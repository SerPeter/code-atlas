# ADR-0024: Upgrade to Memgraph 3.12.0 — 3.7.2 Segfaults on Vector-Index GC

## Status

Accepted — raises the floor set by [ADR-0001](./0001-memgraph-as-database.md) from a "Memgraph 3.7+" DDL baseline to a
hard 3.12.0 minimum. Does not change any decision in [ADR-0017](./0017-calls-edge-weights.md) or
[ADR-0019](./0019-module-granularity-community-detection.md), both of which recorded results "verified live against
memgraph-mage:3.7.2" — those numbers were re-checked, not re-derived.

## Context

Two consecutive full re-indexes produced a silently incomplete graph — one with 3,282 entities where 6,045 were
expected. The application-level symptoms were real and were fixed separately (an abandoned-work reclaim that never
re-ran, embed messages poison-parked and never recovered), but they were downstream. **Memgraph was dying mid-index.**

The evidence took some digging because every obvious source lies about it:

- `docker inspect` reports `ExitCode: 0` and `OOMKilled: false` — both are the _post-restart_ state, not the crash.
- The container runs `--log-level=WARNING`, and Memgraph's own log shows a clean 19-second gap with no shutdown message.
- Memory was never a factor: peak 1.14 GiB against a 4 GiB cgroup limit, 28%.

The truth is in the WSL2 kernel ring buffer (`wsl -d docker-desktop -e dmesg`): four `SIGSEGV`s in a thread named
`Storage GC`, three of them the identical null dereference at offset `0x44`. Crash intervals of 31/53/72 minutes match
the container's restart intervals exactly.

**All four crashes fell on one day, after eight days of uptime with none.** What changed that day was not the workload —
it was that vector indices came into existence for the first time. `ensure_schema`'s "already current" branch had been
silently skipping index creation, so the production instance had been running with _no_ vector indices at all; fixing
that populated six of them, and every full re-index since then has been mass-deleting and re-inserting vector-indexed
nodes.

## Decision

**Pin `memgraph/memgraph-mage:3.12.0`** in `docker-compose.yml` (both the production and test services) and in
`tests/conftest.py`'s testcontainer.

The mechanism was confirmed experimentally rather than argued, on the disposable test instance, isolated to its own
`:GcStress` label:

| stress                                   | vector index | outcome                                      |
| ---------------------------------------- | ------------ | -------------------------------------------- |
| bulk create + concurrent `DETACH DELETE` | none         | 60 rounds, clean                             |
| identical churn                          | dim-768      | **crash at round 59** — `Storage GC` SIGSEGV |

That is the second half of the natural experiment the production timeline offered: same image, same eight days, no
vector indices, no crashes. It also matches Memgraph's own 3.9.0 release note — _"Fixed a segfault during vector index
garbage collection when vertices or edges were being inserted concurrently, which could occur with bulk
create/delete/create workloads"_ — two releases past the version we were on.

3.12.0 rather than 3.9.0 because it is the newest release that exists, and it also carries #4323, a heap-use-after-free
between transaction commit and storage GC.

## Consequences

**Four behaviour changes, all found by running our own code and test suite against 3.12.0 on a disposable instance
before touching production. None of them errors loudly at startup; the first two are silent, and the third only fires
under the delete-heavy workload this upgrade exists to survive.**

`SHOW INDEX INFO` now reports vector-index labels as `:Callable` (the column doubles as an index filter, so `*`,
`:L1|L2` and `:L1&L2` are also valid); `label+property` and `label_text` rows still report a bare `Callable`.
`_reconcile_search_indices` compares that column against `_EMBEDDABLE_LABELS`, so without stripping the colon the two
sets never intersect and every `ensure_schema` concludes all six vector indices are missing. The re-CREATE that follows
is _not_ destructive — a duplicate `CREATE VECTOR INDEX` is a verified no-op that leaves the populated index intact —
but it warns on every startup and permanently blinds the detector to an index that is genuinely gone, which is the exact
failure this reconciliation was added to catch. Use `removeprefix(":")`, not `lstrip(":")`, or a `:A|:B` filter would
lose its second colon too.

`text_search.search_all`'s third parameter changed in 3.11 from a bare integer limit to a config `MAP`. The old form is
a hard `ClientError`, so BM25 returns nothing for every index — and the call site already swallows per-index failures
with a warning, which is exactly the shape that produces an empty result set rather than a stack trace. The key is
`{limit: N}`; `max_results` is rejected.

**A deleted node can still come back from the vector index.** `DROP VECTOR INDEX` returns before its internal state is
cleaned (already documented in `_create_vector_indices`), and deletion does not purge index entries synchronously.
Reading _anything_ off such a node aborts the entire query — `node.uid` raises "Trying to get a property from a deleted
object", `node:Label` raises the labels equivalent — so after a full re-index this would take out semantic search
wholesale rather than drop a row. `id()` is the one thing still legal to read from a dead node, so `vector_search`
re-matches on it (`MATCH (live) WHERE id(live) = id(node)`) and the MATCH yields nothing when the node is gone. Three
integration tests that issue the procedure directly needed the same guard.

**Embeddings now round-trip through float32.** A vector-indexed property written as `0.1` reads back as
`0.10000000149011612`. Cosine similarity is unaffected and nothing in this codebase compares vectors for equality, but
one test asserted an exact float64 round-trip that was never a promised property.

**`text_search.search` is still broken** — it now fails with a Tantivy "Unable to create search query" error rather than
3.7.2's "Unknown exception", so the standing workaround of using `search_all` stays necessary. Do not treat the upgrade
as licence to switch back.

**Not a complete fix.** [memgraph#4473](https://github.com/memgraph/memgraph/issues/4473) is a `Storage GC` SIGSEGV that
still reproduces on 3.12.0; its fix ([#4475](https://github.com/memgraph/memgraph/pull/4475)) is merged but milestoned
for an unreleased 3.13.0. If crashes recur, that is the expected next suspect, not evidence the upgrade failed.

Ruled out with evidence so they are not re-litigated: it is **not** memory pressure, **not** MAGE (#4473 was reported
against the plain `memgraph/memgraph` image), and **not** WAL or snapshots (that reporter disabled both independently
with no effect).

## Alternatives Considered

**Stay on 3.7.2 and drop the vector indices.** Would stop the crashes — the natural experiment proves it — but semantic
search is the reason the vectors exist, so this trades the feature for the bug.

**Raise `--storage-gc-cycle-sec` to narrow the race.** A probabilistic band-aid with no upstream source recommending it
for stability, and it leaves a use-after-free in place.

**Switch to plain `memgraph/memgraph`.** Buys nothing — the same GC crashes are reported there — and costs
`leiden_community_detection`, which ADR-0019 depends on.
