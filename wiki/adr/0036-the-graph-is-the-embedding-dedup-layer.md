# ADR-0036: The graph is the embedding dedup layer, not Valkey

## Status

Accepted — 2026-08-28. Amends ADR-0004, whose tiered pipeline described a Valkey embedding cache as tier 2.

## Context

The embed stage had three tiers: the node's own `embed_hash` (a graph read that was already happening), a Valkey cache
of vectors keyed by text hash, and the provider API. The middle tier shared one `noeviction` Valkey instance with the
event streams and the indexer lease.

That co-tenancy is what forced the decision. A disposable cache filled the instance, Valkey then rejected **all** writes
— stream XADD and lease SET included — and the AST worker crashloopped on `server:OutOfMemoryError`. `noeviction`
protected cache bytes by failing correctness writes, which is backwards, and no `maxmemory-policy` fixes it: the only
TTL'd keys are the cache and the 60-second lease, so any eviction policy is choosing between them (ATL-128 analysed this
and shipped only the memory ceiling).

Four measurements decided it, all taken on the live production instance before any code changed:

| measurement                                             | value                         |
| ------------------------------------------------------- | ----------------------------- |
| Keys in Valkey that were `atlas:emb:*`                  | **32,385 of 32,391 — 99.98%** |
| Cached hashes already on a graph node                   | **31,996 of 32,385 — 98.8%**  |
| Graph vectors _missing_ from the cache                  | **0**                         |
| `embed_hash` values shared across more than one project | **0**                         |

And the realized hit rate, from a real delta index with the cache still in place and warm (32,385 keys):

```
247 entities,  0 graph hits, 0 cache hits,  247 embedded
363 entities,  0 graph hits, 0 cache hits,  363 embedded
1736 entities, 0 graph hits, 0 cache hits, 1736 embedded
```

**2,346 API calls, zero cache hits.** Not an anomaly — a consequence of the tier order. Tier 1 short-circuits unchanged
text before the cache is consulted, so anything reaching tier 2 has _changed_, and changed text is by definition not in
a cache keyed by its own hash. The cache could only ever serve text that changed back to a previous value, or moved.

The vectors it stored were already durably in the graph, keyed by exactly `hash_text(build_embed_text(props))`.

## Decision

**Delete `EmbedCache` and make the graph tier 2.** Before calling the provider, the embed stage asks the graph whether
any node — any project, any label — already carries a vector for the same `embed_hash` under the same model, and copies
the ones it finds.

- `GraphBackend.find_embeddings_by_hash(hashes, model)` returns one exemplar vector per hash, on both backends.
- **Indexed on `:Entity(embed_hash)`.** One index on the marker rather than six per-label ones: the question is
  cross-label and cross-project by nature, so one seek answers it. Unindexed it is one full scan per hash — measured at
  a 10-second timeout for 3,000 hashes over 66k nodes.
- **Filtered on `embed_model`.** A vector only means something inside its model's space, and one database holds several
  — measured, 25,305 vectors at 1536d under one model and 6,691 under another (ADR-0035). Vectors written before that
  stamp existed carry no `embed_model` and are deliberately invisible to this lookup: dedup _copies data_, so it gets
  the strict half of the asymmetry, while search stays permissive.
- **Identical texts inside one batch embed once.** The one hit class the cache could never serve either, because
  `--full` cleared it before the run started.
- `cache_ttl_days` is removed from settings. Settings sections reject unknown keys, so a config still setting it fails
  loudly — which is the point; a dead setting that silently does nothing is the artifact this repo removes.

Rejected: **keeping the cache and tuning `maxmemory-policy`.** Analysed in ATL-128 and rejected there: with cache,
streams and lease sharing an instance, every policy sheds something that matters. Safety needed evicting a _tenant_, not
tuning a policy — and the census above says which tenant.

Rejected: **a separate Valkey instance for the cache.** Fixes the co-tenancy and keeps a layer that measured a 0% hit
rate and 98.8% duplication of data the graph already holds. More infrastructure to carry a copy of the source of truth.

## Consequences

**Valkey returns to coordination only** — streams, consumer groups, the indexer lease. Its working set drops from 266MB
to roughly nothing, and `noeviction` becomes correct rather than merely tolerated, so ATL-128's interim `volatile-lru`
is never needed.

**Cross-project dedup survives and gets better.** It was the cache's one real service; the graph does it too, because
all projects share one Memgraph. It also now covers cases the cache could not: a `--full` reindex used to `clear()` the
cache first, so full reindexes got near-zero hits from prior runs. The graph is not cleared.

**On the SQLite backend, dedup is per project root.** That backend is one database file per root, so its lookup reaches
within a root and its monorepo sub-projects, never across separate repositories. `backend.graph` defaults to `"auto"`
and falls back to SQLite whenever Memgraph is unreachable, so this is the behaviour on any machine without Docker
running — stated here rather than left to be discovered.

**A concurrency window is slightly wider.** Two workers embedding the same brand-new text both miss and both call the
API. That was true before, but Valkey's `put_many` made a vector visible to the _next batch_ immediately, while the
graph write lands only when the current batch completes. The within-batch dedup dict closes the intra-batch half; the
cross-worker half costs one duplicate call on first index of a repo with repeated boilerplate, and the writes are
idempotent.

**Schema v15.** One index added, `:Entity(embed_hash)`. Because the Memgraph instance is shared, the first process to
run `ensure_schema` upgrades it for everyone, and any process still on v14 code then refuses to start. Restart the
daemon and any `atlas mcp` in the same window.

## References

- ATL-127 — the story, with the full measurement set.
- ATL-128 — the eviction-policy analysis this supersedes the need for.
- ADR-0004 — the tiered pipeline; its cache tier is what this replaces.
- ADR-0035 — the `embed_model` stamp this lookup filters on.
- `graph/client.py:find_embeddings_by_hash`, `indexing/consumers.py:_resolve_from_graph`.
