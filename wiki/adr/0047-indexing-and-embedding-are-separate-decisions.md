---
title: "ADR-0047: Indexing and embedding are separate decisions"
tags: [adr, embeddings, search, indexing]
kind: decision
---

# ADR-0047: Indexing and embedding are separate decisions

## Status

Accepted (2026-09-06)

Amended by [ADR-0052](./0052-rrf-rank-space-has-no-epsilon.md), which deletes §4's vector floor without replacing it. §4
claimed the floor earns "the smallest non-zero contribution the channel can pay"; measured, it is 98.4% of an entire
rank-1 BM25 hit, and no constant is both small enough to preserve another channel's ordering and large enough to change
an outcome. §2's sentence about `xml_element`/`xml_setting` is also superseded — see the note there.

## Context

The only controls over embedding were global: `[embeddings] enabled` and `--no-embed`. Parsing a file and buying vectors
for its entities were one decision, so anything the indexer could read, it embedded.

`_DEFAULT_INCLUDE` is broad on purpose (`*.json`, `*.yaml`, `*.toml`, `*.xml` — see the volume note in
`orchestrator.py`), because config files carry architectural signal the graph cannot see otherwise. That trade holds for
the _graph_. It does not hold for the _vector index_.

A generated JSON or XML tree — a UI layout, a report definition, a build manifest — has no dialect handler, so it parses
into `config_setting` leaves named by whatever key sits above them. Those names repeat, they are reached by generated
identifiers rather than by anything a person would search for, and on a large tree they can come to dominate a project's
embeddings outright. A vector is a claim that a node has semantic neighbours, and a key-value pair inside an
unrecognised blob does not.

## Decision

**1. Which entities get a vector is a policy, with two axes.**

```toml
[embeddings]
exclude       = ["vendor/**", "generated/**"] # path axis; empty by default
include       = ["ops/critical/"]             # re-admits; beats both axes
exclude_kinds = ["config_setting", "config_section"] # kind axis; this IS the default
```

`[scope]`'s gitignore dialect, and its replace semantics on `exclude_kinds` — the one field that has a default to
replace. There is deliberately no `extend_exclude` here: the path default is empty, so it would do exactly what
`exclude` does under a second name.

**2. The default is a kind rule, not a path rule.**

`config.py`'s generic structural fallback emits `config_file` / `config_section` / `config_setting`, and _only_ that
fallback emits them — a file a dialect recognised gets `k8s_resource`, `compose_service`, `ci_job`, `dbt_source` and so
on. So "structured data nobody could make sense of" is not a path pattern to be guessed at; it is a kind.

Defaulting on paths, as the original design proposed, would have taken the vector off every recognised dialect too and
then needed an `include` list to hand them back. A GitHub Actions workflow is worth finding semantically; it keeps its
vector, with no line of config.

`config_file` stays embeddable — the file-level node is named after the file and answers "what is this config for". The
XML fallback's twins (`xml_element`, `xml_setting`) stay too: ATL-144 is actively trying to extract _more_ from
Salesforce metadata, and pre-empting it here would work against it. Both are one line away for a user who disagrees.

> **Superseded (ATL-183).** ATL-144 settled it the other way — a recognised Salesforce type now gets its own kind and
> keeps its vector, so `xml_element`/`xml_setting` came to mean precisely "XML no handler recognised" and joined
> `DEFAULT_EXCLUDE_KINDS`. One real permission set contributed 1,944 of them, all named `fieldPermissions`, every one
> with an empty `source`. `xml_document` stays embedded, matching `config_file`.
>
> That widening is also what broke §4: it turned the excluded cohort from a minority into 68.7% of a real Salesforce
> repo, the regime in which the floor inverts. Two individually correct changes composing into a defect — see
> [ADR-0052](./0052-rrf-rank-space-has-no-epsilon.md).

**3. Three gate sites, and the third is the one that bites.**

| #   | site                                     | what it stops                                                                                                          |
| --- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| 1   | `ASTConsumer`, before `build_embed_text` | new work being queued — and the text build and hash it would cost                                                      |
| 2   | `EmbedConsumer._allowed_by_policy`       | what is already on the stream: a poison-parked event, an abandoned PEL entry, work published before the policy changed |
| 3   | `_reconcile_missing_embeddings`          | the sweep that re-queues vector-less entities                                                                          |

Site 3 is not defence in depth. Ungated, it re-queues every excluded entity on **every run**, the embed stage drops them
all, the next pass finds exactly the same set, and `_reconcile_until_embedded` gives up with a warning about lost work —
permanently, on a graph where nothing is wrong. The policy is what teaches it that "no vector" can be intentional.

The kind axis is pushed into the query at site 3 and in the reclaim sweep, so `find_unembedded_entities`' 5,000-row cap
is spent on real holes rather than on excluded entities. It cannot be pushed down when `include` is set, because
`include` beats `exclude_kinds` and that decision needs the path and the kind together; then the query filters nothing
and the policy is applied in full afterwards.

**4. Excluded is not invisible — and a gate must not make it so.**

A node the policy excludes is still parsed, still named, still traversable, still returned by BM25 and by graph search.
Only the vector goes. That is the claim, and two places could quietly break it:

- `hybrid_search`'s RRF fusion. A uid absent from the vector ranked list earns nothing from that channel, and
  `analyze_query` weights vector at **2.0** for any natural-language query of three words or more — so an excluded node
  competes for 1.5 of a 3.5-weight pot and loses to worse matches that happen to own a vector.
- `vector_search(threshold=...)`, a hard filter over the vector index.

The rule: **a similarity gate filters _scored_ candidates; an entity the policy never allowed to be scored is admitted
at the floor.** In `hybrid_search`, a uid another channel surfaced and the policy excludes is appended to the tail of
the vector list, earning `w · 1/(k + tail + 1)` — the smallest non-zero contribution the channel can pay. Barely
passing.

> **Removed (ATL-184), and the claim above is false.** `w · 1/(k + tail + 1)` is the smallest contribution _within the
> vector channel_, which is not the same as small: at the production `fetch_limit` of 63 it is `2/124`, **98.4% of an
> entire rank-1 BM25 hit**. RRF's curve decays only 1.98× across a fetched window, while the adjacent-rank differential
> that decides order is `1/3782`. The floor was ~62× coarser than the ordering it must not disturb, and the interval of
> constants that both preserve that ordering and change any outcome is **empty**. Deleted without replacement by
> [ADR-0052](./0052-rrf-rank-space-has-no-epsilon.md); the two limits below stood, but they bounded the wrong thing.

Two limits, both deliberate. Only uids another channel already surfaced, so nothing enters a search on the strength of
having no vector. And only entities the _policy_ excludes — a missing vector can also be a pipeline hole, and
`_reconcile_missing_embeddings` exists to heal those; a floor score would hide one. `vector_search` itself is unchanged:
it is the vector channel by definition, and a tool named for one channel should not silently serve another.

The floor is applied _after_ provenance is built, so `sources` still reports the ranks the channels actually returned. A
floored uid did not come back from a vector search and must not claim it did.

**5. The policy is recomputed from settings, never stamped on a node.**

A node property would mean a `SCHEMA_VERSION` bump, and a bump drops the vector indices unconditionally while recreating
them only conditionally ([ADR-0024](0024-memgraph-312-for-vector-index-gc.md)) — an outage this project has already
caused itself once. It would also freeze the policy at index time. Recomputing costs one set lookup and at most two
pathspec matches on inputs the caller already holds, and it means a policy change takes effect on the next index — and
at query time, immediately.

**6. Vectors already bought are reclaimed, not left behind.**

`_reclaim_excluded_embeddings` runs at the end of an index: it reads the entities that hold a vector, drops the ones the
policy now excludes, and strips `embedding`, `embed_hash`, `embed_model` and any attached `EmbedChunk`. Pure graph work
— no provider call, nothing recomputed. The alternative was to tell users to run `--reset-embeddings`, which re-bills
every vector in the database for a dimension change nobody made.

On SQLite the `vec0` row goes too, which the project-scoped `clear_embeddings` deliberately leaves behind. It can afford
to: what it clears is about to be re-embedded and the rows are re-keyed on the next write. These nodes never get a
vector again, so a stale row would sit in the shortlist forever, spending `k * _VEC_OVERSAMPLE` slots the rescore then
discards ([ADR-0046](0046-the-vector-index-is-a-shortlist-not-a-copy.md)). The FTS row is kept, in both backends and on
purpose — BM25 is the channel this node now lives in.

## Consequences

- **This is a behaviour change for every existing user.** On the next `atlas index`, generic config leaf and section
  nodes lose their vectors. They stay indexed and findable; they stop appearing in pure vector search.
- Search results shift for queries that were being won by config blobs. That is the point, and it is also the risk: the
  floor is what keeps the shift from becoming a disappearance.
- One more thing to get wrong when adding a gate. Any new site that decides "should this be embedded" has to consult
  `EmbedPolicy` or it re-opens the hole site 3 demonstrates.
- `find_unembedded_entities` now returns `(uid, label, kind, file_path)` and takes `exclude_kinds`;
  `find_embedded_entities` and `clear_embeddings_for_uids` are new on both backends and output-compared by the
  conformance suite.
- The `atlas ui` search path does not get the policy, because `create_app` is never passed an embed client — its vector
  channel does not run at all. A pre-existing gap, not one this ADR creates; when the UI gains a vector channel it must
  gain the policy in the same change.
- The reclaim sweep runs from `atlas index`, not from the daemon's consumer loop. A daemon-only deployment still writes
  no new excluded vectors — gates 1 and 2 are in the consumers — but reclaiming ones bought under an older policy takes
  one `atlas index`.
- `find_embedded_entities` matches the primary labels, not the `:Entity` marker. The marker indexes `uid` and
  `embed_hash` only, so a `project_name` predicate under it scans every node in the graph — affordable for
  `clear_embeddings`, which runs on a destructive flag, and not for something that runs at the end of every index.
