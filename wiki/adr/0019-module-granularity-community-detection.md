# ADR-0019: Community Detection at Module Granularity

## Status

Accepted — amends [ADR-0017](./0017-calls-edge-weights.md) (which documented weighted Leiden as the clustering
mechanism) and narrows [ADR-0015](./0015-embedded-backend-option.md)'s reason for excluding community detection from the
SQLite backend

## Date

2026-07-30

## Context

`find_communities` was meant to answer "what subsystems does this codebase have?". Measured against a real full index of
this repo (174 files, 6009 entities), the shipped implementation returned:

```
6 communities, sizes [1236, 3, 3, 2, 2, 2]
```

One community holding ~95% of production code, plus noise. Useless for its stated purpose.

The obvious fix — retuning resolution, since [ADR-0017](./0017-calls-edge-weights.md) had introduced weights and `gamma`
is divided by total edge weight — does not work. A sweep of MAGE Leiden's `resolution_parameter` found a **cliff** with
no usable middle:

| `resolution_parameter` | communities                      | largest |
| ---------------------- | -------------------------------- | ------- |
| 0.01 (shipped)         | 9                                | 3648    |
| 0.05                   | 25                               | 3605    |
| 0.10                   | 40                               | 3585    |
| 0.30                   | 1809                             | 849     |
| 0.60                   | 2293                             | 787     |
| 1.0 / 2.0              | raises "No communities detected" | —       |

Below 0.3 a giant blob; above it, ~1800 communities of which only four clear the noise threshold. `gamma` was nearly
inert across 0.1–2.0.

The cause is **granularity**, diagnosed by inspecting what the projection actually contained:

```
CALLS    Callable -> Callable   10327 edges
IMPORTS  Module   -> TypeDef      424
IMPORTS  Module   -> Callable     256
IMPORTS  Module   -> Value         73
IMPORTS  Module   -> Module          6   <-- six
```

`IMPORTS` almost never joins two Modules; it joins a Module to the individual **symbol** it imports, so every module
importing a shared symbol hubs through that one node. And `CALLS` alone is no better — projected by itself it yields a
3349-node component out of 3427 (98%), because a real call graph is densely connected through shared helpers.

No resolution parameter fixes a graph whose nodes are the wrong things. "Which subsystems exist" is a question about
**modules** (~174 here), not about individual callables (~5300).

## Decision

Cluster at module granularity, in-process, with a deterministic algorithm:

1. **Aggregate.** Attribute every callable-level CALLS edge to the modules owning its endpoints and **sum** the ADR-0017
   weights per module pair, so a pair joined by many confident production calls outranks one joined by a single
   ambiguous test call. Fold reciprocal directions and CALLS/IMPORTS parallel edges into one undirected weight. Drop
   intra-module calls and self-imports.
2. **Partition** the resulting ~10²-node graph with greedy modularity maximisation (Clauset–Newman–Moore), written
   in-tree — **no new third-party dependency**, and no networkx or igraph.
3. **Determinism is a requirement, not a nicety.** MAGE's Leiden is documented non-deterministic, which makes a tool's
   output unstable between identical calls and impossible to diff. Ties are broken on the lexicographically smallest
   community-key pair, so identical input yields byte-identical output.

The MAGE path is removed entirely: the `project()` projection, the `leiden_community_detection.get(subgraph, "weight")`
call, and the `PROCEDURE_UNAVAILABLE` branch.

**ADR-0017 is not reversed.** Its weights are still the input — they are what makes an aggregated module edge
meaningful. What changes is the consumer: weights now feed a summation into module-pair edges rather than a weighted
Leiden call. ADR-0017's Empirical Validation section documents behaviour of a procedure this code no longer invokes;
read it as a record of why the string `confidence` could not be used as a weight, which remains true and is still the
reason a numeric `weight` property exists.

## Consequences

### Positive

Measured on the same real graph, shipped defaults:

```
granularity=module  module_count=55  edge_count=197  modularity=0.3706  communities=7  ~94ms
```

The blob is gone — the largest community is 15 of 39 connected modules, and it is exactly the `parsing` package.
Eyeballed, the partition maps onto real subsystems: parsing (every language module plus the dispatcher), the indexing
pipeline (correctly pulling in `parsing.detectors` and `search.embeddings`, which the pipeline drives), graph storage
(Memgraph client plus both SQLite backends), search + analysis, and CLI/cross-cutting infra.

- Deterministic output, so results can be diffed across runs and across commits.
- No MAGE dependency for this analysis, and no new Python dependency either.
- External nodes are now structurally unable to bridge communities, rather than needing an explicit exclusion clause.

### Negative

- A hand-written clustering algorithm is code this project now owns and must maintain.
- Greedy modularity is agglomerative and roughly O(n²) in module count. Fine at ~10² modules; the ~10x-scale target
  (~10³) needs measuring before it is assumed safe.
- `_COMMUNITY_SPLIT_MIN_MODULARITY = 0.12` is empirically tuned on **one** repository. The partition is flat across
  0.08–0.17 here, but that plateau is not guaranteed to generalise.

### Risks

- Module attribution keys on `file_path`. Any entity without one is invisible to the clustering.
- The partition quality claim rests on eyeballing one codebase. A high modularity score over a wrong grouping is still
  wrong, so new repositories should be sanity-checked by reading member lists, not by trusting Q.
- `graph.protocol` lands in the CLI/infra community rather than with graph storage — it is a `TYPE_CHECKING`-only
  Protocol with almost no real edges, so it drifts to wherever its few edges point. Mildly wrong, and an inherent limit
  of structural clustering over a file with no runtime coupling.

## Alternatives Considered

### Retune `gamma` / `resolution_parameter`

Rejected on measurement — the sweep table above shows a cliff, not a curve. This was the originally scoped task, and the
measurement is what disproved it.

### Materialise module-level edges so MAGE can still cluster them

Rejected: `project()` is a view over real relationships, so keeping MAGE means writing aggregated edges into the graph.
A tool named `find_communities` must not mutate the graph to answer a question, and the analysis path is read-only by
design.

### Use `igraphalg.community_leiden`

Rejected: it also requires a projected subgraph of real relationships, so it does not avoid the materialisation problem,
and it materialises the whole graph into igraph server-side — a materially different performance profile at the stated
scale.

## References

- `src/code_atlas/server/analysis.py` — `_analyze_communities`, `_fetch_community_inputs`, `_greedy_modularity`,
  `_detect_module_communities`, `_modularity`
- [ADR-0017](./0017-calls-edge-weights.md) — the CALLS weights this consumes
- [ADR-0015](./0015-embedded-backend-option.md) — SQLite backend scope. Community detection is still Memgraph-only, but
  the reason is now the two raw Cypher reads that feed the clustering, **not** MAGE.
