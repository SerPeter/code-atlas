---
title: "ADR-0046: The SQLite vector index is a shortlist, not a second copy"
tags: [adr, backends, performance, search, embeddings]
kind: decision
---

# ADR-0046: The SQLite vector index is a shortlist, not a second copy

## Status

Accepted (2026-09-05)

## Context

The embedded backend stored every embedding twice: once in `nodes.embedding`, which is authoritative and feeds the
[ADR-0036](0036-the-graph-is-the-embedding-dedup-layer.md) dedup lookup, and once in a
`vec0(embedding float[N] distance_metric=cosine)` table per label, used only to answer KNN.

That second copy bought nothing algorithmically. sqlite-vec's KNN is a brute-force scan — measured cost per row is flat
across 10k / 50k / 200k rows, and an independent source read found `vec0Filter_knn` walking every chunk with a top-k
heap. The copy only made the scan read 32 bits per component instead of 1.

Two further facts shaped the decision:

- **The bottleneck is the pager, not arithmetic.** The same query costs 1.39 µs/row in memory against 9.94 µs/row on
  disk. Roughly seven eighths of the time is blob reads, which is why SIMD is the wrong lever here and quantization is
  the right one.
- **The shipped wheels have no SIMD anyway**, and on the version that does, the acceleration is L2-only while this
  schema uses cosine.

Retrieval matters more than indexing here — a search runs on every MCP query, an index runs when files change — but
indexing could not be allowed to regress either.

## Decision

**1. The per-label `vec0` table holds bit-quantized vectors, and the ranking is rescored exactly.**

A query takes `k × _VEC_OVERSAMPLE` candidates from the bit index by hamming distance, then orders those candidates by
real `vec_distance_cosine` against `nodes.embedding` and keeps `k`. The approximation is confined to _which_ rows are
considered; the ranking returned is exact for whatever was shortlisted.

Measured on 9,118 real 1536-d embeddings from a local `.atlas` index:

|                      | query       | build      | recall@10 |
| -------------------- | ----------- | ---------- | --------- |
| float table (before) | 24.23 ms    | 0.33 s     | —         |
| bit + rescore ×8     | **1.00 ms** | **0.09 s** | **1.000** |

24× on retrieval, 3.7× on indexing, a thirty-second of the index storage, same answers. Recall was 1.000 at every
oversample from 4 to 64; 8 is chosen for headroom at larger graphs rather than to buy back accuracy.

**2. Hand-rolled on stable 0.1.9, not `0.1.10-alpha.4`'s built-in `rescore` index.**

The alpha offers the same technique as one line of DDL and measured 6.1× on the same data at oversample 32 (against 6.4×
for the hand-rolled equivalent at the same factor — the same technique, as expected). It was rejected on dependency
risk: an alpha, maintainer silent since 2026-05-18, 155 open issues. Everything needed — `vec_quantize_binary`, `bit[N]`
columns, hamming KNN, `vec_distance_cosine` — already exists in the pinned stable release.

**3. The probe is quantized by the extension, never by us.**

`WHERE embedding MATCH vec_quantize_binary(?)`, so the query and the stored vectors go through one implementation. A
hand-written bit packer that disagreed on bit order would return confidently wrong neighbours with nothing failing.

**4. Binary quantization is sound here regardless of normalization.**

It keeps the sign of each component. Cosine distance is invariant to positive scaling, so the sign pattern is an equally
good proxy whether or not vectors are normalized — which matters, because this repo's index stores normalized vectors
(mean L2 norm 1.000) while at least one production index does not (0.739).

**5. A dimension not divisible by 8 falls back to the float table.**

`vec_quantize_binary` requires it. Every model in practical use qualifies (384, 768, 1024, 1536, 3072), but `dimension`
is user configuration and a 100 must degrade rather than fail every write. One SQL shape serves both: only the MATCH
expression and the candidate count differ, and on the float path the rescore is a re-sort of rows already in cosine
order.

**6. A backend-local `meta.vec_kind` marker, not a `SCHEMA_VERSION` bump.**

The same reasoning as `_ensure_fts_keyed_by_rowid` in [ADR-0045](0045-sqlite-indexes-what-the-planner-can-prove.md):
that number is shared with Memgraph, where advancing it drops and recreates _its_ vector indices. A vec0 column type
cannot be altered, so a shape change is a drop and repopulate — but from `nodes.embedding`, so it costs local time and
never a provider bill.

## Consequences

- Search results are no longer guaranteed byte-identical to an exhaustive scan. They were identical on every query
  measured, and the rescore makes the _ordering_ exact, but a neighbour the bit shortlist misses is a neighbour the
  search will not return.
- **Recall at scale is unverified.** The measurement was 9,118 vectors at 1536 dimensions. At 100k–1M, `k × 8`
  candidates is a much smaller fraction of the table, and `_VEC_OVERSAMPLE` may need to grow. This is the open question,
  and it is a measurement against real vectors — synthetic ones scored recall 0.20–0.32 where real ones scored 1.000, so
  they cannot stand in for it.
- Rejected alternatives: **sqliteai/sqlite-vector** (Elastic License 2.0; its exact path measured slower than
  sqlite-vec, and its fast path is a snapshot that goes stale on every write and needs a full O(N) re-quantize — the
  wrong shape for continuous indexing); **DiskANN** (build did not finish at 200k×768 in ~2 hours); **vectorlite**
  (fastest measured, but PyPI frozen since 2024, a sidecar `.bin` breaks the single-file story, and 3.2 GB RAM at 1M).
