---
title: "ADR-0045: The SQLite backend indexes what the planner can prove"
tags: [adr, backends, performance, schema]
kind: decision
---

# ADR-0045: The SQLite backend indexes what the planner can prove

## Status

Accepted (2026-09-03)

## Context

`schema.py` holds one registry of label-property and composite indices, shared by both backends. Memgraph consumes it
directly: it has no label-free index, so per-label indices are the only shape available and the registry matches its
needs exactly.

`SqliteGraphClient` rendered the same registry as ~124 **partial** indices —
`CREATE INDEX ... ON nodes(<prop>) WHERE labels = '<Label>'` — and had no unqualified index on `nodes` at all.

SQLite will only use a partial index when it can _prove_ the query implies the index's predicate, and it does that proof
at plan time, from the SQL text. Three predicate shapes dominate this backend's write path and none of them can be
proved:

| Predicate             | Why no partial index applies                      |
| --------------------- | ------------------------------------------------- |
| `labels = ?`          | a bound parameter; unknown when the plan is built |
| `labels IN (...)`     | does not imply any single-label predicate         |
| `labels NOT IN (...)` | implies the opposite of one                       |

So the file-hash gate, the per-file diff, the CALLS resolution lookups, the worktree sweep and the
[ADR-0036](0036-the-graph-is-the-embedding-dedup-layer.md) embedding-dedup lookup all planned as `SCAN nodes` — a walk
of the whole node table, on paths that run per file or per batch.

This was found by the benchmark suite, not in production, and `tests/bench/test_bench_writepath.py` measured it rather
than asserting it: 64 row visits at 50 entities, 184 at 146, on a batch held constant. That class deliberately declined
to act — _"whether 124 partial indices are the right shape is a separate decision that should be taken against a
number"_ ([ADR-0043](0043-benchmarks-count-work-and-account-for-waiting.md)). This is that decision.

## Decision

**1. Three unqualified indices, in the SQLite backend's own base schema.**

```sql
CREATE INDEX ix_nodes_project_file       ON nodes(project_name, file_path);
CREATE INDEX ix_nodes_labels_project_name ON nodes(labels, project_name, name);
CREATE INDEX ix_nodes_embed_hash ON nodes(
    json_extract(props_json, '$.embed_hash'),
    json_extract(props_json, '$.embed_model')
) WHERE embedding IS NOT NULL;
```

Verified with `EXPLAIN QUERY PLAN` over the twelve hottest statements: twelve of twelve now plan as index seeks, none
scans. The write-path sweep went from 64/184 row visits to 0/0.

**2. Not by editing `schema.py`.**

The obvious move — make the shared registry emit non-partial indices — would regress Memgraph, which needs the per-label
form. This is a SQLite-side answer to a SQLite planner rule, so it belongs in the SQLite backend's DDL.

**3. Not by adding more partial indices.**

No number of `WHERE labels = 'X'` indices helps a predicate that names no label as a literal. The shape was wrong, not
the coverage.

**4. The ~124 partial indices stay.**

They are redundant for several queries now, and removing them would cut write amplification. That is a real follow-up
and a separate measurement: each one that is genuinely dead has to be shown dead per statement, and a wrong removal is a
silent scan. Not bundled into a change whose point is that scans are expensive.

**5. `ix_nodes_embed_hash` is partial on `embedding IS NOT NULL`.**

The dedup lookup asks whether _any_ node — any project, any label — already holds a vector for this text under this
model, so no per-label index can serve it by construction. The query states `AND embedding IS NOT NULL` as a literal,
which the planner can prove, and the predicate also keeps the index to rows that have vectors rather than to every node
in the graph.

**6. No schema version bump.**

`_BASE_SCHEMA_SQL` is executed on every connection open and is `CREATE ... IF NOT EXISTS` throughout, so existing
databases pick the indices up on next connect. A version bump would have been a lie about what changed, and on the
Memgraph side a bump drops vector indices — a cost with nothing to buy here.

**7. The benchmark assertions invert.**

`TestScanAmplification` was written against the defect: it asserted scans exist and that amplification grows. Both are
now false, which is the suite working. It now fails when a scan comes back, and carries its own control statement — a
`json_extract` on a key nobody indexes, which no schema change can optimise away — because "no scans found" is also what
a silent detector reports. The same substitution was made in `tests/unit/backends/test_instrumentation.py`, which had
been using the file-hash gate as its known-scanning fixture and so was resting on a product defect.

## Consequences

- Three more b-trees on `nodes`: two index inserts per node write, one more per embedded node. Against 124 existing
  partial indices this is small, and it is the trade being made.
- A new predicate shape that uses `labels IN`/`NOT IN`/`= ?` on a column outside these three indices will scan again.
  The benchmark's `test_nothing_on_the_write_path_scans` is what catches that.
- Memgraph's index shape is untouched, and the two backends now deliberately differ here rather than sharing a registry
  that only suits one of them.
