---
title: "ADR-0051: A permission grant is not a reference"
tags: [adr, graph, analysis, salesforce]
kind: decision
---

# ADR-0051: A permission grant is not a reference

## Status

Accepted (2026-09-07)

## Context

ATL-181 gave a Salesforce permission set one node and an `IMPORTS` edge per granted target: fields, objects, Apex
classes, pages, layouts, tabs. That is the whole value of indexing the type — "who can see this field" and "who can call
this class" have no other answer in the graph — and it replaced a node shape that was catastrophically worse (one real
427 KB file became 1,947 contentless nodes).

But `IMPORTS` already meant something, and two consumers read it as meaning that:

- `get_dead_code_candidates` tests `NOT ()-[:CALLS|USES_TYPE|IMPORTS|…]->(n)`. An unqualified `()`.
- `blast_radius` sorts on `min_depth` first, deliberately, with a test pinning it.

Both were correct before permission sets existed in the graph and silently wrong after.

## Decision

**Being _visible to_ a profile is not being _used by_ anything.** Grant-only kinds — `permission_set` and `profile` —
are excluded from liveness and demoted in impact ranking, while their edges are kept.

### 1. The dead-code predicate names its source

An Apex class listed in any `classAccesses` row gained an inbound `IMPORTS`, so `NOT ()-[…]->(n)` became false for it.
In a real org most Apex is granted somewhere, so `analyze_repo(analysis="dead_code")` returned **zero dead Apex** — a
green answer over a denominator the grants had driven to zero, which is the worst shape a regression takes here: nothing
errors, nothing looks empty, the number is simply a lie.

The inbound test now names its source and excludes grant-only kinds. The exclusion is on the **source's kind**, not on
`IMPORTS` itself, so an ordinary `IMPORTS` from real code still proves life.

### 2. `blast_radius` ranks grants below everything that is a dependency

A permission set grants one edge per field it can see — 1,900 from that one measured file — and every one lands at depth
1 with a perfect `confidence_score`, because `IMPORTS` carries no `weight` property. With N such documents granting an
entity, the first N slots of a limit-20 answer were permission sets, the transitive code dependents the tool exists to
surface were cut, and `truncated` claimed the cut entries were the lowest-impact ones — precisely inverted. Depth-1 code
survived only by an accident of the alphabet: `apex.` sorts before `permset.`.

One sort term, above `min_depth` and the only thing above it.

### 3. They are ranked down, not dropped

Deleting a granted field really does break that permission set's deploy, so it _is_ affected. ADR-0029 settled that a
smaller answer is not a better one. `affected_count` is unchanged and the entries still appear — after everything that
is a dependency rather than a visibility rule, which is what `truncation_notice` already promises.

### 4. The parser-side cap is a runaway guard, not the mitigation

`_MAX_PERMISSION_EDGES` sits deliberately above what real files need. Truncating grants would discard the answer and
keep the problem, because a permission set is the highest-degree node in its graph at 400 edges just as surely as at
1,900.

An earlier draft of that docstring named "stop traversing permission edges transitively" as the real fix. **That was
wrong and worth recording as wrong**: these nodes have zero in-degree over every edge set any traversal uses, so no path
ever passes _through_ one and there was never anything transitive to stop. The harm was entirely at the limit, and it
was a ranking problem.

## Consequences

- `_GRANT_ONLY_KINDS` lives in `schema.py`, because `server/analysis.py` — the one place that ranks on it — already
  imports schema and takes no dependency on the graph layer. Literal kind strings, not an import from a parser, the same
  rule `_WAREHOUSE_PRODUCER_KINDS` follows.
- `compute_blast_radius` now returns `kind` from both backends. Read with `.get`, matching `via`, so a row predating the
  column ranks as ordinary code rather than raising.
- Query-time only: no `EXTRACTION_EPOCH` bump and no reindex. Both changes are visible on every existing index
  immediately.
- **Not fixed, and the same shape:** `analyze_repo(analysis="centrality")` still ranks hubs by degree, so its hubs on a
  Salesforce org degenerate into "the fields the most permission sets grant". It needs `get_centrality_data` promoted
  out of the conformance ledger's not-compared set before anyone changes it, which makes it its own story rather than a
  rider on this one.
