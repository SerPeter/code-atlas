# ADR-0029: blast_radius Traverses Dependency, Not Execution

## Status

Accepted. Depends on [ADR-0028](./0028-every-resolved-edge-states-its-evidence.md), which had to land first: widening
this traversal means scoring paths through edges that carried no evidence, and doing that before they spoke would have
ranked a guess above a marked ambiguity.

Narrows the reading of ATL-099's constraint — "a reference is never a CALLS edge, and `blast_radius` must not traverse
it as execution". That holds. Traversing it as _impact_ is a different claim, and the distinction now lives in the
output rather than in an omission.

## Context

`blast_radius(_parse_python, direction="callers")` returned **0**. That function is the entry point every Python parse
goes through; it is reached via `register_language(parse_func=_parse_python)`, a REFERENCES edge, and the default edge
set was `("CALLS",)`.

The registry case was the tip. Measured across `src/`:

- **289 of 1,568 entities (18%)** have zero inbound CALLS but non-zero inbound dependency edges — invisible to the tool
  whose only job is finding them.
- **`GraphClient`: 239 dependents, reported as 0.** `ParsedRelationship` 209, `ParsedEntity` 154, `ParsedFile` 145,
  `AtlasSettings` 64, `EventBus` 59.
- `USES_TYPE` accounts for 1,427 of the 1,806 hidden edges and `IMPLEMENTS` 254.

The cause is structural rather than a missed pattern: **a class is never "called".** It is annotated, constructed,
subclassed, implemented. So the tool was ~100% blind to every TypeDef in the codebase.

This was not a design tension between two defensible readings. The function's own docstring says `"callers"` traverses
"who transitively depends on _uid_", and the MCP description said, in consecutive sentences, _"'what would be affected
if I change this'. Traverses CALLS edges by default."_ The default contradicted the stated contract.

`trace_path` settles it from the other side: the **narrower** question — "how does A reach B" — already traversed
`CALLS|IMPORTS|USES_TYPE`. Impact analysis needs a wider net than pathfinding and had the narrowest in the codebase.

## Decision

**Default to the dependency closure**: `CALLS`, `USES_TYPE`, `IMPLEMENTS`, `OVERRIDES`, `INHERITS`, `REFERENCES`,
`REGISTERED_BY`, `IMPORTS`.

**`DEFINES` and `CONTAINS` stay out.** They are containment, not dependency: including `DEFINES` would make changing one
method "affect" its class and transitively everything the class touches, which is how a blast radius stops meaning
anything.

**Every hit carries `via`** — the edge types incident to the queried entity on the paths that reach it. This is what
keeps ATL-099's constraint intact: a dependent found through `REFERENCES` is reported as such and never as a caller. The
distinction moves from omission to output, which is the only place it was ever useful.

**The resolved-path test coalesces**:
`all(r IN relationships(p) WHERE coalesce(r.confidence, 'resolved') = 'resolved')`. An absent confidence means
structural (ADR-0028) — `IMPORTS` and `DEFINES` are facts, not guesses — and since the resolvers that guess now say so,
a bare edge is exactly the one that should count as resolved. Without this every non-CALLS hop would have been marked
ambiguous, making the flag meaningless the moment it started mattering.

## Consequences

| entity               | before |                after |
| -------------------- | -----: | -------------------: |
| `GraphClient`        |      0 |                  417 |
| `ParsedRelationship` |      0 |                  476 |
| `EventBus`           |      0 |                  166 |
| `_parse_python`      |      0 | 1 (`via=REFERENCES`) |
| `resolve_calls`      |     22 |                   22 |

`resolve_calls` is the control: already CALLS-reachable, unchanged.

**Answers get larger, and that is the correct direction.** `GraphClient` genuinely has 417 things depending on it within
two hops. `limit`, nearest-first ordering, and `confidence_score` are what make that usable; reporting zero was never
the smaller answer, it was the wrong one.

**`via` is per-entity, not per-path.** It collects the distinct types incident to the start node across all paths that
reach the entity at any depth, so an entity reachable both by CALLS and USES_TYPE reports both. It does not describe the
interior of a multi-hop chain.

**Not addressed:** whether the dead-code predicate should weight by confidence. Widening this traversal does not touch
`find_dead_code`, which still counts an inbound CALLS edge regardless of confidence — deliberately, per ADR-0027.

## References

- `server/analysis.py` — `_DEFAULT_BLAST_EDGE_TYPES`, `blast_radius`
- `graph/client.py` — `compute_blast_radius` (`via`, coalesced confidence)
- `tests/integration/test_relationship_coverage.py` — `impact-of-changing-a-type`
- ADR-0028 (edge evidence), ATL-099 (a reference is not a call)
