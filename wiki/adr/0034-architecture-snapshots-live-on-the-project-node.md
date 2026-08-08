# ADR-0034: Architecture snapshots live on the Project node

## Status

Accepted — 2026-08-08. Revisit when a second kind of time series appears (see Consequences).

## Context

The architecture-health view (ADR-adjacent, ATL-119) reports propagation cost, core size, cycle inventory and fan-in
concentration. Those numbers are close to meaningless as single readings: against the published anchors from MacCormack,
Rusnak and Baldwin (2006), a propagation cost of 8.4% sits between refactored Mozilla (~2%) and pre-refactor Mozilla
(~17%) — most of the useful range. The question the view exists to answer is _trajectory_, which needs a history.

So the metrics have to be recorded once per index run and read back. Where they live is the decision.

Three options were on the table:

1. **A new `ArchitectureSnapshot` node label**, one node per run, linked to `Project`.
2. **A bounded list property on the existing `Project` node.**
3. **A file beside the index** (JSON on disk).

## Decision

**Option 2.** Snapshots are stored as a list of JSON strings in an `architecture_snapshots` property on the `Project`
node, capped at 50 entries, written through the existing `GraphBackend.update_project_metadata` and read through
`get_project_status`.

A snapshot carries the coverage it was computed over — module count and any language whose grammar was missing — not
just the metrics.

## Rationale

**Option 1 is the better modelling and was rejected on verifiability.** A new label means a new schema version and a
migration, and this decision was taken while Docker was unavailable, so neither could be run against a real Memgraph.
Shipping an unrunnable migration is the same class of unverifiable claim that ATL-112 exists to eliminate; a property
write through a method both backends already implement has a far smaller unverified surface. This is an honest
constraint, not a claim that option 1 is worse.

**Option 3 was rejected because the history belongs to the index, not to a machine.** A file beside the checkout does
not survive a re-clone, is not shared between the daemon and the MCP server, and would give two agents on the same graph
two different histories.

**Retention is bounded because a daemon-indexed repo indexes continuously.** Fifty runs is enough to read a trend and
costs roughly ten kilobytes. An unbounded list on a hot node is a slow leak that would only surface as a mysteriously
large property months later.

**Coverage travels with the metrics** because without it the history is actively misleading rather than merely
incomplete. A propagation cost that rose because C++ extraction improved (ATL-096 moved it from 0.496 to 0.937) is not a
codebase that decayed — the new dependencies were always there, we just could not see them. The trend reports
`direction: "unclear"` whenever the module count moved more than 10% across the window, rather than picking one of
"better"/"worse" it cannot justify.

**Recording never raises.** It runs on the index path, and telemetry about how healthy the architecture is must not be
able to fail the indexing it is measuring. Failure is a return value.

## Consequences

- No schema migration; `SCHEMA_VERSION` stays at 12.
- Querying across the history in Cypher is awkward — the data is opaque JSON inside a property, so "show me every
  project whose propagation cost rose" is a client-side scan rather than a query. That is acceptable while the only
  consumer is one project's own view.
- **Revisit if a second time series appears** (index duration, entity counts, coverage over time). At that point the
  modelling pressure favours option 1 for all of them together, and the migration becomes worth writing once rather than
  three times.
- The round trip has not been exercised against a real backend. The Memgraph list-of-strings property write and the
  SQLite `json_patch` path are both plausible and both unverified until Docker returns.
