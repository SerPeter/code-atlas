# ADR-0041: A version belongs to the dependency, not the package

## Status

Accepted — 2026-08-30.

## Context

`update_external_package_versions` read a project's manifest — `pyproject.toml`, `package.json`, `go.mod`, `pom.xml` —
and stamped the pinned version onto the `ExternalPackage` node as a node property:

```cypher
MATCH (ep:ExternalPackage {uid: $project + ':ext/' + $pkg}) SET ep.version = $version
```

That is survivable only because of an accident. The uid is `{project}:ext/{name}`, so two projects depending on the same
distribution address two different nodes and cannot overwrite each other. The property is wrong about the world
regardless: **numpy does not have a version.** It has many, and which one applies is a fact about a particular project's
dependency on it. The node property states a per-project fact on a node whose identity is only per-project by
coincidence.

That coincidence is exactly what ATL-088 wants to remove. Globalizing `ExternalPackage` to `ext/{name}` — one node per
package across every indexed repo, so package-level facts have somewhere to live — makes the collision real: N projects,
one `ep.version`, last writer wins, silently. The version had to move before the node could be shared.

There was no `Project -> ExternalPackage` edge at all. `DEPENDS_ON` existed but was project-to-project, asserted in a
comment at `graph/client.py:233`, and the only edge reaching an `ExternalPackage` was `Module -[IMPORTS]->`.

## Decision

**The version lives on a `Project -[DEPENDS_ON]-> ExternalPackage` edge**, one per (project, package), written from the
manifest.

`DEPENDS_ON` was extended to a second endpoint shape rather than given a new relationship type. It is the same relation
— "this project depends on this thing" — and the two producers stay disjoint by target label, which is now load-bearing
rather than decorative and is stated at every site that filters on it.

Rejected alternatives:

- **On `Module -[IMPORTS]-> ExternalPackage`.** A module does not declare a version; a manifest does. This would repeat
  the same string on every importing module and give no answer for a declared-but-unimported dependency.
- **A `version` property on a globalized node, keyed by project.** A map property is a join table with no index and no
  referential integrity, and it re-creates the collision the moment two writers race.
- **Leave it and globalize anyway.** This is the collision, not an alternative to it.

The shape is also the standard SBOM one — name as identity, version as a property of the relationship — which is what
makes it survive globalization: when the node becomes shared, the per-project fact already lives on the edge that is
still per-project.

Schema **v18** migrates existing graphs. The migration **moves** the value: write the edge, then remove the node
property _gated on the edge having actually landed_. A version cannot be re-derived from source text — it is read from a
manifest during `atlas index`, and nothing guarantees an index runs before someone next asks for the dependency report —
so clear-and-reparse, which every earlier migration on this schema could safely do, would report every dependency as
unversioned for an unbounded window with no error to explain it. A package orphaned by a deleted project has no edge to
move to, and keeps its property rather than having "moved" quietly become "dropped".

## Consequences

`ExternalPackage` now has a **structural incoming edge**, which is why it must stay out of the reference-counted sweep.
It was already excluded for its `CONTAINS` edge; ADR-0020's invariant — never give a reference-counted label a
structural incoming edge, it would permanently reference every node and silently disable the sweep — now has a second
instance, and ADR-0020 records that the reason it gave for keeping the node project-scoped no longer holds.

Both backends implement it, and the SQLite side needed more than a translation. Its `edges` table has no foreign keys,
so Memgraph's free `MATCH` guard — an unmatchable manifest coordinate writes nothing — had to be reproduced as
`INSERT ... SELECT ... WHERE labels = 'ExternalPackage'`; a bare `INSERT ... VALUES` would write a dangling row for
every Go, Java or PHP coordinate that does not map to import space. The change also forced out an adjacent SQLite bug:
`resolve_cross_project_imports` proved a stub still alive with an untyped `COUNT(*)` over inbound edges, so the new
`DEPENDS_ON` would have kept every rewired stub alive there while Memgraph's typed predicate deleted it.

**The bug this fixes is not reachable yet.** No two projects can collide while the uid stays project-scoped. This is
enabling work, and the test that guards the project predicate has to seed the post-ATL-088 shape by hand — the natural
version of it passes against a query with the predicate removed, which is how it was written first.

Two behaviours worth stating because nothing surfaces them:

- In a monorepo, a project declaring a **sibling** package in its manifest has this edge written during indexing and
  destroyed by cross-project resolution moments later, when the stub is rewired to the real `Package`. The node property
  died the same way before v18, so this is not new.
- An emptied or unparseable manifest **does not** clear edges: both call sites guard with `if dep_versions:`. A read
  failure must not wipe real data, and that is chosen knowing a stale edge asserts a dependency the manifest no longer
  names — a more visible lie than a stale property was. Only a full reindex clears.

## References

- ADR-0020 — referenced runtime-surface nodes; the reference-counting invariant and the amended rationale
- ATL-146 — this change
- ATL-088 (backlog) — globalizing `ExternalPackage`, which this unblocks
