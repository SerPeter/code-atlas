---
title: "ADR-0054: An external name is weighted by how deliberately it was chosen"
tags: [adr, indexing, external, search, ranking]
kind: decision
---

# ADR-0054: An external name is weighted by how deliberately it was chosen

## Status

Accepted (2026-09-13) — covers ATL-191 P1 and P2. Amends nothing; it adds a factor to the ranking chain
[ADR-0052](./0052-rrf-rank-space-has-no-epsilon.md) constrains, and changes the uid of container-image `ExternalPackage`
nodes.

## Context

An `ExternalPackage` is a bare name. `from pathlib import Path` mints a node called `Path` that says nothing about what
`Path` is, and 557 such names exist on the reference graph. ATL-191 wants to give them signatures, which means indexing
considerably more of the external world than the original design allowed:

> _Why not index everything? Token budget. Indexing all of numpy + pandas + sqlalchemy would add hundreds of thousands
> of nodes, bloat embeddings, and dilute search results._

Dilution is a ranking problem. The objection is answerable if search can tell a dependency somebody chose from a name
that merely turned up — but nothing in the graph recorded the difference, and the three signals that could have are each
individually wrong:

- **A `DEPENDS_ON` edge** was written only from eight language manifests, and only for a declaration that carried a
  version constraint. On this repo that covered 22 of 117 names. `loguru`, declared in plain sight as
  `dependencies = ["loguru"]`, was indistinguishable from a name nobody had ever asked for.
- **`[project.optional-dependencies]` and `[dependency-groups]`** were not read at all, so every dev and extras
  dependency — `pytest`, `litestar`, `jinja2`, every `tree-sitter-*` grammar — read as accidental.
- **Container images** were the largest single distortion. `memgraph/memgraph-mage:3.12.0` is pinned in a file somebody
  wrote, and it reached the graph as an undeclared name with no version, because a Dockerfile and a compose file were
  not in the manifest table.

Images were also being _named_ wrongly. `resolve_imports` derives a package from an unresolved import by taking the part
before the first dot, which is correct for `os.path` and for `java.util.List`. Applied to
`ghcr.io/huggingface/text-embeddings-inference` it produced a package called **`ghcr`**, with the real name demoted to a
sibling `ExternalSymbol`. Every image on a registry collected into one node named after the registry.

## Decision

**1. Every `ExternalPackage` carries a `provenance` tier, and ranking multiplies by it.**

| tier         | test                                                    | factor |
| ------------ | ------------------------------------------------------- | ------ |
| `declared`   | a `Project -[DEPENDS_ON]-> ExternalPackage` edge exists | 1.15   |
| `stdlib`     | the name is in `STDLIB_MODULE_NAMES`                    | 0.85   |
| `undeclared` | neither                                                 | 0.70   |

The edge is written from a manifest and from nothing else, so **its presence _is_ the declaration** — provenance is a
read of data already in the graph, with no new parsing pass and no new node property to keep in sync.
`classify_external_package_provenance` therefore has to run _after_ `update_external_package_versions`; before it,
everything reads undeclared and is never revisited.

`_PROVENANCE_BOOST` is a built-in table in `search/engine.py`, **not** an `ImportanceSettings` rule. Its `is_empty()`
fast path means a default configuration's ranking is byte-identical to what it was before importance existed, which is
the contract that makes importance safe — and exactly what a default of "do nothing" would give here. The dilution this
epic permits has to be absorbed without anybody configuring anything.

**2. The tier is `undeclared`, not `transitive`.**

`transitive` would be a claim about _why_ a package is present, and on the reference graph that claim is mostly false:
the tier's largest members are docker images and GitHub Actions, which arrived through no dependency resolver at all.
Separating a genuine transitive dependency from a foreign-ecosystem name needs `importlib.metadata.requires` plus
ecosystem identity on the node, and neither exists. The name says what the check tests.

**3. A container image is a declaration, and its tag is the version.**

`Dockerfile`, `docker-compose.y[a]ml` and `compose.y[a]ml` join the manifest table. `FROM python:3.14-slim` declares
`python` at `3.14-slim`. An untagged image yields an **empty** version rather than an invented `latest`: the file did
not say `latest`, and the edge alone is what makes the tier read `declared`.

**4. A parser may declare its external name atomic.**

`ParsedRelationship.properties[IMPORT_ATOMIC_NAME]` tells `resolve_imports` that `to_name` is already the package name.
Set by the Dockerfile, compose and Kubernetes parsers for image references.

It is a parser-side marker rather than a resolver-side heuristic because **nothing about the string distinguishes
`ghcr.io/acme/api` from `os.path`** — a resolver guessing from shape would have to be re-guessed for every ecosystem
added later, and the parser already knows which it emitted.

One rule, `schema.split_image_reference`, produces both halves: the repository becomes the node name (via the parsers)
and the tag becomes the `DEPENDS_ON` version (via the manifest parsers). Those two join on the name, so a second
implementation drifting by one character would silently stop writing dependency edges for every image, with nothing
raised. There were two copies before this; there is one now.

**5. The installed-gate switches itself off when it has no evidence.**

Extras and dependency-groups count as declared only when the distribution is installed — an extra nobody selected is a
version this project _would_ pin, not one it depends on. But "installed" can only ever mean _in atlas's own
environment_, and when atlas indexes somebody else's repo that environment is not the project's. The gate would then
demote every extra on a fact about the wrong machine.

So the gate is skipped entirely when **none** of the project's core `[project].dependencies` are installed either. A
gate with no evidence must not filter. This is the same rule already written down for `_distribution_import_names`: a
miss, never a wrong edge.

## Consequences

**Container-image node uids change.** `{project}:ext/ghcr` becomes
`{project}:ext/ghcr.io/huggingface/text-embeddings-inference`, and the sibling `ExternalSymbol` that carried the real
name is no longer minted. The old node is left with no importers and is swept by `gc_orphaned_reference_nodes`. No
`SCHEMA_VERSION` bump — that would drop the vector indices
([ADR-0024](./0024-schema-version-bump-drops-vector-indices.md)) for a change that rewrites itself on the next index.

**A declaration with no version constraint now writes an edge.** `dependencies = ["loguru"]` and `FROM nginx` both
produce `DEPENDS_ON` with `version = ""`, which `atlas deps` renders as `-`. An empty constraint carries no claim, so it
must not _conflict_ with a real one either — otherwise a name declared unpinned in one ecosystem and pinned in another
would collapse, losing both the version and the declaration, which is strictly worse than either input.

**Measured on this repo**, indexing the same 299 files twice into a throwaway SQLite graph, once at the commit before
this change and once after:

| tier         | before | after  |
| ------------ | ------ | ------ |
| `declared`   | 22     | **57** |
| `stdlib`     | 46     | 46     |
| `undeclared` | 49     | **15** |

Ten of the fifteen that remain are GitHub Actions (below). The rest — `markupsafe`, `msgspec`, `rich`, `opentelemetry` —
are genuinely transitive, or in the last case collapsed, because nine `opentelemetry-*` distributions share one import
name and disagree about the constraint. That is the tier working rather than failing.

Every image also gained its version: `memgraph/memgraph-mage` at `3.12.0`, `valkey/valkey` at `8-alpine`, `python` at
`3.14-slim`, all of which read as `None` before.

**GitHub Actions stay undeclared.** `uses: actions/checkout@v4` is a pinned declaration, but it lives in
`.github/workflows/*.yml`, which is a directory of many files rather than a root manifest, and
`_parse_dependency_versions` probes `project_root / filename`. Reaching them needs a manifest concept that is not one
file at one path.

**Go module paths keep the defect this fixes for images.** `github.com/spf13/cobra` still resolves to a package called
`github`. The marker would fix it identically, and deliberately is not applied: it would change the uid of every Go
external package on every indexed graph, which is a migration rather than a fix.

## Alternatives considered

**A `provenance` rule type on `ImportanceSettings`.** Rejected: see decision 1. It would have reused an existing,
well-tested mechanism, but its default-empty contract is load-bearing and inverting it for one rule type would have cost
more than the table saved.

**Re-parsing image names after the fact.** The truncation could have been repaired in a later pass, keying on "looks
like a registry hostname". Rejected: the name is minted in `resolve_imports`, so the repair would run against nodes that
already have edges, and the heuristic it needs is exactly the one decision 4 says cannot be written.

**Ecosystem-qualified uids** (`docker.io/library/python` rather than `python`), which would end the collision between
the Redis _image_ and the Redis _client library_ sharing one `ext/redis` node. Deferred: it needs the parsed language to
reach the graph, and `graph/client.py` cannot import `parsing/ast.py`. Today the two collapse under
`_parse_dependency_versions`'s conflict rule, which is honest — one node, two claims, no answer — rather than wrong.

## References

- ADR-0041 — the version lives on the `DEPENDS_ON` edge, not the node
- [ADR-0052](./0052-rrf-rank-space-has-no-epsilon.md) — the constraint the ranking factor sits inside: a multiplier
  orders what a channel returned, it cannot guarantee that something appears
- `search/engine.py::_boost_results`, `schema.py::split_image_reference`,
  `indexing/orchestrator.py::_optional_declarations`
