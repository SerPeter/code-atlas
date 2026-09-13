---
title: "ADR-0055: A stub is a signature on the node that already exists"
tags: [adr, indexing, external, parsing, schema]
kind: decision
---

# ADR-0055: A stub is a signature on the node that already exists

## Status

Accepted (2026-09-13) — completes ATL-191 (P3–P5) and builds on
[ADR-0054](./0054-an-external-name-is-weighted-by-how-deliberately-it-was-chosen.md), which made dilution a ranking
problem so that indexing more of the external world became affordable. Bumps `SCHEMA_VERSION` to 19.

## Context

ADR-0054 gave every external name a provenance weight. It did not give any of them a _signature_:
`from pathlib import Path` still put a node called `Path` in the graph that said nothing about what `Path` is, and 561
such names existed on the reference graph.

The original design proposed reading type stubs and rejected reading everything:

> _Why not index everything? Token budget. Indexing all of numpy + pandas + sqlalchemy would add hundreds of thousands
> of nodes, bloat embeddings, and dilute search results._

Measured on this repo's environment, that fear is well founded and also mis-aimed. The resolvable third-party surface is
**3,287 files and 37.3 MB**, of which `litellm` alone is 2,068 files and 24.6 MB — 66% of the total for one dependency.
But almost none of that is API surface. The _entrypoint_ of each package — the single file its public names come from —
totals **0.68 MB across 28 packages**, a 55× reduction, and it is precisely the "what can I call on this" the feature
exists to answer.

Two further measurements shaped the design. `py.typed` turned out to be a red herring: it advertises that the _source_
is annotated, so resolving to it means parsing the whole library, which is the cost being avoided. And a purely static
read of entrypoints reaches **100% of public names but only ~26% of their signatures** — the misses are structural, not
incidental.

## Decision

**1. A stub is a signature on the existing `ExternalSymbol`, not a new node type.**

`resolve_imports` already mints an `ExternalSymbol` per imported name. The stub pass fills those in and adds one node
per entrypoint nothing has imported yet. No new label, no second model to keep in sync, and a symbol the project
actually calls and one it merely _could_ answer a search identically.

**2. The entrypoint is the unit, and the whole tree is opt-in.**

`[libraries] full_index` widens one package from its entrypoint to every module beneath it. It is an explicit list
because the difference is 1 KB against 25 MB.

**3. Resolution prefers a hand-written stub, and `py.typed` is not a case.**

`.pyi` bundled beside the module → a `-stubs` distribution → the source. A `py.typed` branch was written first and
deleted: it behaved identically to `source` while implying a stub existed where none did.

A **frozen** stdlib module still resolves. `os`, `io`, `abc` and `codecs` have reported `origin == "frozen"` since 3.11,
and treating that as unresolvable silently lost `os` — one of the most-imported names on any graph.

**4. Static by default; importing is opt-in and named for what it costs.**

`[libraries] introspect = false`. The static path never executes third-party code — `find_spec` locates a module without
running it. What that forgoes is specific and worth stating rather than hiding:

| pattern                                                                             | static                                                 |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `from .x import *`                                                                  | reached — the star targets are scanned                 |
| re-export named in `__all__`                                                        | reached — one hop to the defining module               |
| a package re-exporting from _itself_ absolutely (`from pathlib._local import Path`) | reached — the self-prefix is treated as relative       |
| lazy `__getattr__` (PEP 562)                                                        | **no** — the name-to-module map is computed at runtime |
| compiled extension                                                                  | **no** — there is no Python source                     |
| runtime-generated class                                                             | **no** — it does not exist until import runs           |

Only the last three genuinely require importing. `introspect = true` reaches them by reading `inspect.signature` off the
live object, and pays for it by running that package's import-time code inside the indexer — network calls, CUDA
initialisation and thread spawning are all real in the wild. A package that raises on import is skipped whole and keeps
its static read.

**5. Signatures and docstrings come from the existing Python parser.**

`.pyi` is already a registered Python extension, so a type stub parses with no special case, and `parse_file` already
formats signatures, extracts docstrings, decides visibility and maps labels. `indexing/stubs.py` adds only what
`parse_file` does not expose — the `__all__` list and the re-export map — as 40 lines of `ast`. There is no second
Python parser.

**6. The invalidation key is the distribution version, checked before anything is read.**

The source lives in site-packages, not the repo, so no `file_hash` gate covers it and a stub parsed from numpy 2.4 would
otherwise persist across an upgrade. `stub_version` on the `ExternalPackage` records what was read; an index skips a
package whose installed version has not moved. Stdlib modules use the interpreter version, which is exactly what changes
when they do.

**The check sits between resolution and extraction, and the order is the decision.** Resolving is a `find_spec` and a
`stat`; extracting is 181 files and 6s. Written the other way round first, a re-index did all the work and then
discarded it — 848 symbols rewritten in 9.35s on a graph where nothing had changed.

**7. `ExternalSymbol` becomes embeddable (`SCHEMA_VERSION` 19).**

Gated by `[libraries] embed_stubs`, default on. A library's public API changes far less often than the code calling it,
so these vectors are re-bought rarely — which is what makes the default defensible.

## Consequences

**A `SCHEMA_VERSION` bump drops the vector indices** ([ADR-0024](./0024-schema-version-bump-drops-vector-indices.md)).
The stored `embedding` properties survive; only the indices are rebuilt, so nothing is re-embedded. The existing
`EmbeddingsPresentError` guard still refuses to migrate from a process with embeddings disabled, which is the case that
once took semantic search down silently.

**Measured on this repo**, indexing 300 files into a throwaway SQLite graph: **3,443 entrypoints across 96 packages**,
1,162 of them with a signature, adding about 11s to a 30s index on the first run and ~0 on the next. `asyncio` went from
0 entrypoints to 91 once star targets were scanned; `jinja2`, `typer` and `tiktoken` from 0 to a real surface once a
package with no `__all__` was allowed to expose what it re-exports.

**The stub pass never fails an index.** It runs last, on a graph that is already complete and correct, and every failure
is logged and swallowed. A library whose entrypoint will not parse is a gap in enrichment, not an error.

**Coverage is a property of the indexing machine.** Only packages installed in atlas's own environment resolve. Indexing
a foreign repo yields provenance weights and no signatures — the same limitation ADR-0054 records for its
installed-gate, and the reason that gate switches itself off in exactly this situation.

**`pytest` and `urllib` still yield no signatures**, for opposite reasons: `pytest` re-exports from `_pytest.*`, which
is neither relative nor self-prefixed, and `urllib`'s entrypoint is genuinely empty. Both are honest outcomes of the
rules rather than bugs to paper over.

## Alternatives considered

**Importing by default.** ~98% signature coverage instead of ~26%. Rejected as a default because it executes arbitrary
third-party code inside the indexer; kept as `introspect`, so the decision belongs to whoever owns the machine.

**Vendoring typeshed.** Would give excellent stdlib and third-party coverage with no execution. Rejected for now:
neither `mypy` nor an importable `ty` is a dependency, so it means shipping and tracking a stubs corpus — a larger
decision than this epic, and one the entrypoint read makes less urgent.

**A separate `Stub` label.** Rejected per decision 1: it would double the model for external names and make "imported"
and "available" two different search results for the same function.

## References

- [ADR-0054](./0054-an-external-name-is-weighted-by-how-deliberately-it-was-chosen.md) — the provenance weight that
  makes this volume of external nodes survivable
- ADR-0015 — the embedded backend, whose conformance ledger both new methods join
- `indexing/stubs.py`, `graph/client.py::upsert_external_stubs`, `settings.py::LibrarySettings`
- `.specs/research/code-atlas.md:369-377` — the original design, and the token-budget objection this answers with an
  entrypoint rather than a whole tree
