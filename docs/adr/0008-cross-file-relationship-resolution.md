# ADR-0008: Cross-File Relationship Resolution and Qualified-Name Extensions

## Status

Accepted — amends [ADR-0007](./0007-qualified-name-strategy.md)

## Date

2026-07-12

## Context

A full multi-agent code review (2026-07) found that every relationship type resolved by matching a
parser-emitted name against a graph node uid, name, or file-derived parent path was silently
dropping edges whenever the target lived in a different file or the naming scheme diverged from
what the parser assumed:

- **IMPLEMENTS**: TypeScript, Java, C#, PHP, and Rust parsers all emit the bare interface/trait name
  (`to_name = "Runnable"`), but `_create_relationships` matched IMPLEMENTS by uid — every edge was
  dropped, silently, in every language.
- **Cross-file members**: Go methods declared in a different file than their receiver type, and C++
  out-of-line method definitions (`Widget::draw` in `widget.cpp`), fabricated a parent uid from the
  _member's own file_ module — a uid that never exists when the type lives elsewhere. DEFINES edges
  for these members were dropped entirely.
- **JVM overloads**: `qualified_name` had no parameter-signature component, so overloaded
  Java/C# methods collapsed onto one uid — `_batch_create_entities` silently kept only the
  last-parsed overload.
- **Import resolution**: Python (and, by the same defect class, Java/C#) qualified names included
  the filesystem source-root directory (`src.code_atlas.events`), so absolute internal imports
  (`from code_atlas.events import ...`) never matched and were misclassified as external.

ADR-0007 documents the `uid`/`name`/`qualified_name` property set and the `get_node` matching
cascade, but says nothing about how _relationships between entities in different files_ should be
resolved, or how source roots and overloads should be reflected in `qualified_name`. This ADR fills
that gap with the conventions actually implemented.

## Decision

### IMPLEMENTS: route by shape, not by rel type

`_create_relationships` partitions incoming IMPLEMENTS relationships by whether `to_name` contains
`:`:

- **uid-shaped** (`":" in to_name`, i.e. `{project_name}:{qualified_name}`): matched by uid, exactly
  like CALLS/USES_TYPE. Used only by the Python `ClassOverridesDetector` (Callable→Callable abstract
  method implementations), which has a project-wide symbol table available at detection time.
- **bare-name-shaped** (no `:`): resolved the same way INHERITS already was — `MATCH (a {uid:
from_uid}), (b:TypeDef {project_name, name: to_name})` — exact match on `TypeDef.name`, scoped to the
  same project. Ambiguous names fan out to every match; zero matches silently produce no edge (an
  external/stdlib interface, for example).

No parser change was needed: every bare-name emission site already matched this shape. The fix is a
single discriminator in the graph client, which now generalizes to any future language's IMPLEMENTS
emission without a per-language client change.

### Cross-file members: `parent_type_name` deferred resolution

A DEFINES relationship whose `properties` contains `parent_type_name` is a _deferred member-parent_
relationship, structurally distinct from a direct DEFINES:

- `from_qualified_name` = the declaring file's own Module uid (the fallback parent).
- `to_name` = the member's full uid.
- `properties["parent_type_name"]` = the parent type's bare name, normalized (no pointer sigils, no
  generic type-parameter lists, no `::`/namespace qualifiers).
- `properties["parent_scope"] = "package"` (optional) restricts resolution to the member's directory
  — Go emits this (Go's package = directory); C++ does not (translation-unit scoped).

These relationships are excluded from `_create_relationships`'s immediate-resolution paths (like
CALLS/USES_TYPE) and instead flow to `GraphClient.resolve_member_defines`, which resolves
`parent_type_name` to a concrete `TypeDef` uid — unique match wins; ambiguous or zero matches fall
back to the already-attached Module DEFINES edge rather than guessing. Resolution reruns whenever
either the member's file or the parent's file is reprocessed, so a member's edge self-heals once its
parent's file is indexed, regardless of processing order.

Go and C++ (and, by the same defect class, Rust impl-block methods for externally-defined types)
all emit this shape.

### Overload disambiguation: parameter-type suffix

For Java and C# — the two registered languages with real method overloading — `qualified_name`'s
final segment gets a parameter-type-signature suffix whenever a sibling scope (class/interface/enum/
struct body) contains more than one member with the same declared name:

```
name[<TypeParams>](<t1>,<t2>,...)
```

Types only (no parameter names, no defaults), comma-joined, derived deterministically from the
tree-sitter parameter list so declaration order never affects the suffix. Varargs (`String... args`,
C# `params`) render as `[]` rather than `...` — the qualified-name segment must stay dot-free (dots
are the qualified-name path separator), and `T[]` and `T...` are the same erased type in both
languages, so no collision risk. Non-overloaded members keep the bare segment unchanged. The `name`
node property stays the short, unsuffixed name in all cases — only `qualified_name`/`uid` carry the
suffix.

### Source-root stripping

Python already stripped nothing from `qualified_name` beyond dropping `.py` and collapsing
`__init__.py`; a project laid out as `src/pkg/module.py` (the standard `src`-layout convention)
produced `qualified_name = src.pkg.module`, so `import pkg.module` (what the interpreter and every
other tool sees) never matched. `qualified_name` construction now strips a leading `src` path
segment when present (`parts[0] == "src" and len(parts) > 1`), for Python, Java, and C# alike —
Java/C# gained the same rule to fix package-based import resolution (`import pkg.module.Foo` is
package-based, not file-path-based, and package paths never include a conventional source root).

Relative Python imports (`from . import x`, `from ..pkg import y`) are now resolved at parse time,
using the importing module's own package path — the parser is the only component that knows a
file's package membership, so resolving dots there (rather than passing them through to the graph
client) means `to_name` always lands in the same namespace as stored `qualified_name`s, and
resolution becomes exact-match instead of a client-side guess.

## Consequences

### Positive

- IMPLEMENTS resolution is now a single, language-agnostic code path — a new language's parser needs
  no client-side change as long as it emits bare names for interface/trait targets.
- Cross-file member attachment self-heals regardless of file processing order, without a full
  project-wide symbol table at parse time.
- Overload uids are stable across unrelated edits (parameter-type-derived, not declaration-order- or
  hash-derived) — editing one overload's body never changes a sibling overload's uid.
- Import resolution matches what the language's own import system actually resolves against, not an
  incidental filesystem convention.

### Negative

- **uid churn**: every Python/Java/C# entity's uid changed when source-root stripping landed (schema
  v3 migration forces a full reindex — see [ADR-0009](./0009-event-pipeline-durability-contract.md)).
  A project that genuinely has a top-level package literally named `src` (rare, but not impossible)
  would have its real `src` package silently absorbed into the strip rule; this is a known, accepted
  edge case.
- **Overload uid churn on refactor**: adding or removing an overload changes every other overload's
  suffix only if the added/removed one has a colliding erased-type signature; ordinary edits are
  stable, but this is a real (rare) churn source.
- The `parent_type_name` deferred-resolution path adds one more relationship shape client code must
  recognize; a language whose parser doesn't know about it emits a broken cross-file member edge no
  differently than before this ADR (fails open to the old fallback behavior, not worse).

### Risks

- The `:` discriminator for IMPLEMENTS assumes no language ever legitimately emits a bare interface
  name containing `:` (C++ `ns::Iface`-style scoped names, if IMPLEMENTS support is ever added for
  C++, would need either pre-stripping to a bare name or a different discriminator).
- `resolve_member_defines`'s "ambiguous → fall back to Module DEFINES" rule means a genuinely
  ambiguous cross-file member (two same-named types in the same `parent_scope`) silently gets the
  weaker Module-level edge instead of an error — acceptable given the alternative (guessing wrong)
  is worse, but not surfaced to the user.

## Alternatives Considered

### IMPLEMENTS: parser-side uid emission

Each parser would need a project-wide symbol table at parse time to resolve interface names to
uids before emitting the relationship. Rejected: `parse_file()` is a pure, per-file function by
design (no I/O, no project state) — building this backward would touch five parser files instead of
one client function, and reintroduce exactly the ordering-dependency problem `parent_type_name`
deferred resolution was built to avoid.

### Cross-file members: post-batch resolution alongside CALLS

CALLS/USES_TYPE already have a deferred-resolution path; member-DEFINES could have reused it
directly. Rejected in favor of a dedicated `resolve_member_defines` because the resolution rule
differs materially (name+scope match against `TypeDef`, not a call-heuristic cascade against
`Callable`), and conflating the two would make both harder to reason about.

## References

- [ADR-0007: Qualified Name Resolution Strategy](./0007-qualified-name-strategy.md) — the base
  naming scheme this ADR extends
- [ADR-0009: Event Pipeline Durability Contract](./0009-event-pipeline-durability-contract.md) —
  schema v3 migration that accompanies the uid churn from source-root stripping
- `src/code_atlas/graph/client.py` — `_create_relationships`, `resolve_member_defines`
- `src/code_atlas/parsing/languages/{python,typescript,jvm,cpp,go,rust,php}.py` — per-language
  emission sites
