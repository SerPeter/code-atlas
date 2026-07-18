# ADR-0021: The module_summary Outline Format

## Status

Accepted — supersedes nothing; records a format that had been evolving undocumented since `summarize_module` shipped.
Amends [ADR-0019](./0019-module-granularity-community-detection.md) only insofar as `generate_diagram` now reuses its
partitioner for large import graphs.

## Context

`summarize_module` and `generate_diagram` both return a rendered **string**, not a record set, because repeating
`{"qualified_name": ..., "signature": ...}` keys per entity costs more tokens than the information they label. That
makes the notation itself an interface, and it had no ADR — so its conventions were being extended by whoever touched
the renderer, with no record of which parts were deliberate.

Two questions had never been answered with evidence:

1. **Is the notation standard, or invented?** Mostly invented. `+ - # ~` for public/private/protected/internal is
   genuine UML class-diagram notation; indentation-for-containment and `#`-for-comment are generic.
   `SCOPE`/`NAMES`/`LEGEND`, `a > b`, `L40-67`, `Cn:` and `*N` are ours.
2. **Is it understood by a reader who has never seen it?** Unmeasured until now. Token costs were measured precisely;
   comprehension was asserted from locality statistics.

## Decision

Keep a bespoke format, and justify it rather than assume it.

**Why not `.pyi` stubs**, the obvious standard: they are Python-only across a nine-language indexer, carry no line
numbers, and — decisively — cannot express FAN-IN/FAN-OUT. A stub describes a file; this describes a file's _position in
a graph_, which is the whole reason the tool exists alongside `get_context`. A stub-syntax variant was built and
measured; it is cheaper only because it carries less.

**Validate comprehension empirically, not by assertion.** Four variants were rendered from the same real data and read
by independent agents with no repo context, no tools beyond reading one file, and explicit permission to answer "cannot
tell from the text".

The control that mattered was the format **with the LEGEND line stripped**. That reader independently recovered: the
visibility marker, `L<n>` _including_ the "range only when >= 20 lines" rule, the `>` / `<` direction reversal, the
indentation semantics, the `[k=v]` edge annotations, and the `ext/` prefix. Both readers also resolved the hardest
factual question — which outside file uses the scope most — to the correct two-way tie at 22 edges. The notation is
therefore self-evident rather than legend-dependent, and the legend is a convenience, not a load-bearing decoder ring.

**Fix what blind reading exposed.** Every one of these was invisible from the inside:

- The header read `3 module(s)` above four file blocks, because a package `__init__` is a `Package` node with no
  `Module` row. It now counts rendered file blocks.
- `field` and `enum_member` entities were not indented under their owning class — ~50 of one scope's entities looked
  like module-level globals. They arrive with `parent_qn` unset (no `DEFINES` edge, unlike methods), so the parent is
  now derived from the qualified name.
- The trailing `*` marking an external target was redundant with the `ext/` prefix that every external qualified name
  already carries (verified across all 585 in the index). Removed.
- The import diagram never stated its clustering criterion or the unit of `*N`, and rendered out-edges only — so the
  most depended-upon modules had the emptiest lines. In-degree is now shown as `(<-N)`.

**Keep the tier system keyed on rendered size** (see the `_TIER_*` constants), never on the shape of `path`: one module
can hold 316 entities while a package holds 12.

## Consequences

The format is now documented, and the LEGEND line is known to be redundant for a capable reader — which means it can be
dropped later as a token saving without breaking comprehension, if that is ever worth 56 tokens. It is not dropped now:
it costs little and helps a weaker reader.

Comprehension has been measured **once, on one scope, by one model family, for the Python-only content in this index**.
A cross-language check would need a repo where visibility is a keyword rather than a naming convention; the "emit the
visibility marker only when it is not already recoverable from the name or signature" rule exists for exactly that case
and is untested against real Java or C#.

The notation is still not in any model's training distribution. That cost is real and unquantified — the blind readers
understood it, but a comparison against a standard format for _equal information_ was not run, because no standard
format carries the boundary sections.

## Alternatives Considered

**Abbreviating keywords** (`async def` -> `afn`, `class` -> `cls`). Rejected on measurement: it saves 6,123 characters
and **exactly zero tokens**, because BPE already encodes `async`, `class`, `def` and `constant` as one token each while
`afn` costs two. Abbreviating every keyword in the corpus saves 84 tokens, 0.13%, all of it from `enum_member` -> `em`.

**Dropping indentation.** It genuinely costs 1 token per entity — but it is the cheapest possible encoding of class
membership, and nesting beyond the first level is free. Kept.

**Dropping the redundant `(path)` beside each module's qualified name.** Measured at -399 tokens (0.6%). Rejected: a
dotted name does not tell you the file lives under `src/`, and the outline exists to send a reader to a location.

**JSON for the diagram.** Measured worst of seven candidates — `indent=2` costs _more_ than Mermaid (1.21x), and compact
JSON puts the whole graph on one line, destroying any ability to reference a region of it.
