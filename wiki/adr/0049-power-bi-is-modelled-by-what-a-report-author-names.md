---
title: "ADR-0049: Power BI is modelled by what a report author names"
tags: [adr, parsing, graph, power-bi]
kind: decision
---

# ADR-0049: Power BI is modelled by what a report author names

## Status

Accepted (2026-09-06)

## Context

A Power BI project is two halves that reference each other by name and nothing else. The semantic model (TMDL, under
`<Name>.SemanticModel/definition/`) holds tables, measures and their DAX. The report (PBIR, under
`<Name>.Report/definition/`) holds pages and visuals that display them. Neither half was usable:

- `*.tmdl` was not indexed at all — absent from `_DEFAULT_INCLUDE`, and nothing could have parsed it anyway, because
  `LanguageConfig` required a non-optional tree-sitter `Language` and TMDL has no public grammar.
- PBIR is JSON, so it fell to the generic structural handler and became a tree of `config_setting` leaves addressed by
  generated identifiers, with the field names buried inside their text where nothing searched them.

The whole point of indexing this at all is a question that crosses both halves and the warehouse below them: **what
breaks if I change this dbt model?** Answering it needs the model's tables, the measures over them, the visuals showing
those measures, and an edge joining the pipeline to the report.

## Decision

### 1. A language may have no grammar

`LanguageConfig.language` may be `None`, with `text_parse_func` supplied instead. TMDL is line-oriented and
indentation-scoped, which tree-sitter reaches only with a hand-written external scanner; vendoring one for a format
Microsoft versions is a standing maintenance commitment for a single file type.

Two fields rather than widening `parse_func`'s node parameter to `Node | None`. That widening was the smaller diff and
the worse design: all sixteen grammar-backed handlers would have had to declare a `None` they can never receive.
`__post_init__` enforces that exactly one handler is set and that it matches the grammar, at registration — the
alternative is a `TypeError` deep in `parse_file`, once per file, on a language somebody added months ago.

**The hatch skips the grammar and nothing else.** `_parse_hazard`, the `RecursionError` catch, the per-language timing,
the empty-`ParsedFile`-on-decline rule and content hashing all stay on the shared path. Those are what stop one
pathological file taking the indexer down, and a text parser is no less capable of meeting one.

Consequence for anyone touching `parsing/`: `Parser(config.language)` now needs a guard. `tests/support/langcov.py` has
none — see §6.

### 2. Names, not generated identifiers

TMDL's `lineageTag` and PBIR's 20-character object names are stable and useless to search for. They go to
`extra_properties`, where Cypher still reaches them. The uid carries the name the author typed.

Where an author typed nothing — real `visual.json` files carry no `displayName`, verified against public repositories —
the visual is named for its type and the generated id keeps the uid unique. Two bar charts on one page must not collapse
into one node.

### 3. The file's object is the file's node

Every parser emits exactly one `Module`/`DocFile` per file, and that is not a formality: only those labels appear in
`FILE_HASH_LABELS`, and the per-file hash gate is what lets an unchanged file skip parsing. A language emitting none
re-parses every one of its files on every indexing pass, forever, with no error anywhere — the bug `DocFile` was added
to fix for markdown.

So a PBIR file, which holds exactly one object, makes that object the `Module`. A `TypeDef` visual (as the design sketch
had it) would have carried no `file_hash`. TMDL files can hold several objects, so they get a document `Module` and hang
their tables off it.

### 4. Structural edges stay inside the file that states them

`_recreate_file_relationships` deletes a file's edges by their **source** node's `file_path`. That single rule decides
three otherwise-arbitrary modelling choices:

- **A model relationship is a node, not a table-to-table edge.** Declared in `relationships.tmdl` but sourced at a table
  living in `tables/<T>.tmdl`, the edge would never be deleted when the relationships file changed, and _would_ be
  deleted — unrecoverably — when the table was re-parsed. Its own node has neither problem, and is where cardinality and
  cross-filter direction belong anyway. Reachability becomes two hops, which `salesforce.py` already accepts for the
  same reason.
- **A page does not `DEFINES` its visuals.** They are sibling files. Containment is encoded in the uid
  (`...page.<P>.visual.<V>`), which answers "what is on this page" with a prefix match and needs no edge.
- **A table's measures and columns do `DEFINES`**, because they are in the same file.

### 5. `FEEDS` crosses the warehouse boundary

A BI table names the warehouse object it loads from; a dbt model names the object it produces; neither file mentions the
other. `resolve_warehouse_objects` joins them at monorepo level, because the two live in different sub-projects and
every per-project resolver is scoped to one.

The join is a case fold — warehouse identifiers are conventionally upper case and dbt model names lower case, so the
same object is written two ways by two tools. _Conventionally_, not by rule, which is why the two non-matching outcomes
carry as much design as the match:

| producers | result                                                                                                                                                                                                    |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| one       | `dbt_model -FEEDS-> pbi_table`; the stub's `IMPORTS` edge deleted and the stub removed once nothing points at it                                                                                          |
| several   | no edge; the stub survives carrying `confidence: "ambiguous"`. Two sub-projects owning a model of one name is a real thing in a monorepo, and picking one would be a coin flip written as structural fact |
| none      | the stub survives. A BI table reading an object no model produces is hand-built or is drift, and either way somebody wants to know                                                                        |

`FEEDS` is the only new `RelType`. The parser emits `IMPORTS -> warehouse.<obj>` and reuses `resolve_imports`' existing
stub lifecycle, where the design sketch would have added a second type for it.

`DEPENDS_ON` was rejected: it already carries two endpoint shapes (project-to-project, and Project → ExternalPackage
with a version), and a third would make the deletion queries harder to keep correct than one more name is worth.

### 6. TMDL has no extraction-coverage floor, deliberately

`langcov`'s `LangSpec` is built entirely from tree-sitter node-type names, so a grammar-less language cannot be
expressed in it. Worse, `Coverage.named_funcs` and `Coverage.calls` both return **1.0 when there is nothing to measure**
— so wiring TMDL in naively records a floor of 1.0 that guards nothing, which is worse than no floor at all.

What the floor would have caught, `duplicate_uids`, is covered directly by unit tests instead — and those found a real
collision rather than a hypothetical one: folding five characters to `_` made `'Sales: YTD'` and `'Sales YTD'` one uid.
Only `.` and `:` fold now, and a folded name earns a suffix derived from the name rather than from arrival order.

Making the floor real means a parallel text-based measurement and a change to a ratchet currently protecting nine
languages. That is its own piece of work, not a checkbox.

## Consequences

- Indexing a PBIP project now costs two more parsers and produces far fewer nodes than the generic handler did for the
  same bytes, since a whole report definition folder collapses to one node per object.
- `blast_radius` on a dbt model reaches the tables, measures and visuals downstream of it. That is the claim this whole
  epic exists to make good.
- A wrong case fold shows up as a surviving `ext/warehouse.*` stub, which is queryable — not as a missing edge nobody
  notices.
- Not captured, and each is its own story: `reportExtensions.json` report-level measures (DAX living in the report,
  which the TMDL parser never sees); Direct Lake partitions whose `source` takes children with no `=`; `createOrReplace`
  script-form TMDL, which shifts the whole document one indent deeper; and bookmarks.
- **Read real files before extending either parser.** Both formats are documented and both documents disagree with
  reality in ways that produce silent zero-edge parsers — most sharply, PBIR's own `semanticQuery` schema documents
  `SourceRef.Source` with `additionalProperties: false` while every real `visual.json` carries `SourceRef.Entity`.
