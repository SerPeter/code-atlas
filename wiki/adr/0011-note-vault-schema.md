# ADR-0011: Note Label and the Knowledge Vault Schema

## Status

Accepted

## Date

2026-07-13

## Context

A separate memory system (`neo-memoria`) was designed to give AI agents persistent knowledge — capture, consolidation,
entity linking — on its own storage backend. A parallel investigation
(`.tasks/research/2026-07-11-knowledge-convergence-architecture.md`) concluded that dissolving it into code-atlas's
existing graph is strictly better: code uids (`{project}:{qualified_name}`) already solve entity resolution, the file
watcher already gives live incremental indexing, and a note anchored to code can go stale mechanically (via
`content_hash`) rather than through time-based decay.

This ADR covers the schema decision that makes `docs/` an Obsidian-compatible, zettelkasten-style knowledge vault living
in the same graph as the code it documents — Phase 1 of that architecture.

## Decision

- **New node label `Note`** — one node per file, not per heading (unlike `DocSection`). A markdown file becomes a `Note`
  when its YAML frontmatter matches one of two dialects: the vault's own (`id` + `kind` in `draft|note|decision`) or the
  Claude Code harness memory format (`name` + `description` + `metadata.type`) already used by
  `~/.claude/projects/<slug>/memory/`. Files without either dialect keep today's `DocFile`/`DocSection` heading-level
  extraction — `docs/` mixes both.
- **uid scheme**: `{project}:note:{slug}`, where `slug` is the frontmatter `id`/`name`. The vault convention requires
  filename (sans `.md`) to equal that slug, so Obsidian's filename-based `[[wikilink]]` resolution and the graph's uid
  scheme coincide by construction — no separate entity-resolution or alias table needed.
- **Three new relationship types**: `LINKS_TO` (`[[wikilinks]]`, same-project or `project:slug` cross-project),
  `DERIVED_FROM` and `SUPERSEDES` (frontmatter `derived_from`/`supersedes` lists — future dream-mode consolidation
  provenance). All three resolve via a direct uid-to-uid `MATCH`+`MERGE`: an unresolved target simply matches nothing,
  so a broken wikilink creates no edge and no phantom node.
- **`DOCUMENTS` extended to Note**: the existing heuristic doc→code linker (backtick symbol mentions, file-path
  mentions) now runs over Note bodies too, and `_create_doc_links`'s from-side match accepts `Note` alongside
  `DocSection`.
- **Deleted `MOTIVATED_BY`**: a dead schema slot (declared, never emitted by any code). A decision→code view is the
  query `(n:Note {kind:'decision'})-[:DOCUMENTS]->(e)`, so a second edge type would have carried zero information the
  extended `DOCUMENTS` edge doesn't already provide, while doubling the staleness-maintenance surface once anchor
  invalidation (a future phase) lands.
- **`ParsedEntity.extra_properties`**: a generic `dict[str, Any]` passthrough for frontmatter fields with no fixed
  schema column, merged via Cypher `SET n += $map` (a no-op for the empty dict every non-Note entity carries). Folded
  into `content_hash` only when non-empty.
- **Relationship-routing completeness guard**: a new relationship type that isn't wired into
  `GraphClient._create_relationships`'s routing is silently dropped — the exact failure class the dead `MOTIVATED_BY`
  slot belonged to. `graph/client.py` now keeps an explicit routing registry
  (`_UID_ROUTED_REL_TYPES`/`_NAME_ROUTED_REL_TYPES`/`_POST_BATCH_REL_TYPES`/`_OUT_OF_BAND_REL_TYPES`) and asserts at
  import time that every `RelType` is covered by one of them, mirroring `schema.py`'s `_validate_schema_completeness()`
  guard for node labels.
- **SCHEMA_VERSION 3 → 4**, with a migration that clears stored `file_hash`/`git_hash` project-wide (same shape as the
  v3 migration): `extra_properties` changes `_compute_content_hash`'s formula for every entity, not just Notes —
  appending even an empty element to the `\0`-joined hash-input list still shifts every existing hash value, because the
  extra element changes where the null-byte separators land.

## Consequences

### Positive

- Entity resolution for knowledge — neo-memoria's self-described hardest problem — is nearly free: code uids are the
  canonical entities a note anchors to, and a note's own uid is deterministic from its slug.
- Existing Claude Code memory files become searchable graph nodes the moment their directory is indexed, with no format
  migration required — the harness dialect is recognized as-is.
- The completeness guards (schema.py's node-label one, now graph/client.py's relationship-routing one) make "add an enum
  value, forget to wire it up" a hard import-time failure instead of a silent no-op.

### Negative

- A second, harness-specific frontmatter dialect adds a permanent branch in the parser (`is_harness_dialect` in
  `_parse_markdown_note`) that must be kept in sync if that format ever changes independently.
- The v4 migration forces a full re-hash of every entity in an existing database on upgrade (cheap per-file, but
  non-zero cost proportional to codebase size).

### Risks

- `extra_properties` is an open passthrough dict — nothing currently validates frontmatter shape beyond the two
  dialect-detection checks, so malformed vault files fail closed (silently skip note mode) rather than surfacing a clear
  error. Acceptable for Phase 1; worth a lint pass once dream-mode (Phase 4) exists.
- Filename ≠ frontmatter `id` is not yet enforced anywhere but `docs/SCHEMA.md`'s convention text — nothing currently
  lints for divergence. Deferred to the dream-mode report (Phase 4) per the architecture doc.

## Alternatives Considered

### Alternative 1: A separate `knowledge/` vault directory, distinct from `docs/`

- Keeps ordinary prose docs and vault notes in physically separate trees.
- Rejected: `docs/` already holds ADRs and architecture docs that belong in the same Obsidian-navigable space as new
  notes: two parallel doc trees with different conventions is more surface area for no real benefit, and
  frontmatter-triggered note mode already lets the two kinds of file coexist per-file.

### Alternative 2: Keep `MOTIVATED_BY` and add the new Note edges alongside it

- Would preserve the pre-existing (if dead) schema slot.
- Rejected: it would have carried zero information the extended `DOCUMENTS` edge doesn't already express, while
  requiring its own staleness bookkeeping once anchor invalidation ships — a maintained duplicate of a query, not a
  distinct relationship.

## References

- [Knowledge convergence architecture proposal](../../.tasks/research/2026-07-11-knowledge-convergence-architecture.md)
  (local, gitignored)
- [ADR-0009: Event Pipeline Durability Contract](./0009-event-pipeline-durability-contract.md)
- [docs/SCHEMA.md](../SCHEMA.md) — vault conventions
