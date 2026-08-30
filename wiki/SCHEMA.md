# Knowledge Vault — SCHEMA.md

This directory (`wiki/`) is the knowledge vault: an Obsidian-compatible, zettelkasten-style note collection that lives
in the same graph as the code it documents. Ordinary prose docs and ADRs coexist here with vault notes — a file becomes
a `Note` node the moment it carries vault frontmatter; everything else keeps today's heading-level
`DocFile`/`DocSection` behavior. This file itself carries no frontmatter, so it's indexed as an ordinary doc.

Architecture and roadmap: see `.tasks/research/2026-07-11-knowledge-convergence-architecture.md`.

## Directory layout

```
wiki/
├── SCHEMA.md          # this file
├── HOME.md             # generated landing page (Phase 4 — not built yet)
├── inbox/              # quick-capture drafts — committed, travel with the branch
├── notes/               # durable atomic zettels (dream-mode output only)
├── decisions/           # ADR-style decision notes (frontmattered)
├── archive/             # superseded/merged notes — stub + git-history pointer
└── adr/, architecture.md, ...   # ordinary docs — coexist, migrate per-file
```

## Note frontmatter

A markdown file becomes a `Note` node when its frontmatter matches one of two dialects:

**Vault dialect** — requires `id` + `kind`:

```yaml
---
id: watcher-debounce-selfcancel # REQUIRED. Must equal the filename (sans .md).
kind: draft | note | decision # lifecycle stage
tags: [indexing, asyncio]
aliases: [flush self-cancel] # Obsidian-compatible wikilink aliases
anchors: # explicit code links (Phase 3 — not resolved yet)
  - code-atlas:code_atlas.indexing.watcher.FileWatcher._flush
created: 2026-07-11
derived_from: [inbox-2026-07-10-flush-bug] # dream-mode provenance
supersedes: [] # notes this one replaces — the target is demoted in search
contradicts: [] # notes this one disagrees with, unresolved — both are flagged, neither demoted
archived: false
---
```

**Claude Code memory dialect** — requires `name` + `description` + `metadata.type` (the format already used by
`~/.claude/projects/<slug>/memory/`):

```yaml
---
name: watcher-debounce-selfcancel
description: One-line summary of the finding.
metadata:
  type: user | feedback | project | reference
---
```

Both dialects index into the same `Note` label. `derived_from`/`supersedes`/`contradicts` entries and wikilink targets
are plain slugs for same-project references, or `project:slug` for cross-project references (the global vault, the
memory dir, or any other indexed project).

### Supersession and contradiction

`supersedes:` and `contradicts:` are the two keys that change how a note _ranks_, not just how it links.

- **`supersedes: [older-note]`** creates a `SUPERSEDES` edge, and the target is stamped `superseded_by`. Search demotes
  it and every hit carries the successor's uid. It is demoted, not hidden — a replaced note is the provenance for the
  one that replaced it.
- **`contradicts: [other-note]`** creates a symmetric `CONTRADICTS` edge, and **both** ends are stamped. Neither is
  demoted: in an unresolved contradiction neither side is known wrong, so ranking one down would be the system picking a
  winner nobody picked. Both hits say the dispute exists.

Neither is ever written automatically. There is no contradiction _detection_ — the edge exists so a human, or dream-mode
consolidation landing on "contradiction, can't auto-resolve", has somewhere durable to record the verdict.

## How frontmatter is stored

Frontmatter lands on the node as a single `frontmatter` **map** property, not as individual top-level properties:

```cypher
MATCH (n:Note) WHERE n.frontmatter.metadata.type = 'feedback' RETURN n.name
MATCH (n:DocSection) WHERE 'sre' IN n.frontmatter.audience RETURN n.uid
CREATE INDEX ON :Note(frontmatter.metadata.type);   -- nested keys are indexable
```

Nesting rather than flattening, for three reasons:

- **Flattened keys collided.** `SET n += extra_properties` runs _after_ the schema fields are written, so a frontmatter
  key that happened to share a name with one of them overwrote it. Every Claude Code memory note had its `name` replaced
  by its own slug. `uid`, `file_path`, `source` and `docstring` were equally reachable.
- **A deleted key never disappeared.** `+=` only ever adds. The map is replaced wholesale, so removing a key from the
  file removes it from the graph.
- **Nothing is lost for querying.** Memgraph matches and indexes nested keys directly.

Which keys appear depends on the mode:

| Mode                   | Stored                                                                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Note**               | Everything except the keys note mode _consumes_: `id`, `kind`, `tags`, `derived_from`, `supersedes`, `contradicts`, `anchors` — already queryable as `n.kind`, `n.tags`, or as edges. |
| **DocFile/DocSection** | The block verbatim; doc mode consumes nothing. Propagated from the file to its sections, because a section is what search actually returns for a doc.                                 |

A file with no frontmatter gets `frontmatter = null` on its `DocFile` and nothing at all on its `DocSection`s — sections
are embeddable, and stamping a null on every section of every plain markdown file would change its `content_hash` and
re-embed the repo to record an absence.

YAML values Bolt cannot carry (a `!!set`, a tuple) are coerced to lists or strings rather than failing the write, so
arbitrary frontmatter cannot take a file's indexing down. Strings, numbers, booleans, dates, lists and nested maps all
pass through unchanged.

### Ranking on frontmatter

`[search.importance]` in `atlas.toml` turns a frontmatter key or a path glob into a multiplicative ranking factor — see
the commented block in `atlas.toml`. Keys note mode promotes out of the map stay addressable by their frontmatter name,
so `key = "kind"` and `key = "tags"` work on notes despite not being in `n.frontmatter`.

## Identity

**Filename (sans `.md`) must equal the frontmatter `id` (or `name` for the memory dialect).** This makes Obsidian's
filename-based `[[wikilink]]` resolution and the graph's `{project}:note:{slug}` uid scheme coincide by construction.
Two files sharing an `id` will silently merge into one graph node — don't do that.

## Links

- `[[target]]` / `[[target|alias]]` — resolves to `LINKS_TO` (same-project) or a cross-project Note when `target` is
  `project:slug`. `[[note#heading]]` / `[[note^block]]` resolve to the target note only (v1 — fragment dropped). An
  unresolved target creates no edge (no phantom nodes) — check dangling links with
  `MATCH (n:Note) WHERE NOT (n)-[:LINKS_TO]->() RETURN n` style queries once a dream-mode lint report exists (Phase 4).
- Backtick symbol mentions (`` `FileWatcher._flush` ``) and file-path mentions resolve heuristically to `DOCUMENTS`
  edges onto code entities — the same mechanism ordinary docs use.
- `anchors:` frontmatter → explicit `DOCUMENTS(link_type='anchor')` edges with staleness tracking. **Not implemented
  yet** — Phase 3.

## Workflow

- **Capture** — write a draft to `inbox/` (via the `remember` MCP tool, once built in Phase 2, or by hand).
  Zero-decision, append-only, never read existing files first. Both `wiki/inbox/` and the Claude Code memory dir are
  equivalent draft piles — write wherever is cheapest; consolidation is the authoritative router, not capture-time
  judgment.
- **Consolidate** — `atlas dream --report` (Phase 4 — not built yet) plus the `dream-mode` skill turns drafts into
  durable `notes/`/`decisions/` zettels, or promotes them to the global vault or out of the graph entirely (rules,
  skills, CLAUDE.md).
- **Note style** — zettelkasten-atomic: one subject per note. Meaning lives in links, not prose length. Never restate
  what the code graph already indexes — notes hold rationale, decisions, incidents, cross-cutting behavior only.
