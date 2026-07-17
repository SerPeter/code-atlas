# Knowledge Vault — SCHEMA.md

This directory (`docs/`) is the knowledge vault: an Obsidian-compatible, zettelkasten-style note collection that lives
in the same graph as the code it documents. Ordinary prose docs and ADRs coexist here with vault notes — a file becomes
a `Note` node the moment it carries vault frontmatter; everything else keeps today's heading-level
`DocFile`/`DocSection` behavior. This file itself carries no frontmatter, so it's indexed as an ordinary doc.

Architecture and roadmap: see `.tasks/research/2026-07-11-knowledge-convergence-architecture.md`.

## Directory layout

```
docs/
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
supersedes: []
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

Both dialects index into the same `Note` label. `derived_from`/`supersedes` entries and wikilink targets are plain slugs
for same-project references, or `project:slug` for cross-project references (the global vault, the memory dir, or any
other indexed project).

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
  Zero-decision, append-only, never read existing files first. Both `docs/inbox/` and the Claude Code memory dir are
  equivalent draft piles — write wherever is cheapest; consolidation is the authoritative router, not capture-time
  judgment.
- **Consolidate** — `atlas dream --report` (Phase 4 — not built yet) plus the `dream-mode` skill turns drafts into
  durable `notes/`/`decisions/` zettels, or promotes them to the global vault or out of the graph entirely (rules,
  skills, CLAUDE.md).
- **Note style** — zettelkasten-atomic: one subject per note. Meaning lives in links, not prose length. Never restate
  what the code graph already indexes — notes hold rationale, decisions, incidents, cross-cutting behavior only.
