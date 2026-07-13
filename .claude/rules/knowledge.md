# Knowledge Vault Pointer

This repo's `docs/` directory is a knowledge vault indexed into the code-atlas graph alongside the code — a `Note` node
per frontmattered file, linked to code entities and to each other. Full conventions: `docs/SCHEMA.md`.

This file is a stable pointer, not a status digest — it doesn't change as notes are added; look the state up when you
need it instead of reading it here.

## Finding things

- `hybrid_search(query, mode="knowledge")` — ranks notes/docs above code, for why/decision/gotcha-shaped questions.
  `mode="blended"` (default) still includes them, ranked slightly below code. `mode="code"` excludes them.
- `get_context(uid)` on a code entity returns its linked notes/docs via `DOCUMENTS` edges.
- Cypher: `Note` nodes have `uid = "{project}:note:{slug}"`, `kind` in `draft|note|decision`, `tags`, `docstring` (full
  body). `LINKS_TO` is the wikilink graph; `DERIVED_FROM`/`SUPERSEDES` are dream-mode provenance.

## Writing things

- Quick findings mid-session: drop a frontmattered file in `docs/inbox/` (see SCHEMA.md for the frontmatter shape) — or
  the Claude Code memory dir if it's a private/machine-local observation, not something the repo should carry. Both are
  equivalent draft piles; don't agonize over which.
- A `remember` MCP write tool for this is planned (Phase 2) but not built yet.
- Durable notes in `docs/notes/`/`docs/decisions/` are produced by dream-mode consolidation only — don't write there
  directly.

## Status

Phases 0–1 of the knowledge-convergence architecture are implemented (vault parsing, schema, search integration).
Capture tooling (`remember`), anchor staleness, and dream-mode consolidation are not built yet — see
`.tasks/research/2026-07-11-knowledge-convergence-architecture.md` for the full roadmap.
