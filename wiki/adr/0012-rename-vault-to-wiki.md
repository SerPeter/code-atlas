# ADR-0012: Rename the Default Knowledge Vault Directory to wiki/

## Status

Accepted

## Date

2026-07-17

## Context

ADR-0011 made `docs/` an Obsidian-compatible, zettelkasten-style knowledge vault living in the same graph as the code it
documents, explicitly rejecting a separate `knowledge/` directory in favor of one merged tree (Alternative 1 in that
ADR). The vault root was already exposed as a setting (`[knowledge] vault_path` in `atlas.toml`,
`KnowledgeSettings.vault_path`), defaulted to `"docs"` — configurable from day one, just not renamed.

`"docs"` is a generic, overloaded name that doesn't signal the vault's intended behavior: a living, cross-linked, agent-
and human-editable knowledge base, not a static documentation folder. `"wiki"` names that behavior directly and matches
how the directory is actually used (wikilinks, zettelkasten notes, dream-mode consolidation) once ADR-0011's phases
land.

## Decision

- Change `KnowledgeSettings.vault_path`'s default from `"docs"` to `"wiki"` (`settings.py`). The setting itself is
  unchanged — this only changes what ships as the default.
- Rename this repo's own vault directory `docs/` → `wiki/` to match, along with every internal reference to it
  (`.claude/rules/knowledge.md`, `.claude/commands/dream-mode.md`, `CLAUDE.md`, `README.md`,
  `.gitignore`/`.atlasignore`'s `HOME.md` entry, `pyproject.toml`'s Documentation URL, and the vault's own
  `SCHEMA.md`/`architecture.md` self-references).
- This does **not** reopen ADR-0011's Alternative 1 — it is still one merged directory holding ordinary docs, ADRs, and
  vault notes side by side. Only the name changes, not the structure.

## Consequences

### Positive

- The directory name now describes its behavior (a living wiki) instead of a generic "docs" label that undersells what
  frontmatter-triggered note mode actually does.
- No new setting was needed — `vault_path` already existed, so any project that wants to keep `docs/` (or use something
  else entirely) sets `vault_path` explicitly in `atlas.toml`.

### Negative

- None beyond the breaking-change risk below — this is a pure rename with no structural change.

### Risks

- **Breaking change for existing deployments.** Any project relying on the implicit `"docs"` default without an explicit
  `vault_path` override will have `atlas dream`/`knowledge_health` start scanning an empty `wiki/` instead of their real
  vault, until they either rename their directory or add `vault_path = "docs"` to `atlas.toml`. This is scoped narrowly:
  `vault_path` only feeds the dream-mode filesystem scan (duplicate-id/dangling-link checks) and the `HOME.md` write
  location — note-frontmatter parsing itself is directory-agnostic and unaffected. Flagged via a `BREAKING CHANGE:`
  footer on the settings-change commit for python-semantic-release to version accordingly.

## Alternatives Considered

### Alternative 1: Keep `docs/` as the shipped default, document `wiki` as an opt-in convention

- Zero breaking-change risk — existing deployments keep working unchanged.
- Rejected: the whole point is that "wiki" better signals correct usage; leaving the confusing default in place means
  every new project has to know to opt into to the better name, instead of getting it for free. The breaking-change
  surface is narrow and cheap to work around (one `atlas.toml` line or a directory rename).

### Alternative 2: A separate `wiki/` directory alongside `docs/`

- Would let ordinary project docs stay in `docs/` while a new `wiki/` holds only vault notes.
- Rejected for the same reason ADR-0011 rejected a separate `knowledge/` directory: two parallel doc trees is more
  surface area for no real benefit, and frontmatter-triggered note mode already lets ordinary docs and vault notes
  coexist per-file in one tree.

## References

- [ADR-0011: Note Label and the Knowledge Vault Schema](./0011-note-vault-schema.md) — the merged-directory decision
  this ADR renames but does not reverse
- [wiki/SCHEMA.md](../SCHEMA.md) — vault conventions
