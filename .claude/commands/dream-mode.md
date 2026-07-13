---
description:
  Dream mode (code-atlas edition) — consolidate drafts across the repo inbox, the harness memory dir, and the global
  vault, using atlas dream's deterministic report.
---

# Dream Mode — Code Atlas Edition

Project-local override of the global `dream-mode` command (per that command's own "extend, don't duplicate" convention).
Everything in the global command still applies — philosophy, disposition table, KEEP/MERGE/PROMOTE/DROP semantics,
output format. This file only changes **where the inventory comes from** and adds the knowledge-vault-specific promotion
venues and archive convention that code-atlas's unified graph makes possible.

## Why this repo needs its own version

In code-atlas, repo-inbox drafts (`docs/inbox/`), the harness memory dir, and (if configured) the global vault are **all
Notes in the same Memgraph graph** — one `atlas dream` (or the `knowledge_health` MCP tool) call produces a single
deterministic report spanning all three, instead of three separate manual audits. Capture-time routing is best-effort
(write wherever's cheapest mid-session); this command is the authoritative consolidation pass across all of them, same
as the global command's philosophy — just with a graph-backed inventory instead of a directory listing.

## 1. Inventory (replaces the global command's step 1)

- Run `atlas dream --json` (or call the `knowledge_health` MCP tool) instead of manually listing memory-dir files. It
  returns, across every configured vault (this repo's `docs/`, the harness memory dir, and any
  `[knowledge] extra_vaults`):
  - `inbox_count` / `inbox_paths` — draft notes awaiting disposition
  - `orphan_notes` — notes with no `[[wikilinks]]` in or out (disconnected from the note graph)
  - `dangling_links` — `[[wikilink]]`/`derived_from`/`supersedes` references whose target doesn't exist (typo'd slug, or
    the target was deleted/renamed)
  - `duplicate_ids` — two files in the same vault resolved to the same note uid (only one survives in the graph; the
    report is the only way to see the collision)
  - `similar_pairs` — high-embedding-similarity note pairs (candidates for MERGE)
  - `promotion_candidates` — similar pairs that span _different_ projects (repo-specific finding that recurs elsewhere →
    candidate for the global vault)
  - `memory_index_issues` — MEMORY.md entries with no matching file, or files not indexed
- Still read `MEMORY.md` directly for the human-facing index (byte size, entry count) — the report's
  `memory_index_issues` only flags _consistency_, not size.
- List promotion venues exactly as the global command does (skills, rules, CLAUDE.md, inline comments, docs/) **plus**:
  - **`docs/notes/` / `docs/decisions/`** — durable zettels in this repo's own vault (`kind: note` / `kind: decision`
    frontmatter, per `docs/SCHEMA.md`)
  - **The global vault** — cross-project generalizations (`promotion_candidates` above is the mechanical signal for this
    venue)
- Do not modify anything yet.

## 2. Classify

Same disposition table as the global command (KEEP / MERGE / PROMOTE / CLAUDE.md / inline comment / docs / DROP), with
two additions specific to the graph-unified vault:

| Disposition                | When                                                                                                                       | Action at execute step                                                                                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **PROMOTE → global vault** | A `promotion_candidates` hit, or a finding with no repo-specific anchor that clearly generalizes.                          | `git mv` the file to the global vault's path; add `promoted_from: <old-uid>` to frontmatter so provenance survives the move.                                                   |
| **DROP → archive stub**    | Content worth a provenance pointer but not worth keeping in full (superseded decision, stale draft with historical value). | See §4's archive-stub convention below — this is a refinement of the global command's plain DROP, available because the graph can query `DERIVED_FROM`/`SUPERSEDES` afterward. |

Use `duplicate_ids` and `dangling_links` as blocking findings — resolve them before proceeding to promotions, since a
duplicate-id file will silently overwrite the other on next index regardless of what you decide for either one
individually.

## 3. Plan and confirm

Same as the global command — present the disposition table, wait for approval. Extend the table's venue column with
`global vault` / `archive stub` where applicable.

## 4. Execute

Same ordering as the global command (promotions → merges → deletions → refresh index → backlinks), plus:

- **Archive-stub convention** (instead of outright deletion for anything with lasting reference value): reduce the file
  to a 2-3 line summary, set `archived: true` in frontmatter, and `git mv` it to an `archive/` subdirectory of its
  vault. Point the stub at the commit hash where the full content last existed (`git log --oneline -1 -- <path>` before
  the edit). The stub stays indexed (still searchable, still a real Note) — only the full content leaves the live vault.
  Never do this for a note still referenced by live `DERIVED_FROM`/`SUPERSEDES` edges without first redirecting them.
- **Provenance on MERGE/PROMOTE**: write `derived_from: [<source-uid>, ...]` (and `supersedes: [<uid>]` when the new
  note fully replaces an old one) in the survivor's frontmatter — these become real `DERIVED_FROM`/`SUPERSEDES` graph
  edges on next index, so provenance is queryable (`MATCH (n:Note)-[:DERIVED_FROM]->(m) ...`), not just prose.
- **After execution, re-run `atlas dream`** (not just the manual checks the global command lists) to confirm
  `duplicate_ids`/`dangling_links` actually cleared and the promoted/ merged notes appear where expected — the report is
  cheap and catches mistakes the manual grep-based verification in the global command's step 5 can miss (e.g. a
  `derived_from` slug typo that silently fails to resolve).

## 5. Verify

Everything in the global command's step 5, plus:

- `atlas dream --json` again — `duplicate_ids` and `dangling_links` should be empty (or only contain items explicitly
  deferred, noted as such in the punch list).
- For each `PROMOTE → global vault` row: confirm the note is indexed under the global vault's project name
  (`atlas status` or `hybrid_search(mode="knowledge")`).

## Output format

Same punch list as the global command, plus one line:

```
Vault findings resolved: <duplicate_ids before>→<after>, <dangling_links before>→<after>
```
