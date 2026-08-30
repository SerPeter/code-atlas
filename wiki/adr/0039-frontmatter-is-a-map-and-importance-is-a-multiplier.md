# ADR-0039: Frontmatter is one queryable map, and importance is a multiplier on the existing boost chain

## Status

Accepted (2026-08-30)

## Context

Two requests arrived together: per-path ranking adjustments configured in `atlas.toml`, and a frontmatter representation
that could be queried — including as the input to a ranking adjustment.

Frontmatter was already being stored, badly. `_parse_markdown_note` put every non-consumed key into
`ParsedEntity.extra_properties`, and both upsert paths end with `SET n += e.extra_properties` — _after_ the
`ON CREATE`/`ON MATCH SET` that writes the schema fields. A frontmatter key sharing a name with one of those silently
overwrote it. This was not a latent risk: the Claude Code memory dialect requires a `name:` key, so every memory note in
the graph had its `name` replaced by its own slug. `uid`, `file_path`, `source` and `docstring` sat in the same line of
fire.

Two smaller gaps came with it. `+=` only ever adds, so a key deleted from a file stayed on the node forever. And
frontmatter on markdown that did _not_ trigger note mode was parsed for dialect detection and then discarded, so an
ordinary doc could declare nothing about itself that survived indexing.

Ranking, by contrast, was already in good shape: `_boost_results` is a single chokepoint multiplying RRF score by
visibility, label, project-scope and supersession factors, and `props_by_uid` already carries every node property
returned by the three channels.

## Decision

**1. Frontmatter is one `frontmatter` map property, not N top-level properties.**

Verified against Memgraph 3.12 rather than assumed: nested maps are valid property values,
`WHERE n.frontmatter.metadata.type = …` matches, `CREATE INDEX ON :Note(frontmatter.metadata.type)` registers, nested
temporals and lists-of-maps survive the Bolt round trip, and `SET n += {frontmatter: null}` _removes_ the property. That
last one is what gives deletion semantics for free — the map is replaced wholesale, so a key removed from the file is
removed from the graph.

Note mode keeps excluding the keys it consumes into first-class fields and edges (`id`, `kind`, `tags`, `derived_from`,
`supersedes`, `contradicts`, `anchors`); storing them again in the map would only duplicate what is already queryable.
Doc mode consumes nothing and therefore excludes nothing.

**2. Ordinary docs keep their frontmatter, propagated to their sections.**

A search hit on a doc is a `DocSection`, so frontmatter reachable only from the `DocFile` would never be visible to a
rule. The asymmetry in how absence is recorded is deliberate: a `DocFile` always carries the key (null when there is no
block) so that deleting a block clears the stored map, while a `DocSection` gets the key only when a block exists.
`DocFile` is not embeddable and `DocSection` is — stamping an explicit null on every section of every plain markdown
file would change its `content_hash` and re-embed the entire corpus to record an absence.

**3. Importance is a fifth multiplier in `_boost_results`, configured under `[search.importance]`.**

Two rule lists — `paths` (gitignore-style globs, the dialect `[scope]` and `.atlasignore` already use) and `frontmatter`
(dotted key, optional value). Every matching rule's factor multiplies into the score.

Multiplicative composition rather than first-match-wins, because "in `src/` **and** tagged critical" should compound,
and first-match makes the answer depend on the order rules happen to appear in the file. The product is clamped to
`[0.01, 100]`: a mistyped `factor = 1000` otherwise outranks every genuine signal in the fusion, and the symptom reads
as a broken ranker rather than as bad config.

A frontmatter rule resolves against the map first and falls back to the fields note mode promotes _out_ of it, so
`key = "kind"` and `key = "tags"` work on notes. Without that fallback those two rules would silently never match on
exactly the notes that declare them.

An empty or absent rule set short-circuits, so the default configuration ranks byte-identically to before.

## Consequences

The `name` corruption is fixed by re-parsing: with the key nested, the `ON MATCH SET n.name = e.name` above the `+=`
wins and the title comes back. Schema v16 clears file/git hashes to force that re-parse — the map exists only in the
file's bytes, and the file-hash gate would otherwise skip every unchanged file forever.

**v16 deliberately does not delete the pre-v16 flattened keys.** "Remove every property that is not a schema property"
is one incomplete whitelist away from deleting `embedding` and taking semantic search down silently, which is a failure
this project has already had once (see the v15 vector-index incident). The leftovers are inert — nothing reads an
unknown property — and `atlas project rm` followed by a re-index clears them for anyone who wants them gone.

One residual, recorded rather than hidden: deleting an entire frontmatter block from a _non-note_ doc clears the map on
its `DocFile` but leaves the copy on its `DocSection`s, because sections only carry the key when a block exists.
Correcting it would cost a full doc re-embed on every repo to handle a case that does not occur in normal editing.

The bump is a fleet event: the first process to run `ensure_schema` upgrades the shared Memgraph and every other process
still on older code hard-errors with "newer than code". Restart the daemon and any `atlas mcp` in the same window, and
count the vector indices afterwards rather than assuming the migration left them.
