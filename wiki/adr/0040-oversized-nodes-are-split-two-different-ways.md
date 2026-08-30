# ADR-0040: An oversized node is split two different ways, and which one depends on whether its borders mean anything

## Status

Accepted (2026-08-30)

## Context

Embedding providers cap their input. `gemini-embedding-001` accepts 2048 tokens; `text-embedding-3-small` accepts 8191.
Nothing in the pipeline respected either number reliably, for a reason that only shows up in production:
`EmbedClient._resolve_max_input_tokens` asks litellm's model registry, and the registry has no entry for a **routed**
model name. `openrouter/google/gemini-embedding-001` and `openai/nomic-ai/nomic-embed-code` both raise "isn't mapped
yet", which the code read as "no limit known" and translated into no truncation at all.

The consequence is worse than a truncated vector. `embed_batch` sends up to 128 texts in one call, and a provider that
rejects one rejects all of them — so a single oversized node cost 127 innocent ones their embeddings, and a dropped
embed is never recovered by a later re-index.

Measured on one corpus, the nodes over 2048 tokens were:

| Filetype              | Label      | Total  | Violating | %     |
| --------------------- | ---------- | ------ | --------- | ----- |
| `.md` youtube-archive | DocSection | 945    | 487       | 51.5% |
| `.md` other wiki/docs | DocSection | 15,928 | 131       | 0.8%  |
| `.py`                 | Callable   | 14,351 | 138       | 1.0%  |
| `.py`                 | TypeDef    | 1,804  | 79        | 4.4%  |
| `.py`                 | Value      | 11,737 | 10        | 0.1%  |

The 51.5% row is the shape of the problem. Those files have no headings, and headings are what the markdown parser
splits on — so the whole document became one DocSection. Semantic chunking exists; it just has nothing to work with when
the source contains no borders.

## Decision

**A length cap is a failsafe under semantic chunking, not a replacement for it.** Four changes, in that order.

### 1. State the cap the registry cannot

`embeddings.max_input_tokens` overrides the registry lookup. This is the root-cause fix; everything below is inert
without a known limit, and a limit that resolves to "unknown" is the state a routed model is permanently in.

### 2. Split at the strongest border that fits

`split_embed_text` descends a ladder of separators — blank lines, then newlines, then sentences, then spaces — and packs
greedily at each rung, so a cut lands on the coarsest border that still fits rather than on every border. It reports
whether any cut had to land mid-border, which is what separates "this text is long" from "this text has no structure".

It lives in `code_atlas/chunking.py`, not in `search/embeddings.py`, because both ends of the pipeline need it and
neither may import the other: `search.embeddings` pulls in litellm, which a parser has no business acquiring.

### 3. Documents split into nodes; code gains vectors

This is the decision the title names, and it is a **retrieval** argument rather than a provider one.

A Callable's boundary is meaningful. Half a function is not a thing anyone wants returned, and the node's identity — the
uid other edges point at — is worth preserving. So an oversized code entity stays **one node with several vectors**:
chunk 1 on the node itself, chunks 2..N as `EmbedChunk` nodes. A vector search collapses every row onto one per node at
its **best** chunk's score, because the question a search asks is whether _any_ part of the node matches. Summing or
averaging would make a long node's rank depend on where the splitter happened to cut it.

Half a header-less transcript is not a unit anyone wants returned either — but neither is the whole thing. So an
oversized DocSection becomes **several nodes**, part 1 keeping the original qualified_name so every relationship already
pointing at it stays valid.

`Note` is the exception that proves the rule: it is markdown, but its uid is an address `LINKS_TO` edges point at, so
splitting one would orphan the wikilink graph. Notes take the code treatment.

### 4. Say when a code entity had to be split

A Callable or TypeDef needing several chunks is usually one too large to be a single unit of anything, so that warns.
Documents do not: an oversized section is already split by the parser, and a Note is deliberately never split, so
neither is evidence of a defect.

## Consequences

`EmbedChunk` sits deliberately **outside** `_ENTITY_LABELS`. The `:Entity` marker is what makes a node reachable by uid
alone, and reaching a chunk that way would put it into relationship linking, package containment, the marker sweep and
the embed-dedup lookup, none of which have any business seeing one. The price is that a chunk has no edge to its parent
either — the link is a `parent_uid` property — so a deleted parent strands its chunks, and they need their own orphan
sweep next to the reference-node one.

**Amended the same day.** This originally read that chunk vectors are invisible to embed dedup, which reads through
`:Entity(embed_hash)`, and called that "a little re-embedding" worth paying to keep the marker's meaning intact. That
was wrong about the cost. The splitter descends a ladder starting at blank lines and re-anchors there, so it is already
sticky: measured over 40 random edit positions, a one-line insert into a 298-chunk file changes one chunk and a 200-line
insert changes two. Every other chunk stays byte-identical — and every one of them was being re-embedded anyway, because
the re-embed unit is the parent entity. So `find_embeddings_by_hash` now consults `:EmbedChunk` (`embed_hash`-indexed)
as a second statement. A chunk is in fact the better dedup source of the two: its `embed_hash` is the hash of its own
text, where a split parent's is the hash of the whole text, which no single vector corresponds to. The marker keeps its
meaning — this is a hash lookup, not a uid one, and nothing that asks about _code_ reaches a chunk through it. The
SQLite backend had been doing this all along, matching `nodes` without a label filter.

Splitting a DocSection changes node identity for content that was previously one node. Part 1 keeps the qualified_name,
so the re-parse updates it in place and the AST diff adds the rest; schema v17 clears the freshness markers so an
existing graph actually re-reads its files.

Two smaller extractions fall out of the same "give retrieval something to find" argument and ship alongside it: a Python
string literal over 500 characters becomes a `Value` of kind `text_block` (a prompt template or embedded query inside a
function body previously reached the graph as nothing but a slice of its function's capped `source`), and a SQL CTE
becomes a node the way a view already does.

**The other half of the same failure.** `_migrate_indices` drops every vector index unconditionally and recreates them
only when the running process has embeddings enabled — and `cli.py` disables them _automatically_ when the embedding
endpoint is merely unreachable. One such run took semantic search down on a 73k-node graph and nothing put it back,
because `_reconcile_search_indices` expects no vector indices when embeddings are off. The vectors survive the drop, so
the recovery is one run with embeddings enabled; that asymmetry is why a migration now refuses rather than warns, with
`--force-drop-embeddings` for a graph whose vectors are known to be disposable.

## Alternatives considered

**Truncate and move on.** What the code did when it knew a limit. It loses the tail silently, and for the 51.5% row it
loses most of the document.

**Split code entities into nodes too.** Uniform, and wrong: it would break every uid that CALLS, DEFINES and OVERRIDES
edges point at, to solve a problem affecting 1% of Callables.

**Several vectors on one node, as `embedding_1`, `embedding_2`, …** A Memgraph vector index is per (label, property), so
this means a fixed arity of index slots. A separate node was the only shape that scales with the text.

**A fixed conservative character budget instead of `max_input_tokens`.** Would over-split for the 8191-token models to
protect the 2048-token ones, and still guesses. The cap is knowable; it should be stated.
