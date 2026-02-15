# Code Intelligence Landscape

How AI coding agents understand codebases today, where the gaps are, and how Code Atlas fits in.

---

## The Problem Space

Every time an AI coding agent starts a task, it spends tokens orienting itself — grepping for function names, reading
files to trace call chains, searching docs for context. On larger projects, agents can burn 30-50% of their context
window before writing a single line of code. The tools that help with this fall into several categories, each solving
part of the puzzle but none solving all of it.

## Landscape Overview

### Wiki and Documentation Generators

**DeepWiki** (Cognition) generates interactive wikis for GitHub repositories. Its most interesting insight, presented by
Cognition president Russell Kaplan at LangChain Interrupt 2025, is that the richest codebase understanding comes from
_metadata around the source code_ — pull request history, commit messages, review discussions — not just the code
itself. DeepWiki extracts concepts from this metadata, builds a table of contents, and generates structured
documentation with Mermaid diagrams and source-linked references. Over 50,000 public repos have been pre-indexed.

**DeepWiki-Open** (AsyncFuncAI, ~14K GitHub stars) is an open-source reimplementation using FastAPI, FAISS for vector
search, and configurable AI providers. It generates similar wiki output but works from source code only — no PR or
commit metadata analysis. Development has shifted to the team's AsyncReview product.

**Google Code Wiki** (launched November 2025) takes a different approach: continuous regeneration. Rather than
point-in-time indexing, it rescans and rebuilds documentation after every code change, ensuring the wiki never becomes
stale. It includes a Gemini-powered chat interface for questions about the codebase.

**CodeWiki** (FPT Software / University of Melbourne, arXiv 2510.24428) is an academic framework that combines
Tree-Sitter AST parsing with recursive multi-agent documentation generation. It decomposes repositories into
hierarchical modules via dependency graphs and topological sorting, then delegates documentation of each module to
specialized agents that can recursively subdivide complex modules further. On their CodeWikiBench benchmark (21 repos,
86K-1.4M LOC), it outperformed DeepWiki by ~5 percentage points overall and by ~30% on large-scale projects.

### IDE-Integrated Indexing

**Cursor** uses a Merkle tree-based incremental indexing system. Code is chunked at AST boundaries via Tree-Sitter,
hashed into a Merkle tree, and synchronized with Cursor's servers for embedding generation. Only changed files are
re-uploaded, and chunk-level hash caching allows embedding reuse across team members. Embeddings are stored in
Turbopuffer; no raw code is persisted server-side.

**Continue.dev** originally offered an `@Codebase` context provider backed by LanceDB (embedded, no separate process,
sub-10ms queries even at 1M+ vectors). The team has since deprecated this in favor of Agent mode, where the agent
explores the codebase using built-in file and search tools rather than pre-computed embeddings.

**Augment Code** offers a proprietary 200K-token context engine that maps dependencies and relationships across
codebases, then curates targeted context rather than retrieving raw chunks. Their "ContextWiki" feature generates
wiki-style documentation from indexed codebases. Details of the implementation are proprietary.

### Structural and Graph-Based Approaches

**Aider's Repository Map** creates a condensed structural map of the entire repository, showing the most important
classes and functions ranked by a PageRank-like graph algorithm over a dependency graph built from Tree-Sitter ASTs. The
map fits in ~1K tokens and is sent with every LLM request. This approach is purely structural — no embeddings, no
semantic search — but is highly token-efficient and complements LLM reasoning well.

**Sourcegraph Cody** is notable for having _abandoned_ embeddings in favor of BM25 combined with a structural code
graph. Their reasoning: at enterprise scale (100K+ repositories), the operational overhead of vector databases outweighs
the benefits, and well-tuned BM25 with structural signals performs comparably. Their code graph contains definitions,
references, symbols, and doc comments produced by language-specific indexers.

### MCP-Based Code Graph Tools

Several projects expose code graph analysis via the Model Context Protocol:

**code-graph-mcp** (entrepeneur4lyf) uses ast-grep for fast AST parsing across 25+ languages and rustworkx for in-memory
graph operations. It provides good structural queries and cross-file reference tracking, but has no persistence (the
graph is rebuilt every session), no semantic or keyword search, and development appears to have stopped after an initial
burst of activity.

**code-graph-rag** (vitali87, ~295 issues, actively developed) builds on Memgraph with a full
Project-Package-Module-Class-Method hierarchy — the most complete structural model among open-source tools. However, it
requires an LLM for every query (natural language to Cypher translation), which adds cost and latency. Its Memgraph
instance has native BM25 and vector search capabilities, but the project uses neither — only Cypher graph traversal. The
community has reported issues with cross-project query isolation
([#295](https://github.com/vitali87/code-graph-rag/issues/295)), qualified name resolution in LLM-generated queries
([#278](https://github.com/vitali87/code-graph-rag/issues/278)), and the parser crawling into `site-packages` to index
thousands of third-party files ([#206](https://github.com/vitali87/code-graph-rag/issues/206)).

**Kit** (cased, ~1,200 stars) is the most polished developer experience: Python API, CLI, REST API, and MCP server with
five search modes (text, symbol, AST pattern, semantic via ChromaDB, and docstring-based). It supports 12+ languages and
includes features like incremental symbol extraction, LLM-powered docstring indexing, and context-around-line
extraction. Key lessons from Kit's issue tracker: MCP tool outputs that return full source code caused 40K-token
responses ([#177](https://github.com/cased/kit/issues/177)); only loading the root `.gitignore` caused 88K
`node_modules` files to be included in monorepos ([#144](https://github.com/cased/kit/issues/144)); and ChromaDB batch
size limits caused failures on large repos ([#139](https://github.com/cased/kit/issues/139)). Kit has no graph database,
so it cannot follow relationships like call chains or inheritance hierarchies.

**codegraph-rust** (Jakedismo, 136 stars) is a 100% Rust implementation using SurrealDB. Its strongest feature is
optional LSP integration for type-aware resolution — launching language servers (rust-analyzer, pyright, gopls, etc.) to
resolve types that pure AST parsing cannot. It offers tiered indexing (fast/balanced/full) and consolidated "agentic"
tools that run internal reasoning agents. However, SurrealDB has presented operational challenges: schema functions fail
to load via the CLI ([#61](https://github.com/Jakedismo/codegraph-rust/issues/61)), Go indexing produces nodes but zero
edges ([#51](https://github.com/Jakedismo/codegraph-rust/issues/51)), and JS/TS projects had only 12% of files indexed
with silent fallback to random hash-based embeddings ([#43](https://github.com/Jakedismo/codegraph-rust/issues/43)).

**OpenAI Codex CLI** (~60K stars) is notable for what it _doesn't_ have: no semantic codebase indexing. It relies on
file system exploration and grep-style search. A community proposal for semantic indexing (issue #5181) remains open,
validating the need for external tools that provide this capability.

---

## Capability Matrix

A side-by-side view of indexing and retrieval strategies:

| Capability                   | DeepWiki | Cursor  | Cody    | Aider | Kit     | code-graph-rag | codegraph-rust  | Code Atlas      |
| ---------------------------- | -------- | ------- | ------- | ----- | ------- | -------------- | --------------- | --------------- |
| **AST parsing**              | Unknown  | Yes     | Yes     | Yes   | Yes     | Yes            | Yes             | Yes (10 langs)  |
| **Graph database**           | Unknown  | No      | Yes     | No    | No      | Yes (Memgraph) | Yes (SurrealDB) | Yes (Memgraph)  |
| **Semantic / vector search** | Yes      | Yes     | Dropped | No    | Yes     | No             | Yes             | Yes             |
| **BM25 keyword search**      | Unknown  | No      | Yes     | No    | Partial | Unused         | Yes             | Yes             |
| **Hybrid fusion**            | Unknown  | No      | No      | No    | No      | No             | Partial         | Yes (RRF)       |
| **Relationship traversal**   | Unknown  | No      | Yes     | No    | No      | Yes            | Yes             | Yes             |
| **Documentation indexing**   | Yes      | No      | No      | No    | Partial | No             | Partial         | Yes             |
| **Monorepo support**         | No       | Partial | Yes     | No    | Partial | Partial        | Partial         | Yes             |
| **Incremental updates**      | No       | Yes     | Yes     | No    | Yes     | Yes            | No              | Yes (AST-level) |
| **Self-hosted / local**      | No       | No      | No      | Yes   | Yes     | Yes            | Yes             | Yes             |
| **MCP integration**          | No       | No      | No      | No    | Yes     | No             | Yes             | Yes             |
| **PR/commit metadata**       | Yes      | No      | No      | No    | No      | No             | No              | No              |

### What the Matrix Reveals

No single tool combines all three search paradigms (graph traversal, semantic search, BM25 keyword search) with
documentation intelligence, relationship tracking, and MCP exposure. Several tools have the _infrastructure_ for
multiple search types but don't use it — code-graph-rag sits on Memgraph's BM25 and vector capabilities without enabling
them. Others excel at one dimension but miss the rest: Kit has excellent search variety but no graph database; Cody has
the structural graph but dropped embeddings; Aider has brilliant token efficiency but no persistence.

The gap is integration. The individual technologies exist and are proven. What's been missing is a system that assembles
them into a coherent whole.

---

## Recurring Lessons from the Ecosystem

Studying issue trackers, pull requests, and architectural decisions across these projects surfaces patterns that any
code intelligence tool must address:

### Token discipline is non-negotiable

Kit's most impactful optimization was making `extract_symbols` return metadata-only by default, with a separate tool for
fetching actual source — reducing MCP tool output by 85% ([Kit #177](https://github.com/cased/kit/issues/177)).
codegraph-rust had to reduce graph traversal defaults from 100 to 20 nodes and context limits from 80K to 20K tokens
after performance problems ([PR #27](https://github.com/Jakedismo/codegraph-rust/pull/27)). The lesson: MCP tools must
default to compact output with pagination and let agents request detail on demand.

### Silent failures poison trust

codegraph-rust's fallback to random hash-based embeddings when the embedding provider fails
([#43](https://github.com/Jakedismo/codegraph-rust/issues/43)) produced search results that _looked_ plausible but were
meaningless. code-graph-mcp's use of Rust's `.unwrap()` on unsupported languages caused silent thread panics
([#2](https://github.com/entrepeneur4lyf/code-graph-mcp/issues/2)). code-graph-rag's parser crawled into `site-packages`
without warning ([#206](https://github.com/vitali87/code-graph-rag/issues/206)). When a code intelligence tool silently
degrades, agents make confident decisions based on wrong information.

### Multi-project isolation must be foundational

code-graph-rag added multi-project support as a later feature, leading to a cascade of cross-contamination bugs
([#295](https://github.com/vitali87/code-graph-rag/issues/295),
[#287](https://github.com/vitali87/code-graph-rag/issues/287)). Queries for one project returned results from another.
The fix required adding `project_name` to every node and a WHERE clause to every query — effectively a schema rewrite.

### Qualified name resolution is the hardest recurring problem

In code-graph-rag, LLM-generated Cypher queries searched for `{name: 'VatManager'}` but the graph stored qualified names
like `project.module.VatManager` ([#278](https://github.com/vitali87/code-graph-rag/issues/278)). Kit had duplicate
symbol entries when multiple Tree-Sitter patterns matched the same construct
([#100](https://github.com/cased/kit/issues/100)). codegraph-rust's Go indexer extracted 17K nodes but zero edges
because relationship resolution failed silently ([#51](https://github.com/Jakedismo/codegraph-rust/issues/51)). Robust
name resolution with suffix, prefix, and fuzzy matching is essential.

### Composable tools outperform monolithic ones

Kit built an all-in-one `smart_context` tool that automatically assembled relevant context for queries. They later
removed it entirely ([PR #136](https://github.com/cased/kit/pull/136)), concluding it was "too magic" and that agents
compose simpler tools more effectively than a monolithic orchestrator. This tracks with what we've observed: agents are
better at deciding what context they need when given good primitives.

### Database choice matters more than expected

codegraph-rust's SurrealDB backend has operational issues: schema functions can't be loaded via the CLI
([#61](https://github.com/Jakedismo/codegraph-rust/issues/61)), and the team needed to add RocksDB alongside SurrealDB
for storage patterns SurrealDB couldn't handle ([PR #45](https://github.com/Jakedismo/codegraph-rust/pull/45)).
SurrealDB's non-standard query language (SurrealQL) also means AI agents trained on Cypher can't write queries for it.
code-graph-rag's choice of Memgraph proved more stable, but using only one of its three search capabilities left
significant value on the table.

---

## How Code Atlas Fits

Code Atlas was designed with these ecosystem lessons in mind. Rather than excelling at one dimension of code
intelligence, it integrates all three search paradigms into a single system backed by Memgraph:

**Unified search.** Graph traversal (Cypher), semantic search (vector embeddings via TEI), and BM25 keyword search
(Tantivy) all run against the same database. Results are fused using Reciprocal Rank Fusion, with weights that
auto-adjust based on whether the query looks like an identifier, natural language, or a mix of both.

**Documentation as first-class data.** Markdown files are parsed into sections with hierarchical header breadcrumbs
(e.g., "Architecture > Authentication > Token Validation") and linked to the code entities they describe. Two doc
sections titled "Configuration" under different parent headers get distinct embeddings because the header path is
prepended before embedding.

**Direct tools, no LLM tax.** All 15 MCP tools are self-documented with parameter descriptions, type constraints, and
return shape documentation. An agent can one-shot any tool on first encounter. No intermediate LLM call is needed to
translate queries — the agent's own intelligence drives navigation. This avoids the cost, latency, and fragility of the
NL-to-Cypher pattern.

**No additional API costs.** Many code tools require their own LLM API access for query translation, enrichment, or
documentation generation — adding a separate billing dimension on top of whatever the agent already costs. Code Atlas's
agent-first approach means all intelligence comes from the agent's existing subscription (Claude Code, Cursor, etc.).
Embeddings run locally via TEI with no API keys required. The only infrastructure is Docker containers on your own
machine.

**AST-level incremental indexing.** When a file changes, the parser compares the new AST against stored entity hashes.
Only entities whose content actually changed get re-embedded and updated in the graph. Sibling entities in the same file
are left untouched. A three-tier event pipeline (graph metadata, AST diffing, embeddings) processes changes through
Valkey Streams.

**Monorepo-native from day one.** Sub-projects are auto-detected by project markers (`pyproject.toml`, `package.json`,
`Cargo.toml`, etc.). Each gets its own Project node with DEPENDS_ON relationships. Queries default to the current
project scope and can be expanded to span the full monorepo.

**10-language AST parsing.** Python, TypeScript, JavaScript, Go, Rust, Java, Kotlin, C, C++, Ruby, and PHP are supported
via py-tree-sitter with language-specific grammars. New languages are added by registering a grammar and tree-sitter
query file.

**Pluggable pattern detection.** Six detectors recognize indirect relationships that AST parsing alone misses:
decorator-based routing (`@app.route`), event handler registration, test-to-code mappings (naming + imports), method
overrides, dependency injection, and CLI command handlers.

**Token-budget context assembly.** When expanding context around a search result, the system respects a configurable
token budget with priority ordering: target code, then class context, then callees, callers, documentation, siblings,
and package context — in that order until the budget is exhausted.

### What Code Atlas Does Not Do

Some capabilities in the landscape are explicitly outside Code Atlas's current scope:

- **PR/commit metadata analysis.** DeepWiki's insight that metadata reveals intent is compelling, but requires platform
  API integration (GitHub, GitLab) and adds significant scope. This remains a future opportunity.
- **Wiki generation.** Code Atlas indexes and searches; it does not generate prose documentation. The graph structure
  would be a strong foundation for wiki generation, but that's a different product surface.
- **LSP-based type resolution.** codegraph-rust's approach of launching language servers for type-aware linking produces
  more accurate call graphs than AST-only parsing. This is architecturally compatible with Code Atlas (as an optional
  enrichment tier) but adds significant complexity and has proven fragile in practice.
- **Hosted/cloud deployment.** Code Atlas runs locally with Docker. No data leaves your machine.

---

## Design Decisions and Their Rationale

Several architectural choices were informed directly by observing what worked and what didn't across the ecosystem:

| Decision                       | Rationale                                                                                                                                        |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Memgraph as unified backend    | Only graph DB with native Cypher + BM25 + vector search. Avoids multi-backend integration pain (SurrealDB + RocksDB, ChromaDB + separate graph)  |
| Python with py-tree-sitter     | Same C parsing engine as Rust bindings, avoids tree-sitter Python binding version conflicts that plagued Kit, simpler build than Rust subprocess |
| Direct MCP tools (no LLM)      | NL-to-Cypher fails on qualified names (code-graph-rag #278); every LLM provider version break is a bug (Kit had 10+ provider issues)             |
| TEI for embeddings             | Self-hosted, no API keys needed, GPU-accelerated batching — addresses provider confusion and cost concerns                                       |
| Content-hash embedding cache   | Single biggest performance win in codegraph-rust (100-900x on cache hits). Designed in from day one                                              |
| Compact MCP output defaults    | Kit's 85% token reduction by defaulting to metadata-only validated this approach                                                                 |
| Project scoping on every query | code-graph-rag's cross-contamination bugs (#295, #287) showed this must be foundational, not retrofitted                                         |
| Pluggable pattern detectors    | No existing tool handles decorator routing, DI, events, or test mappings — universal gap                                                         |
| Conservative defaults          | 20-result limits, 8K token budgets, 1-hop caller/callee depth — following codegraph-rust's lesson of starting small                              |

---

## Sources

### Projects Referenced

- [DeepWiki](https://deepwiki.com) — Cognition AI
- [DeepWiki-Open](https://github.com/AsyncFuncAI/deepwiki-open) — AsyncFuncAI (MIT)
- [Google Code Wiki](https://codewiki.google) — Google
- [CodeWiki](https://fsoft-ai4code.github.io/CodeWiki/) — FPT Software / University of Melbourne
- [Aider](https://aider.chat/docs/repomap.html) — Paul Gauthier
- [Continue.dev](https://docs.continue.dev) — Continue
- [Cursor](https://docs.cursor.com) — Anysphere
- [Sourcegraph Cody](https://sourcegraph.com/docs/cody) — Sourcegraph
- [Augment Code](https://www.augmentcode.com) — Augment
- [OpenAI Codex CLI](https://github.com/openai/codex) — OpenAI
- [code-graph-mcp](https://github.com/entrepeneur4lyf/code-graph-mcp) — entrepeneur4lyf
- [code-graph-rag](https://github.com/vitali87/code-graph-rag) — vitali87
- [Kit](https://github.com/cased/kit) — Cased
- [codegraph-rust](https://github.com/Jakedismo/codegraph-rust) — Jakedismo
- [Context7](https://github.com/upstash/context7) — Upstash

### Talks and Articles

- Russell Kaplan, "DeepWiki" — LangChain Interrupt, May 2025
- FPT Software, "CodeWiki: Automated Repository-Level Documentation" — arXiv:2510.24428, November 2025
- Sourcegraph, "How Cody Understands Your Codebase" — sourcegraph.com/blog
- Engineer's Codex, "How Cursor Indexes Codebases Fast" — read.engineerscodex.com
