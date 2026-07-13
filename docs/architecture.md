# Code Atlas Architecture

This document describes the architecture of Code Atlas, a code intelligence graph system that indexes codebases and
exposes them via MCP tools for AI coding agents.

## System Overview

Code Atlas combines three search paradigms in a unified system:

- **Graph traversal** — follow relationships (calls, inheritance, imports)
- **Semantic search** — find code by meaning via embeddings
- **BM25 keyword search** — exact matches for identifiers and strings

All powered by Memgraph as a single backend, exposed through an MCP server. An event-driven pipeline with Valkey
(Redis-compatible) Streams keeps the index fresh as code changes.

```mermaid
graph TB
    subgraph Clients
        CC[Claude Code]
        CU[Cursor]
        WS[Windsurf]
        API[API Clients]
    end

    subgraph "Code Atlas — Daemon"
        FW[File Watcher]
        AST[AST Stage]
        EMB[Embed Stage]
    end

    subgraph "Code Atlas — MCP"
        MCP[MCP Server]
        QR[Query Router]

        subgraph Search
            GS[Graph Search]
            VS[Vector Search]
            BS[BM25 Search]
        end

        RRF[RRF Fusion]
        CE[Context Expander]
    end

    subgraph Infrastructure
        VK[(Valkey)]
        MG[(Memgraph)]
        TEI[TEI Embeddings]
    end

    CC --> MCP
    CU --> MCP
    WS --> MCP
    API --> MCP

    MCP --> QR
    QR --> GS
    QR --> VS
    QR --> BS

    GS --> RRF
    VS --> RRF
    BS --> RRF

    RRF --> CE
    CE --> MCP

    GS --> MG
    VS --> MG
    BS --> MG

    FW --> VK
    VK --> AST
    AST --> VK
    VK --> EMB
    AST --> MG
    EMB --> MG
    EMB --> TEI
```

## Component Architecture

### MCP Server

The MCP server is the primary interface for AI agents. Every capability is MCP-first; the CLI calls the same code paths.
Spawned per agent session via `atlas mcp`, it reads Memgraph directly with no dependency on the daemon.

**Query tools:**

- `cypher_query` — Execute raw Cypher queries against Memgraph
- `text_search` — BM25 keyword search via Memgraph's Tantivy
- `vector_search` — Semantic search via Memgraph's vector index
- `hybrid_search` — All three search types with RRF fusion
- `get_node` — Retrieve a node by qualified name (supports suffix matching)
- `get_context` — Get a node with surrounding context (class, module, callers)

**Index tools:**

- `index` — Trigger indexing of a project
- `status` — Check index health and staleness

**Admin tools:**

- `health` — Infrastructure health check
- `schema_info` — Describe available node types and relationships

### Event-Driven Pipeline

The indexing pipeline is event-driven, with two stages of increasing cost connected via Valkey (Redis) Streams. Each
stage pulls at its own pace, deduplicates within its batch window, and gates downstream work based on significance.

```mermaid
graph LR
    FW[File Watcher] -->|FileChanged| S1[atlas:file-changed]
    S1 -->|XREADGROUP| AST[AST Stage<br/>hash gate + parse + diff]
    AST -->|EmbedDirty| S2[atlas:embed-dirty]
    S2 -->|XREADGROUP| EMB[Embed Stage<br/>dedup by entity]
    AST -->|write| MG[(Memgraph)]
    EMB -->|write| MG
    EMB -->|embed| TEI[TEI]
```

**AST Stage** (medium cost, ~3s batch): Applies a file hash gate to skip unchanged files, re-parses AST via tree-sitter,
diffs entities, updates graph nodes/edges. Evaluates a significance gate — trivial changes (whitespace, formatting) stop
here; semantic changes (signature, body, docstring) publish `EmbedDirty` to the Embed stage.

**Embed Stage** (expensive, ~15s batch): Re-embeds affected entities via TEI, writes vectors to Memgraph. Deduplicates
by entity qualified name across all events in the batch.

**Significance Gate (AST → Embed):**

- Whitespace/formatting only → stop
- Non-docstring comment → stop
- Docstring changed → gate through
- Body changed beyond threshold → gate through
- Signature changed → always gate through
- Entity added/deleted → always gate through

**Error handling:** Failed batches are not acknowledged — Redis re-delivers via the pending entries list (PEL).

See [ADR-0004](adr/0004-event-driven-tiered-pipeline.md) for full rationale.

### Indexing Pipeline

The indexing pipeline transforms source code into a searchable graph. Each stage feeds into the next:

1. **File Scanner** — Walks the project tree, applying exclusion rules (`.gitignore`, `.atlasignore`, `atlas.toml`
   scope). Outputs a list of files to process.
2. **AST Parser** — Parses each file's AST via tree-sitter (in-process via py-tree-sitter), extracts entities (classes,
   functions, methods, imports) and their relationships.
3. **Pattern Detectors** — Pluggable detectors that identify implicit patterns: decorator-based routing, event handlers,
   test-to-code mapping, method overrides.
4. **Embedder** — Batches entities for embedding via TEI. Uses a content-hash cache to skip unchanged code. Operates on
   logical chunks (functions, classes, doc sections).
5. **Graph Writer** — Batch-writes nodes and edges to Memgraph. Updates vector and BM25 indices.

### Query Pipeline

When an agent issues a query:

1. **Query Router** analyzes the query and dispatches to one or more search backends (graph, vector, BM25) in parallel.
2. **RRF Fusion** merges results from all backends using Reciprocal Rank Fusion scoring.
3. **Context Expander** walks the graph around top results — up the hierarchy (class → module → package), along call
   chains, and into documentation links.
4. **Token Assembler** packs expanded results into a response within the configured token budget, prioritizing by
   relevance.

## Graph Schema

### Node Types

**Code nodes (6):** Project, Package, Module — structural containers. TypeDef, Callable, Value — code entities
discriminated by a `kind` property instead of language-specific labels (e.g., `TypeDef {kind: "class"}`,
`Callable {kind: "method"}`). This is language-agnostic and paradigm-agnostic (OOP, FP, procedural).

**Documentation & knowledge nodes (3):** DocFile, DocSection — heading-level extraction from ordinary markdown, linked
to code via DOCUMENTS edges. Note — a frontmatter-triggered atomic zettel (one node per file, not per heading); `docs/`
is an Obsidian-compatible knowledge vault that coexists with ordinary docs in the same tree (see `docs/SCHEMA.md`).
Notes link to each other via LINKS_TO (`[[wikilinks]]`) and DERIVED_FROM/SUPERSEDES (dream-mode provenance), and to code
via the same DOCUMENTS edge (explicit `anchors:` or heuristic symbol/file-path mentions).

**Dependency nodes (2):** ExternalPackage, ExternalSymbol — representing imported libraries and their symbols.

**Meta node (1):** SchemaVersion — singleton tracking the schema version for startup migration.

### Kind Discriminators

| Label    | Kind values                                                                                                  |
| -------- | ------------------------------------------------------------------------------------------------------------ |
| TypeDef  | class, struct, interface, trait, enum, union, type_alias, protocol, record, data_type, typeclass, annotation |
| Callable | function, method, constructor, destructor, static_method, class_method, property, closure                    |
| Value    | variable, constant, field, enum_member                                                                       |

### Common Properties

All entity nodes carry: `uid` (`{project_name}:{qualified_name}`), `project_name`, `name`, `qualified_name`,
`file_path`, `line_start`, `line_end`, `content_hash`, `kind`.

Code entities may also have: `visibility` (public/private/protected/internal), `tags` (extensible list replacing boolean
flags like `is_async`, `is_abstract`), `signature`, `docstring`, `code_snippet`, `embedding`, `complexity`.

### Relationships

```mermaid
erDiagram
    Project ||--o{ Package : CONTAINS
    Project ||--o{ Project : CONTAINS
    Package ||--o{ Module : CONTAINS
    Module ||--o{ TypeDef : DEFINES
    Module ||--o{ Callable : DEFINES
    Module ||--o{ Value : DEFINES
    TypeDef ||--o{ Callable : DEFINES
    TypeDef ||--o{ Value : DEFINES
    TypeDef ||--o{ TypeDef : INHERITS

    Callable ||--o{ Callable : CALLS
    Callable ||--o{ Callable : OVERRIDES
    Module ||--o{ ExternalPackage : IMPORTS
    Callable ||--o{ ExternalSymbol : USES_TYPE

    DocSection ||--o{ Callable : DOCUMENTS
    DocSection ||--o{ TypeDef : DOCUMENTS
    Note ||--o{ Callable : DOCUMENTS
    Note ||--o{ Note : LINKS_TO
    Note ||--o{ Note : DERIVED_FROM

    Callable ||--o{ Callable : HANDLES_ROUTE
    Callable ||--o{ Callable : HANDLES_EVENT
    Callable ||--o{ TypeDef : TESTS
```

**Structural (2):** CONTAINS, DEFINES — hierarchical containment and definition. Node labels provide discrimination (no
separate DEFINES_METHOD or CONTAINS_PROJECT needed). Monorepo roots are `Project` nodes that `CONTAINS` other `Project`
nodes.

**Type hierarchy (2):** INHERITS, IMPLEMENTS — class inheritance and interface/trait implementation.

**Call/Data (4):** CALLS, IMPORTS, USES_TYPE, OVERRIDES — runtime and compile-time dependencies.

**Dependencies (1):** DEPENDS_ON — package-level dependency edges.

**Documentation (2):** DOCUMENTS, MOTIVATED_BY — links between docs/ADRs and code entities.

**Similarity (1):** SIMILAR_TO — computed via embedding cosine similarity (weighted edge).

**Pattern-detected (6):** HANDLES_ROUTE, HANDLES_EVENT, REGISTERED_BY, INJECTED_INTO, TESTS, HANDLES_COMMAND — implicit
relationships made explicit by pattern detectors.

## Deployment

Code Atlas uses a **hybrid deployment model**: a long-running daemon handles continuous indexing, while a lightweight
MCP server is spawned per agent session.

```bash
docker compose up -d                  # Memgraph (7687) + Valkey (6379)
docker compose --profile tei up -d   # Include TEI (8080) for local embeddings
atlas daemon start             # File watcher + AST/Embed consumers (long-running)
atlas mcp                      # MCP server — stdio (Claude Code, Cursor)
atlas mcp --transport http     # MCP server — Streamable HTTP (VS Code, JetBrains)
```

The daemon publishes file change events to Valkey Streams, where the AST and Embed consumers process them and write to
Memgraph. The MCP server reads Memgraph directly — no dependency on the daemon for queries, so agents can query
immediately even with a stale index.

On startup, the daemon runs a reconciliation pass: compares filesystem state against the index and enqueues stale files
through the pipeline progressively (AST stage first, then Embed stage).

See [ADR-0005](adr/0005-deployment-process-model.md) for full rationale.

## Technology Stack

| Layer      | Technology           | Purpose                            |
| ---------- | -------------------- | ---------------------------------- |
| CLI        | Typer                | Command-line interface             |
| MCP        | mcp-python           | Model Context Protocol server      |
| Config     | Pydantic             | Configuration management           |
| Parsing    | Tree-sitter (Python) | Fast AST parsing                   |
| Graph DB   | Memgraph             | Graph storage + vector + BM25      |
| Event Bus  | Valkey (Redis)       | Pipeline streams + embedding cache |
| Embeddings | TEI / LiteLLM        | Code embeddings                    |
| HTTP       | httpx                | Async HTTP client                  |
| Tokens     | tiktoken             | Token counting                     |

## Security

- **Local-first**: All data stays on the developer's machine
- **No external calls by default**: TEI runs locally in Docker
- **Optional cloud embeddings**: LiteLLM fallback requires explicit config
- **No telemetry**: No usage data sent anywhere
- **Git-aware**: Respects .gitignore, never indexes secrets

## Performance Targets

| Operation                | Target  | Notes                       |
| ------------------------ | ------- | --------------------------- |
| Full index (10K files)   | < 60s   | Parallelized parsing        |
| Delta index (10% change) | < 10s   | Entity-level diffing        |
| Simple query (p95)       | < 100ms | Single search type          |
| Hybrid query (p95)       | < 300ms | Three search types + fusion |
| Memory (100K nodes)      | < 2GB   | Memgraph in-memory          |

## Accepted Architectural Decisions

- **[ADR-0004](adr/0004-event-driven-tiered-pipeline.md)**: Event-driven tiered pipeline with Redis Streams,
  significance gating, per-consumer batch policies
- **[ADR-0005](adr/0005-deployment-process-model.md)**: Hybrid deployment — daemon + agent-spawned MCP, stdio/Streamable
  HTTP transport

## Future Considerations

- **Language expansion**: Additional tree-sitter grammars
- **Distributed indexing**: For very large monorepos
- **Remote Memgraph**: Team-shared graph instance
- **Custom detectors**: User-defined pattern plugins
- **IDE integration**: Real-time indexing via file watchers
