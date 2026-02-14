# 🗺️ Code Atlas

**A code intelligence graph that gives AI coding agents deep, token-efficient understanding of your codebase — structure, docs, and dependencies in one searchable graph.**

> Map your codebase. Search it three ways. Feed it to agents.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

---

## What Is This?

Code Atlas builds a **graph database** of your entire codebase — code structure, documentation, and dependencies — and exposes it via **MCP tools** that AI coding agents can use to understand, navigate, and reason about your code.

Three search types, one system:

- **Graph traversal** — follow relationships: who calls this function? What does this class inherit from? What services depend on this library?
- **Semantic search** — find code by meaning: "authentication middleware" finds relevant code even if it's named `verify_token_chain`
- **BM25 keyword search** — exact matches: find that specific error message, config key, or function name

All powered by [Memgraph](https://memgraph.com/) as a single backend.

## Key Features

- **Monorepo-native** — auto-detects sub-projects, tracks cross-project dependencies, scoped queries
- **Documentation as first-class** — indexes markdown docs, ADRs, and READMEs with links to the code they describe
- **AST-level incremental indexing** — only re-indexes the entities that actually changed, not entire files
- **Pattern detection** — pluggable detectors for decorator routing, event handlers, DI, test→code mappings, and more
- **Library awareness** — lightweight stubs for external dependencies, full indexing for internal libraries
- **Self-hosted** — runs locally with Docker. No data leaves your machine
- **Token-efficient** — budget-aware context assembly that prioritizes what matters most
- **Pluggable AI** — TEI for embeddings, LiteLLM for LLM calls, or bring your own
- **MCP server** — works with Claude Code, Cursor, Windsurf, or any MCP-compatible client

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/code-atlas.git
cd code-atlas

# Start infrastructure (Memgraph + TEI)
docker compose up -d

# Install code-atlas
uv sync

# Index your project
atlas index /path/to/your/project

# Check status
atlas status
```

### MCP Integration

Add to your Claude Code / Cursor MCP config:

```json
{
  "mcpServers": {
    "code-atlas": {
      "command": "atlas",
      "args": ["mcp"]
    }
  }
}
```

## Usage

```bash
# Index a codebase
atlas index .

# Index specific paths (monorepo)
atlas index . --scope services/auth --scope libs/shared

# Search
atlas search "authentication middleware"
atlas search --type graph "MATCH (f:Function)-[:CALLS]->(g) WHERE g.name = 'validate_token' RETURN f"
atlas search --type keyword "DATABASE_URL"

# Status and health
atlas status
atlas health
```

## Configuration

Create an `atlas.toml` in your project root:

```toml
[scope]
include_paths = ["services/auth", "services/billing", "libs/shared"]
exclude_patterns = ["*.generated.ts", "testdata/"]

[libraries]
full_index = ["my_company_shared_lib"]
stub_index = ["fastapi", "sqlalchemy"]

[monorepo]
auto_detect = true
always_include = ["libs/shared"]

[embeddings]
model = "nomic-ai/nomic-embed-code"
base_url = "http://localhost:8080"  # self-hosted TEI; omit for cloud providers

[search]
default_token_budget = 8000
test_filter = true  # exclude test files from results by default

[detectors]
enabled = ["decorator_routing", "event_handlers", "test_mapping", "class_overrides", "di_injection", "cli_commands"]
```

File exclusions use `.atlasignore` (same syntax as `.gitignore`):

```
# Generated code
*_pb2.py
*_pb2_grpc.py
# Vendored deps
vendor/
# Migration history
migrations/
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 MCP Server (Python)             │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │  Query   │ │  Index   │ │  Admin/Health   │  │
│  │  Tools   │ │  Tools   │ │  Tools          │  │
│  └────┬─────┘ └────┬─────┘ └─────────────────┘  │
│       │             │                           │
│  ┌────▼─────────────▼───────────────────────┐   │
│  │       Query Router / Orchestrator        │   │
│  └────┬─────────┬───────────┬───────────────┘   │
│       │         │           │                   │
│  ┌────▼───┐ ┌───▼────┐ ┌───▼─────┐              │
│  │ Graph  │ │ Vector │ │  BM25   │   → RRF →    │
│  │ Search │ │ Search │ │ Search  │   Fusion     │
│  └────┬───┘ └───┬────┘ └───┬─────┘              │
│       └─────────┼──────────┘                    │
│            ┌────▼────┐                          │
│            │Memgraph │                          │
│            └─────────┘                          │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Indexing Pipeline                  │
│  File Scanner → AST Parser (tree-sitter) → Diff │
│  → Pattern Detectors → Embeddings (TEI)         │
│  → Graph Writer → Memgraph                      │
└─────────────────────────────────────────────────┘
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint & format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty check

# Pre-commit hooks
uv run pre-commit install
```

## Performance

Benchmarked on a synthetic codebase with deterministic Python files (classes, methods, imports, docstrings).

### Parsing

| Codebase | Files | Entities | Time | Throughput | Peak Memory |
|----------|-------|----------|------|------------|-------------|
| Small    | 100   | 1,500    | 0.14s | **709 files/sec** | 1.4 MB |
| Medium   | 1,000 | 15,000   | 1.6s  | **608 files/sec** | 15.8 MB |

Memory scales linearly (~16 KB per entity) when accumulating all parse results. In production, entities are streamed to the graph and not held in memory simultaneously.

### Query Latency (p50 / p95)

| Search Type | p50 | p95 | p99 |
|-------------|-----|-----|-----|
| BM25 text search | 8 ms | 12 ms | 29 ms |
| Vector search | 58 ms | 73 ms | 84 ms |
| Graph search | 209 ms | 352 ms | 436 ms |

Graph search is slower because it runs a 3-stage cascade (exact → suffix → contains) as sequential Cypher queries. BM25 and vector search use single Memgraph procedure calls.

### Concurrent Queries

| Concurrency | Total Queries | Wall Time | QPS | Errors |
|-------------|---------------|-----------|-----|--------|
| 10 | 50 | 1.9s | 26 | 0 |
| 50 | 250 | 1.1s | **238** | 0 |

Zero errors under load. QPS scales well with concurrency thanks to Memgraph's connection pooling.

### Running Benchmarks

```bash
# Parser + memory (no infra needed)
uv run pytest tests/bench/test_bench_parser.py tests/bench/test_bench_memory.py -m bench -s

# Query + concurrent (requires Memgraph)
uv run pytest tests/bench/ -m bench -s

# Exclude benchmarks from regular test runs
uv run pytest -m "not bench"
```

## How Does This Compare?

Several excellent tools exist in this space. Code Atlas builds on their ideas while addressing gaps that emerge when you need graph, semantic, and keyword search working together.

| Tool                                                                    | Strengths                                      | Gaps                                                                            |
|-------------------------------------------------------------------------|------------------------------------------------|---------------------------------------------------------------------------------|
| **[code-graph-mcp](https://github.com/entrepeneur4lyf/code-graph-mcp)** | Fast ast-grep parsing, broad language coverage | In-memory only (no persistence), no semantic or keyword search                  |
| **[code-graph-rag](https://github.com/vitali87/code-graph-rag)**        | Best hierarchy model, Memgraph-native          | Every query requires an LLM call, no vector/BM25 search, no doc indexing        |
| **[Kit](https://github.com/cased/kit)**                                 | Clean DX, good semantic + text search          | No graph database — can't follow relationships (calls, inheritance)             |
| **[codegraph-rust](https://github.com/Jakedismo/codegraph-rust)**       | 100% Rust, LSP-based type resolution           | SurrealDB's graph traversal unproven, non-standard query language for AI agents |

Code Atlas combines the strengths: code-graph-rag's hierarchy and Memgraph foundation, Kit's search and context assembly, and tree-sitter-based AST parsing — while adding documentation intelligence, monorepo support, pluggable pattern detection, and token-budget-aware context assembly.

## License

[Apache License 2.0](LICENSE)
