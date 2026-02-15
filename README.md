# Code Atlas

**A code intelligence graph that gives AI coding agents deep, token-efficient understanding of your codebase — structure, docs, and dependencies in one searchable graph.**

> Map your codebase. Search it three ways. Feed it to agents.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MCP](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

---

## The Problem

Every time an AI agent touches your codebase, it burns tokens just figuring out where things are. Grep for a function name. Read five files to understand the call chain. Search docs for context. Repeat — across every task, every session. On a large project, agents can spend **30–50% of their context window** on orientation before they write a single line of code.

Many tools solve one piece of this: semantic search, or graph traversal, or keyword lookup. But a developer doesn't understand a codebase through one lens — they build a **mental model** that connects structure, meaning, and names simultaneously. Agents need the same thing.

Code Atlas is that mental model, externalized as a graph.

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
- **No additional API costs** — agent-first design means all intelligence runs through your existing subscription; local embeddings via TEI, no extra API keys
- **Token-efficient** — budget-aware context assembly that prioritizes what matters most
- **Pluggable AI** — TEI for embeddings, LiteLLM for LLM calls, or bring your own
- **MCP server** — works with Claude Code, Cursor, Windsurf, or any MCP-compatible client

## How Does This Compare?

Several excellent tools exist in this space — graph-based analyzers, semantic search engines, wiki generators, and IDE-integrated indexers. Code Atlas builds on their ideas while addressing a gap: no single tool combines graph traversal, semantic search, and BM25 keyword search with documentation intelligence and MCP exposure.

For a detailed comparison covering DeepWiki, Cursor, Sourcegraph Cody, Kit, code-graph-rag, codegraph-rust, and more, see [docs/landscape.md](docs/landscape.md).

## MCP Tools

15 tools exposed via the [Model Context Protocol](https://modelcontextprotocol.io/), designed to minimize context window overhead.

| Tool                   | What it does                                                                                 | Search | Full |
| ---------------------- | -------------------------------------------------------------------------------------------- | -----: | ---: |
| **Search**             |                                                                                              |        |      |
| `hybrid_search`        | **Primary tool** — fuses graph + BM25 + vector via RRF. Auto-adjusts weights by query shape. |   ~117 | ~497 |
| `text_search`          | BM25 keyword search. Quoted phrases, wildcards, field-specific queries.                      |    ~90 | ~275 |
| `vector_search`        | Semantic similarity via embeddings. Finds code by meaning, not name.                         |    ~67 | ~297 |
| `get_node`             | Find entities by name. Cascade: exact uid → name → suffix → prefix → contains.               |   ~100 | ~254 |
| **Navigation**         |                                                                                              |        |      |
| `get_context`          | Expand a node's neighborhood: parent, siblings, callers, callees, docs.                      |    ~64 | ~273 |
| `cypher_query`         | Run read-only Cypher against the graph. Auto-limited, write-protected.                       |    ~59 | ~168 |
| **Analysis**           |                                                                                              |        |      |
| `analyze_repo`         | Structure, centrality, dependencies, or pattern analysis.                                    |    ~41 | ~266 |
| `generate_diagram`     | Mermaid diagrams: packages, imports, inheritance, module detail.                             |    ~37 | ~254 |
| **Guidance**           |                                                                                              |        |      |
| `get_usage_guide`      | Quick-start or topic-specific guidance for the agent.                                        |    ~35 | ~106 |
| `plan_search_strategy` | Recommends which search tool + params for a question.                                        |    ~40 |  ~97 |
| `validate_cypher`      | Catches Cypher errors before execution.                                                      |    ~58 | ~116 |
| `schema_info`          | Full graph schema: labels, relationships, Cypher examples.                                   |    ~75 |  ~96 |
| **Status**             |                                                                                              |        |      |
| `index_status`         | Projects, entity counts, schema version, index health.                                       |    ~72 |  ~93 |
| `list_projects`        | Monorepo project list with dependency relationships.                                         |    ~56 |  ~77 |
| `health_check`         | Infrastructure diagnostics: Memgraph, TEI, Valkey, schema.                                   |    ~55 |  ~76 |

Token counts measured from MCP JSON tool definitions (tiktoken cl100k_base). **Search** = name + description (~966 total); **Full** = name + description + parameter schema with field descriptions, enums, and constraints (~2,945 total). All parameters are self-documented — agents can one-shot any tool without calling `get_usage_guide` first.

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [uv](https://docs.astral.sh/uv/) (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/code-atlas.git
cd code-atlas

# Start infrastructure (Memgraph + Valkey)
docker compose up -d

# Optional: start with local embeddings (TEI)
docker compose --profile tei up -d

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

See [CLI usage guide](docs/guides/usage.md) for more commands and options.

## Performance

| Metric              | Value                 |
| ------------------- | --------------------- |
| Parse throughput    | **600–700 files/sec** |
| Graph search (p50)  | 8 ms                  |
| BM25 search (p50)   | 10 ms                 |
| Vector search (p50) | 47 ms                 |
| Concurrent QPS      | **238** (zero errors) |

Full benchmark tables and methodology: [docs/benchmarks.md](docs/benchmarks.md)

## Documentation

- [Architecture](docs/architecture.md) — system design, pipelines, deployment model
- [Landscape](docs/landscape.md) — code intelligence tools comparison and design rationale
- [Configuration](docs/guides/configuration.md) — atlas.toml, .atlasignore, environment variables
- [CLI Usage](docs/guides/usage.md) — indexing, searching, daemon mode
- [Benchmarks](docs/benchmarks.md) — parsing, query latency, concurrency
- [Repository Guidelines](docs/guides/repo-guidelines.md) — structure your code for better indexing

## Supporting Code Atlas

I built Code Atlas because my AI agents kept burning half their context just figuring out where things are in larger
codebases. Nothing combined the search types I needed in one place, so I built it and open-sourced it so you can
benefit as well.

If Code Atlas saves you time, tokens, or makes your agents noticeably better — consider [sponsoring the project](https://github.com/sponsors/SerPeter).

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SerPeter)

## License

[Apache License 2.0](LICENSE)
