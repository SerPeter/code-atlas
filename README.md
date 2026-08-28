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

> **Which backend am I actually on?** `backend.graph` defaults to `"auto"`, which uses Memgraph when it
> is reachable and silently falls back to an embedded SQLite engine when it is not — so **on a machine
> without Docker running, SQLite is what you get**. It is a fallback, not a parity replacement
> ([ADR-0015](wiki/adr/0015-embedded-backend-option.md)): community detection (`find_communities`,
> and the map in `atlas ui`) is unavailable there, and some analyses differ. `health_check` reports a
> **warning** rather than an OK while it is active, and `index_status` carries a `backend` field. Set
> `backend.graph = "memgraph"` in `atlas.toml` to fail loudly instead of falling back.

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
- **Human-readable too** — `atlas ui` serves the same graph as a local web interface, including an
  architecture-health view (DSM, propagation cost, dependency cycles) for spotting decay

## How Does This Compare?

Several excellent tools exist in this space — graph-based analyzers, semantic search engines, wiki generators, and IDE-integrated indexers. Code Atlas builds on their ideas while addressing a gap: no single tool combines graph traversal, semantic search, and BM25 keyword search with documentation intelligence and MCP exposure.

For a detailed comparison covering DeepWiki, Cursor, Sourcegraph Cody, Kit, code-graph-rag, codegraph-rust, and more, see [wiki/landscape.md](wiki/landscape.md).

## MCP Tools

23 tools exposed via the [Model Context Protocol](https://modelcontextprotocol.io/), designed to minimize context window overhead. On the SQLite fallback the server registers **22** — `find_communities` needs Memgraph and is unregistered rather than left to fail when called.

| Tool                       | What it does                                                                                     | Search | Full | Latency (avg / p95) |
| -------------------------- | ------------------------------------------------------------------------------------------------ | -----: | ---: | ------------------: |
| **Search**                 |                                                                                                  |        |      |                     |
| `hybrid_search`            | **Primary tool** — fuses graph + BM25 + vector via RRF. Auto-adjusts weights by query shape.     |   ~198 | ~672 |        548 / 677 ms |
| `text_search`              | BM25 keyword search. Quoted phrases, wildcards, field-specific queries.                          |   ~111 | ~320 |          34 / 36 ms |
| `vector_search`            | Semantic similarity via embeddings. Finds code by meaning, not name.                             |    ~88 | ~329 |        102 / 125 ms |
| `get_node`                 | Find entities by name. Cascade: exact (uid + name) → partial (suffix > prefix > contains).       |   ~122 | ~369 |            7 / 8 ms |
| **Navigation**             |                                                                                                  |        |      |                     |
| `get_context`              | Expand a node's neighborhood: parent, siblings, callers, callees, docs.                          |    ~90 | ~246 |          34 / 36 ms |
| `summarize_module`         | Dense skeleton of a module or package — signatures, line spans, adjacency, fan-in and fan-out.   |   ~218 | ~407 |                   — |
| `trace_path`               | Shortest path between two entities. Each hop carries its edge type, confidence and strategy.     |   ~114 | ~252 |                   — |
| `cypher_query`             | Run read-only Cypher against the graph. Auto-limited, write-protected.                           |    ~48 | ~128 |            3 / 3 ms |
| **Analysis**               |                                                                                                  |        |      |                     |
| `analyze_repo`             | Structure, centrality, dependencies, pattern, or quality analysis.                               |    ~50 | ~470 |          22 / 23 ms |
| `blast_radius`             | Transitive closure of callers/callees — "what breaks if I change this". Every hit reports `via`. |   ~214 | ~473 |                   — |
| `find_communities`         | Clusters modules into subsystems by deterministic greedy modularity. Memgraph only.              |   ~159 | ~341 |                   — |
| `find_dead_code`           | Entities with no incoming edge. **A lead, not a verdict** — known false positives are listed.    |   ~171 | ~310 |                   — |
| `find_complexity_hotspots` | Top callables by LOC-span — a crude proxy, not cyclomatic complexity.                            |    ~72 | ~213 |                   — |
| `find_hotspots`            | Commit-count hotspots, bus-factor risks and co-change pairs from git history.                    |   ~145 | ~286 |                   — |
| `generate_diagram`         | Mermaid diagrams: packages, imports, inheritance, module detail.                                 |    ~82 | ~284 |            3 / 3 ms |
| **Guidance**               |                                                                                                  |        |      |                     |
| `get_usage_guide`          | Quick-start or topic-specific guidance for the agent.                                            |    ~24 |  ~79 |        < 1 / < 1 ms |
| `plan_search_strategy`     | Recommends which search tool + params for a question.                                            |    ~29 |  ~70 |        < 1 / < 1 ms |
| `validate_cypher`          | Catches Cypher errors before execution.                                                          |    ~47 |  ~89 |            1 / 2 ms |
| `schema_info`              | Full graph schema: labels, relationships, Cypher examples.                                       |    ~64 |  ~79 |        < 1 / < 1 ms |
| **Status**                 |                                                                                                  |        |      |                     |
| `index_status`             | Projects, entity counts, schema version, index health.                                           |    ~61 |  ~76 |          22 / 23 ms |
| `list_projects`            | Monorepo project list with dependency relationships.                                             |    ~45 |  ~60 |          12 / 13 ms |
| `health_check`             | Infrastructure diagnostics: Memgraph, TEI, Valkey, schema.                                       |    ~73 |  ~88 |        218 / 264 ms |
| `knowledge_health`         | Knowledge-vault lint: inbox, orphans, dangling links, duplicate ids, promotion candidates.       |   ~135 | ~150 |                   — |

Token counts measured from the registered MCP tool definitions (tiktoken `cl100k_base`) — reproduce with `uv run python scripts/count_tool_tokens.py`. **Search** = name + description (2,360 total); **Full** = name + description + parameter schema with field descriptions, enums and constraints (5,791 total). All parameters are self-documented, so agents can one-shot any tool without calling `get_usage_guide` first.

**Latency** was measured with local TEI embeddings on the code-atlas repo (~1,400 entities), 5 iterations, with embeddings already present — see `scripts/profile_query.py`. A `—` means that tool is not yet in the profiling harness; it is unmeasured, not instant.

## Quick Start

### Prerequisites

- **Python 3.14+** (`requires-python = ">=3.14"`). A hard floor, not a soft one: the codebase uses
  [PEP 758](https://peps.python.org/pep-0758/) unparenthesized `except` tuples, which are a **syntax
  error** before 3.14 — it will not even import. `litellm` is also held below 1.92, because newer
  releases dropped prebuilt Windows/3.14 wheels.
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose — for Memgraph and Valkey.
- [uv](https://docs.astral.sh/uv/) (Python package manager). `uvx` fetches a matching interpreter for
  you, so you do not need 3.14 already on your PATH.

### 1. Start infrastructure

Download the compose file and start Memgraph + Valkey:

```bash
curl -O https://raw.githubusercontent.com/SerPeter/code-atlas/main/docker-compose.yml
docker compose up -d
```

> `docker-compose.yml` runs the MAGE-enabled `memgraph/memgraph-mage` image (needed for
> `find_communities`/community detection). Existing deployments on an older compose file:
> `docker compose pull && docker compose up -d` to switch over.

Optional — add local embeddings (no API keys needed):

```bash
docker compose --profile tei up -d
```

### 2. Index your project

```bash
uvx --from code-atlas-mcp atlas index /path/to/your/project
uvx --from code-atlas-mcp atlas status
```

### 3. Explore it yourself (optional)

The graph is not only for agents. `atlas ui` serves a local, project-scoped web interface — search,
entity detail with the evidence behind every edge, and an architecture-health view that reports whether
the codebase is trending toward a big ball of mud:

```bash
uvx --from "code-atlas-mcp[ui]" atlas ui   # http://127.0.0.1:8420
```

It binds to loopback and talks to nothing but your local Memgraph.

### 4. Connect to your AI agent

**Claude Code:**

```bash
claude mcp add code-atlas -- uvx --from code-atlas-mcp atlas mcp
```

**Cursor / other MCP clients** — add to your MCP config:

```json
{
  "mcpServers": {
    "code-atlas": {
      "command": "uvx",
      "args": ["--from", "code-atlas-mcp", "atlas", "mcp"]
    }
  }
}
```

See [CLI usage guide](wiki/guides/usage.md) for more commands and options.

### Development

If you want to contribute or run from source:

```bash
git clone https://github.com/SerPeter/code-atlas.git
cd code-atlas
uv sync --group dev
uv run pre-commit install
```

## Performance

| Metric                     | Value                 |
| -------------------------- | --------------------- |
| Full index (107 files)     | **55s** (local TEI)   |
| Parse-only throughput      | **600–700 files/sec** |
| `get_node` / `text_search` | 7 ms / 34 ms          |
| `vector_search`            | 102 ms                |
| Concurrent QPS             | **238** (zero errors) |

Full index includes parsing, graph upserts, and embedding via local TEI (8 concurrent workers). Parse-only is raw tree-sitter CPU time without I/O. Query latencies are averages from `scripts/profile_query.py`. Full benchmark tables: [wiki/benchmarks.md](wiki/benchmarks.md)

## Documentation

- [Architecture](wiki/architecture.md) — system design, pipelines, deployment model
- [Landscape](wiki/landscape.md) — code intelligence tools comparison and design rationale
- [Configuration](wiki/guides/configuration.md) — atlas.toml, .atlasignore, environment variables
- [CLI Usage](wiki/guides/usage.md) — indexing, searching, daemon mode
- [Benchmarks](wiki/benchmarks.md) — parsing, query latency, concurrency
- [Repository Guidelines](wiki/guides/repo-guidelines.md) — structure your code for better indexing

## Supporting Code Atlas

I built Code Atlas because my AI agents kept burning half their context just figuring out where things are in larger
codebases. Nothing combined the search types I needed in one place, so I built it and open-sourced it so you can
benefit as well.

If Code Atlas saves you time, tokens, or makes your agents noticeably better — consider [sponsoring the project](https://github.com/sponsors/SerPeter).

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-pink?logo=github)](https://github.com/sponsors/SerPeter)

## License

[Apache License 2.0](LICENSE)

**Third-party components.** The web UI ships two vendored browser bundles, both MIT, each with its
upstream licence text alongside it: [sigma.js](https://github.com/jacomyal/sigma.js) 3.0.3 and
[graphology](https://github.com/graphology/graphology) 0.26.0, under
`src/code_atlas/server/web/static/vendor/`. Versions, sources and file hashes are recorded in
[PROVENANCE.md](src/code_atlas/server/web/static/vendor/PROVENANCE.md). They are committed rather than
fetched, so `atlas ui` needs no network beyond your local Memgraph.
