# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code Atlas is a code intelligence graph system that indexes codebases and exposes them via MCP tools for AI coding agents. It combines graph traversal, semantic search, and BM25 keyword search using Memgraph as the unified backend.

Python with tree-sitter C extension for AST parsing, called in-process via py-tree-sitter.

## Commands

```bash
# Install dependencies
uv sync                          # Runtime dependencies
uv sync --group dev              # Include dev dependencies

# Run tests
uv run pytest                    # All tests
uv run pytest -m "not slow"      # Skip slow tests
uv run pytest -m integration     # Integration tests only (requires Docker — testcontainers by default, see Testing)
uv run pytest tests/test_foo.py::test_bar  # Single test

# Lint and format
uv run ruff check .              # Lint
uv run ruff check . --fix        # Lint with auto-fix
uv run ruff format .             # Format
uv run ty check                  # Type check

# Pre-commit
uv run pre-commit install        # Install hooks
uv run pre-commit run --all-files  # Run all hooks manually

# Infrastructure
docker compose up -d             # Start Memgraph + Valkey (production index: 7687/6379)
docker compose --profile test up -d  # Optional integration-test fast path (memgraph-test :7688, valkey-test :6380, see Testing)
docker compose --profile tei up -d  # Include local embeddings (TEI)
docker compose --profile telemetry up -d  # Victoria stack + OTel Collector + Grafana (:3000)
docker compose down              # Stop services

# CLI
atlas index /path/to/project     # Index a codebase
atlas index --watch              # Index, then keep watching (holds the indexer lease)
atlas index --watch --force      # ...taking the lease from a holder that is gone
atlas search "query"             # Hybrid search
atlas status                     # Check index status
atlas mcp                        # Start MCP server
atlas mcp --no-index             # Query-only: no watcher/pipeline (2nd+ session in a worktree)
atlas ui                         # Web UI; takes the first free port from 8420 up
atlas daemon start               # Start indexing daemon (watcher + pipeline)
atlas dream                      # Knowledge-vault lint report (inbox, orphans, dangling links, duplicates) + wiki/HOME.md
atlas project rm <name>          # Delete a project's graph data (e.g. a stale worktree project)
```

## Architecture

```
src/code_atlas/
├── __init__.py          # __version__ only
├── schema.py            # Graph schema (labels, relationships, DDL generators)
├── settings.py          # Pydantic configuration (atlas.toml + env vars)
├── events.py            # Event types (FileChanged, EmbedDirty) + Valkey Streams EventBus
├── telemetry.py         # OpenTelemetry integration
├── cli.py               # Typer CLI entrypoint (index, search, status, mcp, daemon commands)
│
├── parsing/
│   ├── ast.py           # Tree-sitter AST parser (py-tree-sitter, in-process)
│   └── detectors.py     # Pluggable pattern detectors (routes, test mappings, overrides)
│
├── graph/
│   └── client.py        # Async Memgraph client (schema, upsert, search)
│
├── search/
│   ├── engine.py        # Hybrid search — RRF fusion across graph/vector/BM25
│   ├── embeddings.py    # Embedding client (litellm) + rate limiter
│   └── guidance.py      # Cypher validation + search strategy for AI agents
│
├── indexing/
│   ├── orchestrator.py  # Full-index, monorepo detection, staleness checking
│   ├── consumers.py     # AST + Embed event consumers (batch-pull pattern)
│   ├── watcher.py       # Filesystem watcher (watchfiles + hybrid debounce)
│   └── daemon.py        # Daemon lifecycle manager (watcher + pipeline)
│
└── server/
    ├── mcp.py           # FastMCP server (tools for AI coding agents)
    └── health.py        # Infrastructure health checks + diagnostics
```

**Event Pipeline:** File Watcher → Valkey Streams → AST stage (hash gate + parse + diff) → Embed stage (embeddings) → Memgraph

**Query Pipeline:** MCP Server → Query Router → [Graph Search | Vector Search | BM25 Search] → RRF Fusion → Results

**Deployment:** Daemon (`atlas daemon start`) for indexing + MCP (`atlas mcp`) per agent session, decoupled via Valkey + Memgraph

**Event model:** Events are atomic — one logical change per event (one file per FileChanged, one entity per EmbedDirty). Never bundle lists of work items into a single event; use `EventBus.publish_many()` for network-efficient batch publishing. The consumer's `max_batch_size` must directly control work volume, not just message count.

**Infrastructure:** Memgraph (graph DB, port 7687), TEI (embeddings, port 8080), Valkey (event bus, port 6379)

**Embedding dedup:** the graph is the dedup layer, not Valkey (ADR-0036). Before calling the provider, the
embed stage asks whether any node — any project, any label — already has a vector for the same `embed_hash`
under the same model, and copies it. Valkey carries streams, consumer groups and the indexer lease only.

## Code Style

- Python 3.14+, line length 120
- Ruff for linting/formatting, ty for type checking
- Known first-party import: `code_atlas`
- Conventional commits: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

## Development Rules

**Code changes:**

- When integrating new behavior that replaces old behavior, remove the old code paths — don't leave dead artifacts
- When removing code unrelated to the current task, ask before deleting
- Edit existing files — search before writing new code
- Integrate, don't isolate — add to existing modules, not new files
- Generate conservatively — only what's explicitly needed
- No speculative code — no "nice to have" features or premature abstractions

**Planning approach:**

- Plan-first for non-trivial tasks: research the codebase to understand:
  - Where the new functionality integrates (callers, config, CLI, exports, tests)
  - What existing behavior it replaces or extends
  - What old code paths should be removed
- Plan must cover both implementation and integration — no dead code

**Working style:**

- Be honest about uncertainty — if unsure about a domain, library, or implementation approach, say so and ask to research first. Don't guess.
- Use subagents to orchestrate complex/large tasks
- Subagents must NOT commit unless explicitly instructed — the parent agent controls commits

## Testing

- Tests in `tests/` directory, async-first with pytest-asyncio (auto mode)
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration`
- `integration` means "needs real Memgraph/Valkey", nothing narrower — it is orthogonal to directory. `tests/bench/` carries both `bench` and `integration`, so `-m integration` from the repo root is the complete infra-requiring set and must collect and pass; `bench` additionally means "slow, measures throughput" and is deselected in CI.
- Infra fixtures (`_infra_endpoints`, `settings`, `graph_client`, `event_bus`, the wipe guard) live in `tests/conftest.py` so both `tests/integration/` and `tests/bench/` see them; `tests/integration/conftest.py` holds only the TEI tier. They are lazy — a unit-only run never starts a container.
- **High gear (default):** Integration tests exercising full workflows and public APIs
- **Low gear (selective):** Unit tests only for complex algorithms or edge cases unreachable via integration
- Don't test every function. Test system behavior.
- Integration tests start session-scoped testcontainers on random ports by default (Docker required; skip if unavailable). Fast path: `docker compose --profile test up -d`, then `ATLAS_TEST_MEMGRAPH_PORT=7688 ATLAS_TEST_VALKEY_PORT=6380 uv run pytest -m integration` — the env vars point tests at any isolated stack (e.g. CI service containers). Never connects to the production Memgraph/Valkey on 7687/6379. A conftest guard refuses to wipe any Memgraph containing project data not prefixed `test`/`bench`; `ATLAS_TEST_DB=1` bypasses it for known-disposable instances only.

### Running tests efficiently — read this before running anything

Measured baselines: unit ~63s serial, integration ~700s scoped / ~1100s from the repo root. Agent workflows
have spent **78-100% of their wall-clock running tests**, one of them re-running the full unit suite 28
times. Almost all of that is avoidable.

- **While iterating, run only what you touched.** `uv run pytest tests/unit/parsing/test_apex.py`, or
  `uv run pytest tests/integration/graph -m integration`. `tests/integration/` mirrors `src/`
  (graph, indexing, search, server, backends), so directory scoping is a natural unit. Do NOT re-run the
  full suite after every edit.
- **Run the full suite once, at the end**, before reporting or committing. That is the gate; the iteration
  loop is not.
- **Use `-n auto` for unit tests** — measured 63s → 31s. `pytest-xdist` is already a dev dependency.
  It is deliberately NOT in `addopts`, because it would also apply to single-test debugging runs where it
  adds startup cost, scrambles output order and breaks `pdb`.
- **NEVER combine `-n` with `ATLAS_TEST_MEMGRAPH_PORT`/`ATLAS_TEST_VALKEY_PORT`.** `graph_client` is
  function-scoped and runs `MATCH (n) DETACH DELETE n` before every test. Under xdist, session fixtures run
  once _per worker_, so with the env overrides unset each worker gets its own container and is isolated —
  but with them set, every worker shares one instance and they wipe each other's data mid-test. The
  failures look like nondeterministic product bugs.
- **Iterate with `--testmon`, gate without it.** `uv run pytest --testmon` runs only the
  tests its dependency database says your changes can affect. The first run pays for a full
  pass to build `.testmondata`; after that a one-file edit runs seconds of tests instead of
  minutes. It is **not compatible with `-n`** (xdist), and it trusts a database rather than
  the test selection you would make by hand -- so it is an iteration tool and never the
  final gate. Delete `.testmondata` if selection ever looks wrong.
- **A hang now fails.** `--timeout=300` (thread method, the portable one) is in `addopts`.
  It is a tripwire for "this will never finish", not a performance budget -- the slowest
  legitimate test is two orders of magnitude under it. If one test genuinely needs longer,
  mark it `@pytest.mark.timeout(N)`; do not raise the global value. This exists because a
  lease-wait regression made one daemon test sit for its full 600s budget: it **passed**,
  and the only symptom was the unit suite going from ~60s to 623s, which reads as green.
- **Watch the suite's wall-clock, not just the count.** Unit is ~50s with `-n auto`. A run
  that takes ten times that has told you something, even when every test passes.
- **Never pass an extra `-q`.** `addopts` already contains one; a second makes `-qq`, which suppresses the
  totals line entirely. That has repeatedly produced "tests pass" reports with no count behind them.

## Commits

- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format: `<type>(<scope>): <description>`
- Commit immediately when task is done
- Amend for feedback: `git add . && git commit --amend --no-edit`
- New commit only for genuinely separate work
- Never unstage changes that would cause data loss (e.g., don't `git reset` if it would discard changes)

**Version bumping (semantic versioning)**

## Configuration

- `atlas.toml` - Project configuration (scope, embeddings, search settings, detectors)
- `.atlasignore` - Gitignore-style exclusion patterns for indexing
- Environment variables: `ATLAS_*` prefix with double-underscore nesting (e.g., `ATLAS_EMBEDDINGS__MODEL`)
