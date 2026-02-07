# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Code Atlas is a code intelligence graph system that indexes codebases and exposes them via MCP tools for AI coding agents. It combines graph traversal, semantic search, and BM25 keyword search using Memgraph as the unified backend.

Python/Rust hybrid: Python for CLI, MCP server, and orchestration; Rust (`crates/atlas-parser`) for fast AST parsing via tree-sitter.

## Commands

```bash
# Install dependencies
uv sync                          # Runtime dependencies
uv sync --group dev              # Include dev dependencies

# Run tests
uv run pytest                    # All tests
uv run pytest -m "not slow"      # Skip slow tests
uv run pytest -m integration     # Integration tests only (requires Memgraph)
uv run pytest tests/test_foo.py::test_bar  # Single test

# Lint and format
uv run ruff check .              # Lint
uv run ruff check . --fix        # Lint with auto-fix
uv run ruff format .             # Format
uv run ty check                  # Type check

# Pre-commit
uv run pre-commit install        # Install hooks
uv run pre-commit run --all-files  # Run all hooks manually

# Rust (from repo root)
cargo build --manifest-path crates/Cargo.toml
cargo test --manifest-path crates/Cargo.toml
cargo clippy --manifest-path crates/Cargo.toml --all-targets -- -D warnings

# Infrastructure
docker compose up -d             # Start Memgraph + TEI
docker compose down              # Stop services

# CLI
atlas index /path/to/project     # Index a codebase
atlas search "query"             # Hybrid search
atlas status                     # Check index status
atlas mcp                        # Start MCP server
```

## Architecture

```
src/code_atlas/
├── cli.py          # Typer CLI entrypoint (index, search, status, mcp commands)
└── settings.py     # Pydantic configuration (atlas.toml + env vars)

crates/
└── atlas-parser/   # Rust AST parser using tree-sitter (outputs JSON)
```

**Indexing Pipeline:** File Scanner → AST Parser (Rust) → Delta Diff → Pattern Detectors → Embeddings (TEI) → Graph Writer → Memgraph

**Query Pipeline:** MCP Server → Query Router → [Graph Search | Vector Search | BM25 Search] → RRF Fusion → Results

**Infrastructure:** Memgraph (graph DB with vector + BM25 support, port 7687), TEI (embeddings, port 8080)

## Code Style

- Python 3.14+, line length 120
- Ruff for linting/formatting, ty for type checking
- Known first-party import: `code_atlas`
- Conventional commits: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`
- Rust: `cargo fmt` + `cargo clippy` with `-D warnings`

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
- Fixtures centralized in `conftest.py`
- **High gear (default):** Integration tests exercising full workflows and public APIs
- **Low gear (selective):** Unit tests only for complex algorithms or edge cases unreachable via integration
- Don't test every function. Test system behavior.

## Commits

- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) format: `<type>(<scope>): <description>`
- Commit immediately when task is done
- Amend for feedback: `git add . && git commit --amend --no-edit`
- New commit only for genuinely separate work
- Never unstage changes that would cause data loss (e.g., don't `git reset` if it would discard changes)

**Version bumping (semantic versioning)**

## Task Tracking

- `.tasks/.roadmap.md` - Master roadmap with all epics and task status tables
- `.tasks/<epic>-<task>.md` - Individual task specs (e.g., `01-foundation-01-memgraph-schema.md`)
- `.tasks/research/` - Research notes and spike findings
- `.tasks/archive/` - Completed tasks moved here on completion
- Read `.tasks/.roadmap.md` before starting work to understand current priorities and status

## Configuration

- `atlas.toml` - Project configuration (scope, embeddings, search settings, detectors)
- `.atlasignore` - Gitignore-style exclusion patterns for indexing
- Environment variables: `ATLAS_*` prefix with double-underscore nesting (e.g., `ATLAS_EMBEDDINGS__PROVIDER`)
