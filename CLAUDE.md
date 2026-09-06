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
atlas index --full               # Re-check every file (re-parse, ignore the file_hash gate) — DESTROYS NOTHING
atlas index --reset --yes        # DESTRUCTIVE: delete the project's graph data and rebuild it
atlas index --reset-embeddings --yes  # DESTRUCTIVE: drop vectors only, keep the graph (model/dimension switch)
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
atlas bench                      # Index a corpus and report per-stage cost (throwaway SQLite db by default)
atlas bench --repo <url> --ref <sha>   # ...against a pinned, cached clone instead
atlas bench --record-baseline    # Store this run as the baseline; never happens automatically
atlas bench --embed-transport    # Route embeddings over a local loopback endpoint (latency, jitter, 429s)
```

**Benchmarks (ADR-0043).** `atlas bench` separates three numbers that a single wall-clock
figure blends: **work**, **contention** (lock/semaphore waits) and **pacing** (drain settle,
batch windows, poll intervals). Production pacing constants are read and reported, never
lowered — on a small corpus pacing is most of the run.

- **Counters are the comparison, not clocks.** Round-trips per `op`, rows, statements, events
  and tokenizer calls reproduce exactly on any machine; `--record-baseline` then compares them
  exactly, while stored timings never drive a verdict.
- **A different machine, backend or corpus commit produces no comparison at all** — a report,
  not a lenient threshold.
- **It never reaches a provider.** Embeddings are stubbed at `litellm.aembedding`, the one line
  on that path that leaves the machine, so chunking, batching, the limiter and the dedup lookup
  stay real and measured.
- It defaults to a **throwaway SQLite database**, pinned explicitly rather than left on `auto`,
  so a benchmark can never write into the index you actually query.

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
under the same model, and copies it. Valkey carries streams, consumer groups, the indexer lease and the
rate-limit buckets — no vectors.

**Not every parser is tree-sitter any more.** A `LanguageConfig` may set `language=None` and
supply `text_parse_func` instead, for line-oriented indentation-scoped formats tree-sitter can
only reach with an external scanner. TMDL (`parsing/languages/powerbi.py`) is the only one, and
Microsoft publishes no grammar for it. Consequences worth knowing before touching `parsing/ast.py`:

- `config.language` and `config.query` are `| None`. Anything that does `Parser(config.language)`
  must guard — `tests/support/langcov.py` does not, which is why TMDL has no coverage floor: its
  `LangSpec` is built from tree-sitter node-type names and cannot express a grammar-less language,
  and `named_funcs`/`calls` both return **1.0 when there is nothing to measure**, so wiring one in
  naively records a floor that guards nothing.
- The hatch skips the grammar and nothing else. `_parse_hazard`, the `RecursionError` catch, the
  empty-ParsedFile-on-decline rule and content hashing all still apply.
- `__post_init__` enforces exactly one handler, matching the grammar, at registration.

**An extension can be claimed by the format inside it (ADR-0048).** `register_dialect(ext, name,
sniff)` in `parsing/ast.py` lets an application format claim files of a suffix somebody else owns;
`.json` is wired to it, with no built-in dialects yet. First registered match wins, the order is
`_BUILTIN_LANGUAGE_MODULES`, a sniff sees 4 KiB and never parses, and anything it does not claim
falls to the generic handler unchanged. Two rules:

- **The generic handler is the floor.** No match, an unknown language name, or a sniff that raises
  all land there. A route that swallowed unmatched files would replace every config entity in the
  graph and never error.
- **Claiming is a commitment.** A dialect that claims a file and then declines gets an _empty_
  `ParsedFile`, not a fallback — so the claimed file's entities are deleted from the graph. A sniff
  sees 4 KiB of bytes and never the path, so it cannot anticipate a path-shaped decline. `.xml` is
  the worked example: `salesforce` is a registered dialect of it, and its handler is a wrapper in
  `config.py` that falls back to the generic structural parse precisely so a decline cannot delete
  anything. (This replaced the direct `_parse_xml` hand-off in ATL-176 — there is one mechanism now.)

`get_language_for_file(..., resolve_content=False)` answers "is this indexable at all" without
reading the file. `FileScope.scan` is the caller that needs it; with resolution on, it reads every
ambiguous-suffix file in the repo to pick between dialects that would answer identically.

**Indexing and embedding are two decisions (ADR-0047).** `[embeddings] exclude` / `include` /
`exclude_kinds` say which entities get a vector, in `[scope]`'s gitignore dialect. `include` beats
both axes; `exclude_kinds` replaces its default when set. The default is a **kind** rule,
`["config_setting", "config_section"]`: those come from exactly one code path, `config.py`'s generic
structural fallback, so they are precisely "data no dialect could read". A recognised dialect
(`ci_job`, `k8s_resource`, `dbt_source`, …) keeps its vector with no re-admit line.

- **Three gate sites, and the third is the one that bites.** The AST stage stops work being queued,
  `EmbedConsumer._allowed_by_policy` catches what is already on the stream, and
  `_reconcile_missing_embeddings` must be gated or it re-queues every excluded entity on every run,
  forever, while shouting that earlier embed work was lost.
- **Excluded is not invisible.** The node keeps its name, edges and FTS document. Where a query gates
  on vector similarity, `_floor_excluded_in_vector_channel` admits it at the tail of the vector list
  instead of dropping it — `analyze_query` weights vector at 2.0, so absence would be a silent
  handicap. Only uids another channel surfaced, and only policy-excluded ones: a vector missing by
  accident is a pipeline hole, and flooring it would hide it.
- **No node property and no `SCHEMA_VERSION` bump** — a bump drops the vector indices (ADR-0024). The
  policy is recomputed from settings wherever it is needed, so a change takes effect on the next index.
- `_reclaim_excluded_embeddings` strips vectors bought under an older policy at the end of an index.
  Pure graph work; it never re-bills anything, which `--reset-embeddings` would.

**Pacing follows the backend (ADR-0044):** `backend.queue = "sqlite"` gets a `SqliteRateLimiter` holding the
same buckets and AIMD factor in `ratelimit.sqlite3`; `"valkey"`/`"auto"` get the Valkey one. The two are
hand-written mirrors (Lua cannot call Python) and are pinned together by
`tests/integration/search/test_ratelimit_conformance.py` — edit both, or that test fails.

**The SQLite side tables are keys, not copies (ADR-0045, ADR-0046).** Both are keyed by `nodes.rowid`,
so neither survives a `VACUUM` — nothing runs one. The FTS document is deleted by rowid because `uid` is an
`UNINDEXED` fts5 column and deleting by it scans the whole table (quadratic in graph size). The `vec0` table
holds **bit-quantized** vectors: a query shortlists `k * _VEC_OVERSAMPLE` by hamming distance, then ranks
those by real `vec_distance_cosine` against `nodes.embedding`, so only the shortlist is approximate. Both
shapes carry a backend-local `meta` marker (`fts_key`, `vec_kind`) rather than a `SCHEMA_VERSION` bump —
that number is shared with Memgraph, where advancing it drops its vector indices.

**`.atlas/` ignores itself.** `ensure_sqlite_data_dir` writes a `.gitignore` when it creates the directory,
because the index is machine-specific and the target repo's `.gitignore` is not ours to edit. Only on first
creation, so deleting the file sticks.

**Oversized nodes (ADR-0040):** set `[embeddings] max_input_tokens` for any routed model name — litellm's
registry has no entry for one, and an unknown cap means no chunking and no truncation, so a single
over-length node fails the whole 128-text provider call it was batched into. Past the cap, a _document_
section splits into several nodes (`#partN`) while _code_ keeps one node and gains `EmbedChunk` overflow
vectors, scored at the node's best chunk. A code entity that needs chunking logs a warning; that is usually
a function too large to be one unit. Notes are never split — their uid is an address `LINKS_TO` points at.

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
  loop is not. The gate is `uv run pytest tests/unit -n auto` (~70s) — a **bare `uv run pytest` also collects
  `tests/integration` and `tests/bench`**, because nothing in `addopts` deselects them, so it needs Docker and
  runs for 15+ minutes. Two runs were abandoned at 400s and 600s before this was noticed; the symptom looks
  like a hang, not like a wider selection.
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
- **Unit tests cannot reach off-box.** `tests/unit/conftest.py` allowlists loopback only, so
  a test that stops intercepting litellm makes a named traceback instead of a real billed
  call. A host allowlist rather than a blanket `disable_socket`, because on Windows
  asyncio's event loop needs a loopback `socketpair()` to exist at all. Integration and
  bench are untouched -- different conftest.
- **Warnings are errors.** Every ignore in `filterwarnings` names why it is not ours to
  fix; new ones need the same justification rather than being appended quietly.
- **Property tests for invariants, not more examples.** `hypothesis` is in the dev group.
  The pattern to copy is `tests/unit/parsing/test_source_shims.py`: the Apex and dbt
  shims promise `len(out) == len(src)` with newline offsets intact, and that is one
  property covering input nobody would think to write down.
- **`time-machine` instead of sleeping.** Leases, TTLs, cooldowns and staleness are all
  time-based; jump the clock (see `tests/unit/backends/test_sqlite_queue.py`) rather than
  waiting or, worse, leaving the path uncovered because covering it took a minute.
- **`pytest-randomly` is wanted but not yet added.** It is the highest-value plugin still
  missing -- this codebase has module-level mutable singletons in `telemetry.py` and a
  one-shot `_discovered` flag in the parser registry, and several tests monkeypatch
  globals, so order-dependence is currently invisible. Expect adding it to surface work
  rather than save it, and note it needs `-p no:randomly` alongside `--testmon`.
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

Resolved most-specific-first: init kwarg > `ATLAS_*` env > `atlas.local.toml` > `atlas.toml`. All of it is
discovered from the **git root**, so the directory you run from never changes what is loaded — a config file
in a sub-directory is not read, and `cli._warn_shadowed_config` says so when one exists.

- `atlas.toml` — committed, describes the codebase (scope, search settings, detectors, monorepo layout, vault)
- `atlas.local.toml` — gitignored, merges per key over the above. For `[redis]`, `[memgraph]`, `[embeddings]`,
  `[backend]`, which differ per machine and should not be in the shared file. The exception is
  `[embeddings]`'s policy keys (`exclude`, `include`, `exclude_kinds`) — those describe the codebase,
  not the machine, and belong in the committed `atlas.toml`
- Environment variables: `ATLAS_*` prefix with double-underscore nesting (e.g. `ATLAS_EMBEDDINGS__MODEL`).
  Atlas never reads `.env` itself — export from `.envrc` (direnv) if you want that
- `.atlasignore` — gitignore-style exclusion patterns for indexing
