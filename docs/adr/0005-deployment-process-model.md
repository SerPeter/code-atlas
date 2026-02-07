# ADR-0005: Deployment & Process Model

## Status

Accepted

## Date

2026-02-07

## Context

Code Atlas consists of several functional components — MCP server (query interface for agents), file watcher (change
detection), indexing pipeline (AST parsing, graph writing, embedding) — that need to run together as a coherent system.
Before implementing these components, we need to decide:

1. **How they're deployed**: Single process? Separate processes? Docker container? Agent-spawned?
2. **Startup behavior**: Cold start from scratch, or reconcile against existing index?
3. **MCP transport**: stdio (agent-spawned) vs HTTP (long-running server)?
4. **Lifecycle management**: Who starts/stops Code Atlas? How does it recover from crashes?

This decision affects the MCP server design (01-foundation-04), the event architecture scope (01-foundation-08), and the
file watcher design (05-delta-04).

Key constraints:

- Must work on Windows, macOS, and Linux
- Must be compatible with MCP clients: Claude Code (stdio), Cursor, Windsurf, VS Code, JetBrains
- Should feel lightweight — a developer tool, not a cloud service
- Memgraph and TEI are already Docker services; Code Atlas itself may or may not be containerized

This pattern is validated by gopls (Go language server), which uses the same daemon + thin forwarder architecture for
sharing state across editor sessions.

## Decision

We adopt **Option D: Hybrid — Daemon + Agent-Spawned MCP**. Two separate processes with distinct responsibilities,
decoupled via Valkey Streams and Memgraph:

### Process Model

```
┌─────────────────────────────────────┐     ┌───────────────────────────┐
│         atlas daemon start          │     │        atlas mcp          │
│  (long-running background process)  │     │   (per-agent-session)     │
│                                     │     │                           │
│  ┌───────────┐  ┌────────────────┐  │     │  ┌─────────────────────┐  │
│  │   File    │  │ Tier Consumers │  │     │  │    MCP Server       │  │
│  │  Watcher  │  │ (1, 2, 3)     │  │     │  │ (stdio / streamable │  │
│  └─────┬─────┘  └───────┬────────┘  │     │  │        HTTP)        │  │
│        │                │           │     │  └──────────┬──────────┘  │
│        └───────┬────────┘           │     │             │             │
│                ▼                    │     │             ▼             │
│         ┌──────────┐               │     │      ┌──────────┐        │
│         │  Valkey   │               │     │      │ Memgraph │        │
│         │ (Streams) │               │     │      │ (read)   │        │
│         └──────────┘               │     │      └──────────┘        │
└─────────────────────────────────────┘     └───────────────────────────┘
                    │                                    ▲
                    └──── writes ──► Memgraph ───────────┘
```

- **`atlas daemon start`** — Long-running foreground process: file watcher publishes `FileChanged` events to Valkey
  Streams, tier consumers (1→2→3) process them and write to Memgraph. Runs in foreground by default (Ctrl+C to stop).
- **`atlas mcp`** — Lightweight, spawned per agent session. Reads Memgraph directly (no dependency on daemon for
  queries). Supports stdio (primary) and Streamable HTTP (secondary) transport.
- **Valkey decouples them** — the daemon indexes via streams, the MCP server queries the graph. They share Memgraph but
  don't communicate directly.

### MCP Transport

| Transport                       | Flag               | Clients                           | Use Case                                      |
| ------------------------------- | ------------------ | --------------------------------- | --------------------------------------------- |
| **stdio** (primary)             | default            | Claude Code, VS Code, Cursor, all | Agent spawns `atlas mcp` as child process     |
| **Streamable HTTP** (secondary) | `--transport http` | VS Code, JetBrains, Roo Code      | Long-running server, manual or daemon-started |

- **stdio** — Agent spawns `atlas mcp` as a child process. Zero config, natural lifecycle. The process lives for the
  agent session, receives JSON-RPC via stdin, responds via stdout.
- **Streamable HTTP** — `atlas mcp --transport http --port 8081` for clients that prefer HTTP. Single endpoint, supports
  both JSON responses and SSE streaming per the MCP spec (2025-11-25).
- **SSE (deprecated)** — The old two-endpoint SSE transport from MCP spec 2024-11-05 is not supported. Clients still on
  SSE (Cursor, Windsurf as of Feb 2026) can use the `mcp-remote` bridge, or will migrate to Streamable HTTP as they
  update.

### Startup Sequence (Daemon)

```
                  ┌─────────┐
                  │  START   │
                  └────┬─────┘
                       ▼
              ┌────────────────┐
              │   Preflight    │    Check Memgraph + Valkey + TEI connectivity
              │   Checks       │    Write PID file
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Create Consumer│    Idempotent XGROUP CREATE for all 3 streams
              │    Groups      │
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Start Tier     │    asyncio.gather(tier1.run(), tier2.run(), tier3.run())
              │  Consumers     │
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │ Start File     │    Watch project root, publish FileChanged events
              │   Watcher      │    (TODO: 05-delta-04)
              └───────┬────────┘
                      ▼
              ┌────────────────┐    Git-based fast path: diff stored_commit..HEAD
              │  Reconcile     │    Fallback: mtime comparison for non-git or rebases
              │  (progressive) │    Enqueue stale files → Tier 1 → 2 → 3
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │    READY       │    All consumers running, watcher active
              └────────────────┘    MCP can already serve queries (stale is OK)
```

### Startup Reconciliation Strategy

On daemon startup, reconcile filesystem state against the indexed state in Memgraph:

1. **First run** (no indexed state in Memgraph) → full index
2. **Same commit** (`stored_commit == HEAD`) → check `git status` for uncommitted changes only, re-index dirty files
3. **Different commit** → `git diff --name-status stored_commit..HEAD` to find changed files, enqueue only those
4. **Non-ancestor commit** (force push, rebase) → fall back to mtime scan: compare file mtimes to stored mtimes,
   re-index files with newer mtime
5. **Non-git project** → mtime scan as primary strategy

Progressive availability: MCP server can query immediately using the existing (possibly stale) index. The daemon
freshens it in the background. `atlas status` shows reconciliation progress.

### Lifecycle State Machine

```
  INIT ──► PREFLIGHT ──► CREATING_GROUPS ──► STARTING_CONSUMERS
                                                      │
              STOPPING ◄── READY ◄── RECONCILING ◄───┘
                  │
                  ▼
               STOPPED
```

- **INIT**: Parse config, validate settings
- **PREFLIGHT**: Validate Memgraph, Valkey, TEI connectivity; check for existing daemon (PID file)
- **CREATING_GROUPS**: Idempotent `XGROUP CREATE` for all streams
- **STARTING_CONSUMERS**: Launch tier consumer tasks
- **RECONCILING**: Filesystem vs index reconciliation running in background
- **READY**: All consumers running, watcher active, accepting file change events
- **STOPPING**: Graceful shutdown — finish in-progress batches, close connections, remove PID file
- **STOPPED**: Clean exit

### Process Management

**Foreground by default** — consistent with developer tool conventions (gopls, rust-analyzer, language servers). The
daemon runs in the foreground and stops on Ctrl+C.

**PID file** for mutual exclusion — prevents multiple daemon instances for the same project:

```
Unix:    $XDG_STATE_HOME/code-atlas/daemon-<project-hash>.pid
         (default: ~/.local/state/code-atlas/)
Windows: %LOCALAPPDATA%\code-atlas\daemon-<project-hash>.pid
```

On `atlas daemon start`:

1. Compute `project-hash` from resolved project root path
2. Check PID file — if exists and process alive, print "daemon already running" and exit
3. If stale PID file (process dead), delete and continue
4. Write PID file on start, delete on clean shutdown

**Graceful shutdown:**

- Unix: handle SIGINT + SIGTERM via `loop.add_signal_handler()`
- Windows: handle SIGBREAK via `signal.signal()` (asyncio signal handlers not supported on Windows)
- Shutdown sequence: stop file watcher → wait for in-progress batches → close Valkey/Memgraph connections → remove PID
  file

### Multi-Workspace

Stream prefix per project root: `atlas:<hash>:file-changed`. This allows multiple daemon instances to share a single
Valkey without collision. The `stream_prefix` in `atlas.toml` can be overridden per project.

Each project root runs its own daemon instance. Multiple daemons share the same Valkey and Memgraph infrastructure.

### Configuration Discovery

Precedence (highest to lowest):

1. **CLI flag**: `atlas --config /path/to/atlas.toml`
2. **Environment variables**: `ATLAS_*` prefix, `__` nesting (e.g., `ATLAS_MEMGRAPH__HOST`)
3. **Project config**: Walk up from CWD to find nearest `atlas.toml`
4. **Global config**: `$XDG_CONFIG_HOME/atlas/atlas.toml` (Unix) / `%APPDATA%\atlas\atlas.toml` (Windows)
5. **Built-in defaults**: Hardcoded in `AtlasSettings` field defaults

Override semantics (Ruff model): nearest config wins completely, no cascading merge. This is simpler to reason about and
debug.

The `project_root` setting is derived from the directory containing the discovered `atlas.toml`, ensuring path
resolution works correctly from subdirectories.

Implementation note: pydantic-settings does not support walk-up discovery for custom TOML files. Requires a custom
`TomlConfigSettingsSource` subclass with `settings_customise_sources()` override.

### User Flow

```
Install:    uv tool install code-atlas     (or: pip install code-atlas)
                    │
                    ▼
Infra:      docker compose up -d           (Memgraph + TEI + Valkey)
                    │
                    ▼
Config:     atlas.toml in project root     (optional — sensible defaults)
                    │
                    ▼
Index:      atlas daemon start             (file watcher + pipeline consumers)
                    │
                    ▼
Agent:      Agent spawns `atlas mcp`       (Claude Code: stdio, VS Code: stdio or HTTP)
                    │
                    ▼
Queries:    Agent calls MCP tools ─────► Memgraph ◄──── Daemon keeps fresh
```

### Data Flow at Runtime

```
  ┌──────────┐     FileChanged      ┌─────────┐       ASTDirty        ┌─────────┐
  │   File   │ ──► events ────────► │ Tier 1  │ ──► events ─────────► │ Tier 2  │
  │  Watcher │     (Valkey Stream)  │ (graph) │     (Valkey Stream)   │  (AST)  │
  └──────────┘                      └─────────┘                       └────┬────┘
                                                                      gate │
                                                                  EmbedDirty│
                                                                  (if sig)  │
                                                                      ┌────▼────┐
                                                                      │ Tier 3  │
                                                                      │ (embed) │
                                                                      └────┬────┘
                                                                           │
                                       ┌──────────┐                       │
  Agent ◄──── MCP Server ◄──── reads   │ Memgraph │ ◄──── writes ────────┘
          (stdio / Streamable HTTP)    └──────────┘
```

## Consequences

### Positive

- **Separation of concerns** — daemon handles continuous indexing, MCP handles queries. Each is simple.
- **MCP is stateless** — spawned fresh per agent session, no daemon dependency for reads
- **Daemon survives agent restarts** — index stays fresh even when no agent is connected
- **Multi-transport** — stdio for most clients, Streamable HTTP for HTTP-preferring clients, same MCP code
- **Progressive startup** — agents can query immediately (stale results), daemon freshens in background
- **Cross-platform** — foreground process with Ctrl+C, no OS-specific service management required
- **Validated pattern** — gopls uses the same daemon + forwarder architecture successfully at scale

### Negative

- **Two processes to manage** — developer must start daemon separately from agent
- **Daemon must be running for freshness** — if daemon is down, index becomes stale (but MCP still works with stale
  data)

### Risks

- Docker volume mount performance on macOS may affect file watcher latency (mitigation: Code Atlas runs natively, not in
  Docker)
- Startup reconciliation of large repos must be fast (target: 10K files < 10s)
- Windows signal handling differs from Unix (SIGBREAK vs SIGTERM) — must test cross-platform shutdown

### Future Considerations

- **Auto-start daemon from MCP**: gopls auto-starts its daemon from the forwarder process. We could do the same — if
  `atlas mcp` detects no running daemon, start one automatically. Deferred to post-v1.
- **`atlas doctor` command**: Comprehensive diagnostics checking Python version, Rust parser binary, all services,
  config validity, index state. High value for support/debugging.
- **Background mode**: `atlas daemon start --background` using `subprocess.Popen` with platform-specific flags
  (`start_new_session=True` on Unix, `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows). Not needed for v1 —
  users can use `nohup`, systemd user services, or Task Scheduler.

## Alternatives Considered

### Option A: Single Long-Running Daemon (MCP + Indexer)

All components in one process. Event bus is in-process asyncio. Agent connects via SSE.

**Why rejected:**

- MCP server lifecycle is coupled to indexing — crash in indexer kills query serving
- stdio transport (needed for Claude Code) doesn't work well with a long-running daemon
- No separation between read path and write path

### Option B: Agent-Spawned Process (MCP + Indexer)

Agent spawns `atlas mcp` which includes watcher and indexer internally.

**Why rejected:**

- Cold start on every agent session — no background indexing
- Index goes stale between sessions
- Heavy process for what should be a lightweight MCP server

### Option C: Docker Compose Service

Code Atlas runs as a container alongside Memgraph and TEI, watches mounted volume.

**Why rejected:**

- Docker volume mount performance is poor on macOS (critical for file watching)
- Forces Docker dependency for the Code Atlas process itself (not just infrastructure)
- Harder to debug and iterate during development

## Impact on Other Tasks

- **01-foundation-04 (MCP Server)**: MCP server is a standalone lightweight process. Reads Memgraph directly. Supports
  stdio (default) + Streamable HTTP (`--transport http`) transport. Uses Python MCP SDK's `FastMCP.run(transport=...)`.
- **01-foundation-08 (Event Architecture)**: Event bus scope is cross-process (Valkey Streams), not in-process. Consumer
  groups enable daemon scaling. _Completed — see ADR-0004._
- **05-delta-04 (File Watcher)**: Watcher runs inside the daemon process, publishes `FileChanged` to Valkey Streams.
- **Settings (settings.py)**: Config discovery needs custom `TomlConfigSettingsSource` subclass for walk-up `atlas.toml`
  discovery. Current `toml_file="atlas.toml"` only checks CWD.

## References

- [Spike task: 01-foundation-09](../../.tasks/01-foundation-09-deployment-model.md)
- [Research notes: deployment model](../../.tasks/research/2026-02-07_deployment-model.md)
- [MCP server task: 01-foundation-04](../../.tasks/01-foundation-04-mcp-server.md)
- [Event architecture spike: 01-foundation-08](../../.tasks/archive/01-foundation-08-event-architecture.md)
- [File watcher task: 05-delta-04](../../.tasks/05-delta-04-file-watcher.md)
- [MCP specification 2025-11-25 — transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
- [gopls daemon mode](https://go.dev/gopls/daemon)
- [How Cursor Indexes Codebases Fast](https://read.engineerscodex.com/p/how-cursor-indexes-codebases-fast)
- [Ruff configuration discovery](https://docs.astral.sh/ruff/configuration/)
- [Competitor research](../../.tasks/research/2026-02-07_competitor_consolidated-insights.md)
