# ADR-0005: Deployment & Process Model

## Status

Accepted

## Date

2026-02-07

## Context

Code Atlas consists of several functional components — MCP server (query interface for agents), file watcher (change detection), indexing pipeline (AST parsing, graph writing, embedding) — that need to run together as a coherent system. Before implementing these components, we need to decide:

1. **How they're deployed**: Single process? Separate processes? Docker container? Agent-spawned?
2. **Startup behavior**: Cold start from scratch, or reconcile against existing index?
3. **MCP transport**: stdio (agent-spawned) vs SSE/HTTP (long-running server)?
4. **Lifecycle management**: Who starts/stops Code Atlas? How does it recover from crashes?

This decision affects the MCP server design (01-foundation-04), the event architecture scope (01-foundation-08), and the file watcher design (05-delta-04).

Key constraints:
- Must work on Windows, macOS, and Linux
- Must be compatible with MCP clients: Claude Code (stdio), Cursor (SSE), Windsurf
- Should feel lightweight — a developer tool, not a cloud service
- Memgraph and TEI are already Docker services; Code Atlas itself may or may not be containerized

## Decision

We adopt **Option D: Hybrid — Daemon + Agent-Spawned MCP**. Two separate processes with distinct responsibilities, decoupled via Redis Streams and Memgraph:

### Process Model

```
┌─────────────────────────────────────┐     ┌───────────────────────┐
│         atlas daemon start          │     │      atlas mcp        │
│  (long-running background process)  │     │ (per-agent-session)   │
│                                     │     │                       │
│  ┌───────────┐  ┌────────────────┐  │     │  ┌─────────────────┐  │
│  │   File    │  │ Tier Consumers │  │     │  │   MCP Server    │  │
│  │  Watcher  │  │ (1, 2, 3)     │  │     │  │  (stdio / SSE)  │  │
│  └─────┬─────┘  └───────┬────────┘  │     │  └────────┬────────┘  │
│        │                │           │     │           │           │
│        └───────┬────────┘           │     │           │           │
│                ▼                    │     │           ▼           │
│         ┌──────────┐               │     │    ┌──────────┐      │
│         │  Valkey   │               │     │    │ Memgraph │      │
│         │ (Streams) │               │     │    │ (read)   │      │
│         └──────────┘               │     │    └──────────┘      │
└─────────────────────────────────────┘     └───────────────────────┘
                    │                                  ▲
                    └──── writes ──► Memgraph ─────────┘
```

- **`atlas daemon start`** — Long-running process: file watcher publishes `FileChanged` events to Redis Streams, tier consumers (1→2→3) process them and write to Memgraph.
- **`atlas mcp`** — Lightweight, spawned per agent session. Reads Memgraph directly (no dependency on daemon for queries). Supports stdio (primary) and SSE (secondary) transport.
- **Redis decouples them** — the daemon indexes via streams, the MCP server queries the graph. They share Memgraph but don't communicate directly.

### MCP Transport

- **stdio** (primary) — Claude Code spawns `atlas mcp` as a child process. Zero config, natural lifecycle.
- **SSE** (secondary) — `atlas mcp --transport sse` for clients that prefer HTTP (Cursor, Windsurf). Requires the user to start the process manually or via daemon integration.

### Startup Sequence

```
                  ┌─────────┐
                  │  START   │
                  └────┬─────┘
                       ▼
              ┌────────────────┐
              │   Preflight    │    Check Memgraph + Valkey + TEI connectivity
              │   Checks       │
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
              ┌────────────────┐
              │  Reconcile     │    Compare filesystem vs index, enqueue stale files
              │  (progressive) │    Tier 1 first → Tier 2 → Tier 3
              └───────┬────────┘
                      ▼
              ┌────────────────┐
              │    READY       │    All consumers running, watcher active
              └────────────────┘
```

### Lifecycle State Machine

```
  INIT ──► PREFLIGHT ──► CREATING_GROUPS ──► STARTING_CONSUMERS
                                                      │
              STOPPING ◄── READY ◄── RECONCILING ◄───┘
                  │
                  ▼
               STOPPED
```

- **PREFLIGHT**: Validate Memgraph, Valkey, TEI connectivity
- **READY**: All consumers running, accepting file change events
- **STOPPING**: Graceful shutdown — finish in-progress batches, close connections
- **STOPPED**: Clean exit

### Multi-Workspace

Stream prefix per project root: `atlas:<hash>:file-changed`. This allows multiple daemon instances to share a single Valkey without collision. The `stream_prefix` in `atlas.toml` can be overridden per project.

### Configuration Discovery

1. Walk up from CWD to find nearest `atlas.toml`
2. Fall back to `~/.config/atlas/atlas.toml`
3. Environment variables (`ATLAS_*`) override everything

### User Flow

```
Install:    pip install code-atlas / uv add code-atlas
                    │
                    ▼
Infra:      docker compose up -d         (Memgraph + TEI + Valkey)
                    │
                    ▼
Daemon:     atlas daemon start           (file watcher + pipeline consumers)
                    │
                    ▼
Agent:      Agent spawns `atlas mcp`     (Claude Code: stdio, Cursor: SSE)
                    │
                    ▼
Queries:    Agent calls MCP tools  ─────► Memgraph ◄──── Daemon keeps fresh
```

### Data Flow at Runtime

```
  ┌──────────┐     FileChanged      ┌─────────┐       ASTDirty        ┌─────────┐
  │   File   │ ──► events ────────► │ Tier 1  │ ──► events ─────────► │ Tier 2  │
  │  Watcher │     (Redis Stream)   │ (graph) │     (Redis Stream)    │  (AST)  │
  └──────────┘                      └─────────┘                       └────┬────┘
                                                                      gate │
                                                                  EmbedDirty│
                                                                  (if sig)  │
                                                                      ┌────▼────┐
                                                                      │ Tier 3  │
                                                                      │ (embed) │
                                                                      └────┬────┘
                                                                           │
                                         ┌──────────┐                     │
        Agent ◄──── MCP Server ◄──── reads│ Memgraph │◄──── writes ───────┘
                   (stdio/SSE)            └──────────┘
```

## Consequences

### Positive

- **Separation of concerns** — daemon handles continuous indexing, MCP handles queries. Each is simple.
- **MCP is stateless** — spawned fresh per agent session, no daemon dependency for reads
- **Daemon survives agent restarts** — index stays fresh even when no agent is connected
- **Multi-transport** — stdio for Claude Code, SSE for Cursor/Windsurf, same MCP code
- **Progressive startup** — agents can query immediately (stale results), daemon freshens in background
- **Cross-platform** — no OS-specific process management; foreground process with Ctrl+C

### Negative

- **Two processes to manage** — developer must start daemon separately from agent
- **Daemon must be running for freshness** — if daemon is down, index becomes stale (but MCP still works with stale data)

### Risks

- Docker volume mount performance on macOS may affect file watcher latency (mitigation: Code Atlas runs natively, not in Docker)
- Startup reconciliation of large repos must be fast (target: 10K files < 10s)

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

- **01-foundation-04 (MCP Server)**: MCP server is a standalone lightweight process. Reads Memgraph directly. Supports stdio + SSE transport.
- **01-foundation-08 (Event Architecture)**: Event bus scope is cross-process (Redis Streams), not in-process. Consumer groups enable daemon scaling.
- **05-delta-04 (File Watcher)**: Watcher runs inside the daemon process, publishes `FileChanged` to Redis Streams.

## References

- [Spike task: 01-foundation-09](../../.tasks/01-foundation-09-deployment-model.md)
- [MCP server task: 01-foundation-04](../../.tasks/01-foundation-04-mcp-server.md)
- [Event architecture spike: 01-foundation-08](../../.tasks/01-foundation-08-event-architecture.md)
- [File watcher task: 05-delta-04](../../.tasks/05-delta-04-file-watcher.md)
- [MCP specification — transports](https://modelcontextprotocol.io/docs/concepts/transports)
- [Competitor research](../../.tasks/research/2026-02-07_competitor_consolidated-insights.md)
