# CLI Usage

## Indexing

```bash
# Index the current directory
atlas index .

# Index a specific project
atlas index /path/to/project

# Index specific paths (monorepo)
atlas index . --scope services/auth --scope libs/shared

# Re-check every file: enumerate all of them and re-parse each one even if its
# bytes are unchanged. Deletes nothing, and content/embedding hashes still decide
# what is rewritten — so a re-check where nothing changed costs zero provider calls.
# This is the one to reach for after a parser or configuration change.
atlas index --full

# DESTRUCTIVE — delete the project's graph data and rebuild it from scratch. On a
# monorepo this deletes every sub-project the run visits. Prints what it is about to
# remove and waits for a yes; --yes is required when there is no TTY.
atlas index --reset --yes

# DESTRUCTIVE — drop this project's vectors, embed hashes and EmbedChunk nodes and
# keep the graph, so the next pass re-embeds without re-parsing. For a model or
# dimension switch. A *dimension* change clears every project in the database,
# because the vector indices are shared.
atlas index --reset-embeddings --yes
```

The three flags set three independent axes and cannot be combined — see
[ADR-0042](../adr/0042-reindex-scope-and-destruction-are-separate-decisions.md).

## Search

```bash
# Hybrid search (default — fuses graph, BM25, and vector)
atlas search "authentication middleware"

# Graph search with Cypher
atlas search --type graph "MATCH (f:Callable)-[:CALLS]->(g) WHERE g.name = 'validate_token' RETURN f"

# Keyword search
atlas search --type keyword "DATABASE_URL"
```

## Status and Health

```bash
# Show indexed projects and entity counts
atlas status

# Infrastructure health check
atlas health
```

## Daemon Mode

```bash
# Start file watcher + indexing pipeline
atlas daemon start

# Stop daemon
atlas daemon stop
```

## MCP Server

```bash
# Start the MCP server (for AI coding agents)
atlas mcp
```

### MCP Client Configuration

Add to your Claude Code / Cursor / Windsurf MCP config:

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

## Configuration

`atlas.toml` is committed and describes the codebase — scope, detectors, rationale markers, monorepo layout, knowledge
vault. It is the same for everyone working on the repository.

Settings that differ per machine — `[redis]`, `[memgraph]`, `[embeddings]`, `[backend]` — belong somewhere else. Two
options, most specific first:

| Layer              | Committed?      | Use it for                                            |
| ------------------ | --------------- | ----------------------------------------------------- |
| `ATLAS_*` env vars | no              | one-off overrides, CI, anything already in your shell |
| `atlas.local.toml` | no (gitignored) | machine-specific values you want to persist           |
| `atlas.toml`       | yes             | everything about the codebase itself                  |

Nested sections use a double underscore: `ATLAS_BACKEND__GRAPH__MEMGRAPH__HOST=box.local`. Atlas never reads `.env`
itself — export from `.envrc` (direnv) if you want them loaded automatically.

`atlas.local.toml` merges per key rather than replacing the file, so this is enough to point one developer at a
different Memgraph while inheriting everything else:

```toml
# atlas.local.toml
[backend.graph.memgraph]
host = "box.local"
```

Declaring a backend is how you select it — there is no separate `graph = "memgraph"` key. Omit `[backend.graph.*]`
entirely and Atlas probes Memgraph and falls back to the embedded SQLite backend when it is unreachable; declare one and
an unreachable backend is an error instead. Since `atlas.local.toml` merges per key, a machine that wants the embedded
backend while the committed file declares Memgraph writes:

```toml
# atlas.local.toml
[backend.graph]
memgraph = false
sqlite = {}
```

Both files are discovered from the **git root**, so it does not matter which directory you run `atlas` from — a
sub-directory of the repo resolves to the same project, the same config and the same indexer lease. A config file inside
a sub-directory is not read.

## Agent Integration

### Claude Code Hooks

Agents reach for Grep, Read and subagents before an MCP tool, whatever the instructions say: the built-in tools are
always loaded, MCP tools are often deferred, and the built-in Explore and Plan subagents never read `CLAUDE.md`. The
hooks put the routing where the decision is made:

```bash
atlas hooks install              # ~/.claude/settings.json, every project
atlas hooks install --strict     # ...and block once before a graph-answerable grep or an exploration subagent
atlas hooks install --scope local  # .claude/settings.local.json in this repo only
atlas hooks uninstall
```

| Hook                           | What the agent sees                                                                                                                                                                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SessionStart`                 | A five-line "which tool for which question" card — only when this repo is indexed and code-atlas is usable. Plus a health notice, shown to you and relayed by the agent, whenever a check is not OK: graph, schema, queue, embedding provider, or this repo not being indexed |
| `SubagentStart`                | The same card, inside every subagent (the only way to reach Explore/Plan)                                                                                                                                                                                                     |
| `PostToolUse` Grep/Bash        | After a symbol-shaped grep the graph can answer: one line with the definition and dependent count                                                                                                                                                                             |
| `PreToolUse` (`--strict` only) | Denies the first such grep, and the first Explore/general-purpose/Plan subagent or workflow, per agent                                                                                                                                                                        |

Grep output is never replaced: grep is exhaustive over literal text and the graph is not, so a substitution would turn
an index gap into a confident wrong answer. Strict blocks fire once per kind per agent — re-issuing the call is allowed
— and stop entirely once that agent has used any code-atlas tool. Every path fails open: a hook error never blocks a
tool call, an unhealthy graph produces the notice and no hints or blocks, and a repo where `~/.claude.json` disables or
does not configure the code-atlas server gets nothing at all. A server configured where the hook cannot see it (a
plugin, managed settings, `--mcp-config`) needs `ATLAS_HOOKS_MCP=1`. A linked worktree whose own project is not indexed
uses the main checkout's index, and says so.

The health notice and `atlas health` reach their conclusions through the same code — `code_atlas.health_verdicts` — so
they report the same status, message and fix for the same situation; a parity test pins that. They differ only in how
they reach a backend: the hook uses bare TCP connects and its own queries, because importing `atlas health`'s clients
costs ~4 s and a session-start hook delays the first reply. Like the rest of Atlas it reads no `.env`: it sees the
environment Claude Code was started with. It adds one check `atlas health` does not have: whether _this_ repository is
indexed. The indexer lease and staleness checks stay `atlas health`-only.

The embedding check is `atlas health`'s own — one real embedding call, which proves config, model and credentials
together. It costs ~5–6 s (litellm's import plus the round trip), so the hook caches a **success** for 24 h, keyed by
provider, model, endpoint and a hash of the credential environment variables. A failure is never cached: a broken key is
re-checked, and reported, every session until fixed. A call that outlasts `check_timeout_s` is reported as slow, not as
a bad key. A background `asyncRewake` hook was tried and dropped: its exit-2 result on `SessionStart` never reached the
model.

How long anything waits before calling a backend down is one setting, shared by the hook, `atlas health`, `atlas doctor`
and the MCP `health_check` tool. It is per machine, so it goes in `atlas.local.toml` (or
`ATLAS_HEALTH__CONNECT_TIMEOUT_S`):

```toml
# atlas.local.toml
[health]
connect_timeout_s = 1.0 # Memgraph / Valkey reachability. Raise (e.g. 5) for a remote backend.
check_timeout_s = 3.0   # atlas health's queries, staleness check and embedding-provider round trip
```

Every backend, and every address of each (`localhost` is both `::1` and `127.0.0.1`), is probed at once, so a down
backend costs at most one `connect_timeout_s` however many are down. `atlas health` runs its Memgraph, Valkey, lease and
embedding checks concurrently too. Measured on Windows with the default 1.0: ~0.9 s healthy, ~1.5 s with Memgraph down,
~1.6 s with Memgraph and Valkey both down, ~2.6 s with both down at 5.

The hooks run `python -m code_atlas.hooks`, pinned by absolute path to the **uv tool install** of code-atlas when there
is one (`uv tool install code-atlas-mcp`), else to the interpreter running `atlas hooks install`; `--python` overrides
both. Installing from a checkout's development venv would tie every Claude Code session to a venv that `uv sync`
rewrites, so the command warns when that is the only choice. Point the MCP server at the same install — an absolute path
to its `atlas` rather than a bare `atlas`, which resolves to whichever venv is active when Claude Code starts. Re-run
after moving the install. A symbol lookup costs ~0.5 s and happens only for identifier-shaped searches; everything else
exits in ~0.13 s.

**Measuring whether it works.** Claude Code's own OpenTelemetry export already records every tool call, so there is
nothing to add to the hooks. With `CLAUDE_CODE_ENABLE_TELEMETRY=1`, `OTEL_LOGS_EXPORTER=otlp` and
`OTEL_LOG_TOOL_DETAILS=1` pointed at the `telemetry` compose profile, VictoriaLogs answers:

```text
# tool mix, including code-atlas calls (tool_name "mcp_tool", server in tool_parameters)
_time:7d event.name:tool_result | stats by (tool_name) count() as n | sort by (n desc)

# strict-mode blocks
_time:7d event.name:tool_decision decision:reject source:hook | stats by (tool_name) count()
```

### Guidelines for Agent Instructions

Copy the following into your project's `CLAUDE.md`, `.cursorrules`, or agent instructions file so your AI agent follows
Code Atlas best practices (see [Repository Guidelines](repo-guidelines.md) for the full rationale):

```markdown
## Code Atlas Guidelines

This codebase is indexed by Code Atlas. Follow these practices for best results:

- Write a concise first-line doc comment on every public function, class, and module — it is embedded for semantic
  search
- Add type annotations to all signatures — they create USES_TYPE graph edges and improve search
- Use named imports not wildcards — each creates an IMPORTS edge for graph analysis
- Keep modules focused (one concept per file) — clean graph neighborhoods, efficient delta detection
- Use descriptive names consistently — graph and BM25 search tokenize on name boundaries
- Use standard decorators for routes, commands, tests — pattern detectors create typed edges
- Use explicit class inheritance — AST parser extracts INHERITS edges for hierarchy analysis
- Exclude generated/vendored code via .atlasignore — reduces graph noise
- Write file-level doc comments — they are embedded for module-level vector search
- Keep functions small and focused — each is a separately searchable, delta-tracked entity
```

### Exploration Subagent

If you use Claude Code, you can define a custom [subagent](https://code.claude.com/docs/en/sub-agents) that explores
your codebase via Code Atlas. The subagent runs on a faster model with pre-granted access to Code Atlas tools, so
exploration stays out of your main context and doesn't prompt for permission on every call.

Save the following as `.claude/agents/explore-atlas.md` (project-level) or `~/.claude/agents/explore-atlas.md` (all
projects):

```markdown
---
name: explore-atlas
description:
  Explore and answer questions about the codebase using Code Atlas graph search. Use proactively when the user needs to
  understand code structure, find entities, or trace dependencies.
tools:
  Read, Glob, Grep, mcp__code-atlas__hybrid_search, mcp__code-atlas__get_node, mcp__code-atlas__get_context,
  mcp__code-atlas__get_usage_guide, mcp__code-atlas__schema_info, mcp__code-atlas__validate_cypher,
  mcp__code-atlas__cypher_query, mcp__code-atlas__plan_search_strategy
disallowedTools: Write, Edit
model: sonnet
mcpServers: code-atlas
memory: project
---

You are a codebase explorer. Use Code Atlas MCP tools to answer questions about this codebase.

Start by calling get_usage_guide() for an overview of available tools, then use hybrid_search as your primary search
tool. Use get_node for exact name lookups and get_context to expand into a node's neighborhood (parent, callers,
callees, docs). Use cypher_query for structural traversals (always call validate_cypher first).

Combine Code Atlas tools with Read/Glob/Grep for source-level detail when graph results need more context.
```

Claude will automatically delegate exploration tasks to this subagent. You can also invoke it explicitly: _"Use the
explore-atlas subagent to find how authentication is implemented."_
