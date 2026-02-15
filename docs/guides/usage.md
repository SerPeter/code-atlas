# CLI Usage

## Indexing

```bash
# Index the current directory
atlas index .

# Index a specific project
atlas index /path/to/project

# Index specific paths (monorepo)
atlas index . --scope services/auth --scope libs/shared

# Full re-index (re-embeds all entities)
atlas index --full
```

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

## Agent Integration

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
