# ADR-0002: Build From Scratch Rather Than Fork

## Status

Accepted

## Date

2025-02-07

## Context

Several open-source tools exist in the code graph intelligence space:

- **code-graph-mcp**: Fast AST parsing via ast-grep, but in-memory only (no persistence)
- **code-graph-rag**: Memgraph-based with good hierarchy model, but every query requires LLM
- **Kit**: Polished DX with semantic search, but no graph database
- **codegraph-rust**: 100% Rust with SurrealDB, but unproven graph performance

The initial analysis suggested forking code-graph-rag as a starting point due to its Memgraph foundation and hierarchy
model.

## Decision

We will build Code Atlas from scratch, borrowing design patterns and query patterns from existing tools rather than
forking.

## Consequences

### Positive

- **Clean architecture**: Design the graph schema to support our full requirements from day one (docs, monorepo,
  external libs, pattern detection)

- **No legacy baggage**: Existing tools have architectural assumptions we'd need to fight (LLM-required queries,
  in-memory only, no hybrid search)

- **Code quality control**: Build with our standards rather than inheriting and reworking

- **Clear ownership**: No confusion about what's shared with upstream; no maintenance burden for heavily modified fork

- **Faster iteration**: Can make breaking changes without coordinating with upstream

### Negative

- **More initial work**: No existing codebase to start from

- **Reinventing patterns**: Some patterns already solved in existing tools must be reimplemented

- **No community contributions**: Can't directly benefit from upstream improvements

### Risks

- May underestimate effort to build features that seem simple in existing tools
- Could miss edge cases that existing tools have already solved

## Alternatives Considered

### Alternative 1: Fork code-graph-rag

**Why rejected:**

- Graph model doesn't accommodate document nodes, external library stubs, monorepo hierarchy, or extended relationship
  types
- Architecture is built around LLM-translates-to-Cypher as the primary query path; our design inverts this (direct tools
  first, LLM optional)
- Would need to strip out and replace core architecture, effectively becoming a rewrite

### Alternative 2: Fork Kit

**Why rejected:**

- No graph database — would need to add Memgraph from scratch
- Flat file/symbol model would need complete restructuring for relationship-based queries
- Different language ecosystem (Kit is more JavaScript-focused)

## What We Borrow

| What                 | Source                              | How                                                                |
| -------------------- | ----------------------------------- | ------------------------------------------------------------------ |
| Hierarchy concept    | code-graph-rag                      | Design inspiration for Project → Package → Module → Class → Method |
| Tree-sitter queries  | Kit, code-graph-rag                 | Study language-specific symbol extraction patterns                 |
| Context assembly     | Kit's ContextAssembler              | Adapt token-budget-aware assembly concept                          |
| Memgraph patterns    | code-graph-rag, memgraph/ai-toolkit | Reference Bolt protocol usage and query patterns                   |
| MCP server structure | memgraph/ai-toolkit                 | Reference MCP ↔ Memgraph bridge patterns                           |

## References

- [code-graph-mcp](https://github.com/entrepeneur4lyf/code-graph-mcp)
- [code-graph-rag](https://github.com/vitali87/code-graph-rag)
- [Kit](https://github.com/cased/kit)
- [codegraph-rust](https://github.com/Jakedismo/codegraph-rust)
