# ADR-0003: Python/Rust Hybrid Architecture

## Status

Superseded by [ADR-0006](./0006-pure-python-tree-sitter.md)

## Date

2025-02-07

## Context

Code Atlas needs to:

1. **Parse source code quickly** — AST extraction for potentially 100K+ files
2. **Orchestrate complex workflows** — indexing pipeline, query routing, context assembly
3. **Serve as an MCP server** — protocol handling, tool definitions
4. **Integrate with AI ecosystems** — LiteLLM, embeddings, agent frameworks

We need to choose an architecture that balances performance for parsing with productivity for orchestration.

## Decision

We will use a **Python/Rust hybrid architecture**:

- **Rust (`crates/atlas-parser`)**: Fast AST parsing via tree-sitter
- **Python (`src/code_atlas`)**: CLI, MCP server, indexing orchestration, query routing

## Consequences

### Positive

- **Best of both worlds**: Rust's performance for the hot path (parsing), Python's productivity for orchestration

- **Tree-sitter native performance**: Tree-sitter is a C library with excellent Rust bindings. Rust parser can achieve
  10,000+ files/second

- **Rich Python ecosystem**: MCP SDK, LiteLLM, httpx, tiktoken, numpy — all Python-native. No FFI overhead for these
  integrations

- **Familiar development**: Most contributors will work in Python. Rust is isolated to a well-defined component

- **Subprocess simplicity**: Parser as a CLI subprocess avoids PyO3/maturin complexity for v1. Can optimize to native
  bindings later if needed

### Negative

- **Two language ecosystems**: Requires Rust toolchain in addition to Python

- **Subprocess overhead**: Calling Rust parser via subprocess adds latency (mitigated by batching)

- **Build complexity**: Need both `uv` (Python) and `cargo` (Rust) in CI/dev environment

### Risks

- If subprocess overhead becomes significant, may need PyO3 bindings
- Contributors unfamiliar with Rust may avoid parser contributions

## Alternatives Considered

### Alternative 1: Pure Python

Use tree-sitter Python bindings (`py-tree-sitter`) for parsing.

**Why rejected:**

- Python GIL limits parallel parsing performance
- tree-sitter Python bindings are less ergonomic than Rust
- Would be 5-10x slower for large codebases

### Alternative 2: Pure Rust

Rewrite everything in Rust for maximum performance.

**Why rejected:**

- MCP SDK is Python-first; Rust MCP support is nascent
- LiteLLM, tiktoken, and other AI libraries are Python
- Significantly more development effort
- Smaller contributor pool

### Alternative 3: PyO3 from Day 1

Use PyO3 to expose Rust parser as a native Python module.

**Why rejected:**

- Adds build complexity (maturin, cross-platform wheels)
- Subprocess is sufficient for v1 performance targets
- Can migrate to PyO3 later if profiling shows subprocess overhead

## Interface Contract

The Rust parser communicates with Python via JSON over stdin/stdout:

```bash
# Single file
echo '{"file": "src/auth/service.py"}' | atlas-parser parse

# Batch mode
atlas-parser parse-batch src/ --output entities.jsonl
```

This contract allows future migration to PyO3 without changing Python code logic.

## References

- [Tree-sitter Rust bindings](https://github.com/tree-sitter/tree-sitter/tree/master/lib/binding_rust)
- [PyO3](https://pyo3.rs/)
- [mcp-python SDK](https://github.com/modelcontextprotocol/python-sdk)
