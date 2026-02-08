# ADR-0006: Pure Python with In-Process Tree-sitter

## Status

Accepted (supersedes [ADR-0003](./0003-python-rust-hybrid.md))

## Date

2026-02-08

## Context

ADR-0003 chose a Python/Rust hybrid where a Rust binary (`atlas-parser`) parsed source files via tree-sitter and
communicated with Python via JSON over stdin/stdout. After prototyping the indexing pipeline (ADR-0004), we measured the
actual cost breakdown:

- **Subprocess overhead** (spawn, JSON serialization, IPC) exceeded the parse time itself for typical files
- **Build complexity** required both `uv` and `cargo` toolchains in dev/CI/Docker
- **Contributor friction** — Rust was isolated to one component, but still required a full toolchain install
- **Parallelism** is already handled by the event bus (multiple Tier 2 consumer instances via Valkey Streams), not by
  Rust's threading model

Meanwhile, `py-tree-sitter` uses the exact same C parsing library (tree-sitter) via Python bindings. The grammar
packages (`tree-sitter-python`, etc.) ship pre-compiled wheels — no compilation step needed.

## Decision

Drop the Rust binary (`crates/atlas-parser`) and use **py-tree-sitter** called in-process within the Tier 2 pipeline
consumer. The parser module lives at `src/code_atlas/parser.py`.

### Architecture

```
Tier 2 Consumer
  └── parser.parse_file(path, source, project_name)
        └── tree-sitter C engine (via py-tree-sitter bindings)
              └── tree-sitter-python grammar (pre-compiled wheel)
```

### Parallelism Model

Multiple Tier 2 consumer instances can run concurrently — each pulls from the `atlas:ast-dirty` Valkey Stream via its
own consumer group member. This gives process-level parallelism without the GIL concern, since each consumer is an
independent process.

## Consequences

### Positive

- **Simpler build**: Single `uv sync` installs everything, no Rust toolchain required
- **Simpler Docker**: Single-stage Python image, no Rust builder stage
- **Same C engine**: py-tree-sitter wraps the identical tree-sitter C library — no parsing quality difference
- **Lower latency**: In-process call eliminates subprocess spawn + JSON serde overhead
- **Easier testing**: Parser is a pure function, unit-testable without subprocess mocking

### Negative

- **GIL-bound within a single process**: Multiple files parsed sequentially in one consumer. Mitigated by running
  multiple consumer instances.
- **No Rust ecosystem access**: If we need Rust-specific optimizations later, we'd need to reintroduce a Rust component.

### Escape Hatch

If profiling shows the Python GIL is a bottleneck for parsing throughput, we can build a PyO3 extension module that
exposes the same `parse_file()` interface but runs tree-sitter in Rust without subprocess overhead. The Python API stays
identical.

## Alternatives Considered

### Keep Rust binary, optimize subprocess

Reduce overhead via batching (send multiple files per invocation) or Unix domain sockets.

**Why rejected:** Still requires dual toolchain. The event bus already provides batching — adding another batching layer
inside the parser adds complexity without clear benefit.

### PyO3 from day one

Build a native Python extension in Rust using PyO3/maturin.

**Why rejected:** Adds maturin build complexity (cross-platform wheels, CI matrix). py-tree-sitter already provides
pre-compiled wheels with the same C engine. Only worth it if profiling shows a bottleneck.

## References

- [ADR-0003](./0003-python-rust-hybrid.md) — Superseded decision
- [py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter) — Python bindings
- [tree-sitter-python](https://github.com/tree-sitter/tree-sitter-python) — Python grammar
