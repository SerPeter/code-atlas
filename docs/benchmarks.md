# Benchmarks

Benchmarked on a synthetic codebase with deterministic Python files (classes, methods, imports, docstrings).

## Parsing

| Codebase | Files | Entities | Time  | Throughput        | Peak Memory |
| -------- | ----- | -------- | ----- | ----------------- | ----------- |
| Small    | 100   | 1,500    | 0.14s | **709 files/sec** | 1.4 MB      |
| Medium   | 1,000 | 15,000   | 1.6s  | **608 files/sec** | 15.8 MB     |

Memory scales linearly (~16 KB per entity) when accumulating all parse results. In production, entities are streamed to
the graph and not held in memory simultaneously.

## Query Latency (p50 / p95)

| Search Type      | p50   | p95   | p99    |
| ---------------- | ----- | ----- | ------ |
| Graph search     | 8 ms  | 63 ms | 66 ms  |
| BM25 text search | 10 ms | 19 ms | 29 ms  |
| Vector search    | 47 ms | 71 ms | 164 ms |

All three search types use single-round-trip queries against Memgraph.

## Concurrent Queries

| Concurrency | Total Queries | Wall Time | QPS     | Errors |
| ----------- | ------------- | --------- | ------- | ------ |
| 10          | 50            | 1.9s      | 26      | 0      |
| 50          | 250           | 1.1s      | **238** | 0      |

Zero errors under load. QPS scales well with concurrency thanks to Memgraph's connection pooling.

## Running Benchmarks

```bash
# Parser + memory (no infra needed)
uv run pytest tests/bench/test_bench_parser.py tests/bench/test_bench_memory.py -m bench -s

# Query + concurrent (requires Memgraph)
uv run pytest tests/bench/ -m bench -s

# Exclude benchmarks from regular test runs
uv run pytest -m "not bench"
```
