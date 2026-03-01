# Benchmarks

Benchmarked on a synthetic codebase with deterministic Python files (classes, methods, imports, docstrings).

## Parsing

| Codebase | Files | Entities | Time  | Throughput        | Peak Memory |
| -------- | ----- | -------- | ----- | ----------------- | ----------- |
| Small    | 100   | 1,500    | 0.14s | **709 files/sec** | 1.4 MB      |
| Medium   | 1,000 | 15,000   | 1.6s  | **608 files/sec** | 15.8 MB     |

Memory scales linearly (~16 KB per entity) when accumulating all parse results. In production, entities are streamed to
the graph and not held in memory simultaneously.

## Query Latency (avg / p95)

Measured on the code-atlas repo (~1,400 entities), 5 iterations, local TEI embeddings. See `scripts/profile_query.py`.

| Tool            | Avg    | p95    | Max    |
| --------------- | ------ | ------ | ------ |
| `hybrid_search` | 548 ms | 677 ms | 677 ms |
| `text_search`   | 34 ms  | 36 ms  | 36 ms  |
| `vector_search` | 102 ms | 125 ms | 125 ms |
| `get_node`      | 7 ms   | 8 ms   | 8 ms   |
| `get_context`   | 34 ms  | 36 ms  | 36 ms  |
| `analyze_repo`  | 22 ms  | 23 ms  | 23 ms  |
| `index_status`  | 22 ms  | 23 ms  | 23 ms  |

Text and vector search queries run in parallel across all 5 indices via `asyncio.gather()`. The `get_node` cascade uses
2-stage UNION ALL queries (max 2 RTTs).

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
