# ADR-0001: Use Memgraph as the Graph Database

## Status

Accepted

## Date

2025-02-07

## Context

Code Atlas requires a database that supports three search paradigms:

1. **Graph traversal** — Cypher queries for navigating code relationships (calls, inheritance, imports)
2. **Vector similarity search** — semantic search via embeddings
3. **BM25 keyword search** — exact matches for identifiers, error messages, config keys

Additionally, the database must:

- Run locally with reasonable resource consumption (developer laptops)
- Support persistent storage across sessions
- Use a query language that AI agents can generate effectively
- Have acceptable performance for interactive queries

We evaluated three candidates: **Memgraph**, **SurrealDB**, and **Neo4j**.

## Decision

We will use **Memgraph** as the unified database backend for Code Atlas.

## Consequences

### Positive

- **All three search types in one system**: Memgraph 3.6+ natively supports Cypher graph queries, Tantivy-based BM25
  text search, and ANN vector similarity search. No external services needed.

- **Cypher compatibility**: AI agents (Claude, GPT-4) have extensive Cypher knowledge from training data. SurrealQL is
  non-standard and would handicap agent query generation.

- **Performance for our workload**: Graph traversal and read-heavy workloads are our primary use pattern. Memgraph
  excels at reads (5-6x faster than Neo4j in benchmarks, sub-millisecond latency).

- **Reasonable resource footprint**: 200-500MB RAM for a typical codebase (50K-100K nodes) is acceptable for local
  development. Neo4j's JVM overhead requires 1-2GB baseline.

- **Existing ecosystem**: code-graph-rag already proves Memgraph + code graph works. memgraph/ai-toolkit provides MCP
  server patterns.

- **Python-native extensions**: Custom query modules in Python align with our stack.

### Negative

- **In-memory model**: Data must fit in RAM. For very large codebases (millions of nodes), this could become a
  constraint.

- **Smaller community**: Fewer Stack Overflow answers and tutorials compared to Neo4j.

- **BSL license**: Business Source License restricts offering as a hosted service, but this is not relevant for
  self-hosted developer tools.

### Risks

- If graph sizes exceed available RAM, we may need to implement pagination or sharding strategies.
- Memgraph updates may introduce breaking changes; pin versions carefully.

## Alternatives Considered

### Alternative 1: Neo4j

Neo4j is the industry-standard graph database with the largest ecosystem.

**Why rejected:**

- **Resource consumption**: JVM overhead requires 1-2GB RAM baseline, excessive for a developer-local tool
- **Performance**: 14-40ms typical query latency vs. Memgraph's 1-5ms
- **License**: Community edition limited to single instance; Enterprise requires commercial license
- **Vector search**: Native support is recent (2025) and less mature

### Alternative 2: SurrealDB

SurrealDB is a multi-model database (graph, document, vector, time-series) in a single Rust binary.

**Why rejected:**

- **Query language**: SurrealQL is proprietary and non-standard. AI agents trained on Cypher won't know SurrealQL,
  reducing agent effectiveness
- **Graph performance unproven**: No published graph traversal benchmarks; community reports suggest it lags behind
  dedicated graph databases for deep traversals
- **Stability concerns**: Multiple GitHub issues about memory leaks and abnormal resource consumption under load
- **Young project**: Breaking changes between versions more likely

**Future consideration:** If SurrealDB publishes graph traversal benchmarks and resolves stability issues, it could
become a compelling single-binary alternative.

## References

- [Memgraph vs Neo4j benchmark](https://memgraph.com/blog/memgraph-vs-neo4j-performance-benchmark-comparison)
- [Independent 3-way comparison](https://t4itech.com/blog/comparative-performance-analysis-of-graph-databases)
- [Critical benchmark analysis](https://maxdemarzi.com/2023/01/11/bullshit-graph-database-performance-benchmarks/)
- [Memgraph text search (Tantivy)](https://memgraph.com/blog/text-search-in-memgraph)
- [SurrealDB stability discussions](https://github.com/surrealdb/surrealdb/issues/5175)
