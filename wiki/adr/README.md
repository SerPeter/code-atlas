# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for Code Atlas.

## What is an ADR?

An ADR is a document that captures an important architectural decision made along with its context and consequences.

## Index

| ADR                                                  | Title                                                          | Status                 | Date       |
| ---------------------------------------------------- | -------------------------------------------------------------- | ---------------------- | ---------- |
| [0000](./0000-template.md)                           | Template                                                       | -                      | -          |
| [0001](./0001-memgraph-as-database.md)               | Use Memgraph as the Graph Database                             | Accepted               | 2025-02-07 |
| [0002](./0002-build-from-scratch.md)                 | Build From Scratch Rather Than Fork                            | Accepted               | 2025-02-07 |
| [0003](./0003-python-rust-hybrid.md)                 | Python/Rust Hybrid Architecture                                | Superseded by ADR-0006 | 2025-02-07 |
| [0004](./0004-event-driven-tiered-pipeline.md)       | Event-Driven Tiered Pipeline                                   | Amended by ADR-0009    | 2026-02-07 |
| [0005](./0005-deployment-process-model.md)           | Deployment & Process Model                                     | Accepted               | 2026-02-07 |
| [0006](./0006-pure-python-tree-sitter.md)            | Pure Python with In-Process Tree-sitter                        | Accepted               | 2026-02-08 |
| [0007](./0007-qualified-name-strategy.md)            | Qualified Name Resolution Strategy                             | Amended by ADR-0008    | 2026-02-08 |
| [0008](./0008-cross-file-relationship-resolution.md) | Cross-File Relationship Resolution & Qualified-Name Extensions | Amended by ADR-0014    | 2026-07-12 |
| [0009](./0009-event-pipeline-durability-contract.md) | Event Pipeline Durability Contract                             | Accepted               | 2026-07-12 |
| [0010](./0010-integration-test-isolation.md)         | Integration Test Database Isolation                            | Accepted               | 2026-07-12 |
| [0011](./0011-note-vault-schema.md)                  | Note Label and the Knowledge Vault Schema                      | Amended by ADR-0012    | 2026-07-13 |
| [0012](./0012-rename-vault-to-wiki.md)               | Rename the Default Knowledge Vault Directory to wiki/          | Accepted               | 2026-07-17 |
| [0013](./0013-mcp-tool-taxonomy.md)                  | MCP Tool Taxonomy — Static Analysis vs. Information Retrieval  | Accepted               | 2026-07-17 |
| [0014](./0014-calls-edge-confidence.md)              | CALLS Edge Confidence                                          | Accepted               | 2026-07-17 |
| [0015](./0015-embedded-backend-option.md)            | Embedded Backend Option (SQLite Graph + Queue)                 | Accepted               | 2026-07-18 |
| [0016](./0016-consistent-test-entity-filtering.md)   | Consistent Test-Entity Filtering Across analyze_repo           | Accepted               | 2026-07-19 |

## Creating a New ADR

1. Copy `0000-template.md` to `NNNN-short-title.md`
2. Fill in the template
3. Update this README index
4. Submit for review

## Status Lifecycle

- **Proposed**: Under discussion
- **Accepted**: Decision made and implemented
- **Deprecated**: No longer applies (superseded or obsolete)
- **Superseded**: Replaced by a newer ADR

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [MADR Template](https://adr.github.io/madr/)
