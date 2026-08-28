# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for Code Atlas.

## What is an ADR?

An ADR is a document that captures an important architectural decision made along with its context and consequences.

## Index

| ADR                                                               | Title                                                          | Status                 | Date       |
| ----------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------- | ---------- |
| [0000](./0000-template.md)                                        | Template                                                       | -                      | -          |
| [0001](./0001-memgraph-as-database.md)                            | Use Memgraph as the Graph Database                             | Accepted               | 2025-02-07 |
| [0002](./0002-build-from-scratch.md)                              | Build From Scratch Rather Than Fork                            | Accepted               | 2025-02-07 |
| [0003](./0003-python-rust-hybrid.md)                              | Python/Rust Hybrid Architecture                                | Superseded by ADR-0006 | 2025-02-07 |
| [0004](./0004-event-driven-tiered-pipeline.md)                    | Event-Driven Tiered Pipeline                                   | Amended by ADR-0009    | 2026-02-07 |
| [0005](./0005-deployment-process-model.md)                        | Deployment & Process Model                                     | Accepted               | 2026-02-07 |
| [0006](./0006-pure-python-tree-sitter.md)                         | Pure Python with In-Process Tree-sitter                        | Accepted               | 2026-02-08 |
| [0007](./0007-qualified-name-strategy.md)                         | Qualified Name Resolution Strategy                             | Amended by ADR-0008    | 2026-02-08 |
| [0008](./0008-cross-file-relationship-resolution.md)              | Cross-File Relationship Resolution & Qualified-Name Extensions | Amended by ADR-0014    | 2026-07-12 |
| [0009](./0009-event-pipeline-durability-contract.md)              | Event Pipeline Durability Contract                             | Accepted               | 2026-07-12 |
| [0010](./0010-integration-test-isolation.md)                      | Integration Test Database Isolation                            | Accepted               | 2026-07-12 |
| [0011](./0011-note-vault-schema.md)                               | Note Label and the Knowledge Vault Schema                      | Amended by ADR-0012    | 2026-07-13 |
| [0012](./0012-rename-vault-to-wiki.md)                            | Rename the Default Knowledge Vault Directory to wiki/          | Accepted               | 2026-07-17 |
| [0013](./0013-mcp-tool-taxonomy.md)                               | MCP Tool Taxonomy — Static Analysis vs. Information Retrieval  | Accepted               | 2026-07-17 |
| [0014](./0014-calls-edge-confidence.md)                           | CALLS Edge Confidence                                          | Accepted               | 2026-07-17 |
| [0015](./0015-embedded-backend-option.md)                         | Embedded Backend Option (SQLite Graph + Queue)                 | Accepted               | 2026-07-18 |
| [0016](./0016-consistent-test-entity-filtering.md)                | Consistent Test-Entity Filtering Across analyze_repo           | Accepted               | 2026-07-19 |
| [0017](./0017-calls-edge-weights.md)                              | CALLS Edge Weights and Test Provenance                         | Amended by ADR-0019    | 2026-07-30 |
| [0018](./0018-non-code-file-parsing.md)                           | Parsing Non-Code Files                                         | Accepted               | 2026-07-30 |
| [0019](./0019-module-granularity-community-detection.md)          | Community Detection at Module Granularity                      | Accepted               | 2026-07-31 |
| [0020](./0020-referenced-runtime-surface-nodes.md)                | Env-Var and Referenced-File Nodes, and When a Node Is Global   | Accepted               | 2026-07-31 |
| [0021](./0021-module-summary-outline-format.md)                   | The module_summary Outline Format                              | Accepted               | 2026-07-18 |
| [0022](./0022-call-resolution-requires-a-grounded-receiver.md)    | Call Resolution Requires a Grounded Receiver                   | Extended by ADR-0023   | 2026-08-01 |
| [0023](./0023-type-directed-call-resolution.md)                   | Type-Directed Call Resolution                                  | Accepted               | 2026-08-02 |
| [0024](./0024-memgraph-312-for-vector-index-gc.md)                | Upgrade to Memgraph 3.12.0 (Vector-Index GC Segfault)          | Accepted               | 2026-08-02 |
| [0025](./0025-structural-protocol-conformance.md)                 | Structural Protocol Conformance by Method-Set Containment      | Accepted               | 2026-08-03 |
| [0026](./0026-resolution-is-replayed-not-batch-final.md)          | Resolution Is Replayed, Not Batch-Final                        | Accepted               | 2026-08-03 |
| [0027](./0027-lexical-strategies-need-a-grounded-receiver.md)     | The Lexical Strategies Need a Grounded Receiver Too            | Accepted               | 2026-08-04 |
| [0028](./0028-every-resolved-edge-states-its-evidence.md)         | Every Resolved Edge States Its Evidence                        | Accepted               | 2026-08-04 |
| [0029](./0029-blast-radius-traverses-dependency-not-execution.md) | blast_radius Traverses Dependency, Not Execution               | Accepted               | 2026-08-04 |
| [0030](./0030-candidate-hygiene-is-asymmetric.md)                 | Candidate Hygiene Is Asymmetric                                | Accepted               | 2026-08-05 |
| [0031](./0031-anonymous-callables-attribute-upward.md)            | Anonymous Callables Attribute Their Calls Upward               | Accepted               | 2026-08-05 |
| [0032](./0032-a-uid-must-identify-exactly-one-definition.md)      | A uid Must Identify Exactly One Definition                     | Accepted               | 2026-08-06 |
| [0033](./0033-graph-renderer-must-be-permissively-licensed.md)    | The Graph Renderer Must Be Permissively Licensed               | Accepted               | 2026-08-08 |
| [0034](./0034-architecture-snapshots-live-on-the-project-node.md) | Architecture Snapshots Live On The Project Node                | Accepted               | 2026-08-08 |
| [0035](./0035-embedding-model-is-per-project-dimension-is-not.md) | Embedding Model Is Per Project, Dimension Is Not               | Accepted               | 2026-08-28 |

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
