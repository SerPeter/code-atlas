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
| [0008](./0008-cross-file-relationship-resolution.md) | Cross-File Relationship Resolution & Qualified-Name Extensions | Accepted               | 2026-07-12 |
| [0009](./0009-event-pipeline-durability-contract.md) | Event Pipeline Durability Contract                             | Accepted               | 2026-07-12 |
| [0010](./0010-integration-test-isolation.md)         | Integration Test Database Isolation                            | Accepted               | 2026-07-12 |

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
