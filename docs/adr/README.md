# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for Code Atlas.

## What is an ADR?

An ADR is a document that captures an important architectural decision made along with its context and consequences.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0000](./0000-template.md) | Template | - | - |
| [0001](./0001-memgraph-as-database.md) | Use Memgraph as the Graph Database | Accepted | 2025-02-07 |
| [0002](./0002-build-from-scratch.md) | Build From Scratch Rather Than Fork | Accepted | 2025-02-07 |
| [0003](./0003-python-rust-hybrid.md) | Python/Rust Hybrid Architecture | Accepted | 2025-02-07 |
| [0004](./0004-event-driven-tiered-pipeline.md) | Event-Driven Tiered Pipeline | Accepted | 2026-02-07 |
| [0005](./0005-deployment-process-model.md) | Deployment & Process Model | Accepted | 2026-02-07 |

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
