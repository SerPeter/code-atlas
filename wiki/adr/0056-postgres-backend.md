---
title: "ADR-0056: Postgres as a third backend, queue first"
tags: [adr, backends, postgres, queue, settings]
kind: decision
---

# ADR-0056: Postgres as a third backend, queue first

## Status

Accepted (2026-09-17). Implements the **queue axis** only. The graph axis below is a recorded direction, not yet built.
Amended the same day: one pool and one listener per process, shared by the bus and the limiter; a schema version row.
Extends [ADR-0053](./0053-declaring-a-backend-is-selecting-it.md) (a third section on the queue axis) and
[ADR-0044](./0044-the-coordination-store-follows-the-backend.md) (a third rate-limiter mirror).

## Context

Code Atlas has two deployment shapes: Memgraph + Valkey, and embedded SQLite. The first is two extra services that most
organisations do not already run; the second is one machine's files. Postgres sits between them: nearly every team
already operates one, it is backed up and monitored, and one database can serve a whole fleet of projects and processes.

Postgres as a graph store was waiting on native SQL/PGQ, which was **reverted from PostgreSQL 19 on 2026-09-07**. The
available route for graph traversal is Apache AGE (1.8.0). So the backend lands one axis at a time, starting with the
one that needs no extension: the event queue, the indexer lease and the embedding rate limiter.

## Decision

`[backend.queue.postgres]` selects a Postgres queue. Like every backend since ADR-0053 it is chosen **only by declaring
it**: an undeclared queue (`auto`) still probes Valkey and falls back to SQLite, and never probes Postgres.

```toml
[backend.queue.postgres]
host = "localhost"
port = 5432
user = "atlas"
password = "..."        # SecretStr; never printed in labels or health output
database = "atlas"
```

The driver is **asyncpg**. psycopg 3's async mode refuses Windows' default `ProactorEventLoop`, which every entry point
(`asyncio.run` in the CLI, FastMCP, pytest-asyncio) runs on; asyncpg uses asyncio transports and works on it unchanged.

### Plain tables with row locks, not a queue extension

Everything lives in its own schema, `atlas_queue`, created idempotently under a transaction-scoped advisory lock (two
processes' `CREATE ... IF NOT EXISTS` can otherwise race on the catalog):

| table                        | role                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `messages`                   | `(project, topic, id)` payloads; `id` is an identity column                         |
| `groups`                     | per-group cursor `last_id`, the newest id ever handed to the group                  |
| `deliveries`                 | the pending-entries list; ack deletes the row; no FK to `messages`                  |
| `consumers`                  | registrations with `seen_at`, for idle time and pruning                             |
| `leases`                     | the indexer lease, one row per project                                              |
| `rate_bucket` / `rate_scale` | the embedding limiter's shared buckets and AIMD factor, keyed per model like Valkey |

**One database, many projects.** Every queue row is keyed by project, the way Valkey keys are prefixed by it.

**Valkey's semantics, not SQLite's, where the two mirrors differ.** A delivery outlives its message: after a trim or
`flush` it replays with empty fields and consumers ack it, as with `XTRIM`. Registrations are real, and `drop_consumer`
destroys the entries it held and reports how many, as `XGROUP DELCONSUMER` does. _(Amended 2026-09-17:)_ a read from a
group that does not exist raises `ConsumerGroupMissingError`, as Valkey answers `NOGROUP`, instead of an empty batch
that looks like an idle stream; a supervised consumer then restarts and re-creates its group.

**Claims lock the group row** (`SELECT ... FOR UPDATE`), then read the messages past the cursor in a fresh snapshot,
record the deliveries and advance the cursor in one transaction, so two consumers can never receive one message.
Reclaiming uses `FOR UPDATE SKIP LOCKED`.

### Ids are assigned in commit order

A cursor is only sound if nothing can commit behind it. With a bare identity column, publisher A draws id 10 and stalls;
publisher B draws 11 and commits; a reader advances the cursor to 11; A commits — and message 10 is never delivered to
anyone. Every publish therefore takes `pg_advisory_xact_lock` on `(project, topic)` before its insert draws ids, and
holds it until commit. `publish_many` is one statement: the lock, the ordered insert and the `pg_notify` are CTEs. A
test reproduces the skip with the lock disabled and passes with it on.

### Blocking reads wake on LISTEN/NOTIFY

`pg_notify` fires inside the publish transaction and is delivered at commit. Each process holds one dedicated listener
connection outside the pool _(amended: per process, in `PostgresConnections`, not per bus)_; a waiting reader holds no
pool connection, clears its wake event before each claim (so a commit racing the claim still wakes it), and falls back
to a bounded 1 s poll if the listener is gone.

### Every time from the server clock

Idle times, reclaim cutoffs, lease expiry and the rate limiter's `now` all come from `clock_timestamp()`, for the reason
the Valkey Lua reads `TIME`: a container clock was measured 16 s away from its host. The limiter reads the clock _after_
its row lock is granted, or a waiter computes refill from a moment its predecessor has already written past.

### The rate limiter is a third mirror

`PostgresRateLimiter` reuses the shared `bucket_decision` / `aimd_increase` / `aimd_penalty`, locks the model's
`rate_scale` row for the whole check-and-debit, and keeps the degraded-retry cooldown. It is shared across processes and
projects on one database, as the Valkey limiter is across one server. `test_ratelimit_conformance.py` runs all three
limiters through the same sequences.

### One pool per process _(amended 2026-09-17)_

The bus and the limiter each used to open a pool (10 and 4, the latter once per embedding client). The composition root
now builds one `PostgresConnections` — a pool of 16, sized for the largest default embed concurrency plus the bus's
concurrent statements, and one listener — and both borrow it (ADR-0038 decision 6). Every connection carries
`application_name` `code-atlas:<pid>`.

### Schema version _(amended 2026-09-17)_

`atlas_queue.meta` holds `schema_version`. A newer version than the code refuses to start; an older one runs the ordered
migrations; a database with tables and no row predates the row and is version 1.

## Direction for the graph axis (not implemented)

- **Traversal:** Apache AGE (openCypher over Postgres), keeping the Cypher the query layer already speaks.
- **Vectors:** pgvector.
- **BM25:** pg_search.

Known risk: AGE's Cypher `MERGE` degrades badly under bulk writes (apache/age#2177, apache/age#2198), and indexing is
bulk upsert by nature. The mitigation to evaluate before committing to it is writing directly into AGE's per-label
tables with SQL for the index path, keeping Cypher for reads.

## Consequences

- `lag` is exact on this backend and never `None`: it counts messages past the cursor that still exist. Its only callers
  (`orchestrator._wait_for_drain`, `DaemonManager.pending_event_counts`) read `None` as "not drained", and an exact
  count is never 0 while a readable message remains.
- Trimming to the retained budget (100 000 per stream, `RedisSettings`' default) is approximate: it runs once per tenth
  of the budget published, not on every publish.
- A third queue implementation is a third mirror to keep in step. The shared scenarios in
  `tests/integration/backends/test_queue_conformance.py` run against all three buses; Postgres-only behaviour is pinned
  in `test_postgres_queue.py`.
- Tests use a separate, lazy `postgres:18-alpine` fixture (`ATLAS_TEST_POSTGRES_PORT` overrides it; compose
  `postgres-test` on 5434). It only truncates in a database named `atlas_test*`.
