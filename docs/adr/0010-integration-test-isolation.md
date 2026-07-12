# ADR-0010: Integration Test Database Isolation

## Status

Accepted

## Date

2026-07-12

## Context

`tests/integration/conftest.py` resolved Memgraph/Valkey endpoints by probing fixed local ports
(7687/6379 — the same ports the production `docker compose up` stack uses) and, if a database was
found there, ran `MATCH (n) DETACH DELETE n` against it as fixture teardown/setup. On a developer
machine running the production stack for day-to-day indexed-search use, this silently connected
integration tests to the live index. It was live-diagnosed doing exactly this: a real project's
index (762 MiB on disk) was destroyed by a routine `pytest tests/integration` run, and a pytest
process was caught mid-run still connected to the production instance during the same investigation
that found the pipeline durability defects in [ADR-0009](./0009-event-pipeline-durability-contract.md).

No existing ADR covers test infrastructure at all — this is new territory, not an amendment.

## Decision

### Default: session-scoped testcontainers on random ports

Integration tests default to spinning up their own Memgraph + Valkey pair via `testcontainers`,
scoped to the pytest session (one pair shared across the whole run, torn down at session end — not
one container per test). Ports are chosen by Docker at container-start time, so there is no fixed
port to accidentally collide with a running production stack, and no port convention to keep in
sync. Memgraph 3.7's containers need exec-probe wait strategies (`echo 'RETURN 1;' | mgconsole`,
mirroring the compose healthcheck) rather than a log-line wait strategy — Memgraph 3.7 never emits a
stable "ready" log line, and accepts TCP connections before the Bolt protocol is actually queryable,
so a plain port-wait strategy produces a flaky "not available yet" failure on the first few tests of
a session.

### Override: explicit env vars select a fixed instance

`ATLAS_TEST_MEMGRAPH_PORT` / `ATLAS_TEST_VALKEY_PORT`, when set, bypass testcontainers entirely and
connect to whatever is listening on those ports. This exists for two cases: (1) a `docker compose
--profile test up -d` pair (`memgraph-test` on 7688, `valkey-test` on 6380 — tmpfs storage, no
persistence, WAL disabled) that a developer starts once and reuses across many local test runs to
skip the ~10s testcontainers startup cost per session; (2) CI environments that provision Memgraph/
Valkey as fixed-port service containers rather than via Docker-in-Docker. These env vars are
harness-only — `AtlasSettings` never reads them, so a developer's ordinary `ATLAS_MEMGRAPH__PORT`
shell export (pointed at production) cannot accidentally redirect a test run there, and conversely a
test-harness override cannot leak into production settings resolution.

### Backstop: refuse to wipe non-disposable data

Independent of how the endpoint was resolved, a session-scope guard inspects the target database
before any destructive fixture runs and aborts the whole test session (loud `pytest.exit`, not a
silent skip) if it finds `Project` nodes whose name isn't `test`/`bench`-prefixed. `ATLAS_TEST_DB=1`
bypasses the guard for instances known to be disposable (the testcontainers and compose-test paths
set it automatically; a developer pointing at some other instance must set it explicitly, which is
the point — it forces a conscious choice rather than an implicit one).

Production ports 7687/6379 are not probed by any code path — not as a fallback, not as a default,
not conditionally. The only way a test run touches those ports is a developer explicitly exporting
`ATLAS_TEST_MEMGRAPH_PORT=7687`, which the guard would then refuse unless `ATLAS_TEST_DB=1` is also
set — at which point that is an unambiguous, deliberate act, not an accident of shared defaults.

## Consequences

### Positive

- Zero-setup safety: a developer who has never heard of the `test` compose profile gets full
  isolation by default, with no port convention to remember or get wrong.
- Session-scoped (not per-test) containers preserve the ability to write multi-test flows against
  shared state — index in one test, search/update/delete across several more — which a
  per-test-container model would have made awkward.
- Parallel test sessions (e.g. multiple agents running integration tests concurrently, as happened
  during the fix effort that produced this ADR) no longer collide, since each session's
  testcontainers pair gets independent random ports.
- The guard is defense-in-depth, not the primary mechanism — it catches misconfiguration classes not
  yet imagined, without being the only thing standing between a test run and production data.

### Negative

- Testcontainers pays a real per-session startup cost (Memgraph image pull on first use, ~5-10s
  container start + health wait) that a long-lived fixed-port instance avoids — this is why the
  compose `test` profile override path exists as an opt-in fast path, not because testcontainers-only
  was rejected on safety grounds.
- Requires Docker Desktop (or equivalent) to be running for any integration test — this was already
  true before this ADR (fixed-port probing also assumed a local Memgraph), so not a new constraint,
  but testcontainers' reaper sidecar and named-pipe socket handling are a source of Windows-specific
  flakiness worth watching.

### Risks

- The guard's heuristic (non-`test`/`bench`-prefixed `Project` node ⇒ refuse) is itself a heuristic,
  not a proof — a project deliberately named `test-something` in a real, non-disposable graph would
  bypass it. Acceptable given the guard is a backstop behind testcontainers-by-default, not the sole
  protection.

## Alternatives Considered

### Testcontainers-only (no fixed-port override)

Random ports for every run, no compose `test` profile. Rejected — not on safety grounds (this option
is strictly safer, eliminating the port-convention risk entirely) but on iteration-speed grounds: a
tight edit/test loop against a single failing integration test pays the full container-startup cost
on every invocation, and Windows Docker Desktop's testcontainers reaper is the flakiest part of the
stack. The override path exists specifically so a developer can trade the marginal extra safety for
a faster local loop, while CI and the default path stay maximally safe.

### Per-test containers instead of session-scoped

A fresh Memgraph/Valkey pair for every individual test. Rejected: several existing integration tests
deliberately exercise a sequence (index → search → update → search again → delete) across what were
originally separate test functions relying on shared prior state; per-test containers would have
required either merging those into single mega-tests or restructuring fixture scope in a way that
provided no additional safety over session scope, for a large, unnecessary startup-cost multiplier.

## References

- [ADR-0009: Event Pipeline Durability Contract](./0009-event-pipeline-durability-contract.md) — the
  investigation that surfaced the production-database-wipe defect this ADR fixes
- `tests/integration/conftest.py` — `_infra_endpoints`, `_assert_disposable_db`
- `docker-compose.yml` — `memgraph-test`/`valkey-test` services (`profiles: [test]`)
- `CLAUDE.md` — Infrastructure and Testing sections (developer-facing instructions)
