---
title: "ADR-0044: The coordination store follows the backend choice"
tags: [adr, backends, performance, embeddings]
kind: decision
---

# ADR-0044: The coordination store follows the backend choice

## Status

Accepted (2026-09-03). Amended 2026-09-17: the limiter is built once per process by the composition root and injected
(ADR-0038 decision 6); see the amended consequence below.

## Context

The embedded backend exists so Code Atlas runs with no Memgraph and no Valkey.
[ADR-0038](0038-backends-are-owned-by-a-composition-root.md) made that a composition-root decision: one place resolves
`backend.graph` and `backend.queue`, and the rest of the process is handed connections.

The rate limiter was not part of that. `EmbedClient` took a `RedisSettings` and, whenever it got one, built a
Valkey-backed `RateLimiter` — regardless of which backend the composition root had just resolved. An embedded deployment
therefore paced its embedding calls against a Valkey that was not there.

The limiter degraded gracefully, so this was invisible in behaviour. It was not invisible in cost. The `atlas bench`
write path measured **4.08 seconds of a 14 second index** inside the limiter, with the provider itself stubbed to return
instantly — 29% of the run spent opening sockets to an absent host. The cause was that `_degraded` suppressed the _log_
and not the _call_: one connect timeout per embed batch, paid again on every batch, for a fact the first call had
already established.

Two things could be wrong here and only one of them is the interesting one.

## Decision

**1. There is a SQLite rate limiter, and `backend.queue` selects it.**

`SqliteRateLimiter` holds the same two token buckets and the same AIMD scale factor in a SQLite file under
`sqlite_data_dir`, using WAL plus `BEGIN IMMEDIATE` where the Valkey side uses a Lua script's atomicity.
`create_rate_limiter(settings, connections)` in `code_atlas.backends` picks on `backend.queue`, the same key
`create_event_bus` uses, and builds the network limiters over the connections the bus uses. _(Amended 2026-09-17: this
was `make_rate_limiter`, called by each `EmbedClient`.)_

**2. Deleting the limiter on the embedded path was rejected.**

It was the smaller change and it was wrong. Pacing is not Valkey's feature; it is the pipeline's. An embedded deployment
calling a cloud embedding provider needs the buckets and the backoff exactly as much as a networked one — arguably more,
since it has no second process to notice.

**3. Cross-process, not in-process.**

The obvious cheap alternative — an in-memory limiter for the embedded path — reintroduces precisely the bug the buckets
were built to prevent. Two daemons on one machine would each get a private budget and issue double the configured rate,
and every test short of a two-process one would pass. SQLite in WAL mode with `BEGIN IMMEDIATE` serialises writers
across processes, which is the property being bought.

**4. `"auto"` resolves to Valkey, not to a guess.**

Probing here would reintroduce the connect timeout the selection exists to remove, and the bus factory has already
probed once at startup. Choosing SQLite for `"auto"` would be worse than the timeout: the process would pace against a
private file while the rest of the fleet paced against Valkey, and the shared budget would quietly stop being shared.
That failure is silent and permanent; a timeout is loud and bounded.

**5. A degraded limiter stops calling, not just stops logging.**

`_DEGRADED_RETRY_MS` (30s) gates the retry on both implementations. An unreachable store now costs a couple of timeouts
across a whole index instead of one per batch. Not a permanent disable — a restarted Valkey is picked up within a batch
or two — and not zero, because pacing genuinely is unavailable for that window and pretending otherwise would be the
same class of lie.

**6. The two implementations are pinned to each other by a conformance test.**

Lua cannot call Python, so the refill, the cost clamp and the AIMD ladder exist twice and will drift the moment one is
edited alone. `tests/integration/search/test_ratelimit_conformance.py` runs both against one sequence.

It cannot compare absolute `wait_ms`: the Lua reads `redis.call('TIME')` on purpose, because a container clock was once
measured drifting 16 seconds from this host. It compares what a caller experiences — which calls are admitted, which
block, how the scale factor moves — and every case carries a non-vacuity guard, because two limiters whose stores are
unreachable agree about everything.

## Consequences

- `EmbedClient(embed_settings, atlas_settings)` replaced `EmbedClient(embed_settings, redis_settings)`. Which store
  paces a client is a backend decision now, not a Redis detail. _(Amended 2026-09-17: now
  `EmbedClient(embed_settings, limiter=backends.limiter)`. The limiter is per process and keyed per call by the client's
  `RateBudget`; the limiter argument is required, and no pacing is `limiter=unpaced()`, spelled out.)_
- One more SQLite file (`ratelimit.sqlite3`) in `sqlite_data_dir`. Deliberately not `queue.sqlite3`: the limiter's
  writes are tiny and constant, and sharing a file with the hot event queue would put them behind its write lock for no
  gain.
- Both implementations must be edited together. The conformance test is the only thing enforcing that, and it needs
  Valkey to run — a unit-only run does not check it.
