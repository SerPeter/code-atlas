# ADR-0038: Backends are owned by a composition root, and closing is not a caller's job

## Status

Accepted (2026-08-30). Amended 2026-09-17: the composition root owns one connection per queue service, and the event bus
and rate limiter borrow it (decision 6). Decision 2 and the ownership paragraph below are updated to match.

## Context

Every entry point opened its own connections and closed them by hand. `cli.py` alone carried nineteen manual
`await graph.close()` / `await bus.close()` calls, each in a `try/finally` written slightly differently, and the MCP
server, the daemon and `health.py` each had their own variation.

Hand-written teardown is wrong in a specific, repeatable way: it guards the block it is attached to and nothing above
it. Eight commands opened a graph, pinged it, and `raise typer.Exit(1)` on failure — _before_ the `try/finally` that
would have closed it, so the failure path leaked the connection it had just opened. The same shape appeared in the four
integration fixtures, which called `pytest.skip()` between constructing a client and reaching their close, and in three
more test sites whose setup — a seeding write, an `ensure_schema`, a whole `index_project` run — sat outside the guard.

None of this was visible. `filterwarnings` ignored `ResourceWarning` wholesale, on the reasoning that the objects
reported were `socket.socket` and `_ProactorSocketTransport` — asyncio internals below our clients, whose teardown no
application `close()` controls on Windows' proactor loop. That reasoning was wrong, and being wrong is what kept it in
place: those objects were downstream of genuinely leaked clients.

## Decision

1. **One composition root.** `backends.use_backends()` opens the connections a process needs and closes them on the way
   out. `connected()` adds the reachability check eight commands were each writing by hand.
2. **Every client that owns a connection is an async context manager.** `GraphClient`, `SqliteGraphClient`,
   `SqliteEventBus`, `SqliteRateLimiter` and `PostgresConnections` carry `__aenter__`/`__aexit__` over `close()`, and so
   does the Valkey client the root builds. A caller that holds one for a scope uses `async with`; a caller that holds
   one across several return paths registers it on an `AsyncExitStack`. _(Amended 2026-09-17: `EventBus`,
   `PostgresEventBus`, `RateLimiter`, `PostgresRateLimiter` and `EmbedClient` own no connection any more and have no
   `close()` — see decision 6.)_
3. **Close what it opened, never what it was handed.** `use_backends` given a live client reuses it untouched. This is
   what lets the MCP server hand its graph to the daemon without either of them guessing who closes it.
4. **A `finally` is not a substitute.** The guard is the block, and the block starts at the constructor. Anything
   between construction and the guard is unguarded, which is where every leak in this codebase was.
5. **`ResourceWarning` is fatal.** This is the enforcement mechanism, not a preference. It is the only thing that
   distinguishes a client that is closed from one that merely looks closed. Exemptions are allowed but must be scoped to
   the class that needs them, never global: the only one is `TestUiInstances`, whose subject is a socket deliberately
   kept bound and handed to uvicorn. A global ignore cannot tell that from a real leak, which is exactly how the
   previous leaks survived.
6. **One connection per service, borrowed by the bus and the limiter** _(added 2026-09-17)_. `use_backends` (and
   `use_queue`, its queue half, for `atlas index`, which needs the queue before the graph) builds `QueueConnections`: at
   most one Valkey client, or one asyncpg pool plus one LISTEN connection. `create_event_bus` and `create_rate_limiter`
   hand those to the bus and to `Backends.limiter`, which never create or close a connection. Every `EmbedClient` used
   to build its own limiter and every limiter its own client or pool, so a process's connection count followed the
   number of embedding clients rather than the work. `EmbedClient` now receives the limiter as a required argument and
   keeps only a `RateBudget` of its own (model, rpm/tpm, concurrency gate); a caller that must not be paced passes
   `unpaced()` by name, as the health probe does.

Ownership is deliberately _not_ uniform where lifetimes differ. _(Amended 2026-09-17: the embedding client no longer
holds a connection, so the dimension probe, the daemon and the MCP root switch simply drop it; the limiter it paced
through stays with the backends, like the bus.)_ The daemon never closes the bus or the limiter it was handed.

## Consequences

`cli.py` went from nineteen manual closes to zero. The test suite went from 149 hand-written closes to 20, and the
twenty that remain are deliberate — clients under test, a simulated restart, and the `__aexit__` contract tests
themselves, which need an explicit close to have anything to assert about.

Making `ResourceWarning` fatal found what the ignore had hidden: an abandoned aiosqlite connection in the MCP tests,
eight `EmbedClient` construction sites with no lifecycle at all, and four Bolt drivers a test had made unclosable with a
class-level `close` mock. With those fixed the suite reports no unclosed objects of any kind.

The cost is that a future leak fails CI rather than passing quietly, and `ResourceWarning` is garbage-collection timed,
so it may name a test that is not the culprit. That is a feature here — it is how the aiosqlite one was found — but it
needs the right debugging reflex: read the _objects_ reported, never the test names. Re-adding `ignore::ResourceWarning`
silences the messenger and is how the previous leaks survived.

`pytest-socket` is a related but separate guard: it stops unit tests reaching off-box, where this stops any test
abandoning a connection it opened.
