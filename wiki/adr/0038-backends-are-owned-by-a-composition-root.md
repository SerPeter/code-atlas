# ADR-0038: Backends are owned by a composition root, and closing is not a caller's job

## Status

Accepted (2026-08-30)

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
2. **Every client is an async context manager.** `GraphClient`, `EventBus`, `SqliteGraphClient`, `SqliteEventBus`,
   `EmbedClient` and `RateLimiter` all carry `__aenter__`/`__aexit__` over their existing `close()`. A caller that holds
   one for a scope uses `async with`; a caller that holds one across several return paths registers it on an
   `AsyncExitStack`.
3. **Close what it opened, never what it was handed.** `use_backends` given a live client reuses it untouched. This is
   what lets the MCP server hand its graph to the daemon without either of them guessing who closes it.
4. **A `finally` is not a substitute.** The guard is the block, and the block starts at the constructor. Anything
   between construction and the guard is unguarded, which is where every leak in this codebase was.
5. **`ResourceWarning` is fatal.** This is the enforcement mechanism, not a preference. It is the only thing that
   distinguishes a client that is closed from one that merely looks closed. Exemptions are allowed but must be scoped to
   the class that needs them, never global: the only one is `TestUiInstances`, whose subject is a socket deliberately
   kept bound and handed to uvicorn. A global ignore cannot tell that from a real leak, which is exactly how the
   previous leaks survived.

Ownership is deliberately _not_ uniform where lifetimes differ: the CLI's embedding-dimension probe takes a block
because it is used once; the daemon closes the `EmbedClient` it constructed but never the bus it was handed; the MCP
lifespan closes whatever `AppContext` currently holds rather than registering on its stack, because a root switch
replaces that object and a stack would close the original twice and the replacement never.

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
