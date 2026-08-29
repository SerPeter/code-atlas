---
id: aiosqlite-thread-warning-flake
kind: draft
tags: [testing, flake, aiosqlite, warnings]
created: 2026-08-29
---

# The unit suite goes red ~1 run in 4, naming a different test each time

`filterwarnings = ["error"]` escalates `PytestUnhandledThreadExceptionWarning`, and pytest's `threadexception` plugin
raises that whenever a **background thread** throws. The thread is aiosqlite's `_connection_worker_thread`, and what it
throws is a `ResourceWarning` — the same category the config already ignores.

The existing `"ignore::ResourceWarning"` entry does not cover it. That filter is applied by pytest inside a per-item
`catch_warnings()` block on the **main** thread; a warning raised on aiosqlite's worker thread can land while the filter
list is not the one the test installed, so it escalates and the plugin re-reports it as a thread exception against
whichever test happens to be running.

Which is why it is never the same test twice. Observed on `test_analysis.py::test_communities_respect_path_scope`,
`test_cli.py::TestProjectRm::test_confirmation_prompt_aborts_on_no`, and
`test_mcp.py::TestSummarizeModule::test_missing_path_is_a_clean_error_not_a_query` — none of which reproduce alone. This
is the same misattribution trap recorded in the `filterwarnings` comment for `ResourceWarning`, arriving through a
second plugin.

## It is pre-existing

Verified rather than assumed: a detached worktree at `4160860` — before the context-manager conversion — reproduced it
on the second run, so moving the closes into fixture teardown did not cause it. Frequency is roughly 1 in 4 full unit
runs at `-n auto`; six consecutive runs of a three-file subset did not reproduce it, so it needs the whole suite's
concurrency.

## Checked: a connection really was being abandoned

The narrow ignore below would have buried a real leak, so it was not applied.

`test_mcp.py::test_hidden_on_sqlite_backend` built a `SqliteGraphClient` and handed it to `_stub_backends`, which yields
it without closing — correct, since the real `use_backends()` closes only what it opened. The test never closed it
either, so a live aiosqlite connection was abandoned on every run. Fixed in `d592eaf`.

That was the whole of it for unit: **2785 tests now pass under `-W error::ResourceWarning`, twice**, where both prior
runs failed deterministically. So the unit suite can carry that flag as a real guard.

## Integration: the drivers were test-side, the redis pools are not

Running `-m integration -W error::ResourceWarning` reported, by object:

| count | object                     | outcome                                       |
| ----- | -------------------------- | --------------------------------------------- |
| 6     | `_ProactorSocketTransport` | below our clients — the known unfixable class |
| 6     | `socket.socket`            | same                                          |
| 4     | `neo4j AsyncBoltDriver`    | **fixed** in `17349c9`                        |
| 2     | `redis.asyncio Connection` | **open** — see below                          |

The four Bolt drivers were three test sites that close in a `finally` but do their _setup_ before the `try`: two seeding
writes in `test_infra_isolation`, and in `test_relationship_coverage` a `pytest.skip()` on ping failure — the same leak
already fixed in the four infra fixtures — followed by `ensure_schema` and a whole `index_project` run, all outside the
guard. Those two files now pass 39 tests with zero unclosed objects.

This also corrects the `filterwarnings` comment's claim that the objects are only transport internals below our clients.
Two of the four categories were above that line.

### Still open: EmbedClient has no lifecycle at all

`EmbedClient` constructs a `RateLimiter` (`embeddings.py:101`), which holds a redis connection pool — and `EmbedClient`
has no `close()`, no `__aenter__`, no `__aexit__`. It is constructed at **eight** sites and closed at none:

- `cli.py:590` (the dimension probe), `cli.py:949`
- `indexing/daemon.py:164`
- `indexing/orchestrator.py:2030`, `:2313`
- `server/health.py:468`, `server/mcp.py:229`, `:822`

`RateLimiter` gained the protocol in `8b3f0de`, so the owner above it is what is missing. This is product code, not test
scaffolding — unlike everything else in this note — so it is left for a deliberate pass rather than folded into a test
cleanup.

## If it gets fixed

A narrow ignore keyed on the thread name would do it, and would keep every other thread exception fatal:

```toml
"ignore:Exception in thread Thread-\d+ \(_connection_worker_thread\):pytest.PytestUnhandledThreadExceptionWarning",
```

Worth checking first whether an aiosqlite connection is genuinely being abandoned somewhere — the narrow ignore silences
the messenger either way, and the reason `ResourceWarning` was ignored wholesale was that the objects were transport
internals below our clients. That argument has not been re-established for this one.
