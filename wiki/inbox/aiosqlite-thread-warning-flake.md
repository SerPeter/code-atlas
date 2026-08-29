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

## Integration is not clean, and not only for the reason recorded

Running `-m integration -W error::ResourceWarning` gives 6 failures and 3 errors. The objects, which are what to read:

| count | object                     | verdict                                               |
| ----- | -------------------------- | ----------------------------------------------------- |
| 6     | `_ProactorSocketTransport` | below our clients — the known unfixable class         |
| 6     | `socket.socket`            | same                                                  |
| 4     | `neo4j AsyncBoltDriver`    | **our layer** — a GraphClient somewhere is not closed |
| 2     | `redis.asyncio Connection` | **our layer** — same for an EventBus                  |

The `filterwarnings` comment says the objects are transport internals. That is now demonstrably incomplete: four
unclosed Bolt drivers and two unclosed redis connections are above that line. They are not in test scaffolding — every
`GraphClient(`/`EventBus(` in `tests/integration/` closes in a `finally` — so they are in the product paths those tests
drive (the CLI commands in `test_git_signals`, the orchestrator drain, the MCP server). Unchased.

## If it gets fixed

A narrow ignore keyed on the thread name would do it, and would keep every other thread exception fatal:

```toml
"ignore:Exception in thread Thread-\d+ \(_connection_worker_thread\):pytest.PytestUnhandledThreadExceptionWarning",
```

Worth checking first whether an aiosqlite connection is genuinely being abandoned somewhere — the narrow ignore silences
the messenger either way, and the reason `ResourceWarning` was ignored wholesale was that the objects were transport
internals below our clients. That argument has not been re-established for this one.
