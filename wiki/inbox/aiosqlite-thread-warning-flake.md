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

## Integration, measured by object across three runs

| object                     | before | after `17349c9` | after `152c13a` |
| -------------------------- | ------ | --------------- | --------------- |
| `_ProactorSocketTransport` | 6      | —               | 4               |
| `socket.socket`            | 6      | —               | 4               |
| `neo4j AsyncBoltDriver`    | 4      | —               | **4**           |
| `redis.asyncio Connection` | 2      | —               | **0**           |

**The EmbedClient fix removed the redis pools.** It had no `close()`, no `__aenter__` and no `__aexit__` while owning a
`RateLimiter` that holds a pool, at eight construction sites. Fixed in `152c13a`; the two unclosed Connections are gone.

**The driver count did not move, and `17349c9`'s commit message implies it should have.** That message reads as though
the three test sites it fixed were the four observed drivers. They were not. Those sites leak only when the setup
_between the constructor and the `try`_ raises — a seeding write, a `pytest.skip` on an unreachable Memgraph, a failing
`index_project`. In a green run none of that raises, so they never leaked and fixing them removed nothing observable.
The fix is still right, because the bug is real the moment anything there fails; it is preventive, not corrective.

So four unclosed `AsyncBoltDriver`s remain and their source is still unfound. They surface against the
`test_git_signals` CLI tests, but that attribution is GC timing and means little. Every `GraphClient` in the tree is
accounted for: `cli.py:605` inside its stack, `use_backends` which closes what it opened, and test sites that are all
now scoped or in a `finally`. Finding it needs `-X tracemalloc` on integration, which is too slow to run casually — one
attempt reached two tests in eight minutes.

## If it gets fixed

A narrow ignore keyed on the thread name would do it, and would keep every other thread exception fatal:

```toml
"ignore:Exception in thread Thread-\d+ \(_connection_worker_thread\):pytest.PytestUnhandledThreadExceptionWarning",
```

Worth checking first whether an aiosqlite connection is genuinely being abandoned somewhere — the narrow ignore silences
the messenger either way, and the reason `ResourceWarning` was ignored wholesale was that the objects were transport
internals below our clients. That argument has not been re-established for this one.
