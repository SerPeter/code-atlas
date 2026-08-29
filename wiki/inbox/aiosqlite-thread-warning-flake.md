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

## Resolved

All of it. The flake was a symptom, never a plumbing problem.

| leak                                                              | fix       |
| ----------------------------------------------------------------- | --------- |
| aiosqlite connection abandoned by `test_hidden_on_sqlite_backend` | `d592eaf` |
| 8 `EmbedClient` sites with no lifecycle at all                    | `152c13a` |
| 4 Bolt drivers held unclosable by a class-level `close` mock      | `8753e43` |

The last one is the interesting one. Two `test_git_signals` tests tried to hand the CLI the fixture's connection with
`monkeypatch.setattr("code_atlas.graph.client.GraphClient", ...)`, but `backends/__init__.py` binds `GraphClient` at
import, so the patch missed and a second real driver was built anyway -- the exact thing it existed to prevent. The
accompanying `monkeypatch.setattr(GraphClient, "close", AsyncMock())` then made that driver, and the fixture's,
impossible to close. Two tests, two drivers each.

The sockets and `_ProactorSocketTransport`s were downstream of those drivers, not the unfixable asyncio internals the
`filterwarnings` comment claimed. With the drivers fixed they disappeared too, so `ignore::ResourceWarning` came out in
`fdbc22a` and the warning is fatal again.

**Eight consecutive clean unit runs** against a rate that was roughly one in four, plus integration clean with the
warning fatal.

### How to find the next one

Not tracemalloc -- it never finished on the integration suite, reaching two tests in eight minutes. Wrap the factory and
hang a `weakref.finalize` on each object, checking a `closed` flag the wrapper sets. A first attempt that looked for
objects _still alive and unclosed at session end_ reported zero and proved nothing: a leaked object is collected
mid-run, and that is exactly when it warns, so the check skipped the entire population it was hunting.
