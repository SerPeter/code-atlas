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

## If it gets fixed

A narrow ignore keyed on the thread name would do it, and would keep every other thread exception fatal:

```toml
"ignore:Exception in thread Thread-\d+ \(_connection_worker_thread\):pytest.PytestUnhandledThreadExceptionWarning",
```

Worth checking first whether an aiosqlite connection is genuinely being abandoned somewhere — the narrow ignore silences
the messenger either way, and the reason `ResourceWarning` was ignored wholesale was that the objects were transport
internals below our clients. That argument has not been re-established for this one.
