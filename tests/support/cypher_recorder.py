"""pytest plugin: record every distinct Cypher statement ``GraphClient`` sends, for a plan audit.

Opt-in, never loaded by default. Its output feeds ``scripts/audit_query_plans.py``, which
EXPLAINs each statement against a graph big enough for plans to mean something::

    CYPHER_RECORD_OUT=statements.jsonl uv run --no-sync python -m pytest \
        -p tests.support.cypher_recorder -m integration tests/integration
    uv run --no-sync python scripts/audit_query_plans.py statements.jsonl

``python -m pytest``, not ``pytest``: ``-p`` plugins load before the repository root is on
``sys.path``, and ``-m`` is what puts it there, so the bare form fails with "No module named
'tests'". The whole suite passes with it loaded: the wrappers are dunder-named, so
``telemetry.caller_name`` skips them and query metrics keep their real method label.

The integration suite is the statement source because it is the one run that reaches every
write and maintenance path -- the reclaim, the resolvers, the GC -- and those are where both
plan regressions of 2026-10 lived. A test graph cannot judge them: Memgraph plans by cost, so
on a few nodes a full scan is the right plan.

Writes one JSON object per distinct statement (whitespace-normalised): ``query``, one ``params``
sample with lists cut to three items, the ``caller`` (file:line:function inside the graph
package, ``?`` for a test's own Cypher) and how many times it was sent. Behaviour is unchanged:
each wrapper records and delegates.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

import pytest

_SEEN: dict[str, dict[str, Any]] = {}
_PASS_THROUGH = frozenset(
    {"execute", "execute_write", "_execute_write_inner", "_execute_write_with_retry", "wrapped_f"}
)


def _shrink(value: Any) -> Any:
    """Small but still plannable: every list keeps its first three items."""
    if isinstance(value, dict):
        return {k: _shrink(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shrink(v) for v in list(value)[:3]]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _caller() -> str:
    for frame in reversed(traceback.extract_stack()):
        path = frame.filename.replace("\\", "/")
        if "/code_atlas/" in path and frame.name not in _PASS_THROUGH and not frame.name.startswith("__"):
            return f"{Path(path).name}:{frame.lineno}:{frame.name}"
    return "?"


def _record(query: str, params: dict[str, Any] | None) -> None:
    key = " ".join(query.split())
    entry = _SEEN.get(key)
    if entry is None:
        sample = _shrink(params or {})
        try:
            json.dumps(sample)
        except TypeError:
            sample = {}
        entry: dict[str, Any] = {"query": query, "params": sample, "caller": _caller(), "count": 0}
        _SEEN[key] = entry
    entry["count"] += 1


def pytest_configure(config: pytest.Config) -> None:
    from code_atlas.graph.client import GraphClient

    for name in ("execute", "execute_write", "_execute_write_inner"):
        original = getattr(GraphClient, name)

        def wrap(orig: Any) -> Any:
            # Dunder-named on purpose: telemetry.caller_name skips such frames, so the metric
            # label stays the GraphClient method rather than this wrapper.
            async def __recorded(
                self: Any, query: str, params: dict[str, Any] | None = None, *a: Any, **kw: Any
            ) -> Any:
                _record(query, params)
                return await orig(self, query, params, *a, **kw)

            return __recorded

        setattr(GraphClient, name, wrap(original))


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("CYPHER_RECORD_OUT")
    if not out:
        return
    with Path(out).open("w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(entry) + "\n" for entry in _SEEN.values())
