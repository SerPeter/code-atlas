"""Unit test fixtures — no infrastructure required."""

from __future__ import annotations

import pytest
from tenacity import wait_none

from code_atlas.settings import AtlasSettings


@pytest.fixture
def settings(tmp_path):
    """Minimal AtlasSettings for unit tests (no Memgraph/Valkey)."""
    return AtlasSettings(project_root=tmp_path)


@pytest.fixture
def no_retry_backoff(monkeypatch):
    """Drop a tenacity-decorated callable's backoff to zero, keeping its policy.

    Patching the *wait* rather than the stop condition is deliberate: the test still
    exercises the real attempt count and the real retry predicate, which is what the
    retry is for. Only the sleeping goes.

    Usage: ``no_retry_backoff(EmbedClient._embed_call)``. The ``.retry`` attribute is
    attached by tenacity's decorator at runtime, so type checkers cannot see it.

    Shared because the pattern is not obvious and was already written once by hand in
    tests/unit/graph/test_client.py; a second hand-rolled copy is how the two drift.
    """

    def _apply(func) -> None:
        monkeypatch.setattr(func.retry, "wait", wait_none())

    return _apply
