"""Unit test fixtures — no infrastructure required."""

from __future__ import annotations

import pytest
import pytest_socket
from tenacity import wait_none

from code_atlas.settings import AtlasSettings

#: Loopback only. Everything this package talks to for real -- a metered embedding
#: provider, a remote Memgraph -- is off-box, and that is what must never be reached.
_LOOPBACK = ["127.0.0.1", "::1", "localhost"]


@pytest.fixture(autouse=True)
def _no_off_box_network():
    """No unit test may connect to anything that is not loopback.

    The failure this prevents is not a slow test. `test_embeddings` intercepts litellm by
    *string* path, so the day that target moves the test would make a real, billed call
    against a live provider and still pass green. This turns that into a named traceback
    at the point of the call.

    Host allowlist rather than a blanket `disable_socket`, because a blanket one is
    unusable here: on Windows asyncio's ProactorEventLoop builds its self-pipe with a
    loopback `socketpair()`, so blocking socket *creation* fails every async test in the
    suite before it runs -- 725 errors when tried. `socket_allow_hosts` patches `connect`
    instead, which leaves the event loop and the UI's port binding alone and still blocks
    every off-box destination.

    Here rather than in addopts because integration and bench legitimately reach real
    services; they sit under a different conftest and are untouched.
    """
    pytest_socket.socket_allow_hosts(_LOOPBACK, allow_unix_socket=True)
    try:
        yield
    finally:
        pytest_socket.enable_socket()


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
