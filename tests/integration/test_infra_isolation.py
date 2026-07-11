"""Meta-tests: integration infra isolation and the production-data guard."""

from __future__ import annotations

import os

import pytest

from code_atlas.graph.client import GraphClient
from code_atlas.settings import AtlasSettings
from tests.integration.conftest import _GUARD_OK, _assert_disposable_db

pytestmark = pytest.mark.integration

_PORTS_OVERRIDDEN = "ATLAS_TEST_MEMGRAPH_PORT" in os.environ or "ATLAS_TEST_VALKEY_PORT" in os.environ


def test_default_endpoints_are_not_production_ports(_infra_endpoints):  # noqa: PT019 — value is used
    """Both resolution paths (env override, testcontainers default) must never yield production ports."""
    assert (_infra_endpoints.memgraph_port, _infra_endpoints.valkey_port) != (7687, 6379)


@pytest.mark.skipif(not _PORTS_OVERRIDDEN, reason="requires ATLAS_TEST_*_PORT env override")
def test_env_override_endpoints_respected(_infra_endpoints):  # noqa: PT019 — value is used
    """Explicit ATLAS_TEST_*_PORT values take precedence over testcontainers."""
    if "ATLAS_TEST_MEMGRAPH_PORT" in os.environ:
        assert _infra_endpoints.memgraph_port == int(os.environ["ATLAS_TEST_MEMGRAPH_PORT"])
    if "ATLAS_TEST_VALKEY_PORT" in os.environ:
        assert _infra_endpoints.valkey_port == int(os.environ["ATLAS_TEST_VALKEY_PORT"])


def test_atlas_env_exported_for_bare_settings(_infra_endpoints, tmp_path):  # noqa: PT019 — value is used
    assert os.environ["ATLAS_MEMGRAPH__PORT"] == str(_infra_endpoints.memgraph_port)
    assert os.environ["ATLAS_REDIS__PORT"] == str(_infra_endpoints.valkey_port)
    bare = AtlasSettings(project_root=tmp_path)
    assert bare.memgraph.port == _infra_endpoints.memgraph_port
    assert bare.redis.port == _infra_endpoints.valkey_port


async def test_guard_refuses_non_test_database(settings, monkeypatch):
    monkeypatch.delenv("ATLAS_TEST_DB", raising=False)
    client = GraphClient(settings)
    await client.execute_write(
        "CREATE (:Project {name: 'trading-bot', project_name: 'trading-bot', uid: 'guard-test:trading-bot'})"
    )
    _GUARD_OK.clear()
    try:
        with pytest.raises(pytest.exit.Exception):
            await _assert_disposable_db(client, settings.memgraph.host, settings.memgraph.port)
    finally:
        await client.execute_write("MATCH (p:Project {uid: 'guard-test:trading-bot'}) DETACH DELETE p")
        _GUARD_OK.clear()
        await client.close()


async def test_guard_allows_test_prefixed_data(settings, monkeypatch):
    monkeypatch.delenv("ATLAS_TEST_DB", raising=False)
    client = GraphClient(settings)
    await client.execute_write(
        "CREATE (:Project {name: 'test_guard_ok', project_name: 'test_guard_ok', uid: 'test_guard_ok:root'}), "
        "(:Module {project_name: 'test_guard_ok', uid: 'test_guard_ok:m'})"
    )
    _GUARD_OK.clear()
    try:
        await _assert_disposable_db(client, settings.memgraph.host, settings.memgraph.port)
    finally:
        await client.execute_write("MATCH (n) WHERE n.uid STARTS WITH 'test_guard_ok' DETACH DELETE n")
        _GUARD_OK.clear()
        await client.close()
