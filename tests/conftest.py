"""Shared test fixtures for Code Atlas."""

from __future__ import annotations

import pytest

from code_atlas.graph import GraphClient
from code_atlas.settings import AtlasSettings


@pytest.fixture
def settings(tmp_path):
    """Create test settings pointing to a temporary directory."""
    return AtlasSettings(project_root=tmp_path)


@pytest.fixture
async def graph_client(settings):
    """Async GraphClient fixture — skips if Memgraph is unreachable.

    Wipes all data after each test for isolation.
    """
    client = GraphClient(settings)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Memgraph not available")

    # Clean slate before test
    await client.execute_write("MATCH (n) DETACH DELETE n")

    yield client

    # Clean up after test
    await client.execute_write("MATCH (n) DETACH DELETE n")
    await client.close()
