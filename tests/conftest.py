"""Shared test fixtures for Code Atlas."""

from __future__ import annotations

import contextlib

import pytest

from code_atlas.graph.client import GraphClient
from code_atlas.schema import generate_drop_text_index_ddl, generate_drop_vector_index_ddl
from code_atlas.settings import AtlasSettings


@pytest.fixture
def settings(tmp_path):
    """Create test settings pointing to a temporary directory."""
    return AtlasSettings(project_root=tmp_path)


@pytest.fixture
async def graph_client(settings):
    """Async GraphClient fixture — skips if Memgraph is unreachable.

    Wipes all data and search indices before each test for isolation.
    Vector/text indices are dropped so ensure_schema can recreate them
    at the dimension specified by the test settings.
    """
    client = GraphClient(settings)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Memgraph not available")

    # Clean slate: wipe nodes and drop search indices (dimension may differ)
    await client.execute_write("MATCH (n) DETACH DELETE n")
    for stmt in generate_drop_vector_index_ddl():
        with contextlib.suppress(Exception):
            await client.execute_write(stmt)
    for stmt in generate_drop_text_index_ddl():
        with contextlib.suppress(Exception):
            await client.execute_write(stmt)

    yield client

    # Clean up after test
    await client.execute_write("MATCH (n) DETACH DELETE n")
    await client.close()
