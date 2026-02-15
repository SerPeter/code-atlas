"""Unit test fixtures — no infrastructure required."""

from __future__ import annotations

import pytest

from code_atlas.settings import AtlasSettings


@pytest.fixture
def settings(tmp_path):
    """Minimal AtlasSettings for unit tests (no Memgraph/Valkey)."""
    return AtlasSettings(project_root=tmp_path)
