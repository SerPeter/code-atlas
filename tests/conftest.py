"""Shared test fixtures for Code Atlas."""

from __future__ import annotations

import pytest

from code_atlas.settings import AtlasSettings


@pytest.fixture
def settings(tmp_path):
    """Create test settings pointing to a temporary directory."""
    return AtlasSettings(project_root=tmp_path)
