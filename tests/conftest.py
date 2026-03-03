"""Shared test constants for Code Atlas."""

from __future__ import annotations

from code_atlas.settings import EmbeddingSettings

# ---------------------------------------------------------------------------
# Constants re-exported to test modules
# ---------------------------------------------------------------------------

TEST_DRAIN_TIMEOUT_S: float = 60.0
"""Shortened drain timeout for integration tests (default 600s is too long)."""

NO_EMBED = EmbeddingSettings(enabled=False)
"""Embedding settings that disable the embed stage entirely — use for pipeline tests
that don't need real or mocked embeddings."""
