"""Unit tests for repository analysis module (mocked graph client — no infrastructure needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from code_atlas.server.analysis import analyze_repo

# ---------------------------------------------------------------------------
# Dependencies: cross-package coupling
# ---------------------------------------------------------------------------


def _graph_with_imports(
    direct: list[dict[str, str]],
    indirect: list[dict[str, str]] | None = None,
) -> MagicMock:
    """Fake GraphClient whose execute() returns the four _analyze_dependencies result sets in call order."""
    graph = MagicMock()
    graph.execute = AsyncMock(side_effect=[direct, indirect or [], [], []])
    return graph


async def test_cross_package_coupling_uses_parent_package():
    """Coupling must group by parent package, not the shared top-level segment.

    Module qualified names are import-system dotted paths (post-S2 namespace,
    e.g. 'code_atlas.indexing.consumers'), so the first segment is identical
    for every internal module and deriving 'package' from it filters out all
    real package-to-package coupling.
    """
    graph = _graph_with_imports(
        direct=[
            {"from_mod": "code_atlas.indexing.consumers", "to_mod": "code_atlas.graph.client"},
            {"from_mod": "code_atlas.indexing.orchestrator", "to_mod": "code_atlas.graph.client"},
            {"from_mod": "code_atlas.search.engine", "to_mod": "code_atlas.graph.client"},
        ],
        indirect=[
            {"from_mod": "code_atlas.search.engine", "to_mod": "code_atlas.graph.client"},
        ],
    )

    result = await analyze_repo(graph, "dependencies", "code-atlas")

    coupling = {(e["from"], e["to"]): e["weight"] for e in result["cross_package_coupling"]}
    assert coupling == {
        ("code_atlas.indexing", "code_atlas.graph"): 2,
        ("code_atlas.search", "code_atlas.graph"): 2,
    }


async def test_cross_package_coupling_excludes_intra_package_imports():
    graph = _graph_with_imports(
        direct=[
            {"from_mod": "code_atlas.indexing.consumers", "to_mod": "code_atlas.indexing.watcher"},
            {"from_mod": "code_atlas.indexing.daemon", "to_mod": "code_atlas.indexing.watcher"},
        ],
    )

    result = await analyze_repo(graph, "dependencies", "code-atlas")

    assert result["cross_package_coupling"] == []
    # The module-level edges themselves are still reported
    assert len(result["internal_imports"]) == 2
