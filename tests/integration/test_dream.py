"""Integration test for the deterministic dream-mode report (code_atlas.dream).

Requires Memgraph + Valkey (provided by conftest fixtures).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.dream import VaultRoot, build_dream_report
from code_atlas.indexing.orchestrator import index_project

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings

pytestmark = pytest.mark.integration


def _write(root: Path, rel_path: str, content: str) -> None:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


async def test_build_dream_report_end_to_end(
    settings: AtlasSettings,
    graph_client: GraphClient,
    event_bus: EventBus,
    tmp_path: Path,
) -> None:
    """A vault with an orphan, a dangling link, and a duplicate id is fully flagged."""
    settings.embeddings.enabled = False
    await graph_client.ensure_schema()

    vault_dir = tmp_path / "vault"
    # note-a links to note-b (neither is an orphan) and to a nonexistent note (dangling).
    # It also has an explicit anchors: reference to a nonexistent target (unresolved).
    _write(
        vault_dir,
        "note-a.md",
        "---\nid: note-a\nkind: note\nanchors: [ghost-anchor]\n---\n\n# A\n\nSee [[note-b]] and [[note-ghost]].\n",
    )
    _write(vault_dir, "note-b.md", "---\nid: note-b\nkind: note\n---\n\n# B\n\nNo outgoing links.\n")
    # note-c has no links at all — a true orphan.
    _write(vault_dir, "note-c.md", "---\nid: note-c\nkind: note\n---\n\n# C\n\nIsolated.\n")
    # Duplicate id: two files resolve to the same note uid.
    _write(vault_dir, "note-d.md", "---\nid: note-d\nkind: note\n---\n\n# D\n\nFirst copy.\n")
    _write(vault_dir, "sub/note-d-copy.md", "---\nid: note-d\nkind: note\n---\n\n# D copy\n\nSecond copy.\n")
    # An inbox draft.
    _write(vault_dir, "inbox/draft-one.md", "---\nid: draft-one\nkind: draft\n---\n\n# Draft\n\nJust a draft.\n")

    result = await index_project(
        settings, graph_client, event_bus, project_name="test-vault", project_root=vault_dir, drain_timeout_s=60.0
    )
    assert result.drained, "indexing the test vault did not drain"

    report = await build_dream_report(graph_client, [VaultRoot(path=vault_dir, project_name="test-vault")])

    orphan_uids = {n.uid for n in report.orphan_notes}
    assert "test-vault:note:note-c" in orphan_uids
    assert "test-vault:note:note-a" not in orphan_uids
    assert "test-vault:note:note-b" not in orphan_uids

    dangling_targets = {d.target_uid for d in report.dangling_links}
    assert "test-vault:note:note-ghost" in dangling_targets

    broken = next((b for b in report.broken_anchors if b.uid == "test-vault:note:note-a"), None)
    assert broken is not None
    assert broken.unresolved_anchors == ["ghost-anchor"]

    dup = next((d for d in report.duplicate_ids if d.qualified_name == "test-vault:note:note-d"), None)
    assert dup is not None
    assert sorted(dup.file_paths) == ["note-d.md", "sub/note-d-copy.md"]

    assert report.inbox_count == 1
    assert report.inbox_paths == ["inbox/draft-one.md"]

    # No embeddings were generated (embeddings disabled), so similarity is empty.
    assert report.similar_pairs == []
    assert report.promotion_candidates == []
