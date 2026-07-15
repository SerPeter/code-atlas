"""Unit tests for the deterministic dream-mode report (code_atlas.dream)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.dream import (
    BrokenAnchor,
    DanglingLink,
    DreamReport,
    DuplicateIdConflict,
    OrphanNote,
    SimilarPair,
    VaultRoot,
    _check_memory_index,
    _cosine_similarity,
    _find_broken_anchors,
    _find_similar_pairs,
    _scan_vault_for_notes,
    render_home_md,
    report_to_dict,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(root: Path, rel_path: str, content: str) -> None:
    full = root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical_vectors_is_one() -> None:
    assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero() -> None:
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# _scan_vault_for_notes
# ---------------------------------------------------------------------------


def test_scan_vault_missing_root_returns_empty(tmp_path: Path) -> None:
    by_qn, links = _scan_vault_for_notes(VaultRoot(path=tmp_path / "missing", project_name="p"))
    assert by_qn == {}
    assert links == []


def test_scan_vault_detects_duplicate_ids(tmp_path: Path) -> None:
    _write(tmp_path, "a.md", "---\nid: dup\nkind: note\n---\n\n# A\n\nbody\n")
    _write(tmp_path, "sub/b.md", "---\nid: dup\nkind: note\n---\n\n# B\n\nbody\n")
    _write(tmp_path, "unique.md", "---\nid: unique\nkind: note\n---\n\n# U\n\nbody\n")

    by_qn, _links = _scan_vault_for_notes(VaultRoot(path=tmp_path, project_name="proj"))

    assert sorted(by_qn["proj:note:dup"]) == ["a.md", "sub/b.md"]
    assert by_qn["proj:note:unique"] == ["unique.md"]


def test_scan_vault_extracts_link_targets(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "a.md",
        "---\nid: a\nkind: note\nderived_from: [b]\n---\n\n# A\n\nSee [[b]] for context.\n",
    )

    _by_qn, links = _scan_vault_for_notes(VaultRoot(path=tmp_path, project_name="proj"))

    targets = {(from_uid, rel_type, to_uid) for from_uid, rel_type, to_uid in links}
    assert ("proj:note:a", "LINKS_TO", "proj:note:b") in targets
    assert ("proj:note:a", "DERIVED_FROM", "proj:note:b") in targets


def test_scan_vault_ignores_non_note_markdown(tmp_path: Path) -> None:
    _write(tmp_path, "plain.md", "# Just a heading\n\nNo frontmatter here.\n")

    by_qn, links = _scan_vault_for_notes(VaultRoot(path=tmp_path, project_name="proj"))

    assert by_qn == {}
    assert links == []


# ---------------------------------------------------------------------------
# _check_memory_index
# ---------------------------------------------------------------------------


def test_check_memory_index_no_memory_md_is_silent(tmp_path: Path) -> None:
    assert _check_memory_index(VaultRoot(path=tmp_path, project_name="p")) == []


def test_check_memory_index_flags_missing_and_unlisted(tmp_path: Path) -> None:
    _write(tmp_path, "MEMORY.md", "- [Listed](listed.md) — exists\n- [Ghost](ghost.md) — does not exist\n")
    _write(tmp_path, "listed.md", "content\n")
    _write(tmp_path, "unlisted.md", "content\n")

    issues = _check_memory_index(VaultRoot(path=tmp_path, project_name="memvault"))

    missing_issue = next(i for i in issues if "references missing" in i)
    unlisted_issue = next(i for i in issues if "not listed" in i)
    assert "ghost.md" in missing_issue
    assert "listed.md" not in missing_issue
    assert "unlisted.md" in unlisted_issue


# ---------------------------------------------------------------------------
# _find_similar_pairs (fake graph — pure computation, no Memgraph needed)
# ---------------------------------------------------------------------------


class _FakeGraph:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def execute(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._rows


async def test_find_similar_pairs_respects_threshold() -> None:
    graph = _FakeGraph(
        [
            {"uid": "p:note:a", "project_name": "p", "embedding": [1.0, 0.0]},
            {"uid": "p:note:b", "project_name": "p", "embedding": [1.0, 0.0]},
            {"uid": "p:note:c", "project_name": "p", "embedding": [0.0, 1.0]},
        ]
    )

    pairs = await _find_similar_pairs(graph, threshold=0.9)  # type: ignore[arg-type]

    assert len(pairs) == 1
    assert {pairs[0].uid_a, pairs[0].uid_b} == {"p:note:a", "p:note:b"}
    assert pairs[0].similarity == pytest.approx(1.0)


async def test_find_similar_pairs_empty_when_no_rows() -> None:
    graph = _FakeGraph([])
    pairs = await _find_similar_pairs(graph, threshold=0.9)  # type: ignore[arg-type]
    assert pairs == []


# ---------------------------------------------------------------------------
# _find_broken_anchors (fake graph — query-row-to-dataclass mapping)
# ---------------------------------------------------------------------------


async def test_find_broken_anchors_maps_rows_to_dataclass() -> None:
    graph = _FakeGraph(
        [
            {
                "uid": "p:note:a",
                "name": "A",
                "project_name": "p",
                "file_path": "a.md",
                "unresolved_anchors": ["missing-target"],
            },
        ]
    )

    broken = await _find_broken_anchors(graph)  # type: ignore[arg-type]

    assert broken == [
        BrokenAnchor(
            uid="p:note:a", name="A", project_name="p", file_path="a.md", unresolved_anchors=["missing-target"]
        )
    ]


async def test_find_broken_anchors_defaults_null_unresolved_to_empty_list() -> None:
    """A note with has_broken_anchors=true (deleted target) but no unresolved_anchors list."""
    graph = _FakeGraph(
        [
            {
                "uid": "p:note:b",
                "name": "B",
                "project_name": "p",
                "file_path": "b.md",
                "unresolved_anchors": None,
            },
        ]
    )

    broken = await _find_broken_anchors(graph)  # type: ignore[arg-type]

    assert broken == [BrokenAnchor(uid="p:note:b", name="B", project_name="p", file_path="b.md", unresolved_anchors=[])]


async def test_find_broken_anchors_empty_when_no_rows() -> None:
    graph = _FakeGraph([])
    broken = await _find_broken_anchors(graph)  # type: ignore[arg-type]
    assert broken == []


# ---------------------------------------------------------------------------
# render_home_md / report_to_dict
# ---------------------------------------------------------------------------


def _sample_report() -> DreamReport:
    return DreamReport(
        inbox_count=1,
        inbox_paths=["docs/inbox/draft.md"],
        orphan_notes=[OrphanNote(uid="p:note:o", name="O", project_name="p", file_path="o.md")],
        duplicate_ids=[DuplicateIdConflict(qualified_name="p:note:dup", project_name="p", file_paths=["a.md", "b.md"])],
        dangling_links=[DanglingLink(from_uid="p:note:a", rel_type="LINKS_TO", target_uid="p:note:missing")],
        similar_pairs=[SimilarPair(uid_a="p:note:a", uid_b="p:note:b", project_a="p", project_b="p", similarity=0.95)],
        promotion_candidates=[],
        broken_anchors=[
            BrokenAnchor(uid="p:note:x", name="X", project_name="p", file_path="x.md", unresolved_anchors=["ghost"])
        ],
        memory_index_issues=["p: MEMORY.md references missing file(s): ['ghost.md']"],
    )


def test_render_home_md_includes_counts() -> None:
    home = render_home_md(_sample_report())

    assert "## Inbox (1)" in home
    assert "docs/inbox/draft.md" in home
    assert "## Orphan notes (1)" in home
    assert "## Duplicate ids (1)" in home
    assert "## Dangling links (1)" in home
    assert "## Broken anchors (1)" in home
    assert "p:note:x" in home
    assert "ghost" in home
    assert "ghost.md" in home


def test_report_to_dict_is_json_safe() -> None:
    import orjson

    payload = report_to_dict(_sample_report())
    # Round-trips through orjson (the same serializer the CLI/MCP tool use)
    encoded = orjson.dumps(payload)
    decoded = orjson.loads(encoded)
    assert decoded["inbox_count"] == 1
    assert decoded["duplicate_ids"][0]["qualified_name"] == "p:note:dup"
    assert decoded["broken_anchors"][0]["uid"] == "p:note:x"
    assert decoded["broken_anchors"][0]["unresolved_anchors"] == ["ghost"]
