"""Tests for Markdown parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_markdown", reason="tree-sitter-markdown not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import NodeLabel, RelType

PROJECT = "test_project"


def _parse(source: str, path: str = "docs/example.md") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entities_by_name(parsed: ParsedFile, name: str) -> list:
    return [e for e in parsed.entities if e.name == name and e.label == NodeLabel.DOC_SECTION]


def _entity_by_name(parsed: ParsedFile, name: str):
    matches = _entities_by_name(parsed, name)
    assert len(matches) == 1, f"Expected 1 DocSection named {name!r}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# 1. Language detection / basic DocFile
# ---------------------------------------------------------------------------


def test_language_detection_md():
    assert get_language_for_file("docs/readme.md") is not None


def test_doc_file_entity():
    parsed = _parse("# Title\ncontent\n", path="docs/readme.md")
    doc_file = next(e for e in parsed.entities if e.label == NodeLabel.DOC_FILE)
    assert doc_file.qualified_name == f"{PROJECT}:docs/readme.md"
    assert doc_file.line_start == 1
    assert doc_file.line_end == 2


def test_single_section_qualified_name():
    parsed = _parse("# A\n\n## X\ncontent one\n", path="docs/dup.md")
    section = _entity_by_name(parsed, "X")
    assert section.qualified_name == f"{PROJECT}:docs/dup.md > A > X"
    assert section.docstring == "content one"


# ---------------------------------------------------------------------------
# 2. Duplicate sibling headings (medium finding, markdown.py:~247)
# ---------------------------------------------------------------------------


def test_duplicate_sibling_headings_get_distinct_qualified_names():
    source = "# A\n\n## X\ncontent one\n\n## X\ncontent two\n"
    parsed = _parse(source, path="docs/dup.md")

    sections = _entities_by_name(parsed, "X")
    assert len(sections) == 2, f"Expected 2 DocSection entities named 'X', got {len(sections)}"

    qns = {s.qualified_name for s in sections}
    assert len(qns) == 2, f"Expected distinct qualified_names, got {qns}"

    # Neither section's content was clobbered.
    docstrings = {s.docstring for s in sections}
    assert docstrings == {"content one", "content two"}

    # The first occurrence keeps the bare breadcrumb qn; only the collision is disambiguated.
    assert f"{PROJECT}:docs/dup.md > A > X" in qns

    # Both DocSection uids must actually be reachable via CONTAINS from the DocFile,
    # not just via subsection nesting (i.e. no dropped relationship for either).
    contains_targets = {
        r.to_name
        for r in parsed.relationships
        if r.rel_type == RelType.CONTAINS and r.from_qualified_name.endswith("docs/dup.md")
    }
    assert qns <= contains_targets


# ---------------------------------------------------------------------------
# 3. DocSection line_end off-by-one without trailing newline (low finding, markdown.py:~367)
# ---------------------------------------------------------------------------


def test_section_line_end_without_trailing_newline():
    # No trailing newline after "content line 2" — DocFile already accounts for this
    # (total_lines includes the last, newline-less line); DocSection must match.
    source = b"# T\ncontent line 2"
    result = parse_file("docs/notrail.md", source, PROJECT)
    assert result is not None

    doc_file = next(e for e in result.entities if e.label == NodeLabel.DOC_FILE)
    assert doc_file.line_end == 2

    section = next(e for e in result.entities if e.label == NodeLabel.DOC_SECTION and e.name == "T")
    assert section.line_end == 2


def test_section_line_end_with_trailing_newline():
    # Regression guard: the common (trailing-newline) case must keep reporting correctly.
    source = b"# T\ncontent line 2\n"
    result = parse_file("docs/trail.md", source, PROJECT)
    assert result is not None

    section = next(e for e in result.entities if e.label == NodeLabel.DOC_SECTION and e.name == "T")
    assert section.line_end == 2
