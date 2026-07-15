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


# ---------------------------------------------------------------------------
# 4. Frontmatter-triggered note mode (Phase 1 — knowledge vault)
# ---------------------------------------------------------------------------


def _note(parsed: ParsedFile):
    notes = [e for e in parsed.entities if e.label == NodeLabel.NOTE]
    assert len(notes) == 1, f"Expected exactly 1 Note entity, got {len(notes)}"
    return notes[0]


def test_vault_dialect_triggers_note_mode():
    source = "---\nid: my-note\nkind: draft\n---\n\n# My Note\n\nBody text.\n"
    parsed = _parse(source, path="docs/inbox/my-note.md")

    note = _note(parsed)
    assert note.qualified_name == f"{PROJECT}:note:my-note"
    assert note.kind == "draft"
    assert note.name == "My Note"
    # docstring holds the full body verbatim (including the heading used for the title) —
    # the graph never mutates file content, it only derives fields from it.
    assert note.docstring == "# My Note\n\nBody text."
    # Note mode replaces DocFile/DocSection entirely.
    assert not any(e.label in (NodeLabel.DOC_FILE, NodeLabel.DOC_SECTION) for e in parsed.entities)


def test_vault_dialect_title_falls_back_to_humanized_slug():
    source = "---\nid: watcher-debounce-selfcancel\nkind: note\n---\n\nNo heading here, just prose.\n"
    parsed = _parse(source, path="docs/notes/watcher-debounce-selfcancel.md")
    note = _note(parsed)
    assert note.name == "Watcher Debounce Selfcancel"


def test_vault_dialect_tags_frontmatter_and_inline():
    source = (
        "---\nid: tagged-note\nkind: note\ntags: [indexing, asyncio]\n---\n\n"
        "# Tagged Note\n\nSee #gotcha and #indexing (dup).\n"
    )
    parsed = _parse(source, path="docs/notes/tagged-note.md")
    note = _note(parsed)
    assert set(note.tags) == {"indexing", "asyncio", "gotcha"}


def test_vault_dialect_heading_hash_not_mistaken_for_tag():
    source = "---\nid: heading-note\nkind: note\n---\n\n# Heading One\n\n## Heading Two\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/heading-note.md")
    note = _note(parsed)
    assert note.tags == []


def test_memory_dialect_triggers_note_mode():
    source = (
        "---\nname: knowledge-convergence-plan\n"
        "description: Plan to dissolve neo-memoria into code-atlas.\n"
        "metadata:\n  type: project\n---\n\nBody content here.\n"
    )
    parsed = _parse(source, path="memory/knowledge-convergence-plan.md")
    note = _note(parsed)
    assert note.qualified_name == f"{PROJECT}:note:knowledge-convergence-plan"
    assert note.name == "Plan to dissolve neo-memoria into code-atlas."
    assert note.kind == "note"
    assert note.extra_properties.get("subtype") == "project"


def test_frontmatter_without_note_dialect_keeps_ordinary_doc_parsing():
    # Frontmatter present, but matching neither dialect (no id+kind, no name+description+metadata.type).
    source = "---\ntitle: Just a title\n---\n\n# Heading\n\ncontent\n"
    parsed = _parse(source, path="docs/plain.md")
    assert any(e.label == NodeLabel.DOC_FILE for e in parsed.entities)
    assert any(e.label == NodeLabel.DOC_SECTION and e.name == "Heading" for e in parsed.entities)
    assert not any(e.label == NodeLabel.NOTE for e in parsed.entities)


def test_no_frontmatter_unaffected():
    parsed = _parse("# Title\ncontent\n", path="docs/readme.md")
    assert any(e.label == NodeLabel.DOC_FILE for e in parsed.entities)
    assert not any(e.label == NodeLabel.NOTE for e in parsed.entities)


# ---------------------------------------------------------------------------
# 5. Wikilinks, derived_from/supersedes (LINKS_TO / DERIVED_FROM / SUPERSEDES)
# ---------------------------------------------------------------------------


def test_wikilink_same_project_resolves_to_note_uid():
    source = "---\nid: a\nkind: note\n---\n\nSee [[b]] for context.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    links = [r for r in parsed.relationships if r.rel_type == RelType.LINKS_TO]
    assert len(links) == 1
    assert links[0].to_name == f"{PROJECT}:note:b"
    assert links[0].properties["alias"] == "b"


def test_wikilink_with_alias():
    source = "---\nid: a\nkind: note\n---\n\nSee [[b|the other note]] for context.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    links = [r for r in parsed.relationships if r.rel_type == RelType.LINKS_TO]
    assert links[0].properties["alias"] == "the other note"
    assert links[0].to_name == f"{PROJECT}:note:b"


def test_wikilink_cross_project():
    source = "---\nid: a\nkind: note\n---\n\nSee [[overspan:ruff-gotcha]].\n"
    parsed = _parse(source, path="docs/notes/a.md")
    links = [r for r in parsed.relationships if r.rel_type == RelType.LINKS_TO]
    assert links[0].to_name == "overspan:note:ruff-gotcha"


def test_wikilink_heading_and_block_refs_drop_fragment():
    source = "---\nid: a\nkind: note\n---\n\nSee [[b#Some Heading]] and [[c^blockid]].\n"
    parsed = _parse(source, path="docs/notes/a.md")
    targets = {r.to_name for r in parsed.relationships if r.rel_type == RelType.LINKS_TO}
    assert targets == {f"{PROJECT}:note:b", f"{PROJECT}:note:c"}


def test_derived_from_and_supersedes_frontmatter():
    source = "---\nid: a\nkind: note\nderived_from: [inbox-x]\nsupersedes: [old-note]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    derived = [r for r in parsed.relationships if r.rel_type == RelType.DERIVED_FROM]
    superseded = [r for r in parsed.relationships if r.rel_type == RelType.SUPERSEDES]
    assert derived[0].to_name == f"{PROJECT}:note:inbox-x"
    assert superseded[0].to_name == f"{PROJECT}:note:old-note"


def test_ordinary_docsection_extracts_wikilinks_too():
    source = "# A\n\n## X\nSee [[some-note]] for background.\n"
    parsed = _parse(source, path="docs/architecture.md")
    links = [r for r in parsed.relationships if r.rel_type == RelType.LINKS_TO]
    assert len(links) == 1
    assert links[0].to_name == f"{PROJECT}:note:some-note"


def test_note_backtick_symbol_emits_documents_edge():
    source = "---\nid: a\nkind: note\n---\n\nSee `FileWatcher._flush` for the bug.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    doc_rels = [r for r in parsed.relationships if r.rel_type == RelType.DOCUMENTS]
    assert any(r.to_name == "_flush" for r in doc_rels)


# ---------------------------------------------------------------------------
# 6. Explicit anchors: frontmatter (Phase 3 — anchors + staleness)
# ---------------------------------------------------------------------------


def _anchor_rels(parsed: ParsedFile):
    return [
        r for r in parsed.relationships if r.rel_type == RelType.DOCUMENTS and r.properties.get("link_type") == "anchor"
    ]


def test_anchor_uid_form():
    source = "---\nid: a\nkind: note\nanchors: [code-atlas:src.code_atlas.foo.Bar]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert len(rels) == 1
    assert rels[0].to_name == "code-atlas:src.code_atlas.foo.Bar"
    assert rels[0].properties["anchor_form"] == "uid"
    assert rels[0].properties["confidence"] == 1.0
    assert "anchor_project" not in rels[0].properties
    assert "anchor_symbol" not in rels[0].properties


def test_anchor_bare_relative_path_form():
    source = "---\nid: a\nkind: note\nanchors: [src/code_atlas/indexing/watcher.py]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert rels[0].to_name == "src/code_atlas/indexing/watcher.py"
    assert rels[0].properties["anchor_form"] == "path"


def test_anchor_project_prefixed_path_form():
    source = "---\nid: a\nkind: note\nanchors: [code-atlas:src/code_atlas/indexing/watcher.py]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert rels[0].to_name == "src/code_atlas/indexing/watcher.py"
    assert rels[0].properties["anchor_form"] == "project_path"
    assert rels[0].properties["anchor_project"] == "code-atlas"


def test_anchor_absolute_path_form():
    source = "---\nid: a\nkind: note\nanchors: ['D:/dev/git/code-atlas/src/code_atlas/foo.py']\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert rels[0].to_name == "D:/dev/git/code-atlas/src/code_atlas/foo.py"
    assert rels[0].properties["anchor_form"] == "absolute_path"


def test_anchor_posix_absolute_path_form():
    source = "---\nid: a\nkind: note\nanchors: [/home/user/repo/src/foo.py]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert rels[0].properties["anchor_form"] == "absolute_path"


def test_anchor_symbol_refinement():
    source = "---\nid: a\nkind: note\nanchors: [src/code_atlas/foo.py#MyClass]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert rels[0].to_name == "src/code_atlas/foo.py"
    assert rels[0].properties["anchor_symbol"] == "MyClass"


def test_anchor_project_prefixed_uid_not_mistaken_for_path():
    # No '/' and no file extension after the project prefix -> uid form, not project_path.
    source = "---\nid: a\nkind: note\nanchors: [code-atlas:src.code_atlas.foo.Bar]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert rels[0].properties["anchor_form"] == "uid"


def test_anchors_excluded_from_extra_properties():
    source = "---\nid: a\nkind: note\nanchors: [foo.py]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    note = _note(parsed)
    assert "anchors" not in note.extra_properties


def test_no_anchors_frontmatter_emits_no_anchor_rels():
    source = "---\nid: a\nkind: note\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    assert _anchor_rels(parsed) == []


def test_anchors_non_list_value_ignored():
    source = "---\nid: a\nkind: note\nanchors: not-a-list\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    assert _anchor_rels(parsed) == []


def test_anchor_short_ambiguous_extension_without_slash_is_uid():
    # "c"/"h" collide with plausible dotted symbol final segments — without a
    # "/" they must NOT be classified as path-form.
    source = "---\nid: a\nkind: note\nanchors: [models.Config.c, physics.Model.h]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = {r.to_name: r.properties["anchor_form"] for r in _anchor_rels(parsed)}
    assert rels["models.Config.c"] == "uid"
    assert rels["physics.Model.h"] == "uid"


def test_anchor_short_extension_with_slash_still_path():
    # Regression guard: a "/" makes the short extension unambiguous again.
    source = "---\nid: a\nkind: note\nanchors: [src/foo.c]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = _anchor_rels(parsed)
    assert rels[0].to_name == "src/foo.c"
    assert rels[0].properties["anchor_form"] == "path"


def test_anchor_conventional_filenames_are_path_form():
    source = "---\nid: a\nkind: note\nanchors: [Dockerfile, Makefile]\n---\n\nBody.\n"
    parsed = _parse(source, path="docs/notes/a.md")
    rels = {r.to_name: r.properties["anchor_form"] for r in _anchor_rels(parsed)}
    assert rels["Dockerfile"] == "path"
    assert rels["Makefile"] == "path"


def test_anchor_bare_symbol_fragment_skipped():
    source = '---\nid: a\nkind: note\nanchors: ["#OrphanSymbol"]\n---\n\nBody.\n'
    parsed = _parse(source, path="docs/notes/a.md")
    assert _anchor_rels(parsed) == []
