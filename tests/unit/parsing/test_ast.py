"""Unit tests for parsing.ast — content hash formula (v4) contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from code_atlas.parsing.ast import ParsedEntity, _compute_content_hash
from code_atlas.schema import NodeLabel


def _entity(**overrides: Any) -> ParsedEntity:
    defaults: dict[str, Any] = {
        "name": "work",
        "qualified_name": "proj:mod.work",
        "label": NodeLabel.CALLABLE,
        "kind": "function",
        "line_start": 1,
        "line_end": 3,
        "file_path": "mod.py",
        "docstring": "Doc.",
        "signature": "def work(x)",
        "tags": ["async"],
        "source": "def work(x):\n    return x\n",
    }
    defaults.update(overrides)
    return ParsedEntity(**defaults)


def test_content_hash_formula_v4():
    """content_hash = sha256 over name/kind/visibility/signature/docstring/sorted tags/source/extra_properties."""
    entity = _entity()
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        "",  # extra_properties empty -> not serialized (see docstring)
    ]
    expected = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    assert _compute_content_hash(entity) == expected


def test_content_hash_source_none_equals_empty():
    """Entities without source (Module, TypeDef, DocSection) hash "" for the source element."""
    assert _compute_content_hash(_entity(source=None)) == _compute_content_hash(_entity(source=""))


def test_content_hash_ignores_positional_fields():
    """line_start/line_end/file_path do not affect the hash."""
    a = _entity()
    b = _entity(line_start=42, line_end=44, file_path="other/mod.py")
    assert _compute_content_hash(a) == _compute_content_hash(b)


def test_content_hash_extra_properties_changes_hash():
    """A Note's frontmatter (extra_properties) is folded into the hash when non-empty."""
    a = _entity(extra_properties={})
    b = _entity(extra_properties={"tags": ["x"]})
    assert _compute_content_hash(a) != _compute_content_hash(b)


def test_content_hash_extra_properties_order_independent():
    """extra_properties is JSON-serialized with sort_keys — dict insertion order doesn't affect the hash."""
    a = _entity(extra_properties={"a": 1, "b": 2})
    b = _entity(extra_properties={"b": 2, "a": 1})
    assert _compute_content_hash(a) == _compute_content_hash(b)


def test_content_hash_extra_properties_matches_json_dumps():
    """The extra_properties hash element is exactly json.dumps(..., sort_keys=True, default=str)."""
    entity = _entity(extra_properties={"id": "foo", "kind": "note"})
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        json.dumps(entity.extra_properties, sort_keys=True, default=str),
    ]
    expected = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    assert _compute_content_hash(entity) == expected


# ---------------------------------------------------------------------------
# Rationale / citations — hash contract
#
# The invariant that matters operationally: an entity with no intent-bearing
# comment must hash EXACTLY as it did before the fields existed, otherwise
# adding the feature reindexes every project.
# ---------------------------------------------------------------------------


def test_content_hash_without_rationale_matches_eight_part_formula():
    """No rationale/citations -> the parts list stays at eight elements, byte-identical."""
    entity = _entity()
    assert entity.rationale is None
    assert entity.citations == []
    parts = [
        entity.name,
        entity.kind,
        entity.visibility,
        entity.signature or "",
        entity.docstring or "",
        ",".join(sorted(entity.tags)),
        entity.source or "",
        "",
    ]
    expected = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]
    assert _compute_content_hash(entity) == expected


def test_content_hash_rationale_changes_hash():
    assert _compute_content_hash(_entity()) != _compute_content_hash(_entity(rationale="WHY: because"))


def test_content_hash_citations_change_hash():
    assert _compute_content_hash(_entity()) != _compute_content_hash(_entity(citations=["ADR-0014"]))


def test_content_hash_citations_order_independent():
    a = _entity(citations=["ADR-0014", "RFC-7231"])
    b = _entity(citations=["RFC-7231", "ADR-0014"])
    assert _compute_content_hash(a) == _compute_content_hash(b)


def test_content_hash_rationale_and_citations_are_distinguishable():
    """The same payload in either field must not collide (each element is key-prefixed)."""
    a = _entity(rationale="ADR-0014")
    b = _entity(citations=["ADR-0014"])
    assert _compute_content_hash(a) != _compute_content_hash(b)


def test_rationale_settings_defaults_match_parser_defaults():
    """settings.py and ast.py each carry the marker defaults — pin them together."""
    from code_atlas.parsing.ast import (
        DEFAULT_CITATION_SCHEMES,
        DEFAULT_RATIONALE_MARKERS,
        DEFAULT_TASK_MARKERS,
    )
    from code_atlas.settings import RationaleSettings

    settings = RationaleSettings()
    assert settings.markers == list(DEFAULT_RATIONALE_MARKERS)
    assert settings.task_markers == list(DEFAULT_TASK_MARKERS)
    assert settings.citation_schemes == list(DEFAULT_CITATION_SCHEMES)
    assert settings.enabled is True
    assert settings.tasks is False
