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
