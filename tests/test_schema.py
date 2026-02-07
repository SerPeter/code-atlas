"""Unit tests for graph schema definitions and DDL generation.

No infrastructure required — these test pure functions and data structures.
"""

from __future__ import annotations

from code_atlas.schema import (
    _CODE_LABELS,
    _DOC_LABELS,
    _ENTITY_LABELS,
    _EXTERNAL_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    EXISTENCE_CONSTRAINTS,
    LABEL_PROPERTY_INDICES,
    SCHEMA_VERSION,
    TEXT_INDICES,
    UNIQUE_CONSTRAINTS,
    NodeLabel,
    generate_existence_constraint_ddl,
    generate_index_ddl,
    generate_text_index_ddl,
    generate_unique_constraint_ddl,
    generate_vector_index_ddl,
)


class TestLabelCompleteness:
    """Every NodeLabel must be accounted for in registries and groupings."""

    def test_unique_constraints_cover_all_labels(self):
        unique_labels = {spec.label for spec in UNIQUE_CONSTRAINTS}
        assert unique_labels == set(NodeLabel)

    def test_existence_constraints_cover_all_labels(self):
        existence_labels = {spec.label for spec in EXISTENCE_CONSTRAINTS}
        assert existence_labels == set(NodeLabel)

    def test_label_sets_cover_all(self):
        grouped = _CODE_LABELS | _DOC_LABELS | _EXTERNAL_LABELS | {NodeLabel.SCHEMA_VERSION}
        assert grouped == set(NodeLabel)

    def test_entity_labels_exclude_meta(self):
        assert NodeLabel.SCHEMA_VERSION not in _ENTITY_LABELS

    def test_index_registry_covers_entity_labels(self):
        index_labels = {spec.label for spec in LABEL_PROPERTY_INDICES}
        assert index_labels == _ENTITY_LABELS


class TestDDLGeneration:
    """DDL generators produce valid Cypher syntax with correct counts."""

    def test_unique_constraint_ddl_syntax(self):
        stmts = generate_unique_constraint_ddl()
        assert len(stmts) == len(UNIQUE_CONSTRAINTS)
        for stmt in stmts:
            assert stmt.startswith("CREATE CONSTRAINT ON")
            assert "IS UNIQUE" in stmt
            assert stmt.endswith(";")

    def test_existence_constraint_ddl_syntax(self):
        stmts = generate_existence_constraint_ddl()
        assert len(stmts) == len(EXISTENCE_CONSTRAINTS)
        for stmt in stmts:
            assert stmt.startswith("CREATE CONSTRAINT ON")
            assert "EXISTS" in stmt
            assert stmt.endswith(";")

    def test_index_ddl_has_expected_properties(self):
        stmts = generate_index_ddl()
        all_text = " ".join(stmts)
        for prop in ("qualified_name", "file_path", "name", "kind", "content_hash"):
            assert prop in all_text, f"Missing index for property: {prop}"

    def test_index_ddl_syntax(self):
        stmts = generate_index_ddl()
        assert len(stmts) == len(LABEL_PROPERTY_INDICES)
        for stmt in stmts:
            assert stmt.startswith("CREATE INDEX ON :")
            assert stmt.endswith(";")

    def test_vector_index_ddl_dimension_parameterized(self):
        stmts_768 = generate_vector_index_ddl(768)
        stmts_384 = generate_vector_index_ddl(384)
        assert len(stmts_768) == len(_TEXT_SEARCHABLE_LABELS)  # same count as embeddable
        for stmt in stmts_768:
            assert "768" in stmt
            assert "vector_index.create" in stmt
        for stmt in stmts_384:
            assert "384" in stmt

    def test_vector_index_ddl_includes_cosine_metric(self):
        stmts = generate_vector_index_ddl(768)
        for stmt in stmts:
            assert "'cosine'" in stmt

    def test_text_index_ddl_one_per_searchable_label(self):
        stmts = generate_text_index_ddl()
        assert len(stmts) == len(TEXT_INDICES)
        for stmt in stmts:
            assert "text_search.create_index" in stmt
            assert stmt.endswith(";")

    def test_text_index_ddl_covers_searchable_labels(self):
        stmts = generate_text_index_ddl()
        all_text = " ".join(stmts)
        for lbl in _TEXT_SEARCHABLE_LABELS:
            assert lbl.value in all_text, f"Missing text index for label: {lbl.value}"


class TestSchemaVersion:
    def test_schema_version_positive(self):
        assert SCHEMA_VERSION >= 1

    def test_schema_version_is_int(self):
        assert isinstance(SCHEMA_VERSION, int)
