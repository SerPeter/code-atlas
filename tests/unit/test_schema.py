"""Unit tests for graph schema definitions and DDL generation.

No infrastructure required — these test pure functions and data structures.
"""

from __future__ import annotations

from code_atlas.schema import (
    _CODE_LABELS,
    _DOC_LABELS,
    _EMBEDDABLE_LABELS,
    _ENTITY_LABELS,
    _EXTERNAL_LABELS,
    _MARKER_LABELS,
    _TEXT_SEARCHABLE_LABELS,
    COMPOSITE_INDICES,
    EXISTENCE_CONSTRAINTS,
    LABEL_PROPERTY_INDICES,
    SCHEMA_VERSION,
    TEXT_INDICES,
    UNIQUE_CONSTRAINTS,
    NodeLabel,
    generate_composite_index_ddl,
    generate_drop_redundant_marker_ddl,
    generate_drop_text_index_ddl,
    generate_drop_vector_index_ddl,
    generate_existence_constraint_ddl,
    generate_index_ddl,
    generate_text_index_ddl,
    generate_unique_constraint_ddl,
    generate_vector_index_ddl,
    primary_label_expr,
)


class TestLabelCompleteness:
    """Every NodeLabel must be accounted for in registries and groupings."""

    def test_unique_constraints_cover_all_labels(self):
        # Markers are exempt: a marker is stamped onto a node that already carries
        # a primary label enforcing uid uniqueness, so constraining it again would
        # re-check every write for nothing.
        unique_labels = {spec.label for spec in UNIQUE_CONSTRAINTS}
        assert unique_labels == set(NodeLabel) - _MARKER_LABELS

    def test_existence_constraints_cover_all_labels(self):
        existence_labels = {spec.label for spec in EXISTENCE_CONSTRAINTS}
        assert existence_labels == set(NodeLabel) - _MARKER_LABELS

    def test_label_sets_cover_all(self):
        grouped = _CODE_LABELS | _DOC_LABELS | _EXTERNAL_LABELS | _MARKER_LABELS | {NodeLabel.SCHEMA_VERSION}
        assert grouped == set(NodeLabel)

    def test_entity_labels_exclude_meta(self):
        assert NodeLabel.SCHEMA_VERSION not in _ENTITY_LABELS

    def test_entity_label_is_marker_not_entity_label(self):
        """NodeLabel.ENTITY is a marker stamped alongside a primary label, not a
        label in its own right.

        It must stay out of _ENTITY_LABELS: that set drives the constraint and index
        registries, and a marker sits on every node in the graph, so each registry it
        joins costs every write (see GraphClient._migrate_v14_trim_marker_indices).
        """
        assert {NodeLabel.ENTITY} == _MARKER_LABELS
        assert NodeLabel.ENTITY not in _ENTITY_LABELS

    def test_index_registry_covers_entity_and_marker_labels(self):
        index_labels = {spec.label for spec in LABEL_PROPERTY_INDICES}
        assert index_labels == _ENTITY_LABELS | _MARKER_LABELS

    def test_marker_label_indexed_only_on_what_is_queried(self):
        """The marker carries exactly the indices whose lookups are genuinely its own.

        - ``uid`` -- the uid-only MATCHes the marker exists to index.
        - ``(project_name, name)`` -- cross-project import resolution.
        - ``embed_hash`` -- the embedding dedup lookup (ATL-127), which asks "does ANY
          node, any label, any project, already carry a vector for this text?" There is
          no primary label to reach that through, and unindexed it is one full scan per
          hash -- measured at a 10s timeout for 3,000 hashes over 66k nodes.

        Anything else is reached through a primary label that has its own index, so
        indexing it here would buy no query anything and cost every node write -- which
        is what v13 did and v14 undid. Adding to this list needs the same argument
        embed_hash makes: a real lookup that no primary-label index can serve.
        """
        assert {s.property for s in LABEL_PROPERTY_INDICES if s.label is NodeLabel.ENTITY} == {"uid", "embed_hash"}
        assert {s.properties for s in COMPOSITE_INDICES if s.label is NodeLabel.ENTITY} == {("project_name", "name")}


class TestPrimaryLabelExpr:
    """A node's own label must survive the marker every node also carries."""

    def test_filters_every_marker_label(self):
        expr = primary_label_expr("n")
        assert expr.startswith("[")
        assert expr.endswith("][0]")
        for lbl in _MARKER_LABELS:
            assert f"'{lbl.value}'" in expr

    def test_binds_the_requested_variable(self):
        assert "labels(affected)" in primary_label_expr("affected")
        assert "labels(n)" in primary_label_expr()

    def test_does_not_index_position_zero(self):
        """Memgraph returns labels in write order, so labels(n)[0] is a convention
        the write sites happen to keep rather than something the database promises.
        A node written ':Entity:Callable' would report its type as 'Entity'."""
        assert "labels(n)[0]" not in primary_label_expr("n")


class TestMarkerDropDDL:
    """v14 drops what v13 created on the marker and nothing queries."""

    def test_drops_every_unqueried_index_and_constraint(self):
        stmts = generate_drop_redundant_marker_ddl()
        joined = " ".join(stmts)
        for prop in ("qualified_name", "file_path", "name", "project_name", "kind", "content_hash"):
            assert f"DROP INDEX ON :Entity({prop});" in stmts
        assert "DROP INDEX ON :Entity(project_name, file_path);" in stmts
        assert "DROP CONSTRAINT ON (n:Entity) ASSERT n.uid IS UNIQUE;" in stmts
        assert "EXISTS (n.uid)" in joined
        assert "EXISTS (n.project_name)" in joined

    def test_keeps_the_two_indices_the_marker_exists_for(self):
        stmts = generate_drop_redundant_marker_ddl()
        assert "DROP INDEX ON :Entity(uid);" not in stmts
        assert "DROP INDEX ON :Entity(project_name, name);" not in stmts


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
        for prop in ("uid", "qualified_name", "file_path", "name", "kind", "content_hash"):
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
        # One vector index per EMBEDDABLE label. This used to be spelled
        # len(_TEXT_SEARCHABLE_LABELS) ("same count as embeddable") — the two
        # sets diverged in v7, when EnvVar/ResourceFile became text-searchable
        # but deliberately not embeddable.
        assert len(stmts_768) == len(_EMBEDDABLE_LABELS)
        for stmt in stmts_768:
            assert "768" in stmt
            assert stmt.startswith("CREATE VECTOR INDEX")
        for stmt in stmts_384:
            assert "384" in stmt

    def test_vector_index_ddl_includes_cos_metric(self):
        stmts = generate_vector_index_ddl(768)
        for stmt in stmts:
            assert '"cos"' in stmt

    def test_vector_index_ddl_declarative_syntax(self):
        stmts = generate_vector_index_ddl(768)
        for stmt in stmts:
            assert "ON :" in stmt
            assert "WITH CONFIG" in stmt
            assert "(embedding)" in stmt
            assert stmt.endswith(";")

    def test_drop_vector_index_ddl_syntax(self):
        stmts = generate_drop_vector_index_ddl()
        assert len(stmts) == len(_EMBEDDABLE_LABELS)
        for stmt in stmts:
            assert stmt.startswith("DROP VECTOR INDEX")
            assert stmt.endswith(";")
            assert "CALL" not in stmt

    def test_text_index_ddl_one_per_searchable_label(self):
        stmts = generate_text_index_ddl()
        assert len(stmts) == len(TEXT_INDICES)
        for stmt in stmts:
            assert stmt.startswith("CREATE TEXT INDEX")
            assert "ON :" in stmt
            assert stmt.endswith(";")

    def test_text_index_ddl_covers_searchable_labels(self):
        stmts = generate_text_index_ddl()
        all_text = " ".join(stmts)
        for lbl in _TEXT_SEARCHABLE_LABELS:
            assert lbl.value in all_text, f"Missing text index for label: {lbl.value}"

    def test_drop_text_index_ddl_syntax(self):
        stmts = generate_drop_text_index_ddl()
        assert len(stmts) == len(TEXT_INDICES)
        for stmt in stmts:
            assert stmt.startswith("DROP TEXT INDEX")
            assert stmt.endswith(";")
            assert "CALL" not in stmt


class TestCompositeIndexDDL:
    """Composite index DDL generators produce valid Cypher syntax."""

    def test_composite_index_ddl_syntax(self):
        stmts = generate_composite_index_ddl()
        assert len(stmts) == len(COMPOSITE_INDICES)
        for stmt in stmts:
            assert stmt.startswith("CREATE INDEX ON :")
            assert ", " in stmt  # composite has multiple properties
            assert stmt.endswith(";")

    def test_composite_index_covers_entity_labels(self):
        index_labels = {spec.label for spec in COMPOSITE_INDICES}
        assert index_labels == _ENTITY_LABELS | _MARKER_LABELS

    def test_composite_index_has_expected_property_combos(self):
        stmts = generate_composite_index_ddl()
        all_text = " ".join(stmts)
        assert "project_name, file_path" in all_text
        assert "project_name, name" in all_text


class TestSchemaVersion:
    def test_schema_version_positive(self):
        assert SCHEMA_VERSION >= 1

    def test_schema_version_is_int(self):
        assert isinstance(SCHEMA_VERSION, int)
