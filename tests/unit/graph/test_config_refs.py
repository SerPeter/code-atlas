"""Unit tests for the EnvVar / ResourceFile reference planner.

``_plan_config_refs`` is the single place both graph backends turn a parsed
READS_ENV / REFERENCES_FILE reference into nodes and edges, so the scoping
decisions (global env vars, project-scoped resource files) and the
names-only security invariant are all pinned here rather than per backend.
"""

from __future__ import annotations

from code_atlas.graph.client import _normalize_resource_path, _plan_config_refs
from code_atlas.parsing.ast import ParsedRelationship
from code_atlas.schema import GLOBAL_PROJECT, RelType, env_var_uid, resource_file_uid


def _env_ref(from_uid: str, name: str, **props: object) -> ParsedRelationship:
    return ParsedRelationship(
        from_qualified_name=from_uid, rel_type=RelType.READS_ENV, to_name=name, properties=dict(props)
    )


def _file_ref(from_uid: str, path: str, **props: object) -> ParsedRelationship:
    return ParsedRelationship(
        from_qualified_name=from_uid, rel_type=RelType.REFERENCES_FILE, to_name=path, properties=dict(props)
    )


# ---------------------------------------------------------------------------
# SECURITY: names only, never values
# ---------------------------------------------------------------------------


class TestNamesOnlyInvariant:
    """``os.getenv("API_KEY", "sk-live-abc123")`` puts a live secret in the
    default argument. Nothing a parser attaches to a reference may reach the
    graph — see graph/client.py's "capture NAMES, never VALUES" block.
    """

    def test_env_node_carries_no_parser_properties(self) -> None:
        secret = "sk-live-abc123"
        plan = _plan_config_refs(
            "proj",
            [_env_ref("proj:m.f", "API_KEY", default=secret, value=secret, fallback=secret, line=12)],
        )

        node = plan.env_nodes[env_var_uid("API_KEY")]
        assert set(node) == {"uid", "project_name", "name", "qualified_name"}
        assert secret not in "".join(node.values())

    def test_resource_node_carries_no_parser_properties(self) -> None:
        secret = "hunter2"
        plan = _plan_config_refs("proj", [_file_ref("proj:m.f", "conf/creds.yaml", contents=secret, mode="r")])

        node = plan.file_nodes[resource_file_uid("proj", "conf/creds.yaml")]
        # file_path is admissible under the names-never-values invariant: it is derived
        # from rel.to_name, the same source as name and qualified_name, and carries no
        # parser-observed content. The file's *contents* and open mode stay out.
        assert set(node) == {"uid", "project_name", "name", "qualified_name", "file_path"}
        assert secret not in "".join(node.values())
        assert node["file_path"] == "conf/creds.yaml"

    def test_edges_carry_no_properties_at_all(self) -> None:
        """Edges are bare 3-tuples — there is no channel for a default value to
        ride along on, even if the node allowlist were widened later.
        """
        plan = _plan_config_refs("proj", [_env_ref("proj:m.f", "API_KEY", default="sk-live-abc123")])

        assert plan.edges == [("proj:m.f", "env/API_KEY", "READS_ENV")]

    def test_neither_label_is_embeddable(self) -> None:
        """Defense in depth: even a leaked value could not reach the embedding
        API, because these labels have no vector index.
        """
        from code_atlas.schema import _EMBEDDABLE_LABELS, NodeLabel

        assert NodeLabel.ENV_VAR not in _EMBEDDABLE_LABELS
        assert NodeLabel.RESOURCE_FILE not in _EMBEDDABLE_LABELS


# ---------------------------------------------------------------------------
# Scoping: env vars are global, resource files are not
# ---------------------------------------------------------------------------


class TestScoping:
    def test_env_var_uid_has_no_project_prefix(self) -> None:
        plan = _plan_config_refs("proj", [_env_ref("proj:m.f", "DATABASE_URL")])

        node = plan.env_nodes["env/DATABASE_URL"]
        assert node["uid"] == "env/DATABASE_URL"
        assert node["project_name"] == GLOBAL_PROJECT

    def test_same_env_var_in_two_projects_is_one_node(self) -> None:
        a = _plan_config_refs("alpha", [_env_ref("alpha:m.f", "DATABASE_URL")])
        b = _plan_config_refs("beta", [_env_ref("beta:m.g", "DATABASE_URL")])

        assert set(a.env_nodes) == set(b.env_nodes) == {"env/DATABASE_URL"}

    def test_same_path_in_two_projects_is_two_nodes(self) -> None:
        """Deliberate asymmetry with env vars — a path is only meaningful
        relative to a project root.
        """
        a = _plan_config_refs("alpha", [_file_ref("alpha:m.f", "data/fixtures.json")])
        b = _plan_config_refs("beta", [_file_ref("beta:m.g", "data/fixtures.json")])

        assert set(a.file_nodes) == {"alpha:res/data/fixtures.json"}
        assert set(b.file_nodes) == {"beta:res/data/fixtures.json"}

    def test_resource_node_fields(self) -> None:
        plan = _plan_config_refs("proj", [_file_ref("proj:m.f", "data/fixtures.json")])

        node = plan.file_nodes["proj:res/data/fixtures.json"]
        assert node["project_name"] == "proj"
        assert node["name"] == "fixtures.json"
        assert node["qualified_name"] == "res/data/fixtures.json"


# ---------------------------------------------------------------------------
# Normalization + dedup
# ---------------------------------------------------------------------------


class TestNormalization:
    def test_backslashes_and_dot_prefix_converge(self) -> None:
        assert _normalize_resource_path(r".\data\fixtures.json") == "data/fixtures.json"
        assert _normalize_resource_path("./data/fixtures.json") == "data/fixtures.json"
        assert _normalize_resource_path("  data/fixtures.json  ") == "data/fixtures.json"

    def test_parent_segments_are_preserved(self) -> None:
        """``..`` is NOT collapsed: two references from different directories
        would otherwise merge into one wrong node.
        """
        assert _normalize_resource_path("../shared/config.yaml") == "../shared/config.yaml"

    def test_equivalent_paths_produce_one_node_two_edges(self) -> None:
        plan = _plan_config_refs(
            "proj",
            [_file_ref("proj:m.f", "./data/fixtures.json"), _file_ref("proj:m.g", r"data\fixtures.json")],
        )

        assert set(plan.file_nodes) == {"proj:res/data/fixtures.json"}
        assert len(plan.edges) == 2

    def test_duplicate_reference_yields_one_edge(self) -> None:
        """Two ``os.getenv("X")`` calls in one function are one edge — the graph
        stores no call-site multiplicity.
        """
        plan = _plan_config_refs("proj", [_env_ref("proj:m.f", "X"), _env_ref("proj:m.f", "X")])

        assert plan.edges == [("proj:m.f", "env/X", "READS_ENV")]

    def test_blank_names_are_dropped(self) -> None:
        plan = _plan_config_refs("proj", [_env_ref("proj:m.f", "   "), _file_ref("proj:m.f", "./")])

        assert plan.env_nodes == {}
        assert plan.file_nodes == {}
        assert plan.edges == []

    def test_unrelated_rel_types_are_ignored(self) -> None:
        other = ParsedRelationship(from_qualified_name="proj:m.f", rel_type=RelType.CALLS, to_name="helper")

        plan = _plan_config_refs("proj", [other, _env_ref("proj:m.f", "X")])

        assert plan.edges == [("proj:m.f", "env/X", "READS_ENV")]
