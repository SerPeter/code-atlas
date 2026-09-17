"""The routing hooks' raw Cypher, against a real Memgraph.

The hooks bypass GraphClient (it takes seconds to import, and they run on every grep), so
their queries are not covered by anything else that exercises the graph layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas import hooks
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import SCHEMA_VERSION, NodeLabel, Visibility

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings

pytestmark = pytest.mark.integration

PROJECT = "test_hooks"


def _entity(name: str, label: NodeLabel = NodeLabel.CALLABLE) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=f"{PROJECT}:mod.{name}",
        label=label,
        kind="function" if label == NodeLabel.CALLABLE else "class",
        line_start=7,
        line_end=9,
        file_path="mod.py",
        docstring=None,
        signature=None,
        visibility=Visibility.PUBLIC,
        content_hash="h",
    )


async def test_memgraph_context_and_lookup(graph_client: GraphClient, settings: AtlasSettings) -> None:
    await graph_client.ensure_schema()  # the fixture wipes every node, the SchemaVersion marker included
    entities = [_entity("target"), _entity("caller_a"), _entity("caller_b"), _entity("Holder", NodeLabel.TYPE_DEF)]
    await graph_client.upsert_file_entities(PROJECT, "mod.py", entities, [])
    await graph_client.execute_write(
        "UNWIND $edges AS e MATCH (a {uid: e[0]}), (b {uid: e[1]}) "
        "FOREACH (_ IN CASE WHEN e[2] = 'CALLS' THEN [1] ELSE [] END | CREATE (a)-[:CALLS]->(b)) "
        "FOREACH (_ IN CASE WHEN e[2] = 'REFERENCES' THEN [1] ELSE [] END | CREATE (a)-[:REFERENCES]->(b))",
        {
            "edges": [
                [f"{PROJECT}:mod.caller_a", f"{PROJECT}:mod.target", "CALLS"],
                [f"{PROJECT}:mod.caller_b", f"{PROJECT}:mod.target", "CALLS"],
                [f"{PROJECT}:mod.Holder", f"{PROJECT}:mod.target", "REFERENCES"],
            ]
        },
    )

    mg = settings.memgraph
    ctx: dict[str, Any] = {
        "project": PROJECT,
        "backend": "memgraph",
        "uri": f"bolt://{mg.host}:{mg.port}",
        "timeout_s": settings.health.connect_timeout_s,
    }
    picked = hooks._pick_project(ctx, hooks._run(ctx, hooks._count_query(ctx), projects=[PROJECT, "test_absent"]))
    assert picked["available"] is True
    assert picked["entities"] == 4
    assert picked["schema"] == SCHEMA_VERSION

    rows = hooks.lookup(picked, ["target", "missing"])
    assert len(rows) == 1
    row = rows[0]
    assert (row["uid"], row["label"], row["file"], row["line"]) == (f"{PROJECT}:mod.target", "Callable", "mod.py", 7)
    assert (row["callers"], row["refs"]) == (2, 3)
