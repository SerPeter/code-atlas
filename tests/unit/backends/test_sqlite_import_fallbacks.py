"""`resolve_imports`' two fallbacks after the exact match: alias widening and the case fold.

Both exist because Salesforce names things in ways an exact string match cannot follow:

* ``<c:foo>`` in Aura markup, and a FlexiPage ``componentName``, name one component
  identity that the platform forbids an Aura and an LWC component from sharing — but
  nothing in the *file* says which kind it is. The parsers emit ``cmp.foo`` and the
  resolver widens it.
* Salesforce API names are case-insensitive, so ``FROM ACCOUNT`` in Apex and
  ``Account.object-meta.xml`` are the same object.

Neither fallback may fire outside its own namespace: Python, TypeScript, Go and Rust
imports are case-sensitive, and folding those would mint edges the language rejects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.graph.client import build_case_folded_map, resolve_component_alias
from code_atlas.parsing.ast import ParsedEntity, ParsedRelationship
from code_atlas.schema import NodeLabel, RelType, Visibility

if TYPE_CHECKING:
    from pathlib import Path

PROJECT = "p"


def _entity(qualified_name: str, *, path: str = "src/a.cls", label: NodeLabel = NodeLabel.TYPE_DEF) -> ParsedEntity:
    return ParsedEntity(
        name=qualified_name.rsplit(".", 1)[-1],
        qualified_name=qualified_name,
        label=label,
        kind="class",
        line_start=1,
        line_end=2,
        file_path=path,
        visibility=Visibility.PUBLIC,
        content_hash=f"h-{qualified_name}",
    )


def _imports(from_uid: str, to_name: str) -> ParsedRelationship:
    return ParsedRelationship(from_qualified_name=from_uid, rel_type=RelType.IMPORTS, to_name=to_name)


@pytest.fixture
async def client(tmp_path: Path):
    async with SqliteGraphClient(tmp_path / "g.sqlite3", dimension=8) as graph:
        await graph.ensure_schema()
        yield graph


async def _edges(graph: SqliteGraphClient) -> set[tuple[str, str]]:
    conn = await graph._get_conn()
    cursor = await conn.execute("SELECT from_uid, to_uid FROM edges WHERE rel_type = 'IMPORTS'")
    return {tuple(row) for row in await cursor.fetchall()}


# ---------------------------------------------------------------------------
# The two builders, in isolation
# ---------------------------------------------------------------------------


def test_the_fold_map_is_empty_without_salesforce_nodes():
    """The gate is on the node side, so a repo with no Salesforce metadata pays nothing.

    Six of the folded namespaces (`app`, `page`, `component`, `layout`, `tab`,
    `profile`) are also plausible directory names, but a path-derived qualified name
    is rooted at a top-level directory, so a bare one never appears first.
    """
    folded = build_case_folded_map(
        [
            ("src.services.user.UserService", "u1"),
            ("web.components.Button", "u2"),
            ("app", "u3"),
            ("mypackage.app.routes.handler", "u4"),
        ]
    )
    assert folded == {}


def test_the_fold_map_refuses_a_collision_instead_of_picking():
    """Two nodes folding to one key resolve to nothing at all.

    Two SObjects differing only by case cannot both exist in one org, but two nodes
    can -- a monorepo, a managed package, a parse artefact. Measured: six such
    collisions inside `apex.` in one production org, including `apex.Class` against
    `apex.class`. Picking one would point an edge at an arbitrary member of the pair.
    """
    folded = build_case_folded_map(
        [("apex.Thing.exampleObject", "u1"), ("apex.Thing.ExampleObject", "u2"), ("sobject.Account", "u3")]
    )
    assert folded["apex.thing.exampleobject"] is None
    assert folded["sobject.account"] == "u3"


def test_alias_widening_only_touches_the_cmp_prefix():
    internal = {"aura.Foo": "a1", "lwc.Bar": "l1", "sobject.Baz": "s1"}
    assert resolve_component_alias("cmp.Foo", internal) == "a1"
    assert resolve_component_alias("cmp.Bar", internal) == "l1"
    assert resolve_component_alias("cmp.Missing", internal) is None
    # Anything not `cmp.`-prefixed is none of its business.
    assert resolve_component_alias("sobject.Baz", internal) is None


# ---------------------------------------------------------------------------
# End to end, through the backend
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("defined", "expected_kind"),
    [("aura.errorPanel", "an Aura bundle"), ("lwc.errorPanel", "an LWC bundle")],
)
async def test_a_cmp_target_resolves_to_whichever_kind_exists(
    client: SqliteGraphClient, defined: str, expected_kind: str
):
    """One reference, either kind, no stub — the point of the whole mechanism."""
    await client.upsert_file_entities(PROJECT, "src/a.cls", [_entity("apex.Caller"), _entity(defined)], [])
    await client.resolve_imports(PROJECT, [_imports("apex.Caller", "cmp.errorPanel")])

    assert ("apex.Caller", defined) in await _edges(client), f"should have resolved to {expected_kind}"


async def test_a_cmp_target_with_no_component_still_becomes_a_stub(client: SqliteGraphClient):
    """A managed-package or platform component genuinely IS external.

    The stub is correct here and must survive; only the false twin was the problem.
    """
    await client.upsert_file_entities(PROJECT, "src/a.cls", [_entity("apex.Caller")], [])
    await client.resolve_imports(PROJECT, [_imports("apex.Caller", "cmp.someManagedThing")])

    assert any(to_uid.startswith(f"{PROJECT}:ext/") for _from, to_uid in await _edges(client))


async def test_a_salesforce_target_resolves_case_insensitively(client: SqliteGraphClient):
    """`FROM ACCOUNT` in Apex and `Account.object-meta.xml` are the same object."""
    await client.upsert_file_entities(PROJECT, "src/a.cls", [_entity("apex.Caller"), _entity("sobject.Account")], [])
    await client.resolve_imports(PROJECT, [_imports("apex.Caller", "sobject.ACCOUNT")])

    assert ("apex.Caller", "sobject.Account") in await _edges(client)


async def test_a_python_import_is_never_case_folded(client: SqliteGraphClient):
    """Python is case-sensitive; folding it would mint an edge the language rejects."""
    await client.upsert_file_entities(
        PROJECT,
        "src/a.py",
        [_entity("src.a.Caller", path="src/a.py"), _entity("src.services.UserService", path="src/a.py")],
        [],
    )
    await client.resolve_imports(PROJECT, [_imports("src.a.Caller", "src.services.userservice")])

    assert ("src.a.Caller", "src.services.UserService") not in await _edges(client)


async def test_an_exact_match_still_wins_over_a_fold(client: SqliteGraphClient):
    """The fold is a fallback, not a replacement: exact casing is tried first."""
    await client.upsert_file_entities(
        PROJECT,
        "src/a.cls",
        [_entity("apex.Caller"), _entity("sobject.Account"), _entity("sobject.ACCOUNT")],
        [],
    )
    await client.resolve_imports(PROJECT, [_imports("apex.Caller", "sobject.ACCOUNT")])

    assert ("apex.Caller", "sobject.ACCOUNT") in await _edges(client)
    assert ("apex.Caller", "sobject.Account") not in await _edges(client)
