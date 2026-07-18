"""End-to-end integration tests for the embedded (SQLite) backend.

Unlike the rest of ``tests/integration/``, this tier needs no Docker/
testcontainers — SQLite is a plain file. It drives the real production code
paths (the ``atlas`` CLI via ``typer.testing.CliRunner``, and the real
``@mcp.tool`` implementations via ``_invoke_tool``) against
``ATLAS_BACKEND__GRAPH=sqlite`` / ``ATLAS_BACKEND__QUEUE=sqlite`` — never
``SqliteGraphClient`` methods directly, that's what the unit tests already
cover.

This is the tier the embedded-backend plan's Verification section required
but that was never built (see the plan's "Post-implementation finding: Phase
3.5") — its absence is exactly why ``SqliteGraphClient.execute()``'s
``NotImplementedError`` shipped undetected: ``atlas status`` and
``get_context`` both crashed on a real sqlite-backed project before Phase 3.5
ported every domain query behind named ``GraphBackend`` methods.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from code_atlas.backends import create_graph_client
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.cli import app
from code_atlas.server.mcp import AppContext, create_mcp_server
from code_atlas.settings import AtlasSettings, BackendSettings, EmbeddingSettings, derive_project_name
from tests.unit.server.test_mcp import _invoke_tool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

runner = CliRunner()

# Explicit (not "auto") backend choice — never probes/falls back to Memgraph
# or Valkey, so this tier has zero external dependencies.
_ENV = {"ATLAS_BACKEND__GRAPH": "sqlite", "ATLAS_BACKEND__QUEUE": "sqlite"}

# ---------------------------------------------------------------------------
# Fixture project — small, real, git-tracked. Mirrors the inline-project
# convention used throughout tests/integration/indexing/*.py (a `_write`
# helper + git init), since no single reusable static fixture project exists
# in this repo yet. Gives every analysis/diagram tool real material: an
# inheritance edge (User -> Base), internal + external imports, a resolved
# CALLS chain (handle_request -> User()/save/helper), one undocumented
# entity (MAX_RETRIES) for docstring-coverage, and dead code (to_json,
# unused_function have no inbound CALLS).
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str) -> None:
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _build_fixture_project(root: Path) -> None:
    _write(root, "src/__init__.py", "")
    _write(
        root,
        "src/models.py",
        "class Base:\n"
        '    """Base model."""\n'
        "\n"
        "\n"
        "class User(Base):\n"
        '    """A user account."""\n'
        "\n"
        "    def save(self) -> None:\n"
        '        """Persist the user."""\n',
    )
    _write(
        root,
        "src/utils.py",
        "import json\n"
        "\n"
        "MAX_RETRIES = 3\n"
        "\n"
        "\n"
        "def helper(x: int) -> int:\n"
        '    """A helper function."""\n'
        "    return x + 1\n"
        "\n"
        "\n"
        "def to_json(x: int) -> str:\n"
        '    """Serialize x to JSON."""\n'
        "    return json.dumps(x)\n"
        "\n"
        "\n"
        "def unused_function() -> None:\n"
        '    """Never called by anything — dead code fixture."""\n',
    )
    _write(
        root,
        "src/app.py",
        "from src.models import User\n"
        "from src.utils import helper\n"
        "\n"
        "\n"
        "def handle_request(x: int) -> int:\n"
        '    """Handle an incoming request end to end."""\n'
        "    u = User()\n"
        "    u.save()\n"
        "    return helper(x)\n",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "test@test.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "initial")


@pytest.fixture(scope="module")
def indexed_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Index a small real git-tracked project once via the real ``atlas index``
    CLI path against the SQLite fallback backend; shared read-only across
    every test in this module (each test that queries it opens its own fresh
    connection — see ``app_ctx``)."""
    root = tmp_path_factory.mktemp("embedded_mode")
    _build_fixture_project(root)

    result = runner.invoke(app, ["--json", "index", str(root), "--no-embed"], env=_ENV)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["entities_total"] > 0
    assert payload["mode"] == "full"
    return root


@pytest.fixture(scope="module")
def project_name(indexed_project: Path) -> str:
    return derive_project_name(indexed_project)


@pytest.fixture
def embedded_settings(indexed_project: Path) -> AtlasSettings:
    return AtlasSettings(
        project_root=indexed_project,
        backend=BackendSettings(graph="sqlite", queue="sqlite"),
        embeddings=EmbeddingSettings(enabled=False),
    )


@pytest.fixture
async def app_ctx(embedded_settings: AtlasSettings) -> AsyncIterator[AppContext]:
    """A fresh SqliteGraphClient connection to the already-indexed on-disk
    file, built through the real ``create_graph_client`` factory — the same
    "restart" a new ``atlas mcp`` process would perform. Tools are exercised
    via ``_invoke_tool`` (the real ``@mcp.tool`` bodies), never by calling
    SqliteGraphClient methods directly."""
    graph = await create_graph_client(embedded_settings)
    assert isinstance(graph, SqliteGraphClient)
    await graph.ping()
    try:
        yield AppContext(
            graph=graph,  # type: ignore[invalid-argument-type]
            settings=embedded_settings,
            embed=None,
            vector_enabled=False,
        )
    finally:
        await graph.close()


# ---------------------------------------------------------------------------
# CLI surface (real CliRunner invocations, mirrors tests/unit/test_cli.py)
# ---------------------------------------------------------------------------


class TestCliSurface:
    def test_atlas_status_reports_indexed_project(self, indexed_project: Path, project_name: str) -> None:
        """Live repro target from the plan's Phase 3.5 finding: 'atlas status'
        used to crash with an unhandled traceback on the sqlite backend."""
        env = {**_ENV, "ATLAS_PROJECT_ROOT": str(indexed_project)}
        result = runner.invoke(app, ["--json", "status"], env=env)
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        entry = next((p for p in payload["projects"] if p["name"] == project_name), None)
        assert entry is not None, payload
        assert entry["git_hash"]
        assert entry["file_count"]
        assert entry["file_count"] > 0

    def test_restart_delta_index_sees_no_changes(self, indexed_project: Path) -> None:
        """A second 'atlas index' run (a fresh SqliteGraphClient/SqliteEventBus
        constructed against the same on-disk .atlas/*.sqlite3 files, exactly
        like a real process restart) must see the git_hash the first run
        stored and do nothing — the core disk-cache-survives-restart
        requirement the original plan required. (entities_unchanged only
        counts entities from files the AST consumer actually reprocessed —
        zero changed files means it never runs, so it stays 0 on both
        backends; entities_total staying put is the real "nothing changed"
        signal, mirroring tests/integration/indexing/test_orchestrator.py's
        test_delta_index_preserves_unchanged.)"""
        baseline = runner.invoke(app, ["--json", "status"], env={**_ENV, "ATLAS_PROJECT_ROOT": str(indexed_project)})
        assert baseline.exit_code == 0, baseline.output
        baseline_entities = json.loads(baseline.output)["projects"][0]["entity_count"]

        result = runner.invoke(app, ["--json", "index", str(indexed_project), "--no-embed"], env=_ENV)
        assert result.exit_code == 0, result.output

        payload = json.loads(result.output)
        assert payload["mode"] == "delta"
        stats = payload["delta_stats"]
        assert stats is not None
        assert stats["files_added"] == 0
        assert stats["files_modified"] == 0
        assert stats["files_deleted"] == 0
        assert payload["entities_total"] == baseline_entities


# ---------------------------------------------------------------------------
# First-index readiness gate against a real, already-provisioned backend
# ---------------------------------------------------------------------------


class TestReadinessGate:
    async def test_restart_gate_opens_immediately(self, embedded_settings: AtlasSettings) -> None:
        """A fresh MCP lifespan entry against an already-indexed sqlite backend
        must never block: needs_first_index is computed from
        get_schema_version() BEFORE any indexing runs, and 'atlas index' (via
        the indexed_project fixture) already called ensure_schema()."""
        mcp = create_mcp_server(embedded_settings, catchup=False)
        lifespan = mcp.settings.lifespan
        assert lifespan is not None
        async with lifespan(mcp) as app_ctx:
            assert app_ctx.needs_first_index is False
            assert app_ctx.first_index_ready.is_set() is True
            assert isinstance(app_ctx.graph, SqliteGraphClient)


# ---------------------------------------------------------------------------
# Core lookup / navigation tools
# ---------------------------------------------------------------------------


class TestCoreTools:
    async def test_hybrid_search_finds_entity(self, app_ctx: AppContext) -> None:
        result = await _invoke_tool(app_ctx, "hybrid_search", query="helper")
        assert "error" not in result
        names = {r["name"] for r in result["results"]}
        assert "helper" in names

    async def test_get_node_finds_entity(self, app_ctx: AppContext) -> None:
        result = await _invoke_tool(app_ctx, "get_node", name="handle_request")
        assert result["count"] >= 1
        assert any(r["name"] == "handle_request" for r in result["results"])

    async def test_get_context_expands_neighborhood(self, app_ctx: AppContext) -> None:
        """Live repro target from the plan's Phase 3.5 finding: get_context
        used to raise directly on the sqlite backend."""
        found = await _invoke_tool(app_ctx, "get_node", name="User")
        uid = found["results"][0]["uid"]

        result = await _invoke_tool(app_ctx, "get_context", uid=uid)
        assert "error" not in result
        assert result["node"]["name"] == "User"
        assert result["parent"] is not None

    async def test_trace_path_finds_call_chain(self, app_ctx: AppContext) -> None:
        from_node = await _invoke_tool(app_ctx, "get_node", name="handle_request")
        to_node = await _invoke_tool(app_ctx, "get_node", name="helper")
        from_uid = from_node["results"][0]["uid"]
        to_uid = to_node["results"][0]["uid"]

        result = await _invoke_tool(app_ctx, "trace_path", from_uid=from_uid, to_uid=to_uid)
        assert "error" not in result
        assert result["found"] is True
        assert result["hop_count"] >= 1

    async def test_blast_radius_finds_callers(self, app_ctx: AppContext) -> None:
        found = await _invoke_tool(app_ctx, "get_node", name="helper")
        uid = found["results"][0]["uid"]

        result = await _invoke_tool(app_ctx, "blast_radius", uid=uid, direction="callers")
        assert "error" not in result
        names = {a["name"] for a in result["affected"]}
        assert "handle_request" in names

    async def test_list_projects_includes_indexed_project(self, app_ctx: AppContext, project_name: str) -> None:
        result = await _invoke_tool(app_ctx, "list_projects")
        names = {r["name"] for r in result["results"]}
        assert project_name in names

    async def test_index_status_reports_schema_and_entities(self, app_ctx: AppContext, project_name: str) -> None:
        result = await _invoke_tool(app_ctx, "index_status")
        assert result["schema_version"] > 0
        names = {p["name"] for p in result["projects"]}
        assert project_name in names


# ---------------------------------------------------------------------------
# find_* shortcut tools (ADR-0013)
# ---------------------------------------------------------------------------


class TestShortcutTools:
    async def test_find_dead_code(self, app_ctx: AppContext, project_name: str) -> None:
        result = await _invoke_tool(app_ctx, "find_dead_code", project=project_name)
        assert "error" not in result
        names = {c["name"] for c in result["dead_code"]}
        assert "unused_function" in names
        assert "helper" not in names  # helper IS called, by handle_request

    async def test_find_complexity_hotspots(self, app_ctx: AppContext, project_name: str) -> None:
        result = await _invoke_tool(app_ctx, "find_complexity_hotspots", project=project_name)
        assert "error" not in result
        assert len(result["hotspots"]) > 0

    async def test_find_hotspots_without_mining_is_sane_not_crashing(
        self, app_ctx: AppContext, project_name: str
    ) -> None:
        """'atlas mine-git-history' was never run against this fixture — a sane
        result here is mined=false with empty lists, not an error."""
        result = await _invoke_tool(app_ctx, "find_hotspots", project=project_name)
        assert "error" not in result
        assert result["mined"] is False
        assert result["hotspots"] == []


# ---------------------------------------------------------------------------
# analyze_repo — every sub-analysis except communities (documented non-goal
# on the embedded backend, covered separately in TestUnsupportedOnSqlite)
# ---------------------------------------------------------------------------


class TestAnalyzeRepo:
    @pytest.mark.parametrize("analysis", ["structure", "centrality", "dependencies", "patterns", "quality"])
    async def test_sub_analysis_returns_sane_result(
        self, app_ctx: AppContext, project_name: str, analysis: str
    ) -> None:
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis=analysis, project=project_name)
        assert "error" not in result, result
        assert result["analysis"] == analysis


# ---------------------------------------------------------------------------
# generate_diagram — every diagram type
# ---------------------------------------------------------------------------


class TestGenerateDiagram:
    @pytest.mark.parametrize("diagram_type", ["packages", "imports", "inheritance"])
    async def test_diagram_type_renders(self, app_ctx: AppContext, project_name: str, diagram_type: str) -> None:
        result = await _invoke_tool(app_ctx, "generate_diagram", type=diagram_type, project=project_name)
        assert "error" not in result, result
        assert result["type"] == diagram_type
        assert result["mermaid"]

    async def test_module_detail_diagram_renders(self, app_ctx: AppContext, project_name: str) -> None:
        result = await _invoke_tool(
            app_ctx, "generate_diagram", type="module_detail", project=project_name, path="src/models"
        )
        assert "error" not in result, result
        assert result["type"] == "module_detail"
        assert "User" in result["mermaid"]
        assert "Base" in result["mermaid"]


# ---------------------------------------------------------------------------
# Deliberate backend-specific exceptions — must fail cleanly, not crash
# ---------------------------------------------------------------------------


class TestUnsupportedOnSqlite:
    async def test_find_communities_returns_documented_unsupported_error(
        self, app_ctx: AppContext, project_name: str
    ) -> None:
        result = await _invoke_tool(app_ctx, "find_communities", project=project_name)
        assert result["analysis"] == "communities"
        assert "unsupported on the sqlite backend" in result["error"]

    async def test_analyze_repo_communities_returns_documented_unsupported_error(
        self, app_ctx: AppContext, project_name: str
    ) -> None:
        result = await _invoke_tool(app_ctx, "analyze_repo", analysis="communities", project=project_name)
        assert result["analysis"] == "communities"
        assert "unsupported on the sqlite backend" in result["error"]

    async def test_cypher_query_returns_clean_structured_error(self, app_ctx: AppContext) -> None:
        result = await _invoke_tool(app_ctx, "cypher_query", query="MATCH (n:Callable) RETURN n LIMIT 10")
        assert result["code"] == "UNSUPPORTED_BACKEND"

    async def test_validate_cypher_does_not_crash(self, app_ctx: AppContext) -> None:
        result = await _invoke_tool(app_ctx, "validate_cypher", query="MATCH (n:Callable) RETURN n LIMIT 10")
        assert "issues" in result
        assert any("skipped" in i["message"].lower() for i in result["issues"])


# ---------------------------------------------------------------------------
# health_check — backend-honesty fix (Phase 3.5 "also fold in")
# ---------------------------------------------------------------------------


class TestHealthCheck:
    async def test_health_check_names_the_active_backend(self, app_ctx: AppContext) -> None:
        result = await _invoke_tool(app_ctx, "health_check")
        memgraph_check = next(c for c in result["checks"] if c["name"] == "memgraph")
        assert "SQLite (embedded)" in memgraph_check["message"]
