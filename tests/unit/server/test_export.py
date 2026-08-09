"""Static HTML export (ATL-120).

The export's whole justification is that it is a *second renderer over the same
components*, so the tests that matter are the ones that would catch it becoming a second
implementation — and the ones that would catch it reaching the network, since a
`file://` document has an empty hostname and no origin check protects it.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

from code_atlas.server.analysis import ModuleGraph
from code_atlas.server.web.export import StaticExporter, export_project
from code_atlas.server.web.services import ProjectNotIndexedError

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend


class _Graph:
    """The reads the three exported views make, and nothing else."""

    def __init__(self, *, projects: list[dict[str, Any]] | None = None):
        self._projects = (
            projects
            if projects is not None
            else [
                {
                    "name": "demo",
                    "entity_count": 42,
                    "last_indexed_at": "2026-08-08T00:00:00Z",
                    "git_hash": "abc123def4567",
                }
            ]
        )

    async def get_project_status(self, project_name: str | None = None) -> list[dict[str, Any]]:
        # `[{"n": <node>}]` with the node's own property names — the real backend shape.
        return [{"n": p} for p in self._projects]

    async def get_structure_overview(self, project: str, path: str, limit: int) -> dict[str, list[dict[str, Any]]]:
        return {"counts": [{"label": "Module", "kind": "module", "cnt": 3}], "largest_modules": [], "packages": []}

    async def get_module_import_edges(self, project: str, path: str) -> dict[str, list[dict[str, Any]]]:
        return {
            "direct": [
                {"from_mod": "app.api", "to_mod": "app.service"},
                {"from_mod": "app.service", "to_mod": "app.repo"},
            ],
            "indirect": [],
        }

    async def close(self) -> None: ...


def _patch_map(monkeypatch, *, modules: tuple[str, ...] = ("app.api", "app.service", "app.repo")) -> None:
    async def _fake_graph(graph, project, path, *, test_patterns=()):
        return ModuleGraph(
            modules={
                m: {"uid": f"u:{m}", "name": m.rsplit(".", 1)[-1], "qn": m, "file_path": f"{m}.py"} for m in modules
            },
            edges={(modules[0], modules[1]): 3.0} if len(modules) > 1 else {},
            directed={(modules[0], modules[1]): 3.0} if len(modules) > 1 else {},
            partition=[list(modules)],
        )

    async def _fake_external(graph, project):
        return []

    monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _fake_graph)
    monkeypatch.setattr("code_atlas.server.analysis.fetch_first_hop_external", _fake_external)


def _exporter(graph: _Graph | None = None) -> StaticExporter:
    return StaticExporter(cast("GraphBackend", graph or _Graph()), "demo")


class TestSelfContained:
    """ "Self-contained" has to mean it, or the file is a trap on an offline machine."""

    async def test_no_external_references_survive(self, monkeypatch):
        _patch_map(monkeypatch)

        html = await _exporter().render()

        # `src=`/`href=` attributes pointing anywhere would break offline. The inlined
        # map.js contains `location.href` assignments, which are not document loads.
        external = re.findall(r'<[a-z][^>]*\s(?:src|href)\s*=\s*"([^"]*)"', html)
        assert all(ref.startswith("#") for ref in external), f"non-anchor references: {external}"

    async def test_nothing_points_at_the_network(self, monkeypatch):
        """A file:// origin has an empty hostname, so no localhost guard protects it."""
        _patch_map(monkeypatch)

        html = await _exporter().render()

        for scheme in ("http://", "https://", "//cdn", "ws://"):
            assert scheme not in html, f"{scheme} appears in a supposedly offline document"

    async def test_the_renderer_is_inlined_not_linked(self, monkeypatch):
        _patch_map(monkeypatch)

        html = await _exporter().render()

        assert "/static/" not in html
        assert "data-atlas-canvas" in html, "the canvas renderer itself must be in the file"
        assert "data:font/woff2;base64," in html, "the fonts ride along as data URIs"

    async def test_the_map_payload_is_embedded_so_no_fetch_runs(self, monkeypatch):
        _patch_map(monkeypatch)

        html = await _exporter().render()

        assert "window.ATLAS_EMBED = " in html
        assert "app.service" in html, "the module data must be in the document"


class TestSameComponents:
    """A parallel implementation is the failure mode this story exists to avoid."""

    async def test_the_export_renders_through_the_same_island_the_server_serves(self, monkeypatch):
        """The inlined map.js is byte-identical to the served file — one renderer."""
        _patch_map(monkeypatch)

        from code_atlas.server.web import STATIC_DIR

        html = await _exporter().render()

        assert 'id="map-aside"' in html
        assert 'id="map-main"' in html
        assert (STATIC_DIR / "map.js").read_text(encoding="utf-8") in html

    async def test_a_missing_template_variable_fails_loudly(self, monkeypatch):
        """StrictUndefined on purpose: an export is written once and mailed to someone.

        The server can survive a blank cell on a page that gets reloaded; a silently
        empty section would ship.
        """
        _patch_map(monkeypatch)
        env = _exporter()._environment()

        from jinja2 import UndefinedError

        template = env.from_string("{{ never_provided.attribute }}")
        with pytest.raises(UndefinedError):
            template.render()


class TestProvenance:
    """A snapshot that does not say when it was taken looks current forever."""

    async def test_project_commit_and_generation_time_are_stated(self, monkeypatch):
        _patch_map(monkeypatch)

        html = await _exporter().render(generated_at=datetime(2026, 8, 8, 12, 30, tzinfo=UTC))

        assert "demo" in html
        assert "abc123d" in html, "the commit must be on the page"
        assert "2026-08-08 12:30:00 UTC" in html
        assert "static export" in html

    async def test_what_the_export_cannot_do_explains_itself(self, monkeypatch):
        """Filters and the entity level need the server; the fallback says so rather
        than rendering an empty canvas."""
        _patch_map(monkeypatch)

        html = await _exporter().render()

        assert "need the live server" in html


class TestScriptEmbedding:
    async def test_a_script_close_tag_in_the_data_cannot_break_the_document(self, monkeypatch):
        """The HTML parser sees the raw text before any JSON parser does."""
        _patch_map(monkeypatch, modules=("app.</script><img>evil", "app.other"))

        html = await _exporter().render()

        blob = html.split("window.ATLAS_EMBED = ")[1].split("</script>")[0]
        assert "</script" not in blob
        assert "\\u003c" in blob

    async def test_the_embedded_payload_is_valid_json(self, monkeypatch):
        import json

        _patch_map(monkeypatch)

        html = await _exporter().render()
        blob = html.split("window.ATLAS_EMBED = ")[1].split("</script>")[0].rstrip().rstrip(";")

        data = json.loads(blob)
        assert {n["id"] for n in data["nodes"]} == {"app.api", "app.service", "app.repo"}


class TestWriting:
    async def test_it_writes_a_file_and_reports_what_it_covers(self, monkeypatch, tmp_path):
        _patch_map(monkeypatch)
        target = tmp_path / "nested" / "snapshot.html"

        result = await export_project(cast("GraphBackend", _Graph()), "demo", target)

        assert target.exists()
        assert result.node_count == 3
        assert result.map_available
        assert result.bytes_written == len(target.read_text(encoding="utf-8").encode("utf-8"))

    async def test_an_unindexed_project_refuses_rather_than_writing_zeroes(self, monkeypatch, tmp_path):
        """A file of zeroes is indistinguishable from a project that really is empty."""
        _patch_map(monkeypatch)
        graph = _Graph(projects=[{"name": "other", "entity_count": 1}])
        target = tmp_path / "snapshot.html"

        with pytest.raises(ProjectNotIndexedError):
            await export_project(cast("GraphBackend", graph), "demo", target)

        assert not target.exists(), "nothing may be written when the project is not indexed"

    async def test_the_document_is_rendered_once(self, monkeypatch, tmp_path):
        """Re-querying to build the summary could describe a different graph than the file."""
        _patch_map(monkeypatch)
        calls = 0
        real = __import__("code_atlas.server.analysis", fromlist=["build_module_graph"]).build_module_graph

        async def _counting(*args, **kwargs):
            nonlocal calls
            calls += 1
            return await real(*args, **kwargs)

        monkeypatch.setattr("code_atlas.server.analysis.build_module_graph", _counting)

        await export_project(cast("GraphBackend", _Graph()), "demo", tmp_path / "s.html")

        assert calls == 1, f"the module graph was built {calls} times"
