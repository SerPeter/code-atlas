"""Blast radius and trace path in the web UI (ATL-118).

The analysis functions are patched: they are the same ones the MCP tools call and are
covered by their own tests. What needs proving here is that the view does not lose the
distinctions those functions worked to produce — above all ``via``, which is the only
thing stopping a REFERENCES dependent from reading as a caller (ADR-0029).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from code_atlas.server.web.services import ImpactViewService

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend


class _Graph:
    async def close(self) -> None: ...


def _service() -> ImpactViewService:
    return ImpactViewService(cast("GraphBackend", _Graph()), "demo")


def _entry(uid: str, **kw: Any) -> dict[str, Any]:
    return {
        "uid": uid,
        "name": kw.get("name", uid.rsplit(":", 1)[-1]),
        "qualified_name": kw.get("qn", uid.rsplit(":", 1)[-1]),
        "label": kw.get("label", "Callable"),
        "file_path": kw.get("file_path", "app.py"),
        "min_depth": kw.get("depth", 1),
        "via": kw.get("via", ["CALLS"]),
        "via_lines": kw.get("via_lines", []),
        "ambiguous_only": kw.get("ambiguous_only", False),
        "test_only": kw.get("test_only", False),
        "confidence_score": kw.get("confidence_score", 1.0),
    }


def _patch_blast(monkeypatch, affected: list[dict[str, Any]], *, total: int | None = None) -> dict[str, Any]:
    """Patch the analysis and record the kwargs it was called with."""
    seen: dict[str, Any] = {}

    async def _fake(graph, uid, *, direction, max_depth, edge_types, limit, test_patterns):
        seen.update(uid=uid, direction=direction, max_depth=max_depth, limit=limit, edge_types=edge_types)
        return {
            "uid": uid,
            "direction": direction,
            "max_depth": max_depth,
            "affected_count": total if total is not None else len(affected),
            "affected": affected[:limit],
        }

    monkeypatch.setattr("code_atlas.server.analysis.blast_radius", _fake)
    return seen


class TestBlastRadiusView:
    async def test_via_survives_into_the_view(self, monkeypatch):
        """A dependent found through REFERENCES must never read as a caller."""
        _patch_blast(
            monkeypatch,
            [
                _entry("u:caller", via=["CALLS"]),
                _entry("u:referrer", via=["REFERENCES"]),
                _entry("u:subclass", via=["INHERITS", "USES_TYPE"]),
            ],
        )

        view = await _service().blast("u:target")

        by_uid = {e.uid: e for group in view.groups for e in group.entities}
        assert by_uid["u:referrer"].via == ("REFERENCES",)
        assert not by_uid["u:referrer"].is_call, "REFERENCES is not a call"
        assert by_uid["u:caller"].is_call
        assert by_uid["u:subclass"].via == ("INHERITS", "USES_TYPE")

    async def test_results_are_grouped_by_distance_nearest_first(self, monkeypatch):
        _patch_blast(
            monkeypatch,
            [
                _entry("u:near", depth=1),
                _entry("u:far", depth=3),
                _entry("u:mid", depth=2),
            ],
        )

        view = await _service().blast("u:target")

        assert [g.depth for g in view.groups] == [1, 2, 3]
        assert [e.uid for g in view.groups for e in g.entities] == ["u:near", "u:mid", "u:far"]

    async def test_the_traversal_is_delegated_not_reimplemented(self, monkeypatch):
        """Constraint: call the same analysis the MCP tool calls, or the two drift."""
        seen = _patch_blast(monkeypatch, [])

        await _service().blast("u:x", direction="callees", max_depth=5)

        assert seen["uid"] == "u:x"
        assert seen["direction"] == "callees"
        assert seen["max_depth"] == 5
        assert "DEFINES" not in seen["edge_types"], "containment is excluded by ADR-0029"
        assert "CONTAINS" not in seen["edge_types"]

    async def test_the_reported_total_is_the_traversals_own(self, monkeypatch):
        """Unlike a search, this one genuinely knows — the closure is computed, then sliced."""
        _patch_blast(monkeypatch, [_entry(f"u:{i}") for i in range(10)], total=137)

        view = await _service().blast("u:target", limit=5)

        assert view.affected_count == 137
        assert view.shown == 5
        assert view.truncated is True
        assert view.remedy

    async def test_a_complete_result_is_not_marked_truncated(self, monkeypatch):
        _patch_blast(monkeypatch, [_entry("u:a"), _entry("u:b")])

        view = await _service().blast("u:target", limit=50)

        assert view.truncated is False
        assert view.shown == 2

    async def test_a_missing_entity_is_reported_not_rendered_as_empty(self, monkeypatch):
        async def _missing(graph, uid, **kwargs):
            return {"error": f"Node not found: {uid}", "code": "NOT_FOUND"}

        monkeypatch.setattr("code_atlas.server.analysis.blast_radius", _missing)

        view = await _service().blast("u:nope")

        assert not view.is_found
        assert "not found" in view.error.lower()
        assert view.groups == ()


class TestResolvedOnly:
    """The toggle hides entities no fully-resolved path reaches."""

    async def test_guessed_entities_are_dropped(self, monkeypatch):
        _patch_blast(
            monkeypatch,
            [
                _entry("u:solid", ambiguous_only=False),
                _entry("u:guessed", ambiguous_only=True),
            ],
        )

        view = await _service().blast("u:target", resolved_only=True)

        assert [e.uid for g in view.groups for e in g.entities] == ["u:solid"]

    async def test_the_default_shows_everything(self, monkeypatch):
        _patch_blast(
            monkeypatch,
            [_entry("u:solid"), _entry("u:guessed", ambiguous_only=True)],
        )

        view = await _service().blast("u:target")

        assert len({e.uid for g in view.groups for e in g.entities}) == 2

    async def test_the_filter_runs_over_the_considered_set_not_one_page(self, monkeypatch):
        """Filtering a page and paging a filtered set give different answers.

        With 60 guessed entities ahead of 5 solid ones, filtering a 50-row page would
        return nothing; filtering first returns the 5.
        """
        affected = [_entry(f"u:guess{i}", ambiguous_only=True) for i in range(60)]
        affected += [_entry(f"u:solid{i}") for i in range(5)]
        _patch_blast(monkeypatch, affected)

        view = await _service().blast("u:target", limit=50, resolved_only=True)

        shown = [e.uid for g in view.groups for e in g.entities]
        assert len(shown) == 5
        assert all(uid.startswith("u:solid") for uid in shown)

    async def test_the_caveat_refuses_to_call_a_filtered_list_verified(self, monkeypatch):
        """`ambiguous_only` is a heuristic — an edge with no confidence counts as unresolved."""
        _patch_blast(monkeypatch, [_entry("u:a"), _entry("u:b", ambiguous_only=True)])

        view = await _service().blast("u:target", resolved_only=True)

        assert "heuristic" in view.caveat.note
        assert "verified" in view.caveat.note


class TestTracePath:
    @staticmethod
    def _patch(monkeypatch, result: dict[str, Any]) -> None:
        async def _fake(graph, from_uid, to_uid, max_depth=6):
            return result

        monkeypatch.setattr("code_atlas.server.analysis.trace_path", _fake)

    async def test_hops_carry_their_edge_type_and_line(self, monkeypatch):
        self._patch(
            monkeypatch,
            {
                "found": True,
                "hop_count": 2,
                "path_weight": 0.5,
                "hops": [
                    {
                        "from": {"uid": "u:a", "name": "a"},
                        "to": {"uid": "u:b", "name": "b"},
                        "edge_type": "CALLS",
                        "confidence": "resolved",
                        "strategy": "import",
                        "at_line": 42,
                    },
                    {
                        "from": {"uid": "u:b", "name": "b"},
                        "to": {"uid": "u:c", "name": "c"},
                        "edge_type": "IMPORTS",
                    },
                ],
            },
        )

        view = await _service().trace("u:a", "u:c")

        assert view.found
        assert [h.edge_type for h in view.hops] == ["CALLS", "IMPORTS"]
        assert view.hops[0].at_line == 42
        assert view.hops[1].is_structural, "an edge with no confidence is a fact, not a guess"

    async def test_a_guessed_hop_taints_the_whole_path(self, monkeypatch):
        """A path is only as trustworthy as its weakest hop."""
        self._patch(
            monkeypatch,
            {
                "found": True,
                "hop_count": 2,
                "hops": [
                    {
                        "from": {"uid": "u:a", "name": "a"},
                        "to": {"uid": "u:b", "name": "b"},
                        "edge_type": "CALLS",
                        "confidence": "resolved",
                    },
                    {
                        "from": {"uid": "u:b", "name": "b"},
                        "to": {"uid": "u:c", "name": "c"},
                        "edge_type": "CALLS",
                        "confidence": "ambiguous",
                    },
                ],
            },
        )

        view = await _service().trace("u:a", "u:c")

        assert view.has_guessed_hop
        assert view.hops[1].is_guess
        assert not view.hops[0].is_guess

    async def test_no_path_is_not_an_error(self, monkeypatch):
        self._patch(monkeypatch, {"found": False, "message": "No path found within 6 hops"})

        view = await _service().trace("u:a", "u:z")

        assert not view.found
        assert not view.error
        assert "No path" in view.message

    async def test_a_missing_endpoint_is_an_error(self, monkeypatch):
        self._patch(monkeypatch, {"error": "Node not found: u:z", "code": "NOT_FOUND"})

        view = await _service().trace("u:a", "u:z")

        assert not view.found
        assert "not found" in view.error.lower()


class TestImpactEndpoint:
    def test_the_bare_page_asks_for_an_entity(self):
        pytest.importorskip("litestar")
        from litestar.testing import TestClient

        from code_atlas.server.web.app import create_app

        with TestClient(app=create_app(cast("GraphBackend", _Graph()), "demo")) as client:
            response = client.get("/impact")

        assert response.status_code == 200
        assert "uid" in response.text

    def test_the_page_labels_each_hit_with_the_edge_that_reached_it(self, monkeypatch):
        pytest.importorskip("litestar")
        from litestar.testing import TestClient

        from code_atlas.server.web.app import create_app

        _patch_blast(monkeypatch, [_entry("u:ref", qn="app.referrer", via=["REFERENCES"])])

        with TestClient(app=create_app(cast("GraphBackend", _Graph()), "demo")) as client:
            body = client.get("/impact?uid=u:target").text

        assert "REFERENCES" in body, "the edge type must be on the page, not just in the model"
        assert "not a call" in body
        assert "app.referrer" in body

    def test_depth_is_clamped_at_the_edge(self, monkeypatch):
        pytest.importorskip("litestar")
        from litestar.testing import TestClient

        from code_atlas.server.web.app import create_app

        seen = _patch_blast(monkeypatch, [])

        with TestClient(app=create_app(cast("GraphBackend", _Graph()), "demo")) as client:
            client.get("/impact?uid=u:x&depth=99&direction=sideways")

        assert seen["max_depth"] == 10, "an unbounded depth is a DoS against the reader's own machine"
        assert seen["direction"] == "callers", "an unknown direction falls back rather than erroring"

    def test_the_api_returns_the_same_view_model(self, monkeypatch):
        pytest.importorskip("litestar")
        from litestar.testing import TestClient

        from code_atlas.server.web.app import create_app

        _patch_blast(monkeypatch, [_entry("u:a", via=["USES_TYPE"])], total=9)

        with TestClient(app=create_app(cast("GraphBackend", _Graph()), "demo")) as client:
            payload = client.get("/impact/api/blast?uid=u:target").json()

        assert payload["affected_count"] == 9
        assert payload["groups"][0]["entities"][0]["via"] == ["USES_TYPE"]
