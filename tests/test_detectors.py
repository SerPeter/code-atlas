"""Tests for the detector framework (registry, run_detectors, output merging)."""

from __future__ import annotations

import pytest

from code_atlas.detectors import (
    DetectorResult,
    PropertyEnrichment,
    get_enabled_detectors,
    register_detector,
    run_detectors,
)
from code_atlas.parser import ParsedEntity, ParsedFile, ParsedRelationship
from code_atlas.schema import CallableKind, NodeLabel, RelType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeDetector:
    """A concrete detector for testing."""

    def __init__(self, name: str, result: DetectorResult | None = None) -> None:
        self._name = name
        self._result = result or DetectorResult()

    @property
    def name(self) -> str:
        return self._name

    async def detect(self, parsed, project_name, graph):
        return self._result


class FailingDetector:
    """A detector that always raises."""

    @property
    def name(self) -> str:
        return "failing"

    async def detect(self, parsed, project_name, graph):
        msg = "boom"
        raise RuntimeError(msg)


def _make_parsed_file() -> ParsedFile:
    """Create a minimal ParsedFile for testing."""
    return ParsedFile(
        file_path="src/app.py",
        language="python",
        entities=[
            ParsedEntity(
                name="handler",
                qualified_name="proj:src.app.handler",
                label=NodeLabel.CALLABLE,
                kind=CallableKind.FUNCTION,
                line_start=1,
                line_end=5,
                file_path="src/app.py",
            ),
        ],
        relationships=[],
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_register_and_get(monkeypatch):
    """register_detector adds to registry; get_enabled_detectors retrieves by name."""
    monkeypatch.setattr("code_atlas.detectors._REGISTRY", {})

    det_a = FakeDetector("alpha")
    det_b = FakeDetector("beta")
    register_detector(det_a)
    register_detector(det_b)

    result = get_enabled_detectors(["alpha", "beta"])
    assert len(result) == 2
    assert result[0] is det_a
    assert result[1] is det_b


def test_register_duplicate_raises(monkeypatch):
    """Registering a detector with the same name raises ValueError."""
    monkeypatch.setattr("code_atlas.detectors._REGISTRY", {})

    register_detector(FakeDetector("dup"))
    with pytest.raises(ValueError, match="already registered"):
        register_detector(FakeDetector("dup"))


def test_get_enabled_skips_unknown(monkeypatch):
    """Unknown detector names are skipped, not errors."""
    monkeypatch.setattr("code_atlas.detectors._REGISTRY", {})

    det = FakeDetector("known")
    register_detector(det)

    result = get_enabled_detectors(["known", "unknown_future_detector"])
    assert len(result) == 1
    assert result[0] is det


def test_get_enabled_empty_registry(monkeypatch):
    """Empty enabled list returns empty list."""
    monkeypatch.setattr("code_atlas.detectors._REGISTRY", {})
    assert get_enabled_detectors([]) == []


# ---------------------------------------------------------------------------
# run_detectors tests
# ---------------------------------------------------------------------------


async def test_run_detectors_empty_list():
    """No detectors -> empty result."""
    parsed = _make_parsed_file()
    result = await run_detectors([], parsed, "proj", None)  # type: ignore[arg-type]
    assert result.relationships == []
    assert result.enrichments == []


async def test_run_detectors_merges_results():
    """Multiple detectors' outputs are merged into a single DetectorResult."""
    rel = ParsedRelationship(
        from_qualified_name="proj:src.app.Child.method",
        rel_type=RelType.OVERRIDES,
        to_name="proj:src.base.Base.method",
    )
    enrichment = PropertyEnrichment(
        qualified_name="proj:src.app.handler",
        properties={"route_path": "/users", "http_method": "GET"},
    )

    det_a = FakeDetector("det_a", DetectorResult(relationships=[rel]))
    det_b = FakeDetector("det_b", DetectorResult(enrichments=[enrichment]))

    parsed = _make_parsed_file()
    result = await run_detectors([det_a, det_b], parsed, "proj", None)  # type: ignore[arg-type]

    assert len(result.relationships) == 1
    assert result.relationships[0].rel_type == RelType.OVERRIDES
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["route_path"] == "/users"


async def test_run_detectors_isolates_failures():
    """A failing detector is skipped; other detectors still run."""
    enrichment = PropertyEnrichment(
        qualified_name="proj:src.app.handler",
        properties={"command_name": "deploy"},
    )
    good_det = FakeDetector("good", DetectorResult(enrichments=[enrichment]))
    bad_det = FailingDetector()

    parsed = _make_parsed_file()
    result = await run_detectors([bad_det, good_det], parsed, "proj", None)  # type: ignore[arg-type]

    # Good detector's output is present despite bad detector raising
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["command_name"] == "deploy"
    assert result.relationships == []


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


def test_detector_result_defaults():
    """DetectorResult has empty lists by default."""
    result = DetectorResult()
    assert result.relationships == []
    assert result.enrichments == []


def test_property_enrichment_frozen():
    """PropertyEnrichment is immutable."""
    enrichment = PropertyEnrichment(qualified_name="proj:x", properties={"a": 1})
    with pytest.raises(AttributeError):
        enrichment.qualified_name = "proj:y"  # type: ignore[misc]
