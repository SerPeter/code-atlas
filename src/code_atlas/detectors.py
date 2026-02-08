"""Pluggable pattern detector framework.

Detectors discover implicit semantic patterns (route handlers, test-to-code
mappings, method overrides, etc.) that the structural parser misses.  Each
detector receives a ParsedFile and optional GraphClient for cross-file
resolution, returning relationships and/or property enrichments.

The framework supports two output types:
  - **Relationships**: Cross-file edges (OVERRIDES, TESTS) added to the graph.
  - **Property enrichments**: Extra properties SET on existing entity nodes
    (route_path, http_method, etc.).

Detectors self-register at import time via ``register_detector()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

if TYPE_CHECKING:
    from code_atlas.graph import GraphClient
    from code_atlas.parser import ParsedFile, ParsedRelationship


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PropertyEnrichment:
    """Extra properties to SET on an existing entity node."""

    qualified_name: str  # Target entity UID (project:qn)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorResult:
    """Merged output of detector(s) for one file."""

    relationships: list[ParsedRelationship] = field(default_factory=list)
    enrichments: list[PropertyEnrichment] = field(default_factory=list)


_EMPTY_RESULT = DetectorResult()


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Detector(Protocol):
    """Interface that all pattern detectors must satisfy."""

    @property
    def name(self) -> str: ...

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,
        graph: GraphClient,
    ) -> DetectorResult: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Detector] = {}


def register_detector(detector: Detector) -> None:
    """Register a detector instance by name. Raises on duplicate."""
    if detector.name in _REGISTRY:
        msg = f"Detector already registered: {detector.name!r}"
        raise ValueError(msg)
    _REGISTRY[detector.name] = detector
    logger.debug("Registered detector: {}", detector.name)


def get_enabled_detectors(enabled: list[str]) -> list[Detector]:
    """Return detector instances for the given enabled names.

    Unknown names are logged and skipped — this allows settings to reference
    detectors that haven't been implemented yet without crashing.
    """
    detectors: list[Detector] = []
    for name in enabled:
        detector = _REGISTRY.get(name)
        if detector is None:
            logger.debug("Detector {!r} not found in registry, skipping", name)
        else:
            detectors.append(detector)
    return detectors


async def run_detectors(
    detectors: list[Detector],
    parsed: ParsedFile,
    project_name: str,
    graph: GraphClient,
) -> DetectorResult:
    """Run all detectors and merge their outputs into a single result.

    Individual detector failures are logged and skipped — one broken
    detector should not block the rest of the pipeline.
    """
    if not detectors:
        return _EMPTY_RESULT

    all_relationships: list[ParsedRelationship] = []
    all_enrichments: list[PropertyEnrichment] = []

    for detector in detectors:
        try:
            result = await detector.detect(parsed, project_name, graph)
            all_relationships.extend(result.relationships)
            all_enrichments.extend(result.enrichments)
        except Exception:
            logger.exception("Detector {!r} failed on {}", detector.name, parsed.file_path)

    return DetectorResult(
        relationships=all_relationships,
        enrichments=all_enrichments,
    )
