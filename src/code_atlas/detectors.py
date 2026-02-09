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

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from loguru import logger

from code_atlas.parser import ParsedRelationship
from code_atlas.schema import CallableKind, NodeLabel, RelType

if TYPE_CHECKING:
    from code_atlas.graph import GraphClient
    from code_atlas.parser import ParsedEntity, ParsedFile


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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_STRING_RE = re.compile(r"""(['"])((?:(?!\1).)*)\1""")
_DEPENDS_RE = re.compile(r"Depends\(\s*([A-Za-z_]\w*)\s*\)")


def _parse_decorator_tag(tag: str) -> tuple[str, str]:
    """Split a decorator tag into (name, args_text).

    >>> _parse_decorator_tag("decorator:app.get('/users')")
    ("app.get", "'/users'")
    >>> _parse_decorator_tag("decorator:staticmethod")
    ("staticmethod", "")
    >>> _parse_decorator_tag("not_a_decorator")
    ("", "")
    """
    if not tag.startswith("decorator:"):
        return ("", "")
    body = tag[len("decorator:") :]
    paren = body.find("(")
    if paren < 0:
        return (body, "")
    name = body[:paren]
    args = body[paren + 1 :].rstrip(")")
    return (name, args)


def _extract_first_string_arg(text: str) -> str | None:
    """Extract the first string literal value from argument text.

    >>> _extract_first_string_arg("'/users/{id}', response_model=User")
    '/users/{id}'
    """
    match = _STRING_RE.search(text)
    return match.group(2) if match else None


def _extract_depends_names(text: str) -> list[str]:
    """Find all ``Depends(name)`` references in text (e.g. a signature).

    >>> _extract_depends_names("def f(db=Depends(get_db), cache=Depends(get_cache))")
    ['get_db', 'get_cache']
    """
    return _DEPENDS_RE.findall(text)


# ---------------------------------------------------------------------------
# Concrete detector implementations
# ---------------------------------------------------------------------------

# HTTP method suffixes recognized on route decorators
_ROUTE_SUFFIXES: frozenset[str] = frozenset(
    {".get", ".post", ".put", ".delete", ".patch", ".head", ".options", ".route", ".api_route"}
)

# Map decorator suffix to HTTP method
_SUFFIX_TO_METHOD: dict[str, str] = {
    ".get": "GET",
    ".post": "POST",
    ".put": "PUT",
    ".delete": "DELETE",
    ".patch": "PATCH",
    ".head": "HEAD",
    ".options": "OPTIONS",
    ".route": "ANY",
    ".api_route": "ANY",
}

# Known event-handler decorator names (suffix or full name)
_EVENT_PATTERNS: dict[str, str] = {
    "app.task": "celery",
    "shared_task": "celery",
    "celery.task": "celery",
    "receiver": "django",
    "dramatiq.actor": "dramatiq",
    "event_handler": "generic",
    "on_event": "generic",
}


class DecoratorRoutingDetector:
    """Detect HTTP route handlers from framework decorators."""

    @property
    def name(self) -> str:
        return "decorator_routing"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        enrichments: list[PropertyEnrichment] = []
        for entity in parsed.entities:
            for tag in entity.tags:
                dec_name, args_text = _parse_decorator_tag(tag)
                if not dec_name:
                    continue
                # Check if decorator ends with a route suffix
                for suffix, method in _SUFFIX_TO_METHOD.items():
                    if dec_name.endswith(suffix):
                        route_path = _extract_first_string_arg(args_text) if args_text else None
                        if route_path is None:
                            break
                        enrichments.append(
                            PropertyEnrichment(
                                qualified_name=entity.qualified_name,
                                properties={"route_path": route_path, "http_method": method},
                            )
                        )
                        break
        return DetectorResult(enrichments=enrichments)


class EventHandlerDetector:
    """Detect event/task handlers from framework decorators."""

    @property
    def name(self) -> str:
        return "event_handlers"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        enrichments: list[PropertyEnrichment] = []
        for entity in parsed.entities:
            for tag in entity.tags:
                dec_name, args_text = _parse_decorator_tag(tag)
                if not dec_name:
                    continue
                framework = _EVENT_PATTERNS.get(dec_name)
                if framework is None:
                    continue
                event_name = _extract_first_string_arg(args_text) if args_text else None
                if event_name is None:
                    # Celery tasks use the function name as the task name
                    event_name = entity.name
                enrichments.append(
                    PropertyEnrichment(
                        qualified_name=entity.qualified_name,
                        properties={"event_name": event_name, "event_framework": framework},
                    )
                )
        return DetectorResult(enrichments=enrichments)


class TestMappingDetector:
    """Map test classes/functions to their subjects via naming conventions."""

    @property
    def name(self) -> str:
        return "test_mapping"

    async def detect(self, parsed: ParsedFile, project_name: str, graph: GraphClient) -> DetectorResult:
        relationships: list[ParsedRelationship] = []
        for entity in parsed.entities:
            target_name = self._extract_target_name(entity)
            if target_name is None:
                continue
            # Look up target in graph
            target_uid = await self._find_target(graph, project_name, entity, target_name)
            if target_uid:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=entity.qualified_name,
                        rel_type=RelType.TESTS,
                        to_name=target_uid,
                    )
                )
        return DetectorResult(relationships=relationships)

    @staticmethod
    def _extract_target_name(entity: ParsedEntity) -> str | None:
        """Derive the subject name from a test entity name."""
        if entity.label == NodeLabel.TYPE_DEF and entity.name.startswith("Test"):
            return entity.name[4:] or None
        if entity.label == NodeLabel.CALLABLE and entity.name.startswith("test_"):
            return entity.name[5:] or None
        return None

    @staticmethod
    async def _find_target(graph: GraphClient, project_name: str, source: ParsedEntity, target_name: str) -> str | None:
        if graph is None:
            return None
        # TypeDef test → look for TypeDef; Callable test → look for Callable
        label = "TypeDef" if source.label == NodeLabel.TYPE_DEF else "Callable"
        records = await graph.execute(
            f"MATCH (n:{label} {{project_name: $p, name: $n}}) RETURN n.uid AS uid LIMIT 1",
            {"p": project_name, "n": target_name},
        )
        return records[0]["uid"] if records else None


class ClassOverridesDetector:
    """Detect method overrides by checking parent classes for same-name methods."""

    @property
    def name(self) -> str:
        return "class_overrides"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,
    ) -> DetectorResult:
        if graph is None:
            return _EMPTY_RESULT

        # Build class_qn → [base_names] map from INHERITS relationships
        class_bases: dict[str, list[str]] = {}
        for rel in parsed.relationships:
            if rel.rel_type == RelType.INHERITS:
                class_bases.setdefault(rel.from_qualified_name, []).append(rel.to_name)

        if not class_bases:
            return _EMPTY_RESULT

        relationships: list[ParsedRelationship] = []
        for entity in parsed.entities:
            if entity.kind not in (
                CallableKind.METHOD,
                CallableKind.CONSTRUCTOR,
                CallableKind.DESTRUCTOR,
                CallableKind.STATIC_METHOD,
                CallableKind.CLASS_METHOD,
            ):
                continue
            # Derive class qualified_name: strip ".method_name" from entity qn
            dot_pos = entity.qualified_name.rfind(".")
            if dot_pos < 0:
                continue
            class_qn = entity.qualified_name[:dot_pos]
            bases = class_bases.get(class_qn, [])
            if not bases:
                continue
            # Query graph for parent method
            records = await graph.execute(
                "MATCH (base:TypeDef)-[:DEFINES]->(m:Callable)"
                " WHERE base.name IN $bases AND m.name = $method"
                " RETURN m.uid AS uid LIMIT 1",
                {"bases": bases, "method": entity.name},
            )
            if records:
                relationships.append(
                    ParsedRelationship(
                        from_qualified_name=entity.qualified_name,
                        rel_type=RelType.OVERRIDES,
                        to_name=records[0]["uid"],
                    )
                )
        return DetectorResult(relationships=relationships)


class DIInjectionDetector:
    """Detect FastAPI Depends() injection patterns."""

    @property
    def name(self) -> str:
        return "di_injection"

    async def detect(self, parsed: ParsedFile, project_name: str, graph: GraphClient) -> DetectorResult:
        enrichments: list[PropertyEnrichment] = []
        relationships: list[ParsedRelationship] = []
        for entity in parsed.entities:
            if not entity.signature:
                continue
            dep_names = _extract_depends_names(entity.signature)
            if not dep_names:
                continue
            enrichments.append(
                PropertyEnrichment(
                    qualified_name=entity.qualified_name,
                    properties={"di_framework": "fastapi", "dependencies": dep_names},
                )
            )
            # Try to resolve provider UIDs in graph
            if graph is None:
                continue
            for dep_name in dep_names:
                records = await graph.execute(
                    "MATCH (n:Callable {project_name: $p, name: $n}) RETURN n.uid AS uid LIMIT 1",
                    {"p": project_name, "n": dep_name},
                )
                if records:
                    relationships.append(
                        ParsedRelationship(
                            from_qualified_name=records[0]["uid"],
                            rel_type=RelType.INJECTED_INTO,
                            to_name=entity.qualified_name,
                        )
                    )
        return DetectorResult(relationships=relationships, enrichments=enrichments)


class CLICommandDetector:
    """Detect CLI command handlers from click/typer decorators."""

    @property
    def name(self) -> str:
        return "cli_commands"

    async def detect(
        self,
        parsed: ParsedFile,
        project_name: str,  # noqa: ARG002
        graph: GraphClient,  # noqa: ARG002
    ) -> DetectorResult:
        enrichments: list[PropertyEnrichment] = []
        for entity in parsed.entities:
            for tag in entity.tags:
                dec_name, args_text = _parse_decorator_tag(tag)
                if not dec_name:
                    continue
                if not dec_name.endswith(".command"):
                    continue
                command_name = _extract_first_string_arg(args_text) if args_text else None
                if command_name is None:
                    command_name = entity.name
                # Heuristic: typer uses app.command(), click uses cli.command() or @click.command()
                framework = "typer" if "typer" in dec_name.lower() else "click"
                enrichments.append(
                    PropertyEnrichment(
                        qualified_name=entity.qualified_name,
                        properties={
                            "command_name": command_name,
                            "cli_framework": framework,
                        },
                    )
                )
                break  # One command decorator per entity is enough
        return DetectorResult(enrichments=enrichments)


# ---------------------------------------------------------------------------
# Auto-registration
# ---------------------------------------------------------------------------

register_detector(DecoratorRoutingDetector())
register_detector(EventHandlerDetector())
register_detector(TestMappingDetector())
register_detector(ClassOverridesDetector())
register_detector(DIInjectionDetector())
register_detector(CLICommandDetector())
