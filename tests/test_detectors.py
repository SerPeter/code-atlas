"""Tests for the detector framework (registry, run_detectors, output merging)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from code_atlas.detectors import (
    ClassOverridesDetector,
    CLICommandDetector,
    DecoratorRoutingDetector,
    DetectorResult,
    DIInjectionDetector,
    EventHandlerDetector,
    PropertyEnrichment,
    TestMappingDetector,
    _extract_depends_names,
    _extract_first_string_arg,
    _parse_decorator_tag,
    get_enabled_detectors,
    register_detector,
    run_detectors,
)
from code_atlas.parser import ParsedEntity, ParsedFile, ParsedRelationship
from code_atlas.schema import CallableKind, NodeLabel, RelType, TypeDefKind

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


# ---------------------------------------------------------------------------
# Helper utility tests
# ---------------------------------------------------------------------------


class TestParseDecoratorTag:
    def test_with_args(self):
        name, args = _parse_decorator_tag("decorator:app.get('/users')")
        assert name == "app.get"
        assert args == "'/users'"

    def test_without_args(self):
        name, args = _parse_decorator_tag("decorator:staticmethod")
        assert name == "staticmethod"
        assert args == ""

    def test_not_a_decorator_tag(self):
        assert _parse_decorator_tag("other:foo") == ("", "")

    def test_complex_args(self):
        name, args = _parse_decorator_tag("decorator:app.get('/users/{id}', response_model=User)")
        assert name == "app.get"
        assert args == "'/users/{id}', response_model=User"


class TestExtractFirstStringArg:
    def test_single_quoted(self):
        assert _extract_first_string_arg("'/users/{id}'") == "/users/{id}"

    def test_double_quoted(self):
        assert _extract_first_string_arg('"/api/v1"') == "/api/v1"

    def test_with_extra_args(self):
        assert _extract_first_string_arg("'/path', response_model=User") == "/path"

    def test_no_string(self):
        assert _extract_first_string_arg("response_model=User") is None

    def test_empty(self):
        assert _extract_first_string_arg("") is None


class TestExtractDependsNames:
    def test_single_depends(self):
        assert _extract_depends_names("def f(db=Depends(get_db))") == ["get_db"]

    def test_multiple_depends(self):
        result = _extract_depends_names("def f(db=Depends(get_db), cache=Depends(get_cache))")
        assert result == ["get_db", "get_cache"]

    def test_no_depends(self):
        assert _extract_depends_names("def f(x: int, y: str)") == []

    def test_depends_with_spaces(self):
        assert _extract_depends_names("Depends( get_db )") == ["get_db"]


# ---------------------------------------------------------------------------
# Concrete detector tests — enrichment-only (no graph needed)
# ---------------------------------------------------------------------------


def _make_entity(
    name: str = "handler",
    qn: str = "proj:src.app.handler",
    label: NodeLabel = NodeLabel.CALLABLE,
    kind: str = CallableKind.FUNCTION,
    tags: list[str] | None = None,
    signature: str | None = None,
) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=qn,
        label=label,
        kind=kind,
        line_start=1,
        line_end=5,
        file_path="src/app.py",
        tags=tags or [],
        signature=signature,
    )


def _make_parsed(
    entities: list[ParsedEntity] | None = None,
    relationships: list[ParsedRelationship] | None = None,
) -> ParsedFile:
    return ParsedFile(
        file_path="src/app.py",
        language="python",
        entities=entities or [],
        relationships=relationships or [],
    )


async def test_routing_fastapi_get():
    """Entity with app.get('/users') tag → enrichment with route_path and http_method."""
    entity = _make_entity(tags=["decorator:app.get('/users')"])
    parsed = _make_parsed(entities=[entity])
    det = DecoratorRoutingDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["route_path"] == "/users"
    assert result.enrichments[0].properties["http_method"] == "GET"


async def test_routing_post():
    """POST route detected correctly."""
    entity = _make_entity(tags=['decorator:router.post("/items")'])
    parsed = _make_parsed(entities=[entity])
    det = DecoratorRoutingDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["http_method"] == "POST"


async def test_routing_missing_path():
    """Decorator without a string arg → no enrichment."""
    entity = _make_entity(tags=["decorator:app.get()"])
    parsed = _make_parsed(entities=[entity])
    det = DecoratorRoutingDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert result.enrichments == []


async def test_routing_non_route_decorator():
    """Non-route decorator → no enrichment."""
    entity = _make_entity(tags=["decorator:app.middleware('http')"])
    parsed = _make_parsed(entities=[entity])
    det = DecoratorRoutingDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert result.enrichments == []


async def test_event_handler_celery_task():
    """Celery @app.task decorator → event enrichment."""
    entity = _make_entity(name="send_email", tags=["decorator:app.task"])
    parsed = _make_parsed(entities=[entity])
    det = EventHandlerDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["event_framework"] == "celery"
    assert result.enrichments[0].properties["event_name"] == "send_email"


async def test_event_handler_django_receiver():
    """Django receiver with signal name → event enrichment."""
    entity = _make_entity(tags=["decorator:receiver('post_save')"])
    parsed = _make_parsed(entities=[entity])
    det = EventHandlerDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["event_framework"] == "django"
    assert result.enrichments[0].properties["event_name"] == "post_save"


async def test_cli_command_typer():
    """Typer @app.command('build') → command enrichment."""
    entity = _make_entity(tags=["decorator:app.command('build')"])
    parsed = _make_parsed(entities=[entity])
    det = CLICommandDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["command_name"] == "build"
    assert result.enrichments[0].properties["cli_framework"] == "click"


async def test_cli_command_no_arg_uses_function_name():
    """@app.command() without string arg → function name as command_name."""
    entity = _make_entity(name="deploy", tags=["decorator:app.command()"])
    parsed = _make_parsed(entities=[entity])
    det = CLICommandDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["command_name"] == "deploy"


async def test_cli_command_typer_framework():
    """@typer_app.command() → cli_framework = 'click' (no 'typer' in name)."""
    entity = _make_entity(tags=["decorator:typer.command('run')"])
    parsed = _make_parsed(entities=[entity])
    det = CLICommandDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert result.enrichments[0].properties["cli_framework"] == "typer"


async def test_di_injection_depends():
    """Signature with Depends(get_db) → enrichment with dependencies list."""
    entity = _make_entity(signature="def handler(db=Depends(get_db))")
    parsed = _make_parsed(entities=[entity])
    det = DIInjectionDetector()
    result = await det.detect(parsed, "proj", None)  # type: ignore[arg-type]
    assert len(result.enrichments) == 1
    assert result.enrichments[0].properties["di_framework"] == "fastapi"
    assert result.enrichments[0].properties["dependencies"] == ["get_db"]
    # No graph → no relationships
    assert result.relationships == []


# ---------------------------------------------------------------------------
# Graph-dependent detector tests (using AsyncMock)
# ---------------------------------------------------------------------------


async def test_test_mapping_class():
    """TestFoo entity + graph returns Foo UID → TESTS relationship."""
    entity = _make_entity(
        name="TestFoo",
        qn="proj:tests.test_app.TestFoo",
        label=NodeLabel.TYPE_DEF,
        kind=TypeDefKind.CLASS,
    )
    parsed = _make_parsed(entities=[entity])

    graph = AsyncMock()
    graph.execute = AsyncMock(return_value=[{"uid": "proj:src.app.Foo"}])

    det = TestMappingDetector()
    result = await det.detect(parsed, "proj", graph)
    assert len(result.relationships) == 1
    assert result.relationships[0].rel_type == RelType.TESTS
    assert result.relationships[0].to_name == "proj:src.app.Foo"


async def test_test_mapping_function():
    """test_foo function + graph returns foo UID → TESTS relationship."""
    entity = _make_entity(
        name="test_create_user",
        qn="proj:tests.test_app.test_create_user",
        label=NodeLabel.CALLABLE,
        kind=CallableKind.FUNCTION,
    )
    parsed = _make_parsed(entities=[entity])

    graph = AsyncMock()
    graph.execute = AsyncMock(return_value=[{"uid": "proj:src.app.create_user"}])

    det = TestMappingDetector()
    result = await det.detect(parsed, "proj", graph)
    assert len(result.relationships) == 1
    assert result.relationships[0].rel_type == RelType.TESTS


async def test_test_mapping_not_found():
    """If target not found in graph → no relationship."""
    entity = _make_entity(
        name="TestMissing",
        qn="proj:tests.test_app.TestMissing",
        label=NodeLabel.TYPE_DEF,
        kind=TypeDefKind.CLASS,
    )
    parsed = _make_parsed(entities=[entity])

    graph = AsyncMock()
    graph.execute = AsyncMock(return_value=[])

    det = TestMappingDetector()
    result = await det.detect(parsed, "proj", graph)
    assert result.relationships == []


async def test_overrides_found():
    """METHOD entity + INHERITS rel + graph returns parent UID → OVERRIDES relationship."""
    method = _make_entity(
        name="save",
        qn="proj:src.app.Child.save",
        label=NodeLabel.CALLABLE,
        kind=CallableKind.METHOD,
    )
    inherits_rel = ParsedRelationship(
        from_qualified_name="proj:src.app.Child",
        rel_type=RelType.INHERITS,
        to_name="Base",
    )
    parsed = _make_parsed(entities=[method], relationships=[inherits_rel])

    graph = AsyncMock()
    graph.execute = AsyncMock(return_value=[{"uid": "proj:src.base.Base.save"}])

    det = ClassOverridesDetector()
    result = await det.detect(parsed, "proj", graph)
    assert len(result.relationships) == 1
    assert result.relationships[0].rel_type == RelType.OVERRIDES
    assert result.relationships[0].to_name == "proj:src.base.Base.save"


async def test_overrides_no_inheritance():
    """No INHERITS relationships → no overrides detected."""
    method = _make_entity(
        name="save",
        qn="proj:src.app.MyClass.save",
        label=NodeLabel.CALLABLE,
        kind=CallableKind.METHOD,
    )
    parsed = _make_parsed(entities=[method])

    graph = AsyncMock()
    det = ClassOverridesDetector()
    result = await det.detect(parsed, "proj", graph)
    assert result.relationships == []


async def test_di_injection_relationship():
    """DI injection with graph → INJECTED_INTO relationship."""
    entity = _make_entity(signature="def handler(db=Depends(get_db))")
    parsed = _make_parsed(entities=[entity])

    graph = AsyncMock()
    graph.execute = AsyncMock(return_value=[{"uid": "proj:src.deps.get_db"}])

    det = DIInjectionDetector()
    result = await det.detect(parsed, "proj", graph)
    assert len(result.enrichments) == 1
    assert len(result.relationships) == 1
    assert result.relationships[0].rel_type == RelType.INJECTED_INTO
    assert result.relationships[0].from_qualified_name == "proj:src.deps.get_db"
    assert result.relationships[0].to_name == "proj:src.app.handler"
