"""Does the graph make common Python design patterns visible, or leave loose objects?

A codebase is not a bag of functions — it is objects wired together by patterns a reader
already knows: a factory builds a thing, a registry owns a set of things, a service is
handed its collaborators. If the graph does not carry those edges, every one of those
questions returns an empty or misleading answer while looking perfectly healthy.

This module indexes a corpus of canonical patterns ONCE and then asks, per pattern, the
question a developer would actually ask. Each case records its measured status:

- ``LINKED``   — the edge exists today. The test fails if it disappears.
- ``MISSING``  — the edge does not exist today. The test fails if it *appears*, which is
  the point: a gap that gets fixed must show up as a red test telling you to promote it,
  not vanish silently into a suite that was already green.

So the suite measures coverage in both directions rather than merely asserting the parts
that happen to work. `COVERAGE_FLOOR` is the aggregate ratchet.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio

from code_atlas.graph.client import GraphClient
from code_atlas.indexing.orchestrator import index_project
from code_atlas.settings import AtlasSettings, MemgraphSettings, RedisSettings
from tests.conftest import NO_EMBED, TEST_DRAIN_TIMEOUT_S

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# The corpus is indexed once per module, so the fixture and the tests that read it must
# share one event loop — a module-scoped async fixture on the default function-scoped
# loop hands out a driver bound to a loop that is already closed.
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]


# ---------------------------------------------------------------------------
# The corpus — one module per pattern family, written the way people write them
# ---------------------------------------------------------------------------

CORPUS: dict[str, str] = {
    "ports.py": '''
"""Interfaces: an ABC, a Protocol, and an external-base enum."""
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Protocol


class Severity(StrEnum):
    """Subclasses an EXTERNAL base."""

    LOW = "low"
    HIGH = "high"


class StorageError(Exception):
    """Subclasses an external base too."""


class Repository(ABC):
    """Abstract base whose method body is `...` — the Protocol/ABC idiom."""

    @abstractmethod
    def load(self, key: str) -> str: ...


class BaseJob:
    """Base whose method has a REAL body, for contrast with the stub above."""

    def run(self) -> str:
        """Concrete default implementation."""
        return "base"


class Notifier(Protocol):
    """Structural — implementations declare no base."""

    def notify(self, message: str) -> None: ...
''',
    "adapters.py": '''
"""Implementations: one nominal (ABC), one structural (Protocol)."""
from patterns.ports import BaseJob, Repository


class SqlRepository(Repository):
    """Nominal subclass — declares its base."""

    def load(self, key: str) -> str:
        return f"sql:{key}"


class LogNotifier:
    """Structurally satisfies Notifier without naming it."""

    def notify(self, message: str) -> None:
        print(message)


class NightlyJob(BaseJob):
    """Overrides a CONCRETE base method."""

    def run(self) -> str:
        return "nightly"
''',
    "service.py": '''
"""Constructor dependency injection and delegation through a declared type."""
from patterns.ports import Repository


class Service:
    """Collaborators are handed in, never constructed here."""

    def __init__(self, repo: Repository) -> None:
        self.repo = repo

    def fetch(self, key: str) -> str:
        # Delegation through an annotated attribute — the receiver type is known.
        return self.repo.load(key)
''',
    "factories.py": '''
"""Factory function, classmethod factory, and a module-level singleton."""
from patterns.adapters import SqlRepository
from patterns.service import Service


def build_service() -> Service:
    """Plain factory function."""
    return Service(SqlRepository())


class ServiceFactory:
    @classmethod
    def create(cls) -> Service:
        """Classmethod factory."""
        return Service(SqlRepository())


DEFAULT_SERVICE = Service(SqlRepository())
"""Module-level singleton built at import time."""
''',
    "registry.py": '''
"""Two registries: decorator-driven and a plain dict of callables."""

__all__ = ["dispatch", "register"]

_HANDLERS: dict[str, object] = {}


def register(name: str):
    """Decorator-based registration."""

    def _wrap(fn):
        _HANDLERS[name] = fn
        return fn

    return _wrap


@register("greet")
def handle_greet(payload: str) -> str:
    """Only ever reached through the registry."""
    return f"hello {payload}"


@register("farewell")
def handle_farewell(payload: str) -> str:
    return f"bye {payload}"


def dispatch(name: str, payload: str) -> str:
    """Table-driven dispatch — the call target is a value, not a name."""
    return _HANDLERS[name](payload)


TABLE = {"greet": handle_greet, "farewell": handle_farewell}
"""A dict of callables, the other common registry shape."""
''',
    "composition.py": '''
"""Composition: nested config objects, the shape every settings tree has."""
from dataclasses import dataclass, field


@dataclass
class DbConfig:
    host: str = "localhost"
    port: int = 5432


@dataclass
class CacheConfig:
    ttl_s: int = 60


@dataclass
class AppConfig:
    """Made OUT OF the two above — the classic composition question."""

    db: DbConfig = field(default_factory=DbConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
''',
    "callbacks.py": '''
"""Callbacks, closures and nested definitions."""
from collections.abc import Callable


def on_complete(result: str) -> None:
    """Passed by reference, never called by name at the call site."""
    print(result)


def run_with(callback: Callable[[str], None]) -> None:
    callback("done")


def wire() -> None:
    """Passes a function as a value."""
    run_with(on_complete)


def outer(prefix: str) -> Callable[[str], str]:
    """Encloses a nested definition that itself calls out."""

    def inner(value: str) -> str:
        return helper(prefix + value)

    return inner


def helper(text: str) -> str:
    """Only ever called from inside a nested function."""
    return text.upper()
''',
}


@dataclass(frozen=True)
class Case:
    """One pattern, the question it answers, and the query that answers it."""

    name: str
    question: str
    cypher: str
    status: str  # "LINKED" (must stay) | "MISSING" (must stay absent until promoted)
    note: str = ""


# `$p` is bound to the corpus project name in every query.
CASES: tuple[Case, ...] = (
    Case(
        "abc-subclass",
        "what implements the Repository ABC?",
        "MATCH (a)-[:INHERITS|IMPLEMENTS]->(b) WHERE a.project_name=$p AND b.name='Repository' RETURN a.name AS hit",
        "LINKED",
    ),
    Case(
        "override-of-concrete-base",
        "which methods override BaseJob.run?",
        "MATCH (a)-[:OVERRIDES]->(b) WHERE a.project_name=$p AND b.qualified_name ENDS WITH 'BaseJob.run' "
        "RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "implements-abstract-method",
        "which methods implement the abstract Repository.load?",
        # Ask by QUESTION, not by edge type: an abstract implementation is IMPLEMENTS, an
        # ordinary redefinition is OVERRIDES, and asserting only one of them measures the
        # schema rather than the capability.
        "MATCH (a)-[r]->(b) WHERE a.project_name=$p AND b.qualified_name ENDS WITH 'Repository.load' "
        "AND type(r) IN ['IMPLEMENTS','OVERRIDES'] RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "protocol-method-implementation",
        "which methods implement the structural Notifier.notify?",
        "MATCH (a)-[r]->(b) WHERE a.project_name=$p AND b.qualified_name ENDS WITH 'Notifier.notify' "
        "AND type(r) IN ['IMPLEMENTS','OVERRIDES'] RETURN a.qualified_name AS hit",
        "MISSING",
        "structural conformance: LogNotifier.notify names no base. Measured on the real repo, "
        "88 of 102 `...`-bodied stub methods are GraphBackend Protocol methods with no inbound "
        "IMPLEMENTS at all.",
    ),
    Case(
        "constructor-injection",
        "what type does Service depend on?",
        "MATCH (a)-[:USES_TYPE]->(b) WHERE a.project_name=$p AND a.qualified_name CONTAINS 'Service.' "
        "AND b.name='Repository' RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "delegation-through-declared-type",
        "does Service.fetch reach Repository.load through self.repo?",
        "MATCH (a)-[:CALLS]->(b) WHERE a.project_name=$p AND a.qualified_name ENDS WITH 'Service.fetch' "
        "AND b.name='load' RETURN b.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "factory-function",
        "who constructs a Service?",
        "MATCH (a)-[:CALLS]->(b) WHERE a.project_name=$p AND a.qualified_name ENDS WITH 'build_service' "
        "AND b.name IN ['Service', '__init__'] RETURN b.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "classmethod-factory",
        "does ServiceFactory.create construct a Service?",
        "MATCH (a)-[:CALLS]->(b) WHERE a.project_name=$p AND a.qualified_name ENDS WITH 'ServiceFactory.create' "
        "AND b.name IN ['Service', '__init__'] RETURN b.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "decorator-registration",
        "is a registry-only handler linked to its registrar?",
        # Any edge will do — the question is whether the link exists, not which type
        # carries it. Asking only for CALLS would have answered "no" even once
        # REGISTERED_BY was written, which is the third time this suite caught me
        # measuring the schema instead of the capability.
        "MATCH (a)-[]->(b) WHERE a.project_name=$p AND b.name='register' AND a.name STARTS WITH 'handle_' "
        "RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "registry-dispatch",
        "who can dispatch() actually reach?",
        "MATCH (a)-[:CALLS]->(b) WHERE a.project_name=$p AND a.qualified_name ENDS WITH 'dispatch' "
        "AND b.name STARTS WITH 'handle_' RETURN b.qualified_name AS hit",
        "MISSING",
        "table-driven dispatch: the target is a value, not a name",
    ),
    Case(
        "callback-by-reference",
        "who uses on_complete?",
        # "Uses" includes naming it as a value. Restricting to CALLS would measure the
        # schema instead of the capability — and would answer "no" for a callback, which
        # is by definition handed over rather than invoked.
        "MATCH (a)-[r]->(b) WHERE a.project_name=$p AND b.name='on_complete' "
        "AND type(r) IN ['CALLS','USES_TYPE','REFERENCES'] RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "external-base-class",
        "what subclasses StrEnum?",
        "MATCH (a)-[:INHERITS|IMPLEMENTS]->(b) WHERE a.project_name=$p AND b.name='StrEnum' RETURN a.name AS hit",
        "LINKED",
    ),
    Case(
        "exception-hierarchy",
        "show me every exception type",
        "MATCH (a)-[:INHERITS|IMPLEMENTS]->(b) WHERE a.project_name=$p AND b.name='Exception' RETURN a.name AS hit",
        "MISSING",
        "`Exception` is a builtin: never imported, so no ExternalSymbol node exists for "
        "resolve_inherits to point at — unlike StrEnum/ABC/Protocol/BaseSettings, which "
        "arrive via an import and now resolve. Fixing this means deciding whether builtin "
        "bases deserve nodes at all, which is a separate call (see ADR-0020).",
    ),
    Case(
        "protocol-conformance",
        "what implements the Notifier Protocol?",
        "MATCH (a)-[:INHERITS|IMPLEMENTS]->(b) WHERE a.project_name=$p AND b.name='Notifier' RETURN a.name AS hit",
        "MISSING",
        "structural conformance — LogNotifier names no base",
    ),
    Case(
        "injected-attribute",
        "what collaborator does a Service instance hold?",
        "MATCH (n) WHERE n.project_name=$p AND n.qualified_name ENDS WITH 'Service.repo' "
        "RETURN n.qualified_name AS hit",
        "LINKED",
        "`self.repo = repo` produces no node at all — measured on the real repo, "
        "ASTConsumer.graph and TierConsumer.bus are both absent.",
    ),
    Case(
        "dataclass-composition",
        "what is AppConfig made of?",
        # Two hops, deliberately: the type edge hangs off the FIELD, not the class. A
        # direct class->class edge would answer "made of a DbConfig" while losing WHICH
        # field holds it, and a class with three fields of the same type would collapse to
        # one indistinguishable edge. DEFINES->USES_TYPE keeps the attribution.
        "MATCH (a:TypeDef {name: 'AppConfig'})-[:DEFINES]->(f)-[:USES_TYPE]->(b) "
        "WHERE a.project_name=$p AND b.name IN ['DbConfig','CacheConfig'] RETURN b.name AS hit",
        "LINKED",
    ),
    Case(
        "nested-function-entity",
        "is a nested function an entity at all?",
        "MATCH (n:Callable) WHERE n.project_name=$p AND n.name='inner' RETURN n.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "call-from-nested-body",
        "who calls helper(), which is only reached from inside a closure?",
        "MATCH (a)-[:CALLS]->(b) WHERE a.project_name=$p AND b.name='helper' RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "public-api-surface",
        "what does this module export as its public API?",
        "MATCH (m:Module)-[:EXPORTS]->(x) WHERE m.project_name=$p RETURN x.name AS hit",
        "LINKED",
    ),
    Case(
        "module-level-singleton",
        "who constructs DEFAULT_SERVICE?",
        "MATCH (n:Value) WHERE n.project_name=$p AND n.name='DEFAULT_SERVICE' RETURN n.qualified_name AS hit",
        "LINKED",
    ),
)

# Ratchet: the share of pattern questions the graph can answer today. Raising this is the
# point of the work; it must never fall.
COVERAGE_FLOOR = sum(1 for c in CASES if c.status == "LINKED") / len(CASES)


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("patterns_project")
    pkg = root / "patterns"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for rel, body in CORPUS.items():
        (pkg / rel).write_text(body.lstrip(), encoding="utf-8")
    return root


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def indexed_corpus(corpus_root, _infra_endpoints) -> AsyncIterator[tuple[GraphClient, str]]:
    """Index the pattern corpus ONCE for the whole module.

    Deliberately not the function-scoped ``graph_client``: that wipes every node before
    each test, which would delete the corpus between the cases that query it.
    """
    project = f"test-patterns-{uuid.uuid4().hex[:8]}"
    settings = AtlasSettings(
        project_root=corpus_root,
        memgraph=MemgraphSettings(host=_infra_endpoints.memgraph_host, port=_infra_endpoints.memgraph_port),
        redis=RedisSettings(
            host=_infra_endpoints.valkey_host,
            port=_infra_endpoints.valkey_port,
            stream_prefix=f"test-{uuid.uuid4().hex[:8]}",
        ),
        embeddings=NO_EMBED,
    )
    client = GraphClient(settings)
    try:
        await client.ping()
    except Exception:
        pytest.skip("Memgraph not available")

    from code_atlas.events import EventBus

    bus = EventBus(settings.redis, project_name=project)
    await client.ensure_schema()
    await index_project(
        settings,
        client,
        bus,
        project_name=project,
        project_root=corpus_root,
        drain_timeout_s=TEST_DRAIN_TIMEOUT_S,
    )
    try:
        yield client, project
    finally:
        await client.execute_write("MATCH (n) WHERE n.project_name = $p DETACH DELETE n", {"p": project})
        await bus.close()
        await client.close()


async def _hits(client: GraphClient, project: str, case: Case) -> list[Any]:
    rows = await client.execute(case.cypher, {"p": project})
    return [r["hit"] for r in rows]


async def test_the_corpus_indexed_at_all(indexed_corpus):
    """Guard: a corpus that failed to index would make every MISSING case pass vacuously."""
    client, project = indexed_corpus
    rows = await client.execute("MATCH (n:Callable) WHERE n.project_name=$p RETURN count(n) AS c", {"p": project})
    assert rows[0]["c"] >= 10, "pattern corpus did not index — the rest of this module proves nothing"


@pytest.mark.parametrize("case", [c for c in CASES if c.status == "LINKED"], ids=lambda c: c.name)
async def test_pattern_is_visible(indexed_corpus, case: Case):
    """These questions the graph can answer today. Losing one is a regression."""
    client, project = indexed_corpus
    hits = await _hits(client, project, case)
    assert hits, f"REGRESSION — {case.question!r} used to be answerable, now returns nothing"


@pytest.mark.parametrize("case", [c for c in CASES if c.status == "MISSING"], ids=lambda c: c.name)
async def test_known_gap_is_still_a_gap(indexed_corpus, case: Case):
    """These return nothing today.

    Failing here is GOOD NEWS: the pattern started resolving. Flip the case to LINKED so
    the ratchet rises and the new edge is protected from regressing.
    """
    client, project = indexed_corpus
    hits = await _hits(client, project, case)
    assert not hits, (
        f"FIXED — {case.question!r} now returns {hits[:3]}. "
        f"Promote case {case.name!r} to status='LINKED' to lock it in."
    )


async def test_pattern_coverage_ratchet(indexed_corpus):
    """The headline number, so coverage is a measurement and not an anecdote."""
    client, project = indexed_corpus
    answered = [case.name for case in CASES if await _hits(client, project, case)]
    coverage = len(answered) / len(CASES)
    assert coverage >= COVERAGE_FLOOR, (
        f"pattern coverage fell to {coverage:.0%} (floor {COVERAGE_FLOOR:.0%}); answered: {sorted(answered)}"
    )
