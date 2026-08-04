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
    "endpoints.py": '''
"""Registration surfaces from four different frameworks, none of them known to the parser."""


@app.get("/users/{id}")
def read_user(id: int) -> dict:
    """FastAPI: the verb is in the decorator name, the path is the string."""
    return {}


@app.route("/legacy", methods=["GET"])
def legacy() -> dict:
    """Flask: the verb is a kwarg, so a first-string rule gets the path only."""
    return {}


@cli.command("mine-git-history")
def mine_git_history() -> None:
    """Typer: the command name differs from the function name."""


@celery.task(name="send.email")
def send_email() -> None:
    """Celery: the surface key arrives as a keyword argument."""
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
    "sinks.py": '''
"""A second `flush` implementation, deliberately in a different file from drain()."""


class BackupSink:
    def flush(self) -> None:
        """Same name as PrimarySink.flush, different file — the duck-typed twin."""
''',
    "boot.py": '''
"""Everything that runs at import time, which is not a function body."""
import asyncio
from dataclasses import dataclass, field


class Settings:
    """Constructed at module scope, never inside any function."""


def _new_flag() -> asyncio.Event:
    return asyncio.Event()


def _validate_wiring() -> None:
    """Called once at the foot of this module and nowhere else."""


class Limit:
    """Named only by a module constant's annotation."""


SETTINGS = Settings()
LIMITS: tuple[Limit, ...] = ()


@dataclass
class Boot:
    """A DECORATED class body — the guard for decorated functions used to skip these too."""

    flag: asyncio.Event = field(default_factory=_new_flag)


class Scanner:
    def scan(self) -> None:
        """Hands its own bound method to a scheduler instead of calling it."""
        asyncio.get_event_loop().run_in_executor(None, self._walk)

    def _walk(self) -> None:
        """Only ever passed as a value via `self._walk`."""


class PrimarySink:
    def flush(self) -> None:
        """Co-located with the only caller, which does NOT mean it is the callee."""


def drain(sink) -> None:
    """`sink` is untyped, so `sink.flush()` must not bind to the same-file PrimarySink."""
    sink.flush()


_validate_wiring()
''',
    "ledger.py": '''
"""A production method whose name a test double also uses."""


class Ledger:
    def commit(self) -> None:
        """The real implementation, and the only legitimate target of submit()."""


def submit(ledger) -> None:
    """`ledger` is untyped, so the receiver is ungrounded: strategies 2 and 3 are skipped
    and resolution falls through to the project-wide pool — which is exactly where a
    same-named test double would otherwise compete for the edge."""
    ledger.commit()
''',
    "test_doubles.py": '''
"""A test double. The FILENAME is what marks it — it matches the default `test_*` pattern."""


class FakeLedger:
    def commit(self) -> None:
        """Same method name as Ledger.commit, and must never absorb a production call."""
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
        "LINKED",
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
        "framework-route-surface",
        "what serves /users/{id}?",
        # No framework knowledge anywhere: the parser records "decorated by X with string
        # Y" and the FRAMEWORK is a filter written here, at query time. A new framework
        # needs no parser change — which is the whole point of the generic extraction.
        "MATCH (n:Callable) WHERE n.project_name=$p AND n.decorator_arg = '/users/{id}' RETURN n.name AS hit",
        "LINKED",
    ),
    Case(
        "framework-cli-surface",
        "which function handles the `mine-git-history` command?",
        "MATCH (n:Callable) WHERE n.project_name=$p AND n.decorator_name ENDS WITH '.command' "
        "AND n.decorator_arg = 'mine-git-history' RETURN n.name AS hit",
        "LINKED",
        "The decorator argument differs from the function name, which is the case a name-based guess gets wrong.",
    ),
    Case(
        "registry-dispatch",
        "what does dispatch() dispatch through?",
        # A subscript callee has no name to resolve, so the honest edge runs to the TABLE.
        # Never CALLS, and never fanned out: collapsing this into direct edges would give
        # one call site as many full-confidence targets as the table has entries.
        "MATCH (a)-[:REFERENCES]->(tbl) WHERE a.project_name=$p AND a.name = 'dispatch' RETURN tbl.name AS hit",
        "LINKED",
        "Reaching the MEMBERS of `_HANDLERS` is a separate matter and stays unanswered on "
        "purpose: it is populated at runtime by the decorator (`_HANDLERS[name] = fn` "
        "inside a closure), so no static pass can know its contents without executing it. "
        "A dict LITERAL is decidable and is covered — the module references handle_greet "
        "and handle_farewell through TABLE — and the decorator path is covered by "
        "REGISTERED_BY. Only the runtime-populated shape is out of reach.",
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
        "LINKED",
    ),
    Case(
        "protocol-conformance",
        "what implements the Notifier Protocol?",
        "MATCH (a)-[:INHERITS|IMPLEMENTS]->(b) WHERE a.project_name=$p AND b.name='Notifier' RETURN a.name AS hit",
        "LINKED",
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
    Case(
        "import-time-call",
        "what reaches _validate_wiring(), called only at the foot of its own module?",
        # Asked as "any edge", not CALLS: the parser cannot tell an import-time invocation
        # from a construction, so it emits both and lets each resolver claim its own.
        "MATCH (a)-[r]->(b) WHERE a.project_name=$p AND b.name='_validate_wiring' "
        "AND type(r) IN ['CALLS','REFERENCES','USES_TYPE'] RETURN a.qualified_name AS hit",
        "LINKED",
        "every CALLS edge used to have a Callable source and not one had a Module source, so "
        "a function invoked only at import time looked unreachable.",
    ),
    Case(
        "module-scope-construction",
        "what uses Settings, constructed at module scope with no annotation?",
        "MATCH (a)-[r]->(b) WHERE a.project_name=$p AND b.name='Settings' "
        "AND type(r) IN ['USES_TYPE','CALLS','REFERENCES'] RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "annotated-module-constant",
        "what uses Limit, named only inside a module constant's annotation?",
        "MATCH (a)-[:USES_TYPE]->(b) WHERE a.project_name=$p AND b.name='Limit' RETURN a.qualified_name AS hit",
        "LINKED",
        "the annotation scan was gated to class fields, so `LIMITS: tuple[Limit, ...]` said nothing.",
    ),
    Case(
        "decorated-class-body-default",
        "what reaches _new_flag, handed to field(default_factory=...) inside a @dataclass?",
        "MATCH (a)-[:REFERENCES]->(b) WHERE a.project_name=$p AND b.name='_new_flag' RETURN a.qualified_name AS hit",
        "LINKED",
        "the decorated_definition guard exists to skip decorated FUNCTIONS; it skipped decorated "
        "classes too, and @dataclass is how most classes here are written.",
    ),
    Case(
        "impact-of-changing-a-type",
        "what would be affected if I changed Repository — the question blast_radius exists for?",
        # A class is never CALLED; it is annotated, constructed and implemented. Reading
        # CALLS only, this answered ZERO for every type in the codebase — on the real repo
        # that was 18% of src entities, GraphClient's 239 dependents among them.
        "MATCH (a)-[r:CALLS|USES_TYPE|IMPLEMENTS|OVERRIDES|INHERITS|REFERENCES|IMPORTS]->(b) "
        "WHERE a.project_name=$p AND b.name='Repository' RETURN a.qualified_name AS hit",
        "LINKED",
    ),
    Case(
        "duck-typed-twin-not-stolen-by-co-location",
        "does sink.flush() reach the OTHER implementation, not just the co-located one?",
        "MATCH (a)-[:CALLS]->(b) WHERE a.project_name=$p AND a.qualified_name ENDS WITH 'boot.drain' "
        "AND b.name='flush' AND NOT b.qualified_name CONTAINS 'PrimarySink' RETURN b.qualified_name AS hit",
        "LINKED",
        "same-file matching is a LEXICAL lookup, and `sink.flush()` is not looked up lexically. "
        "Ungated it awarded the call to the co-located class at confidence 'resolved', so the real "
        "implementation in another file read dead while the wrong one absorbed its only caller.",
    ),
    Case(
        "self-method-as-value",
        "what reaches Scanner._walk, only ever passed as `self._walk`?",
        "MATCH (a)-[:REFERENCES]->(b) WHERE a.project_name=$p AND b.qualified_name ENDS WITH 'Scanner._walk' "
        "RETURN a.qualified_name AS hit",
        "LINKED",
        "`self.` pins the name to one class, so the methods-only exclusion (which exists because a "
        "BARE name matching a method is coincidence) does not apply.",
    ),
    Case(
        "production-call-skips-the-test-double",
        "does submit reach the production Ledger.commit?",
        "MATCH (a)-[:CALLS]->(b) WHERE a.project_name=$p AND a.qualified_name ENDS WITH 'submit' "
        "AND b.qualified_name CONTAINS 'Ledger.commit' AND NOT b.qualified_name CONTAINS 'FakeLedger' "
        "RETURN b.qualified_name AS hit",
        "LINKED",
        "ATL-103: production code cannot depend on test code, so a test definition is dropped from "
        "the candidate pool before the name-matching strategies read it.",
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


async def test_a_test_double_never_absorbs_a_production_call(indexed_corpus):
    """The other half of ATL-103, which the LINKED/MISSING harness cannot express.

    A LINKED case proves the right edge EXISTS; it says nothing about a wrong edge sitting
    beside it. The candidate_count assertion is the half that actually bites: leave the
    fixture in the pool and the production edge is written at candidate_count 2, so
    ``weight`` is 0.5 and the real call reaches Leiden and blast_radius ranking at half
    strength — a quiet mis-ranking rather than a visibly wrong answer.
    """
    client, project = indexed_corpus
    rows = await client.execute(
        "MATCH (a)-[r:CALLS]->(b) WHERE a.project_name=$p AND a.qualified_name ENDS WITH 'submit' "
        "AND b.name='commit' RETURN b.qualified_name AS target, r.candidate_count AS candidates",
        {"p": project},
    )
    targets = {r["target"] for r in rows}
    assert targets, "submit -> commit did not resolve at all; this test proves nothing as written"
    assert not any("FakeLedger" in t for t in targets), f"a test double absorbed a production call: {sorted(targets)}"
    assert all(r["candidates"] == 1 for r in rows), (
        f"the test double still inflated candidate_count, halving the real edge's weight: {rows}"
    )


async def test_pattern_coverage_ratchet(indexed_corpus):
    """The headline number, so coverage is a measurement and not an anecdote."""
    client, project = indexed_corpus
    answered = [case.name for case in CASES if await _hits(client, project, case)]
    coverage = len(answered) / len(CASES)
    assert coverage >= COVERAGE_FLOOR, (
        f"pattern coverage fell to {coverage:.0%} (floor {COVERAGE_FLOOR:.0%}); answered: {sorted(answered)}"
    )
