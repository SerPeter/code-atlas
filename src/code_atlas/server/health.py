"""Health check and diagnostics for Code Atlas infrastructure."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from code_atlas.backends.postgres_queue import PostgresEventBus
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.health_verdicts import (
    CheckResult,
    CheckStatus,
    embeddings_verdict,
    memgraph_verdict,
    schema_verdict,
    valkey_verdict,
)
from code_atlas.indexing.orchestrator import StalenessChecker
from code_atlas.schema import SCHEMA_VERSION
from code_atlas.search.embeddings import EmbedClient
from code_atlas.search.ratelimit import unpaced
from code_atlas.settings import HealthSettings, _find_atlas_toml, derive_project_name, find_git_root

if TYPE_CHECKING:
    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import AtlasSettings, EmbeddingSettings, MemgraphSettings, RedisSettings

# Defaults for callers that run one check directly; run_health_checks passes [health].
_DEFAULTS = HealthSettings()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


# CheckStatus and CheckResult live in code_atlas.health_verdicts, beside the verdicts that
# produce them; importing them from here keeps working because this module is where callers look.


@dataclass(frozen=True)
class HealthReport:
    """Aggregated results from all health checks."""

    checks: list[CheckResult]
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        """True when no check has FAIL status (WARN is treated as passing)."""
        return all(c.status != CheckStatus.FAIL for c in self.checks)

    @property
    def degraded(self) -> bool:
        """True when any check is not fully OK (WARN or FAIL).

        Surfaces non-fatal degradations (e.g. Valkey down = indexing disabled,
        embeddings unreachable = vector search disabled) that ``ok`` alone hides.
        """
        return any(c.status != CheckStatus.OK for c in self.checks)


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


async def check_memgraph(
    graph: GraphClient | SqliteGraphClient | None,
    mg_settings: MemgraphSettings,
    *,
    timeout_s: float = _DEFAULTS.connect_timeout_s,
    chosen: bool = False,
) -> CheckResult:
    """Verify connectivity of the active graph backend, honestly naming which one it is.

    *graph* may be a real ``GraphClient`` (Memgraph) or the ``SqliteGraphClient``
    embedded fallback — whichever ``create_graph_client`` actually returned. *chosen*
    is whether the config declared that backend. The verdict is
    :func:`~code_atlas.health_verdicts.memgraph_verdict`, shared with the session-start hook.
    """
    addr = f"{mg_settings.host}:{mg_settings.port}"
    if graph is None:
        return CheckResult(
            "memgraph", CheckStatus.FAIL, f"No client ({addr})", suggestion="Check Memgraph connection settings."
        )
    embedded = isinstance(graph, SqliteGraphClient)
    try:
        ok = await asyncio.wait_for(graph.ping(), timeout=timeout_s)
    except Exception as exc:
        return memgraph_verdict(addr, reachable=False, embedded=embedded, detail=str(exc))
    detail = "" if ok else "ping returned false"
    return memgraph_verdict(addr, reachable=bool(ok), embedded=embedded, chosen=chosen, detail=detail)


async def check_schema(graph: GraphClient, *, timeout_s: float = _DEFAULTS.check_timeout_s) -> CheckResult:
    """Verify graph schema version matches the code."""
    try:
        stored = await asyncio.wait_for(graph.get_schema_version(), timeout=timeout_s)
    except Exception as exc:
        return schema_verdict(None, SCHEMA_VERSION, error=str(exc))
    return schema_verdict(stored, SCHEMA_VERSION)


async def check_embeddings(
    embed: EmbedClient | None,
    embed_settings: EmbeddingSettings,
    *,
    timeout_s: float = _DEFAULTS.check_timeout_s,
) -> CheckResult:
    """Verify the embedding provider answers: config, model and credentials in one real call."""
    if not embed_settings.enabled or embed is None:
        return embeddings_verdict(embed_settings, ok=None if embed_settings.enabled else True)
    try:
        ok = await asyncio.wait_for(embed.health_check(), timeout=timeout_s)
    except TimeoutError:
        return embeddings_verdict(embed_settings, ok=False, timed_out=True)
    except Exception as exc:
        return embeddings_verdict(embed_settings, ok=False, detail=str(exc))
    return embeddings_verdict(embed_settings, ok=bool(ok))


async def check_valkey(
    bus: EventBus | SqliteEventBus | PostgresEventBus | None,
    redis_settings: RedisSettings,
    *,
    timeout_s: float = _DEFAULTS.connect_timeout_s,
    chosen: bool = False,
) -> CheckResult:
    """Verify connectivity of the active queue backend, honestly naming which one it is.

    *bus* may be a real ``EventBus`` (Valkey), a ``PostgresEventBus``, or the
    ``SqliteEventBus`` embedded fallback — whichever ``create_event_bus`` actually returned (or
    the daemon's live bus, when one is running). Ownership (construction/closing) is the
    caller's responsibility — mirrors ``check_memgraph``, as does the shared verdict.
    """
    addr = f"{redis_settings.host}:{redis_settings.port}"
    if bus is None:
        return CheckResult(
            "valkey", CheckStatus.WARN, f"No client ({addr})", suggestion="Check Valkey connection settings."
        )
    embedded = isinstance(bus, SqliteEventBus)
    postgres = bus.address if isinstance(bus, PostgresEventBus) else None
    try:
        ok = await asyncio.wait_for(bus.ping(), timeout=timeout_s)
    except Exception as exc:
        return valkey_verdict(addr, reachable=False, embedded=embedded, postgres=postgres, detail=str(exc))
    detail = "" if ok else "ping returned false"
    return valkey_verdict(addr, reachable=bool(ok), embedded=embedded, postgres=postgres, chosen=chosen, detail=detail)


async def check_config(settings: AtlasSettings) -> CheckResult:
    """Verify project root, git repo, and loaded config files."""
    name = "config"
    root = settings.project_root

    if not root.exists():
        return CheckResult(
            name,
            CheckStatus.FAIL,
            f"Root does not exist: {root}",
            suggestion="Set project_root in atlas.toml or pass a valid path.",
        )

    # Which config file was loaded. No .env: Atlas reads none -- the environment is the caller's.
    config_match = _find_atlas_toml()
    detail = f"config: {config_match.path if config_match else 'not found'}"

    git_root = find_git_root(root)
    if git_root is None:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"No git repo at {root}",
            detail=f"Staleness checks and delta indexing require git. {detail}",
            suggestion="Run 'git init' or check project_root setting.",
        )

    return CheckResult(name, CheckStatus.OK, f"Valid root: {root}", detail=detail)


async def check_embedding_model(
    graph: GraphClient,
    embed_settings: EmbeddingSettings,
    project: str = "",
    *,
    timeout_s: float = _DEFAULTS.check_timeout_s,
) -> CheckResult:
    """Check whether *project*'s embedding model matches the configured one.

    Per project, not database-wide: one store holds several projects, each with its
    own model, and comparing against the database default reported a mismatch — and
    disabled vector search — for every project that was not the last one to index
    (ATL-135). The dimension half stays global, because the vector indices are.
    """
    name = "embedding_model"
    if not embed_settings.enabled:
        return CheckResult(name, CheckStatus.OK, "Skipped (embeddings disabled)")
    try:
        stored = await asyncio.wait_for(graph.get_embedding_config(), timeout=timeout_s)
        project_model = (
            await asyncio.wait_for(graph.get_project_embedding_model(project), timeout=timeout_s) if project else None
        )
    except Exception as exc:
        return CheckResult(name, CheckStatus.WARN, "Cannot read embedding config", detail=str(exc))

    if stored is None:
        return CheckResult(name, CheckStatus.OK, "No model lock (fresh database)")
    _stored_model, stored_dim = stored
    if project_model is None:
        return CheckResult(name, CheckStatus.OK, f"No model recorded for this project yet ({stored_dim}d)")
    if project_model == embed_settings.model:
        return CheckResult(name, CheckStatus.OK, f"Model matches: {project_model} ({stored_dim}d)")
    return CheckResult(
        name,
        CheckStatus.WARN,
        f"Mismatch: this project indexed under '{project_model}', configured='{embed_settings.model}'",
        detail=f"Stored dimension: {stored_dim}. Vector search disabled until re-indexed.",
        suggestion="Run 'atlas index --reset-embeddings' to re-embed this project with the new model.",
    )


async def check_index(graph: GraphClient, settings: AtlasSettings) -> CheckResult:
    """Check indexed project status."""
    timeout_s = settings.health.check_timeout_s
    name = "index"
    try:
        projects = await asyncio.wait_for(graph.get_project_status(), timeout=timeout_s)
    except Exception as exc:
        return CheckResult(name, CheckStatus.WARN, "Cannot read projects", detail=str(exc))

    if not projects:
        return CheckResult(
            name,
            CheckStatus.WARN,
            "No indexed projects",
            suggestion="Run 'atlas index <path>' to index a project.",
        )

    # Check staleness for the current project
    project_names = []
    for row in projects:
        node = row.get("n")
        if node is not None:
            props = dict(node.items()) if hasattr(node, "items") else node
            project_names.append(props.get("name", "?"))

    detail = f"Projects: {', '.join(project_names)}"

    checker = StalenessChecker(settings.project_root)
    try:
        info = await asyncio.wait_for(checker.check(graph, include_changed=False), timeout=timeout_s)
    except Exception:
        return CheckResult(name, CheckStatus.OK, f"{len(project_names)} project(s) indexed", detail=detail)

    if info.stale:
        commit = info.last_indexed_commit[:8] if info.last_indexed_commit else "never"
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Index is stale (last: {commit})",
            detail=detail,
            suggestion="Run 'atlas index' to update.",
        )

    return CheckResult(name, CheckStatus.OK, f"{len(project_names)} project(s) up to date", detail=detail)


async def check_indexer_lease(
    bus: EventBus | SqliteEventBus | PostgresEventBus | None, *, timeout_s: float = _DEFAULTS.connect_timeout_s
) -> CheckResult | None:
    """Report a foreign indexer holding the lease.

    Without this a second indexer is invisible: Redis identifies a consumer by name only,
    so two processes sharing one reported ``consumers=1`` and the pipeline looked healthy
    while a single index run was being split between them.

    Returns ``None`` when the lease is free — a check that says nothing when there is
    nothing to say, rather than adding a permanent OK line to every report.
    """
    if bus is None:
        return None
    try:
        holder = await asyncio.wait_for(bus.read_indexer_lease(), timeout=timeout_s)
    except Exception:
        return None
    if not holder:
        return None
    return CheckResult(
        "indexer_lease",
        CheckStatus.WARN,
        f"Another indexer is running ({holder})",
        detail="This project's pipeline is paused until that indexer finishes.",
        suggestion="Wait for it to finish, or stop the other 'atlas index' / daemon.",
    )


def check_pipeline(daemon: DaemonManager) -> CheckResult:
    """Report in-process indexing pipeline liveness from the DaemonManager."""
    st = daemon.status()
    if st["disabled_reason"]:
        # Distinct from a pipeline that started and died. Both leave zero tasks running,
        # and "0 task(s) running -- OK" is the report that sends someone looking for a
        # bug instead of reading their own configuration.
        return CheckResult("pipeline", CheckStatus.OK, f"not running — {st['disabled_reason']}")
    if st["crash_counts"]:
        worst = max(st["crash_counts"], key=st["crash_counts"].get)
        return CheckResult(
            "pipeline",
            CheckStatus.WARN,
            f"{st['tasks_running']}/{st['tasks_total']} task(s) running; "
            f"'{worst}' crashed {st['crash_counts'][worst]}x (supervised restart)",
            detail=st["last_crash"].get(worst, ""),
        )
    if st["tasks_running"] < st["tasks_total"]:
        return CheckResult(
            "pipeline", CheckStatus.FAIL, f"{st['tasks_total'] - st['tasks_running']} pipeline task(s) dead"
        )
    return CheckResult("pipeline", CheckStatus.OK, f"{st['tasks_running']} pipeline task(s) running")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_SKIPPED_DETAIL = "Skipped — Memgraph unreachable"


async def run_health_checks(
    settings: AtlasSettings,
    *,
    graph: GraphClient | SqliteGraphClient,
    bus: EventBus | SqliteEventBus | PostgresEventBus,
    embed: EmbedClient | None = None,
    daemon: DaemonManager | None = None,
) -> HealthReport:
    """Run all health checks and return an aggregated report.

    Independent checks (config, memgraph, embeddings, valkey) run concurrently.
    Dependent checks (schema, index) only run if the graph backend is reachable.

    *graph* and *bus* are required and never created here. A health check that opens
    its own connections is answering a slightly different question than the one asked --
    "can a fresh connection reach these services", not "are the connections this process
    is using healthy" -- and it used to track which ones it owned with a pair of booleans
    and a finally. The caller holds them: the CLI from a `use_backends` scope, the MCP
    server from its AppContext.

    *embed* stays optional because it is genuinely optional (lightweight mode has none)
    and needs no closing.
    """
    t0 = time.monotonic()

    if embed is None and settings.embeddings.enabled:
        # No limiter on purpose: a health probe must answer "is the provider
        # reachable" immediately. Handing it the rate limiter would let a drained
        # bucket block the check and report the provider down when it is merely busy.
        embed = EmbedClient(settings.embeddings, limiter=unpaced())

    # Mode indicator
    mode_label = "full" if settings.embeddings.enabled else "lightweight (no embeddings)"
    mode_res = CheckResult("mode", CheckStatus.OK, mode_label)

    # Phase 1: independent checks
    # The lease read belongs here too: it only needs the bus, and run after this phase it waited,
    # unbounded, on a Valkey that check_valkey had already found down.
    config_res, mg_res, embed_res, valkey_res, lease_res = await asyncio.gather(
        check_config(settings),
        check_memgraph(
            graph,
            settings.memgraph,
            timeout_s=settings.health.connect_timeout_s,
            chosen=settings.backend.graph_choice == "sqlite",
        ),
        check_embeddings(embed, settings.embeddings, timeout_s=settings.health.check_timeout_s),
        check_valkey(
            bus,
            settings.redis,
            timeout_s=settings.health.connect_timeout_s,
            chosen=settings.backend.queue_choice == "sqlite",
        ),
        check_indexer_lease(bus, timeout_s=settings.health.connect_timeout_s),
    )

    results = [mode_res, config_res, mg_res, embed_res, valkey_res]

    if lease_res is not None:
        results.append(lease_res)

    # Phase 2: Memgraph-dependent checks
    if mg_res.status == CheckStatus.FAIL:
        results.append(CheckResult("schema", CheckStatus.FAIL, _SKIPPED_DETAIL))
        results.append(CheckResult("embedding_model", CheckStatus.FAIL, _SKIPPED_DETAIL))
        results.append(CheckResult("index", CheckStatus.FAIL, _SKIPPED_DETAIL))
    else:
        assert graph is not None
        # check_schema/check_embedding_model/check_index stay declared as GraphClient-only
        # (same "deferred retyping" convention as the ~10 construction call sites elsewhere) —
        # they only call methods both backends implement, so the SqliteGraphClient case is safe.
        schema_res, model_res, index_res = await asyncio.gather(
            check_schema(graph, timeout_s=settings.health.check_timeout_s),  # ty: ignore[invalid-argument-type]
            check_embedding_model(
                graph,  # ty: ignore[invalid-argument-type]
                settings.embeddings,
                derive_project_name(settings.project_root),
                timeout_s=settings.health.check_timeout_s,
            ),
            check_index(graph, settings),  # ty: ignore[invalid-argument-type]
        )
        results.append(schema_res)
        results.append(model_res)
        results.append(index_res)

    # In-process pipeline liveness — only when a live DaemonManager is passed (MCP)
    if daemon is not None:
        results.append(check_pipeline(daemon))

    elapsed = (time.monotonic() - t0) * 1000
    return HealthReport(checks=results, elapsed_ms=elapsed)
