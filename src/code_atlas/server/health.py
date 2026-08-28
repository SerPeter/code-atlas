"""Health check and diagnostics for Code Atlas infrastructure."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from dotenv import find_dotenv

from code_atlas.backends import create_event_bus, create_graph_client
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.backends.sqlite_queue import SqliteEventBus
from code_atlas.indexing.orchestrator import StalenessChecker
from code_atlas.schema import SCHEMA_VERSION
from code_atlas.search.embeddings import EmbedClient
from code_atlas.settings import _find_atlas_toml, find_git_root

if TYPE_CHECKING:
    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import AtlasSettings, EmbeddingSettings, MemgraphSettings, RedisSettings

_CHECK_TIMEOUT = 3.0  # seconds per individual check


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str
    detail: str = ""
    suggestion: str = ""


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
) -> CheckResult:
    """Verify connectivity of the active graph backend, honestly naming which one it is.

    *graph* may be a real ``GraphClient`` (Memgraph) or the ``SqliteGraphClient``
    embedded fallback — whichever ``create_graph_client`` actually returned. The
    reported message always names the real backend instead of assuming Memgraph.
    """
    name = "memgraph"
    addr = f"{mg_settings.host}:{mg_settings.port}"
    if graph is None:
        return CheckResult(
            name, CheckStatus.FAIL, f"No client ({addr})", suggestion="Check Memgraph connection settings."
        )

    embedded = isinstance(graph, SqliteGraphClient)
    backend = "SQLite (embedded)" if embedded else f"Memgraph ({addr})"

    try:
        ok = await asyncio.wait_for(graph.ping(), timeout=_CHECK_TIMEOUT)
        if ok and embedded:
            # WARN, not OK. `backend.graph = "auto"` falls back here whenever Memgraph is
            # unreachable, so this is the *default* outcome on a machine without Docker
            # running — and ADR-0015 calls SQLite explicitly not a parity replacement.
            # Reporting an unqualified OK meant a fully-degraded install looked healthy,
            # which is the one thing a health check must never do.
            return CheckResult(
                name,
                CheckStatus.WARN,
                f"Connected — {backend}, NOT Memgraph",
                detail=(
                    "The embedded fallback is active. Community detection is unavailable and "
                    "some analyses differ from Memgraph; see the README."
                ),
                suggestion=f"Start Memgraph ({addr}) with: docker compose up -d memgraph",
            )
        if ok:
            return CheckResult(name, CheckStatus.OK, f"Connected — {backend}")
        return CheckResult(
            name, CheckStatus.FAIL, f"Ping failed — {backend}", suggestion="docker compose up -d memgraph"
        )
    except Exception as exc:
        return CheckResult(
            name,
            CheckStatus.FAIL,
            f"Unreachable — {backend}",
            detail=str(exc),
            suggestion="docker compose up -d memgraph",
        )


async def check_schema(graph: GraphClient) -> CheckResult:
    """Verify graph schema version matches the code."""
    name = "schema"
    try:
        stored = await asyncio.wait_for(graph.get_schema_version(), timeout=_CHECK_TIMEOUT)
    except Exception as exc:
        return CheckResult(name, CheckStatus.FAIL, "Cannot read schema version", detail=str(exc))

    if stored is None:
        return CheckResult(
            name,
            CheckStatus.WARN,
            "No schema version found",
            detail="Database may be empty.",
            suggestion="Run 'atlas index' to initialize the schema.",
        )
    if stored == SCHEMA_VERSION:
        return CheckResult(name, CheckStatus.OK, f"Version {stored} (current)")
    if stored < SCHEMA_VERSION:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Version {stored} (expected {SCHEMA_VERSION})",
            detail="Schema is outdated.",
            suggestion="Run 'atlas index' to migrate the schema.",
        )
    # stored > SCHEMA_VERSION
    return CheckResult(
        name,
        CheckStatus.FAIL,
        f"Version {stored} > code {SCHEMA_VERSION}",
        detail="Database schema is newer than the installed code.",
        suggestion="Update your Code Atlas installation.",
    )


async def check_embeddings(
    embed: EmbedClient | None,
    embed_settings: EmbeddingSettings,
) -> CheckResult:
    """Verify the embedding service is reachable."""
    name = "embeddings"
    if not embed_settings.enabled:
        return CheckResult(name, CheckStatus.OK, "Disabled (lightweight mode)")

    # Provider-aware display and suggestions
    if embed_settings.provider == "tei":
        info = f"tei @ {embed_settings.base_url}"
        suggestion = "docker compose --profile tei up -d"
    elif embed_settings.provider == "ollama":
        info = f"ollama @ {embed_settings.base_url}"
        suggestion = "Start Ollama and pull the model: ollama pull " + embed_settings.model
    else:
        info = f"{embed_settings.provider} ({embed_settings.model})"
        suggestion = "Check your API key in .env (e.g. OPENAI_API_KEY) and network connectivity."

    if embed is None:
        return CheckResult(name, CheckStatus.WARN, f"No client ({info})", suggestion="Check embedding settings.")

    try:
        ok = await asyncio.wait_for(embed.health_check(), timeout=_CHECK_TIMEOUT)
        if ok:
            return CheckResult(name, CheckStatus.OK, f"Responding ({info})")
        return CheckResult(name, CheckStatus.WARN, f"Unreachable ({info})", suggestion=suggestion)
    except Exception as exc:
        return CheckResult(name, CheckStatus.WARN, f"Unreachable ({info})", detail=str(exc), suggestion=suggestion)


async def check_valkey(
    bus: EventBus | SqliteEventBus | None,
    redis_settings: RedisSettings,
) -> CheckResult:
    """Verify connectivity of the active queue backend, honestly naming which one it is.

    *bus* may be a real ``EventBus`` (Valkey) or the ``SqliteEventBus`` embedded
    fallback — whichever ``create_event_bus`` actually returned (or the daemon's
    live bus, when one is running). The reported message always names the real
    backend instead of assuming Valkey. Ownership (construction/closing) is the
    caller's responsibility — mirrors ``check_memgraph``.
    """
    name = "valkey"
    addr = f"{redis_settings.host}:{redis_settings.port}"
    if bus is None:
        return CheckResult(
            name, CheckStatus.WARN, f"No client ({addr})", suggestion="Check Valkey connection settings."
        )

    is_sqlite = isinstance(bus, SqliteEventBus)
    backend = "SQLite (embedded)" if is_sqlite else f"Valkey ({addr})"
    indexing_disabled = (
        "auto-indexing disabled — file changes will NOT be indexed until the embedded queue is available"
        if is_sqlite
        else "auto-indexing disabled — file changes will NOT be indexed until Valkey is reachable"
    )
    try:
        ok = await asyncio.wait_for(bus.ping(), timeout=_CHECK_TIMEOUT)
        if ok and is_sqlite:
            # Same reasoning as check_memgraph: ADR-0015 calls the embedded path "not a
            # parity replacement for the Memgraph+Valkey path", and names a concrete gap
            # — blocking reads are emulated by short polling, since SQLite has no
            # server-side blocking. The queue works, so this stays WARN rather than FAIL,
            # but a plain OK would let a fallback nobody chose read as the supported
            # configuration.
            return CheckResult(
                name,
                CheckStatus.WARN,
                f"Connected — {backend}, NOT Valkey",
                detail="Blocking reads are emulated by polling; throughput is lower than the Valkey path.",
                suggestion=f"Start Valkey ({addr}) with: docker compose up -d valkey",
            )
        if ok:
            return CheckResult(name, CheckStatus.OK, f"Connected — {backend}")
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Ping failed — {backend} — {indexing_disabled}",
            suggestion="docker compose up -d valkey",
        )
    except Exception as exc:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Unreachable — {backend} — {indexing_disabled}",
            detail=str(exc),
            suggestion="docker compose up -d valkey",
        )


async def check_config(settings: AtlasSettings, *, dotenv_path: str = "") -> CheckResult:
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

    # Build detail string showing which config files were loaded
    config_match = _find_atlas_toml()
    # Callers that know the path pass it (the CLI captures it at load time). The MCP
    # server has no such handle — cli.py loads the .env before handing off, so re-resolve
    # it here rather than reporting "not found" for a file that is demonstrably loaded.
    resolved_dotenv = dotenv_path or find_dotenv(usecwd=True)
    detail_parts: list[str] = []
    detail_parts.append(f"config: {config_match.path if config_match else 'not found'}")
    detail_parts.append(f".env: {resolved_dotenv or 'not found'}")
    detail = " | ".join(detail_parts)

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


async def check_embedding_model(graph: GraphClient, embed_settings: EmbeddingSettings) -> CheckResult:
    """Check whether the stored embedding model matches the configured model."""
    name = "embedding_model"
    if not embed_settings.enabled:
        return CheckResult(name, CheckStatus.OK, "Skipped (embeddings disabled)")
    try:
        stored = await asyncio.wait_for(graph.get_embedding_config(), timeout=_CHECK_TIMEOUT)
    except Exception as exc:
        return CheckResult(name, CheckStatus.WARN, "Cannot read embedding config", detail=str(exc))

    if stored is None:
        return CheckResult(name, CheckStatus.OK, "No model lock (fresh database)")
    stored_model, stored_dim = stored
    if stored_model == embed_settings.model:
        return CheckResult(name, CheckStatus.OK, f"Model matches: {stored_model} ({stored_dim}d)")
    return CheckResult(
        name,
        CheckStatus.WARN,
        f"Mismatch: stored='{stored_model}', configured='{embed_settings.model}'",
        detail=f"Stored dimension: {stored_dim}. Vector search disabled until re-indexed.",
        suggestion="Run 'atlas index --full' to re-embed with the new model.",
    )


async def check_index(graph: GraphClient, settings: AtlasSettings) -> CheckResult:
    """Check indexed project status."""
    name = "index"
    try:
        projects = await asyncio.wait_for(graph.get_project_status(), timeout=_CHECK_TIMEOUT)
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
        info = await asyncio.wait_for(checker.check(graph, include_changed=False), timeout=_CHECK_TIMEOUT)
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


async def check_indexer_lease(bus: EventBus | SqliteEventBus | None) -> CheckResult | None:
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
        holder = await bus.read_indexer_lease()
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
    graph: GraphClient | SqliteGraphClient | None = None,
    embed: EmbedClient | None = None,
    daemon: DaemonManager | None = None,
    bus: EventBus | SqliteEventBus | None = None,
    dotenv_path: str = "",
) -> HealthReport:
    """Run all health checks and return an aggregated report.

    Independent checks (config, memgraph, embeddings, valkey) run concurrently.
    Dependent checks (schema, index) only run if the graph backend is reachable.

    When called from CLI, *graph*/*embed*/*bus* are ``None`` — temporary clients
    are created (via the same backend-selection factories used everywhere else)
    and closed.  MCP passes existing clients from AppContext; if *bus* isn't
    passed explicitly but *daemon* is, the daemon's own live bus is used instead
    of opening a redundant connection.
    """
    t0 = time.monotonic()

    # Create temporary clients if not provided
    own_graph = graph is None
    if own_graph:
        graph = await create_graph_client(settings)
    if embed is None and settings.embeddings.enabled:
        # No redis_settings on purpose: a health probe must answer "is the provider
        # reachable" immediately. Handing it the rate limiter would let a drained
        # bucket block the check and report the provider down when it is merely busy.
        embed = EmbedClient(settings.embeddings)
    if bus is None and daemon is not None:
        bus = daemon.bus
    own_bus = bus is None
    if own_bus:
        bus = await create_event_bus(settings)

    try:
        # Mode indicator
        mode_label = "full" if settings.embeddings.enabled else "lightweight (no embeddings)"
        mode_res = CheckResult("mode", CheckStatus.OK, mode_label)

        # Phase 1: independent checks
        config_res, mg_res, embed_res, valkey_res = await asyncio.gather(
            check_config(settings, dotenv_path=dotenv_path),
            check_memgraph(graph, settings.memgraph),
            check_embeddings(embed, settings.embeddings),
            check_valkey(bus, settings.redis),
        )

        results = [mode_res, config_res, mg_res, embed_res, valkey_res]

        lease_res = await check_indexer_lease(bus)
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
                check_schema(graph),  # type: ignore[invalid-argument-type]
                check_embedding_model(graph, settings.embeddings),  # type: ignore[invalid-argument-type]
                check_index(graph, settings),  # type: ignore[invalid-argument-type]
            )
            results.append(schema_res)
            results.append(model_res)
            results.append(index_res)

        # In-process pipeline liveness — only when a live DaemonManager is passed (MCP)
        if daemon is not None:
            results.append(check_pipeline(daemon))
    finally:
        if own_graph:
            assert graph is not None
            await graph.close()
        if own_bus:
            assert bus is not None
            await bus.close()

    elapsed = (time.monotonic() - t0) * 1000
    return HealthReport(checks=results, elapsed_ms=elapsed)
