"""Health check and diagnostics for Code Atlas infrastructure."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from code_atlas.events import EventBus
from code_atlas.graph.client import GraphClient
from code_atlas.indexing.orchestrator import StalenessChecker
from code_atlas.schema import SCHEMA_VERSION
from code_atlas.search.embeddings import EmbedClient
from code_atlas.settings import find_git_root

if TYPE_CHECKING:
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


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------


async def check_memgraph(
    graph: GraphClient | None,
    mg_settings: MemgraphSettings,
) -> CheckResult:
    """Verify Memgraph connectivity."""
    name = "memgraph"
    addr = f"{mg_settings.host}:{mg_settings.port}"
    if graph is None:
        return CheckResult(
            name, CheckStatus.FAIL, f"No client ({addr})", suggestion="Check Memgraph connection settings."
        )

    try:
        ok = await asyncio.wait_for(graph.ping(), timeout=_CHECK_TIMEOUT)
        if ok:
            return CheckResult(name, CheckStatus.OK, f"Connected ({addr})")
        return CheckResult(name, CheckStatus.FAIL, f"Ping failed ({addr})", suggestion="docker compose up -d memgraph")
    except Exception as exc:
        return CheckResult(
            name,
            CheckStatus.FAIL,
            f"Unreachable ({addr})",
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
    info = f"{embed_settings.provider} @ {embed_settings.base_url}"
    if embed is None:
        return CheckResult(name, CheckStatus.WARN, f"No client ({info})", suggestion="Check embedding settings.")

    try:
        ok = await asyncio.wait_for(embed.health_check(), timeout=_CHECK_TIMEOUT)
        if ok:
            return CheckResult(name, CheckStatus.OK, f"Responding ({info})")
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Unreachable ({info})",
            suggestion="docker compose up -d tei",
        )
    except Exception as exc:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Unreachable ({info})",
            detail=str(exc),
            suggestion="docker compose up -d tei",
        )


async def check_valkey(redis_settings: RedisSettings) -> CheckResult:
    """Verify Valkey/Redis connectivity."""
    name = "valkey"
    addr = f"{redis_settings.host}:{redis_settings.port}"
    bus = EventBus(redis_settings)
    try:
        ok = await asyncio.wait_for(bus.ping(), timeout=_CHECK_TIMEOUT)
        if ok:
            return CheckResult(name, CheckStatus.OK, f"Connected ({addr})")
        return CheckResult(name, CheckStatus.WARN, f"Ping failed ({addr})", suggestion="docker compose up -d valkey")
    except Exception as exc:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Unreachable ({addr})",
            detail=str(exc),
            suggestion="docker compose up -d valkey",
        )
    finally:
        await bus.close()


async def check_config(settings: AtlasSettings) -> CheckResult:
    """Verify project root and git repo."""
    name = "config"
    root = settings.project_root

    if not root.exists():
        return CheckResult(
            name,
            CheckStatus.FAIL,
            f"Root does not exist: {root}",
            suggestion="Set project_root in atlas.toml or pass a valid path.",
        )

    git_root = find_git_root(root)
    if git_root is None:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"No git repo at {root}",
            detail="Staleness checks and delta indexing require git.",
            suggestion="Run 'git init' or check project_root setting.",
        )

    return CheckResult(name, CheckStatus.OK, f"Valid root: {root}")


async def check_embedding_model(graph: GraphClient, embed_settings: EmbeddingSettings) -> CheckResult:
    """Check whether the stored embedding model matches the configured model."""
    name = "embedding_model"
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


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_SKIPPED_DETAIL = "Skipped — Memgraph unreachable"


async def run_health_checks(
    settings: AtlasSettings,
    *,
    graph: GraphClient | None = None,
    embed: EmbedClient | None = None,
) -> HealthReport:
    """Run all health checks and return an aggregated report.

    Independent checks (config, memgraph, embeddings, valkey) run concurrently.
    Dependent checks (schema, index) only run if Memgraph is reachable.

    When called from CLI, *graph* and *embed* are ``None`` — temporary clients
    are created and closed.  MCP passes existing clients from AppContext.
    """
    t0 = time.monotonic()

    # Create temporary clients if not provided
    own_graph = graph is None
    if own_graph:
        graph = GraphClient(settings)
    if embed is None:
        embed = EmbedClient(settings.embeddings)

    try:
        # Phase 1: independent checks
        config_res, mg_res, embed_res, valkey_res = await asyncio.gather(
            check_config(settings),
            check_memgraph(graph, settings.memgraph),
            check_embeddings(embed, settings.embeddings),
            check_valkey(settings.redis),
        )

        results = [config_res, mg_res, embed_res, valkey_res]

        # Phase 2: Memgraph-dependent checks
        if mg_res.status == CheckStatus.FAIL:
            results.append(CheckResult("schema", CheckStatus.FAIL, _SKIPPED_DETAIL))
            results.append(CheckResult("embedding_model", CheckStatus.FAIL, _SKIPPED_DETAIL))
            results.append(CheckResult("index", CheckStatus.FAIL, _SKIPPED_DETAIL))
        else:
            assert graph is not None
            schema_res, model_res, index_res = await asyncio.gather(
                check_schema(graph),
                check_embedding_model(graph, settings.embeddings),
                check_index(graph, settings),
            )
            results.append(schema_res)
            results.append(model_res)
            results.append(index_res)
    finally:
        if own_graph:
            assert graph is not None
            await graph.close()

    elapsed = (time.monotonic() - t0) * 1000
    return HealthReport(checks=results, elapsed_ms=elapsed)
