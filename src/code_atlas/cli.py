"""CLI entrypoint for Code Atlas."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import asdict, dataclass
from typing import Any

import typer
from dotenv import load_dotenv
from loguru import logger

load_dotenv()  # Load .env into os.environ (ATLAS_* + provider API keys)

app = typer.Typer(
    name="atlas",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Output mode (global flags)
# ---------------------------------------------------------------------------


@dataclass
class OutputMode:
    quiet: bool = False
    json: bool = False
    verbose: int = 0  # 0=normal, 1=debug, 2=trace
    no_color: bool = False


_output = OutputMode()


def _configure_logger() -> None:
    """Reconfigure loguru based on global output flags."""
    logger.remove()

    if _output.json:
        # JSON mode: only errors on stderr, no formatting noise
        logger.add(sys.stderr, level="ERROR", colorize=False, format="{message}")
        return

    if _output.quiet:
        level = "WARNING"
    elif _output.verbose >= 2:
        level = "TRACE"
    elif _output.verbose >= 1:
        level = "DEBUG"
    else:
        level = "INFO"

    logger.add(sys.stderr, level=level, colorize=False if _output.no_color else None)


def _json_output(payload: dict[str, Any]) -> None:
    """Write a JSON object to stdout."""
    import json as _json

    print(_json.dumps(payload, indent=2, default=str))


@app.callback()
def main(
    quiet: bool = typer.Option(False, "--quiet", "-q", envvar="ATLAS_QUIET", help="Suppress info output (CI mode)."),
    json_flag: bool = typer.Option(False, "--json", envvar="ATLAS_JSON", help="Machine-readable JSON output."),
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (-v debug, -vv trace)."),
    no_color: bool = typer.Option(False, "--no-color", envvar="NO_COLOR", help="Disable colored output."),
) -> None:
    """Code Atlas — map your codebase, search it three ways, feed it to agents."""
    _output.quiet = quiet
    _output.json = json_flag
    _output.verbose = verbose
    _output.no_color = no_color
    _configure_logger()


daemon_app = typer.Typer(name="daemon", help="Manage the Code Atlas indexing daemon.")
app.add_typer(daemon_app)


@app.command()
def index(
    path: str = typer.Argument(".", help="Path to the project root to index."),
    scope: list[str] | None = typer.Option(None, help="Scope indexing to specific paths (repeatable)."),
    project: list[str] | None = typer.Option(
        None, "--project", "-p", help="Index specific sub-projects (repeatable, globs)."
    ),
    full_reindex: bool = typer.Option(False, "--full", help="Force full re-index, ignoring delta cache."),
) -> None:
    """Index a codebase into the graph."""
    asyncio.run(_run_index(path, scope, full_reindex, projects=project))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (natural language, keyword, or identifier)."),
    type_: str = typer.Option("hybrid", "--type", "-t", help="Search type: hybrid, graph, vector, bm25."),
    scope: str | None = typer.Option(None, help="Scope search to a project name."),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results to return."),
    include_tests: bool = typer.Option(False, "--include-tests", help="Include test entities in results."),
    include_stubs: bool = typer.Option(False, "--include-stubs", help="Include .pyi type stubs in results."),
    include_generated: bool = typer.Option(False, "--include-generated", help="Include generated code in results."),
) -> None:
    """Search the code graph."""
    asyncio.run(
        _run_search(
            query,
            type_,
            scope,
            limit,
            exclude_tests=False if include_tests else None,
            exclude_stubs=False if include_stubs else None,
            exclude_generated=False if include_generated else None,
        )
    )


@app.command()
def status() -> None:
    """Show index status and health."""
    asyncio.run(_run_status())


@app.command()
def health() -> None:
    """Quick infrastructure health check (exit 0 = ok, 1 = any failed)."""
    asyncio.run(_run_health())


@app.command()
def doctor() -> None:
    """Detailed diagnostic report with fix suggestions."""
    asyncio.run(_run_doctor())


@app.command()
def watch(
    path: str = typer.Argument(".", help="Path to the project root to watch."),
    debounce: float | None = typer.Option(None, "--debounce", help="Debounce timer in seconds (default: 5)."),
    max_wait: float | None = typer.Option(None, "--max-wait", help="Max-wait ceiling in seconds (default: 30)."),
) -> None:
    """Watch a project for file changes and auto-index."""
    try:
        asyncio.run(_run_watch(path, debounce=debounce, max_wait=max_wait))
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio, streamable-http"),
    strict: bool = typer.Option(False, "--strict", help="Refuse to start if embedding model mismatch."),
) -> None:
    """Start the MCP server for AI agent connections."""
    from code_atlas.server.mcp import create_mcp_server
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = AtlasSettings()
    init_telemetry(settings.observability)
    try:
        server = create_mcp_server(settings, strict=strict)
        logger.info("Starting MCP server (transport={})", transport)
        server.run(transport=transport)  # type: ignore[arg-type]  # typer gives str, FastMCP expects Literal
    finally:
        shutdown_telemetry()


# ---------------------------------------------------------------------------
# Index / Status async helpers
# ---------------------------------------------------------------------------


async def _run_index(  # noqa: PLR0915
    path: str,
    scope: list[str] | None,
    full_reindex: bool,
    *,
    projects: list[str] | None = None,
) -> None:
    """Async implementation of the ``atlas index`` command."""
    from pathlib import Path

    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.orchestrator import detect_sub_projects, index_monorepo, index_project
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    project_root = Path(path).resolve()
    settings = AtlasSettings(project_root=project_root)
    init_telemetry(settings.observability)

    # Connect to Valkey
    bus = EventBus(settings.redis)
    try:
        await bus.ping()
    except Exception as exc:
        logger.error("Cannot reach Valkey at {}:{} — {}", settings.redis.host, settings.redis.port, exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to Valkey at {}:{}", settings.redis.host, settings.redis.port)

    # Resolve embedding dimension before graph construction (vector indices need it)
    if settings.embeddings.dimension is None:
        from code_atlas.search.embeddings import EmbedClient as _EmbedClient

        _probe = _EmbedClient(settings.embeddings)
        try:
            resolved_dim = await _probe.detect_dimension()
        except Exception as exc:
            logger.error("Cannot auto-detect embedding dimension: {}", exc)
            logger.error("Set 'dimension' in atlas.toml [embeddings] or start the embedding service.")
            await bus.close()
            raise typer.Exit(code=1) from exc
        settings.embeddings.dimension = resolved_dim
        logger.info("Auto-detected embedding dimension: {}", resolved_dim)

    # Connect to Memgraph
    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
        await bus.close()
        raise typer.Exit(code=1) from exc
    logger.info("Connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)

    await graph.ensure_schema()

    try:
        # Auto-detect monorepo: if sub-projects detected or --project specified → monorepo mode
        sub_projects = detect_sub_projects(project_root, settings.monorepo)
        is_monorepo = bool(sub_projects) or bool(projects)

        if is_monorepo:
            results = await index_monorepo(
                settings,
                graph,
                bus,
                scope_projects=projects,
                full_reindex=full_reindex,
            )
            total_files = sum(r.files_scanned for r in results)
            total_entities = sum(r.entities_total for r in results)
            total_duration = sum(r.duration_s for r in results)
            if _output.json:
                _json_output(
                    {
                        "projects": [asdict(r) for r in results],
                        "total_files": total_files,
                        "total_entities": total_entities,
                        "total_duration_s": round(total_duration, 1),
                    }
                )
            else:
                logger.info(
                    "Monorepo indexing complete — {} sub-project(s), {} files, {} entities, {:.1f}s total",
                    len(results),
                    total_files,
                    total_entities,
                    total_duration,
                )
        else:
            result = await index_project(
                settings,
                graph,
                bus,
                scope_paths=scope or None,
                full_reindex=full_reindex,
            )
            if _output.json:
                _json_output(asdict(result))
            else:
                logger.info(
                    "Done ({}) — {} files scanned, {} published, {} entities in {:.1f}s",
                    result.mode,
                    result.files_scanned,
                    result.files_published,
                    result.entities_total,
                    result.duration_s,
                )
                if result.delta_stats is not None:
                    ds = result.delta_stats
                    logger.info(
                        "Delta: files +{} ~{} -{} | entities +{} ~{} -{} ={} unchanged",
                        ds.files_added,
                        ds.files_modified,
                        ds.files_deleted,
                        ds.entities_added,
                        ds.entities_modified,
                        ds.entities_deleted,
                        ds.entities_unchanged,
                    )
    finally:
        await graph.close()
        await bus.close()
        shutdown_telemetry()


async def _run_search(  # noqa: PLR0915
    query: str,
    type_: str,
    scope: str | None,
    limit: int,
    *,
    exclude_tests: bool | None = None,
    exclude_stubs: bool | None = None,
    exclude_generated: bool | None = None,
) -> None:
    """Async implementation of the ``atlas search`` command."""
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.orchestrator import StalenessChecker
    from code_atlas.search.embeddings import EmbedClient
    from code_atlas.search.engine import SearchType, hybrid_search
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = AtlasSettings()
    init_telemetry(settings.observability)
    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
        raise typer.Exit(code=1) from exc

    # Map CLI type names to SearchType lists
    type_map: dict[str, list[SearchType] | None] = {
        "hybrid": None,  # all channels
        "graph": [SearchType.GRAPH],
        "vector": [SearchType.VECTOR],
        "bm25": [SearchType.BM25],
    }
    search_types = type_map.get(type_)
    if type_ not in type_map:
        logger.error("Unknown search type '{}' — use hybrid, graph, vector, or bm25", type_)
        await graph.close()
        raise typer.Exit(code=1)

    # Check model lock — warn and disable vector if mismatch
    embed: EmbedClient | None = None
    stored_config = await graph.get_embedding_config()
    model_mismatch = stored_config is not None and stored_config[0] != settings.embeddings.model
    if model_mismatch:
        stored_model = stored_config[0]  # type: ignore[index]
        if search_types and SearchType.VECTOR in search_types:
            logger.error(
                "Cannot use vector search: model mismatch (stored='{}', current='{}'). Run 'atlas index --full'.",
                stored_model,
                settings.embeddings.model,
            )
            await graph.close()
            raise typer.Exit(code=1)
        logger.warning(
            "Embedding model mismatch (stored='{}', current='{}') — vector search disabled",
            stored_model,
            settings.embeddings.model,
        )

    if not model_mismatch and (search_types is None or SearchType.VECTOR in search_types):
        embed = EmbedClient(settings.embeddings)

    try:
        results = await hybrid_search(
            graph=graph,
            embed=embed,
            settings=settings.search,
            query=query,
            search_types=search_types,
            limit=limit,
            scope=scope or "",
            exclude_tests=exclude_tests,
            exclude_stubs=exclude_stubs,
            exclude_generated=exclude_generated,
        )

        # Staleness check (before output so JSON can include it)
        checker = StalenessChecker(settings.project_root)
        info = await checker.check(graph, include_changed=True)

        if _output.json:
            _json_output(
                {
                    "query": query,
                    "type": type_,
                    "results": [asdict(r) for r in results],
                    "stale": info.stale if info else None,
                }
            )
            return

        if not results:
            logger.info("No results found for '{}'", query)
            return
        for i, r in enumerate(results, 1):
            sources = ", ".join(f"{ch}#{rank}" for ch, rank in r.sources.items())
            loc = f"{r.file_path}:{r.line_start}" if r.file_path and r.line_start else ""
            logger.info(
                "{}. {} ({}) — rrf={:.4f} [{}] {}",
                i,
                r.qualified_name or r.name,
                r.kind or ", ".join(r.labels),
                r.rrf_score,
                sources,
                loc,
            )

        if info.stale:
            commit_str = info.last_indexed_commit[:8] if info.last_indexed_commit else "never"
            logger.warning("Index is stale (last indexed: {})", commit_str)
            if info.changed_files:
                logger.warning("  {} file(s) changed since last index", len(info.changed_files))
    finally:
        await graph.close()
        shutdown_telemetry()


async def _run_status() -> None:
    """Async implementation of the ``atlas status`` command."""
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings()
    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
        raise typer.Exit(code=1) from exc

    try:
        projects = await graph.get_project_status()

        import datetime

        # Collect DEPENDS_ON relationships
        depends_on = await graph.execute(
            "MATCH (a:Project)-[:DEPENDS_ON]->(b:Project) RETURN a.name AS from_proj, b.name AS to_proj"
        )
        deps_by_project: dict[str, list[str]] = {}
        for row in depends_on:
            deps_by_project.setdefault(row["from_proj"], []).append(row["to_proj"])

        if _output.json:
            _json_output(
                {
                    "projects": [
                        {
                            "name": row["n"].get("name"),
                            "last_indexed_at": (
                                datetime.datetime.fromtimestamp(
                                    row["n"]["last_indexed_at"], tz=datetime.UTC
                                ).isoformat()
                                if row["n"].get("last_indexed_at")
                                else None
                            ),
                            "file_count": row["n"].get("file_count"),
                            "entity_count": row["n"].get("entity_count"),
                            "git_hash": row["n"].get("git_hash"),
                            "depends_on": sorted(deps_by_project.get(row["n"].get("name", ""), [])),
                        }
                        for row in projects
                    ],
                }
            )
            return

        if not projects:
            logger.info("No indexed projects found.")
            return

        for row in projects:
            node = row["n"]
            name = node.get("name", "?")
            last = node.get("last_indexed_at")
            files = node.get("file_count", "?")
            entities = node.get("entity_count", "?")
            git_hash = node.get("git_hash", "?")

            ts = datetime.datetime.fromtimestamp(last, tz=datetime.UTC).isoformat() if last else "never"
            deps = deps_by_project.get(name, [])
            deps_str = f" | depends_on: {', '.join(sorted(deps))}" if deps else ""
            logger.info(
                "Project: {} | indexed: {} | files: {} | entities: {} | git: {}{}",
                name,
                ts,
                files,
                entities,
                git_hash,
                deps_str,
            )
    finally:
        await graph.close()


# ---------------------------------------------------------------------------
# Health / Doctor async helpers
# ---------------------------------------------------------------------------


async def _run_health() -> None:
    from code_atlas.server.health import run_health_checks
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings()
    report = await run_health_checks(settings)
    _print_report(report, detailed=False)
    raise typer.Exit(code=0 if report.ok else 1)


async def _run_doctor() -> None:
    from code_atlas.server.health import run_health_checks
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings()
    report = await run_health_checks(settings)
    _print_report(report, detailed=True)
    raise typer.Exit(code=0 if report.ok else 1)


def _print_report(report: object, *, detailed: bool) -> None:
    from code_atlas.server.health import CheckStatus, HealthReport

    rpt: HealthReport = report  # type: ignore[assignment]

    if _output.json:
        _json_output(
            {
                "ok": rpt.ok,
                "checks": [
                    {
                        "name": c.name,
                        "status": c.status.value,
                        "message": c.message,
                        "detail": c.detail,
                        "suggestion": c.suggestion,
                    }
                    for c in rpt.checks
                ],
                "elapsed_ms": round(rpt.elapsed_ms, 1),
            }
        )
        return

    status_icon = {
        CheckStatus.OK: "<green>\u2713</green>",
        CheckStatus.WARN: "<yellow>!</yellow>",
        CheckStatus.FAIL: "<red>\u2717</red>",
    }

    for c in rpt.checks:
        icon = status_icon.get(c.status, "?")
        logger.opt(colors=True).info("{} {:<20} {}", icon, c.name, c.message)
        if detailed:
            if c.detail:
                logger.info("    {}", c.detail)
            if c.suggestion:
                logger.opt(colors=True).info("    <dim>Suggestion: {}</dim>", c.suggestion)

    logger.info("Completed in {:.0f}ms", rpt.elapsed_ms)


# ---------------------------------------------------------------------------
# Watch async helper
# ---------------------------------------------------------------------------


async def _run_watch(path: str, *, debounce: float | None, max_wait: float | None) -> None:
    """Async implementation of the ``atlas watch`` command."""
    from pathlib import Path

    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    project_root = Path(path).resolve()
    settings = AtlasSettings(project_root=project_root)
    init_telemetry(settings.observability)
    if debounce is not None:
        settings.watcher.debounce_s = debounce
    if max_wait is not None:
        settings.watcher.max_wait_s = max_wait

    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)
    await graph.ensure_schema()

    daemon = DaemonManager()
    started = await daemon.start(settings, graph)
    if not started:
        logger.error("Valkey required for watch mode")
        await graph.close()
        raise typer.Exit(code=1)

    try:
        await daemon.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.stop()
        await graph.close()
        shutdown_telemetry()
        logger.info("Watch stopped")


# ---------------------------------------------------------------------------
# Daemon subcommands
# ---------------------------------------------------------------------------


async def _run_daemon() -> None:
    """Start the EventBus and all tier consumers, run until interrupted."""
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = AtlasSettings()
    init_telemetry(settings.observability)

    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)
    await graph.ensure_schema()

    daemon = DaemonManager()
    started = await daemon.start(settings, graph, include_watcher=False)
    if not started:
        logger.error("Valkey required for daemon mode")
        await graph.close()
        raise typer.Exit(code=1)

    try:
        await daemon.wait()
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.stop()
        await graph.close()
        shutdown_telemetry()
        logger.info("Daemon stopped")


@daemon_app.command("start")
def daemon_start(
    foreground: bool = typer.Option(True, "--foreground/--background", help="Run in foreground (Ctrl+C to stop)."),
) -> None:
    """Start the indexing daemon (file watcher + tier consumers)."""
    if not foreground:
        logger.error("Background mode not yet implemented — use --foreground")
        raise typer.Exit(code=1)

    logger.info("Starting Code Atlas daemon (foreground)")
    try:
        asyncio.run(_run_daemon())
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")


if __name__ == "__main__":
    app()
