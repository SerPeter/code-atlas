"""CLI entrypoint for Code Atlas."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from rich.console import Console

_dotenv_path = find_dotenv(usecwd=True)  # '' when not found
load_dotenv(_dotenv_path)  # Load .env into os.environ (ATLAS_* + provider API keys)

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
    verbose: int = 0  # 0=warning, 1=info, 2=debug, 3=trace
    no_color: bool = False


_output = OutputMode()

# Shared Rich console — used by both loguru sink and Progress bars so Rich can
# coordinate output (log lines render *above* any live progress bar).
_console = Console(stderr=True)


def _configure_logger() -> None:
    """Reconfigure loguru based on global output flags."""
    logger.remove()
    logger.configure(extra={"consumer": ""})

    if _output.json:
        # JSON mode: only errors on stderr, no formatting noise
        logger.add(sys.stderr, level="ERROR", colorize=False, format="{message}")
        return

    if _output.quiet:
        level = "ERROR"
    elif _output.verbose >= 3:
        level = "TRACE"
    elif _output.verbose >= 2:
        level = "DEBUG"
    elif _output.verbose >= 1:
        level = "INFO"
    else:
        level = "WARNING"

    if _output.verbose >= 2:
        fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[consumer]:<14} | {name}:{function}:{line} - {message}"
        )
    elif _output.verbose >= 1:
        fmt = "{time:HH:mm:ss.SSS} | {level:<8} | {message}"
    else:
        fmt = "{message}"

    # Route loguru through the shared Rich console so log lines render above
    # any active Progress bar instead of clobbering it.
    def _rich_sink(message: str) -> None:
        _console.print(message, end="", highlight=False, markup=False)

    logger.add(_rich_sink, level=level, colorize=not _output.no_color, format=fmt)


def _echo(msg: str) -> None:
    """Print a message to stderr (visible in default mode, suppressed by --json/--quiet)."""
    if not _output.json and not _output.quiet:
        typer.echo(msg, err=True)


def _json_output(payload: dict[str, Any]) -> None:
    """Write a JSON object to stdout."""
    import json as _json

    print(_json.dumps(payload, indent=2, default=str))


@app.callback()
def main(
    quiet: bool = typer.Option(False, "--quiet", "-q", envvar="ATLAS_QUIET", help="Suppress info output (CI mode)."),
    json_flag: bool = typer.Option(False, "--json", envvar="ATLAS_JSON", help="Machine-readable JSON output."),
    verbose: int = typer.Option(
        0, "--verbose", "-v", count=True, help="Increase verbosity (-v info, -vv debug, -vvv trace)."
    ),
    no_color: bool = typer.Option(False, "--no-color", envvar="NO_COLOR", help="Disable colored output."),
) -> None:
    """Code Atlas — map your codebase, search it three ways, feed it to agents."""
    _output.quiet = quiet
    _output.json = json_flag
    _output.verbose = verbose
    _output.no_color = no_color
    if no_color:
        _console.no_color = True
    _configure_logger()


daemon_app = typer.Typer(name="daemon", help="Manage the Code Atlas indexing daemon.")
app.add_typer(daemon_app)


# ---------------------------------------------------------------------------
# Git root resolution
# ---------------------------------------------------------------------------


def _resolve_project_root(path: str, *, no_git_check: bool = False) -> tuple[Path, str | None]:
    """Resolve project root from a user-supplied path.

    Walks up to find the git root. If the target is a subdirectory of the repo,
    returns the git root as project root and the relative path as a scope prefix.

    Returns ``(project_root, scope_prefix)`` — *scope_prefix* is ``None`` when
    *path* IS the git root.
    """
    from code_atlas.settings import find_git_root

    target = Path(path).resolve()
    if no_git_check:
        return target, None

    git_root = find_git_root(target)
    if git_root is None:
        logger.error("No git repository found at or above {}", target)
        logger.error("Use --no-git-check to index a non-git directory")
        raise typer.Exit(code=1)

    if git_root == target:
        return git_root, None

    scope_prefix = target.relative_to(git_root).as_posix()
    logger.info("Git root: {} — auto-scoping to {}/", git_root, scope_prefix)
    return git_root, scope_prefix


@app.command()
def index(
    path: str = typer.Argument(".", help="Path to the project root to index."),
    scope: list[str] | None = typer.Option(None, help="Scope indexing to specific paths (repeatable)."),
    project: list[str] | None = typer.Option(
        None, "--project", "-p", help="Index specific sub-projects (repeatable, globs)."
    ),
    full_reindex: bool = typer.Option(False, "--full", help="Force full re-index, ignoring delta cache."),
    no_embed: bool = typer.Option(False, "--no-embed", help="Disable embeddings (lightweight mode)."),
    no_git_check: bool = typer.Option(False, "--no-git-check", help="Allow indexing outside a git repository."),
) -> None:
    """Index a codebase into the graph."""
    asyncio.run(_run_index(path, scope, full_reindex, projects=project, no_embed=no_embed, no_git_check=no_git_check))


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
    no_git_check: bool = typer.Option(False, "--no-git-check", help="Allow watching outside a git repository."),
) -> None:
    """Watch a project for file changes and auto-index."""
    try:
        asyncio.run(_run_watch(path, debounce=debounce, max_wait=max_wait, no_git_check=no_git_check))
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")


@app.command()
def mcp(
    transport: str = typer.Option(None, "--transport", "-t", help="Transport: stdio, streamable-http."),
    host: str = typer.Option(None, "--host", help="Bind address for HTTP transports (ignored for stdio)."),
    port: int = typer.Option(None, "--port", "-p", help="Bind port for HTTP transports (ignored for stdio)."),
    strict: bool = typer.Option(None, "--strict", help="Refuse to start if embedding model mismatch."),
) -> None:
    """Start the MCP server for AI agent connections."""
    from code_atlas.server.mcp import create_mcp_server
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = AtlasSettings()
    # CLI args override settings (None = use settings default)
    mcp_cfg = settings.mcp
    transport = transport or mcp_cfg.transport
    host = host or mcp_cfg.host
    port = port or mcp_cfg.port
    strict = strict if strict is not None else mcp_cfg.strict

    init_telemetry(settings.observability)
    try:
        server = create_mcp_server(settings, strict=strict, host=host, port=port)
        logger.info("Starting MCP server (transport={}, host={}, port={})", transport, host, port)
        server.run(transport=transport)  # type: ignore[arg-type]  # typer gives str, FastMCP expects Literal
    finally:
        shutdown_telemetry()


# ---------------------------------------------------------------------------
# Index / Status async helpers
# ---------------------------------------------------------------------------


async def _run_index(  # noqa: PLR0912, PLR0915
    path: str,
    scope: list[str] | None,
    full_reindex: bool,
    *,
    projects: list[str] | None = None,
    no_embed: bool = False,
    no_git_check: bool = False,
) -> None:
    """Async implementation of the ``atlas index`` command."""
    from code_atlas.events import EventBus
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.orchestrator import detect_sub_projects
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    project_root, auto_scope = _resolve_project_root(path, no_git_check=no_git_check)
    if auto_scope:
        scope = [auto_scope, *(scope or [])]
    settings = AtlasSettings(project_root=project_root)
    if no_embed:
        settings.embeddings.enabled = False
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
    if settings.embeddings.enabled and settings.embeddings.dimension is None:
        from code_atlas.search.embeddings import EmbedClient as _EmbedClient

        _probe = _EmbedClient(settings.embeddings)
        try:
            resolved_dim = await _probe.detect_dimension()
        except Exception:
            logger.warning("Embedding service unreachable — running in lightweight mode. Vector search disabled.")
            settings.embeddings.enabled = False
            resolved_dim = None
        if resolved_dim is not None:
            settings.embeddings.dimension = resolved_dim
            logger.debug("Auto-detected embedding dimension: {}", resolved_dim)

    if not settings.embeddings.enabled:
        logger.info("Lightweight mode: embeddings disabled, using graph + BM25 only")

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
            results = await _index_monorepo_with_progress(
                settings, graph, bus, projects=projects, full_reindex=full_reindex
            )
            total_files = sum(r.files_scanned for r in results)
            total_entities = sum(r.entities_total for r in results)
            total_duration = max((r.duration_s for r in results), default=0.0)
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
                _echo(
                    f"Done — {len(results)} projects, {total_files} files,"
                    f" {total_entities} entities in {total_duration:.1f}s"
                )
        else:
            result = await _index_single_with_spinner(settings, graph, bus, scope=scope, full_reindex=full_reindex)
            if _output.json:
                _json_output(asdict(result))
            else:
                _echo(
                    f"Done ({result.mode}) — {result.files_scanned} files,"
                    f" {result.entities_total} entities in {result.duration_s:.1f}s"
                )
                if result.delta_stats is not None:
                    ds = result.delta_stats
                    _echo(
                        f"Delta: files +{ds.files_added} ~{ds.files_modified} -{ds.files_deleted}"
                        f" | entities +{ds.entities_added} ~{ds.entities_modified} -{ds.entities_deleted}"
                        f" ={ds.entities_unchanged} unchanged"
                    )
    finally:
        await graph.close()
        await bus.close()
        shutdown_telemetry()


async def _index_monorepo_with_progress(
    settings: Any,
    graph: Any,
    bus: Any,
    *,
    projects: list[str] | None,
    full_reindex: bool,
) -> list[Any]:
    """Run monorepo indexing with a Rich progress bar (unless --json or --quiet)."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from code_atlas.indexing.orchestrator import IndexResult, index_monorepo

    show_progress = not _output.json and not _output.quiet

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        disable=not show_progress,
        console=_console,
    ) as progress:
        task = progress.add_task("Indexing", total=None)

        def on_progress(name: str, current: int, total: int) -> None:
            progress.update(task, total=total, completed=current, description=name)

        drain_prev_remaining: int | None = None
        drain_processed = 0

        def on_drain(t1: int, t2: int, t3: int) -> None:
            nonlocal drain_prev_remaining, drain_processed
            remaining = t1 + t2 + t3
            if drain_prev_remaining is None:
                # First drain tick — switch from project-count to event-count bar
                drain_processed = 0
            else:
                consumed = drain_prev_remaining - remaining
                if consumed > 0:
                    drain_processed += consumed
            drain_prev_remaining = remaining
            total = drain_processed + remaining
            if remaining > 0:
                progress.update(task, total=total, completed=drain_processed, description="Processing events")
            else:
                progress.update(task, total=total, completed=total, description="Done")

        results: list[IndexResult] = await index_monorepo(
            settings,
            graph,
            bus,
            scope_projects=projects,
            full_reindex=full_reindex,
            on_progress=on_progress,
            on_drain_progress=on_drain,
        )

    return results


async def _index_single_with_spinner(
    settings: Any,
    graph: Any,
    bus: Any,
    *,
    scope: list[str] | None,
    full_reindex: bool,
) -> Any:
    """Run single-project indexing with a Rich spinner (unless --json or --quiet)."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from code_atlas.indexing.orchestrator import index_project

    show_progress = not _output.json and not _output.quiet

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        TimeElapsedColumn(),
        disable=not show_progress,
        console=_console,
    ) as progress:
        task = progress.add_task("Indexing...", total=None)

        def on_drain(t1: int, t2: int, t3: int) -> None:
            remaining = t1 + t2 + t3
            if remaining > 0:
                progress.update(task, description=f"Processing {remaining} event(s)...")
            else:
                progress.update(task, description="Finalizing...")

        result = await index_project(
            settings,
            graph,
            bus,
            scope_paths=scope or None,
            full_reindex=full_reindex,
            on_drain_progress=on_drain,
        )

    return result  # noqa: RET504


async def _run_search(  # noqa: PLR0912, PLR0915
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

    # Check embeddings disabled — error on explicit vector search
    if not settings.embeddings.enabled and search_types and SearchType.VECTOR in search_types:
        logger.error("Vector search unavailable — embeddings are disabled")
        await graph.close()
        raise typer.Exit(code=1)

    # Check model lock — warn and disable vector if mismatch
    embed: EmbedClient | None = None
    if settings.embeddings.enabled:
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
            _echo(f"No results found for '{query}'")
            return
        for i, r in enumerate(results, 1):
            sources = ", ".join(f"{ch}#{rank}" for ch, rank in r.sources.items())
            loc = f"{r.file_path}:{r.line_start}" if r.file_path and r.line_start else ""
            kind = r.kind or ", ".join(r.labels)
            _echo(f"{i}. {r.qualified_name or r.name} ({kind}) — rrf={r.rrf_score:.4f} [{sources}] {loc}")

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
            _echo("No indexed projects found.")
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
            _echo(
                f"Project: {name} | indexed: {ts} | files: {files} | entities: {entities} | git: {git_hash}{deps_str}"
            )
    finally:
        await graph.close()


# ---------------------------------------------------------------------------
# Health / Doctor async helpers
# ---------------------------------------------------------------------------


async def _run_health() -> None:
    from code_atlas.server.health import run_health_checks
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings(project_root=Path.cwd())
    report = await run_health_checks(settings, dotenv_path=_dotenv_path)
    _print_report(report, detailed=False)
    raise typer.Exit(code=0 if report.ok else 1)


async def _run_doctor() -> None:
    from code_atlas.server.health import run_health_checks
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings(project_root=Path.cwd())
    report = await run_health_checks(settings, dotenv_path=_dotenv_path)
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

    if _output.quiet:
        return

    status_icon = {
        CheckStatus.OK: "[green]\u2713[/green]",
        CheckStatus.WARN: "[yellow]![/yellow]",
        CheckStatus.FAIL: "[red]\u2717[/red]",
    }

    console = _console
    for c in rpt.checks:
        icon = status_icon.get(c.status, "?")
        console.print(f"{icon} {c.name:<20} {c.message}")
        if detailed:
            if c.detail:
                console.print(f"    {c.detail}")
            if c.suggestion:
                console.print(f"    [dim]Suggestion: {c.suggestion}[/dim]")

    _echo(f"Completed in {rpt.elapsed_ms:.0f}ms")


# ---------------------------------------------------------------------------
# Watch async helper
# ---------------------------------------------------------------------------


async def _run_watch(path: str, *, debounce: float | None, max_wait: float | None, no_git_check: bool = False) -> None:
    """Async implementation of the ``atlas watch`` command."""
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    project_root, _auto_scope = _resolve_project_root(path, no_git_check=no_git_check)
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


async def _run_daemon(*, no_embed: bool = False) -> None:
    """Start the EventBus and all tier consumers, run until interrupted."""
    from code_atlas.graph.client import GraphClient
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import AtlasSettings
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = AtlasSettings()
    if no_embed:
        settings.embeddings.enabled = False
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
    no_embed: bool = typer.Option(False, "--no-embed", help="Disable embeddings (lightweight mode)."),
) -> None:
    """Start the indexing daemon (file watcher + tier consumers)."""
    if not foreground:
        logger.error("Background mode not yet implemented — use --foreground")
        raise typer.Exit(code=1)

    logger.info("Starting Code Atlas daemon (foreground)")
    try:
        asyncio.run(_run_daemon(no_embed=no_embed))
    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")


if __name__ == "__main__":
    app()
