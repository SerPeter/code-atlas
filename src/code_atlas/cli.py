"""CLI entrypoint for Code Atlas."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from rich.console import Console

if TYPE_CHECKING:
    from code_atlas.graph.protocol import GraphBackend
    from code_atlas.indexing.orchestrator import IndexResult
    from code_atlas.settings import AtlasSettings

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


def _warn_partial_index(result: IndexResult) -> None:
    """Say so when an index is partial, instead of reporting a clean run.

    Two distinct silences, both of which produced `Done - 4823 files, 0 entities`
    with exit 0 on a TypeScript repo (ATL-110):

    * files the scope wanted that no installed grammar could read -- a default install
      ships only the Python and Markdown grammars, and every other language sits behind
      an extra that is documented nowhere a user would look;
    * a scan that found files and produced nothing, which is either the above or a
      scope so narrow it matched no code. Either way it is not success.

    Written to stderr like the rest of the summary, so `--json` and `--quiet` still get
    a clean stream; the counts ride in the JSON payload instead.
    """
    # Local, matching this module's lazy-import style: the CLI keeps startup cheap by
    # importing the heavy packages only on the paths that need them.
    from code_atlas.parsing.languages import install_hint, missing_grammar_extras

    if result.skipped_no_grammar:
        total = sum(result.skipped_no_grammar.values())
        by_ext = ", ".join(
            f"{ext} x{count}" for ext, count in sorted(result.skipped_no_grammar.items(), key=lambda kv: -kv[1])
        )
        extras = missing_grammar_extras(result.skipped_no_grammar)
        _echo(f"  ! {total} file(s) skipped - no grammar installed: {by_ext}")
        if extras:
            _echo(f"    install with: pip install 'code-atlas-mcp[{install_hint(extras.values())}]'")

    if result.files_scanned > 0 and result.entities_total == 0:
        _echo("  ! Scanned files but produced no entities - the index is empty.")
        if not result.skipped_no_grammar:
            _echo("    Check the [scope] section of atlas.toml, or run 'atlas doctor'.")


def _json_output(payload: dict[str, Any]) -> None:
    """Write a JSON object to stdout."""
    import orjson

    print(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS, default=str).decode())


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

project_app = typer.Typer(name="project", help="Manage indexed projects.")
app.add_typer(project_app)


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


def _load_settings(**overrides: object) -> AtlasSettings:
    """Load ``AtlasSettings``, converting the git-root RuntimeError into a user-friendly exit."""
    from code_atlas.settings import AtlasSettings

    try:
        return AtlasSettings(**overrides)  # type: ignore[arg-type]
    except RuntimeError as exc:
        logger.error("{}", exc)
        raise typer.Exit(code=1) from None


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
    with_git_signals: bool = typer.Option(
        False, "--with-git-signals", help="Mine git history for hotspot/co-change signals after indexing."
    ),
    co_change_threshold: int = typer.Option(
        3,
        "--co-change-threshold",
        help="Minimum shared commits for a CO_CHANGES_WITH edge (used with --with-git-signals).",
    ),
) -> None:
    """Index a codebase into the graph."""
    asyncio.run(
        _run_index(
            path,
            scope,
            full_reindex,
            projects=project,
            no_embed=no_embed,
            no_git_check=no_git_check,
            with_git_signals=with_git_signals,
            co_change_threshold=co_change_threshold,
        )
    )


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


@app.command("mine-git-history")
def mine_git_history(
    path: str = typer.Argument(".", help="Path to the project root to mine git history for."),
    co_change_threshold: int = typer.Option(
        3, "--co-change-threshold", help="Minimum shared commits for a CO_CHANGES_WITH edge."
    ),
    no_git_check: bool = typer.Option(False, "--no-git-check", help="Allow running outside a git repository."),
) -> None:
    """Mine git history for hotspot/bus-factor/co-change signals (see find_hotspots).

    One-shot batch job over the full commit history — not part of the
    continuous indexing pipeline. Re-run periodically (e.g. from CI) to
    refresh the mined signals; results are written onto existing Module/
    DocFile nodes, so index the project first.
    """
    asyncio.run(_run_mine_git_history(path, co_change_threshold, no_git_check=no_git_check))


@app.command()
def dream() -> None:
    """Deterministic dream-mode report: inbox, orphans, dangling links, duplicates, similarity.

    Scans this project's vault plus any configured extra vaults, refreshes
    HOME.md inside the vault, and prints the report (--json for machine-readable output).
    The disposition step (KEEP/MERGE/PROMOTE/DROP) is agent-side — see the
    dream-mode command.
    """
    asyncio.run(_run_dream())


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address. Non-loopback exposes the whole graph."),
    port: int = typer.Option(
        8420, "--port", "-p", help="Preferred bind port. Moves to the next free one if it is taken."
    ),
    project: str = typer.Option("", "--project", help="Project to view. Empty = detect from the working directory."),
    reload: bool = typer.Option(False, "--reload", help="Enable debug mode and template auto-reload."),
    export: Path = typer.Option(
        None, "--export", help="Write a self-contained HTML snapshot to this path instead of serving."
    ),
) -> None:
    """Start the local web interface for exploring the graph."""
    if export is not None:
        asyncio.run(_run_export(path=export, project=project))
        return
    asyncio.run(_run_ui(host=host, port=port, project=project, debug=reload))


async def _run_export(*, path: Path, project: str) -> None:
    """Write a static snapshot instead of serving one.

    A second renderer over the same view services and template partials, not a second
    implementation — see `code_atlas.server.web.export`.
    """
    try:
        import jinja2  # noqa: F401  # presence check; the exporter imports what it needs
    except ImportError:
        logger.error("The HTML export needs the 'ui' extra. Install it with: pip install 'code-atlas-mcp[ui]'")
        raise typer.Exit(code=1) from None

    from code_atlas.backends import create_graph_client
    from code_atlas.settings import derive_project_name

    settings = _load_settings()
    project_name = project or derive_project_name(settings.project_root)
    graph = await create_graph_client(settings)
    try:
        from code_atlas.server.web.export import ProjectNotIndexedError, export_project

        try:
            result = await export_project(graph, project_name, path)
        except ProjectNotIndexedError:
            # Same distinction the CLI draws everywhere else (ATL-110): "not indexed" is
            # not "empty", and writing a file of zeroes would erase the difference.
            logger.error("Project '{}' has no index. Run 'atlas index' first.", project_name)
            raise typer.Exit(code=1) from None

        _echo(f"Wrote {result.path} — {result.size_mb:.1f} MB, {result.node_count} modules.")
        if not result.map_available:
            _echo("  The map is not in this export: community detection needs the Memgraph backend.")
        _echo("  Self-contained: open it directly, no server and no network.")
    finally:
        await graph.close()


async def _run_ui(*, host: str, port: int, project: str, debug: bool) -> None:
    """Serve the web UI against the configured backend.

    The `ui` extra is optional, so its absence is reported the way a missing grammar is
    (ATL-110): name the thing and the command that installs it, never fail obscurely on
    an ImportError traceback.
    """
    try:
        import uvicorn
    except ImportError:
        logger.error("The web UI needs the 'ui' extra. Install it with: pip install 'code-atlas-mcp[ui]'")
        raise typer.Exit(code=1) from None

    from code_atlas.backends import create_graph_client
    from code_atlas.server.web.instances import claim_port, live_instances, registered
    from code_atlas.settings import derive_project_name
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = _load_settings()
    project_name = project or derive_project_name(settings.project_root)
    # The UI was the one entry point that never initialised telemetry, so it produced
    # no signals at all even with everything else exporting.
    init_telemetry(
        settings.observability,
        role="web",
        project=project_name,
        root=str(settings.project_root),
        indexing=False,
    )
    graph = await create_graph_client(settings)
    try:
        from code_atlas.server.web.app import create_app

        app_instance = create_app(graph, project_name, debug=debug)

        if host not in ("127.0.0.1", "localhost", "::1"):
            # The UI is unauthenticated and reaches the entire graph. Binding it
            # outward is a deliberate act and is worth saying out loud once.
            logger.warning("Binding {} — the UI has no authentication and exposes the whole graph.", host)

        # Several UIs at once is the normal case, not an error: one per worktree, and
        # sometimes two sessions in the same one. They all defaulted to 8420, so the
        # second invocation died on "address already in use". Bind here rather than
        # letting uvicorn do it, and hand the socket over: a check-then-bind would leave
        # two simultaneous invocations able to pick the same port and both think they won.
        peers = live_instances()
        try:
            sock, port = claim_port(host, port)
        except OSError as exc:
            logger.error("{}", exc)
            raise typer.Exit(code=1) from exc

        _echo(f"code-atlas UI for '{project_name}' — http://{host}:{port}")
        for peer in peers:
            if peer.port != port:
                _echo(f"  also serving: {peer.url} — {peer.project}")

        config = uvicorn.Config(app_instance, host=host, port=port, log_level="warning")
        with registered(host, port, project_name, str(settings.project_root)):
            await uvicorn.Server(config).serve(sockets=[sock])
    finally:
        await graph.close()
        shutdown_telemetry()


@app.command()
def mcp(
    transport: str = typer.Option(None, "--transport", "-t", help="Transport: stdio, streamable-http."),
    host: str = typer.Option(None, "--host", help="Bind address for HTTP transports (ignored for stdio)."),
    port: int = typer.Option(None, "--port", "-p", help="Bind port for HTTP transports (ignored for stdio)."),
    strict: bool = typer.Option(None, "--strict", help="Refuse to start if embedding model mismatch."),
    no_index: bool = typer.Option(
        False,
        "--no-index",
        help="Serve queries only: no file watcher, no pipeline, no startup catch-up. "
        "For the second and later agent sessions sharing one worktree — indexing is "
        "per-worktree, so exactly one indexer (a daemon, or one server without this "
        "flag) must still cover that checkout.",
    ),
) -> None:
    """Start the MCP server for AI agent connections."""
    from code_atlas.server.mcp import create_mcp_server
    from code_atlas.settings import derive_project_name
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = _load_settings()
    # CLI args override settings (None = use settings default)
    mcp_cfg = settings.mcp
    transport = transport or mcp_cfg.transport
    host = host or mcp_cfg.host
    port = port or mcp_cfg.port
    strict = strict if strict is not None else mcp_cfg.strict
    # The flag can only switch indexing off, never force it on over a config that
    # deliberately disabled it.
    auto_index = mcp_cfg.auto_index and not no_index

    init_telemetry(
        settings.observability,
        role="mcp",
        project=derive_project_name(settings.project_root),
        root=str(settings.project_root),
        indexing=auto_index,
    )
    try:
        server = create_mcp_server(settings, strict=strict, host=host, port=port, auto_index=auto_index)
        logger.info(
            "Starting MCP server (transport={}, host={}, port={}, indexing={})",
            transport,
            host,
            port,
            "on" if auto_index else "off",
        )
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
    with_git_signals: bool = False,
    co_change_threshold: int = 3,
) -> None:
    """Async implementation of the ``atlas index`` command."""
    from code_atlas.backends import create_event_bus, create_graph_client, graph_backend_label, queue_backend_label
    from code_atlas.indexing.orchestrator import detect_sub_projects
    from code_atlas.settings import AtlasSettings, derive_project_name
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    project_root, auto_scope = _resolve_project_root(path, no_git_check=no_git_check)
    if auto_scope:
        scope = [auto_scope, *(scope or [])]
    settings = AtlasSettings(project_root=project_root)
    if no_embed:
        settings.embeddings.enabled = False
    init_telemetry(
        settings.observability,
        role="index",
        project=derive_project_name(settings.project_root),
        root=str(settings.project_root),
        indexing=True,
    )

    project_name = derive_project_name(settings.project_root)

    # Connect to the event queue backend (Valkey or the SQLite fallback)
    bus = await create_event_bus(settings)
    try:
        await bus.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", queue_backend_label(bus, settings), exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to {}", queue_backend_label(bus, settings))

    # Resolve embedding dimension before graph construction (vector indices need it)
    if settings.embeddings.enabled and settings.embeddings.dimension is None:
        from code_atlas.search.embeddings import EmbedClient as _EmbedClient

        _probe = _EmbedClient(settings.embeddings, settings.redis)
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

    # Connect to the graph backend (Memgraph or the SQLite fallback)
    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
        await bus.close()
        raise typer.Exit(code=1) from exc
    logger.info("Connected to {}", graph_backend_label(graph, settings))

    await graph.ensure_schema()

    # Wait rather than index alongside another process. Two indexers writing the same
    # nodes is how one run got split across two code versions, and how Memgraph's MVCC
    # conflicts turned into dropped files. Waiting is visible (a log line names the
    # holder) and interruptible; refusing outright made a concurrent daemon catch-up
    # into an exit code 1 for a human who only had to wait.
    from code_atlas.events import IndexerBusyError, hold_indexer_lease

    try:
        lease = hold_indexer_lease(bus, wait_s=settings.index.lease_wait_s)
        owner = await lease.__aenter__()
    except IndexerBusyError as exc:
        logger.error("{}", exc)
        await graph.close()
        await bus.close()
        raise typer.Exit(code=1) from exc
    logger.debug("Holding indexer lease ({})", owner)

    # An undrained pipeline means the graph does not reflect the working tree. Printing
    # "Done" and exiting 0 for that made two incomplete indexes read as successes, to a
    # human and to any CI step gating on the exit code.
    incomplete = False

    try:
        # Auto-detect monorepo: if sub-projects detected or --project specified → monorepo mode
        sub_projects = detect_sub_projects(project_root, settings.monorepo)
        is_monorepo = bool(sub_projects) or bool(projects)

        # A --scope path (explicit or auto-derived from a subdirectory target) must
        # not be silently discarded when monorepo mode kicks in — translate it into
        # the sub-project(s) it touches. If it touches none, it's entirely within
        # root-only territory, so fall back to a plain scoped single-project index
        # instead of indexing the whole monorepo.
        if is_monorepo and scope:
            normalized_scope = [s.replace("\\", "/").rstrip("/") for s in scope]
            matched = {
                sp.name
                for sp in sub_projects
                for s in normalized_scope
                if s == sp.path or s.startswith(sp.path + "/") or sp.path.startswith(s + "/")
            }
            if matched:
                projects = sorted(set(projects or []) | matched)
            else:
                is_monorepo = False

        if is_monorepo:
            results = await _index_monorepo_with_progress(
                settings, graph, bus, projects=projects, full_reindex=full_reindex
            )
            total_files = sum(r.files_scanned for r in results)
            total_entities = sum(r.entities_total for r in results)
            total_duration = max((r.duration_s for r in results), default=0.0)

            incomplete = any(not r.drained for r in results)

            git_signals_stats = (
                await _mine_and_write_git_signals(project_root, project_name, graph, co_change_threshold)
                if with_git_signals
                else None
            )

            if _output.json:
                payload = {
                    "projects": [asdict(r) for r in results],
                    "total_files": total_files,
                    "total_entities": total_entities,
                    "total_duration_s": round(total_duration, 1),
                }
                if git_signals_stats is not None:
                    payload["git_signals"] = git_signals_stats
                _json_output(payload)
            else:
                _echo(
                    f"Done — {len(results)} projects, {total_files} files,"
                    f" {total_entities} entities in {total_duration:.1f}s"
                )
                if any(not r.drained for r in results):
                    _echo("WARNING: pipeline did not drain — index incomplete; re-run 'atlas index' to retry")
                if git_signals_stats is not None:
                    _echo(_git_signals_summary_line(git_signals_stats, co_change_threshold))
        else:
            result = await _index_single_with_spinner(settings, graph, bus, scope=scope, full_reindex=full_reindex)
            incomplete = not result.drained

            git_signals_stats = (
                await _mine_and_write_git_signals(project_root, project_name, graph, co_change_threshold)
                if with_git_signals
                else None
            )

            if _output.json:
                payload = asdict(result)
                if git_signals_stats is not None:
                    payload["git_signals"] = git_signals_stats
                _json_output(payload)
            else:
                _echo(
                    f"Done ({result.mode}) — {result.files_scanned} files,"
                    f" {result.entities_total} entities in {result.duration_s:.1f}s"
                )
                _warn_partial_index(result)
                if result.delta_stats is not None:
                    ds = result.delta_stats
                    _echo(
                        f"Delta: files +{ds.files_added} ~{ds.files_modified} -{ds.files_deleted}"
                        f" | entities +{ds.entities_added} ~{ds.entities_modified} -{ds.entities_deleted}"
                        f" ={ds.entities_unchanged} unchanged"
                    )
                if not result.drained:
                    _echo("WARNING: pipeline did not drain — index incomplete; re-run 'atlas index' to retry")
                if git_signals_stats is not None:
                    _echo(_git_signals_summary_line(git_signals_stats, co_change_threshold))
    finally:
        # Release before closing the bus — the release is a compare-and-delete over it.
        await lease.__aexit__(None, None, None)
        await graph.close()
        await bus.close()

    if incomplete:
        raise typer.Exit(code=1)
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

    from code_atlas.indexing.orchestrator import index_monorepo

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
            drain_timeout_s=settings.index.drain_timeout_s,
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
            drain_timeout_s=settings.index.drain_timeout_s,
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
    from code_atlas.backends import create_graph_client, graph_backend_label
    from code_atlas.indexing.orchestrator import StalenessChecker
    from code_atlas.search.embeddings import EmbedClient
    from code_atlas.search.engine import SearchType, hybrid_search
    from code_atlas.settings import derive_project_name
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = _load_settings()
    init_telemetry(
        settings.observability,
        role="search",
        project=derive_project_name(settings.project_root),
        root=str(settings.project_root),
    )
    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
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
        # Per project: the database default belongs to whichever project indexed
        # last, and comparing against it disabled vector search for all the others
        # (ATL-135).
        from code_atlas.settings import derive_project_name

        stored_model = await graph.get_project_embedding_model(derive_project_name(settings.project_root))
        model_mismatch = stored_model is not None and stored_model != settings.embeddings.model
        if model_mismatch:
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
            embed = EmbedClient(settings.embeddings, settings.redis)

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
        info = await checker.check(graph, include_changed=True)  # type: ignore[invalid-argument-type]

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
            _echo(f"{i}. {r.qualified_name or r.name} ({kind}) — score={r.ranked_score:.4f} [{sources}] {loc}")

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
    from code_atlas.backends import create_graph_client, graph_backend_label

    settings = _load_settings()
    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
        raise typer.Exit(code=1) from exc

    try:
        projects = await graph.get_project_status()

        import datetime

        # Collect DEPENDS_ON relationships
        depends_on = await graph.get_project_dependency_edges()
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


@project_app.command("rm")
def project_rm(
    name: str = typer.Argument(..., help="Project name to remove (as shown by 'atlas status')."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete a project's graph data (nodes, relationships, embeddings).

    Mainly for worktree-project cleanup: a linked worktree indexes as its own
    'base@branch' project, and the daemon auto-GCs these once the checkout is
    gone — this command covers the manual case (immediate cleanup, or a
    non-worktree project you want to drop).
    """
    asyncio.run(_run_project_rm(name, skip_confirm=yes))


async def _run_project_rm(name: str, *, skip_confirm: bool) -> None:
    """Async implementation of the ``atlas project rm`` command."""
    from code_atlas.backends import create_graph_client, graph_backend_label

    settings = _load_settings()
    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
        raise typer.Exit(code=1) from exc

    try:
        rows = await graph.get_project_status(name)
        if not rows:
            logger.error("No project named '{}' found in the graph.", name)
            raise typer.Exit(code=1)

        if not skip_confirm:
            if _output.json:
                logger.error(
                    "Refusing to delete '{}' without confirmation in --json mode — pass --yes to confirm.", name
                )
                raise typer.Exit(code=1)
            if not typer.confirm(f"Delete all graph data for '{name}'?"):
                _echo("Aborted.")
                raise typer.Exit(code=1)

        await graph.delete_project_data(name)
        _echo(f"Removed project '{name}'.")
        if _output.json:
            _json_output({"removed": name})
    finally:
        await graph.close()


# ---------------------------------------------------------------------------
# Git signals mining async helper
# ---------------------------------------------------------------------------


async def _run_mine_git_history(path: str, co_change_threshold: int, *, no_git_check: bool) -> None:
    """Async implementation of the ``atlas mine-git-history`` command."""
    from git.exc import InvalidGitRepositoryError, NoSuchPathError

    from code_atlas.backends import create_graph_client, graph_backend_label
    from code_atlas.indexing.git_signals import mine_git_signals, write_git_signals
    from code_atlas.settings import AtlasSettings, derive_project_name

    project_root, _auto_scope = _resolve_project_root(path, no_git_check=no_git_check)
    settings = AtlasSettings(project_root=project_root)
    project_name = derive_project_name(settings.project_root)

    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
        raise typer.Exit(code=1) from exc

    try:
        _echo(f"Mining git history for '{project_name}'...")
        try:
            result = mine_git_signals(project_root, co_change_threshold=co_change_threshold)
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            logger.error("Not a git repository: {} — {}", project_root, exc)
            raise typer.Exit(code=1) from exc

        stats = await write_git_signals(graph, project_name, result)
        if _output.json:
            _json_output(stats)
        else:
            _echo(
                f"Scanned {stats['commits_scanned']} commits — "
                f"{stats['files_matched']}/{stats['files_mined']} files matched, "
                f"{stats['co_change_edges']} co-change edges ({stats['co_change_pairs_mined']} pairs mined, "
                f"threshold={co_change_threshold})"
            )
    finally:
        await graph.close()


def _git_signals_summary_line(stats: dict[str, int], co_change_threshold: int) -> str:
    """Format the git-signals mining summary line for ``atlas index --with-git-signals``."""
    return (
        f"Scanned {stats['commits_scanned']} commits — "
        f"{stats['files_matched']}/{stats['files_mined']} files matched, "
        f"{stats['co_change_edges']} co-change edges ({stats['co_change_pairs_mined']} pairs mined, "
        f"threshold={co_change_threshold})"
    )


async def _mine_and_write_git_signals(
    project_root: Path, project_name: str, graph: GraphBackend, co_change_threshold: int
) -> dict[str, int]:
    """Mine git history and write the resulting signals, for ``atlas index --with-git-signals``.

    Runs the same ``mine_git_signals``/``write_git_signals`` pair that ``atlas
    mine-git-history`` uses (see ``_run_mine_git_history``), invoked here after
    an ``atlas index`` pass completes so signals land on the just-indexed nodes.
    """
    from git.exc import InvalidGitRepositoryError, NoSuchPathError

    from code_atlas.indexing.git_signals import mine_git_signals, write_git_signals

    _echo(f"Mining git history for '{project_name}'...")
    try:
        result = mine_git_signals(project_root, co_change_threshold=co_change_threshold)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        logger.error("Not a git repository: {} — {}", project_root, exc)
        raise typer.Exit(code=1) from exc

    return await write_git_signals(graph, project_name, result)


# ---------------------------------------------------------------------------
# Dream-mode async helper
# ---------------------------------------------------------------------------


async def _run_dream() -> None:
    """Async implementation of the ``atlas dream`` command."""
    from code_atlas.backends import create_graph_client, graph_backend_label
    from code_atlas.dream import VaultRoot, build_dream_report, render_home_md, report_to_dict
    from code_atlas.settings import derive_project_name

    settings = _load_settings()
    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
        raise typer.Exit(code=1) from exc

    try:
        project_name = derive_project_name(settings.project_root)
        vault_roots = [VaultRoot(path=settings.project_root / settings.knowledge.vault_path, project_name=project_name)]
        vault_roots.extend(
            VaultRoot(path=Path(v.path).expanduser().resolve(), project_name=v.project_name)
            for v in settings.knowledge.extra_vaults
        )

        report = await build_dream_report(graph, vault_roots)

        home_path = settings.project_root / settings.knowledge.vault_path / "HOME.md"
        home_path.parent.mkdir(parents=True, exist_ok=True)
        home_path.write_text(render_home_md(report), encoding="utf-8")

        if _output.json:
            _json_output(report_to_dict(report))
        else:
            _print_dream_report(report)
            _echo(f"Wrote {home_path}")
    finally:
        await graph.close()


def _print_dream_report(report: Any) -> None:
    _echo(f"Inbox: {report.inbox_count} draft(s)")
    _echo(f"Orphan notes: {len(report.orphan_notes)}")
    _echo(f"Dangling links: {len(report.dangling_links)}")
    _echo(f"Duplicate ids: {len(report.duplicate_ids)}")
    _echo(f"Similar pairs: {len(report.similar_pairs)}")
    _echo(f"Promotion candidates: {len(report.promotion_candidates)}")
    for issue in report.memory_index_issues:
        _echo(f"MEMORY.md: {issue}")


# ---------------------------------------------------------------------------
# Health / Doctor async helpers
# ---------------------------------------------------------------------------


async def _run_health() -> None:
    from code_atlas.server.health import run_health_checks

    settings = _load_settings()
    report = await run_health_checks(settings, dotenv_path=_dotenv_path)
    _print_report(report, detailed=False)
    raise typer.Exit(code=0 if report.ok else 1)


async def _run_doctor() -> None:
    from code_atlas.server.health import run_health_checks

    settings = _load_settings()
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
    from code_atlas.backends import create_graph_client, graph_backend_label
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import AtlasSettings, derive_project_name
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    project_root, _auto_scope = _resolve_project_root(path, no_git_check=no_git_check)
    settings = AtlasSettings(project_root=project_root)
    init_telemetry(
        settings.observability,
        role="watch",
        project=derive_project_name(settings.project_root),
        root=str(settings.project_root),
        indexing=True,
    )
    if debounce is not None:
        settings.watcher.debounce_s = debounce
    if max_wait is not None:
        settings.watcher.max_wait_s = max_wait

    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to {}", graph_backend_label(graph, settings))
    await graph.ensure_schema()

    daemon = DaemonManager()
    started = await daemon.start(settings, graph)  # type: ignore[invalid-argument-type]
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
    """Start the EventBus, file watcher, and all tier consumers, run until interrupted."""
    from code_atlas.backends import create_graph_client, graph_backend_label
    from code_atlas.indexing.daemon import DaemonManager
    from code_atlas.settings import derive_project_name
    from code_atlas.telemetry import init_telemetry, shutdown_telemetry

    settings = _load_settings()
    if no_embed:
        settings.embeddings.enabled = False
    init_telemetry(
        settings.observability,
        role="daemon",
        project=derive_project_name(settings.project_root),
        root=str(settings.project_root),
        indexing=True,
    )

    graph = await create_graph_client(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach {} — {}", graph_backend_label(graph, settings), exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to {}", graph_backend_label(graph, settings))
    await graph.ensure_schema()

    daemon = DaemonManager()
    started = await daemon.start(settings, graph, include_watcher=True)  # type: ignore[invalid-argument-type]
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
