"""CLI entrypoint for Code Atlas."""

from __future__ import annotations

import asyncio

import typer
from loguru import logger

app = typer.Typer(
    name="atlas",
    help="Code Atlas — map your codebase, search it three ways, feed it to agents.",
    no_args_is_help=True,
)

daemon_app = typer.Typer(name="daemon", help="Manage the Code Atlas indexing daemon.")
app.add_typer(daemon_app)


@app.command()
def index(
    path: str = typer.Argument(".", help="Path to the project root to index."),
    scope: list[str] | None = typer.Option(None, help="Scope indexing to specific paths (repeatable)."),
    full_reindex: bool = typer.Option(False, "--full", help="Force full re-index, ignoring delta cache."),
) -> None:
    """Index a codebase into the graph."""
    asyncio.run(_run_index(path, scope, full_reindex))


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
) -> None:
    """Start the MCP server for AI agent connections."""
    from code_atlas.mcp_server import create_mcp_server
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings()
    server = create_mcp_server(settings)
    logger.info("Starting MCP server (transport={})", transport)
    server.run(transport=transport)  # type: ignore[arg-type]  # typer gives str, FastMCP expects Literal


# ---------------------------------------------------------------------------
# Index / Status async helpers
# ---------------------------------------------------------------------------


async def _run_index(path: str, scope: list[str] | None, full_reindex: bool) -> None:
    """Async implementation of the ``atlas index`` command."""
    from pathlib import Path

    from code_atlas.events import EventBus
    from code_atlas.graph import GraphClient
    from code_atlas.indexer import index_project
    from code_atlas.settings import AtlasSettings

    project_root = Path(path).resolve()
    settings = AtlasSettings(project_root=project_root)

    # Connect to Valkey
    bus = EventBus(settings.redis)
    try:
        await bus.ping()
    except Exception as exc:
        logger.error("Cannot reach Valkey at {}:{} — {}", settings.redis.host, settings.redis.port, exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to Valkey at {}:{}", settings.redis.host, settings.redis.port)

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
        result = await index_project(
            settings,
            graph,
            bus,
            scope_paths=scope or None,
            full_reindex=full_reindex,
        )
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


async def _run_search(
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
    from code_atlas.embeddings import EmbedClient
    from code_atlas.graph import GraphClient
    from code_atlas.indexer import StalenessChecker
    from code_atlas.search import SearchType, hybrid_search
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings()
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

    embed: EmbedClient | None = None
    if search_types is None or SearchType.VECTOR in search_types:
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

        # Staleness check
        checker = StalenessChecker(settings.project_root)
        info = await checker.check(graph, include_changed=True)
        if info.stale:
            commit_str = info.last_indexed_commit[:8] if info.last_indexed_commit else "never"
            logger.warning("Index is stale (last indexed: {})", commit_str)
            if info.changed_files:
                logger.warning("  {} file(s) changed since last index", len(info.changed_files))
    finally:
        await graph.close()


async def _run_status() -> None:
    """Async implementation of the ``atlas status`` command."""
    from code_atlas.graph import GraphClient
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
            import datetime

            ts = datetime.datetime.fromtimestamp(last, tz=datetime.UTC).isoformat() if last else "never"
            logger.info(
                "Project: {} | indexed: {} | files: {} | entities: {} | git: {}",
                name,
                ts,
                files,
                entities,
                git_hash,
            )
    finally:
        await graph.close()


# ---------------------------------------------------------------------------
# Shared infrastructure connection
# ---------------------------------------------------------------------------


async def _connect_bus_and_graph(settings):
    """Connect to Valkey and Memgraph, returning ``(bus, graph)``.

    Exits with code 1 if either service is unreachable.
    Accepts an :class:`~code_atlas.settings.AtlasSettings` instance.
    """
    from code_atlas.events import EventBus
    from code_atlas.graph import GraphClient

    bus = EventBus(settings.redis)
    try:
        await bus.ping()
    except Exception as exc:
        logger.error("Cannot reach Valkey at {}:{} — {}", settings.redis.host, settings.redis.port, exc)
        raise typer.Exit(code=1) from exc
    logger.info("Connected to Valkey at {}:{}", settings.redis.host, settings.redis.port)

    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
        await bus.close()
        raise typer.Exit(code=1) from exc
    logger.info("Connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)
    await graph.ensure_schema()

    return bus, graph


# ---------------------------------------------------------------------------
# Watch async helper
# ---------------------------------------------------------------------------


async def _run_watch(path: str, *, debounce: float | None, max_wait: float | None) -> None:
    """Async implementation of the ``atlas watch`` command."""
    from pathlib import Path

    from code_atlas.embeddings import EmbedCache, EmbedClient
    from code_atlas.indexer import FileScope
    from code_atlas.pipeline import Tier1GraphConsumer, Tier2ASTConsumer, Tier3EmbedConsumer
    from code_atlas.settings import AtlasSettings
    from code_atlas.watcher import FileWatcher

    project_root = Path(path).resolve()
    settings = AtlasSettings(project_root=project_root)
    if debounce is not None:
        settings.watcher.debounce_s = debounce
    if max_wait is not None:
        settings.watcher.max_wait_s = max_wait

    bus, graph = await _connect_bus_and_graph(settings)

    embed = EmbedClient(settings.embeddings)
    cache: EmbedCache | None = None
    if settings.embeddings.cache_ttl_days > 0:
        cache = EmbedCache(settings.redis, settings.embeddings)

    scope = FileScope(project_root, settings)
    watcher = FileWatcher(project_root, bus, scope, settings.watcher)
    consumers = [
        Tier1GraphConsumer(bus, graph, settings),
        Tier2ASTConsumer(bus, graph, settings),
        Tier3EmbedConsumer(bus, graph, embed, cache=cache),
    ]

    try:
        await asyncio.gather(watcher.run(), *(c.run() for c in consumers))
    except asyncio.CancelledError:
        pass
    finally:
        watcher.stop()
        for c in consumers:
            c.stop()
        if cache is not None:
            await cache.close()
        await graph.close()
        await bus.close()
        logger.info("Watch stopped")


# ---------------------------------------------------------------------------
# Daemon subcommands
# ---------------------------------------------------------------------------


async def _run_daemon() -> None:
    """Start the EventBus and all tier consumers, run until interrupted."""
    from code_atlas.embeddings import EmbedCache, EmbedClient
    from code_atlas.events import EventBus
    from code_atlas.graph import GraphClient
    from code_atlas.pipeline import Tier1GraphConsumer, Tier2ASTConsumer, Tier3EmbedConsumer
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings()
    bus = EventBus(settings.redis)

    # Verify Redis is reachable
    try:
        await bus.ping()
    except Exception as exc:
        logger.error("Cannot reach Redis/Valkey at {}:{} — {}", settings.redis.host, settings.redis.port, exc)
        raise typer.Exit(code=1) from exc

    logger.info("Connected to Redis/Valkey at {}:{}", settings.redis.host, settings.redis.port)

    # Verify Memgraph is reachable and apply schema
    graph = GraphClient(settings)
    try:
        await graph.ping()
    except Exception as exc:
        logger.error("Cannot reach Memgraph at {}:{} — {}", settings.memgraph.host, settings.memgraph.port, exc)
        await bus.close()
        raise typer.Exit(code=1) from exc

    logger.info("Connected to Memgraph at {}:{}", settings.memgraph.host, settings.memgraph.port)
    await graph.ensure_schema()

    embed = EmbedClient(settings.embeddings)
    cache: EmbedCache | None = None
    if settings.embeddings.cache_ttl_days > 0:
        cache = EmbedCache(settings.redis, settings.embeddings)

    consumers = [
        Tier1GraphConsumer(bus, graph, settings),
        Tier2ASTConsumer(bus, graph, settings),
        Tier3EmbedConsumer(bus, graph, embed, cache=cache),
    ]

    try:
        await asyncio.gather(*(c.run() for c in consumers))
    except asyncio.CancelledError:
        pass
    finally:
        for c in consumers:
            c.stop()
        if cache is not None:
            await cache.close()
        await graph.close()
        await bus.close()
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
