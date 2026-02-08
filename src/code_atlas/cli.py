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
    query: str = typer.Argument(..., help="Search query (natural language, keyword, or Cypher)."),
    type_: str = typer.Option("hybrid", "--type", "-t", help="Search type: hybrid, graph, semantic, keyword."),
    scope: str | None = typer.Option(None, help="Scope search to a sub-project path."),
    budget: int = typer.Option(8000, "--budget", "-b", help="Token budget for context assembly."),
) -> None:
    """Search the code graph."""
    logger.info("Searching for '{}' (type={})", query, type_)
    raise typer.Exit(code=0)


@app.command()
def status() -> None:
    """Show index status and health."""
    asyncio.run(_run_status())


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
            "Done — {} files scanned, {} entities indexed in {:.1f}s",
            result.files_scanned,
            result.entities_total,
            result.duration_s,
        )
    finally:
        await graph.close()
        await bus.close()


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
# Daemon subcommands
# ---------------------------------------------------------------------------


async def _run_daemon() -> None:
    """Start the EventBus and all tier consumers, run until interrupted."""
    from code_atlas.embeddings import EmbedClient
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
    consumers = [
        Tier1GraphConsumer(bus, graph, settings),
        Tier2ASTConsumer(bus, graph, settings),
        Tier3EmbedConsumer(bus, graph, embed),
    ]

    try:
        await asyncio.gather(*(c.run() for c in consumers))
    except asyncio.CancelledError:
        pass
    finally:
        for c in consumers:
            c.stop()
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
