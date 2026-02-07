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
    logger.info("Indexing {}", path)
    raise typer.Exit(code=0)


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
    logger.info("Status check")
    raise typer.Exit(code=0)


@app.command()
def mcp() -> None:
    """Start the MCP server."""
    logger.info("Starting MCP server")
    raise typer.Exit(code=0)


# ---------------------------------------------------------------------------
# Daemon subcommands
# ---------------------------------------------------------------------------


async def _run_daemon() -> None:
    """Start the EventBus and all tier consumers, run until interrupted."""
    from code_atlas.events import EventBus
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

    consumers = [
        Tier1GraphConsumer(bus),
        Tier2ASTConsumer(bus),
        Tier3EmbedConsumer(bus),
    ]

    try:
        await asyncio.gather(*(c.run() for c in consumers))
    except asyncio.CancelledError:
        pass
    finally:
        for c in consumers:
            c.stop()
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
