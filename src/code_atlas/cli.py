"""CLI entrypoint for Code Atlas."""

from __future__ import annotations

import typer
from loguru import logger

app = typer.Typer(
    name="atlas",
    help="Code Atlas — map your codebase, search it three ways, feed it to agents.",
    no_args_is_help=True,
)


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


if __name__ == "__main__":
    app()
