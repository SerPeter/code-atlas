"""Server package — MCP server and health checks."""

from __future__ import annotations

from code_atlas.server.health import CheckResult, CheckStatus, HealthReport, run_health_checks
from code_atlas.server.mcp import AppContext, create_mcp_server

__all__ = [
    "AppContext",
    "CheckResult",
    "CheckStatus",
    "HealthReport",
    "create_mcp_server",
    "run_health_checks",
]
