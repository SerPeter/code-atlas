"""Code Atlas — code intelligence graph for AI coding agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("code-atlas-mcp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
