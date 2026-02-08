"""Python indexer — scans files and drives the event pipeline for atlas index."""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pathspec
from loguru import logger

from code_atlas.events import EventBus, FileChanged, Topic
from code_atlas.parser import get_language_for_file
from code_atlas.pipeline import Tier1GraphConsumer, Tier2ASTConsumer

if TYPE_CHECKING:
    from code_atlas.graph import GraphClient
    from code_atlas.settings import AtlasSettings

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_DEFAULT_EXCLUDES: list[str] = [
    ".git/",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    "build/",
    "dist/",
    ".tox/",
    ".mypy_cache/",
    ".ruff_cache/",
    "site-packages/",
    ".eggs/",
    "*.pyc",
    ".env/",
]


@dataclass(frozen=True)
class IndexResult:
    """Summary of an indexing run."""

    files_scanned: int
    files_published: int
    entities_total: int
    duration_s: float


# ---------------------------------------------------------------------------
# File scanner (pure function, no I/O beyond filesystem)
# ---------------------------------------------------------------------------


def scan_files(
    project_root: str | Path,
    settings: AtlasSettings,
    scope_paths: list[str] | None = None,
) -> list[str]:
    """Discover indexable files under *project_root*.

    Returns a sorted list of **relative POSIX paths** (forward slashes,
    relative to *project_root*).

    Exclusion order (P0):
      1. Default excludes (`.git/`, `__pycache__/`, etc.)
      2. Root `.gitignore` patterns
      3. Root `.atlasignore` patterns
      4. ``settings.scope.exclude_patterns``
      5. Filter to ``scope_paths`` / ``settings.scope.include_paths`` (if set)
      6. Filter to files with registered language support
    """
    root = Path(project_root).resolve()

    # -- build combined ignore spec ------------------------------------------
    patterns: list[str] = list(_DEFAULT_EXCLUDES)

    gitignore = root / ".gitignore"
    if gitignore.is_file():
        patterns.extend(_read_ignore_file(gitignore))

    atlasignore = root / ".atlasignore"
    if atlasignore.is_file():
        patterns.extend(_read_ignore_file(atlasignore))

    patterns.extend(settings.scope.exclude_patterns)

    spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    # -- walk and filter -----------------------------------------------------
    include_paths = scope_paths or settings.scope.include_paths or []
    # Normalise include paths to POSIX (forward-slash, no trailing slash)
    include_prefixes = [p.replace("\\", "/").rstrip("/") for p in include_paths]

    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        # Prune excluded directories (modify dirnames in-place)
        dirnames[:] = [d for d in dirnames if not spec.match_file(f"{rel_dir}/{d}/" if rel_dir else f"{d}/")]

        for fname in filenames:
            rel_path = f"{rel_dir}/{fname}" if rel_dir else fname
            # 1. Exclude check
            if spec.match_file(rel_path):
                continue
            # 2. Include check (if include_paths set)
            if include_prefixes and not _matches_any_prefix(rel_path, include_prefixes):
                continue
            # 3. Language support check
            if get_language_for_file(rel_path) is None:
                continue
            result.append(rel_path)

    result.sort()
    return result


def _read_ignore_file(path: Path) -> list[str]:
    """Read a .gitignore-style file, stripping comments and blank lines."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def _matches_any_prefix(rel_path: str, prefixes: list[str]) -> bool:
    """Check if a relative path starts with any of the given prefixes."""
    return any(rel_path == prefix or rel_path.startswith(prefix + "/") for prefix in prefixes)


# ---------------------------------------------------------------------------
# Package detection
# ---------------------------------------------------------------------------


def _detect_packages(project_root: Path) -> list[tuple[str, str]]:
    """Find Python packages (dirs with __init__.py).

    Returns list of ``(qualified_name, relative_posix_path)`` sorted by depth.
    """
    root = project_root.resolve()
    packages: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "__init__.py" in filenames:
            rel = Path(dirpath).relative_to(root).as_posix()
            if rel == ".":
                continue
            # qualified name: replace / with .
            qn = rel.replace("/", ".")
            packages.append((qn, rel))
    packages.sort(key=lambda t: t[0].count("."))
    return packages


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _get_git_hash(project_root: Path) -> str | None:
    """Get the current git HEAD short hash, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError, subprocess.TimeoutExpired:
        pass
    return None


# ---------------------------------------------------------------------------
# Main indexing orchestration
# ---------------------------------------------------------------------------


async def index_project(
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    *,
    scope_paths: list[str] | None = None,
    full_reindex: bool = False,
    drain_timeout_s: float = 120.0,
) -> IndexResult:
    """Run a full index of the project through the event pipeline.

    1. Scan files
    2. Optionally wipe old data (full reindex)
    3. Create Project + Package hierarchy in the graph
    4. Publish FileChanged events to Valkey
    5. Run inline Tier 1 + Tier 2 consumers until the pipeline drains
    6. Update Project metadata (counts, git hash)
    """
    start = time.monotonic()
    project_name = settings.project_root.name
    project_root = Path(settings.project_root).resolve()

    # 1. Scan files
    files = scan_files(project_root, settings, scope_paths=scope_paths)
    logger.info("Scanned {} indexable files", len(files))

    if not files:
        return IndexResult(files_scanned=0, files_published=0, entities_total=0, duration_s=time.monotonic() - start)

    # 2. Full reindex — wipe project data
    if full_reindex:
        logger.info("Full reindex: deleting existing data for '{}'", project_name)
        await graph.delete_project_data(project_name)

    # 3. Create Project + Package hierarchy
    await graph.merge_project_node(project_name)
    packages = _detect_packages(project_root)
    for qn, rel_path in packages:
        pkg_name = qn.rsplit(".", 1)[-1]
        await graph.merge_package_node(project_name, qn, pkg_name, f"{rel_path}/__init__.py")
        # CONTAINS edge from parent
        parent_qn = qn.rsplit(".", 1)[0] if "." in qn else None
        parent_uid = f"{project_name}:{parent_qn}" if parent_qn else project_name
        await graph.create_contains_edge(parent_uid, f"{project_name}:{qn}")

    logger.info("Created {} package node(s)", len(packages))

    # 4. Publish FileChanged events
    for file_path in files:
        await bus.publish(Topic.FILE_CHANGED, FileChanged(path=file_path, change_type="created"))

    logger.info("Published {} FileChanged events", len(files))

    # 5. Start inline consumers and wait for drain
    await bus.ensure_group(Topic.FILE_CHANGED, "tier1-graph")
    await bus.ensure_group(Topic.AST_DIRTY, "tier2-ast")

    tier1 = Tier1GraphConsumer(bus, graph, settings)
    tier2 = Tier2ASTConsumer(bus, graph, settings)

    task1 = asyncio.create_task(tier1.run())
    task2 = asyncio.create_task(tier2.run())

    try:
        await _wait_for_drain(bus, drain_timeout_s)
    finally:
        tier1.stop()
        tier2.stop()
        # Give consumers time to finish their current iteration
        await asyncio.sleep(0.5)
        task1.cancel()
        task2.cancel()
        for t in (task1, task2):
            with contextlib.suppress(asyncio.CancelledError):
                await t

    # 6. Update Project metadata
    entity_count = await graph.count_entities(project_name)
    git_hash = _get_git_hash(project_root)
    await graph.update_project_metadata(
        project_name,
        last_indexed_at=time.time(),
        file_count=len(files),
        entity_count=entity_count,
        **({"git_hash": git_hash} if git_hash else {}),
    )

    duration = time.monotonic() - start
    logger.info("Indexing complete: {} files, {} entities, {:.1f}s", len(files), entity_count, duration)

    return IndexResult(
        files_scanned=len(files),
        files_published=len(files),
        entities_total=entity_count,
        duration_s=duration,
    )


async def _wait_for_drain(bus: EventBus, timeout_s: float) -> None:
    """Poll stream groups until both Tier 1 and Tier 2 are drained."""
    deadline = time.monotonic() + timeout_s
    settled_since: float | None = None

    while time.monotonic() < deadline:
        t1_info = await bus.stream_group_info(Topic.FILE_CHANGED, "tier1-graph")
        t2_info = await bus.stream_group_info(Topic.AST_DIRTY, "tier2-ast")

        t1_remaining = t1_info["pending"] + t1_info["lag"]
        t2_remaining = t2_info["pending"] + t2_info["lag"]

        if t1_remaining == 0 and t2_remaining == 0:
            if settled_since is None:
                settled_since = time.monotonic()
            elif time.monotonic() - settled_since >= 1.0:
                logger.debug("Pipeline drained after {:.1f}s settling", time.monotonic() - settled_since)
                return
        else:
            settled_since = None

        await asyncio.sleep(0.5)

    logger.warning("Pipeline drain timed out after {:.0f}s", timeout_s)
