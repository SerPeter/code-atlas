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

from code_atlas.embeddings import EmbedCache, EmbedClient
from code_atlas.events import EventBus, FileChanged, Topic
from code_atlas.parser import get_language_for_file
from code_atlas.pipeline import Tier1GraphConsumer, Tier2ASTConsumer, Tier3EmbedConsumer

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
    "vendor/",
    "build/",
    "dist/",
    "target/",
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
# File scope filter (cached, reusable)
# ---------------------------------------------------------------------------


class FileScope:
    """Cached, reusable file scope filter.

    Compiles ignore patterns once on construction and supports nested
    ``.gitignore`` files discovered during :meth:`scan`.  The
    :meth:`is_included` method can be called independently (e.g. by a
    file watcher) without re-reading ignore files.

    Exclusion order:
      1. Default excludes (``.git/``, ``__pycache__/``, etc.)
      2. Root ``.gitignore`` patterns
      3. Root ``.atlasignore`` patterns
      4. ``settings.scope.exclude_patterns``
      5. Nested ``.gitignore`` files (discovered during :meth:`scan`)
      6. Include-path prefix filter (if set)
    """

    def __init__(
        self,
        project_root: str | Path,
        settings: AtlasSettings,
        scope_paths: list[str] | None = None,
    ) -> None:
        self._root = Path(project_root).resolve()
        self._global_spec = self._build_global_spec(settings)
        self._include_prefixes = self._build_include_prefixes(scope_paths, settings)
        # Nested gitignore specs populated during scan()
        self._nested_specs: dict[str, pathspec.PathSpec] = {}

    # -- public API ----------------------------------------------------------

    def scan(self) -> list[str]:
        """Walk the project tree and return sorted relative POSIX paths.

        Files are filtered through the global ignore spec, nested
        ``.gitignore`` files, include-path prefixes, and language support.
        """
        result: list[str] = []

        for dirpath, dirnames, filenames in os.walk(self._root):
            rel_dir = Path(dirpath).relative_to(self._root).as_posix()
            if rel_dir == ".":
                rel_dir = ""

            # Discover nested .gitignore (non-root directories only)
            if rel_dir:
                nested_gi = Path(dirpath) / ".gitignore"
                if nested_gi.is_file():
                    patterns = _read_ignore_file(nested_gi)
                    if patterns:
                        self._nested_specs[rel_dir] = pathspec.PathSpec.from_lines("gitignore", patterns)
                        logger.debug("Loaded {} patterns from {}", len(patterns), nested_gi)

            # Prune excluded and symlinked directories (modify dirnames in-place)
            dirnames[:] = [
                d
                for d in dirnames
                if not self._is_dir_excluded(f"{rel_dir}/{d}" if rel_dir else d) and not Path(dirpath, d).is_symlink()
            ]

            for fname in filenames:
                # Skip broken symlinks
                fpath = Path(dirpath, fname)
                if fpath.is_symlink() and not fpath.exists():
                    logger.debug("Skipping broken symlink: {}", f"{rel_dir}/{fname}" if rel_dir else fname)
                    continue
                rel_path = f"{rel_dir}/{fname}" if rel_dir else fname
                if not self.is_included(rel_path):
                    continue
                # Language support check (not in is_included — watcher may skip this)
                if get_language_for_file(rel_path) is None:
                    continue
                result.append(rel_path)

        result.sort()
        return result

    def is_included(self, rel_path: str) -> bool:
        """Check whether *rel_path* passes all scope filters.

        Does **not** check language support — callers handle that separately.
        """
        # 1. Global exclude
        if self._global_spec.match_file(rel_path):
            logger.trace("EXCLUDE {}: matched global pattern", rel_path)
            return False

        # 2. Nested gitignore exclude
        parts = rel_path.split("/")
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            spec = self._nested_specs.get(ancestor)
            if spec is not None:
                # Match relative to the ancestor directory
                sub_path = "/".join(parts[depth:])
                if spec.match_file(sub_path):
                    logger.trace("EXCLUDE {}: matched nested .gitignore in {}/", rel_path, ancestor)
                    return False

        # 3. Include-path prefix filter
        if self._include_prefixes and not _matches_any_prefix(rel_path, self._include_prefixes):
            logger.trace("EXCLUDE {}: not under any include path", rel_path)
            return False

        logger.trace("INCLUDE {}", rel_path)
        return True

    # -- private helpers -----------------------------------------------------

    def _build_global_spec(self, settings: AtlasSettings) -> pathspec.PathSpec:
        """Compile the global ignore spec from defaults, root ignore files, and settings."""
        patterns: list[str] = list(_DEFAULT_EXCLUDES)

        gitignore = self._root / ".gitignore"
        if gitignore.is_file():
            gi_patterns = _read_ignore_file(gitignore)
            patterns.extend(gi_patterns)
            logger.debug("Loaded {} patterns from {}", len(gi_patterns), gitignore)

        atlasignore = self._root / ".atlasignore"
        if atlasignore.is_file():
            ai_patterns = _read_ignore_file(atlasignore)
            patterns.extend(ai_patterns)
            logger.debug("Loaded {} patterns from {}", len(ai_patterns), atlasignore)

        patterns.extend(settings.scope.exclude_patterns)

        return pathspec.PathSpec.from_lines("gitignore", patterns)

    def _build_include_prefixes(self, scope_paths: list[str] | None, settings: AtlasSettings) -> list[str]:
        """Normalise include paths to POSIX form."""
        include_paths = scope_paths or settings.scope.include_paths or []
        return [p.replace("\\", "/").rstrip("/") for p in include_paths]

    def _is_dir_excluded(self, rel_dir: str) -> bool:
        """Check whether a directory should be pruned from the walk."""
        dir_pattern = f"{rel_dir}/"
        if self._global_spec.match_file(dir_pattern):
            return True

        # Check nested gitignore specs
        parts = rel_dir.split("/")
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            spec = self._nested_specs.get(ancestor)
            if spec is not None:
                sub_path = "/".join(parts[depth:]) + "/"
                if spec.match_file(sub_path):
                    return True

        return False


# ---------------------------------------------------------------------------
# File scanner (thin wrapper for backward compatibility)
# ---------------------------------------------------------------------------


def scan_files(
    project_root: str | Path,
    settings: AtlasSettings,
    scope_paths: list[str] | None = None,
) -> list[str]:
    """Discover indexable files under *project_root*.

    Returns a sorted list of **relative POSIX paths** (forward slashes,
    relative to *project_root*).  Delegates to :class:`FileScope`.
    """
    return FileScope(project_root, settings, scope_paths).scan()


def _read_ignore_file(path: Path) -> list[str]:
    """Read a .gitignore-style file, stripping comments and blank lines."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
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
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune symlinked directories
        dirnames[:] = [d for d in dirnames if not Path(dirpath, d).is_symlink()]
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
# Embedding model lock
# ---------------------------------------------------------------------------


async def _check_model_lock(graph: GraphClient, model: str, dimension: int, *, reindex: bool) -> None:
    """Enforce embedding model lock on the SchemaVersion node.

    - First run (no stored model): write current config.
    - Reindex: clear embeddings, write new config.
    - Mismatch: raise RuntimeError with clear guidance.
    """
    if reindex:
        await graph.clear_all_embeddings()
        await graph.set_embedding_config(model, dimension)
        return

    stored = await graph.get_embedding_config()
    if stored is not None:
        stored_model, _stored_dim = stored
        if stored_model != model:
            msg = (
                f"Embedding model changed from '{stored_model}' to '{model}'. "
                f"Run `atlas index --reindex` to rebuild all embeddings."
            )
            raise RuntimeError(msg)
    else:
        await graph.set_embedding_config(model, dimension)


# ---------------------------------------------------------------------------
# Main indexing orchestration
# ---------------------------------------------------------------------------


async def _create_package_hierarchy(graph: GraphClient, project_name: str, project_root: Path) -> int:
    """Create Project + Package nodes and CONTAINS edges. Returns package count."""
    await graph.merge_project_node(project_name)
    packages = _detect_packages(project_root)
    for qn, rel_path in packages:
        pkg_name = qn.rsplit(".", 1)[-1]
        await graph.merge_package_node(project_name, qn, pkg_name, f"{rel_path}/__init__.py")
        parent_qn = qn.rsplit(".", 1)[0] if "." in qn else None
        parent_uid = f"{project_name}:{parent_qn}" if parent_qn else project_name
        await graph.create_contains_edge(parent_uid, f"{project_name}:{qn}")
    return len(packages)


async def _run_pipeline(
    bus: EventBus,
    graph: GraphClient,
    settings: AtlasSettings,
    embed: EmbedClient,
    cache: EmbedCache | None,
    drain_timeout_s: float,
) -> None:
    """Start inline tier consumers and wait for the pipeline to drain."""
    await bus.ensure_group(Topic.FILE_CHANGED, "tier1-graph")
    await bus.ensure_group(Topic.AST_DIRTY, "tier2-ast")
    await bus.ensure_group(Topic.EMBED_DIRTY, "tier3-embed")

    tier1 = Tier1GraphConsumer(bus, graph, settings)
    tier2 = Tier2ASTConsumer(bus, graph, settings)
    tier3 = Tier3EmbedConsumer(bus, graph, embed, cache=cache)

    task1 = asyncio.create_task(tier1.run())
    task2 = asyncio.create_task(tier2.run())
    task3 = asyncio.create_task(tier3.run())

    try:
        await _wait_for_drain(bus, drain_timeout_s)
    finally:
        tier1.stop()
        tier2.stop()
        tier3.stop()
        await asyncio.sleep(0.5)
        task1.cancel()
        task2.cancel()
        task3.cancel()
        for t in (task1, task2, task3):
            with contextlib.suppress(asyncio.CancelledError):
                await t
        if cache is not None:
            await cache.close()


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

    # 2. Model lock check + full reindex
    embed = EmbedClient(settings.embeddings)
    cache: EmbedCache | None = None
    if settings.embeddings.cache_ttl_days > 0:
        cache = EmbedCache(settings.redis, settings.embeddings)

    if full_reindex:
        logger.info("Full reindex: deleting existing data for '{}'", project_name)
        await graph.delete_project_data(project_name)
        if cache is not None:
            await cache.clear()

    await _check_model_lock(graph, settings.embeddings.model, settings.embeddings.dimension, reindex=full_reindex)

    # 3. Create Project + Package hierarchy
    pkg_count = await _create_package_hierarchy(graph, project_name, project_root)
    logger.info("Created {} package node(s)", pkg_count)

    # 4. Publish FileChanged events
    for file_path in files:
        await bus.publish(Topic.FILE_CHANGED, FileChanged(path=file_path, change_type="created"))
    logger.info("Published {} FileChanged events", len(files))

    # 5. Start inline consumers and wait for drain
    await _run_pipeline(bus, graph, settings, embed, cache, drain_timeout_s)

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
    """Poll stream groups until Tier 1, Tier 2, and Tier 3 are drained."""
    deadline = time.monotonic() + timeout_s
    settled_since: float | None = None

    while time.monotonic() < deadline:
        t1_info = await bus.stream_group_info(Topic.FILE_CHANGED, "tier1-graph")
        t2_info = await bus.stream_group_info(Topic.AST_DIRTY, "tier2-ast")
        t3_info = await bus.stream_group_info(Topic.EMBED_DIRTY, "tier3-embed")

        t1_remaining = t1_info["pending"] + t1_info["lag"]
        t2_remaining = t2_info["pending"] + t2_info["lag"]
        t3_remaining = t3_info["pending"] + t3_info["lag"]

        if t1_remaining == 0 and t2_remaining == 0 and t3_remaining == 0:
            if settled_since is None:
                settled_since = time.monotonic()
            elif time.monotonic() - settled_since >= 1.0:
                logger.debug("Pipeline drained after {:.1f}s settling", time.monotonic() - settled_since)
                return
        else:
            settled_since = None

        await asyncio.sleep(0.5)

    logger.warning("Pipeline drain timed out after {:.0f}s", timeout_s)
