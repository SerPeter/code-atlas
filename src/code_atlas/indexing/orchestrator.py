"""Python indexer — scans files and drives the event pipeline for atlas index."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import os
import re
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pathspec
from loguru import logger

from code_atlas.events import Event, EventBus, FileChanged, Topic
from code_atlas.indexing.consumers import Tier1GraphConsumer, Tier2ASTConsumer, Tier3EmbedConsumer
from code_atlas.parsing.ast import get_language_for_file
from code_atlas.search.embeddings import EmbedCache, EmbedClient
from code_atlas.settings import derive_project_name, resolve_git_dir
from code_atlas.telemetry import get_metrics, get_tracer

if TYPE_CHECKING:
    from collections.abc import Callable

    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings, MonorepoSettings

_tracer = get_tracer(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectedProject:
    """A detected sub-project within a monorepo."""

    name: str  # project_name for the graph
    path: str  # relative POSIX path from monorepo root
    root: Path  # absolute path
    marker: str  # which marker file (or "explicit")


_DEFAULT_EXCLUDE: list[str] = [
    # Version control
    ".git/",
    ".hg/",
    ".svn/",
    ".bzr/",
    ".fossil/",
    # IDE / editor
    ".idea/",
    ".vscode/",
    ".vs/",
    ".eclipse/",
    # Python
    "__pycache__/",
    ".venv/",
    "venv/",
    ".eggs/",
    "*.pyc",
    "*.pyo",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    ".pytype/",
    ".tox/",
    ".nox/",
    "site-packages/",
    "__pypackages__/",
    ".pdm-build/",
    # JavaScript / TypeScript
    "node_modules/",
    "bower_components/",
    ".next/",
    ".nuxt/",
    ".svelte-kit/",
    # Rust
    "target/",
    # Java / JVM
    ".gradle/",
    # General build / dist
    "build/",
    "dist/",
    "out/",
    "vendor/",
    # Environment / config
    ".env/",
    ".direnv/",
    # Caches
    ".cache/",
    ".parcel-cache/",
    ".turbo/",
    # AI agents
    ".claude/",
    ".cursor/",
    ".copilot/",
    # Code Atlas
    ".atlas/",
]

_DEFAULT_INCLUDE: list[str] = [
    # Python
    "*.py",
    "*.pyi",
    # TypeScript / JavaScript
    "*.ts",
    "*.tsx",
    "*.js",
    "*.jsx",
    "*.mjs",
    "*.cjs",
    # Go
    "*.go",
    # Rust
    "*.rs",
    # C / C++
    "*.c",
    "*.h",
    "*.cpp",
    "*.cc",
    "*.cxx",
    "*.hpp",
    "*.hxx",
    "*.hh",
    # Java
    "*.java",
    # C#
    "*.cs",
    # Ruby
    "*.rb",
    "*.rake",
    "*.gemspec",
    # PHP
    "*.php",
    # Markdown (documentation indexing)
    "*.md",
]

# Directories to always skip during sub-project detection walk
_DETECT_PRUNE_DIRS = frozenset(d.rstrip("/") for d in _DEFAULT_EXCLUDE if d.endswith("/"))


# ---------------------------------------------------------------------------
# Monorepo sub-project detection
# ---------------------------------------------------------------------------


def _resolve_project_name(
    proj: DetectedProject,
    path: str,
    name_counts: dict[str, list[str]],
) -> DetectedProject:
    """Resolve a DetectedProject's name, handling collisions."""
    if proj.name:
        return proj
    base = path.rsplit("/", 1)[-1]
    resolved = path.replace("/", "-") if len(name_counts.get(base, [])) > 1 else base
    return DetectedProject(name=resolved, path=proj.path, root=proj.root, marker=proj.marker)


def detect_sub_projects(
    project_root: Path,
    monorepo_settings: MonorepoSettings,
) -> list[DetectedProject]:
    """Detect sub-projects within a monorepo root.

    1. Start with explicit ``monorepo_settings.projects`` entries.
    2. If ``auto_detect`` is True, walk the tree looking for marker files.
    3. Skip the root directory itself (root = the monorepo, not a sub-project).
    4. Prune default-excluded directories during the walk.
    5. Explicit entries override auto-detected at the same path.
    6. Naming: basename of path. On collision, use ``path.replace("/", "-")``.
    7. Sort by path depth (shallow first).
    """
    root = project_root.resolve()

    # 1. Explicit projects
    explicit_by_path: dict[str, DetectedProject] = {}
    for entry in monorepo_settings.projects:
        raw_path = entry.get("path", "").replace("\\", "/").strip("/")
        if not raw_path:
            continue
        name = entry.get("name", "") or raw_path.replace("/", "-")
        explicit_by_path[raw_path] = DetectedProject(
            name=name,
            path=raw_path,
            root=root / raw_path.replace("/", os.sep),
            marker="explicit",
        )

    # 2. Auto-detect
    auto_by_path: dict[str, DetectedProject] = {}
    if monorepo_settings.auto_detect:
        markers = set(monorepo_settings.markers)
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).relative_to(root).as_posix()
            if rel_dir == ".":
                # Skip root — prune excluded dirs
                dirnames[:] = [d for d in dirnames if d not in _DETECT_PRUNE_DIRS]
                continue

            # Prune excluded dirs
            dirnames[:] = [d for d in dirnames if d not in _DETECT_PRUNE_DIRS]

            # Check for marker files
            matched_markers = markers & set(filenames)
            if matched_markers:
                marker = sorted(matched_markers)[0]  # deterministic
                if rel_dir not in explicit_by_path:
                    auto_by_path[rel_dir] = DetectedProject(
                        name="",  # placeholder — resolved below
                        path=rel_dir,
                        root=root / rel_dir.replace("/", os.sep),
                        marker=marker,
                    )

    # 3. Merge: explicit overrides auto-detected at same path
    all_paths: dict[str, DetectedProject] = {**auto_by_path, **explicit_by_path}

    # 4. Resolve names (basename, with collision fallback to full-path-dashed)
    name_counts: dict[str, list[str]] = {}
    for path, proj in all_paths.items():
        base = proj.name or path.rsplit("/", 1)[-1]
        name_counts.setdefault(base, []).append(path)

    result: list[DetectedProject] = []
    for path, proj in all_paths.items():
        result.append(_resolve_project_name(proj, path, name_counts))

    # 5. Sort by path depth (shallow first), then alphabetically
    result.sort(key=lambda dp: (dp.path.count("/"), dp.path))
    return result


def classify_file_project(rel_path: str, sub_projects: list[DetectedProject]) -> str:
    """Return the project_name for the most specific (longest prefix) matching sub-project.

    Returns empty string if the file doesn't belong to any sub-project.
    """
    best_name = ""
    best_len = -1
    for proj in sub_projects:
        prefix = proj.path
        if (rel_path == prefix or rel_path.startswith(prefix + "/")) and len(prefix) > best_len:
            best_len = len(prefix)
            best_name = proj.name
    return best_name


@dataclass(frozen=True)
class DeltaStats:
    """File- and entity-level delta statistics for an indexing run."""

    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    entities_added: int = 0
    entities_modified: int = 0
    entities_deleted: int = 0
    entities_unchanged: int = 0


@dataclass(frozen=True)
class IndexResult:
    """Summary of an indexing run."""

    files_scanned: int
    files_published: int
    entities_total: int
    duration_s: float
    mode: str = "full"  # "full" | "delta"
    delta_stats: DeltaStats | None = None


# ---------------------------------------------------------------------------
# File scope filter (cached, reusable)
# ---------------------------------------------------------------------------


class FileScope:
    """Cached, reusable file scope filter.

    Compiles ignore patterns once on construction and supports nested
    ``.gitignore`` files discovered during :meth:`scan`.  The
    :meth:`is_included` method can be called independently (e.g. by a
    file watcher) without re-reading ignore files.

    Uses ruff-style include/exclude semantics:
      - ``exclude`` replaces ``_DEFAULT_EXCLUDE``; ``extend_exclude`` appends.
      - ``include`` replaces ``_DEFAULT_INCLUDE``; ``extend_include`` appends.
      - ``.gitignore`` and ``.atlasignore`` always apply regardless of ``exclude``.
    """

    def __init__(
        self,
        project_root: str | Path,
        settings: AtlasSettings,
        scope_paths: list[str] | None = None,
    ) -> None:
        self._root = Path(project_root).resolve()
        self._exclude_spec = self._build_exclude_spec(settings)
        self._include_spec = self._build_include_spec(settings)
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
        # 1. Exclude spec (defaults or custom + .gitignore + .atlasignore)
        if self._exclude_spec.match_file(rel_path):
            logger.trace("EXCLUDE {}: matched exclude pattern", rel_path)
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

        # 3. Include spec (file-extension filter)
        if self._include_spec is not None and not self._include_spec.match_file(rel_path):
            logger.trace("EXCLUDE {}: not matched by include patterns", rel_path)
            return False

        # 4. Include-path prefix filter (monorepo scoping)
        if self._include_prefixes and not _matches_any_prefix(rel_path, self._include_prefixes):
            logger.trace("EXCLUDE {}: not under any scope path", rel_path)
            return False

        logger.trace("INCLUDE {}", rel_path)
        return True

    # -- private helpers -----------------------------------------------------

    def _build_exclude_spec(self, settings: AtlasSettings) -> pathspec.PathSpec:
        """Compile the exclude spec from defaults (or override), ignore files, and settings."""
        base = list(settings.scope.exclude) if settings.scope.exclude is not None else list(_DEFAULT_EXCLUDE)
        base.extend(settings.scope.extend_exclude)

        # .gitignore and .atlasignore always applied regardless of exclude override
        gitignore = self._root / ".gitignore"
        if gitignore.is_file():
            gi_patterns = _read_ignore_file(gitignore)
            base.extend(gi_patterns)
            logger.debug("Loaded {} patterns from {}", len(gi_patterns), gitignore)

        atlasignore = self._root / ".atlasignore"
        if atlasignore.is_file():
            ai_patterns = _read_ignore_file(atlasignore)
            base.extend(ai_patterns)
            logger.debug("Loaded {} patterns from {}", len(ai_patterns), atlasignore)

        return pathspec.PathSpec.from_lines("gitignore", base)

    def _build_include_spec(self, settings: AtlasSettings) -> pathspec.PathSpec | None:
        """Compile include patterns into a PathSpec for file-extension filtering."""
        base = list(settings.scope.include) if settings.scope.include is not None else list(_DEFAULT_INCLUDE)
        base.extend(settings.scope.extend_include)
        return pathspec.PathSpec.from_lines("gitignore", base) if base else None

    def _build_include_prefixes(self, scope_paths: list[str] | None, settings: AtlasSettings) -> list[str]:
        """Normalise scope paths to POSIX form."""
        paths = scope_paths or settings.scope.paths or []
        return [p.replace("\\", "/").rstrip("/") for p in paths]

    def _is_dir_excluded(self, rel_dir: str) -> bool:
        """Check whether a directory should be pruned from the walk."""
        dir_pattern = f"{rel_dir}/"
        if self._exclude_spec.match_file(dir_pattern):
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
    Prunes directories in ``_DETECT_PRUNE_DIRS`` (e.g. .venv, node_modules).
    """
    root = project_root.resolve()
    packages: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune symlinked and excluded directories
        dirnames[:] = [d for d in dirnames if not Path(dirpath, d).is_symlink() and d not in _DETECT_PRUNE_DIRS]
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
    """Get the current git HEAD full hash, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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


_GIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")

_GIT_STATUS_MAP = {"A": "created", "M": "modified", "D": "deleted"}


def _git_changed_files(project_root: Path, from_hash: str) -> list[tuple[str, str]] | None:
    """Return files changed between *from_hash* and HEAD as ``[(path, change_type), ...]``.

    Uses ``git diff --name-status --no-renames`` so renames appear as delete+add.
    Returns ``None`` if git fails (invalid hash, not a repo, etc.) — caller
    falls back to full mode.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "--no-renames", from_hash, "--", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("git diff failed (rc={}): {}", result.returncode, result.stderr.strip())
            return None
    except FileNotFoundError, subprocess.TimeoutExpired:
        return None

    changes: list[tuple[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        status_code, file_path = parts
        change_type = _GIT_STATUS_MAP.get(status_code, "modified")
        # Normalise to forward slashes
        changes.append((file_path.replace("\\", "/"), change_type))
    return changes


# ---------------------------------------------------------------------------
# Pure-Python HEAD reader (no subprocess)
# ---------------------------------------------------------------------------


def _read_git_head(project_root: Path) -> str | None:  # noqa: PLR0911
    """Read the current git HEAD hash without spawning a subprocess.

    - If ``.git/HEAD`` contains ``ref: refs/heads/...``, read the ref file.
    - If the ref file is missing, fall back to ``.git/packed-refs``.
    - If HEAD is a detached 40-char hex hash, return directly.
    - Returns ``None`` for non-git directories or on any read error.

    Supports linked worktrees (where ``.git`` is a file pointing to the
    real git directory).
    """
    git_dir = resolve_git_dir(project_root)
    if git_dir is None:
        return None
    try:
        head_content = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None

    # Detached HEAD — raw 40-char hex
    if _GIT_HEX_RE.match(head_content):
        return head_content

    # Symbolic ref: "ref: refs/heads/main"
    if not head_content.startswith("ref: "):
        return None

    ref_path = head_content[5:].strip()
    ref_file = git_dir / ref_path.replace("/", os.sep)

    # Try loose ref file first
    try:
        return ref_file.read_text(encoding="utf-8").strip()
    except OSError:
        pass

    # Fall back to packed-refs
    packed_refs = git_dir / "packed-refs"
    try:
        for raw_line in packed_refs.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith(("#", "^")):
                continue
            parts = stripped.split(" ", 1)
            if len(parts) == 2 and parts[1] == ref_path:
                return parts[0]
    except OSError:
        pass

    return None


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StalenessInfo:
    """Result of a staleness check."""

    stale: bool
    last_indexed_commit: str | None = None
    current_commit: str | None = None
    changed_files: list[str] = field(default_factory=list)


class StalenessChecker:
    """Mtime-cached staleness checker for a git project.

    ``current_head()`` reads ``.git/HEAD`` via :func:`_read_git_head`,
    caching the result until the HEAD or ref file mtime changes.
    ``check()`` compares the current HEAD against the stored ``git_hash``
    on the Project node and optionally lists changed files.
    """

    def __init__(self, project_root: Path, *, project_name: str | None = None) -> None:
        self._root = project_root.resolve()
        self._project_name = project_name or derive_project_name(self._root)
        self._cached_hash: str | None = None
        self._cached_head_mtime: float | None = None
        self._cached_ref_mtime: float | None = None
        self._cached_ref_path: Path | None = None

    @property
    def project_name(self) -> str:
        return self._project_name

    def current_head(self) -> str | None:
        """Return the current HEAD hash, cached by file mtime."""
        git_dir = resolve_git_dir(self._root)
        if git_dir is None:
            self._cached_hash = None
            return None
        head_file = git_dir / "HEAD"

        try:
            head_mtime = head_file.stat().st_mtime
        except OSError:
            self._cached_hash = None
            return None

        # Determine ref file path for mtime tracking
        ref_path: Path | None = None
        try:
            head_content = head_file.read_text(encoding="utf-8").strip()
            if head_content.startswith("ref: "):
                ref_rel = head_content[5:].strip()
                ref_path = git_dir / ref_rel.replace("/", os.sep)
        except OSError:
            pass

        ref_mtime: float | None = None
        if ref_path is not None:
            with contextlib.suppress(OSError):
                ref_mtime = ref_path.stat().st_mtime

        # Check cache validity
        if (
            self._cached_hash is not None
            and head_mtime == self._cached_head_mtime
            and ref_path == self._cached_ref_path
            and ref_mtime == self._cached_ref_mtime
        ):
            return self._cached_hash

        # Cache miss — re-read
        result = _read_git_head(self._root)
        self._cached_hash = result
        self._cached_head_mtime = head_mtime
        self._cached_ref_path = ref_path
        self._cached_ref_mtime = ref_mtime
        return result

    async def check(self, graph: GraphClient, *, include_changed: bool = True) -> StalenessInfo:
        """Compare current HEAD against the stored git_hash on the Project node."""
        current = self.current_head()
        stored = await graph.get_project_git_hash(self.project_name)

        # Non-git directory — not stale by definition
        if current is None:
            return StalenessInfo(stale=False)

        # Never indexed — stale
        if stored is None:
            return StalenessInfo(stale=True, current_commit=current)

        # Hashes match — not stale
        if current == stored:
            return StalenessInfo(stale=False, last_indexed_commit=stored, current_commit=current)

        # Stale — optionally list changed files
        changed: list[str] = []
        if include_changed:
            raw = await asyncio.to_thread(_git_changed_files, self._root, stored)
            if raw is not None:
                changed = [path for path, _ in raw]

        return StalenessInfo(
            stale=True,
            last_indexed_commit=stored,
            current_commit=current,
            changed_files=changed,
        )


# ---------------------------------------------------------------------------
# Embedding model lock
# ---------------------------------------------------------------------------


async def _check_model_lock(
    graph: GraphClient,
    model: str,
    dimension: int,
    *,
    reindex: bool,
    cache: EmbedCache | None = None,
) -> None:
    """Enforce embedding model lock on the SchemaVersion node.

    - First run (no stored model): write current config.
    - Reindex: clear embeddings + embed_hash, rebuild vector indices,
      flush all model keys from Valkey cache, write new config.
    - Mismatch: raise RuntimeError with clear guidance.
    """
    if reindex:
        await graph.clear_all_embeddings()
        await graph.rebuild_vector_indices(dimension)
        if cache is not None:
            await cache.clear_all_models()
        await graph.set_embedding_config(model, dimension)
        return

    stored = await graph.get_embedding_config()
    if stored is not None:
        stored_model, _stored_dim = stored
        if stored_model != model:
            msg = (
                f"Embedding model changed from '{stored_model}' to '{model}'. "
                "Run 'atlas index --full' to rebuild all embeddings."
            )
            raise RuntimeError(msg)
    else:
        await graph.set_embedding_config(model, dimension)


# ---------------------------------------------------------------------------
# Dependency version extraction
# ---------------------------------------------------------------------------

_PEP508_RE = re.compile(r"^([A-Za-z0-9][\w.-]*)\s*(.*)")


def _parse_dependency_versions(project_root: Path) -> dict[str, str]:
    """Extract package name → version constraint from pyproject.toml dependencies."""
    pyproject = project_root / "pyproject.toml"
    if not pyproject.is_file():
        return {}
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except Exception:
        return {}
    deps: list[str] = data.get("project", {}).get("dependencies", [])
    versions: dict[str, str] = {}
    for dep in deps:
        match = _PEP508_RE.match(dep.strip())
        if match:
            pkg_name = match.group(1).lower().replace("-", "_")
            constraint = match.group(2).strip().rstrip(";").strip()
            if constraint:
                versions[pkg_name] = constraint
    return versions


# ---------------------------------------------------------------------------
# Main indexing orchestration
# ---------------------------------------------------------------------------


async def _create_package_hierarchy(graph: GraphClient, project_name: str, project_root: Path) -> int:
    """Create Project + Package nodes and CONTAINS edges. Returns package count."""
    await graph.merge_project_node(project_name)
    packages = _detect_packages(project_root)
    batch = [(qn, qn.rsplit(".", 1)[-1], f"{rel_path}/__init__.py") for qn, rel_path in packages]
    await graph.merge_package_batch(project_name, batch)
    return len(packages)


async def _run_pipeline(
    bus: EventBus,
    graph: GraphClient,
    settings: AtlasSettings,
    embed: EmbedClient | None,
    cache: EmbedCache | None,
    drain_timeout_s: float,
    *,
    project_root: Path | None = None,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
) -> Tier2ASTConsumer:
    """Start inline tier consumers and wait for the pipeline to drain.

    Returns the Tier2 consumer so callers can read accumulated stats.
    """
    await bus.ensure_group(Topic.FILE_CHANGED, "tier1-graph")
    await bus.ensure_group(Topic.AST_DIRTY, "tier2-ast")

    tier1 = Tier1GraphConsumer(bus, graph, settings)
    tier2 = Tier2ASTConsumer(bus, graph, settings, project_root=project_root)

    task1 = asyncio.create_task(tier1.run())
    task2 = asyncio.create_task(tier2.run())

    tier3: Tier3EmbedConsumer | None = None
    task3: asyncio.Task[None] | None = None
    if embed is not None:
        await bus.ensure_group(Topic.EMBED_DIRTY, "tier3-embed")
        tier3 = Tier3EmbedConsumer(bus, graph, embed, cache=cache)
        task3 = asyncio.create_task(tier3.run())

    try:
        await _wait_for_drain(
            bus,
            drain_timeout_s,
            embed_enabled=embed is not None,
            on_drain_progress=on_drain_progress,
        )
    finally:
        tier1.stop()
        tier2.stop()
        if tier3 is not None:
            tier3.stop()
        await asyncio.sleep(0.5)
        task1.cancel()
        task2.cancel()
        if task3 is not None:
            task3.cancel()
        for t in (task1, task2, task3):
            if t is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        if cache is not None:
            await cache.close()

    return tier2


@dataclass
class _DeltaDecision:
    """Result of the delta vs. full mode decision."""

    mode: str  # "full" | "delta"
    files_added: set[str]
    files_modified: set[str]
    files_deleted: set[str]


async def _decide_delta_mode(
    settings: AtlasSettings,
    graph: GraphClient,
    project_name: str,
    project_root: Path,
    current_file_set: set[str],
) -> _DeltaDecision:
    """Determine whether to use delta or full mode based on git diff and threshold."""
    stored_hash = await graph.get_project_git_hash(project_name)
    if stored_hash is None:
        return _DeltaDecision("full", set(), set(), set())

    git_changes = await asyncio.to_thread(_git_changed_files, project_root, stored_hash)
    if git_changes is None:
        return _DeltaDecision("full", set(), set(), set())

    old_file_paths = await graph.get_project_file_paths(project_name)
    git_changed_paths = {path for path, _ in git_changes}
    files_deleted = old_file_paths - current_file_set
    files_added = current_file_set - old_file_paths
    files_modified = (git_changed_paths & current_file_set) - files_added

    all_affected = files_added | files_modified | files_deleted
    ratio = len(all_affected) / len(current_file_set) if current_file_set else 1.0

    if ratio > settings.index.delta_threshold:
        logger.debug(
            "Delta ratio {:.0%} exceeds threshold {:.0%} — falling back to full mode",
            ratio,
            settings.index.delta_threshold,
        )
        return _DeltaDecision("full", set(), set(), set())

    if all_affected:
        logger.debug(
            "Delta mode: {} added, {} modified, {} deleted ({:.0%} of {} files)",
            len(files_added),
            len(files_modified),
            len(files_deleted),
            ratio,
            len(current_file_set),
        )
    else:
        logger.debug("Delta mode: no changes detected")

    return _DeltaDecision("delta", files_added, files_modified, files_deleted)


async def _publish_events(
    bus: EventBus,
    mode: str,
    files: list[str],
    decision: _DeltaDecision,
    *,
    project_name: str = "",
    project_root: str = "",
) -> int:
    """Publish FileChanged events and return the count published."""
    if mode == "delta":
        events: list[Event] = []
        events.extend(
            FileChanged(path=fp, change_type="created", project_name=project_name, project_root=project_root)
            for fp in decision.files_added
        )
        events.extend(
            FileChanged(path=fp, change_type="modified", project_name=project_name, project_root=project_root)
            for fp in decision.files_modified
        )
        events.extend(
            FileChanged(path=fp, change_type="deleted", project_name=project_name, project_root=project_root)
            for fp in decision.files_deleted
        )
        if events:
            await bus.publish_many(Topic.FILE_CHANGED, events)
        logger.debug("Published {} FileChanged events (delta)", len(events))
        return len(events)

    full_events: list[Event] = [
        FileChanged(path=file_path, change_type="created", project_name=project_name, project_root=project_root)
        for file_path in files
    ]
    if full_events:
        await bus.publish_many(Topic.FILE_CHANGED, full_events)
    logger.debug("Published {} FileChanged events (full)", len(full_events))
    return len(full_events)


def _record_index_metrics(span: Any, mode: str, files: int, entities: int, duration: float) -> None:
    """Record OTel span attributes and metrics for an indexing run."""
    if span is not None:
        span.set_attribute("mode", mode)
        span.set_attribute("files_scanned", files)
        span.set_attribute("entities_total", entities)
    m = get_metrics()
    m.index_files_total.add(files)
    m.index_entities_total.add(entities)
    m.index_duration.record(duration)


async def index_project(
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    *,
    scope_paths: list[str] | None = None,
    full_reindex: bool = False,
    drain_timeout_s: float = 600.0,
    project_name: str | None = None,
    project_root: Path | None = None,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
) -> IndexResult:
    """Run a full or delta index of the project through the event pipeline.

    1. Scan files
    2. Optionally wipe old data (full reindex)
    3. Decide full vs. delta mode (git diff, threshold check)
    4. Create Project + Package hierarchy in the graph
    5. Publish FileChanged events to Valkey (all or delta-only)
    6. Run inline Tier 1 + Tier 2 consumers until the pipeline drains
    7. Update Project metadata (counts, git hash, delta stats)

    In monorepo mode, *project_name* and *project_root* override the
    settings-derived defaults so that each sub-project can be indexed
    with its own root while sharing infra config from the monorepo settings.
    """
    project_name = project_name or derive_project_name(Path(settings.project_root))
    with _tracer.start_as_current_span("index_project", attributes={"project_name": project_name}) as idx_span:
        return await _index_project_inner(
            settings,
            graph,
            bus,
            scope_paths=scope_paths,
            full_reindex=full_reindex,
            drain_timeout_s=drain_timeout_s,
            project_name=project_name,
            project_root=project_root,
            span=idx_span,
            on_drain_progress=on_drain_progress,
        )


async def _index_project_inner(
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    *,
    scope_paths: list[str] | None = None,
    full_reindex: bool = False,
    drain_timeout_s: float = 600.0,
    project_name: str,
    project_root: Path | None = None,
    span: Any = None,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
) -> IndexResult:
    """Inner implementation of index_project with active span."""
    start = time.monotonic()
    project_root = (project_root or Path(settings.project_root)).resolve()

    # 1. Scan files
    files = scan_files(project_root, settings, scope_paths=scope_paths)
    logger.debug("Scanned {} indexable files", len(files))

    if not files:
        return IndexResult(files_scanned=0, files_published=0, entities_total=0, duration_s=time.monotonic() - start)

    # 2. Embedding setup + model lock check (skipped in lightweight mode)
    embed: EmbedClient | None = None
    cache: EmbedCache | None = None
    if settings.embeddings.enabled:
        embed = EmbedClient(settings.embeddings)
        if settings.embeddings.cache_ttl_days > 0:
            cache = EmbedCache(settings.redis, settings.embeddings)

    if full_reindex:
        logger.debug("Full reindex: deleting existing data for '{}'", project_name)
        await bus.flush()
        await graph.delete_project_data(project_name)
        if cache is not None:
            await cache.clear()

    if settings.embeddings.enabled:
        dimension = settings.embeddings.dimension or 768
        await _check_model_lock(graph, settings.embeddings.model, dimension, reindex=full_reindex, cache=cache)

    # 3. Decide full vs. delta mode
    if full_reindex:
        decision = _DeltaDecision("full", set(), set(), set())
    else:
        decision = await _decide_delta_mode(settings, graph, project_name, project_root, set(files))

    # 4. Create Project + Package hierarchy
    pkg_count = await _create_package_hierarchy(graph, project_name, project_root)
    logger.debug("Created {} package node(s)", pkg_count)

    # 5. Publish FileChanged events
    published = await _publish_events(bus, decision.mode, files, decision, project_name=project_name)

    # 6. Start inline consumers and wait for drain
    t2stats = None
    if published > 0:
        tier2 = await _run_pipeline(
            bus,
            graph,
            settings,
            embed,
            cache,
            drain_timeout_s,
            project_root=project_root,
            on_drain_progress=on_drain_progress,
        )
        t2stats = tier2.stats

    # 7. Set dependency versions on ExternalPackage nodes
    dep_versions = _parse_dependency_versions(project_root)
    if dep_versions:
        await graph.update_external_package_versions(project_name, dep_versions)

    # 8. Update Project metadata
    entity_count = await graph.count_entities(project_name)
    git_hash = _get_git_hash(project_root)
    metadata: dict[str, Any] = {
        "last_indexed_at": time.time(),
        "file_count": len(files),
        "entity_count": entity_count,
        "index_mode": decision.mode,
    }
    if git_hash:
        metadata["git_hash"] = git_hash
    if decision.mode == "delta":
        metadata["delta_files_added"] = len(decision.files_added)
        metadata["delta_files_modified"] = len(decision.files_modified)
        metadata["delta_files_deleted"] = len(decision.files_deleted)
    await graph.update_project_metadata(project_name, **metadata)

    duration = time.monotonic() - start
    delta_stats = _build_delta_stats(decision, t2stats) if decision.mode == "delta" else None

    logger.debug(
        "Indexing complete ({}): {} files scanned, {} published, {} entities, {:.1f}s",
        decision.mode,
        len(files),
        published,
        entity_count,
        duration,
    )

    _record_index_metrics(span, decision.mode, len(files), entity_count, duration)

    return IndexResult(
        files_scanned=len(files),
        files_published=published,
        entities_total=entity_count,
        duration_s=duration,
        mode=decision.mode,
        delta_stats=delta_stats,
    )


async def index_monorepo(
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    *,
    scope_projects: list[str] | None = None,
    full_reindex: bool = False,
    drain_timeout_s: float = 600.0,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
) -> list[IndexResult]:
    """Index a monorepo: detect sub-projects, index each, resolve cross-project imports.

    Flow:
    1. Detect sub-projects via markers and explicit config.
    2. Filter by *scope_projects* if specified (supports exact match + glob).
    3. Index each sub-project via ``index_project()`` with overridden root + name.
    4. Index root project (files not inside any sub-project).
    5. Resolve cross-project imports and create DEPENDS_ON edges.

    If *on_progress* is provided, it is called after each sub-project finishes
    with ``(project_name, current_1based, total)``.
    """
    with _tracer.start_as_current_span("index_monorepo"):
        return await _index_monorepo_inner(
            settings,
            graph,
            bus,
            scope_projects=scope_projects,
            full_reindex=full_reindex,
            drain_timeout_s=drain_timeout_s,
            on_progress=on_progress,
            on_drain_progress=on_drain_progress,
        )


@dataclass
class _ProjectPublishResult:
    """Result of the publish phase for a single project within a monorepo."""

    project_name: str
    project_root: Path
    files_scanned: int
    files_published: int
    mode: str  # "full" | "delta"
    decision: _DeltaDecision


async def _publish_project(
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    project_name: str,
    project_root: Path,
    files: list[str],
    *,
    full_reindex: bool = False,
) -> _ProjectPublishResult:
    """Scan, decide delta/full, create packages, and publish events for one project.

    Does NOT create consumers or wait for drain — callers manage the shared pipeline.
    """
    if full_reindex:
        logger.debug("Full reindex: deleting existing data for '{}'", project_name)
        await graph.delete_project_data(project_name)

    # Decide full vs. delta mode
    if full_reindex:
        decision = _DeltaDecision("full", set(), set(), set())
    else:
        decision = await _decide_delta_mode(settings, graph, project_name, project_root, set(files))

    # Create Project + Package hierarchy
    pkg_count = await _create_package_hierarchy(graph, project_name, project_root)
    logger.debug("'{}': {} package node(s)", project_name, pkg_count)

    # Publish FileChanged events
    published = await _publish_events(
        bus, decision.mode, files, decision, project_name=project_name, project_root=str(project_root)
    )

    return _ProjectPublishResult(
        project_name=project_name,
        project_root=project_root,
        files_scanned=len(files),
        files_published=published,
        mode=decision.mode,
        decision=decision,
    )


async def _index_monorepo_inner(  # noqa: PLR0912, PLR0915
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    *,
    scope_projects: list[str] | None = None,
    full_reindex: bool = False,
    drain_timeout_s: float = 600.0,
    on_progress: Callable[[str, int, int], None] | None = None,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
) -> list[IndexResult]:
    """Inner implementation of index_monorepo.

    Uses a shared consumer pipeline across all sub-projects: file discovery
    and publishing run per sub-project, but tier consumers run continuously
    with a single drain at the end.
    """
    project_root = Path(settings.project_root).resolve()
    sub_projects = detect_sub_projects(project_root, settings.monorepo)

    if not sub_projects:
        logger.info("No sub-projects detected — falling back to single-project index")
        result = await index_project(settings, graph, bus, full_reindex=full_reindex, drain_timeout_s=drain_timeout_s)
        return [result]

    logger.info("Detected {} sub-project(s): {}", len(sub_projects), ", ".join(sp.name for sp in sub_projects))

    # Filter by scope_projects if specified (matches on bare DetectedProject.name)
    if scope_projects:
        filtered: list[DetectedProject] = []
        for sp in sub_projects:
            for pattern in scope_projects:
                if sp.name == pattern or fnmatch.fnmatch(sp.name, pattern):
                    filtered.append(sp)
                    break
        sub_projects = filtered
        logger.info("Scoped to {} sub-project(s): {}", len(sub_projects), ", ".join(sp.name for sp in sub_projects))

    # Compute the root name once — sub-projects are prefixed with it
    root_name = derive_project_name(project_root)

    # Pre-scan root files to determine if there's a root project (needed for total count)
    sub_paths = [sp.path for sp in sub_projects]
    root_scope = FileScope(project_root, settings)
    root_files = root_scope.scan()
    root_only_files = [f for f in root_files if not any(f == sp or f.startswith(sp + "/") for sp in sub_paths)]
    has_root = bool(root_only_files)
    total = len(sub_projects) + (1 if has_root else 0)

    # --- Shared embedding resources (created once) ---
    embed: EmbedClient | None = None
    cache: EmbedCache | None = None
    if settings.embeddings.enabled:
        embed = EmbedClient(settings.embeddings)
        if settings.embeddings.cache_ttl_days > 0:
            cache = EmbedCache(settings.redis, settings.embeddings)
        dimension = settings.embeddings.dimension or 768
        await _check_model_lock(graph, settings.embeddings.model, dimension, reindex=full_reindex, cache=cache)

    if full_reindex:
        await bus.flush()
        if cache is not None:
            await cache.clear()

    # --- Start shared consumers (once for entire monorepo) ---
    await bus.ensure_group(Topic.FILE_CHANGED, "tier1-graph")
    await bus.ensure_group(Topic.AST_DIRTY, "tier2-ast")

    tier1 = Tier1GraphConsumer(bus, graph, settings)
    tier2 = Tier2ASTConsumer(bus, graph, settings, project_root=project_root)

    consumer_tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(tier1.run()),
        asyncio.create_task(tier2.run()),
    ]

    tier3: Tier3EmbedConsumer | None = None
    if embed is not None:
        await bus.ensure_group(Topic.EMBED_DIRTY, "tier3-embed")
        tier3 = Tier3EmbedConsumer(bus, graph, embed, cache=cache)
        consumer_tasks.append(asyncio.create_task(tier3.run()))

    start = time.monotonic()
    publish_results: list[_ProjectPublishResult] = []

    try:
        # --- Publish phase: scan + publish per sub-project (fast) ---
        for i, sub in enumerate(sub_projects, 1):
            prefixed_name = f"{root_name}/{sub.name}"
            logger.debug("Publishing sub-project '{}' at {}", prefixed_name, sub.path)
            if on_progress is not None:
                on_progress(prefixed_name, i - 1, total)

            sub_files = scan_files(sub.root, settings)
            pr = await _publish_project(
                settings, graph, bus, prefixed_name, sub.root, sub_files, full_reindex=full_reindex
            )
            publish_results.append(pr)

            if on_progress is not None:
                on_progress(prefixed_name, i, total)

        # Publish root project files (outside any sub-project)
        root_pr: _ProjectPublishResult | None = None
        if root_only_files:
            logger.debug(
                "Publishing root project '{}' ({} file(s) outside sub-projects)", root_name, len(root_only_files)
            )
            if on_progress is not None:
                on_progress(root_name, total - 1, total)

            root_pr = await _publish_project(
                settings, graph, bus, root_name, project_root, root_only_files, full_reindex=full_reindex
            )
            publish_results.append(root_pr)

            if on_progress is not None:
                on_progress(root_name, total, total)

        # --- Wait for ALL tiers to drain (once) ---
        await _wait_for_drain(
            bus,
            drain_timeout_s,
            embed_enabled=embed is not None,
            on_drain_progress=on_drain_progress,
        )

    finally:
        # --- Tear down consumers (once) ---
        tier1.stop()
        tier2.stop()
        if tier3 is not None:
            tier3.stop()
        await asyncio.sleep(0.5)
        for task in consumer_tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if cache is not None:
            await cache.close()

    # --- Update metadata + build results per project ---
    results: list[IndexResult] = []
    for pr in publish_results:
        # Set dependency versions
        dep_versions = _parse_dependency_versions(pr.project_root)
        if dep_versions:
            await graph.update_external_package_versions(pr.project_name, dep_versions)

        entity_count = await graph.count_entities(pr.project_name)
        git_hash = _get_git_hash(pr.project_root)
        metadata: dict[str, Any] = {
            "last_indexed_at": time.time(),
            "file_count": pr.files_scanned,
            "entity_count": entity_count,
            "index_mode": pr.mode,
        }
        if git_hash:
            metadata["git_hash"] = git_hash
        if pr.mode == "delta":
            metadata["delta_files_added"] = len(pr.decision.files_added)
            metadata["delta_files_modified"] = len(pr.decision.files_modified)
            metadata["delta_files_deleted"] = len(pr.decision.files_deleted)
        await graph.update_project_metadata(pr.project_name, **metadata)

        # Use shared start time — in monorepo mode all projects share one pipeline,
        # so per-project publish timestamps don't reflect actual processing duration.
        duration = time.monotonic() - start
        delta_stats = _build_delta_stats(pr.decision, tier2.stats) if pr.mode == "delta" else None
        results.append(
            IndexResult(
                files_scanned=pr.files_scanned,
                files_published=pr.files_published,
                entities_total=entity_count,
                duration_s=duration,
                mode=pr.mode,
                delta_stats=delta_stats,
            )
        )

    # Cross-project import resolution
    all_project_names = [f"{root_name}/{sp.name}" for sp in sub_projects]
    if root_only_files:
        all_project_names.append(root_name)

    if len(all_project_names) > 1:
        rewired = await graph.resolve_cross_project_imports(all_project_names)
        logger.debug("Cross-project import resolution: {} imports rewired", rewired)
        depends_count = await graph.create_depends_on_edges(all_project_names)
        logger.debug("Created {} DEPENDS_ON edge(s)", depends_count)

    logger.debug("Monorepo indexing completed in {:.1f}s", time.monotonic() - start)

    return results


def _build_delta_stats(decision: _DeltaDecision, t2stats: Any) -> DeltaStats:
    """Build DeltaStats from the decision and Tier2 stats."""
    return DeltaStats(
        files_added=len(decision.files_added),
        files_modified=len(decision.files_modified),
        files_deleted=len(decision.files_deleted),
        entities_added=t2stats.entities_added if t2stats else 0,
        entities_modified=t2stats.entities_modified if t2stats else 0,
        entities_deleted=t2stats.entities_deleted if t2stats else 0,
        entities_unchanged=t2stats.entities_unchanged if t2stats else 0,
    )


async def _wait_for_drain(
    bus: EventBus,
    timeout_s: float,
    *,
    embed_enabled: bool = True,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
) -> None:
    """Poll stream groups until Tier 1, Tier 2, and (optionally) Tier 3 are drained.

    If *on_drain_progress* is provided, it is called each poll cycle with
    ``(t1_remaining, t2_remaining, t3_remaining)`` so callers can display
    pipeline progress to the user.
    """
    deadline = time.monotonic() + timeout_s
    settled_since: float | None = None
    poll_interval = 0.5

    while time.monotonic() < deadline:
        queries: list[tuple[Topic, str]] = [
            (Topic.FILE_CHANGED, "tier1-graph"),
            (Topic.AST_DIRTY, "tier2-ast"),
        ]
        if embed_enabled:
            queries.append((Topic.EMBED_DIRTY, "tier3-embed"))

        infos = await bus.stream_group_info_multi(queries)

        t1_remaining = infos[0]["pending"] + infos[0]["lag"]
        t2_remaining = infos[1]["pending"] + infos[1]["lag"]
        t3_remaining = infos[2]["pending"] + infos[2]["lag"] if embed_enabled else 0

        if on_drain_progress is not None:
            on_drain_progress(t1_remaining, t2_remaining, t3_remaining)

        if t1_remaining == 0 and t2_remaining == 0 and t3_remaining == 0:
            if settled_since is None:
                settled_since = time.monotonic()
            elif time.monotonic() - settled_since >= 1.0:
                logger.debug("Pipeline drained after {:.1f}s settling", time.monotonic() - settled_since)
                return
            # Adaptive backoff: poll less frequently once idle
            poll_interval = min(2.0, poll_interval * 1.5)
        else:
            settled_since = None
            poll_interval = 0.5  # reset to fast polling when work is happening

        await asyncio.sleep(poll_interval)

    logger.warning("Pipeline drain timed out after {:.0f}s", timeout_s)
