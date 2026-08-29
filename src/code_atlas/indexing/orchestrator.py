"""Python indexer — scans files and drives the event pipeline for atlas index."""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import os
import re
import subprocess
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

import pathspec
from loguru import logger

from code_atlas.events import EmbedDirty, EntityRef, Event, EventBus, FileChanged, Significance, Topic
from code_atlas.indexing.consumers import ASTConsumer, BatchPolicy, EmbedConsumer
from code_atlas.parsing.ast import get_language_for_file
from code_atlas.parsing.languages.python import module_qualified_name
from code_atlas.search.embeddings import EmbedClient, EmbeddingError
from code_atlas.settings import derive_project_name, resolve_git_dir
from code_atlas.telemetry import get_metrics, get_tracer

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Protocol

    from code_atlas.graph.protocol import GraphBackend

    class _HasProgress(Protocol):
        """Anything teardown can ask "are you still working?".

        Structural on purpose: only the heartbeat is read, so the waiter does not
        need to import a concrete consumer, and a test double is a legitimate
        implementation rather than something to cast around.
        """

        @property
        def progress_at(self) -> float: ...

    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings, MonorepoSettings

    type ManifestParser = Callable[[str], dict[str, str]]
    """Parses dependency-manifest text into import-space name → version constraint."""

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
    # Terraform provider cache — the .tf equivalent of node_modules, and now
    # reachable because *.tf/*.hcl are indexable.
    ".terraform/",
    # Lock / generated manifests. Machine-written, enormous, and semantically
    # empty; they exist here rather than being left to .gitignore because
    # lockfiles are deliberately *committed*, so .gitignore never covers them.
    # Only the ones whose suffix is now indexable can actually reach the
    # scanner (package-lock.json, pnpm-lock.yaml, .terraform.lock.hcl); the
    # rest are listed defensively so a future *.lock include cannot regress.
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "Cargo.lock",
    "composer.lock",
    "Gemfile.lock",
    ".terraform.lock.hcl",
    "*.min.json",
    # Secret-bearing files. This deny-list is a HARD BACKSTOP, not a convenience:
    # widening the include list to *.yaml/*.json/*.toml/*.tfvars made files like
    # secrets.yaml and gcp-key.json reachable by the scanner for the first time,
    # and an indexed entity is also an *embedded* one — its content leaves the
    # machine for the embedding API. Verified before this list existed:
    # secrets.yaml, gcp-key.json, service-account.json, local.settings.json,
    # terraform.tfvars and credentials.toml all passed is_included().
    #
    # .gitignore is NOT an adequate substitute. It is the usual place these are
    # hidden, but atlas never reads .git/info/exclude or core.excludesFile, drops
    # the repo-root .gitignore entirely when a monorepo sub-project is rooted in a
    # subdirectory, and matches case-sensitively where git on Windows does not —
    # so any of those alone re-exposes the file. Deny by name here regardless.
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "*.jks",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".npmrc",
    ".netrc",
    ".pypirc",
    "credentials",
    "credentials.*",
    "secrets.*",
    "secret.*",
    "*-secrets.*",
    "*.secrets.*",
    "service-account*.json",
    "*-key.json",
    "*.tfvars",
    "*.tfvars.json",
    "local.settings.json",
    ".ssh/",
    ".aws/",
    ".gnupg/",
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
    # Terraform / HCL
    "*.tf",
    "*.tfvars",
    "*.hcl",
    # Shell
    "*.sh",
    "*.bash",
    "*.zsh",
    # Container builds. Extensionless variants are dispatched by
    # LanguageConfig.filenames, not by suffix — both casings are listed because
    # pathspec matching is case-sensitive while filename dispatch lowercases
    # the basename, so the registry key is always the lowercase form.
    "*.dockerfile",
    "*.containerfile",
    "Dockerfile",
    "dockerfile",
    "Containerfile",
    "containerfile",
    # SQL
    "*.sql",
    # Structured config / data. Deliberately broad — see the volume note below.
    "*.yaml",
    "*.yml",
    "*.json",
    "*.toml",
    "*.xml",
    # Salesforce Apex. Registered ahead of the parser module so the scope
    # plumbing lands once; until an apex language module exists these files
    # pass is_included() but are dropped by scan()'s language-support gate.
    "*.cls",
    "*.trigger",
]

# NOTE: the config/data globs above (*.yaml, *.yml, *.json, *.toml, *.xml) put
# every committed config file in every repo into indexing scope by default. That
# is a deliberate trade: those files carry real architectural signal (CI
# pipelines, k8s manifests, compose topologies) that the graph cannot see
# otherwise. The cost is bounded by three things and no more:
#   1. .gitignore is always applied (_build_exclude_spec), so build output and
#      test artefacts — the bulk of generated config noise — are already gone.
#   2. Committed lockfiles, which .gitignore by definition does not cover, are
#      excluded explicitly in _DEFAULT_EXCLUDE above.
#   3. A parser that recognises no dialect in a file returns an empty
#      ParsedFile, which creates no nodes and therefore no embeddings.
# What remains is per-file scan + hash + one tree-sitter parse on first index
# (amortised to a hash check afterwards), plus a larger denominator in the
# delta_threshold ratio that decides full-vs-incremental reindex. Users who do
# not want config indexed at all should set `[scope] include` in atlas.toml.

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
                marker = min(matched_markers)  # deterministic
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


def classify_file_project(rel_path: str, sub_projects: list[DetectedProject]) -> DetectedProject | None:
    """Return the most specific (longest path prefix) sub-project owning *rel_path*.

    Returns ``None`` if the file doesn't belong to any sub-project.
    """
    best: DetectedProject | None = None
    best_len = -1
    for proj in sub_projects:
        prefix = proj.path
        if (rel_path == prefix or rel_path.startswith(prefix + "/")) and len(prefix) > best_len:
            best_len = len(prefix)
            best = proj
    return best


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
    drained: bool = True  # False when the pipeline drain timed out; git_hash was NOT advanced
    # extension -> count of files the scope wanted but no installed grammar could read.
    # Empty on a complete install; non-empty means the index is partial BY INSTALL, which
    # the caller must say out loud rather than reporting a clean run (ATL-110).
    skipped_no_grammar: dict[str, int] = field(default_factory=dict)


async def _record_architecture_snapshot(
    graph: GraphBackend, project_name: str, git_hash: str, skipped_no_grammar: dict[str, int]
) -> None:
    """Capture the architecture-health numbers for this index run.

    On the index path, not the view path. Computing these when someone opens a page would
    make the history a record of who looked at it rather than of how the code changed.

    Every failure is swallowed: this is telemetry about the run, and it must not be able
    to fail the run. The snapshot carries the module count and any language whose grammar
    was missing, because a propagation cost that rose when C++ extraction improved is not
    a codebase that decayed — without the coverage the two are indistinguishable.
    """
    try:
        from code_atlas.server.analysis import fetch_architecture_pairs  # noqa: PLC0415
        from code_atlas.server.architecture import analyse  # noqa: PLC0415
        from code_atlas.server.architecture_history import record, snapshot_from_metrics  # noqa: PLC0415

        # Same edge source and defaults as the architecture page (production
        # modules, structural+resolved evidence) — the history used to record the
        # near-empty direct Module->Module import edges, so its numbers were not
        # comparable to anything the page showed. The trend view's own guard
        # marks the definition switch "unclear" once, since the module count jumps.
        source = await fetch_architecture_pairs(graph, project_name)
        edges = sorted(source.pairs)
        if not edges:
            return
        nodes = sorted({n for edge in edges for n in edge})
        snapshot = snapshot_from_metrics(
            analyse(nodes, edges),
            commit=git_hash,
            skipped_languages=tuple(sorted(skipped_no_grammar)),
        )
        await record(graph, project_name, snapshot)
    except Exception as exc:  # telemetry may never break indexing
        logger.debug("Skipped architecture snapshot for {}: {}", project_name, exc)


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
        # Nested gitignore specs, discovered lazily (see _check_nested_gitignore)
        self._nested_specs: dict[str, pathspec.PathSpec] = {}
        self._nested_checked: set[str] = set()
        # extension -> files the scope wanted but no installed grammar could read.
        # Populated by scan(); read by the caller to tell "no code here" apart from
        # "you did not install the extra" (ATL-110).
        self.skipped_no_grammar: Counter[str] = Counter()

    # -- public API ----------------------------------------------------------

    def scan(self) -> list[str]:
        """Walk the project tree and return sorted relative POSIX paths.

        Files are filtered through the global ignore spec, nested
        ``.gitignore`` files, include-path prefixes, and language support.

        Also refreshes :attr:`skipped_no_grammar`, so a caller can report a partial
        index rather than a clean one. Reset per call — a scope is reusable, and a
        stale count would outlive the scan it describes.
        """
        self.skipped_no_grammar.clear()
        result: list[str] = []

        for dirpath, dirnames, filenames in os.walk(self._root):
            rel_dir = Path(dirpath).relative_to(self._root).as_posix()
            if rel_dir == ".":
                rel_dir = ""

            # Discover nested .gitignore (non-root directories only)
            if rel_dir:
                self._check_nested_gitignore(rel_dir)

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
                    # Recorded, not merely skipped. A file the scope WANTED and no
                    # grammar could read is the difference between "this repo has no
                    # code" and "you did not install the extra" (ATL-110).
                    self.skipped_no_grammar[PurePosixPath(rel_path).suffix.lower()] += 1
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
            self._check_nested_gitignore(ancestor)
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
        """Normalise scope paths to POSIX form.

        ``scope_paths is None`` means "not overridden by the caller" -- fall back to
        ``settings.scope.paths``. An explicit ``[]`` must survive as-is (unrestricted),
        not fall back too: the monorepo indexer passes exactly that for a sub-project
        an ancestor scope path already covers in full (see ``_sub_project_scope_paths``)
        -- an ``or`` chain here would silently replace it with the un-translated,
        repo-root-relative global list, which matches nothing under a sub-project's
        own root and is how every scoped monorepo sub-project indexed zero files.
        """
        paths = settings.scope.paths if scope_paths is None else scope_paths
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
            self._check_nested_gitignore(ancestor)
            spec = self._nested_specs.get(ancestor)
            if spec is not None:
                sub_path = "/".join(parts[depth:]) + "/"
                if spec.match_file(sub_path):
                    return True

        return False

    def _check_nested_gitignore(self, rel_dir: str) -> None:
        """Discover and cache *rel_dir*'s own ``.gitignore``, if not already checked.

        Populated lazily on first access (from ``scan()``'s walk, or directly
        from ``is_included()``/``_is_dir_excluded()``) so callers that never
        call ``scan()`` — e.g. the file watcher, which queries ``is_included()``
        one path at a time — still see nested ``.gitignore`` exclusions.
        """
        if rel_dir in self._nested_checked:
            return
        self._nested_checked.add(rel_dir)
        nested_gi = self._root / rel_dir.replace("/", os.sep) / ".gitignore"
        if nested_gi.is_file():
            patterns = _read_ignore_file(nested_gi)
            if patterns:
                self._nested_specs[rel_dir] = pathspec.PathSpec.from_lines("gitignore", patterns)
                logger.debug("Loaded {} patterns from {}", len(patterns), nested_gi)


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

    Use :func:`scan_files_reporting_gaps` where the caller needs to know what was
    skipped for want of a grammar; this signature stays list-only because 27 of its
    28 call sites want exactly that and should not pay for the one that does not.
    """
    return FileScope(project_root, settings, scope_paths).scan()


def scan_files_reporting_gaps(
    project_root: str | Path,
    settings: AtlasSettings,
    scope_paths: list[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """:func:`scan_files`, plus the extensions no installed grammar could read.

    The second element is empty on a complete install. Non-empty means the index is
    partial **by install choice**, which is a different thing from an empty repository
    and has to be said out loud -- see :class:`IndexResult.skipped_no_grammar`.
    """
    scope = FileScope(project_root, settings, scope_paths)
    files = scope.scan()
    return files, dict(scope.skipped_no_grammar)


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


def _sub_project_in_scope(global_paths: list[str], sub_path: str) -> bool:
    """Whether a monorepo sub-project overlaps ``settings.scope.paths`` at all.

    Both directions count as overlap: an ancestor global path covering the whole
    sub-project, or a global path nested inside it. A sub-project with neither is
    entirely outside the configured scope and must not be indexed at all -- not
    indexed-with-zero-files, which is indistinguishable from "nothing changed".
    """
    for raw in global_paths:
        p = raw.rstrip("/")
        if p == sub_path or sub_path == p or sub_path.startswith(p + "/") or p.startswith(sub_path + "/"):
            return True
    return False


def _sub_project_scope_paths(global_paths: list[str], sub_path: str) -> list[str]:
    """Translate repo-root-relative ``scope.paths`` into *sub_path*-relative include
    prefixes for one monorepo sub-project's own :class:`FileScope`.

    Only called for a sub-project ``_sub_project_in_scope`` already confirmed
    overlaps -- a sub-project with zero overlap is dropped before reaching here, so
    this never needs to represent "nothing in scope" (which an empty list cannot:
    ``FileScope`` treats ``[]`` as unrestricted, not as "match nothing").

    Returns ``[]`` (unrestricted) when an ancestor or exact-match global path covers
    the whole sub-project; otherwise the non-empty list of paths nested under it,
    each relative to the sub-project's own root rather than the repo root.
    """
    translated: list[str] = []
    for raw in global_paths:
        p = raw.rstrip("/")
        if p == sub_path or sub_path == p or sub_path.startswith(p + "/"):
            return []  # an ancestor (or exact match) covers the whole sub-project
        if p.startswith(sub_path + "/"):
            translated.append(p[len(sub_path) + 1 :])
    return translated


# ---------------------------------------------------------------------------
# Package detection
# ---------------------------------------------------------------------------


def _detect_packages(project_root: Path, *, exclude_dirs: list[str] | None = None) -> list[tuple[str, str]]:
    """Find Python packages (dirs with __init__.py).

    Returns list of ``(qualified_name, relative_posix_path)`` sorted by depth.
    Prunes directories in ``_DETECT_PRUNE_DIRS`` (e.g. .venv, node_modules) and,
    if given, *exclude_dirs* (relative POSIX paths owned by sub-projects — keeps
    a monorepo root's package hierarchy from reaching into sub-project trees).
    """
    root = project_root.resolve()
    packages: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir

        # Prune symlinked, excluded, and sub-project directories
        dirnames[:] = [
            d
            for d in dirnames
            if not Path(dirpath, d).is_symlink()
            and d not in _DETECT_PRUNE_DIRS
            and not (exclude_dirs and _matches_any_prefix(f"{rel_dir}/{d}" if rel_dir else d, exclude_dirs))
        ]

        if "__init__.py" in filenames and rel_dir:
            # Shared parser derivation so Package uids converge with parsed
            # __init__.py entities (strips source roots like 'src/')
            qn = module_qualified_name(f"{rel_dir}/__init__.py")
            packages.append((qn, rel_dir))
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


def _git_tracked_files(project_root: Path) -> list[str] | None:
    """Return git's tracked (index) file list for *project_root*, or None if unavailable.

    Reads from git's index, independent of the atlas scope/.atlasignore filters
    and of the working tree's current contents — used to corroborate a
    zero-file scan before trusting it as a genuine full deletion.
    ``None`` means "no independent signal" (not a git repo, git failed, or
    *project_root* itself is inaccessible) — callers must NOT treat that as
    confirmation of anything.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
    except OSError, subprocess.TimeoutExpired:
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


_GIT_HEX_RE = re.compile(r"^[0-9a-f]{40}$")

_GIT_STATUS_MAP = {"A": "created", "M": "modified", "D": "deleted"}


def _git_changed_files(project_root: Path, from_hash: str) -> list[tuple[str, str]] | None:
    """Return files changed between *from_hash* and HEAD as ``[(path, change_type), ...]``.

    Uses ``git diff --name-status --no-renames`` so renames appear as delete+add.
    Paths are relative to *project_root* (POSIX separators), matching
    ``scan_files()`` output — required for monorepo sub-projects and any
    project_root below the git top-level.
    Returns ``None`` if git fails (invalid hash, not a repo, etc.) — caller
    falls back to full mode.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", "--no-renames", "--relative", from_hash, "--", "."],
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


async def _resolve_dimension(embed: EmbedClient, configured: int | None) -> int:
    """Return the embedding dimension, verified against the service that will produce it.

    A configured dimension used to be returned on faith, never probed. That is how the
    shipped defaults could disagree with each other and nothing noticed: ``atlas.toml``
    hardcodes ``dimension = 768``, ``settings.embeddings.model`` defaults to
    ``nomic-ai/nomic-embed-code``, and the bundled TEI container serves
    ``Qwen/Qwen3-Embedding-0.6B`` — three values, no two of which have to agree, and TEI
    ignores the requested model name entirely so nothing downstream could tell (ATL-111).

    The consequence was not a bad number in a report. A dimension that disagrees with the
    vector index fails every embed batch, which the consumer retries five times and then
    parks as poison — while indexing reports success.

    So: always probe, and treat a disagreement as fatal. The service is the authority on
    what it emits; the config is a claim about it. When the probe itself fails the claim
    cannot be checked, which is a reason to warn and continue rather than to stop —
    embeddings may simply be unreachable right now, and that is already handled downstream.
    """
    try:
        detected = await embed.detect_dimension()
    except EmbeddingError as exc:
        if configured is None:
            raise
        logger.warning(
            "Could not verify embedding dimension against the service ({}); trusting the configured {}",
            exc,
            configured,
        )
        return configured

    if configured is None:
        logger.info("Auto-detected embedding dimension: {}", detected)
        return detected

    if configured != detected:
        msg = (
            f"Configured embedding dimension {configured} does not match the {detected} "
            f"the service actually returns. Every embed batch would fail and be parked as "
            f"poison while indexing reported success. Either unset [embeddings] dimension "
            f"to auto-detect, or point the service at a model that emits {configured}."
        )
        raise RuntimeError(msg)
    return configured


def _describe_other_projects(models: dict[str, str], project: str) -> str:
    """Render the other projects' models for a lock error message."""
    others = sorted((p, m) for p, m in models.items() if p != project)
    if not others:
        return ""
    listed = ", ".join(f"{p}='{m}'" for p, m in others)
    return f" Other projects in this database: {listed}."


async def _check_model_lock(
    graph: GraphClient,
    model: str,
    dimension: int,
    *,
    project: str,
    reindex: bool,
) -> None:
    """Enforce the embedding locks: dimension database-wide, model per project.

    The split is not a preference, it is what the storage is. Vector indices are one
    per label for the whole database and carry a single dimension, so **dimension has
    to be global** — a change rebuilds indices everyone shares. A **model is per
    project**: it decides which space a vector lives in, and nothing about it is
    shared.

    Both were global before ATL-135, and that made two projects on one Memgraph
    mutually exclusive. Whichever indexed last owned the lock; every other project
    got ``Embedding model changed from X to Y`` on every run, with `--full` as the
    only offered remedy — and `--full` cleared embeddings database-wide, silently
    destroying the other projects' vectors. Measured here: 25,305 vectors under one
    model and 6,691 under another, coexisting at the same 1536 dimensions.
    """
    stored = await graph.get_embedding_config()

    # -- First run against this store ---------------------------------------- #
    if stored is None:
        await graph.set_embedding_config(model, dimension)
        await graph.set_project_embedding_model(project, model)
        return

    _stored_model, stored_dim = stored

    # -- Dimension: global, because the vector indices are ------------------- #
    if stored_dim != dimension:
        counts = await graph.count_embeddings_by_project()
        if not reindex:
            affected = ", ".join(f"{p} ({c:,} vectors)" for p, c in sorted(counts.items()))
            msg = (
                f"Embedding dimension changed from {stored_dim} to {dimension}. "
                "Vector indices are shared by every project in this database, so this "
                "rebuilds all of them. Run 'atlas index --full' to proceed."
                + (f" This will re-embed: {affected}." if affected else "")
            )
            raise RuntimeError(msg)
        if counts:
            logger.warning(
                "Dimension {} → {}: clearing embeddings for EVERY project in this database "
                "({}). Vector indices are shared and cannot be rebuilt per project.",
                stored_dim,
                dimension,
                ", ".join(f"{p}={c:,}" for p, c in sorted(counts.items())),
            )
        cleared = await graph.clear_embeddings(None)
        await graph.rebuild_vector_indices(dimension)
        await graph.set_embedding_config(model, dimension)
        await graph.set_project_embedding_model(project, model)
        logger.info("Cleared {:,} vectors for the dimension change", cleared)
        return

    # -- Model: per project -------------------------------------------------- #
    project_model = await graph.get_project_embedding_model(project)
    if project_model == model:
        return

    if project_model is None:
        # A project indexed before the per-project lock existed. Its vectors were
        # written by runs using its own configuration, so the configured model is
        # the right thing to record — but say so, because the one case this gets
        # wrong is a model changed while indexing was already failing.
        existing = (await graph.count_embeddings_by_project()).get(project, 0)
        await graph.set_project_embedding_model(project, model)
        if existing:
            logger.warning(
                "Project '{}' has {:,} vectors but no recorded embedding model; adopting the "
                "configured '{}'. If you changed the model while indexing was failing, those "
                "vectors are from the old one — run 'atlas index --full' to re-embed.",
                project,
                existing,
                model,
            )
        return

    if not reindex:
        models = await graph.get_embedding_models_by_project()
        msg = (
            f"Embedding model for project '{project}' changed from '{project_model}' to "
            f"'{model}'. Run 'atlas index --full' to re-embed this project — other projects "
            "in this database are unaffected." + _describe_other_projects(models, project)
        )
        raise RuntimeError(msg)

    cleared = await graph.clear_embeddings(project)
    await graph.set_project_embedding_model(project, model)
    logger.info(
        "Model '{}' → '{}' for project '{}': cleared {:,} vectors, other projects untouched",
        project_model,
        model,
        project,
        cleared,
    )


# ---------------------------------------------------------------------------
# Dependency manifest parsing
# ---------------------------------------------------------------------------
#
# Manifests declare *distribution* names; source code imports *module* names.
# ``update_external_package_versions`` joins on ``{project}:ext/{key}``, where
# ``key`` is what ``GraphClient.resolve_imports`` derives from an import
# statement (``to_name.split(".")[0]``). Every parser below therefore returns
# keys in *import* space, using the deterministic rule for its ecosystem:
#
#   pyproject.toml    distribution name lowered, ``-`` → ``_`` (pre-existing)
#   package.json      verbatim — an npm name *is* the specifier root, scope
#                     included (``@scope/pkg``)
#   Cargo.toml        table key with ``-`` → ``_`` (cargo's default lib-target
#                     name; the key is already the name code writes, even for
#                     renamed ``{ package = "..." }`` dependencies)
#   Gemfile           verbatim — a gem name is the usual ``require`` path
#
# For go.mod, pom.xml, build.gradle(.kts) and composer.json the two namespaces
# CANNOT be reconciled from the manifest alone: the import root is a hosting
# domain (``github``), a TLD segment (``com``/``org``) or a PSR-4 namespace
# that only the dependency's own metadata declares. Collapsing a coordinate
# onto such a token would stamp a version onto an aggregate node shared by
# unrelated packages, so those parsers emit the declared coordinate verbatim
# (``github.com/spf13/cobra``, ``org.slf4j:slf4j-api``, ``monolog/monolog``).
# Those keys match no ExternalPackage under today's uid scheme — the version
# write is a deliberate no-op rather than a wrong mapping.

_PEP508_RE = re.compile(r"^([A-Za-z0-9][\w.-]*)\s*(.*)")
_GRADLE_COORD_RE = re.compile(
    r"""['"]([A-Za-z0-9_.\-]+):([A-Za-z0-9_.\-]+):([A-Za-z0-9_.\-+]+)(?::[A-Za-z0-9_.\-]+)?(?:@[A-Za-z0-9]+)?['"]"""
)
_GRADLE_MAP_RE = re.compile(
    r"""group\s*:\s*['"]([^'"]+)['"]\s*,\s*name\s*:\s*['"]([^'"]+)['"]\s*,\s*version\s*:\s*['"]([^'"]+)['"]"""
)
_GEM_RE = re.compile(r"""^\s*gem\s+['"]([^'"]+)['"](.*)$""")
_GEM_CONSTRAINT_RE = re.compile(r"""\s*,\s*['"]([^'"]+)['"]""")
_POM_PROPERTY_RE = re.compile(r"\$\{([^}]+)\}")
_COMPOSER_PLATFORM_RE = re.compile(r"^(php|hhvm|composer|(ext|lib)-.+|(php|composer)-.+)$")


def _nested_table(data: Any, *keys: str) -> dict[Any, Any]:
    """Return a nested TOML/JSON table, or an empty dict if any level is missing or not a table."""
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return {}
        node = node.get(key)
    return node if isinstance(node, dict) else {}


def _parse_pyproject_deps(text: str) -> dict[str, str]:
    """PEP 621 ``[project].dependencies`` → import name → PEP 508 constraint."""
    deps = _nested_table(tomllib.loads(text), "project").get("dependencies", [])
    versions: dict[str, str] = {}
    for dep in deps:
        if not isinstance(dep, str):
            continue
        match = _PEP508_RE.match(dep.strip())
        if match:
            pkg_name = match.group(1).lower().replace("-", "_")
            constraint = match.group(2).strip().rstrip(";").strip()
            if constraint:
                versions[pkg_name] = constraint
    return versions


def _parse_package_json_deps(text: str) -> dict[str, str]:
    """npm/pnpm/yarn ``package.json`` → specifier root → semver range."""
    data = json.loads(text)
    versions: dict[str, str] = {}
    # Priority order: a runtime dependency's range wins over a dev/peer echo of it.
    for section in ("dependencies", "devDependencies", "optionalDependencies", "peerDependencies"):
        for name, constraint in _nested_table(data, section).items():
            if isinstance(constraint, str) and constraint.strip():
                versions.setdefault(name, constraint.strip())
    return versions


def _parse_cargo_toml_deps(text: str) -> dict[str, str]:
    """Cargo ``[dependencies]`` (plus dev/build/workspace) → crate name → version req."""
    data = tomllib.loads(text)
    versions: dict[str, str] = {}
    tables = (
        _nested_table(data, "dependencies"),
        _nested_table(data, "dev-dependencies"),
        _nested_table(data, "build-dependencies"),
        _nested_table(data, "workspace", "dependencies"),
    )
    for table in tables:
        for name, spec in table.items():
            constraint = spec if isinstance(spec, str) else spec.get("version") if isinstance(spec, dict) else None
            # Path/git/workspace-inherited deps carry no version requirement.
            if isinstance(constraint, str) and constraint.strip():
                versions.setdefault(name.replace("-", "_"), constraint.strip())
    return versions


def _parse_go_mod_deps(text: str) -> dict[str, str]:
    """``go.mod`` require directives → module path (verbatim) → version.

    Both the single-line (``require path v1.2.3``) and block forms are read;
    ``// indirect`` requirements count too. ``replace``/``exclude`` blocks are
    skipped — their contents are not dependency declarations.
    """
    versions: dict[str, str] = {}
    in_block = False
    for raw in text.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if in_block:
            if line == ")":
                in_block = False
                continue
            entry = line
        elif line == "require" or line.startswith(("require(", "require (")):
            in_block = line.endswith("(")
            continue
        elif line.startswith("require "):
            entry = line[len("require ") :].strip()
        else:
            continue
        parts = entry.split()
        if len(parts) >= 2 and parts[1].startswith("v"):
            versions.setdefault(parts[0], parts[1])
    return versions


def _pom_child_text(element: ET.Element, tag: str) -> str:
    """Return the text of a namespace-agnostic direct child, or an empty string."""
    child = element.find(f"./{{*}}{tag}")
    return (child.text or "").strip() if child is not None else ""


def _parse_pom_xml_deps(text: str) -> dict[str, str]:
    """Maven ``pom.xml`` → ``groupId:artifactId`` (verbatim) → version.

    ``${...}`` placeholders are resolved against ``<properties>``; entries whose
    version stays unresolved (inherited from a parent pom or dependency
    management we cannot see) are skipped rather than recorded as a literal.
    """
    root = ET.fromstring(text)
    properties = {prop.tag.rpartition("}")[2]: (prop.text or "").strip() for prop in root.iterfind("./{*}properties/*")}
    versions: dict[str, str] = {}
    for dep in root.iterfind(".//{*}dependency"):
        group = _pom_child_text(dep, "groupId")
        artifact = _pom_child_text(dep, "artifactId")
        version = _pom_child_text(dep, "version")
        if not (group and artifact and version):
            continue
        resolved = _POM_PROPERTY_RE.sub(lambda m: properties.get(m.group(1), m.group(0)), version)
        if "${" in resolved:
            continue
        versions.setdefault(f"{group}:{artifact}", resolved)
    return versions


def _parse_gradle_deps(text: str) -> dict[str, str]:
    """Gradle Groovy/Kotlin DSL → ``group:artifact`` (verbatim) → version.

    Both the string-coordinate form (``implementation "g:a:v"``) and the Groovy
    map form (``group: 'g', name: 'a', version: 'v'``) are read. Interpolated
    versions (``$ktorVersion``) do not match the coordinate pattern and are
    skipped — a variable name is not a version.
    """
    versions: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith(("//", "*", "/*", "#")):
            continue
        for pattern in (_GRADLE_COORD_RE, _GRADLE_MAP_RE):
            for group, artifact, version in pattern.findall(line):
                versions.setdefault(f"{group}:{artifact}", version)
    return versions


def _parse_composer_json_deps(text: str) -> dict[str, str]:
    """Composer ``require``/``require-dev`` → ``vendor/package`` (verbatim) → constraint."""
    data = json.loads(text)
    versions: dict[str, str] = {}
    for section in ("require", "require-dev"):
        for name, constraint in _nested_table(data, section).items():
            # php, hhvm, ext-*, lib-* are platform requirements, not packages.
            if _COMPOSER_PLATFORM_RE.match(name):
                continue
            if isinstance(constraint, str) and constraint.strip():
                versions.setdefault(name, constraint.strip())
    return versions


def _parse_gemfile_deps(text: str) -> dict[str, str]:
    """Bundler ``Gemfile`` → gem name → joined version requirements.

    Only the ``gem "name", "req", ...`` form is read (the Gemfile is Ruby, not
    a declarative format). Options such as ``require:``/``git:``/``group:``
    terminate the requirement list, and gems declared without a requirement
    yield no entry.
    """
    versions: dict[str, str] = {}
    for raw in text.splitlines():
        match = _GEM_RE.match(raw.split("#", 1)[0])
        if match is None:
            continue
        rest = match.group(2)
        constraints: list[str] = []
        while (constraint := _GEM_CONSTRAINT_RE.match(rest)) is not None:
            constraints.append(constraint.group(1))
            rest = rest[constraint.end() :]
        if constraints:
            versions.setdefault(match.group(1), ", ".join(constraints))
    return versions


# The dispatch table: adding an ecosystem is an entry here plus its parser.
_MANIFEST_PARSERS: dict[str, ManifestParser] = {
    "pyproject.toml": _parse_pyproject_deps,
    "package.json": _parse_package_json_deps,
    "Cargo.toml": _parse_cargo_toml_deps,
    "go.mod": _parse_go_mod_deps,
    "pom.xml": _parse_pom_xml_deps,
    "build.gradle": _parse_gradle_deps,
    "build.gradle.kts": _parse_gradle_deps,
    "composer.json": _parse_composer_json_deps,
    "Gemfile": _parse_gemfile_deps,
}


def register_manifest_parser(filename: str, parser: ManifestParser) -> None:
    """Register a dependency-manifest parser under its exact filename.

    Extension point for ecosystems outside the built-in table; the parser takes
    the manifest text and returns import-space name → version constraint.
    """
    _MANIFEST_PARSERS[filename] = parser
    logger.debug("Registered manifest parser: {}", filename)


def _parse_dependency_versions(project_root: Path) -> dict[str, str]:
    """Extract package name → version constraint from every known manifest in *project_root*.

    Unknown filenames are simply never probed. A manifest that fails to parse is
    skipped, not fatal. A key claimed with different constraints by two
    manifests (a polyglot root declaring the same name in two ecosystems) is
    dropped — there is one language-blind ExternalPackage node per name, so
    picking a winner would be a coin flip.
    """
    versions: dict[str, str] = {}
    conflicting: set[str] = set()
    for filename, parser in _MANIFEST_PARSERS.items():
        manifest = project_root / filename
        if not manifest.is_file():
            continue
        try:
            parsed = parser(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Skipping unparsable manifest {}: {}", manifest, exc)
            continue
        for name, constraint in parsed.items():
            if versions.get(name, constraint) != constraint:
                conflicting.add(name)
            versions[name] = constraint
    for name in conflicting:
        del versions[name]
    if conflicting:
        logger.debug("Dropped {} dependency name(s) declared inconsistently across manifests", len(conflicting))
    return versions


# ---------------------------------------------------------------------------
# Main indexing orchestration
# ---------------------------------------------------------------------------


async def _create_package_hierarchy(
    graph: GraphClient, project_name: str, project_root: Path, *, exclude_dirs: list[str] | None = None
) -> int:
    """Create Project + Package nodes and CONTAINS edges. Returns package count."""
    # root_path enables absolute-path anchor resolution for the knowledge vault
    # (a note's anchors: entry can point at a file by absolute path).
    await graph.merge_project_node(project_name, root_path=str(project_root.resolve()).replace("\\", "/"))
    packages = _detect_packages(project_root, exclude_dirs=exclude_dirs)
    batch = [(qn, qn.rsplit(".", 1)[-1], f"{rel_path}/__init__.py") for qn, rel_path in packages]
    await graph.merge_package_batch(project_name, batch)
    return len(packages)


# How long a stopped consumer gets to finish before it is cancelled.
#
# The AST consumer's ``run()`` ends with ``_flush_deferred_resolution(final=True)``
# — the deferred CALLS/IMPORTS/USES_TYPE resolution, the withheld file-hash
# writes, and the end-of-run citation retry sweep that makes ADR references
# resolve at all on a cold index. This replaced a flat ``sleep(0.5)`` before an
# unconditional ``cancel()``, which was short enough that cancellation
# routinely landed *inside* that flush; ``contextlib.suppress(CancelledError)``
# then swallowed it, so the sweep silently never completed.
#
# How long a stopped consumer may go WITHOUT completing a step before teardown
# gives up on it. Not a budget for the whole final flush: that flush is unbounded
# work — since ADR-0026 it replays every rel a project-wide strategy resolved —
# so any fixed ceiling is wrong at some project size, and being wrong means the
# whole-project sweeps at the end of it are cancelled and silently skipped.
#
# That has now happened twice, both times invisibly. At 60s the cancel landed
# inside the flush, ``suppress(CancelledError)`` swallowed it, and the
# protocol-conformance sweep never ran: IMPLEMENTS 258 -> 11 and find_dead_code
# 15 -> 120, with nothing in the output saying so but one warning buried in the
# progress display. Raising the number only moves the cliff, so teardown now
# measures progress instead — a consumer that completed a resolver step or a
# batch inside this window is working, not wedged, and is left alone.
_CONSUMER_STALL_S = 120.0
# Absolute backstop so a consumer that heartbeats forever cannot wedge the CLI.
_CONSUMER_TEARDOWN_CAP_S = 3600.0
_TEARDOWN_POLL_S = 2.0

# Drain polling: start fast, back off while nothing is moving, snap back to fast the
# moment work appears. Named rather than left as locals so tests can shrink them -- as
# locals they were unpatchable, and three drain tests each paid ~1s of real sleeping.
_DRAIN_POLL_S = 0.5
_DRAIN_POLL_MAX_S = 2.0
# How long `lag == 0` must hold before a drain is believed. Costs more than it reads:
# the poll interval grows 1.5x per idle poll, so with the defaults the sleeps after the
# first idle poll are 0.75, 1.125 and 1.6875 and the `>= settle_s` check first passes at
# t=3.56s, not 2.0s. Named so the integration suite can shrink it -- it was five inline
# literals, and every real index in that suite paid the overshoot.
_DRAIN_SETTLE_S = 2.0


async def _stop_consumer_tasks(
    tasks: Sequence[asyncio.Task[None] | None],
    consumers: Sequence[_HasProgress | None] = (),
) -> bool:
    """Let already-stopped consumers finish their final flush, then cancel stragglers.

    Callers must have invoked ``stop()`` on every consumer first — this only
    waits. Exceptions propagate exactly as they did before (only
    ``CancelledError`` is suppressed).

    Returns True when every task finished on its own. False means at least one was
    cancelled mid-flush, so its end-of-run whole-project sweeps did not run and the
    graph is missing edges the next full index would restore — the caller is
    expected to say so out loud rather than let it pass as success.

    *consumers* is optional only so existing callers keep working; without it there
    is no heartbeat to read and the stall window becomes a flat timeout.
    """
    live = [t for t in tasks if t is not None]
    if not live:
        return True
    watched = [c for c in consumers if c is not None]

    loop = asyncio.get_event_loop()
    started = loop.time()
    last_progress = started
    while True:
        _done, still_running = await asyncio.wait(live, timeout=_TEARDOWN_POLL_S)
        if not still_running:
            return True
        # A heartbeat from ANY watched consumer counts: they are torn down together
        # and one still writing means the graph is still being completed.
        newest = max((c.progress_at for c in watched), default=0.0)
        last_progress = max(last_progress, newest)
        now = loop.time()
        stalled = now - last_progress
        if stalled < _CONSUMER_STALL_S and (now - started) < _CONSUMER_TEARDOWN_CAP_S:
            continue

        logger.warning(
            "{} consumer task(s) made no progress for {:.0f}s after stop() — cancelling. "
            "The final resolution flush is INCOMPLETE: the end-of-run protocol-conformance "
            "and citation sweeps did not run, so the graph is missing edges until the next "
            "full index.",
            len(still_running),
            stalled,
        )
        for task in still_running:
            task.cancel()
        for task in live:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return False


async def _run_pipeline(
    bus: EventBus,
    graph: GraphClient,
    settings: AtlasSettings,
    embed: EmbedClient | None,
    drain_timeout_s: float,
    *,
    project_root: Path | None = None,
    project_filter: set[str] | None = None,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
    reindex_mode: bool = False,
) -> tuple[ASTConsumer, bool]:
    """Start inline consumers and wait for the pipeline to drain.

    Returns the AST consumer (so callers can read accumulated stats) and
    whether the pipeline fully drained before the timeout.
    When *reindex_mode* is True, reindex-tuned policies are used for
    faster polling.
    """
    await bus.ensure_group(Topic.FILE_CHANGED, "ast")

    # Reindex-tuned policies: flush immediately, short blocking reads
    ast_policy = BatchPolicy(time_window_s=0, max_batch_size=30, block_ms=50) if reindex_mode else None
    embed_policy = (
        BatchPolicy(time_window_s=1.0, max_batch_size=embed.batch_size, block_ms=50)
        if reindex_mode and embed is not None
        else None
    )

    ast_consumer = ASTConsumer(
        bus, graph, settings, project_root=project_root, project_filter=project_filter, policy=ast_policy
    )
    ast_task = asyncio.create_task(ast_consumer.run())

    embed_consumer: EmbedConsumer | None = None
    embed_task: asyncio.Task[None] | None = None
    if embed is not None:
        await bus.ensure_group(Topic.EMBED_DIRTY, "embed")
        embed_consumer = EmbedConsumer(
            bus,
            graph,
            embed,
            project_filter=project_filter,
            policy=embed_policy,
        )
        embed_task = asyncio.create_task(embed_consumer.run())

    try:
        drained = await _wait_for_drain(
            bus,
            drain_timeout_s,
            embed_enabled=embed is not None,
            on_drain_progress=on_drain_progress,
            settle_s=_DRAIN_SETTLE_S,
        )
        # Only worth reconciling once the run's own work has settled, and only while
        # these consumers are still alive to act on what it finds.
        if (
            drained
            and embed is not None
            and project_filter
            and await _reconcile_missing_embeddings(graph, bus, project_filter)
        ):
            drained = await _wait_for_drain(
                bus,
                drain_timeout_s,
                embed_enabled=True,
                on_drain_progress=on_drain_progress,
                settle_s=_DRAIN_SETTLE_S,
            )
    finally:
        ast_consumer.stop()
        if embed_consumer is not None:
            embed_consumer.stop()
        finished = await _stop_consumer_tasks([ast_task, embed_task], [ast_consumer, embed_consumer])
        if not finished:
            drained = False

    return ast_consumer, drained


@dataclass
class _DeltaDecision:
    """Result of the delta vs. full mode decision."""

    mode: str  # "full" | "delta"
    files_added: set[str]
    files_modified: set[str]
    files_deleted: set[str]


def _git_worktree_list(base_root: Path) -> list[Path] | None:
    """Return the list of worktree paths git knows about for the repo at *base_root*, or None if unavailable.

    Used to corroborate that a vanished ``base@branch`` project's checkout was
    actually removed (``git worktree remove`` drops it from this list) rather
    than merely being transiently unavailable — a directory that's still
    unmounted or otherwise inaccessible but was never properly removed still
    shows up here, often marked "prunable". ``None`` means "no independent
    signal" — callers must NOT treat that as confirmation of anything, same
    contract as _git_tracked_files.
    """
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=base_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
    except OSError, subprocess.TimeoutExpired:
        return None
    return [
        Path(line[len("worktree ") :]).resolve() for line in result.stdout.splitlines() if line.startswith("worktree ")
    ]


async def _decide_empty_scan_deletion(graph: GraphClient, project_name: str, project_root: Path) -> _DeltaDecision:
    """Decide the delta outcome when a scan finds zero indexable files.

    Nothing left to scan could mean files were deleted/moved, a scope
    misconfiguration, or project_root being transiently unavailable —
    os.walk silently yields nothing rather than raising. Only trust this as
    a genuine deletion when git independently corroborates it: git's index
    (unaffected by atlas scope/.atlasignore) must also report zero tracked
    files. This is deliberately NOT gated by delta_threshold — an empty scan
    would compute ratio=1.0 regardless of threshold — but it must not be
    trusted blind, since destructively wiping the entire project's graph
    data is irreversible.
    """
    old_file_paths = await graph.get_project_file_paths(project_name)
    if not old_file_paths:
        return _DeltaDecision("full", set(), set(), set())

    tracked = await asyncio.to_thread(_git_tracked_files, project_root)
    if tracked is not None and not tracked:
        return _DeltaDecision("delta", set(), set(), old_file_paths)

    logger.warning(
        "Scan of '{}' found zero indexable files but the graph has {} indexed for "
        "'{}' — skipping deletion reconciliation this run (not corroborated by git; "
        "possible transient scan failure or scope misconfiguration)",
        project_root,
        len(old_file_paths),
        project_name,
    )
    return _DeltaDecision("full", set(), set(), set())


async def _decide_delta_mode(
    settings: AtlasSettings,
    graph: GraphClient,
    project_name: str,
    project_root: Path,
    current_file_set: set[str],
) -> _DeltaDecision:
    """Determine whether to use delta or full mode based on git diff and threshold."""
    if not current_file_set:
        return await _decide_empty_scan_deletion(graph, project_name, project_root)

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


def _sort_files_for_indexing(files: list[str]) -> list[str]:
    """Sort files so deep modules come before shallow re-exporters.

    Ordering:
    1. Sort by depth descending (deeper files first).
    2. Within same depth: ``__init__.py`` files come LAST (they import from siblings).
    3. Stable sort preserves alphabetical order within same priority.
    """

    def _sort_key(path: str) -> tuple[int, int]:
        depth = path.count("/")
        is_init = 1 if path.endswith("__init__.py") else 0
        return (-depth, is_init)

    return sorted(files, key=_sort_key)


def _build_project_dep_graph(
    sub_projects: list[DetectedProject],
) -> dict[str, set[str]]:
    """Parse ``pyproject.toml`` ``[tool.uv.sources]`` for workspace deps.

    Returns ``{project_name: set of project_names it depends on}``.
    Handles patterns like::

        [tool.uv.sources]
        trading-core = { workspace = true }
    """
    # Build a map from normalised package name → DetectedProject.name
    pkg_to_project: dict[str, str] = {}
    for sp in sub_projects:
        # Try reading project name from pyproject.toml
        pyproject = sp.root / "pyproject.toml"
        if pyproject.is_file():
            try:
                data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
                pkg_name = data.get("project", {}).get("name", "")
                if pkg_name:
                    pkg_to_project[pkg_name.lower().replace("-", "_")] = sp.name
            except Exception:
                pass
        # Also map the directory basename
        pkg_to_project[sp.name.lower().replace("-", "_")] = sp.name

    dep_graph: dict[str, set[str]] = {sp.name: set() for sp in sub_projects}

    for sp in sub_projects:
        pyproject = sp.root / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except Exception:
            continue

        uv_sources = data.get("tool", {}).get("uv", {}).get("sources", {})
        for pkg_name, source_spec in uv_sources.items():
            if isinstance(source_spec, dict) and source_spec.get("workspace"):
                normalised = pkg_name.lower().replace("-", "_")
                dep_name = pkg_to_project.get(normalised)
                if dep_name and dep_name != sp.name:
                    dep_graph[sp.name].add(dep_name)

    return dep_graph


def _topo_sort_projects(
    sub_projects: list[DetectedProject],
    dep_graph: dict[str, set[str]],
) -> list[DetectedProject]:
    """Topological sort: dependencies first, then dependents.

    Falls back to original order on cycles.
    """
    by_name = {sp.name: sp for sp in sub_projects}
    result: list[str] = []
    visited: set[str] = set()
    in_stack: set[str] = set()
    has_cycle = False

    def _visit(name: str) -> None:
        nonlocal has_cycle
        if name in in_stack:
            has_cycle = True
            return
        if name in visited:
            return
        in_stack.add(name)
        for dep in dep_graph.get(name, ()):
            if dep in by_name:
                _visit(dep)
        in_stack.discard(name)
        visited.add(name)
        result.append(name)

    for sp in sub_projects:
        _visit(sp.name)

    if has_cycle:
        logger.warning("Cycle detected in project dependency graph — using original order")
        return sub_projects

    return [by_name[name] for name in result if name in by_name]


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
    6. Run inline AST + Embed consumers until the pipeline drains
    7. Update Project metadata (counts, git hash, delta stats)

    In monorepo mode, *project_name* and *project_root* override the
    settings-derived defaults so that each sub-project can be indexed
    with its own root while sharing infra config from the monorepo settings.

    On drain timeout the run returns ``drained=False`` and git_hash is not
    advanced, so the next run retries the delta.
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
    files, skipped_no_grammar = scan_files_reporting_gaps(project_root, settings, scope_paths)
    logger.debug("Scanned {} indexable files", len(files))
    # Deliberately no early return on `not files`: an empty scan (all source
    # files deleted/moved, or a scope misconfiguration) must still flow through
    # the delta decision below so stale entities are reconciled and Project
    # metadata is updated — matching the monorepo path (publish_project_changes),
    # which has no such early return.

    # 2. Embedding setup + model lock check (skipped in lightweight mode)
    embed: EmbedClient | None = None
    if settings.embeddings.enabled:
        embed = EmbedClient(settings.embeddings, settings.redis)

    if full_reindex:
        logger.debug("Full reindex: deleting existing data for '{}'", project_name)
        await bus.flush()
        await graph.delete_project_data(project_name)

    if settings.embeddings.enabled and embed is not None:
        dimension = await _resolve_dimension(embed, settings.embeddings.dimension)
        await _check_model_lock(
            graph,
            settings.embeddings.model,
            dimension,
            project=project_name,
            reindex=full_reindex,
        )

    # 3. Decide full vs. delta mode
    if full_reindex:
        decision = _DeltaDecision("full", set(), set(), set())
    else:
        decision = await _decide_delta_mode(settings, graph, project_name, project_root, set(files))

    # 4. Create Project + Package hierarchy
    pkg_count = await _create_package_hierarchy(graph, project_name, project_root)
    logger.debug("Created {} package node(s)", pkg_count)

    # 5. Sort files for optimal resolution order (deep modules before __init__.py)
    files = _sort_files_for_indexing(files)

    # 6. Publish events
    published = await _publish_events(
        bus, decision.mode, files, decision, project_name=project_name, project_root=str(project_root)
    )

    # 7. Start inline consumers and wait for drain
    reindex_mode = full_reindex or decision.mode == "full"
    ast_stats = None
    drained = True
    if published > 0:
        ast_consumer, drained = await _run_pipeline(
            bus,
            graph,
            settings,
            embed,
            drain_timeout_s,
            project_root=project_root,
            project_filter={project_name},
            on_drain_progress=on_drain_progress,
            reindex_mode=reindex_mode,
        )
        ast_stats = ast_consumer.stats

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
    # Only advance git_hash when the pipeline drained: an un-advanced git_hash
    # makes the next delta run republish the missed files and drain the
    # leftover backlog (durability contract #5).
    if git_hash and drained:
        metadata["git_hash"] = git_hash
    if decision.mode == "delta":
        metadata["delta_files_added"] = len(decision.files_added)
        metadata["delta_files_modified"] = len(decision.files_modified)
        metadata["delta_files_deleted"] = len(decision.files_deleted)
    await graph.update_project_metadata(project_name, **metadata)

    # 9. Record the architecture snapshot (ATL-121)
    await _record_architecture_snapshot(graph, project_name, git_hash or "", skipped_no_grammar)

    duration = time.monotonic() - start
    delta_stats = _build_delta_stats(decision, ast_stats) if decision.mode == "delta" else None

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
        drained=drained,
        skipped_no_grammar=skipped_no_grammar,
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


async def publish_project_changes(
    settings: AtlasSettings,
    graph: GraphClient,
    bus: EventBus,
    project_name: str,
    project_root: Path,
    files: list[str],
    *,
    full_reindex: bool = False,
    exclude_package_dirs: list[str] | None = None,
) -> _ProjectPublishResult:
    """Scan, decide delta/full, create packages, and publish events for one project.

    *exclude_package_dirs* (relative POSIX paths) are pruned from the package
    hierarchy walk — used by the monorepo root project so its Package nodes
    never reach into sub-project directories.

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
    pkg_count = await _create_package_hierarchy(graph, project_name, project_root, exclude_dirs=exclude_package_dirs)
    logger.debug("'{}': {} package node(s)", project_name, pkg_count)

    # Sort files for optimal resolution order
    files = _sort_files_for_indexing(files)

    # Publish events
    published = await _publish_events(
        bus,
        decision.mode,
        files,
        decision,
        project_name=project_name,
        project_root=str(project_root),
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

    # Full set of sub-project paths, captured BEFORE scope filtering — files
    # under an excluded sub-project still belong to that sub-project, not the
    # root, and must stay excluded from both root_only_files and the root
    # project's package hierarchy.
    all_sub_paths = [sp.path for sp in sub_projects]

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

    # Filter by settings.scope.paths (repo-root-relative) if configured -- a
    # sub-project with zero overlap is outside the configured scope and must not
    # be indexed at all. One that does overlap is scanned with a translated,
    # sub-relative prefix list (see _sub_project_scope_paths at its call site below).
    if settings.scope.paths:
        in_scope = [sp for sp in sub_projects if _sub_project_in_scope(settings.scope.paths, sp.path)]
        if len(in_scope) != len(sub_projects):
            dropped = sorted({sp.name for sp in sub_projects} - {sp.name for sp in in_scope})
            logger.info("Outside scope.paths, skipping sub-project(s): {}", ", ".join(dropped))
        sub_projects = in_scope

    # Topological sort: process dependency packages before dependents
    dep_graph = _build_project_dep_graph(sub_projects)
    sub_projects = _topo_sort_projects(sub_projects, dep_graph)

    # Compute the root name once — sub-projects are prefixed with it
    root_name = derive_project_name(project_root)

    # Pre-scan root files to determine if there's a root project (needed for total count)
    root_scope = FileScope(project_root, settings)
    root_files = root_scope.scan()
    root_only_files = [f for f in root_files if not any(f == sp or f.startswith(sp + "/") for sp in all_sub_paths)]
    has_root = bool(root_only_files)
    total = len(sub_projects) + (1 if has_root else 0)

    # --- Shared embedding resources (created once) ---
    embed: EmbedClient | None = None
    if settings.embeddings.enabled:
        embed = EmbedClient(settings.embeddings, settings.redis)
        dimension = await _resolve_dimension(embed, settings.embeddings.dimension)
        await _check_model_lock(
            graph,
            settings.embeddings.model,
            dimension,
            project=root_name,
            reindex=full_reindex,
        )

    if full_reindex:
        await bus.flush()

    # --- Start shared consumers (once for entire monorepo) ---
    reindex_mode = full_reindex

    await bus.ensure_group(Topic.FILE_CHANGED, "ast")

    ast_policy = BatchPolicy(time_window_s=0, max_batch_size=30, block_ms=50) if reindex_mode else None
    embed_policy = (
        BatchPolicy(time_window_s=1.0, max_batch_size=embed.batch_size, block_ms=50)
        if reindex_mode and embed is not None
        else None
    )

    ast_consumer = ASTConsumer(bus, graph, settings, project_root=project_root, policy=ast_policy)

    consumer_tasks: list[asyncio.Task[None]] = []
    consumer_tasks.append(asyncio.create_task(ast_consumer.run()))

    embed_consumer: EmbedConsumer | None = None
    if embed is not None:
        await bus.ensure_group(Topic.EMBED_DIRTY, "embed")
        embed_consumer = EmbedConsumer(bus, graph, embed, policy=embed_policy)
        consumer_tasks.append(asyncio.create_task(embed_consumer.run()))

    start = time.monotonic()
    publish_results: list[_ProjectPublishResult] = []
    drained = False

    try:
        # --- Publish phase: scan + publish per sub-project (fast) ---
        for i, sub in enumerate(sub_projects, 1):
            prefixed_name = f"{root_name}/{sub.name}"
            logger.debug("Publishing sub-project '{}' at {}", prefixed_name, sub.path)
            if on_progress is not None:
                on_progress(prefixed_name, i - 1, total)

            sub_scope_paths = _sub_project_scope_paths(settings.scope.paths, sub.path) if settings.scope.paths else None
            sub_files = scan_files(sub.root, settings, sub_scope_paths)
            pr = await publish_project_changes(
                settings,
                graph,
                bus,
                prefixed_name,
                sub.root,
                sub_files,
                full_reindex=full_reindex,
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

            root_pr = await publish_project_changes(
                settings,
                graph,
                bus,
                root_name,
                project_root,
                root_only_files,
                full_reindex=full_reindex,
                exclude_package_dirs=all_sub_paths,
            )
            publish_results.append(root_pr)

            if on_progress is not None:
                on_progress(root_name, total, total)

        # --- Wait for ALL stages to drain (once) ---
        drained = await _wait_for_drain(
            bus,
            drain_timeout_s,
            embed_enabled=embed is not None,
            on_drain_progress=on_drain_progress,
            settle_s=_DRAIN_SETTLE_S,
        )
        if drained and embed is not None:
            names = [pr.project_name for pr in publish_results]
            if await _reconcile_missing_embeddings(graph, bus, names):
                drained = await _wait_for_drain(
                    bus,
                    drain_timeout_s,
                    embed_enabled=True,
                    on_drain_progress=on_drain_progress,
                    settle_s=_DRAIN_SETTLE_S,
                )

    finally:
        # --- Tear down consumers (once) ---
        ast_consumer.stop()
        if embed_consumer is not None:
            embed_consumer.stop()
        await _stop_consumer_tasks(consumer_tasks, [ast_consumer, embed_consumer])

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
        # Only advance git_hash when the shared pipeline drained (coarse per-run
        # gate): an un-advanced git_hash makes the next delta run republish the
        # missed files and drain the leftover backlog (durability contract #5).
        if git_hash and drained:
            metadata["git_hash"] = git_hash
        if pr.mode == "delta":
            metadata["delta_files_added"] = len(pr.decision.files_added)
            metadata["delta_files_modified"] = len(pr.decision.files_modified)
            metadata["delta_files_deleted"] = len(pr.decision.files_deleted)
        await graph.update_project_metadata(pr.project_name, **metadata)

        # Use shared start time — in monorepo mode all projects share one pipeline,
        # so per-project publish timestamps don't reflect actual processing duration.
        duration = time.monotonic() - start
        delta_stats = _build_delta_stats(pr.decision, ast_consumer.stats) if pr.mode == "delta" else None
        results.append(
            IndexResult(
                files_scanned=pr.files_scanned,
                files_published=pr.files_published,
                entities_total=entity_count,
                duration_s=duration,
                mode=pr.mode,
                delta_stats=delta_stats,
                drained=drained,
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


def _build_delta_stats(decision: _DeltaDecision, ast_stats: Any) -> DeltaStats:
    """Build DeltaStats from the decision and AST consumer stats."""
    return DeltaStats(
        files_added=len(decision.files_added),
        files_modified=len(decision.files_modified),
        files_deleted=len(decision.files_deleted),
        entities_added=ast_stats.entities_added if ast_stats else 0,
        entities_modified=ast_stats.entities_modified if ast_stats else 0,
        entities_deleted=ast_stats.entities_deleted if ast_stats else 0,
        entities_unchanged=ast_stats.entities_unchanged if ast_stats else 0,
    )


async def _reconcile_missing_embeddings(graph: GraphClient, bus: EventBus, project_names: Iterable[str]) -> int:
    """Republish embed work for entities that hold no vector, and return how many.

    The AST stage already refuses to re-embed an entity that has one
    (``has_embedding``, consumers.py) — but that check is downstream of two
    content-based skips, so it is unreachable for the entities that need it most.
    A file whose hash is unchanged is never parsed, and in delta mode (which is
    every daemon-driven index) it is never even published. So an ``EmbedDirty``
    lost to a poison-park or an abandoned PEL is lost for good: measured, 144
    entities across 4 files stayed unembedded through a subsequent *full* re-index.

    Fixing it at the gate cannot work, because in delta mode there is no gate —
    the file is not in the batch at all. Reconciling desired state against actual
    state is what makes the pipeline self-healing rather than merely retryable,
    and it catches every cause rather than the one that happened to be found.
    """
    refs: list[Event] = []
    for project_name in project_names:
        for uid, label, file_path in await graph.find_unembedded_entities(project_name):
            # EntityRef.qualified_name carries the *uid* — the embed consumer feeds it
            # straight to read_entity_texts(uids=...), so a real qualified name silently
            # matches nothing and the batch completes having done no work.
            refs.append(
                EmbedDirty(
                    entity=EntityRef(qualified_name=uid, node_type=label, file_path=file_path),
                    significance=Significance.HIGH,
                )
            )
    if refs:
        logger.warning(
            "Re-queued {} entity(ies) that had no embedding — earlier embed work was lost, "
            "not skipped; re-embedding now",
            len(refs),
        )
        await bus.publish_many(Topic.EMBED_DIRTY, refs)
    return len(refs)


async def _wait_for_drain(
    bus: EventBus,
    timeout_s: float,
    *,
    embed_enabled: bool = True,
    on_drain_progress: Callable[[int, int, int], None] | None = None,
    settle_s: float = _DRAIN_SETTLE_S,
) -> bool:
    """Poll stream groups until AST and (optionally) Embed consumers are drained.

    Returns ``True`` when every queried group has ``pending == 0`` and
    ``lag == 0`` sustained for *settle_s*, ``False`` on timeout.  A ``lag``
    of ``None`` means unknown (the stream was trimmed past the group's read
    position) and is treated as NOT drained.

    If *on_drain_progress* is provided, it is called each poll cycle with
    ``(t1_remaining, t2_remaining, t3_remaining)`` so callers can display
    pipeline progress to the user.  ``t1_remaining`` is always 0 (kept for
    callback signature compatibility).
    """
    deadline = time.monotonic() + timeout_s
    settled_since: float | None = None
    poll_interval = _DRAIN_POLL_S
    t2_remaining: int | None = -1
    t3_remaining: int | None = -1
    infos = []

    while time.monotonic() < deadline:
        queries: list[tuple[Topic, str]] = [(Topic.FILE_CHANGED, "ast")]
        if embed_enabled:
            queries.append((Topic.EMBED_DIRTY, "embed"))

        infos = await bus.stream_group_info_multi(queries)

        # Build topic→remaining maps so we don't need fragile index tracking.
        # lag=None → remaining unknown (not drained); display pending only.
        remaining: dict[Topic, int | None] = {}
        display: dict[Topic, int] = {}
        for (topic, _), info in zip(queries, infos, strict=True):
            pending, lag = info["pending"], info["lag"]
            remaining[topic] = None if lag is None else pending + lag
            display[topic] = pending if lag is None else pending + lag
        t2_remaining = remaining.get(Topic.FILE_CHANGED, 0)
        t3_remaining = remaining.get(Topic.EMBED_DIRTY, 0)

        if on_drain_progress is not None:
            on_drain_progress(0, display.get(Topic.FILE_CHANGED, 0), display.get(Topic.EMBED_DIRTY, 0))

        if t2_remaining == 0 and t3_remaining == 0:
            if settled_since is None:
                settled_since = time.monotonic()
            elif time.monotonic() - settled_since >= settle_s:
                logger.debug("Pipeline drained after {:.1f}s settling", time.monotonic() - settled_since)
                return True
            # Adaptive backoff: poll less frequently once idle
            poll_interval = min(_DRAIN_POLL_MAX_S, poll_interval * 1.5)
        else:
            settled_since = None
            poll_interval = _DRAIN_POLL_S  # reset to fast polling when work is happening

        await asyncio.sleep(poll_interval)

    logger.error(
        "Pipeline drain timed out after {:.0f}s — t2={} t3={}; "
        "index metadata will NOT advance; re-run 'atlas index' to retry the missed files",
        timeout_s,
        t2_remaining,
        t3_remaining,
    )
    return False


# ---------------------------------------------------------------------------
# Worktree-project GC (§3.9)
# ---------------------------------------------------------------------------


async def gc_vanished_worktree_projects(graph: GraphClient) -> list[str]:
    """Delete graph data for worktree projects whose checkout no longer exists on disk.

    A linked git worktree indexes as its own ``base@branch`` Project
    (``derive_project_name``); once the worktree is removed
    (``git worktree remove`` or a manual ``rm``), its ``root_path`` — stored on
    the Project node by ``_create_package_hierarchy`` — points at a directory
    that's gone. This runs at the startup of *every* daemon for *any* project
    sharing the Memgraph instance, so a directory that merely looks gone
    (transient mount hiccup, unmounted drive) must not be trusted blind —
    destructively wiping a project's graph data is irreversible. A vanished
    root_path is therefore only a candidate; it must be corroborated against
    the base project's own ``git worktree list`` (via ``_git_worktree_list``)
    before deletion proceeds — a real ``git worktree remove`` drops the entry
    from that list, while a merely-unavailable checkout still shows up there.
    If the base project can't be found in the graph, its root_path also isn't
    a live directory, or git is unavailable, the candidate is skipped this
    run rather than guessed at — same "skip, don't guess" behavior as
    ``_decide_empty_scan_deletion``. Meant to run once at daemon startup, not
    on a timer — a removed worktree is a one-time event, not something that
    needs continuous polling.
    """
    removed: list[str] = []
    records = await graph.get_project_status()

    roots_by_name: dict[str, str] = {}
    for record in records:
        node = record.get("n")
        if node is None:
            continue
        name = node.get("project_name")
        root_path = node.get("root_path")
        if name and root_path:
            roots_by_name[name] = root_path

    for record in records:
        node = record.get("n")
        if node is None:
            continue
        name = node.get("project_name")
        root_path = node.get("root_path")
        if not name or "@" not in name or not root_path:
            continue
        if Path(root_path).is_dir():
            continue

        base = name.rpartition("@")[0]
        base_root = roots_by_name.get(base)
        if not base_root or not Path(base_root).is_dir():
            logger.warning(
                "GC: worktree project '{}' checkout vanished ({}) but base project '{}' has no live root_path "
                "to corroborate against — skipping deletion this run",
                name,
                root_path,
                base,
            )
            continue

        worktrees = await asyncio.to_thread(_git_worktree_list, Path(base_root))
        if worktrees is None:
            logger.warning(
                "GC: worktree project '{}' checkout vanished ({}) but 'git worktree list' is unavailable for "
                "base '{}' — skipping deletion this run (not corroborated)",
                name,
                root_path,
                base,
            )
            continue

        if Path(root_path).resolve() in worktrees:
            logger.warning(
                "GC: worktree project '{}' checkout looks vanished ({}) but git still lists it as a live "
                "worktree — skipping deletion this run (likely transient unavailability, not a real removal)",
                name,
                root_path,
            )
            continue

        logger.info("GC: worktree project '{}' checkout vanished ({}) — removing graph data", name, root_path)
        await graph.delete_project_data(name)
        removed.append(name)
    return removed
