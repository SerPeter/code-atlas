"""Configuration management for Code Atlas."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    PyprojectTomlConfigSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def find_git_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for a ``.git`` directory.

    Returns the containing directory or ``None`` if no ``.git`` is found.
    """
    current = (start or Path.cwd()).resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def resolve_git_dir(project_root: Path) -> Path | None:
    """Resolve the actual git directory for *project_root*.

    - If ``.git`` is a directory → return it (normal repo / main worktree).
    - If ``.git`` is a file → parse ``gitdir: <path>``, resolve relative
      paths against *project_root*, return the target directory.
    - Otherwise → ``None``.
    """
    dot_git = project_root / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        try:
            content = dot_git.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if content.startswith("gitdir:"):
            raw = content[len("gitdir:") :].strip()
            resolved = Path(raw) if Path(raw).is_absolute() else (project_root / raw).resolve()
            return resolved if resolved.is_dir() else None
    return None


def get_worktree_branch(project_root: Path) -> str | None:
    """Return the branch name if *project_root* is a linked git worktree.

    Returns ``None`` for the main worktree or non-git directories.
    """
    dot_git = project_root / ".git"
    if not dot_git.is_file():
        return None  # main worktree or non-git

    git_dir = resolve_git_dir(project_root)
    if git_dir is None:
        return None

    head_file = git_dir / "HEAD"
    try:
        head_content = head_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if head_content.startswith("ref: refs/heads/"):
        return head_content[len("ref: refs/heads/") :]

    # Detached HEAD fallback — use the worktree directory name
    # (git stores linked worktrees under `.git/worktrees/<name>`)
    return git_dir.name


def _explicit_project_name(project_root: Path) -> str | None:
    """Read an explicit ``[project] name`` override from atlas.toml, if set.

    Reads the file directly (rather than going through :class:`AtlasSettings`)
    because callers of :func:`derive_project_name` only have a bare *project_root*
    path, not a settings instance.
    """
    match = _find_atlas_toml(project_root)
    if match is None:
        return None
    if match.table_header:
        # GAP: match is a pyproject.toml [tool.atlas] fallback. Its own top-level
        # [project] table is PEP 621 package metadata (name/version/...), not an
        # atlas override — reading it here would silently pick up the wrong name.
        # A [tool.atlas] project-name override isn't read from this fallback yet;
        # left out of scope for this change.
        return None
    try:
        with match.path.open("rb") as fh:
            data = tomllib.load(fh)
    except OSError, tomllib.TOMLDecodeError:
        return None
    project_section = data.get("project")
    if not isinstance(project_section, dict):
        return None
    name = project_section.get("name")
    return name if isinstance(name, str) and name.strip() else None


def derive_project_name(project_root: Path) -> str:
    """Derive the canonical project name for *project_root*.

    - Base name = explicit ``[project] name`` override from atlas.toml if set,
      otherwise the resolved directory basename. Two unrelated repositories
      checked out under different paths but sharing a folder name (e.g. two
      "backend" checkouts) will otherwise collide in the graph and event
      streams — set ``[project] name`` in atlas.toml to disambiguate. This is
      opt-in by design: auto-hashing names would churn uids for every existing
      single-project user.
    - If *project_root* is a linked worktree → ``base@branch``.
    """
    root = project_root.resolve()
    base = _explicit_project_name(root) or root.name
    branch = get_worktree_branch(project_root)
    if branch is not None:
        return f"{base}@{branch}"
    return base


def _default_project_root() -> Path:
    """Git root if found, otherwise the current directory.

    This used to raise, and the raise was both redundant and actively harmful. The
    commands that genuinely need a repository — ``index``, ``watch``,
    ``mine-git-history`` — enforce it themselves through ``_resolve_project_root``, each
    with a ``--no-git-check`` escape hatch. Raising here only reached the commands that
    do NOT need one.

    The cost was concrete: the README's own quickstart has you `curl` a compose file into
    a fresh directory and then run ``atlas status``, which exited 1 with "Run from inside
    a git repo or pass an explicit path" — advice no user could follow, because ``status``
    accepts no path. ``atlas mcp`` failed the same way, and an MCP client launches it from
    whatever directory it likes, so a globally-registered server simply would not start.

    Falling back to the working directory is what the caller meant in every one of those
    cases: operate on wherever I am.
    """
    return find_git_root() or Path.cwd()


@dataclass(frozen=True)
class _ConfigFileMatch:
    """The config file :func:`_find_atlas_toml` discovered, and where AtlasSettings'
    table lives within it.

    ``table_header`` is empty for a standalone ``atlas.toml`` (settings live at the
    file's root, matching today's behavior) and ``("tool", "atlas")`` for a
    ``pyproject.toml`` ``[tool.atlas]`` fallback match.
    """

    path: Path
    table_header: tuple[str, ...] = ()


def _has_tool_atlas_table(pyproject_path: Path) -> bool:
    """Check whether *pyproject_path* has a ``[tool.atlas]`` table."""
    try:
        with pyproject_path.open("rb") as fh:
            data = tomllib.load(fh)
    except OSError, tomllib.TOMLDecodeError:
        return False
    tool_section = data.get("tool")
    return isinstance(tool_section, dict) and isinstance(tool_section.get("atlas"), dict)


def _find_atlas_toml(start: Path | None = None) -> _ConfigFileMatch | None:
    """Walk up from *start* (default: cwd) looking for Atlas config.

    Ruff-style dual-file discovery: at each directory level, a standalone
    ``atlas.toml`` always wins. Only when absent is ``pyproject.toml`` considered,
    and only if it has a ``[tool.atlas]`` table — a ``pyproject.toml`` without one
    is transparent to the search and the walk continues past it. The first
    directory level where either resolves wins; there is no cross-directory
    merging.
    """
    current = (start or Path.cwd()).resolve()
    while True:
        atlas_candidate = current / "atlas.toml"
        if atlas_candidate.is_file():
            return _ConfigFileMatch(path=atlas_candidate)

        pyproject_candidate = current / "pyproject.toml"
        if pyproject_candidate.is_file() and _has_tool_atlas_table(pyproject_candidate):
            return _ConfigFileMatch(path=pyproject_candidate, table_header=("tool", "atlas"))

        parent = current.parent
        if parent == current:
            return None
        current = parent


class StrictSection(BaseModel):
    """Base for every ``atlas.toml`` section: an unknown key is an error, not a no-op.

    The root settings model has always been ``extra="forbid"``, but a nested
    ``BaseModel`` defaults to ``ignore`` — so a typo *inside* a section vanished
    without a word. Measured before the fix::

        ScopeSettings(include_paths=[...], exclude_patterns=[...])
        -> {'paths': [], 'include': None, 'exclude': None}

    Both values silently dropped; the real fields are ``paths``/``include``/``exclude``.
    Someone scoping indexing to three services would have indexed the whole monorepo and
    been told nothing (ATL-111).

    Inherited rather than repeated on seventeen classes, so a section added later is
    strict by default instead of strict only if its author remembered.
    """

    model_config = ConfigDict(extra="forbid")


class ScopeSettings(StrictSection):
    """File scope and ignore settings (ruff-style include/exclude semantics)."""

    paths: list[str] = Field(
        default_factory=list,
        description="Restrict indexing to these directory paths (monorepo scoping).",
    )
    include: list[str] | None = Field(
        default=None,
        description="File patterns to index. Overrides default language-based patterns when set.",
    )
    extend_include: list[str] = Field(
        default_factory=list,
        description="Additional file patterns to index, appended to defaults.",
    )
    exclude: list[str] | None = Field(
        default=None,
        description="Patterns to exclude from indexing. Overrides defaults when set.",
    )
    extend_exclude: list[str] = Field(
        default_factory=list,
        description="Additional patterns to exclude, appended to defaults.",
    )


class LibrarySettings(StrictSection):
    """Library and dependency indexing settings."""

    full_index: list[str] = Field(default_factory=list, description="Libraries to fully parse and index.")
    stub_index: list[str] = Field(default_factory=list, description="Libraries to index at type-stub level only.")


class MonorepoSettings(StrictSection):
    """Monorepo detection and scoping settings."""

    auto_detect: bool = Field(default=True, description="Auto-detect sub-projects by project markers.")
    projects: list[dict[str, str]] = Field(
        default_factory=list,
        description='Explicit sub-project definitions: [{"path": "services/auth", "name": "auth"}].',
    )
    always_include: list[str] = Field(
        default_factory=list, description="Project names always included when scoping queries (e.g., shared libs)."
    )
    markers: list[str] = Field(
        default_factory=lambda: [
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
        ],
        description="Files that indicate a directory is a sub-project root.",
    )


# ``rpm``/``tpm`` of 0 mean unlimited. That is the last-resort default on purpose:
# litellm's registry publishes rate limits for 4 of its 134 embedding models, so a
# non-zero guess here would throttle the overwhelming majority of users based on
# nothing. Unlimited is not unbounded -- max_concurrency still caps calls in flight,
# and the AIMD backoff reacts to a real 429 whatever the configured limits are.
_PROVIDER_DEFAULTS: dict[str, dict[str, int]] = {
    "tei": {"batch_size": 32, "max_concurrency": 4, "rpm": 0, "tpm": 0},
    "ollama": {"batch_size": 32, "max_concurrency": 2, "rpm": 0, "tpm": 0},
    "litellm": {"batch_size": 128, "max_concurrency": 8, "rpm": 0, "tpm": 0},
}


class EmbeddingSettings(StrictSection):
    """Embedding settings — routes through litellm for any provider."""

    enabled: bool = Field(
        default=True, description="Enable embedding pipeline and vector search. False for lightweight mode."
    )
    provider: str = Field(default="tei", description="Embedding provider: 'tei', 'litellm', or 'ollama'.")
    model: str = Field(default="nomic-ai/nomic-embed-code", description="Embedding model name.")
    base_url: str = Field(default="http://localhost:8080", description="OpenAI-compatible embedding endpoint URL.")
    dimension: int | None = Field(default=None, description="Embedding vector dimension. Auto-detected when None.")
    batch_size: int | None = Field(default=None, description="Max texts per embedding API call. Auto from provider.")
    max_concurrency: int | None = Field(
        default=None, description="Max concurrent embedding API calls / embed consumers. Auto from provider."
    )
    rpm: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Provider requests-per-minute budget, shared across processes via Valkey. "
            "None resolves from litellm's model registry, then the provider default. 0 means unlimited."
        ),
    )
    tpm: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Provider tokens-per-minute budget, shared across processes via Valkey. "
            "None resolves from litellm's model registry, then the provider default. 0 means unlimited."
        ),
    )
    timeout_s: float = Field(default=30.0, description="Timeout in seconds for embedding API calls.")
    truncate_ratio: float = Field(
        default=0.9, gt=0, le=1, description="Fraction of max input tokens to use as truncation limit."
    )
    query_cache_size: int = Field(default=128, description="Max cached query embeddings (LRU eviction).")
    cache_ttl_days: int = Field(default=7, description="Embedding cache TTL in days. 0 disables Valkey caching.")

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> EmbeddingSettings:
        defaults = _PROVIDER_DEFAULTS.get(self.provider, _PROVIDER_DEFAULTS["tei"])
        if self.batch_size is None:
            self.batch_size = defaults["batch_size"]
        if self.max_concurrency is None:
            self.max_concurrency = defaults["max_concurrency"]
        # rpm/tpm are deliberately NOT resolved here. The registry is keyed by the
        # litellm model string, which EmbeddingClient builds (it prefixes "openai/"
        # for TEI), so only the client can look them up. None survives to there and
        # means "not configured"; the provider default below is the floor of that
        # lookup, applied by EmbeddingClient._resolve_rate_limit.
        return self


class BackendSettings(StrictSection):
    """Backend selection for the graph store and event queue.

    ``"auto"`` probes the network backend at startup (``GraphClient.ping()`` /
    ``EventBus.ping()``) and falls back to the in-process SQLite backend if
    unreachable. An explicit ``"memgraph"``/``"valkey"`` fails loudly if
    unreachable rather than silently falling back.
    """

    graph: Literal["memgraph", "sqlite", "auto"] = Field(
        default="auto", description="Graph backend: 'memgraph', 'sqlite', or 'auto'."
    )
    queue: Literal["valkey", "sqlite", "auto"] = Field(
        default="auto", description="Event queue backend: 'valkey', 'sqlite', or 'auto'."
    )
    sqlite_data_dir: Path = Field(
        default=Path(".atlas"),
        description="Directory (relative to project_root) holding graph.sqlite3 and queue.sqlite3.",
    )


class MemgraphSettings(StrictSection):
    """Memgraph connection settings."""

    host: str = Field(default="localhost", description="Memgraph host.")
    port: int = Field(default=7687, description="Memgraph Bolt port.")
    username: str = Field(default="", description="Memgraph username.")
    password: str = Field(default="", description="Memgraph password.")
    query_timeout_s: float = Field(default=10.0, description="Timeout in seconds for read queries.")
    write_timeout_s: float = Field(default=60.0, description="Timeout in seconds for write queries.")


class SearchSettings(StrictSection):
    """Search and retrieval settings."""

    default_token_budget: int = Field(default=8000, description="Default token budget for context assembly.")
    max_token_budget: int = Field(default=32000, description="Maximum allowed token budget for context assembly.")
    tokenizer: str = Field(default="cl100k_base", description="Tiktoken encoding name for token counting.")
    test_filter: bool = Field(default=True, description="Exclude test files from results by default.")
    stub_filter: bool = Field(default=True, description="Exclude .pyi type stubs from results by default.")
    generated_filter: bool = Field(default=True, description="Exclude generated code patterns from results by default.")
    test_patterns: list[str] = Field(
        default_factory=lambda: ["test_*", "*_test.py", "tests/", "__tests__/"],
        description="Glob patterns matching test file paths.",
    )
    generated_patterns: list[str] = Field(
        default_factory=lambda: ["*_pb2.py", "*_pb2_grpc.py", "*.generated.*"],
        description="Glob patterns matching generated code file paths.",
    )
    max_caller_depth: int = Field(default=1, description="Default hop depth for caller/callee expansion.")
    max_callers: int = Field(default=10, description="Max callers to return before ranking/filtering.")
    max_siblings: int = Field(default=5, description="Max sibling entities in context expansion.")
    rrf_k: int = Field(default=60, description="RRF k parameter (higher = more weight to lower-ranked results).")
    default_weights: dict[str, float] = Field(
        default_factory=lambda: {"graph": 1.0, "vector": 1.0, "bm25": 1.0},
        description="Default per-channel weights for hybrid search RRF fusion.",
    )


class DetectorSettings(StrictSection):
    """Pattern detector settings."""

    enabled: list[str] = Field(
        default_factory=lambda: [
            "test_mapping",
            "class_overrides",
            "di_injection",
            # Emits EXPORTS (Module -> the Callable/TypeDef named in __all__). Registered
            # since it was written but never enabled, so the public-API surface every
            # __all__ already declares was captured as a property and never as an edge.
            "module_exports",
        ],
        description="Enabled pattern detectors.",
    )


class RationaleSettings(StrictSection):
    """Extraction of intent-bearing comments (``# NOTE:``, ``# WHY:``, ``# HACK:``).

    Matched comments are attached to the smallest enclosing entity as the
    ``rationale`` node property, and ADR/RFC references as ``citations``.
    Marker matching is case-sensitive and uppercase-only — ``Note:`` in
    ordinary prose is not a marker.

    Defaults mirror ``parsing.ast.DEFAULT_RATIONALE_MARKERS`` /
    ``DEFAULT_TASK_MARKERS`` / ``DEFAULT_CITATION_SCHEMES``, which are what a
    caller that passes no settings object gets.
    """

    enabled: bool = Field(default=True, description="Master switch for rationale extraction.")
    markers: list[str] = Field(
        default_factory=lambda: ["NOTE", "WHY", "HACK"],
        description="Uppercase comment markers treated as rationale.",
    )
    tasks: bool = Field(
        default=False,
        description="Also extract work-tracking markers (TODO/FIXME). Off by default — high volume, short lived.",
    )
    task_markers: list[str] = Field(
        default_factory=lambda: ["TODO", "FIXME"],
        description="Work-tracking markers, extracted only when 'tasks' is true.",
    )
    citations: bool = Field(default=True, description="Record ADR/RFC style references found in comments.")
    citation_schemes: list[str] = Field(
        default_factory=lambda: ["ADR", "RFC"],
        description="Reference schemes to record, e.g. 'ADR-0014' from 'see ADR 14'.",
    )


class IndexSettings(StrictSection):
    """Indexing delta settings."""

    delta_threshold: float = Field(
        default=0.3, description="If more than this fraction of files changed, fall back to full re-index."
    )
    stale_mode: Literal["warn", "lock", "ignore"] = Field(
        default="warn",
        description="Stale index behavior: 'warn' (annotate), 'lock' (refuse), 'ignore' (skip).",
    )
    max_source_chars: int = Field(default=2000, description="Max characters for entity source text (0 to disable).")
    max_parse_bytes: int = Field(
        default=1_048_576,
        description="Skip files larger than this many bytes instead of parsing them (0 disables the ceiling). "
        "Tree-sitter error recovery is superlinear, so one committed dump can stall indexing: an unparseable "
        "T-SQL dump measured 5s at 1 MiB, 45s at 2 MiB and 4min at 4 MiB. Must stay in sync with "
        "parsing.ast.DEFAULT_MAX_PARSE_BYTES (asserted by a unit test).",
    )
    file_hash_gate: bool = Field(default=True, description="Skip files whose content hash hasn't changed.")
    strip_whitespace: bool = Field(
        default=True, description="Normalize whitespace before hashing (ignores formatting-only changes)."
    )
    drain_timeout_s: float = Field(
        default=600.0,
        description="Max seconds to wait for the AST/embed pipeline to drain after publishing. "
        "git_hash (the delta checkpoint) only advances on a full drain, so a run that times out "
        "here republishes the ENTIRE file set on the next invocation -- on top of whatever the "
        "timed-out run left undrained. A workload that structurally cannot drain within this "
        "window can never converge; raise it rather than retrying the default indefinitely.",
    )


class ObservabilitySettings(StrictSection):
    """OpenTelemetry observability settings (requires ``[otel]`` extra)."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing and metrics.")
    exporter: str = Field(default="otlp", description="Exporter type: 'otlp', 'console', or 'none'.")
    endpoint: str = Field(default="http://localhost:4317", description="OTLP collector endpoint.")
    service_name: str = Field(default="code-atlas", description="OTel service.name resource attribute.")
    sample_rate: float = Field(default=1.0, description="Trace sample rate (1.0 = all, 0.1 = 10%).")


class WatcherSettings(StrictSection):
    """File watcher debounce settings."""

    debounce_s: float = Field(default=5.0, description="Debounce timer in seconds (resets per change).")
    max_wait_s: float = Field(default=30.0, description="Max-wait ceiling in seconds (per batch).")
    cooldown_s: float = Field(default=10.0, description="Per-file cooldown after processing (seconds). 0 disables.")


class McpSettings(StrictSection):
    """MCP server settings."""

    host: str = Field(default="127.0.0.1", description="Bind address for HTTP transports (ignored for stdio).")
    port: int = Field(default=8000, description="Bind port for HTTP transports (ignored for stdio).")
    transport: str = Field(default="stdio", description="Transport protocol: 'stdio' or 'streamable-http'.")
    strict: bool = Field(default=False, description="Refuse to start if embedding model mismatch.")


class RedisSettings(StrictSection):
    """Redis/Valkey connection settings for event bus."""

    host: str = Field(default="localhost", description="Redis/Valkey host.")
    port: int = Field(default=6379, description="Redis/Valkey port.")
    db: int = Field(default=0, description="Redis database number.")
    password: str = Field(default="", description="Redis/Valkey password.")
    stream_prefix: str = Field(default="atlas", description="Prefix for Redis Stream keys.")
    stream_maxlen: int = Field(
        default=100_000,
        description="Max entries per Redis Stream (XADD maxlen, approximate). "
        "Must exceed the largest expected publish backlog, but a retained entry costs ~227 bytes: "
        "the ceiling is a memory budget, not just a backlog guard. At 100k a single stream tops out "
        "near 23MB, so one flooding project cannot exhaust the shared bus and OOM every other "
        "project's streams. 0 disables trimming.",
    )


class ProjectSettings(StrictSection):
    """Project identity overrides."""

    name: str | None = Field(
        default=None,
        description="Explicit project name override (see derive_project_name). Set this to "
        "disambiguate two checkouts that share a directory basename — otherwise they collide "
        "in the graph and event streams.",
    )


class ExtraVaultSettings(StrictSection):
    """A knowledge vault indexed as a sibling project alongside this repo's own vault.

    Used for the overspanning (cross-project) vault and the Claude Code
    harness memory directory — both are ordinary projects in the same graph,
    just rooted outside this repo.
    """

    path: str = Field(description="Filesystem path to the vault root (absolute, or ~-expanded).")
    project_name: str = Field(description="Project name this vault indexes under (see derive_project_name).")


class KnowledgeSettings(StrictSection):
    """Knowledge vault settings — the Obsidian-compatible note vault living alongside code."""

    vault_path: str = Field(
        default="wiki",
        description="Repo-relative path to this project's knowledge vault. This directory IS the "
        "vault — frontmatter-triggered note mode lets ordinary docs and vault notes coexist in the same tree.",
    )
    extra_vaults: list[ExtraVaultSettings] = Field(
        default_factory=list,
        description="Additional vaults (global overspanning vault, harness memory dir) indexed as "
        "sibling projects in the same graph. Always included in query scope alongside the current project. "
        "Each gets its own live FileWatcher instance (multi-root watching) plus a one-time startup "
        "catch-up scan — see DaemonManager.start().",
    )

    @model_validator(mode="after")
    def _validate_extra_vaults_unique(self) -> KnowledgeSettings:
        seen_names: set[str] = set()
        seen_paths: set[Path] = set()
        for vault in self.extra_vaults:
            if vault.project_name in seen_names:
                msg = f"Duplicate [knowledge] extra_vaults project_name: '{vault.project_name}'"
                raise ValueError(msg)
            seen_names.add(vault.project_name)

            resolved_path = Path(vault.path).expanduser().resolve()
            if resolved_path in seen_paths:
                msg = f"Duplicate [knowledge] extra_vaults path (resolves to {resolved_path}): '{vault.path}'"
                raise ValueError(msg)
            seen_paths.add(resolved_path)
        return self


class AtlasSettings(BaseSettings):
    """Root configuration for Code Atlas."""

    model_config = SettingsConfigDict(
        toml_file="atlas.toml",
        pyproject_toml_table_header=("tool", "atlas"),
        env_prefix="ATLAS_",
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Discover atlas.toml (or a pyproject.toml [tool.atlas] fallback) relative to
        # the target project_root (when explicitly passed, e.g. `atlas index <other-path>`),
        # not the process cwd — otherwise indexing a project other than the cwd's own
        # applies the wrong config.
        init_kwargs = getattr(init_settings, "init_kwargs", {})
        target_root = init_kwargs.get("project_root")
        config_match = _find_atlas_toml(Path(target_root) if target_root else None)
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if config_match is not None:
            if config_match.table_header:
                # pyproject.toml [tool.atlas] fallback — PyprojectTomlConfigSettingsSource
                # drills into model_config's pyproject_toml_table_header before matching
                # fields, unlike plain TomlConfigSettingsSource which has no table-header
                # concept at all (root-level keys only).
                sources.append(PyprojectTomlConfigSettingsSource(settings_cls, toml_file=config_match.path))
            else:
                sources.append(TomlConfigSettingsSource(settings_cls, toml_file=config_match.path))
        sources.append(file_secret_settings)
        return tuple(sources)

    project_root: Path = Field(default_factory=_default_project_root, description="Project root path.")
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    scope: ScopeSettings = Field(default_factory=ScopeSettings)
    libraries: LibrarySettings = Field(default_factory=LibrarySettings)
    monorepo: MonorepoSettings = Field(default_factory=MonorepoSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    memgraph: MemgraphSettings = Field(default_factory=MemgraphSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    rationale: RationaleSettings = Field(default_factory=RationaleSettings)
    watcher: WatcherSettings = Field(default_factory=WatcherSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    detectors: DetectorSettings = Field(default_factory=DetectorSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
