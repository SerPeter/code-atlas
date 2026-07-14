"""Configuration management for Code Atlas."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

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
    toml_path = _find_atlas_toml(project_root)
    if toml_path is None:
        return None
    try:
        with toml_path.open("rb") as fh:
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
    """Git root if found, otherwise raise."""
    root = find_git_root()
    if root is None:
        msg = f"No git repository found at or above {Path.cwd()}. Run from inside a git repo or pass an explicit path."
        raise RuntimeError(msg)
    return root


def _find_atlas_toml(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for ``atlas.toml``."""
    current = (start or Path.cwd()).resolve()
    while True:
        candidate = current / "atlas.toml"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


class ScopeSettings(BaseModel):
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


class LibrarySettings(BaseModel):
    """Library and dependency indexing settings."""

    full_index: list[str] = Field(default_factory=list, description="Libraries to fully parse and index.")
    stub_index: list[str] = Field(default_factory=list, description="Libraries to index at type-stub level only.")


class MonorepoSettings(BaseModel):
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


_PROVIDER_DEFAULTS: dict[str, dict[str, int]] = {
    "tei": {"batch_size": 32, "max_concurrency": 4},
    "ollama": {"batch_size": 32, "max_concurrency": 2},
    "litellm": {"batch_size": 128, "max_concurrency": 8},
}


class EmbeddingSettings(BaseModel):
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
        return self


class MemgraphSettings(BaseModel):
    """Memgraph connection settings."""

    host: str = Field(default="localhost", description="Memgraph host.")
    port: int = Field(default=7687, description="Memgraph Bolt port.")
    username: str = Field(default="", description="Memgraph username.")
    password: str = Field(default="", description="Memgraph password.")
    query_timeout_s: float = Field(default=10.0, description="Timeout in seconds for read queries.")
    write_timeout_s: float = Field(default=60.0, description="Timeout in seconds for write queries.")


class SearchSettings(BaseModel):
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


class DetectorSettings(BaseModel):
    """Pattern detector settings."""

    enabled: list[str] = Field(
        default_factory=lambda: [
            "decorator_routing",
            "event_handlers",
            "test_mapping",
            "class_overrides",
            "di_injection",
            "cli_commands",
        ],
        description="Enabled pattern detectors.",
    )


class IndexSettings(BaseModel):
    """Indexing delta settings."""

    delta_threshold: float = Field(
        default=0.3, description="If more than this fraction of files changed, fall back to full re-index."
    )
    stale_mode: str = Field(
        default="warn",
        description="Stale index behavior: 'warn' (annotate), 'lock' (refuse), 'ignore' (skip).",
    )
    max_source_chars: int = Field(default=2000, description="Max characters for entity source text (0 to disable).")
    file_hash_gate: bool = Field(default=True, description="Skip files whose content hash hasn't changed.")
    strip_whitespace: bool = Field(
        default=True, description="Normalize whitespace before hashing (ignores formatting-only changes)."
    )


class ObservabilitySettings(BaseModel):
    """OpenTelemetry observability settings (requires ``[otel]`` extra)."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing and metrics.")
    exporter: str = Field(default="otlp", description="Exporter type: 'otlp', 'console', or 'none'.")
    endpoint: str = Field(default="http://localhost:4317", description="OTLP collector endpoint.")
    service_name: str = Field(default="code-atlas", description="OTel service.name resource attribute.")
    sample_rate: float = Field(default=1.0, description="Trace sample rate (1.0 = all, 0.1 = 10%).")


class WatcherSettings(BaseModel):
    """File watcher debounce settings."""

    debounce_s: float = Field(default=5.0, description="Debounce timer in seconds (resets per change).")
    max_wait_s: float = Field(default=30.0, description="Max-wait ceiling in seconds (per batch).")
    cooldown_s: float = Field(default=10.0, description="Per-file cooldown after processing (seconds). 0 disables.")


class McpSettings(BaseModel):
    """MCP server settings."""

    host: str = Field(default="127.0.0.1", description="Bind address for HTTP transports (ignored for stdio).")
    port: int = Field(default=8000, description="Bind port for HTTP transports (ignored for stdio).")
    transport: str = Field(default="stdio", description="Transport protocol: 'stdio' or 'streamable-http'.")
    strict: bool = Field(default=False, description="Refuse to start if embedding model mismatch.")


class RedisSettings(BaseModel):
    """Redis/Valkey connection settings for event bus."""

    host: str = Field(default="localhost", description="Redis/Valkey host.")
    port: int = Field(default=6379, description="Redis/Valkey port.")
    db: int = Field(default=0, description="Redis database number.")
    password: str = Field(default="", description="Redis/Valkey password.")
    stream_prefix: str = Field(default="atlas", description="Prefix for Redis Stream keys.")
    stream_maxlen: int = Field(
        default=1_000_000,
        description="Max entries per Redis Stream (XADD maxlen, approximate). "
        "Must exceed the largest expected publish backlog. 0 disables trimming.",
    )


class ProjectSettings(BaseModel):
    """Project identity overrides."""

    name: str | None = Field(
        default=None,
        description="Explicit project name override (see derive_project_name). Set this to "
        "disambiguate two checkouts that share a directory basename — otherwise they collide "
        "in the graph and event streams.",
    )


class ExtraVaultSettings(BaseModel):
    """A knowledge vault indexed as a sibling project alongside this repo's own vault.

    Used for the overspanning (cross-project) vault and the Claude Code
    harness memory directory — both are ordinary projects in the same graph,
    just rooted outside this repo.
    """

    path: str = Field(description="Filesystem path to the vault root (absolute, or ~-expanded).")
    project_name: str = Field(description="Project name this vault indexes under (see derive_project_name).")


class KnowledgeSettings(BaseModel):
    """Knowledge vault settings — the Obsidian-compatible note vault living alongside code."""

    vault_path: str = Field(
        default="docs",
        description="Repo-relative path to this project's knowledge vault. docs/ IS the vault — "
        "frontmatter-triggered note mode lets ordinary docs and vault notes coexist in the same tree.",
    )
    extra_vaults: list[ExtraVaultSettings] = Field(
        default_factory=list,
        description="Additional vaults (global overspanning vault, harness memory dir) indexed as "
        "sibling projects in the same graph. Always included in query scope alongside the current project. "
        "Each gets its own live FileWatcher instance (multi-root watching) plus a one-time startup "
        "catch-up scan — see DaemonManager.start().",
    )


class AtlasSettings(BaseSettings):
    """Root configuration for Code Atlas."""

    model_config = SettingsConfigDict(
        toml_file="atlas.toml",
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
        # Discover atlas.toml relative to the target project_root (when explicitly
        # passed, e.g. `atlas index <other-path>`), not the process cwd — otherwise
        # indexing a project other than the cwd's own applies the wrong config.
        init_kwargs = getattr(init_settings, "init_kwargs", {})
        target_root = init_kwargs.get("project_root")
        toml_path = _find_atlas_toml(Path(target_root) if target_root else None)
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if toml_path:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    project_root: Path = Field(default_factory=_default_project_root, description="Project root path.")
    project: ProjectSettings = Field(default_factory=ProjectSettings)
    scope: ScopeSettings = Field(default_factory=ScopeSettings)
    libraries: LibrarySettings = Field(default_factory=LibrarySettings)
    monorepo: MonorepoSettings = Field(default_factory=MonorepoSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    memgraph: MemgraphSettings = Field(default_factory=MemgraphSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    watcher: WatcherSettings = Field(default_factory=WatcherSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    knowledge: KnowledgeSettings = Field(default_factory=KnowledgeSettings)
    detectors: DetectorSettings = Field(default_factory=DetectorSettings)
    mcp: McpSettings = Field(default_factory=McpSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
