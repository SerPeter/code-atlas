"""Configuration management for Code Atlas."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
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


def _default_project_root() -> Path:
    """Git root if found, otherwise cwd."""
    return find_git_root() or Path.cwd()


def _find_atlas_toml() -> Path | None:
    """Walk up from cwd looking for ``atlas.toml``."""
    current = Path.cwd().resolve()
    while True:
        candidate = current / "atlas.toml"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


class ScopeSettings(BaseSettings):
    """File scope and ignore settings."""

    include_paths: list[str] = Field(default_factory=list, description="Whitelist of paths to index (monorepo roots).")
    exclude_patterns: list[str] = Field(
        default_factory=list, description="Additional glob patterns to exclude beyond .gitignore."
    )


class LibrarySettings(BaseSettings):
    """Library and dependency indexing settings."""

    full_index: list[str] = Field(default_factory=list, description="Libraries to fully parse and index.")
    stub_index: list[str] = Field(default_factory=list, description="Libraries to index at type-stub level only.")


class MonorepoSettings(BaseSettings):
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


class EmbeddingSettings(BaseSettings):
    """Embedding settings — routes through litellm for any provider."""

    provider: str = Field(default="tei", description="Embedding provider: 'tei', 'litellm', or 'ollama'.")
    model: str = Field(default="nomic-ai/nomic-embed-code", description="Embedding model name.")
    base_url: str = Field(default="http://localhost:8080", description="OpenAI-compatible embedding endpoint URL.")
    dimension: int | None = Field(default=None, description="Embedding vector dimension. Auto-detected when None.")
    batch_size: int = Field(default=32, description="Max texts per embedding API call.")
    timeout_s: float = Field(default=30.0, description="Timeout in seconds for embedding API calls.")
    query_cache_size: int = Field(default=128, description="Max cached query embeddings (LRU eviction).")
    cache_ttl_days: int = Field(default=7, description="Embedding cache TTL in days. 0 disables Valkey caching.")


class MemgraphSettings(BaseSettings):
    """Memgraph connection settings."""

    host: str = Field(default="localhost", description="Memgraph host.")
    port: int = Field(default=7687, description="Memgraph Bolt port.")
    username: str = Field(default="", description="Memgraph username.")
    password: str = Field(default="", description="Memgraph password.")
    query_timeout_s: float = Field(default=10.0, description="Timeout in seconds for read queries.")
    write_timeout_s: float = Field(default=60.0, description="Timeout in seconds for write queries.")


class SearchSettings(BaseSettings):
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


class DetectorSettings(BaseSettings):
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


class IndexSettings(BaseSettings):
    """Indexing delta settings."""

    delta_threshold: float = Field(
        default=0.3, description="If more than this fraction of files changed, fall back to full re-index."
    )
    stale_mode: str = Field(
        default="warn",
        description="Stale index behavior: 'warn' (annotate), 'lock' (refuse), 'ignore' (skip).",
    )


class ObservabilitySettings(BaseSettings):
    """OpenTelemetry observability settings (requires ``[otel]`` extra)."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing and metrics.")
    exporter: str = Field(default="otlp", description="Exporter type: 'otlp', 'console', or 'none'.")
    endpoint: str = Field(default="http://localhost:4317", description="OTLP collector endpoint.")
    service_name: str = Field(default="code-atlas", description="OTel service.name resource attribute.")
    sample_rate: float = Field(default=1.0, description="Trace sample rate (1.0 = all, 0.1 = 10%).")


class WatcherSettings(BaseSettings):
    """File watcher debounce settings."""

    debounce_s: float = Field(default=5.0, description="Debounce timer in seconds (resets per change).")
    max_wait_s: float = Field(default=30.0, description="Max-wait ceiling in seconds (per batch).")


class RedisSettings(BaseSettings):
    """Redis/Valkey connection settings for event bus."""

    host: str = Field(default="localhost", description="Redis/Valkey host.")
    port: int = Field(default=6379, description="Redis/Valkey port.")
    db: int = Field(default=0, description="Redis database number.")
    password: str = Field(default="", description="Redis/Valkey password.")
    stream_prefix: str = Field(default="atlas", description="Prefix for Redis Stream keys.")


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
        toml_path = _find_atlas_toml()
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if toml_path:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        sources.append(file_secret_settings)
        return tuple(sources)

    project_root: Path = Field(default_factory=_default_project_root, description="Project root path.")
    scope: ScopeSettings = Field(default_factory=ScopeSettings)
    libraries: LibrarySettings = Field(default_factory=LibrarySettings)
    monorepo: MonorepoSettings = Field(default_factory=MonorepoSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    memgraph: MemgraphSettings = Field(default_factory=MemgraphSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    watcher: WatcherSettings = Field(default_factory=WatcherSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    detectors: DetectorSettings = Field(default_factory=DetectorSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
