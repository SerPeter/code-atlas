"""Unit tests for settings — env var scoping, nested overrides, and defaults."""

from __future__ import annotations

import os

import pytest

from code_atlas.settings import AtlasSettings, MemgraphSettings, RedisSettings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Isolate settings from the host: no atlas.toml discovery, no ATLAS_* env vars."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("ATLAS_"):
            monkeypatch.delenv(key)
    return tmp_path


class TestEnvVarScoping:
    def test_bare_env_vars_do_not_leak_into_nested_sections(self, clean_env, monkeypatch):
        """Unprefixed env vars (e.g. Windows USERNAME) must not bind to nested fields."""
        monkeypatch.setenv("USERNAME", "windows-logon-name")
        monkeypatch.setenv("HOST", "bare-host")
        monkeypatch.setenv("PORT", "9999")
        monkeypatch.setenv("MODEL", "bare-model")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.memgraph.username == ""
        assert settings.memgraph.host == "localhost"
        assert settings.memgraph.port == 7687
        assert settings.redis.host == "localhost"
        assert settings.redis.port == 6379
        assert settings.mcp.port == 8000
        assert settings.embeddings.model == "nomic-ai/nomic-embed-code"

    def test_prefixed_nested_env_overrides_apply(self, clean_env, monkeypatch):
        """ATLAS_SECTION__FIELD env vars override nested fields; siblings keep defaults."""
        monkeypatch.setenv("ATLAS_MEMGRAPH__PORT", "7999")
        monkeypatch.setenv("ATLAS_REDIS__STREAM_MAXLEN", "500")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.memgraph.port == 7999
        assert settings.memgraph.host == "localhost"
        assert settings.redis.stream_maxlen == 500

    def test_toml_section_still_loads(self, clean_env):
        """atlas.toml sections populate nested settings."""
        (clean_env / "atlas.toml").write_text("[memgraph]\nport = 7777\n", encoding="utf-8")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.memgraph.port == 7777

    def test_env_override_beats_toml_within_section(self, clean_env, monkeypatch):
        """Env vars beat atlas.toml for the same nested field; other toml keys survive (test isolation needs this)."""
        (clean_env / "atlas.toml").write_text('[memgraph]\nport = 7687\nhost = "tomlhost"\n', encoding="utf-8")
        monkeypatch.setenv("ATLAS_MEMGRAPH__PORT", "7688")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.memgraph.port == 7688
        assert settings.memgraph.host == "tomlhost"

    def test_init_kwargs_override_env(self, clean_env, monkeypatch):
        """Explicitly passed nested settings win over env vars (integration conftest relies on this)."""
        monkeypatch.setenv("ATLAS_MEMGRAPH__PORT", "7999")

        settings = AtlasSettings(project_root=clean_env, memgraph=MemgraphSettings(port=7688))

        assert settings.memgraph.port == 7688


class TestStreamMaxlen:
    def test_default_is_one_million(self, clean_env):
        assert RedisSettings().stream_maxlen == 1_000_000

    def test_zero_disables_trimming(self, clean_env):
        assert RedisSettings(stream_maxlen=0).stream_maxlen == 0
