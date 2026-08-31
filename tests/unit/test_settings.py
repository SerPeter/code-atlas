"""Unit tests for settings — env var scoping, nested overrides, and defaults."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from code_atlas import schema
from code_atlas.settings import (
    AtlasSettings,
    BackendSettings,
    DetectorSettings,
    EmbeddingSettings,
    ExtraVaultSettings,
    IndexSettings,
    KnowledgeSettings,
    LibrarySettings,
    McpSettings,
    MemgraphSettings,
    MonorepoSettings,
    ObservabilitySettings,
    ProjectSettings,
    RationaleSettings,
    RedisSettings,
    ScopeSettings,
    SearchSettings,
    WatcherSettings,
    derive_project_name,
    extraction_key,
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Isolate settings from the host: no atlas.toml discovery, no ATLAS_* env vars."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("ATLAS_"):
            monkeypatch.delenv(key)
    return tmp_path


def _extraction_key_in_child(project_root: Path, hash_seed: str) -> str:
    """``extraction_key`` for *project_root*, computed in a fresh interpreter.

    ``PYTHONHASHSEED`` is fixed at interpreter start, so the only way to observe a
    per-process salt is from another process. ATLAS_* is scrubbed for the same reason
    ``clean_env`` scrubs it in-process.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("ATLAS_")}
    env["PYTHONHASHSEED"] = hash_seed
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from code_atlas.settings import AtlasSettings, extraction_key; "
                "print(extraction_key(AtlasSettings(project_root=sys.argv[1])))"
            ),
            str(project_root),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=True,
    )
    return result.stdout.strip()


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
    def test_default_is_one_hundred_thousand(self, clean_env):
        assert RedisSettings().stream_maxlen == 100_000

    def test_zero_disables_trimming(self, clean_env):
        assert RedisSettings(stream_maxlen=0).stream_maxlen == 0


class TestAtlasTomlDiscovery:
    """atlas.toml must be discovered relative to project_root, not the process cwd —
    otherwise `atlas index <other-path>` picks up the config of whatever project the
    caller's shell happens to be sitting in.
    """

    def test_toml_discovered_from_project_root_not_cwd(self, tmp_path, monkeypatch):
        for key in list(os.environ):
            if key.startswith("ATLAS_"):
                monkeypatch.delenv(key)

        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        (project_dir / "atlas.toml").write_text("[memgraph]\nport = 7777\n", encoding="utf-8")

        monkeypatch.chdir(cwd_dir)

        settings = AtlasSettings(project_root=project_dir)

        assert settings.memgraph.port == 7777

    def test_cwd_toml_not_applied_to_unrelated_project_root(self, tmp_path, monkeypatch):
        """A stray atlas.toml sitting in cwd must not leak into a different project_root."""
        for key in list(os.environ):
            if key.startswith("ATLAS_"):
                monkeypatch.delenv(key)

        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()
        (cwd_dir / "atlas.toml").write_text("[memgraph]\nport = 6666\n", encoding="utf-8")
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        monkeypatch.chdir(cwd_dir)

        settings = AtlasSettings(project_root=project_dir)

        assert settings.memgraph.port == 7687  # default, not cwd's 6666


class TestBackendSettings:
    def test_defaults(self):
        settings = BackendSettings()

        assert settings.graph == "auto"
        assert settings.queue == "auto"
        assert settings.sqlite_data_dir == Path(".atlas")


class TestBackendConfigDiscovery:
    """Backend selection is config-driven via the same atlas.toml / pyproject.toml
    [tool.atlas] dual-file discovery as the rest of AtlasSettings.
    """

    def test_atlas_toml_overrides_backend_settings(self, clean_env):
        (clean_env / "atlas.toml").write_text('[backend]\ngraph = "sqlite"\nqueue = "sqlite"\n', encoding="utf-8")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.backend.graph == "sqlite"
        assert settings.backend.queue == "sqlite"

    def test_pyproject_tool_atlas_fallback_picked_up_when_no_atlas_toml(self, clean_env):
        (clean_env / "pyproject.toml").write_text('[tool.atlas.backend]\ngraph = "sqlite"\n', encoding="utf-8")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.backend.graph == "sqlite"

    def test_pyproject_without_tool_atlas_table_is_skipped(self, clean_env):
        """A pyproject.toml with no [tool.atlas] table is transparent — the walk
        continues up to a parent directory's atlas.toml instead of stopping there.
        """
        (clean_env / "atlas.toml").write_text('[backend]\ngraph = "sqlite"\n', encoding="utf-8")
        project_dir = clean_env / "project"
        project_dir.mkdir()
        (project_dir / "pyproject.toml").write_text('[project]\nname = "some-pkg"\n', encoding="utf-8")

        settings = AtlasSettings(project_root=project_dir)

        assert settings.backend.graph == "sqlite"

    def test_env_var_overrides_atlas_toml_backend(self, clean_env, monkeypatch):
        (clean_env / "atlas.toml").write_text('[backend]\ngraph = "sqlite"\n', encoding="utf-8")
        monkeypatch.setenv("ATLAS_BACKEND__GRAPH", "memgraph")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.backend.graph == "memgraph"

    def test_env_var_overrides_pyproject_fallback_backend(self, clean_env, monkeypatch):
        (clean_env / "pyproject.toml").write_text('[tool.atlas.backend]\ngraph = "sqlite"\n', encoding="utf-8")
        monkeypatch.setenv("ATLAS_BACKEND__GRAPH", "memgraph")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.backend.graph == "memgraph"


class TestProjectNameOverride:
    """Two checkouts sharing a folder name collide in the graph/streams unless
    disambiguated via an explicit [project] name override in atlas.toml.
    """

    def test_basename_default_when_no_override(self, tmp_path):
        project_dir = tmp_path / "my-repo"
        project_dir.mkdir()

        assert derive_project_name(project_dir) == "my-repo"

    def test_explicit_override_wins_over_basename(self, tmp_path):
        project_dir = tmp_path / "backend"
        project_dir.mkdir()
        (project_dir / "atlas.toml").write_text('[project]\nname = "acme-backend"\n', encoding="utf-8")

        assert derive_project_name(project_dir) == "acme-backend"

    def test_same_basename_collides_without_override(self, tmp_path):
        repo_a = tmp_path / "a" / "backend"
        repo_a.mkdir(parents=True)
        repo_b = tmp_path / "b" / "backend"
        repo_b.mkdir(parents=True)

        assert derive_project_name(repo_a) == derive_project_name(repo_b) == "backend"

    def test_override_disambiguates_same_basename(self, tmp_path):
        repo_a = tmp_path / "a" / "backend"
        repo_a.mkdir(parents=True)
        (repo_a / "atlas.toml").write_text('[project]\nname = "team-a-backend"\n', encoding="utf-8")
        repo_b = tmp_path / "b" / "backend"
        repo_b.mkdir(parents=True)

        assert derive_project_name(repo_a) == "team-a-backend"
        assert derive_project_name(repo_b) == "backend"

    def test_project_section_loads_through_atlas_settings(self, clean_env):
        """[project] must be a recognized section — extra=forbid would otherwise reject it."""
        (clean_env / "atlas.toml").write_text('[project]\nname = "acme-backend"\n', encoding="utf-8")

        settings = AtlasSettings(project_root=clean_env)

        assert settings.project.name == "acme-backend"


class TestExtraVaultsUniqueness:
    """Duplicate extra_vaults entries would merge unrelated vault data under one project_name,
    or spin up two independent FileWatcher instances double-watching the same directory.
    """

    def test_duplicate_project_name_raises(self, tmp_path):
        vault_a = tmp_path / "vault-a"
        vault_a.mkdir()
        vault_b = tmp_path / "vault-b"
        vault_b.mkdir()

        with pytest.raises(ValidationError, match="project_name"):
            KnowledgeSettings(
                extra_vaults=[
                    ExtraVaultSettings(path=str(vault_a), project_name="shared-vault"),
                    ExtraVaultSettings(path=str(vault_b), project_name="shared-vault"),
                ]
            )

    def test_duplicate_resolved_path_raises(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()

        with pytest.raises(ValidationError, match="path"):
            KnowledgeSettings(
                extra_vaults=[
                    ExtraVaultSettings(path=str(vault), project_name="vault-one"),
                    ExtraVaultSettings(path=str(vault) + "/", project_name="vault-two"),
                ]
            )

    def test_distinct_names_and_paths_do_not_raise(self, tmp_path):
        vault_a = tmp_path / "vault-a"
        vault_a.mkdir()
        vault_b = tmp_path / "vault-b"
        vault_b.mkdir()

        settings = KnowledgeSettings(
            extra_vaults=[
                ExtraVaultSettings(path=str(vault_a), project_name="vault-one"),
                ExtraVaultSettings(path=str(vault_b), project_name="vault-two"),
            ]
        )

        assert len(settings.extra_vaults) == 2


# ---------------------------------------------------------------------------
# Section strictness (ATL-111)
# ---------------------------------------------------------------------------


class TestEverySectionRejectsUnknownKeys:
    """A typo inside a section used to vanish without a word.

    The root model has always been ``extra="forbid"``, but a nested ``BaseModel``
    defaults to ``ignore``. Measured before the fix::

        ScopeSettings(include_paths=[...], exclude_patterns=[...])
        -> {'paths': [], 'include': None, 'exclude': None}

    Someone scoping indexing to three services would have indexed the whole monorepo and
    been told nothing.
    """

    @staticmethod
    def _sections() -> dict[str, type]:
        """Every nested section model reachable from AtlasSettings.

        Discovered from the model rather than listed by hand: a hand-written list is
        exactly what a newly-added section would not appear in, and the whole point is
        that the *next* section is strict too.
        """
        from pydantic import BaseModel

        from code_atlas.settings import AtlasSettings

        found: dict[str, type] = {}
        for name, field in AtlasSettings.model_fields.items():
            annotation = field.annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                found[name] = annotation
        return found

    def test_the_discovery_actually_finds_the_sections(self):
        """Guard the guard: an empty mapping would make every assertion below vacuous."""
        sections = self._sections()

        assert len(sections) >= 15, f"only found {sorted(sections)}"
        assert "scope" in sections
        assert "search" in sections

    def test_no_section_silently_accepts_an_unknown_key(self):
        from pydantic import ValidationError

        accepting = []
        for name, model in self._sections().items():
            try:
                model(definitely_not_a_real_setting_xyz=1)
            except ValidationError:
                continue
            except Exception:
                # A section whose constructor fails for another reason is still strict
                # about extras; only silent acceptance is the defect.
                continue
            accepting.append(name)

        assert not accepting, f"these sections ignore unknown keys: {accepting}"

    def test_the_original_scope_typo_is_now_an_error(self):
        """The exact call that silently produced an empty scope."""
        import pytest
        from pydantic import ValidationError

        from code_atlas.settings import ScopeSettings

        with pytest.raises(ValidationError):
            ScopeSettings(include_paths=["a"], exclude_patterns=["b"])  # ty: ignore[unknown-argument]

    def test_a_correct_key_still_works(self):
        """Strictness must not break the fields that do exist."""
        from code_atlas.settings import ScopeSettings

        assert ScopeSettings(paths=["services/a"]).paths == ["services/a"]


class TestImportanceSettings:
    """[search.importance] — the atlas.toml surface for ranking adjustments."""

    def test_absent_by_default(self, clean_env):
        settings = AtlasSettings(project_root=clean_env)
        assert settings.search.importance.paths == []
        assert settings.search.importance.frontmatter == []
        assert settings.search.importance.is_empty()

    def test_array_of_tables_loads_both_rule_kinds(self, clean_env):
        (clean_env / "atlas.toml").write_text(
            "[[search.importance.paths]]\n"
            'glob = "src/code_atlas/search/**"\n'
            "factor = 1.5\n"
            "\n"
            "[[search.importance.paths]]\n"
            'glob = "wiki/inbox/"\n'
            "factor = 0.5\n"
            "\n"
            "[[search.importance.frontmatter]]\n"
            'key = "metadata.type"\n'
            'value = "decision"\n'
            "factor = 2.0\n"
            "\n"
            "[[search.importance.frontmatter]]\n"
            'key = "deprecated"\n'
            "factor = 0.25\n",
            encoding="utf-8",
        )
        importance = AtlasSettings(project_root=clean_env).search.importance

        assert [(r.glob, r.factor) for r in importance.paths] == [
            ("src/code_atlas/search/**", 1.5),
            ("wiki/inbox/", 0.5),
        ]
        assert [(r.key, r.value, r.factor) for r in importance.frontmatter] == [
            ("metadata.type", "decision", 2.0),
            ("deprecated", None, 0.25),
        ]
        assert not importance.is_empty()

    def test_typo_in_a_rule_is_an_error_not_a_silent_no_op(self, clean_env):
        """StrictSection all the way down — a misspelled key must not vanish."""
        (clean_env / "atlas.toml").write_text(
            '[[search.importance.paths]]\nglob = "src/**"\nfactr = 1.5\n', encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            AtlasSettings(project_root=clean_env)

    def test_non_positive_factor_is_rejected(self, clean_env):
        (clean_env / "atlas.toml").write_text(
            '[[search.importance.paths]]\nglob = "src/**"\nfactor = 0.0\n', encoding="utf-8"
        )
        with pytest.raises(ValidationError):
            AtlasSettings(project_root=clean_env)


class TestExtractionKey:
    """The gate's key must cover extraction and nothing else, and never move on its own.

    An over-inclusive key costs one spurious re-parse. An under-inclusive one leaves the
    graph wrong with no signal, which is the defect ATL-152 exists to remove. An UNSTABLE
    one is worse than both: the daemon always trusts the gate, so a key that differs run to
    run re-parses the whole project forever at watcher cadence and reads as a performance
    regression rather than as a key that moved.
    """

    def test_identical_config_gives_an_identical_key(self, clean_env):
        assert extraction_key(AtlasSettings(project_root=clean_env)) == extraction_key(
            AtlasSettings(project_root=clean_env)
        )

    def test_the_key_does_not_depend_on_list_order(self, clean_env):
        """Reordering a list in atlas.toml must not re-parse the world.

        Extraction reads markers as a frozenset and merely iterates the detector list, so
        their order is provably not output-bearing.
        """
        forward = AtlasSettings(
            project_root=clean_env,
            detectors=DetectorSettings(enabled=["test_mapping", "class_overrides"]),
            rationale=RationaleSettings(markers=["NOTE", "WHY"]),
        )
        reversed_ = AtlasSettings(
            project_root=clean_env,
            detectors=DetectorSettings(enabled=["class_overrides", "test_mapping"]),
            rationale=RationaleSettings(markers=["WHY", "NOTE"]),
        )

        assert extraction_key(forward) == extraction_key(reversed_)

    def test_the_epoch_is_part_of_the_key(self, clean_env, monkeypatch):
        """The sabotage check ATL-152 asks for, in the one place it can be made.

        Drop EXTRACTION_EPOCH from the payload and this is the test that fails.
        """
        settings = AtlasSettings(project_root=clean_env)
        before = extraction_key(settings)

        monkeypatch.setattr(schema, "EXTRACTION_EPOCH", schema.EXTRACTION_EPOCH + 1)

        assert extraction_key(settings) != before

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("index", IndexSettings(max_source_chars=2000)),
            ("index", IndexSettings(max_doc_section_chars=1234)),
            ("index", IndexSettings(max_parse_bytes=4096)),
            ("detectors", DetectorSettings(enabled=["test_mapping"])),
            ("rationale", RationaleSettings(enabled=False)),
            ("rationale", RationaleSettings(markers=["NOTE"])),
            ("rationale", RationaleSettings(tasks=True)),
            ("rationale", RationaleSettings(task_markers=["XXX"])),
            ("rationale", RationaleSettings(citations=False)),
            ("rationale", RationaleSettings(citation_schemes=["ADR"])),
        ],
    )
    def test_every_extraction_affecting_setting_moves_the_key(self, clean_env, field, value):
        """One case per member of extraction_key's IN list.

        ``max_source_chars`` is the 2026-08-31 incident: raising it left 2,169 entities
        holding truncated source while every file's bytes were unchanged.
        """
        baseline = extraction_key(AtlasSettings(project_root=clean_env))

        assert extraction_key(AtlasSettings(project_root=clean_env, **{field: value})) != baseline

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            # Changes what is HASHED, not what is parsed — _compute_file_hash already
            # normalizes on it, so folding it in here would double-count.
            ("index", IndexSettings(strip_whitespace=False)),
            # Pipeline control, not extraction.
            ("index", IndexSettings(delta_threshold=0.9)),
            ("index", IndexSettings(file_hash_gate=False)),
            # Downstream of extraction: gated by embed_hash and by query-time weights.
            ("search", SearchSettings(rrf_k=99)),
            ("memgraph", MemgraphSettings(port=7688)),
            # Never reaches the parser: note mode is triggered by frontmatter, not by path.
            ("knowledge", KnowledgeSettings(vault_path="notes")),
            # The rest of the docstring's OUT list, one case per section, so "both lists
            # are exhaustive as of this writing" is a claim something checks rather than a
            # comment. The tempting member is embeddings: max_input_tokens drives ADR-0040
            # chunking, which reads like extraction and is not — it splits the embed *text*
            # of an entity the parser has already produced, and is gated by embed_hash.
            ("embeddings", EmbeddingSettings(max_input_tokens=1000)),
            ("backend", BackendSettings(graph="sqlite")),
            ("redis", RedisSettings(port=6380)),
            ("watcher", WatcherSettings(debounce_s=5.0)),
            ("mcp", McpSettings(port=9000)),
            ("observability", ObservabilitySettings(enabled=True)),
            ("libraries", LibrarySettings(full_index=["requests"])),
            ("monorepo", MonorepoSettings(auto_detect=False)),
            # Enumeration and project identity. The gate self-heals on both: it is keyed
            # (project_name, file_path), so a newly in-scope file has no stored hash under
            # that key and parses, and a renamed project has none for any of its files.
            ("scope", ScopeSettings(paths=["src"])),
            ("project", ProjectSettings(name="renamed")),
        ],
    )
    def test_a_non_extraction_setting_leaves_the_key_alone(self, clean_env, field, value):
        baseline = extraction_key(AtlasSettings(project_root=clean_env))

        assert extraction_key(AtlasSettings(project_root=clean_env, **{field: value})) == baseline

    def test_the_key_is_the_same_however_the_config_arrived(self, clean_env, monkeypatch):
        """Resolved values, not the route they arrived by.

        An explicit kwarg, an ``ATLAS_*`` env var and an ``atlas.toml`` key are three ways
        to set the same field, and the daemon, the CLI and an MCP session do not all take
        the same one. A key that digested the route would have two processes disagree about
        an identically-configured project — and because the gate is always trusted, the
        symptom is not an error but a project re-parsed forever at watcher cadence.
        """
        baseline = extraction_key(AtlasSettings(project_root=clean_env))
        from_kwarg = extraction_key(AtlasSettings(project_root=clean_env, index=IndexSettings(max_source_chars=1234)))
        assert from_kwarg != baseline, "1234 has to differ from the default, or every equality below is vacuous"

        monkeypatch.setenv("ATLAS_INDEX__MAX_SOURCE_CHARS", "1234")
        from_env = extraction_key(AtlasSettings(project_root=clean_env))
        monkeypatch.delenv("ATLAS_INDEX__MAX_SOURCE_CHARS")

        (clean_env / "atlas.toml").write_text("[index]\nmax_source_chars = 1234\n", encoding="utf-8")
        from_toml = extraction_key(AtlasSettings(project_root=clean_env))

        assert from_kwarg == from_env == from_toml

    def test_the_key_is_the_same_in_another_process(self, clean_env):
        """``hashlib``, never the builtin ``hash()``.

        ``hash()`` of a str is salted per interpreter, so a key built with it would differ
        between the daemon and the CLI while every in-process assertion in this class
        stayed green. Two children with different ``PYTHONHASHSEED`` values is the only
        place that shows up, and it is worth two interpreter starts because the failure it
        catches is silent: the losing process re-parses the whole project on every run.
        """
        keys = {_extraction_key_in_child(clean_env, seed) for seed in ("0", "12345")}

        assert len(keys) == 1, f"the key moved between processes: {sorted(keys)}"
        assert keys == {extraction_key(AtlasSettings(project_root=clean_env))}

    def test_the_key_does_not_depend_on_where_the_project_lives(self, clean_env, tmp_path):
        """project_root is per-worktree and per-machine.

        Keying on it would make two checkouts of the same repo invalidate each other's
        stored hashes, and the gate already keys on (project_name, file_path) anyway.
        """
        other = tmp_path / "elsewhere"
        other.mkdir()

        assert extraction_key(AtlasSettings(project_root=clean_env)) == extraction_key(
            AtlasSettings(project_root=other)
        )
