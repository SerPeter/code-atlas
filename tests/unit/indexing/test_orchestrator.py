"""Unit tests for the indexer/orchestrator module (no infrastructure needed)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from typing import TYPE_CHECKING

import pathspec
import pytest

from code_atlas.indexing import orchestrator as orchestrator_module
from code_atlas.indexing.orchestrator import (
    _DEFAULT_EXCLUDE,
    _DEFAULT_INCLUDE,
    FileScope,
    StalenessChecker,
    _check_model_lock,
    _detect_packages,
    _git_changed_files,
    _parse_dependency_versions,
    _read_git_head,
    _stop_consumer_tasks,
    _wait_for_drain,
    gc_vanished_worktree_projects,
    register_manifest_parser,
    scan_files,
)
from code_atlas.parsing.ast import _EXTENSION_MAP, _FILENAME_MAP
from code_atlas.parsing.languages import discover_plugins
from code_atlas.settings import AtlasSettings, ScopeSettings, derive_project_name, resolve_git_dir

if TYPE_CHECKING:
    from pathlib import Path

# Skip decorator for tests that require symlink support (needs admin/dev mode on Windows)
needs_symlinks = pytest.mark.skipif(
    not os.environ.get("CI") and os.name == "nt",
    reason="Symlinks require admin or Developer Mode on Windows",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str = "") -> Path:
    """Write a file at root/rel_path, creating parent dirs."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _make_settings(root: Path, **scope_kwargs) -> AtlasSettings:
    """Create AtlasSettings pointing at *root* with optional scope overrides."""
    return AtlasSettings(
        project_root=root,
        scope=ScopeSettings(**scope_kwargs),
    )


# ---------------------------------------------------------------------------
# scan_files — unit tests (no infrastructure needed)
# ---------------------------------------------------------------------------


class TestScanFiles:
    def test_discovers_py_files(self, tmp_path):
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "lib/utils.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "app.py" in result
        assert "lib/utils.py" in result

    def test_ignores_unsupported_files(self, tmp_path):
        # NOTE: .json used to be the example of an unsupported file here. It is
        # now indexable (structured-config support), so the example moved to
        # extensions no parser claims.
        _write(tmp_path, "data.csv", "a,b\n1,2\n")
        _write(tmp_path, "image.png", "")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "readme.md", "# Hello")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py", "readme.md"]

    def test_default_excludes(self, tmp_path):
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "__pycache__/module.pyc", "")
        _write(tmp_path, ".git/config", "")
        _write(tmp_path, "node_modules/pkg/index.py", "")
        _write(tmp_path, ".venv/lib/site.py", "")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_respects_gitignore(self, tmp_path):
        _write(tmp_path, ".gitignore", "generated/\n*.generated.py\n")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "generated/out.py", "y = 2")
        _write(tmp_path, "foo.generated.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_respects_atlasignore(self, tmp_path):
        _write(tmp_path, ".atlasignore", "vendor/\n")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "vendor/lib.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_scope_paths(self, tmp_path):
        _write(tmp_path, "src/app.py", "x = 1")
        _write(tmp_path, "tests/test_app.py", "y = 2")
        _write(tmp_path, "scripts/deploy.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path), scope_paths=["src"])

        assert result == ["src/app.py"]

    def test_scope_paths_via_settings(self, tmp_path):
        _write(tmp_path, "src/app.py", "x = 1")
        _write(tmp_path, "tests/test_app.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path, paths=["src"]))

        assert result == ["src/app.py"]

    def test_explicit_empty_scope_paths_is_unrestricted_not_a_fallback(self, tmp_path):
        """scope_paths=[] must mean "no restriction", not "fall back to settings.scope.paths".

        The monorepo indexer relies on this: it passes [] for a sub-project an
        ancestor global scope path already covers in full (see
        _sub_project_scope_paths). Falling back to the un-translated,
        repo-root-relative global list here is what made every scoped sub-project
        scan zero files -- the paths never matched relative to the sub's own root.
        """
        _write(tmp_path, "src/app.py", "x = 1")
        _write(tmp_path, "tests/test_app.py", "y = 2")

        # settings.scope.paths restricts to "src", but the explicit [] passed
        # directly to scan_files must win and see everything.
        result = scan_files(tmp_path, _make_settings(tmp_path, paths=["src"]), scope_paths=[])

        assert result == ["src/app.py", "tests/test_app.py"]

    def test_exclude_before_include(self, tmp_path):
        """Excluded files are NOT rescued by scope paths."""
        _write(tmp_path, ".atlasignore", "src/generated/\n")
        _write(tmp_path, "src/app.py", "x = 1")
        _write(tmp_path, "src/generated/out.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path, paths=["src"]))

        assert result == ["src/app.py"]

    def test_extend_exclude_from_settings(self, tmp_path):
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "tmp/scratch.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path, extend_exclude=["tmp/"]))

        assert result == ["app.py"]

    def test_returns_sorted_posix_paths(self, tmp_path):
        _write(tmp_path, "z.py", "")
        _write(tmp_path, "a.py", "")
        _write(tmp_path, "m/b.py", "")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["a.py", "m/b.py", "z.py"]

    def test_empty_directory(self, tmp_path):
        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == []

    def test_pyi_files_included(self, tmp_path):
        _write(tmp_path, "app.pyi", "x: int")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.pyi"]

    def test_nested_gitignore(self, tmp_path):
        """A .gitignore in a subdirectory excludes files below it."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "lib/core.py", "y = 2")
        _write(tmp_path, "lib/.gitignore", "generated/\n")
        _write(tmp_path, "lib/generated/out.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "app.py" in result
        assert "lib/core.py" in result
        assert "lib/generated/out.py" not in result

    def test_nested_gitignore_does_not_leak_up(self, tmp_path):
        """Patterns in a nested .gitignore don't affect sibling dirs."""
        _write(tmp_path, "a/skip.py", "x = 1")
        _write(tmp_path, "b/.gitignore", "skip.py\n")
        _write(tmp_path, "b/skip.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "a/skip.py" in result
        assert "b/skip.py" not in result

    def test_is_included_discovers_nested_gitignore_without_scan(self, tmp_path):
        """is_included() must honor nested .gitignore even if scan() was never called.

        Watcher-mode constructs FileScope and calls is_included() directly
        (DaemonManager.start never calls scan()) — nested-gitignore exclusion
        must not depend on scan() having run first.
        """
        _write(tmp_path, "lib/.gitignore", "generated/\n")
        _write(tmp_path, "lib/generated/out.py", "z = 3")
        _write(tmp_path, "lib/core.py", "y = 2")

        scope = FileScope(tmp_path, _make_settings(tmp_path))
        # Deliberately NOT calling scope.scan() — simulates watcher mode.

        assert scope.is_included("lib/generated/out.py") is False
        assert scope.is_included("lib/core.py") is True

    def test_default_excludes_vendor_and_target(self, tmp_path):
        """vendor/ and target/ are excluded by default."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "vendor/lib.py", "y = 2")
        _write(tmp_path, "target/out.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_file_scope_reuse(self, tmp_path):
        """FileScope can be constructed once and queried multiple times."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "lib/utils.py", "y = 2")
        _write(tmp_path, ".gitignore", "ignored/\n")
        _write(tmp_path, "ignored/secret.py", "z = 3")

        scope = FileScope(tmp_path, _make_settings(tmp_path))
        # Must call scan() first to populate nested specs (and verify it works)
        files = scope.scan()
        assert "app.py" in files
        assert "lib/utils.py" in files

        # is_included() works without re-scanning
        assert scope.is_included("app.py") is True
        assert scope.is_included("lib/utils.py") is True
        assert scope.is_included("ignored/secret.py") is False

    def test_bom_in_gitignore(self, tmp_path):
        """A .gitignore saved with UTF-8 BOM should still work."""
        bom = b"\xef\xbb\xbf"
        gi = tmp_path / ".gitignore"
        gi.write_bytes(bom + b"secret/\n")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "secret/key.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    @needs_symlinks
    def test_symlinked_dir_not_followed(self, tmp_path):
        """Symlinked directories are skipped (matches git default behavior)."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        _write(tmp_path, "real/module.py", "x = 1")
        _write(tmp_path, "app.py", "y = 2")

        link = tmp_path / "linked"
        link.symlink_to(real_dir, target_is_directory=True)

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "app.py" in result
        assert "real/module.py" in result
        # Symlinked directory content should NOT appear
        assert "linked/module.py" not in result

    @needs_symlinks
    def test_broken_symlink_skipped(self, tmp_path):
        """Broken file symlinks don't crash the scanner."""
        _write(tmp_path, "app.py", "x = 1")
        broken = tmp_path / "broken.py"
        broken.symlink_to(tmp_path / "nonexistent.py")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert result == ["app.py"]

    def test_non_ascii_paths(self, tmp_path):
        """Non-ASCII characters in paths don't crash the scanner."""
        _write(tmp_path, "über/app.py", "x = 1")
        _write(tmp_path, "日本語/lib.py", "y = 2")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "über/app.py" in result
        assert "日本語/lib.py" in result

    def test_exclude_overrides_defaults(self, tmp_path):
        """Setting exclude replaces _DEFAULT_EXCLUDE entirely."""
        _write(tmp_path, "app.py", "x = 1")
        # node_modules/ is in _DEFAULT_EXCLUDE but NOT in our custom exclude
        _write(tmp_path, "node_modules/pkg/index.py", "y = 2")
        # custom_skip/ is in our custom exclude
        _write(tmp_path, "custom_skip/lib.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path, exclude=["custom_skip/", ".git/"]))

        assert "app.py" in result
        assert "node_modules/pkg/index.py" in result  # no longer excluded
        assert "custom_skip/lib.py" not in result

    def test_extend_exclude_appends_to_defaults(self, tmp_path):
        """extend_exclude adds to defaults without replacing them."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "node_modules/pkg/index.py", "y = 2")  # default exclude
        _write(tmp_path, "extra_skip/lib.py", "z = 3")

        result = scan_files(tmp_path, _make_settings(tmp_path, extend_exclude=["extra_skip/"]))

        assert result == ["app.py"]

    def test_include_restricts_files(self, tmp_path):
        """Setting include restricts to only those patterns."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "lib.ts", "y = 2")
        _write(tmp_path, "readme.md", "# Hello")

        result = scan_files(tmp_path, _make_settings(tmp_path, include=["*.py"]))

        assert result == ["app.py"]

    def test_extend_include_adds_patterns(self, tmp_path):
        """extend_include adds to default include patterns."""
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "data.csv", "a,b\n1,2\n")

        # .csv is not in _DEFAULT_INCLUDE, but extend_include adds it.
        # data.csv has no language parser so scan() won't return it, but
        # is_included() should pass it through. (This used to use *.yaml, which
        # became a default include when structured-config support landed and so
        # no longer proved anything.)
        scope = FileScope(tmp_path, _make_settings(tmp_path, extend_include=["*.csv"]))
        assert scope.is_included("data.csv") is True
        assert scope.is_included("app.py") is True

    def test_default_include_covers_every_registered_language(self):
        """_DEFAULT_INCLUDE must cover every extension and filename in the parser registry.

        This is the guard against the project's sharpest silent-failure mode.
        Two registries decide whether a file is ever indexed, and they are
        maintained by hand in different files:

          * ``_DEFAULT_INCLUDE`` (here)   — the scope allowlist
          * ``_EXTENSION_MAP``/``_FILENAME_MAP`` (parsing.ast) — the parser registry

        ``FileScope.scan`` consults both, but the live ``FileWatcher`` consults
        only the allowlist. So registering a language without adding its globs
        here yields a parser that never runs: no error, no warning at INFO, the
        files simply never appear.

        Asserted as a superset, not equality — ``_DEFAULT_INCLUDE`` legitimately
        carries patterns with no registered parser (Apex ``*.cls``/``*.trigger``
        are plumbed ahead of their module).

        The probe is behavioural rather than a string-set comparison: it compiles
        the real PathSpec and asks it about a synthetic path per registry entry.
        That way non-``*.ext`` patterns (bare ``Dockerfile``) and pathspec's
        case-sensitivity are exercised as the scanner actually experiences them.

        Note this can only under-report: a grammar missing from the venv keeps
        its extension out of the registry, so the assertion passes. It fails
        loudly in the dev venv and in CI, where all grammars are installed.
        """
        discover_plugins()
        assert _EXTENSION_MAP, "language registry is empty — discover_plugins() did not run"

        spec = pathspec.PathSpec.from_lines("gitignore", _DEFAULT_INCLUDE)
        # {probe path: what it represents in the failure message}
        probes = {f"probe{ext}": ext for ext in _EXTENSION_MAP}
        probes.update({filename: f"filename {filename!r}" for filename in _FILENAME_MAP})

        missing = sorted(label for probe, label in probes.items() if not spec.match_file(probe))
        assert not missing, (
            f"registered by a parser but not matched by _DEFAULT_INCLUDE: {missing}. "
            "Files with these extensions are silently never scanned and never watched. "
            "Add the matching glob(s) to _DEFAULT_INCLUDE in indexing/orchestrator.py."
        )

    def test_default_include_covers_new_config_and_infra_extensions(self):
        """Pin the extensions this release added, independent of what happens to register."""
        spec = pathspec.PathSpec.from_lines("gitignore", _DEFAULT_INCLUDE)
        for ext in (".tf", ".tfvars", ".hcl", ".sh", ".bash", ".zsh", ".sql", ".cls", ".trigger"):
            assert spec.match_file(f"probe{ext}"), f"{ext} missing from _DEFAULT_INCLUDE"
        for ext in (".yaml", ".yml", ".json", ".toml", ".xml"):
            assert spec.match_file(f"probe{ext}"), f"{ext} missing from _DEFAULT_INCLUDE"
        # Extensionless container builds, in the casings that occur in the wild.
        for name in ("Dockerfile", "dockerfile", "Containerfile", "containerfile"):
            assert spec.match_file(name), f"{name} missing from _DEFAULT_INCLUDE"
        assert spec.match_file("build/Dockerfile"), "Dockerfile must match at any depth"

    def test_secret_bearing_files_are_denied_by_name(self):
        """Hard backstop: an indexed entity is an *embedded* one, so its content
        leaves the machine for the embedding API. Widening the include list to
        *.yaml/*.json/*.toml/*.tfvars made these reachable for the first time —
        before this deny-list, every name below passed is_included().

        .gitignore is not an adequate substitute: atlas never reads
        .git/info/exclude or core.excludesFile, drops the repo-root .gitignore
        when a monorepo sub-project is rooted in a subdirectory, and matches
        case-sensitively where git on Windows does not.
        """
        spec = pathspec.PathSpec.from_lines("gitignore", _DEFAULT_EXCLUDE)
        for secret in (
            ".env",
            ".env.local",
            "prod.env",
            "secrets.yaml",
            "app-secrets.yaml",
            "gcp-key.json",
            "service-account.json",
            "local.settings.json",
            "credentials.toml",
            "terraform.tfvars",
            "id_rsa",
            "id_ed25519",
            "server.pem",
            "private.key",
            "store.p12",
            ".npmrc",
            ".netrc",
            ".pypirc",
            ".aws/credentials",
            ".ssh/id_ed25519",
        ):
            assert spec.match_file(secret), f"{secret} is NOT denied — secrets would be indexed and embedded"

    def test_secret_deny_list_does_not_swallow_ordinary_config(self):
        """The deny-list must not be so broad it defeats the config parsers."""
        spec = pathspec.PathSpec.from_lines("gitignore", _DEFAULT_EXCLUDE)
        for benign in (
            "values-prod.yaml",
            "tsconfig.json",
            "pyproject.toml",
            "main.tf",
            "docker-compose.yml",
            "k8s/deployment.yaml",
            "package.json",
            ".github/workflows/ci.yml",
        ):
            assert not spec.match_file(benign), f"{benign} wrongly denied — config parsing would never see it"

    def test_lockfiles_excluded_but_real_manifests_are_not(self):
        """Committed lockfiles must not ride in on the new *.json/*.yaml includes.

        .gitignore covers most generated config, but lockfiles are deliberately
        committed, so only _DEFAULT_EXCLUDE can keep them out.
        """
        spec = pathspec.PathSpec.from_lines("gitignore", _DEFAULT_EXCLUDE)
        for locked in (
            "package-lock.json",
            "npm-shrinkwrap.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "poetry.lock",
            "uv.lock",
            "Cargo.lock",
            "composer.lock",
            "Gemfile.lock",
            ".terraform.lock.hcl",
            "bundle.min.json",
            "frontend/package-lock.json",
        ):
            assert spec.match_file(locked), f"{locked} should be excluded"

        # Hand-written manifests carry real signal and must survive.
        for kept in ("package.json", "pyproject.toml", "docker-compose.yml", "infra/main.tf"):
            assert not spec.match_file(kept), f"{kept} must NOT be excluded"

    def test_terraform_provider_cache_is_pruned(self):
        """.terraform/ is the .tf equivalent of node_modules — vendored provider schemas."""
        assert ".terraform/" in _DEFAULT_EXCLUDE
        assert ".terraform" in orchestrator_module._DETECT_PRUNE_DIRS

    def test_scan_surfaces_filename_dispatched_dockerfile(self, tmp_path):
        """End-to-end proof the two-registry trap is closed for an extensionless file.

        scan() applies the allowlist AND the parser registry, so a Dockerfile
        coming back from here means both halves agree.
        """
        pytest.importorskip("tree_sitter_containerfile", reason="tree-sitter-containerfile not installed")
        _write(tmp_path, "Dockerfile", "FROM alpine:3\nRUN echo hi\n")
        # NOT "build/" — that is a _DEFAULT_EXCLUDE directory and would be pruned.
        _write(tmp_path, "docker/api.dockerfile", "FROM alpine:3\n")
        _write(tmp_path, "notes.txt", "not indexable")

        result = scan_files(tmp_path, _make_settings(tmp_path))

        assert "Dockerfile" in result
        assert "docker/api.dockerfile" in result
        assert "notes.txt" not in result

    def test_scan_picks_up_apex_now_that_it_has_a_parser(self, tmp_path):
        """``.cls``/``.trigger`` clear both gates: the allowlist and scan()'s parser check.

        Inverted from the pre-parser assertion this replaces — ``parsing.languages.apex``
        registers both extensions, so the files the watcher already published events
        for now actually produce entities.
        """
        _write(tmp_path, "Foo.cls", "public class Foo {}")
        _write(tmp_path, "Bar.trigger", "trigger Bar on Account (before insert) {}")

        scope = FileScope(tmp_path, _make_settings(tmp_path))
        assert scope.is_included("Foo.cls") is True
        assert sorted(scan_files(tmp_path, _make_settings(tmp_path))) == ["Bar.trigger", "Foo.cls"]

    def test_gitignore_applied_with_custom_exclude(self, tmp_path):
        """.gitignore still works when exclude overrides defaults."""
        _write(tmp_path, ".gitignore", "secret/\n")
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "secret/key.py", "y = 2")

        # Custom exclude replaces defaults but .gitignore still applies
        result = scan_files(tmp_path, _make_settings(tmp_path, exclude=[".git/"]))

        assert result == ["app.py"]

    def test_default_exclude_is_comprehensive(self, tmp_path):
        """_DEFAULT_EXCLUDE covers all common build/cache/VCS directories."""
        expected_dirs = [".git/", "node_modules/", "__pycache__/", ".venv/", "target/", "build/", "dist/"]
        for pattern in expected_dirs:
            assert pattern in _DEFAULT_EXCLUDE, f"{pattern} missing from _DEFAULT_EXCLUDE"


# ---------------------------------------------------------------------------
# _sub_project_in_scope / _sub_project_scope_paths — unit tests
#
# scope.paths is repo-root-relative; a monorepo sub-project's own FileScope is
# rooted at the sub-project's directory. These two functions translate between
# the two coordinate systems for _index_monorepo_inner's per-sub-project scan.
# ---------------------------------------------------------------------------


class TestSubProjectScopeTranslation:
    def test_ancestor_path_covers_whole_sub_project(self):
        # global "training" covers sub-project "training/core" entirely
        assert orchestrator_module._sub_project_in_scope(["training"], "training/core") is True
        assert orchestrator_module._sub_project_scope_paths(["training"], "training/core") == []

    def test_exact_match_covers_whole_sub_project(self):
        assert orchestrator_module._sub_project_in_scope(["training/core"], "training/core") is True
        assert orchestrator_module._sub_project_scope_paths(["training/core"], "training/core") == []

    def test_nested_path_translated_relative_to_sub_root(self):
        global_paths = ["training/core/src", "training/core/tests", "training/pipeline/src"]
        assert orchestrator_module._sub_project_in_scope(global_paths, "training/core") is True
        assert orchestrator_module._sub_project_scope_paths(global_paths, "training/core") == ["src", "tests"]

    def test_unrelated_sub_project_has_no_overlap(self):
        global_paths = ["training/core/src", "wiki"]
        assert orchestrator_module._sub_project_in_scope(global_paths, "training/pipeline") is False

    def test_trailing_slash_normalised(self):
        assert orchestrator_module._sub_project_in_scope(["training/core/"], "training/core") is True
        assert orchestrator_module._sub_project_scope_paths(["training/core/"], "training/core") == []


# ---------------------------------------------------------------------------
# _detect_packages — unit tests (no infrastructure needed)
# ---------------------------------------------------------------------------


class TestDetectPackages:
    def test_detect_packages_strips_source_root(self, tmp_path):
        """Package qns must converge with parser uids: 'src/' stripped, rel_path stays physical."""
        _write(tmp_path, "src/mypkg/__init__.py")
        _write(tmp_path, "src/mypkg/sub/__init__.py")

        result = _detect_packages(tmp_path)

        assert ("mypkg", "src/mypkg") in result
        assert ("mypkg.sub", "src/mypkg/sub") in result
        assert all(not qn.startswith("src.") for qn, _ in result)

    def test_detect_packages_flat_layout_unchanged(self, tmp_path):
        _write(tmp_path, "mypkg/__init__.py")
        _write(tmp_path, "mypkg/sub/__init__.py")

        result = _detect_packages(tmp_path)

        assert ("mypkg", "mypkg") in result
        assert ("mypkg.sub", "mypkg/sub") in result

    def test_detect_packages_src_itself_a_package(self, tmp_path):
        """A source root that is itself a package keeps qn 'src'."""
        _write(tmp_path, "src/__init__.py")

        assert ("src", "src") in _detect_packages(tmp_path)


# ---------------------------------------------------------------------------
# _git_changed_files — unit tests (requires git CLI, no infrastructure)
# ---------------------------------------------------------------------------


def _git(cwd, *args):
    """Run a git command in cwd."""
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _init_git_repo(tmp_path):
    """Initialise a git repo with an initial commit."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")


def _get_head(tmp_path):
    """Return the full HEAD hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class TestGitChangedFiles:
    def test_added_file(self, tmp_path):
        """New files since the base commit are detected as 'created'."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        _write(tmp_path, "new.py", "y = 2")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "add new")

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        paths = {p for p, _ in changes}
        types = dict(changes)
        assert "new.py" in paths
        assert types["new.py"] == "created"

    def test_modified_file(self, tmp_path):
        """Modified files are detected as 'modified'."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        _write(tmp_path, "app.py", "x = 2")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "modify")

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        types = dict(changes)
        assert types["app.py"] == "modified"

    def test_deleted_file(self, tmp_path):
        """Deleted files are detected as 'deleted'."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _write(tmp_path, "old.py", "y = 2")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        (tmp_path / "old.py").unlink()
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "delete")

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        types = dict(changes)
        assert types["old.py"] == "deleted"

    def test_invalid_hash_returns_none(self, tmp_path):
        """An invalid base hash returns None (fallback to full mode)."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")

        result = _git_changed_files(tmp_path, "0000000000000000000000000000000000000000")

        assert result is None

    def test_no_changes(self, tmp_path):
        """No changes returns empty list."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "app.py", "x = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        changes = _git_changed_files(tmp_path, base_hash)

        assert changes is not None
        assert changes == []

    def test_subdir_project_paths_relative_to_project_root(self, tmp_path):
        """Paths from a project_root below the git top-level are project-root-relative.

        Without ``--relative`` git prints repo-root-relative paths, so
        ``git_changed_paths & current_file_set`` is permanently empty for
        monorepo sub-projects.
        """
        _init_git_repo(tmp_path)
        _write(tmp_path, "packages/core/src/mod.py", "x = 1")
        _write(tmp_path, "root.py", "y = 1")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        base_hash = _get_head(tmp_path)

        _write(tmp_path, "packages/core/src/mod.py", "x = 2")
        _write(tmp_path, "root.py", "y = 2")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "modify")

        changes = _git_changed_files(tmp_path / "packages" / "core", base_hash)

        assert changes == [("src/mod.py", "modified")]


# ---------------------------------------------------------------------------
# resolve_git_dir — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestResolveGitDir:
    def test_normal_repo(self, tmp_path):
        """Normal repo with .git directory returns it."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        assert resolve_git_dir(tmp_path) == git_dir

    def test_worktree(self, tmp_path):
        """.git file with gitdir: pointer follows to the real git dir."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "feature"
        real_git_dir.mkdir(parents=True)
        worktree = tmp_path / "feature"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        result = resolve_git_dir(worktree)
        assert result == real_git_dir

    def test_relative_gitdir(self, tmp_path):
        """Relative gitdir: path is resolved against project_root."""
        real_git_dir = tmp_path / "main" / ".git" / "worktrees" / "wt"
        real_git_dir.mkdir(parents=True)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        # Write a relative path from worktree dir to real git dir
        rel = os.path.relpath(real_git_dir, worktree).replace("\\", "/")
        (worktree / ".git").write_text(f"gitdir: {rel}\n", encoding="utf-8")

        result = resolve_git_dir(worktree)
        assert result is not None
        assert result.resolve() == real_git_dir.resolve()

    def test_no_git(self, tmp_path):
        """No .git file or directory returns None."""
        assert resolve_git_dir(tmp_path) is None


# ---------------------------------------------------------------------------
# derive_project_name — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestDeriveProjectName:
    def test_normal_repo(self, tmp_path):
        """Normal repo returns directory basename."""
        (tmp_path / ".git").mkdir()
        assert derive_project_name(tmp_path) == tmp_path.resolve().name

    def test_worktree_with_branch(self, tmp_path):
        """Worktree returns basename@branch."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "feat-login"
        real_git_dir.mkdir(parents=True)
        (real_git_dir / "HEAD").write_text("ref: refs/heads/feat/login\n", encoding="utf-8")

        worktree = tmp_path / "myapp"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        assert derive_project_name(worktree) == "myapp@feat/login"

    def test_detached_worktree_fallback(self, tmp_path):
        """Detached HEAD worktree uses git dir basename as fallback."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "hotfix-42"
        real_git_dir.mkdir(parents=True)
        detached_hash = "a" * 40
        (real_git_dir / "HEAD").write_text(detached_hash + "\n", encoding="utf-8")

        worktree = tmp_path / "myapp"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        assert derive_project_name(worktree) == "myapp@hotfix-42"

    def test_non_git_directory(self, tmp_path):
        """Non-git directory returns plain basename."""
        assert derive_project_name(tmp_path) == tmp_path.resolve().name


# ---------------------------------------------------------------------------
# _read_git_head — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestReadGitHead:
    def test_reads_ref_head(self, tmp_path):
        """Reads HEAD via ref: pointer to a loose ref file."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        fake_hash = "a" * 40
        (refs_dir / "main").write_text(fake_hash + "\n", encoding="utf-8")

        assert _read_git_head(tmp_path) == fake_hash

    def test_reads_detached_head(self, tmp_path):
        """Detached HEAD with a raw 40-char hex hash."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        fake_hash = "b" * 40
        (git_dir / "HEAD").write_text(fake_hash + "\n", encoding="utf-8")

        assert _read_git_head(tmp_path) == fake_hash

    def test_reads_packed_ref(self, tmp_path):
        """Falls back to packed-refs when the loose ref file is missing."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        fake_hash = "c" * 40
        (git_dir / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled sorted\n{fake_hash} refs/heads/main\n",
            encoding="utf-8",
        )

        assert _read_git_head(tmp_path) == fake_hash

    def test_returns_none_no_git(self, tmp_path):
        """Returns None for directories without .git."""
        assert _read_git_head(tmp_path) is None

    def test_reads_head_in_worktree(self, tmp_path):
        """_read_git_head follows .git file -> real git dir."""
        real_git_dir = tmp_path / "main-repo" / ".git" / "worktrees" / "wt"
        real_git_dir.mkdir(parents=True)
        fake_hash = "f" * 40
        (real_git_dir / "HEAD").write_text(fake_hash + "\n", encoding="utf-8")

        worktree = tmp_path / "wt"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        assert _read_git_head(worktree) == fake_hash


# ---------------------------------------------------------------------------
# StalenessChecker — unit tests (no infrastructure)
# ---------------------------------------------------------------------------


class TestStalenessChecker:
    def test_current_head_caches_by_mtime(self, tmp_path):
        """Same mtime returns cached value without re-reading the file."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        fake_hash = "d" * 40
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (refs_dir / "main").write_text(fake_hash + "\n", encoding="utf-8")

        checker = StalenessChecker(tmp_path)
        h1 = checker.current_head()
        assert h1 == fake_hash

        # Overwrite file content but keep the same text — mtime hasn't changed (fast enough)
        # The cache should still return the same hash
        h2 = checker.current_head()
        assert h2 == fake_hash

    def test_current_head_invalidates_on_mtime_change(self, tmp_path):
        """New ref content with changed mtime triggers re-read."""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        refs_dir = git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        hash1 = "d" * 40
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        ref_file = refs_dir / "main"
        ref_file.write_text(hash1 + "\n", encoding="utf-8")

        checker = StalenessChecker(tmp_path)
        assert checker.current_head() == hash1

        # Write a new hash and force mtime change
        time.sleep(0.05)
        hash2 = "e" * 40
        ref_file.write_text(hash2 + "\n", encoding="utf-8")

        assert checker.current_head() == hash2

    def test_current_head_non_git_dir(self, tmp_path):
        """Non-git directory returns None."""
        checker = StalenessChecker(tmp_path)
        assert checker.current_head() is None

    def test_project_name(self, tmp_path):
        """project_name is derived via derive_project_name()."""
        checker = StalenessChecker(tmp_path)
        assert checker.project_name == derive_project_name(tmp_path)

    def test_project_name_worktree(self, tmp_path):
        """project_name includes worktree branch suffix."""
        real_git_dir = tmp_path / "main" / ".git" / "worktrees" / "dev"
        real_git_dir.mkdir(parents=True)
        (real_git_dir / "HEAD").write_text("ref: refs/heads/dev\n", encoding="utf-8")
        refs_dir = real_git_dir / "refs" / "heads"
        refs_dir.mkdir(parents=True)
        (refs_dir / "dev").write_text("a" * 40 + "\n", encoding="utf-8")

        worktree = tmp_path / "myapp"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {real_git_dir}\n", encoding="utf-8")

        checker = StalenessChecker(worktree)
        assert checker.project_name == "myapp@dev"


# ---------------------------------------------------------------------------
# _wait_for_drain — unit tests with a fake bus (no infrastructure)
# ---------------------------------------------------------------------------


class FakeDrainBus:
    """Stub bus returning a canned info dict for every queried group."""

    def __init__(self, info: dict) -> None:
        self._info = info

    async def stream_group_info_multi(self, queries):
        return [dict(self._info) for _ in queries]


class TestWaitForDrain:
    async def test_drained_returns_true(self):
        """pending == 0 and lag == 0 sustained for settle_s returns True."""
        bus = FakeDrainBus({"pending": 0, "lag": 0})

        drained = await _wait_for_drain(bus, 5.0, embed_enabled=False, settle_s=0.0)  # type: ignore[arg-type]

        assert drained is True

    async def test_timeout_returns_false(self):
        """Outstanding work at the deadline returns False instead of falling through."""
        bus = FakeDrainBus({"pending": 3, "lag": 2})

        drained = await _wait_for_drain(bus, 0.6, embed_enabled=False, settle_s=0.0)  # type: ignore[arg-type]

        assert drained is False

    async def test_null_lag_is_not_drained(self):
        """lag=None (stream trimmed past group read position) means unknown → not drained."""
        bus = FakeDrainBus({"pending": 0, "lag": None})

        drained = await _wait_for_drain(bus, 0.6, embed_enabled=False, settle_s=0.0)  # type: ignore[arg-type]

        assert drained is False

    async def test_zero_timeout_returns_false_without_error(self):
        """timeout_s <= 0 returns False immediately (no NameError on unset locals)."""
        bus = FakeDrainBus({"pending": 0, "lag": 0})

        drained = await _wait_for_drain(bus, 0.0, embed_enabled=False, settle_s=0.0)  # type: ignore[arg-type]

        assert drained is False

    async def test_progress_callback_reports_pending_when_lag_unknown(self):
        """With lag=None the progress callback receives the pending count (display only)."""
        seen: list[tuple[int, int, int]] = []
        bus = FakeDrainBus({"pending": 4, "lag": None})

        await _wait_for_drain(
            bus,  # type: ignore[arg-type]
            0.3,
            embed_enabled=False,
            settle_s=0.0,
            on_drain_progress=lambda t1, t2, t3: seen.append((t1, t2, t3)),
        )

        assert seen
        assert seen[0] == (0, 4, 0)


# ---------------------------------------------------------------------------
# _check_model_lock — unit tests with fakes (no infrastructure)
# ---------------------------------------------------------------------------


class FakeLockGraph:
    """Records destructive/config calls made by _check_model_lock."""

    def __init__(
        self,
        stored: tuple[str, int] | None = None,
        *,
        project_models: dict[str, str] | None = None,
        counts: dict[str, int] | None = None,
    ) -> None:
        self.stored = stored
        self.project_models = dict(project_models or {})
        self.counts = dict(counts or {})
        self.calls: list[tuple] = []

    async def get_embedding_config(self):
        return self.stored

    async def set_embedding_config(self, model, dimension):
        self.calls.append(("set_embedding_config", model, dimension))

    async def get_project_embedding_model(self, project):
        return self.project_models.get(project)

    async def set_project_embedding_model(self, project, model):
        self.project_models[project] = model
        self.calls.append(("set_project_embedding_model", project, model))

    async def get_embedding_models_by_project(self):
        return dict(self.project_models)

    async def count_embeddings_by_project(self):
        return dict(self.counts)

    async def clear_embeddings(self, project=None):
        self.calls.append(("clear_embeddings", project))
        return self.counts.get(project, 0) if project else sum(self.counts.values())

    async def rebuild_vector_indices(self, dimension):
        self.calls.append(("rebuild_vector_indices", dimension))


class TestCheckModelLock:
    """The lock splits: dimension is database-wide, model is per project (ATL-135).

    The split is forced by the storage. Vector indices are one per label for the whole
    database and carry a single dimension, so a dimension change rebuilds indices every
    project shares. A model decides only which space a vector lives in, and nothing
    about that is shared -- so one project changing model must not touch another's.
    """

    async def test_full_reindex_unchanged_config_is_not_destructive(self):
        """--full with unchanged model/dimension must not wipe other projects' embeddings."""
        graph = FakeLockGraph(stored=("model-a", 768), project_models={"proj": "model-a"})

        await _check_model_lock(graph, "model-a", 768, project="proj", reindex=True)  # type: ignore[arg-type]

        call_names = {c[0] for c in graph.calls}
        assert "clear_embeddings" not in call_names
        assert "rebuild_vector_indices" not in call_names

    async def test_full_reindex_first_run_writes_config_without_wipe(self):
        """--full on a fresh database writes config without a destructive pass."""
        graph = FakeLockGraph(stored=None)

        await _check_model_lock(graph, "model-a", 768, project="proj", reindex=True)  # type: ignore[arg-type]

        assert graph.calls == [
            ("set_embedding_config", "model-a", 768),
            ("set_project_embedding_model", "proj", "model-a"),
        ]

    async def test_model_change_clears_only_the_changing_project(self):
        """The bug this class exists for: one project's model change destroyed the rest.

        `other` keeps its 5,000 vectors. Before ATL-135 this call wiped them, because
        the lock was global and its only remedy was `clear_all_embeddings`.
        """
        graph = FakeLockGraph(
            stored=("model-a", 768),
            project_models={"proj": "model-a", "other": "model-a"},
            counts={"proj": 100, "other": 5_000},
        )

        await _check_model_lock(graph, "model-b", 768, project="proj", reindex=True)  # type: ignore[arg-type]

        assert ("clear_embeddings", "proj") in graph.calls
        assert ("clear_embeddings", None) not in graph.calls
        assert ("set_project_embedding_model", "proj", "model-b") in graph.calls
        # Shared vector indices are not rebuilt: the dimension did not change.
        assert "rebuild_vector_indices" not in {c[0] for c in graph.calls}
        # The other project keeps both its vectors and its model.
        assert graph.project_models["other"] == "model-a"

    async def test_dimension_change_is_database_wide_because_indices_are(self):
        graph = FakeLockGraph(
            stored=("model-a", 512),
            project_models={"proj": "model-a"},
            counts={"proj": 100, "other": 5_000},
        )

        await _check_model_lock(graph, "model-a", 768, project="proj", reindex=True)  # type: ignore[arg-type]

        assert ("clear_embeddings", None) in graph.calls
        assert ("rebuild_vector_indices", 768) in graph.calls
        assert ("set_embedding_config", "model-a", 768) in graph.calls

    async def test_model_mismatch_without_reindex_raises_and_names_the_other_projects(self):
        """A message naming only two model strings cannot be acted on when the cause is
        another project the reader was not thinking about."""
        graph = FakeLockGraph(
            stored=("model-a", 768),
            project_models={"proj": "model-a", "other": "model-z"},
        )

        with pytest.raises(RuntimeError, match="model for project 'proj' changed") as exc:
            await _check_model_lock(graph, "model-b", 768, project="proj", reindex=False)  # type: ignore[arg-type]

        assert "other='model-z'" in str(exc.value)
        assert "unaffected" in str(exc.value)

    async def test_dimension_mismatch_without_reindex_raises(self):
        graph = FakeLockGraph(stored=("model-a", 512), counts={"proj": 100})

        with pytest.raises(RuntimeError, match="dimension changed"):
            await _check_model_lock(graph, "model-a", 768, project="proj", reindex=False)  # type: ignore[arg-type]

    async def test_a_project_with_no_recorded_model_adopts_the_configured_one(self):
        """Bootstrap: vectors written before the per-project lock existed were written
        under that project's own configuration, so adopting it is right -- and must not
        clear anything."""
        graph = FakeLockGraph(stored=("model-a", 768), counts={"proj": 6_691})

        await _check_model_lock(graph, "model-b", 768, project="proj", reindex=False)  # type: ignore[arg-type]

        assert ("set_project_embedding_model", "proj", "model-b") in graph.calls
        assert "clear_embeddings" not in {c[0] for c in graph.calls}

    async def test_two_projects_on_different_models_both_index(self):
        """The deadlock, directly: neither project may lock the other out."""
        graph = FakeLockGraph(
            stored=("model-a", 768),
            project_models={"alpha": "model-a", "beta": "model-b"},
        )

        # Both are already on their own model, so both are no-ops rather than errors.
        await _check_model_lock(graph, "model-a", 768, project="alpha", reindex=False)  # type: ignore[arg-type]
        await _check_model_lock(graph, "model-b", 768, project="beta", reindex=False)  # type: ignore[arg-type]

        assert graph.calls == []


# ---------------------------------------------------------------------------
# Worktree-project GC (Phase 5)
# ---------------------------------------------------------------------------


class FakeGCGraph:
    """Records delete_project_data calls made by gc_vanished_worktree_projects."""

    def __init__(self, projects: list[dict]) -> None:
        self._projects = projects
        self.deleted: list[str] = []

    async def get_project_status(self):
        return self._projects

    async def delete_project_data(self, project_name: str) -> None:
        self.deleted.append(project_name)


class TestGcVanishedWorktreeProjects:
    async def test_removes_worktree_project_whose_root_path_vanished(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        graph = FakeGCGraph(
            [
                {"n": {"project_name": "code-atlas", "root_path": str(tmp_path)}},
                {"n": {"project_name": "code-atlas@feature-x", "root_path": str(tmp_path / "gone")}},
            ]
        )
        monkeypatch.setattr(orchestrator_module, "_git_worktree_list", lambda base_root: [tmp_path.resolve()])

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == ["code-atlas@feature-x"]
        assert graph.deleted == ["code-atlas@feature-x"]

    async def test_spares_worktree_project_whose_root_path_still_exists(self, tmp_path: Path):
        graph = FakeGCGraph([{"n": {"project_name": "code-atlas@feature-x", "root_path": str(tmp_path)}}])

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []

    async def test_spares_non_worktree_project_regardless_of_root_path(self, tmp_path: Path):
        """Only '@branch'-shaped names are worktree candidates — a plain project
        with a vanished root_path (e.g. moved checkout) is not auto-GC'd."""
        graph = FakeGCGraph([{"n": {"project_name": "code-atlas", "root_path": str(tmp_path / "gone")}}])

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []

    async def test_spares_worktree_project_with_no_stored_root_path(self, tmp_path: Path):
        """No root_path stored (pre-orchestrator-fix data) — nothing to check against, so skip rather than guess."""
        graph = FakeGCGraph([{"n": {"project_name": "code-atlas@feature-x"}}])

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []

    async def test_spares_worktree_project_when_base_project_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No base-project record in the graph to corroborate against — skip rather than guess."""
        graph = FakeGCGraph([{"n": {"project_name": "code-atlas@feature-x", "root_path": str(tmp_path / "gone")}}])
        calls: list[Path] = []
        monkeypatch.setattr(orchestrator_module, "_git_worktree_list", calls.append)

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []
        assert calls == []  # git shouldn't even be consulted without a base root_path to run it against

    async def test_spares_worktree_project_when_base_root_path_not_a_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Base project's own root_path also isn't a live directory — no independent signal, skip."""
        graph = FakeGCGraph(
            [
                {"n": {"project_name": "code-atlas", "root_path": str(tmp_path / "base-gone")}},
                {"n": {"project_name": "code-atlas@feature-x", "root_path": str(tmp_path / "gone")}},
            ]
        )
        calls: list[Path] = []
        monkeypatch.setattr(orchestrator_module, "_git_worktree_list", calls.append)

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []
        assert calls == []

    async def test_spares_worktree_project_when_git_worktree_list_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """'git worktree list' fails or git is unavailable (returns None) — not corroborated, skip."""
        graph = FakeGCGraph(
            [
                {"n": {"project_name": "code-atlas", "root_path": str(tmp_path)}},
                {"n": {"project_name": "code-atlas@feature-x", "root_path": str(tmp_path / "gone")}},
            ]
        )
        monkeypatch.setattr(orchestrator_module, "_git_worktree_list", lambda base_root: None)

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []

    async def test_spares_worktree_project_when_git_still_lists_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """git still lists the vanished path as a live worktree — transient unavailability, not a real removal."""
        vanished = tmp_path / "gone"
        graph = FakeGCGraph(
            [
                {"n": {"project_name": "code-atlas", "root_path": str(tmp_path)}},
                {"n": {"project_name": "code-atlas@feature-x", "root_path": str(vanished)}},
            ]
        )
        monkeypatch.setattr(
            orchestrator_module, "_git_worktree_list", lambda base_root: [tmp_path.resolve(), vanished.resolve()]
        )

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []

    async def test_empty_project_list_is_noop(self):
        graph = FakeGCGraph([])

        removed = await gc_vanished_worktree_projects(graph)  # type: ignore[arg-type]

        assert removed == []
        assert graph.deleted == []


# ---------------------------------------------------------------------------
# Dependency manifest parsing
# ---------------------------------------------------------------------------


@pytest.fixture
def manifest_registry():
    """Snapshot/restore the manifest parser registry so extension tests stay isolated."""
    original = dict(orchestrator_module._MANIFEST_PARSERS)
    yield orchestrator_module._MANIFEST_PARSERS
    orchestrator_module._MANIFEST_PARSERS.clear()
    orchestrator_module._MANIFEST_PARSERS.update(original)


class TestDependencyManifests:
    """One realistic sample per supported manifest format, parsed through the registry."""

    def test_pyproject_toml(self, tmp_path: Path):
        _write(
            tmp_path,
            "pyproject.toml",
            """
[project]
name = "demo"
requires-python = ">=3.12"
dependencies = [
    "loguru",
    "pydantic>=2.0",
    "PyYAML~=6.0",
    "python-dotenv==1.0.1",
    "requests[socks]>=2.31",
]

[dependency-groups]
dev = ["pytest>=8.0"]
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "pydantic": ">=2.0",
            "pyyaml": "~=6.0",
            "python_dotenv": "==1.0.1",
            "requests": "[socks]>=2.31",
        }
        # Unpinned deps carry no constraint, and [dependency-groups] is not read.
        assert "loguru" not in result
        assert "pytest" not in result

    def test_package_json(self, tmp_path: Path):
        _write(
            tmp_path,
            "package.json",
            """
{
  "name": "@acme/web",
  "private": true,
  "dependencies": {
    "react": "^18.3.1",
    "@tanstack/react-query": "^5.36.0",
    "lodash.debounce": "^4.0.8"
  },
  "devDependencies": {
    "typescript": "~5.4.5",
    "react": "^18.0.0"
  },
  "peerDependencies": {
    "react-dom": ">=18"
  }
}
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "react": "^18.3.1",  # runtime range wins over the devDependencies echo
            "@tanstack/react-query": "^5.36.0",
            "lodash.debounce": "^4.0.8",
            "typescript": "~5.4.5",
            "react-dom": ">=18",
        }

    def test_cargo_toml(self, tmp_path: Path):
        _write(
            tmp_path,
            "Cargo.toml",
            """
[package]
name = "demo"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tracing-subscriber = "0.3"
local-lib = { path = "../local-lib" }

[dev-dependencies]
criterion = "0.5"

[build-dependencies]
cc = "1.0"
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "serde": "1.0",
            "serde_json": "1.0",
            "tracing_subscriber": "0.3",  # crate name, not the dashed package name
            "criterion": "0.5",
            "cc": "1.0",
        }
        # Path dependencies declare no version requirement.
        assert "local_lib" not in result

    def test_go_mod(self, tmp_path: Path):
        _write(
            tmp_path,
            "go.mod",
            "module github.com/example/demo\n"
            "\n"
            "go 1.22\n"
            "\n"
            "require (\n"
            "\tgithub.com/spf13/cobra v1.8.0\n"
            "\tgolang.org/x/text v0.14.0 // indirect\n"
            ")\n"
            "\n"
            "require github.com/stretchr/testify v1.9.0\n"
            "\n"
            "replace github.com/example/other => ../other\n"
            "\n"
            "exclude (\n"
            "\tgithub.com/bad/pkg v0.1.0\n"
            ")\n",
        )

        result = _parse_dependency_versions(tmp_path)

        # Keys are module paths verbatim — see the module-level note on why a Go
        # import root ("github") is not a usable ExternalPackage key.
        assert result == {
            "github.com/spf13/cobra": "v1.8.0",
            "golang.org/x/text": "v0.14.0",
            "github.com/stretchr/testify": "v1.9.0",
        }
        assert "github.com/bad/pkg" not in result
        assert "github.com/example/other" not in result

    def test_pom_xml(self, tmp_path: Path):
        _write(
            tmp_path,
            "pom.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.example</groupId>
  <artifactId>demo</artifactId>
  <version>1.0.0</version>
  <properties>
    <junit.version>5.10.2</junit.version>
  </properties>
  <dependencies>
    <dependency>
      <groupId>org.slf4j</groupId>
      <artifactId>slf4j-api</artifactId>
      <version>2.0.13</version>
    </dependency>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <version>${junit.version}</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>com.google.guava</groupId>
      <artifactId>guava</artifactId>
    </dependency>
  </dependencies>
</project>
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "org.slf4j:slf4j-api": "2.0.13",
            "org.junit.jupiter:junit-jupiter": "5.10.2",  # ${junit.version} resolved
        }
        # Version inherited from a parent pom / dependencyManagement we cannot see.
        assert "com.google.guava:guava" not in result

    def test_build_gradle_groovy(self, tmp_path: Path):
        _write(
            tmp_path,
            "build.gradle",
            """
plugins {
    id 'java'
}

repositories {
    mavenCentral()
    maven { url "https://repo.spring.io/milestone" }
}

dependencies {
    implementation 'com.google.guava:guava:33.1.0-jre'
    implementation "org.slf4j:slf4j-api:$slf4jVersion"
    testImplementation group: 'junit', name: 'junit', version: '4.13.2'
    runtimeOnly 'org.postgresql:postgresql:42.7.3'
    // implementation 'commented:out:1.0'
}
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "com.google.guava:guava": "33.1.0-jre",
            "junit:junit": "4.13.2",
            "org.postgresql:postgresql": "42.7.3",
        }
        # An interpolated version is a variable name, not a version.
        assert "org.slf4j:slf4j-api" not in result
        assert "commented:out" not in result

    def test_build_gradle_kts(self, tmp_path: Path):
        _write(
            tmp_path,
            "build.gradle.kts",
            """
dependencies {
    implementation(platform("org.springframework.boot:spring-boot-dependencies:3.2.5"))
    implementation("io.ktor:ktor-client-core:2.3.11")
    testImplementation("org.jetbrains.kotlin:kotlin-test:1.9.23")
}
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "org.springframework.boot:spring-boot-dependencies": "3.2.5",
            "io.ktor:ktor-client-core": "2.3.11",
            "org.jetbrains.kotlin:kotlin-test": "1.9.23",
        }

    def test_composer_json(self, tmp_path: Path):
        _write(
            tmp_path,
            "composer.json",
            """
{
  "name": "example/demo",
  "require": {
    "php": ">=8.2",
    "ext-json": "*",
    "monolog/monolog": "^3.6",
    "symfony/console": "^7.0"
  },
  "require-dev": {
    "phpunit/phpunit": "^11.0"
  }
}
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "monolog/monolog": "^3.6",
            "symfony/console": "^7.0",
            "phpunit/phpunit": "^11.0",
        }
        # Platform requirements are not packages.
        assert "php" not in result
        assert "ext-json" not in result

    def test_gemfile(self, tmp_path: Path):
        _write(
            tmp_path,
            "Gemfile",
            """
source "https://rubygems.org"

gem "rails", "~> 7.1.3"
gem "puma", ">= 6.0", "< 7.0"
gem "nokogiri"
gem "pg", "~> 1.5", require: false
gem "dotenv-rails", group: :development
# gem "commented", "1.0"

group :test do
  gem "rspec", "~> 3.13"
end
""",
        )

        result = _parse_dependency_versions(tmp_path)

        assert result == {
            "rails": "~> 7.1.3",
            "puma": ">= 6.0, < 7.0",
            "pg": "~> 1.5",
            "rspec": "~> 3.13",
        }
        # Gems declared without a requirement contribute nothing.
        assert "nokogiri" not in result
        assert "dotenv-rails" not in result
        assert "commented" not in result

    def test_unknown_manifest_filename_is_ignored(self, tmp_path: Path):
        """An unregistered manifest is skipped silently — not an error, not a guess."""
        _write(tmp_path, "requirements.txt", "requests==2.31.0\n")
        _write(tmp_path, "Pipfile", "[packages]\nflask = '*'\n")
        _write(tmp_path, "setup.py", "from setuptools import setup\n\nsetup(install_requires=['click'])\n")

        assert _parse_dependency_versions(tmp_path) == {}

    def test_empty_project_root(self, tmp_path: Path):
        assert _parse_dependency_versions(tmp_path) == {}

    def test_polyglot_root_merges_manifests(self, tmp_path: Path):
        _write(tmp_path, "pyproject.toml", '[project]\nname = "api"\ndependencies = ["fastapi>=0.111"]\n')
        _write(tmp_path, "package.json", '{"dependencies": {"react": "^18.3.1"}}')

        result = _parse_dependency_versions(tmp_path)

        assert result == {"fastapi": ">=0.111", "react": "^18.3.1"}

    def test_conflicting_name_across_manifests_is_dropped(self, tmp_path: Path):
        """One ExternalPackage node per name — two ecosystems disagreeing means no answer."""
        _write(tmp_path, "pyproject.toml", '[project]\nname = "api"\ndependencies = ["redis>=5.0", "httpx>=0.27"]\n')
        _write(tmp_path, "package.json", '{"dependencies": {"redis": "^4.6.0"}}')

        result = _parse_dependency_versions(tmp_path)

        assert result == {"httpx": ">=0.27"}

    def test_malformed_manifest_does_not_break_siblings(self, tmp_path: Path):
        _write(tmp_path, "package.json", "{ this is not json")
        _write(tmp_path, "pom.xml", "<project><dependencies>")
        _write(tmp_path, "Cargo.toml", "[dependencies\nbroken =")
        _write(tmp_path, "pyproject.toml", '[project]\nname = "api"\ndependencies = ["httpx>=0.27"]\n')

        assert _parse_dependency_versions(tmp_path) == {"httpx": ">=0.27"}

    def test_register_manifest_parser_extends_the_table(self, tmp_path: Path, manifest_registry):
        _write(tmp_path, "deps.txt", "leftpad 1.0.0\n")

        assert _parse_dependency_versions(tmp_path) == {}

        def _parse_deps_txt(text: str) -> dict[str, str]:
            versions: dict[str, str] = {}
            for line in text.splitlines():
                if line.strip():
                    name, constraint = line.split(maxsplit=1)
                    versions[name] = constraint
            return versions

        register_manifest_parser("deps.txt", _parse_deps_txt)

        assert "deps.txt" in manifest_registry
        assert _parse_dependency_versions(tmp_path) == {"leftpad": "1.0.0"}


# ---------------------------------------------------------------------------
# Consumer teardown
# ---------------------------------------------------------------------------


class TestStopConsumerTasks:
    """The AST consumer's ``finally`` runs the end-of-run resolution flush —
    deferred CALLS/IMPORTS, the withheld file-hash writes, and the citation
    retry sweep. Teardown has to let it finish."""

    async def test_a_final_flush_longer_than_the_old_budget_still_completes(self):
        """0.6s: longer than the flat ``sleep(0.5)`` this replaced, which used
        to cancel straight into the middle of the flush."""
        flushed = asyncio.Event()

        async def consumer() -> None:
            try:
                await asyncio.sleep(0)
            finally:
                await asyncio.sleep(0.6)
                flushed.set()

        task = asyncio.create_task(consumer())

        await _stop_consumer_tasks([task, None])

        assert flushed.is_set()
        assert not task.cancelled()

    async def test_a_straggler_is_still_cancelled_when_it_stops_progressing(self, monkeypatch):
        monkeypatch.setattr(orchestrator_module, "_CONSUMER_STALL_S", 0.05)
        monkeypatch.setattr(orchestrator_module, "_TEARDOWN_POLL_S", 0.01)
        task = asyncio.create_task(asyncio.sleep(30))

        finished = await _stop_consumer_tasks([task])

        assert task.cancelled()
        assert finished is False, "a truncated flush must be reported, not pass as success"

    async def test_a_slow_but_progressing_consumer_is_not_cancelled(self, monkeypatch):
        """The regression that made this a mechanism instead of a number.

        A final flush scales with the project, so it will always outgrow any fixed
        grace period. Twice, the cancel landed inside it and the whole-project
        conformance sweep was skipped in silence — IMPLEMENTS 258 -> 11.
        """
        monkeypatch.setattr(orchestrator_module, "_CONSUMER_STALL_S", 0.2)
        monkeypatch.setattr(orchestrator_module, "_TEARDOWN_POLL_S", 0.01)

        swept = asyncio.Event()

        class _Beating:
            """Heartbeats while it works, exactly as TierConsumer.note_progress does."""

            def __init__(self) -> None:
                self.progress_at = 0.0

        consumer = _Beating()

        async def work() -> None:
            # Ten steps at 0.1s each: 1s total, five times the stall window, but no
            # single gap exceeds it.
            for _ in range(10):
                await asyncio.sleep(0.1)
                consumer.progress_at = asyncio.get_event_loop().time()
            swept.set()

        task = asyncio.create_task(work())

        finished = await _stop_consumer_tasks([task], [consumer])

        assert swept.is_set(), "cancelled a consumer that was still doing real work"
        assert not task.cancelled()
        assert finished is True

    async def test_the_absolute_cap_still_bounds_a_forever_heartbeat(self, monkeypatch):
        """A consumer that heartbeats but never finishes must not wedge the CLI."""
        monkeypatch.setattr(orchestrator_module, "_CONSUMER_STALL_S", 30.0)
        monkeypatch.setattr(orchestrator_module, "_CONSUMER_TEARDOWN_CAP_S", 0.1)
        monkeypatch.setattr(orchestrator_module, "_TEARDOWN_POLL_S", 0.01)

        class _Beating:
            def __init__(self) -> None:
                self.progress_at = 0.0

        consumer = _Beating()

        async def forever() -> None:
            while True:
                await asyncio.sleep(0.01)
                consumer.progress_at = asyncio.get_event_loop().time()

        task = asyncio.create_task(forever())

        finished = await _stop_consumer_tasks([task], [consumer])

        assert task.cancelled()
        assert finished is False

    async def test_no_tasks_is_a_no_op(self):
        await _stop_consumer_tasks([None])


class TestGrammarGapReporting:
    """The scan records what it dropped for want of a grammar (ATL-110)."""

    def test_scan_records_files_no_grammar_could_read(self, tmp_path, monkeypatch):
        from code_atlas.indexing.orchestrator import FileScope

        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "b.ts").write_text("const x = 1", encoding="utf-8")
        (tmp_path / "c.ts").write_text("const y = 2", encoding="utf-8")

        # Simulate TypeScript's grammar not being installed. Capture the real function
        # BEFORE patching — resolving it through the module afterwards recurses.
        import code_atlas.indexing.orchestrator as orch

        real_lookup = orch.get_language_for_file

        def _no_ts(path: str):
            return None if path.endswith(".ts") else real_lookup(path)

        monkeypatch.setattr(orch, "get_language_for_file", _no_ts)

        scope = FileScope(tmp_path, _make_settings(tmp_path))
        files = scope.scan()

        assert "a.py" in files
        assert "b.ts" not in files
        assert scope.skipped_no_grammar == {".ts": 2}

    def test_the_counter_is_reset_per_scan(self, tmp_path, monkeypatch):
        """A scope is reusable, so a stale count must not outlive the scan it describes."""
        import code_atlas.indexing.orchestrator as orch
        from code_atlas.indexing.orchestrator import FileScope

        (tmp_path / "b.ts").write_text("const x = 1", encoding="utf-8")
        monkeypatch.setattr(orch, "get_language_for_file", lambda _p: None)

        scope = FileScope(tmp_path, _make_settings(tmp_path))
        scope.scan()
        assert scope.skipped_no_grammar == {".ts": 1}
        scope.scan()
        assert scope.skipped_no_grammar == {".ts": 1}, "counts must not accumulate across scans"
