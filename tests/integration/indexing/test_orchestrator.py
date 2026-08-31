"""Integration tests for the indexer/orchestrator module (require Memgraph + Valkey)."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from code_atlas import schema
from code_atlas.chunking import SplitResult
from code_atlas.events import Topic
from code_atlas.indexing.consumers import ASTConsumer
from code_atlas.indexing.orchestrator import StalenessChecker, index_monorepo, index_project
from code_atlas.schema import (
    _EMBEDDABLE_LABELS,
    FILE_HASH_LABELS,
    NodeLabel,
    RelType,
    generate_clear_file_hashes_ddl,
)
from code_atlas.settings import (
    AtlasSettings,
    EmbeddingSettings,
    IndexSettings,
    RationaleSettings,
    SearchSettings,
    derive_project_name,
)
from tests.conftest import NO_EMBED, TEST_DRAIN_TIMEOUT_S

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel_path: str, content: str = "") -> Path:
    """Write a file at root/rel_path, creating parent dirs."""
    p = root / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def parse_calls(monkeypatch):
    """Every path the AST stage genuinely handed to tree-sitter, in order.

    A run that skipped every file and a run that re-parsed every file and found nothing
    changed return the identical ``IndexResult``, so this is the only signal that
    separates "distrusted the gate" from "trusted it".

    Module-scoped rather than owned by one class: the flag split (axis B) and the gate
    key (ATL-152) both turn on exactly this distinction.
    """
    from code_atlas.indexing import consumers

    real = consumers.parse_file
    seen: list[str] = []

    def _counting(path, source, project_name, **kwargs):
        seen.append(path)
        return real(path, source, project_name, **kwargs)

    monkeypatch.setattr(consumers, "parse_file", _counting)
    return seen


def _git(cwd, *args):
    """Run a git command in cwd."""
    subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def _init_git_repo(tmp_path):
    """Initialise a git repo with an initial commit."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@test.com")
    _git(tmp_path, "config", "user.name", "Test")


async def _citation_edges(graph_client, project: str) -> list[tuple[str, str, str]]:
    """``(cited document path, citing entity name, citation)``, doc → code."""
    records = await graph_client.execute(
        f"MATCH (doc {{project_name: $p}})-[r:{RelType.DOCUMENTS} {{link_type: 'citation'}}]->(entity) "
        "RETURN doc.file_path AS doc_path, entity.name AS entity_name, r.citation AS citation "
        "ORDER BY doc_path, entity_name",
        {"p": project},
    )
    return [(r["doc_path"], r["entity_name"], r["citation"]) for r in records]


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


# ---------------------------------------------------------------------------
# Integration tests (require Memgraph + Valkey)
# ---------------------------------------------------------------------------


class TestIndexProjectIntegration:
    @pytest.fixture
    def project_dir(self, tmp_path):
        """Create a minimal Python project for indexing."""
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        return tmp_path

    async def test_index_project_creates_graph(self, project_dir, graph_client, event_bus):
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.files_scanned >= 2
        assert result.entities_total > 0
        assert result.duration_s > 0

    async def test_index_project_creates_hierarchy(self, project_dir, graph_client, event_bus):
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Project node exists
        projects = await graph_client.execute(f"MATCH (p:{NodeLabel.PROJECT}) RETURN p.name AS name")
        assert len(projects) >= 1

        # Package node for src/ exists
        packages = await graph_client.execute(
            f"MATCH (p:{NodeLabel.PACKAGE} {{project_name: $pn}}) RETURN p.qualified_name AS qn",
            {"pn": project_dir.name},
        )
        assert any(p["qn"] == "src" for p in packages)

        # CONTAINS edge exists: Project -> Package
        contains = await graph_client.execute("MATCH (a)-[:CONTAINS]->(b) RETURN a.uid AS from_uid, b.uid AS to_uid")
        assert len(contains) > 0

    async def test_index_project_full_reindex(self, project_dir, graph_client, event_bus):
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.entities_total > 0

        # Full reindex — should work cleanly
        r2 = await index_project(
            settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )
        assert r2.entities_total > 0

    async def test_index_project_error_resilience(self, tmp_path, graph_client, event_bus):
        """One unparseable file shouldn't abort the whole indexing run."""
        _write(tmp_path, "good.py", "x = 1\n")
        _write(tmp_path, "bad.py", "def (\n")  # syntax error — tree-sitter handles gracefully

        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Should still complete and index the good file
        assert result.files_scanned >= 2
        assert result.entities_total > 0


# ---------------------------------------------------------------------------
# Delta indexing — integration tests (require Memgraph + Valkey + git)
# ---------------------------------------------------------------------------


class TestDeltaIndexIntegration:
    @pytest.fixture
    def git_project(self, tmp_path):
        """Create a git-tracked Python project with initial commit.

        Uses 5 files so that modifying 1 file stays under the 30% delta
        threshold (1/5 = 20%).
        """
        _init_git_repo(tmp_path)
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        _write(tmp_path, "src/config.py", "DEBUG = False\n\ndef get_config():\n    return {}\n")
        _write(tmp_path, "src/models.py", "class User:\n    name: str\n    email: str\n")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        return tmp_path

    async def test_citation_resolves_when_only_the_adr_is_published(self, git_project, graph_client, event_bus):
        """The daemon's shape: an ADR written days after the code it explains.

        The citing file is unchanged, so delta mode publishes only the new
        document and the citing entity is never re-parsed — nothing but the
        retry sweep that rides on indexing a document can link it.
        """
        _write(git_project, "src/mod.py", "# WHY: retry cascade documented in ADR-14.\ndef resolve():\n    return 1\n")
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "cite an adr")
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(git_project)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert await _citation_edges(graph_client, project) == []

        _write(git_project, "wiki/adr/0014-calls.md", "# ADR-0014: CALLS Edge Confidence\n\nBody.\n")
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "write the adr")
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert r2.files_published == 1, "only the ADR should be republished"
        assert await _citation_edges(graph_client, project) == [("wiki/adr/0014-calls.md", "resolve", "ADR-14")]

    async def test_delta_index_mode(self, git_project, graph_client, event_bus):
        """Re-indexing without changes uses delta mode."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index — full mode
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.mode == "full"
        assert r1.entities_total > 0

        # Re-index without changes — delta mode, 0 published
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.files_published == 0
        assert r2.entities_total == r1.entities_total

    async def test_delta_index_publishes_only_changed(self, git_project, graph_client, event_bus):
        """Modifying one file only publishes that file in delta mode."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.mode == "full"

        # Modify one file and commit
        _write(git_project, "src/app.py", 'def hello():\n    """Say hello!""""\n    return "hello world"\n')
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "modify app")

        # Delta re-index
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_modified >= 1
        assert r2.files_published >= 1

    async def test_delta_index_detects_new_files(self, git_project, graph_client, event_bus):
        """New files are picked up in delta mode."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Add new file and commit
        _write(git_project, "src/new_module.py", "NEW_CONST = 99\n")
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "add new module")

        # Delta re-index
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_added >= 1

    async def test_delta_index_handles_deletion(self, git_project, graph_client, event_bus):
        """Deleted files' entities are removed from the graph."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        e1 = r1.entities_total

        # Delete a file and commit
        (git_project / "src" / "utils.py").unlink()
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "remove utils")

        # Delta re-index
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_deleted >= 1
        # Entity count should have dropped
        assert r2.entities_total < e1

    async def test_reindex_reconciles_when_scan_finds_zero_files(self, git_project, graph_client, event_bus):
        """All source files removed: the graph must be reconciled, not left stale.

        Before the fix, index_project returned early on an empty scan,
        silently keeping every previously indexed entity and never updating
        Project metadata.
        """
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project_name = derive_project_name(git_project)

        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.entities_total > 0

        # Delete every source file and commit — the next scan finds nothing.
        for f in (git_project / "src").iterdir():
            f.unlink()
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "remove all sources")

        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert r2.files_scanned == 0
        assert r2.mode == "delta"
        assert r2.delta_stats is not None
        assert r2.delta_stats.files_deleted == 5
        # Entities reconciled (dropped), not silently left stale.
        assert r2.entities_total < r1.entities_total

        # Project metadata must reflect the empty state, not be skipped.
        projects = await graph_client.execute(
            f"MATCH (p:{NodeLabel.PROJECT} {{uid: $pn}}) RETURN p.file_count AS fc",
            {"pn": project_name},
        )
        assert len(projects) == 1
        assert projects[0]["fc"] == 0

    async def test_empty_scan_not_corroborated_by_git_skips_reconciliation(self, git_project, graph_client, event_bus):
        """A zero-file scan that git does NOT corroborate must not wipe the graph.

        Simulates a transient/misconfiguration scenario (e.g. the CI race of
        `rm -rf src && git checkout src`, or an unmounted path): the files are
        gone from disk but git's index still tracks them (nothing was staged
        or committed). The scan legitimately finds zero files, but this must
        NOT be treated as a genuine deletion.
        """
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r1.entities_total > 0

        # Remove every source file from disk WITHOUT staging/committing —
        # git's index still lists them as tracked.
        for f in (git_project / "src").iterdir():
            f.unlink()

        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert r2.files_scanned == 0
        # No reconciliation happened — entities must be untouched, not wiped.
        assert r2.entities_total == r1.entities_total

    async def test_delta_index_full_fallback_on_threshold(self, git_project, graph_client, event_bus):
        """Exceeding the threshold triggers full mode."""
        # Set a very low threshold so any change exceeds it
        settings = AtlasSettings(
            project_root=git_project, index=IndexSettings(delta_threshold=0.0), embeddings=NO_EMBED
        )
        await graph_client.ensure_schema()

        # First index
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Modify a file and commit
        _write(git_project, "src/app.py", 'def hello():\n    return "changed"\n')
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "change")

        # Re-index with threshold=0.0 — should fall back to full
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "full"

    async def test_delta_index_preserves_unchanged(self, git_project, graph_client, event_bus):
        """Unchanged entities keep their entity count after delta re-index."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # First index
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Re-index without changes
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert r2.mode == "delta"
        assert r2.entities_total == r1.entities_total

    async def test_full_reindex_flag_overrides_delta(self, git_project, graph_client, event_bus):
        """--full flag forces full mode even when delta is available."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        r2 = await index_project(
            settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )
        assert r2.mode == "full"


# ---------------------------------------------------------------------------
# Pipeline durability — integration tests (require Memgraph + Valkey + git)
# ---------------------------------------------------------------------------


class TestPipelineDurabilityIntegration:
    async def test_drain_timeout_does_not_advance_git_hash(self, tmp_path, graph_client, event_bus):
        """S7(f): a timed-out drain must not advance git_hash — the next run retries the delta.

        Before the fix the timeout only logged a warning, git_hash advanced to
        HEAD anyway, and the IndexResult carried no failure signal.
        """
        _init_git_repo(tmp_path)
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    return "hello"\n')
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project_name = derive_project_name(tmp_path)

        # The pipeline's drain settle window (_DRAIN_SETTLE_S, shrunk to 0.1s for this
        # suite) still makes draining within 0.01s impossible -- a 10x margin, not the
        # 200x the unpatched 2.0s gave. Raise drain_timeout_s here if that ever tightens.
        r1 = await index_project(settings, graph_client, event_bus, drain_timeout_s=0.01)

        assert r1.drained is False
        assert await graph_client.get_project_git_hash(project_name) is None

        # A follow-up run with a normal timeout processes the files, THEN advances git_hash
        r2 = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert r2.drained is True
        assert r2.entities_total > 0
        assert await graph_client.get_project_git_hash(project_name) == _get_head(tmp_path)

    async def test_reset_preserves_foreign_consumer_group(self, tmp_path, graph_client, event_bus):
        """S7(e): a destructive reindex must not destroy consumer groups a live daemon depends on.

        Before the fix ``bus.flush()`` destroyed every consumer group on the
        pipeline streams, permanently killing a concurrently running daemon's
        consumers.

        ``reset``, not ``full_reindex``: since ADR-0042 the flush is the one thing
        ``--reset`` does and ``--full`` does not, because the streams are shared and a
        run that deletes nothing has nothing to discard. Left on ``full_reindex=True``
        this test went green by never flushing at all -- passing for the absence of the
        very call it exists to guard.
        """
        _write(tmp_path, "app.py", "x = 1\n")
        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        # Simulate a live daemon's consumer group on the FileChanged stream
        await event_bus.ensure_group(Topic.FILE_CHANGED, "daemon-sim")
        key = event_bus._stream_key(Topic.FILE_CHANGED)

        try:
            await index_project(settings, graph_client, event_bus, reset=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

            groups = await event_bus._redis.xinfo_groups(key)
            names = set()
            for g in groups:
                name = g.get(b"name", g.get("name", b""))
                names.add(name.decode() if isinstance(name, bytes) else name)
            assert "daemon-sim" in names
        finally:
            await event_bus._redis.xgroup_destroy(key, "daemon-sim")


# ---------------------------------------------------------------------------
# Monorepo scoping/package-hierarchy — integration tests (require Memgraph + Valkey)
# ---------------------------------------------------------------------------


class TestIndexMonorepoScopingIntegration:
    async def test_scoped_monorepo_excludes_unscoped_project_files_from_root(self, graph_client, event_bus, tmp_path):
        """scope_projects filtering must not leak an excluded sub-project's files into the root project.

        Before the fix, ``sub_paths`` was computed from the scope-FILTERED
        sub-project list, so files belonging to an excluded sub-project were
        misclassified as root-only files and indexed under the bare
        root project_name — duplicating the excluded sub-project's entities
        under a different uid namespace.
        """
        _write(tmp_path, "services/auth/pyproject.toml", '[project]\nname = "auth"\n')
        _write(tmp_path, "services/auth/auth/__init__.py", "")
        _write(tmp_path, "services/auth/auth/service.py", "def authenticate():\n    return True\n")

        _write(tmp_path, "libs/shared/pyproject.toml", '[project]\nname = "shared"\n')
        _write(tmp_path, "libs/shared/shared/__init__.py", "")
        _write(tmp_path, "libs/shared/shared/utils.py", "def validate():\n    return True\n")

        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        root_name = tmp_path.resolve().name

        await index_monorepo(
            settings, graph_client, event_bus, scope_projects=["auth"], drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )

        # The excluded 'shared' sub-project must never be published under the
        # bare root project_name.
        root_callables = await graph_client.execute(
            f"MATCH (c:{NodeLabel.CALLABLE} {{project_name: $pn}}) RETURN c.name AS name",
            {"pn": root_name},
        )
        assert root_callables == []

    async def test_root_package_hierarchy_excludes_sub_project_dirs(self, graph_client, event_bus, tmp_path):
        """The root project's package hierarchy must not reach into sub-project directories.

        Before the fix, ``_create_package_hierarchy(root_name, project_root)``
        walked the ENTIRE monorepo tree, creating Package nodes for the
        sub-project's ``__init__.py`` files under the ROOT project_name —
        churned (created then deleted every delta run) since the sub-project's
        own files are never part of the root's current file set.
        """
        _write(tmp_path, "libs/shared/pyproject.toml", '[project]\nname = "shared"\n')
        _write(tmp_path, "libs/shared/shared/__init__.py", "")
        _write(tmp_path, "libs/shared/shared/utils.py", "def validate():\n    return True\n")

        # A genuine root-level file so the root project actually gets published.
        _write(tmp_path, "tools/run.py", "def main():\n    pass\n")

        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        root_name = tmp_path.resolve().name

        await index_monorepo(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        root_packages = await graph_client.execute(
            f"MATCH (p:{NodeLabel.PACKAGE} {{project_name: $pn}}) RETURN p.file_path AS fp",
            {"pn": root_name},
        )
        assert not any(r["fp"].startswith("libs/shared") for r in root_packages)


# ---------------------------------------------------------------------------
# Staleness check — integration tests (require Memgraph + Valkey + git)
# ---------------------------------------------------------------------------


class TestStalenessCheckIntegration:
    @pytest.fixture
    def git_project(self, tmp_path):
        """Create a git-tracked Python project with initial commit."""
        _init_git_repo(tmp_path)
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        return tmp_path

    async def test_not_stale_when_hashes_match(self, git_project, graph_client, event_bus):
        """After indexing, the checker reports not stale."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        checker = StalenessChecker(git_project)
        info = await checker.check(graph_client)

        assert info.stale is False
        assert info.current_commit is not None
        assert info.last_indexed_commit is not None

    async def test_stale_when_new_commit(self, git_project, graph_client, event_bus):
        """A new commit after indexing makes the checker report stale."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # Make a new commit
        _write(git_project, "src/new.py", "x = 1\n")
        _git(git_project, "add", ".")
        _git(git_project, "commit", "-m", "new file")

        checker = StalenessChecker(git_project)
        info = await checker.check(graph_client)

        assert info.stale is True
        assert info.changed_files  # should list at least src/new.py

    async def test_not_stale_non_git_dir(self, tmp_path, graph_client):
        """Non-git directory is never stale."""
        checker = StalenessChecker(tmp_path)
        info = await checker.check(graph_client)

        assert info.stale is False

    async def test_stale_never_indexed(self, git_project, graph_client):
        """Git project with no stored hash is stale."""
        await graph_client.ensure_schema()

        checker = StalenessChecker(git_project)
        info = await checker.check(graph_client)

        assert info.stale is True
        assert info.last_indexed_commit is None
        assert info.current_commit is not None


class TestManifestVersionsIntegration:
    """End-to-end proof of which manifest → ExternalPackage joins actually land."""

    async def _external_versions(self, graph_client, project_root: Path) -> dict[str, str | None]:
        """Every ExternalPackage the index created, with the version its project pinned.

        The OPTIONAL MATCH is what makes ``None`` mean "no manifest entry joined" rather
        than "no such package": since v18 the version is a property of the
        ``Project -[DEPENDS_ON]->`` edge, so a package with no manifest entry has no edge
        at all, and an inner match would drop it from this map entirely — which is the
        answer the go.mod case below is specifically asserting the shape of.
        """
        rows = await graph_client.execute(
            f"MATCH (p:{NodeLabel.EXTERNAL_PACKAGE} {{project_name: $pn}}) "
            f"OPTIONAL MATCH (:{NodeLabel.PROJECT} {{uid: $pn}})-[d:{RelType.DEPENDS_ON}]->(p) "
            "RETURN p.name AS name, d.version AS version",
            {"pn": derive_project_name(project_root)},
        )
        return {r["name"]: r["version"] for r in rows}

    async def test_package_json_versions_land_on_external_packages(self, tmp_path, graph_client, event_bus):
        """An npm name *is* the import specifier root, scope included — so the version joins."""
        _write(
            tmp_path,
            "package.json",
            '{"name": "web", "dependencies": {"react": "^18.3.1", "@tanstack/react-query": "^5.36.0"}}',
        )
        _write(
            tmp_path,
            "src/app.ts",
            'import { useState } from "react";\n'
            'import { QueryClient } from "@tanstack/react-query";\n'
            "\n"
            "export function App() {\n"
            "  return useState(new QueryClient());\n"
            "}\n",
        )
        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        versions = await self._external_versions(graph_client, tmp_path)
        assert versions.get("react") == "^18.3.1"
        assert versions.get("@tanstack/react-query") == "^5.36.0"

    async def test_go_mod_version_is_not_stamped_on_the_shared_import_root(self, tmp_path, graph_client, event_bus):
        """A Go import root is a hosting domain, so go.mod deliberately joins nothing.

        ``github.com/spf13/cobra`` collapses to an ExternalPackage named ``github``
        that aggregates every GitHub-hosted module. Stamping one module's version
        onto it would be a wrong mapping, so the parser emits the module path
        verbatim and the write is a no-op.

        Since v18 the no-op is stronger than it was: nothing joined means no
        ``DEPENDS_ON`` edge exists at all, rather than a node whose ``version``
        property was never set.
        """
        _write(tmp_path, "go.mod", "module example.com/demo\n\ngo 1.22\n\nrequire github.com/spf13/cobra v1.8.0\n")
        _write(
            tmp_path,
            "main.go",
            'package main\n\nimport "github.com/spf13/cobra"\n\nfunc main() {\n\tvar c cobra.Command\n\t_ = c\n}\n',
        )
        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        versions = await self._external_versions(graph_client, tmp_path)
        assert "github" in versions, "the collapsed aggregate node should still be created by import resolution"
        assert versions["github"] is None


class TestEmbeddingReconciliation:
    """A lost embedding must not be permanent.

    Measured on the live index: 144 entities across 4 files kept `embedding IS NULL`
    through a subsequent FULL re-index, because their EmbedDirty was poison-parked
    during a Memgraph outage and both AST-stage gates are content-based — an unchanged
    file is never re-parsed, and in delta mode never even published, so the
    `has_embedding` check that would have caught it is unreachable.
    """

    @pytest.fixture
    def project_dir(self, tmp_path):
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        return tmp_path

    async def test_find_unembedded_entities_sees_only_searchable_labels(
        self, project_dir, graph_client, event_bus
    ) -> None:
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        missing = await graph_client.find_unembedded_entities(project)
        assert missing, "embeddings were disabled, so every embeddable entity should be reported"

        # ExternalPackage/ExternalSymbol have no vector index — re-embedding them would
        # buy a vector nothing can search, so they must stay out of the reconcile set.
        from code_atlas.schema import _EMBEDDABLE_LABELS

        embeddable = {lbl.value for lbl in _EMBEDDABLE_LABELS}
        assert {label for _uid, label, _fp in missing} <= embeddable

        # uid, not qualified_name: the embed consumer feeds this field straight into
        # read_entity_texts(uids=...), so a bare qualified name matches nothing and the
        # batch silently completes having embedded zero entities.
        assert all(uid.startswith(f"{project}:") for uid, _label, _fp in missing)

    async def test_reconcile_requeues_an_entity_whose_embed_was_lost(
        self, project_dir, graph_client, event_bus
    ) -> None:
        from code_atlas.events import EmbedDirty, decode_event
        from code_atlas.indexing.orchestrator import _reconcile_missing_embeddings

        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        project = derive_project_name(project_dir)
        await graph_client.ensure_schema()
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        unembedded = {uid for uid, _, _ in await graph_client.find_unembedded_entities(project)}
        assert unembedded

        await event_bus.ensure_group(Topic.EMBED_DIRTY, "reconcile-test")
        # Returns the uids it re-queued, not a count: the caller loops until a pass
        # re-queues the same set twice, which is how it tells "still working" from
        # "these will never embed" without paying for the second one.
        queued = await _reconcile_missing_embeddings(graph_client, event_bus, [project])
        assert queued == unembedded

        published = await event_bus.read_batch(Topic.EMBED_DIRTY, "reconcile-test", "c1", count=1, block_ms=500)
        assert published
        event = decode_event(Topic.EMBED_DIRTY, published[0][1])
        assert isinstance(event, EmbedDirty)
        # The republished ref must round-trip through the same lookup the consumer uses.
        props = await graph_client.read_entity_texts([event.entity.qualified_name])
        assert props, "republished ref did not resolve — the consumer would no-op on it"


# ---------------------------------------------------------------------------
# Reindex scope vs. destruction — ADR-0042 / ATL-148 (require Memgraph + Valkey)
# ---------------------------------------------------------------------------


_FAKE_MODEL = "atlas-test/fake-embed"
"""Recorded as the project's embedding model, so the per-project model lock sees the
same value across the several index runs each test below performs and never decides
it has vectors to clear on its own."""


class _CountingEmbedClient:
    """A stand-in provider that records what it was asked to embed.

    Spend is asserted on this counter and never on elapsed time. The claim ADR-0042
    makes is that a ``--full`` re-check is free *in provider calls*; a timing assertion
    would go green on a fast machine that billed for every entity in the project.

    Implements only the surface the index path actually touches: the async-context
    protocol (``index_project`` enters the client on an exit stack), ``detect_dimension``
    for the dimension probe, the three attributes ``EmbedConsumer`` reads in its
    constructor, and ``split_text`` — part of the client contract since ATL-140, and a
    fake without it stalls the consumer in a retry loop that reads as a hang.
    """

    def __init__(self, model: str, dimension: int) -> None:
        self.max_concurrency = 1
        self.batch_size = 32
        self.configured_model = model
        self._dimension = dimension
        self.calls = 0
        self.texts: list[str] = []

    async def __aenter__(self) -> _CountingEmbedClient:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False

    async def detect_dimension(self) -> int:
        return self._dimension

    def split_text(self, text: str) -> SplitResult:
        return SplitResult([text], False, 0)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts.extend(texts)
        return [[0.1] * self._dimension for _ in texts]


def _embedding_settings(dimension: int) -> EmbeddingSettings:
    """Embeddings on, pinned to the vector indices' dimension and to the fake model."""
    return EmbeddingSettings(model=_FAKE_MODEL, dimension=dimension)


@pytest.fixture
def provider(graph_client):
    """Patch the embedding provider into both orchestrator entry points.

    Module-scoped for the same reason ``parse_calls`` is: ADR-0042's claim that a
    re-check costs nothing at the provider is made by the flag split (axis B) and again
    by the gate key (ATL-152), and the two reach the re-parse by different routes.
    """
    embed = _CountingEmbedClient(_FAKE_MODEL, graph_client._dimension)
    with patch("code_atlas.indexing.orchestrator.EmbedClient", return_value=embed):
        yield embed


async def _blast_radius(graph_client, project: str) -> dict[str, int]:
    """The row ``count_project_data`` reports for *project* itself.

    Deliberately the same read the destructive preflight prints, so a test asserting
    "nothing was removed" asserts on the number a user would have been shown.
    """
    rows = await graph_client.count_project_data(project)
    row = next((r for r in rows if r["name"] == project), None)
    # Not a convenience default: ADR-0042 makes "nothing there" and "the count failed"
    # mean opposite things, so a missing row is a failure and must say so rather than
    # surfacing as `RuntimeError: coroutine raised StopIteration`.
    assert row is not None, f"count_project_data reported no row for {project!r}"
    return row


async def _file_hashes(graph_client, project: str) -> dict[str, str | None]:
    """``file_path -> file_hash`` across every label the gate stores one on.

    ``None`` for a node carrying no hash — the state a schema migration's
    ``generate_clear_file_hashes_ddl`` leaves behind, and the state a gate-distrusting
    run that stopped *writing* hashes when it stopped *reading* them would leave behind
    for every file it parsed.
    """
    out: dict[str, str | None] = {}
    for label in FILE_HASH_LABELS:
        records = await graph_client.execute(
            f"MATCH (n:{label} {{project_name: $p}}) RETURN n.file_path AS fp, n.file_hash AS fh",
            {"p": project},
        )
        out.update({r["fp"]: r["fh"] for r in records})
    return out


async def _entities_for_file(graph_client, project: str, file_path: str) -> set[str]:
    """Every entity uid the graph still holds for *file_path*, node label included."""
    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.ENTITY} {{project_name: $p, file_path: $fp}}) RETURN n.uid AS uid",
        {"p": project, "fp": file_path},
    )
    return {r["uid"] for r in records}


async def _mark_module(graph_client, project: str, file_path: str) -> None:
    """Stamp a property no writer sets onto the Module node for *file_path*."""
    await graph_client.execute_write(
        f"MATCH (n:{NodeLabel.MODULE} {{project_name: $p, file_path: $fp}}) SET n.atl148_marker = 'survived'",
        {"p": project, "fp": file_path},
    )


async def _module_mark(graph_client, project: str, file_path: str) -> str | None:
    """The mark, or ``None`` once the node it sat on has been destroyed and rebuilt.

    Counts alone cannot separate a re-check from a delete-and-rebuild: both end with
    the same numbers, which is exactly why the old ``--full`` could be swapped for the
    new one and every count-based test would still pass.
    """
    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.MODULE} {{project_name: $p, file_path: $fp}}) RETURN n.atl148_marker AS m",
        {"p": project, "fp": file_path},
    )
    return records[0]["m"] if records else None


async def _entity_source(graph_client, project: str, name: str) -> str:
    """The stored ``source`` of the single Callable called *name*."""
    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: $p, name: $n}}) RETURN n.source AS source",
        {"p": project, "n": name},
    )
    assert len(records) == 1, f"expected exactly one callable named {name}, got {len(records)}"
    return records[0]["source"] or ""


async def _searchable_vectors(graph_client, project: str) -> int:
    """How many of *project*'s vectors a vector index actually serves.

    Deliberately narrower than ``count_project_data``'s ``embedded_nodes``, which counts
    every vector a destructive run would remove. Package and DocFile nodes are given a
    vector by the AST stage but have no vector index, so ``find_unembedded_entities``
    excludes them on purpose and a clear is one-way for exactly those two labels. This
    is the number that has to come back.
    """
    labels = "|".join(sorted(lbl.value for lbl in _EMBEDDABLE_LABELS))
    records = await graph_client.execute(
        f"MATCH (n:{labels}) WHERE n.project_name = $p AND n.embedding IS NOT NULL RETURN count(n) AS c",
        {"p": project},
    )
    return records[0]["c"]


async def _content_hashes(graph_client, project: str) -> dict[str, str]:
    """``name -> content_hash`` for the project's callables — ADR-0042's layer 2."""
    records = await graph_client.execute(
        f"MATCH (n:{NodeLabel.CALLABLE} {{project_name: $p}}) RETURN n.name AS name, n.content_hash AS h",
        {"p": project},
    )
    return {r["name"]: r["h"] for r in records}


class TestReindexScopeIsNotDestruction:
    """ADR-0042's three axes, one flag each: ``--full`` re-checks, ``--reset`` destroys.

    Every project here is deliberately **not** a git repo, so ``_decide_delta_mode``
    returns ``full`` on every run and enumeration is never the variable under test. What
    separates ``atlas index`` from ``atlas index --full`` in these tests is then exactly
    one thing — whether the ``file_hash`` gate is trusted — which is axis B, the axis
    that had no flag at all before this epic.
    """

    @pytest.fixture
    def project_dir(self, tmp_path):
        """A minimal Python project: three files, one of them a package marker."""
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        return tmp_path

    # -- axis C: --full destroys nothing ------------------------------------ #

    async def test_full_on_an_unchanged_project_costs_no_provider_call(
        self, project_dir, graph_client, event_bus, provider, parse_calls
    ):
        """The headline claim: re-checking everything is free when nothing changed.

        Cost ladders down four layers and money only enters at the bottom. ``--full``
        skips layer 1 (``file_hash``) by design; layers 2-4 (``content_hash``,
        ``embed_hash``, the ``(embed_hash, embed_model)`` dedup) are untouched and stop
        the run before it reaches the provider.
        """
        settings = AtlasSettings(project_root=project_dir, embeddings=_embedding_settings(graph_client._dimension))
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        embedded = (await graph_client.count_embeddings_by_project()).get(project, 0)
        assert embedded > 0, "the first index has to buy vectors, or a free re-check proves nothing"

        provider.calls = 0
        parse_calls.clear()
        result = await index_project(
            settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )

        assert result.mode == "full"
        # Non-vacuous: a run that skipped every file would also report zero calls, for
        # entirely the wrong reason.
        assert parse_calls, "--full must distrust the gate and re-parse, or the zero below means nothing"
        assert provider.calls == 0
        assert (await graph_client.count_embeddings_by_project()).get(project, 0) == embedded

    async def test_full_deletes_nothing(self, project_dir, graph_client, event_bus, parse_calls):
        """``--full`` re-reads and re-parses every file and removes no node or edge."""
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        await _mark_module(graph_client, project, "src/app.py")
        before = await _blast_radius(graph_client, project)
        assert before["nodes"] > 0
        assert before["relationships"] > 0

        parse_calls.clear()
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        after = await _blast_radius(graph_client, project)
        assert parse_calls
        assert (after["nodes"], after["relationships"]) == (before["nodes"], before["relationships"])
        # The counts above survive a delete-and-rebuild too. This does not.
        assert await _module_mark(graph_client, project, "src/app.py") == "survived"

    async def test_full_still_reconciles_a_file_deleted_from_disk(
        self, project_dir, graph_client, event_bus, parse_calls
    ):
        """Destroying nothing must not mean reconciling nothing.

        ``_publish_events``' full branch emits one ``created`` per file that exists, so
        nothing tells the AST stage about a file deleted since the last index. Deleting
        the whole project first used to cover that; once ``--full`` stopped doing so, a
        deleted or renamed file would keep its entities, edges and vectors forever —
        and the hash gate guarantees no later run revisits them. Running ``--full``
        again would not help, which is what makes it permanent rather than merely late.
        """
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert await _entities_for_file(graph_client, project, "src/utils.py")

        (project_dir / "src" / "utils.py").unlink()
        parse_calls.clear()
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert not await _entities_for_file(graph_client, project, "src/utils.py")
        # The surviving files are untouched — this reconciles the deletion, it does not
        # fall back to rebuilding the project.
        assert await _entities_for_file(graph_client, project, "src/app.py")

    async def test_reset_deletes_and_rebuilds(self, project_dir, graph_client, event_bus, parse_calls):
        """``--reset`` is the old ``--full``: the project's data goes, then comes back."""
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        await _mark_module(graph_client, project, "src/app.py")
        before = await _blast_radius(graph_client, project)

        parse_calls.clear()
        result = await index_project(
            settings, graph_client, event_bus, reset=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )

        after = await _blast_radius(graph_client, project)
        assert result.mode == "full"
        assert parse_calls
        assert (after["nodes"], after["relationships"]) == (before["nodes"], before["relationships"])
        # Same numbers, different nodes — which is the whole difference between the two
        # flags, and the reason this pair of tests is written as a pair.
        assert await _module_mark(graph_client, project, "src/app.py") is None

    # -- axis B: distrusting the gate, without breaking it ------------------ #

    async def test_full_repopulates_file_hash_for_every_file_it_parsed(
        self, project_dir, graph_client, event_bus, parse_calls
    ):
        """Distrusting the gate must not stop the run writing to it.

        ADR-0042 states it as a consequence worth naming: the first ``--full`` after an
        upgrade is also what populates ``file_hash`` for files and labels that never had
        one. The obvious implementation of axis B — compute the new hashes only on the
        branch that also compares them — leaves a ``--full`` writing no hash at all, so
        every following delta run re-parses the same files forever.

        Clearing the hashes first is what makes this specific to ``--full``: it is the
        state twelve schema migrations deliberately create with
        ``generate_clear_file_hashes_ddl`` to force exactly this run.
        """
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        baseline = await _file_hashes(graph_client, project)
        assert baseline, "the ordinary path has to store hashes first"
        assert all(baseline.values())

        await graph_client.execute_write(generate_clear_file_hashes_ddl())
        assert not any((await _file_hashes(graph_client, project)).values())

        parse_calls.clear()
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert parse_calls
        assert await _file_hashes(graph_client, project) == baseline

    async def test_full_rewrites_exactly_what_a_config_change_moved(
        self, tmp_path, graph_client, event_bus, provider, parse_calls
    ):
        """The case the epic exists for: bytes unchanged, extraction changed.

        Widening ``[rationale] markers`` makes the parser attach a ``# WHY:`` comment to
        the callable below it. No file's bytes moved, so the gate would skip everything;
        ``--full`` distrusts it, re-parses, and layer 2 (``content_hash``) then decides
        that exactly one entity is different.

        The provider is still never asked about the entity that did not move — which is
        the ladder working, not an accident of this fixture.
        """
        _write(
            tmp_path,
            "src/mod.py",
            "# WHY: the retry cascade needs a wider window.\n"
            "def widened():\n"
            "    return 1\n"
            "\n"
            "def untouched():\n"
            "    return 2\n",
        )
        narrow = AtlasSettings(
            project_root=tmp_path,
            rationale=RationaleSettings(markers=["NOTE"]),
            embeddings=_embedding_settings(graph_client._dimension),
        )
        wide = AtlasSettings(
            project_root=tmp_path,
            rationale=RationaleSettings(markers=["NOTE", "WHY"]),
            embeddings=_embedding_settings(graph_client._dimension),
        )
        await graph_client.ensure_schema()
        project = derive_project_name(tmp_path)

        await index_project(narrow, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        before = await _content_hashes(graph_client, project)
        assert set(before) == {"widened", "untouched"}

        provider.texts.clear()
        parse_calls.clear()
        await index_project(wide, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert "src/mod.py" in parse_calls
        after = await _content_hashes(graph_client, project)
        assert after["widened"] != before["widened"], "the entity the config change reached must be rewritten"
        assert after["untouched"] == before["untouched"], "and nothing else may be"
        assert not [t for t in provider.texts if "untouched" in t], "an unmoved entity must not be re-embedded"

    async def test_full_does_not_repair_a_source_truncated_under_a_narrower_cap(
        self, tmp_path, graph_client, event_bus, parse_calls
    ):
        """Raising ``index.max_source_chars`` and running ``--full`` leaves ``source`` short.

        Pinned, not endorsed. ``content_hash`` is computed *before* truncation
        (``parsing/ast.py`` ``_finalize``) so that moving the cap is not a content change,
        and the consequence is that layer 2 never fires for a cap change: the file is
        genuinely re-read and re-parsed, every entity classifies as ``unchanged``, only
        line positions are written, and the stored ``source`` stays cut at the old cap.

        Axis B is doing its job here — ``parse_calls`` proves the re-parse happened — so
        the gap is not in this epic's flag split. ATL-152's extraction epoch would not
        close it either: that invalidates ``file_hash``, which is layer 1. Repairing this
        means folding the cap into ``content_hash``.
        """
        body = "\n".join(f"    value_{i} = {i}" for i in range(60))
        _write(tmp_path, "src/big.py", f"def big():\n{body}\n    return 1\n")
        narrow = AtlasSettings(project_root=tmp_path, index=IndexSettings(max_source_chars=80), embeddings=NO_EMBED)
        wide = AtlasSettings(project_root=tmp_path, index=IndexSettings(max_source_chars=50_000), embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(tmp_path)

        await index_project(narrow, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert len(await _entity_source(graph_client, project, "big")) == 80

        parse_calls.clear()
        await index_project(wide, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert "src/big.py" in parse_calls, "the file has to be re-parsed, or this pins the wrong thing"
        assert len(await _entity_source(graph_client, project, "big")) == 80

    # -- axis C: --reset-embeddings drops vectors and nothing else ---------- #

    async def test_reset_embeddings_keeps_the_graph_and_re_embeds_without_reparsing(
        self, project_dir, graph_client, event_bus, provider, parse_calls
    ):
        """Vectors go, the graph stays, and the next run pays the provider and nothing else.

        Recovery is measured on the *searchable* vectors, not on the raw count, because
        the two genuinely differ: Package and DocFile nodes are handed a vector by the
        AST stage but no vector index serves them, so ``find_unembedded_entities``
        excludes them and only a re-parse would ever write one again -- which the gate
        correctly refuses. The clear is therefore one-way for those two labels. Nothing
        searchable is lost, so it is a wart rather than a hole, but a test asserting the
        raw count returns to its old value fails on it.
        """
        embedding = AtlasSettings(project_root=project_dir, embeddings=_embedding_settings(graph_client._dimension))
        lightweight = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(embedding, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        before = await _blast_radius(graph_client, project)
        before_searchable = await _searchable_vectors(graph_client, project)
        hashes = await _file_hashes(graph_client, project)
        assert before["embedded_nodes"] > 0

        # --no-embed so the cleared state is observable at all: with the embed stage
        # running, the same run's reconcile pass re-fills what it just dropped and the
        # zero never exists anywhere a test can see it.
        parse_calls.clear()
        await index_project(
            lightweight, graph_client, event_bus, reset_embeddings=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
        )

        cleared = await _blast_radius(graph_client, project)
        assert (cleared["embedded_nodes"], cleared["embed_chunks"]) == (0, 0)
        assert (cleared["nodes"], cleared["relationships"]) == (before["nodes"], before["relationships"])
        assert await _file_hashes(graph_client, project) == hashes
        assert parse_calls == [], "dropping vectors is not a reason to re-parse anything"

        # The next ordinary index re-embeds off the surviving graph. The gate still holds,
        # so nothing is re-parsed and only the provider is paid.
        provider.calls = 0
        await index_project(embedding, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert parse_calls == []
        assert provider.calls > 0
        assert await _searchable_vectors(graph_client, project) == before_searchable
        assert await graph_client.find_unembedded_entities(project) == []
        assert await _file_hashes(graph_client, project) == hashes

    async def test_reset_embeddings_on_a_sub_project_leaves_its_siblings_alone(
        self, tmp_path, graph_client, event_bus, provider
    ):
        """ATL-135, at monorepo granularity: one project's clear reaches one project.

        ``clear_embeddings`` matches name-or-prefix, so a clear scoped to
        ``{root}/auth`` must reach ``{root}/auth`` and its own children and stop there —
        not the sibling ``{root}/shared``, and not the bare root. Clearing every project
        for one project's model change is how ATL-135 destroyed 6,691 vectors belonging
        to a project nobody had asked about.
        """
        _write(tmp_path, "services/auth/pyproject.toml", '[project]\nname = "auth"\n')
        _write(tmp_path, "services/auth/auth/service.py", "def authenticate():\n    return True\n")
        _write(tmp_path, "libs/shared/pyproject.toml", '[project]\nname = "shared"\n')
        _write(tmp_path, "libs/shared/shared/utils.py", "def validate():\n    return True\n")
        _write(tmp_path, "tools/run.py", "def main():\n    return None\n")

        embedding = AtlasSettings(project_root=tmp_path, embeddings=_embedding_settings(graph_client._dimension))
        lightweight = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        root = derive_project_name(tmp_path)

        await index_monorepo(embedding, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        before = await graph_client.count_embeddings_by_project()
        assert before.get(f"{root}/auth", 0) > 0
        assert before.get(f"{root}/shared", 0) > 0

        # The sub-project entry point: index_project with the prefixed name and the
        # sub-project's own root, which is how every monorepo member is addressed.
        await index_project(
            lightweight,
            graph_client,
            event_bus,
            reset_embeddings=True,
            project_name=f"{root}/auth",
            project_root=tmp_path / "services" / "auth",
            drain_timeout_s=TEST_DRAIN_TIMEOUT_S,
        )

        after = await graph_client.count_embeddings_by_project()
        assert after.get(f"{root}/auth", 0) == 0
        assert after.get(f"{root}/shared") == before.get(f"{root}/shared")
        assert after.get(root) == before.get(root)


class TestTheExtractionKeyGatesEnumeration:
    """ATL-152: an extraction change invalidates the gate by itself.

    The file-hash gate keyed on file *bytes* while the extracted result also depends on
    the parser and on configuration, so its key was narrower than the thing it gated.
    After any extraction change the graph could be wrong for every file while the gate
    insisted nothing needed doing — recovered on 2026-08-31 by a hand-written script
    clearing ``file_hash`` on 845 nodes and ``git_hash`` on 9 projects.

    A git-backed project on purpose. On a non-git tree ``_decide_delta_mode`` falls back
    to full mode on its own, so nothing here would be testing the gate at all: the whole
    question is whether an unchanged HEAD still gets enumerated.
    """

    @pytest.fixture
    def git_project(self, tmp_path):
        _init_git_repo(tmp_path)
        _write(tmp_path, "src/__init__.py", "")
        _write(tmp_path, "src/app.py", 'def hello():\n    """Say hello."""\n    return "hello"\n')
        _write(tmp_path, "src/utils.py", "MAGIC = 42\n\ndef add(a, b):\n    return a + b\n")
        _git(tmp_path, "add", ".")
        _git(tmp_path, "commit", "-m", "initial")
        return tmp_path

    async def test_an_unchanged_key_still_skips_every_file(self, git_project, graph_client, event_bus, parse_calls):
        """The behaviour that must survive: the ordinary re-index stays free."""
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        parse_calls.clear()
        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.mode == "delta"
        assert parse_calls == []

    async def test_a_bumped_epoch_re_parses_a_project_git_calls_unchanged(
        self, git_project, graph_client, event_bus, parse_calls, monkeypatch
    ):
        """The story's headline scenario, and the reason both halves of the key exist.

        Bumping the epoch has to reach files git reports as unchanged, which takes BOTH
        gates: ``Project.extraction_key`` opens enumeration, and the epoch inside each
        ``file_hash`` is what then declines to skip. Remove either and this passes for
        the wrong reason or not at all — with only the file-hash half, delta mode
        publishes nothing and the gate is never even consulted.
        """
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        parse_calls.clear()

        monkeypatch.setattr(schema, "EXTRACTION_EPOCH", schema.EXTRACTION_EPOCH + 1)
        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.mode == "full"
        assert {"src/app.py", "src/utils.py"} <= set(parse_calls)

        # ...and exactly once. This is also the migration ATL-152 owes every graph indexed
        # before the key existed: those hashes were computed keyless, so the first run
        # afterwards re-reads everything and the run that re-checked also re-keyed.
        parse_calls.clear()
        again = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert again.mode == "delta"
        assert parse_calls == []

    async def test_a_bumped_epoch_costs_no_provider_call_when_nothing_moved(
        self, git_project, graph_client, event_bus, provider, parse_calls, monkeypatch
    ):
        """The whole point of keying the gate rather than clearing it: the epoch is free.

        Deliberately not a duplicate of ``test_full_on_an_unchanged_project_costs_no_provider_call``.
        That one reaches the re-parse through the ``--full`` FLAG, which sets
        ``force_reparse=True`` and turns the per-file gate off entirely. The epoch reaches
        it with the gate still ON and still trusted: ``full_reindex`` is False, every file
        is asked about, and the gate declines to skip only because the stored hashes were
        keyed on the old epoch. Same claim about spend, opposite setting of the switch that
        makes the claim interesting — and this is the route a version upgrade takes, where
        nobody chose to pay anything.
        """
        settings = AtlasSettings(project_root=git_project, embeddings=_embedding_settings(graph_client._dimension))
        await graph_client.ensure_schema()
        project = derive_project_name(git_project)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        embedded = (await graph_client.count_embeddings_by_project()).get(project, 0)
        assert embedded > 0, "the first index has to buy vectors, or a free re-check proves nothing"

        provider.calls = 0
        parse_calls.clear()
        monkeypatch.setattr(schema, "EXTRACTION_EPOCH", schema.EXTRACTION_EPOCH + 1)
        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.mode == "full"
        # Non-vacuous: a run that skipped every file would report zero calls too, for
        # entirely the wrong reason.
        assert {"src/app.py", "src/utils.py"} <= set(parse_calls)
        assert provider.calls == 0, "layer 2 (content_hash) has to stop the re-parse before the provider"
        assert (await graph_client.count_embeddings_by_project()).get(project, 0) == embedded

    async def test_an_extraction_affecting_setting_re_parses_without_manual_clearing(
        self, git_project, graph_client, event_bus, parse_calls
    ):
        """The ``max_source_chars`` case the story exists for, on the enumeration side.

        NOTE this proves the file is re-READ, not that its stored ``source`` is repaired:
        ``content_hash`` is computed before truncation, so layer 2 still classifies every
        entity as unchanged. That gap is pinned separately by
        ``test_full_does_not_repair_a_source_truncated_under_a_narrower_cap`` and closing
        it means folding the cap into ``content_hash``, which is not this story.
        """
        await graph_client.ensure_schema()
        narrow = AtlasSettings(
            project_root=git_project, index=IndexSettings(max_source_chars=2000), embeddings=NO_EMBED
        )
        wide = AtlasSettings(
            project_root=git_project, index=IndexSettings(max_source_chars=48_000), embeddings=NO_EMBED
        )

        await index_project(narrow, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        parse_calls.clear()
        result = await index_project(wide, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.mode == "full"
        assert {"src/app.py", "src/utils.py"} <= set(parse_calls)

    async def test_a_setting_that_cannot_move_extraction_still_skips(
        self, git_project, graph_client, event_bus, parse_calls
    ):
        """The other direction, and the one that makes the key worth having.

        A key that moved on any config change would re-parse the world every time someone
        tuned RRF weights, which is how a gate stops being trusted.
        """
        await graph_client.ensure_schema()
        before = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        after = AtlasSettings(project_root=git_project, search=SearchSettings(rrf_k=99), embeddings=NO_EMBED)

        await index_project(before, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        parse_calls.clear()
        result = await index_project(after, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.mode == "delta"
        assert parse_calls == []

    async def test_a_project_with_no_stored_key_enumerates_everything(self, git_project, graph_client, event_bus):
        """An absent key has to read as "differs" — this is what carries the upgrade.

        A graph indexed before this shipped has no ``Project.extraction_key`` AND stored
        every ``file_hash`` without one. This test pins the half an absent key is
        responsible for: enumeration reopens, so the per-file gate is asked about every
        file instead of about nothing. The other half — the gate then declining to skip
        them — is what test_a_bumped_epoch_re_parses_a_project_git_calls_unchanged
        covers, and only both together re-parse the project.

        Deliberately does not assert on parse_calls: these hashes were written by the
        current code under the current key, so the gate skipping them here is correct.

        Doing it this way rather than with a ``SCHEMA_VERSION`` bump is a decision, not an
        omission. A bump would deliver the one-time re-check once and then need repeating
        for every future epoch move, and ``ensure_schema``'s migration branch drops and
        rebuilds every vector and text index (``_migrate_indices``) — a cost this graph
        has already paid once for nothing, and one a lightweight install cannot pay at all.
        """
        settings = AtlasSettings(project_root=git_project, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(git_project)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert await graph_client.get_project_extraction_key(project) is not None
        assert (await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)).mode == (
            "delta"
        ), "without this the assertion below would pass for the wrong reason"

        await graph_client.execute_write(
            f"MATCH (p:{NodeLabel.PROJECT} {{uid: $uid}}) REMOVE p.extraction_key", {"uid": project}
        )
        result = await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert result.mode == "full"
        assert result.files_published == 3


class _RelWriteSpy:
    """Counts the relationship statements the per-file write path issues, and their payloads.

    Statement count alone measures batch count rather than work: the delete is one
    statement per *batch* over a ``$fps`` list and the create is one UNWIND per rel_type
    over the pooled rels, so a thirty-file batch is a handful of statements whether one
    file changed or thirty. ``deleted_file_paths`` and ``created_rels`` are the numbers
    that move.

    Matches on query shape rather than wrapping ``_recreate_batch_relationships``,
    because the claim under test is about statements actually reaching the database.
    Deliberately blind to the post-batch resolvers' own MERGEs (CALLS/IMPORTS/USES_TYPE/
    DOCUMENTS), which ATL-151 does not touch — the replay buffer is kept, so those still
    run for every file including the ones whose rewrite was skipped.
    """

    def __init__(self, graph) -> None:
        self._graph = graph
        self._orig = graph.execute_write
        self.delete_statements = 0
        self.deleted_file_paths = 0
        self.create_statements = 0
        self.created_rels = 0

    def __enter__(self) -> _RelWriteSpy:
        async def _spy(query: str, params: dict | None = None, **kwargs):
            p = params or {}
            if "n.file_path IN $fps AND NOT n:" in query and "DELETE r" in query:
                self.delete_statements += 1
                self.deleted_file_paths += len(p.get("fps") or [])
            elif "SET e += r.props" in query or "CREATE (a)-[:IMPLEMENTS" in query:
                self.create_statements += 1
                self.created_rels += len(p.get("rels") or [])
            return await self._orig(query, params, **kwargs)

        self._graph.execute_write = _spy
        return self

    def __exit__(self, *exc: object) -> None:
        self._graph.execute_write = self._orig

    @property
    def total(self) -> int:
        return self.delete_statements + self.create_statements

    def __str__(self) -> str:
        return (
            f"{self.delete_statements} delete(s) over {self.deleted_file_paths} file(s), "
            f"{self.create_statements} create(s) of {self.created_rels} rel(s)"
        )


async def _rel_census(graph_client, project: str) -> dict[str, int]:
    """``rel_type -> count`` for every edge leaving one of *project*'s nodes."""
    records = await graph_client.execute(
        "MATCH (n {project_name: $p})-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY t",
        {"p": project},
    )
    return {r["t"]: r["c"] for r in records}


async def _mark_rels(graph_client, project: str, file_path: str | None = None) -> int:
    """Stamp a property no writer sets onto the project's edges. Returns how many.

    Counting edges cannot separate "left alone" from "deleted and recreated identically"
    — both end on the same number, which is exactly why the unconditional rewrite could
    be swapped for the skip with every count-based test still passing.
    """
    pattern = (
        f"MATCH (n:{NodeLabel.ENTITY} {{project_name: $p, file_path: $f}})-[r]->()"
        if file_path
        else f"MATCH (n:{NodeLabel.ENTITY} {{project_name: $p}})-[r]->()"
    )
    records = await graph_client.execute(
        f"{pattern} SET r.atl151_marker = 'survived' RETURN count(r) AS c",
        {"p": project, "f": file_path},
    )
    return records[0]["c"]


async def _rewritten_rels(graph_client, project: str, file_path: str | None = None) -> list[str]:
    """Every unmarked edge out of the scope, as ``"file_path:REL_TYPE"``, sorted.

    A rewrite deletes and recreates, so anything it touched comes back unmarked. An edge
    created *since* the mark was taken is unmarked too and shows up here — a post-batch
    resolver's new CALLS edge, say — so a caller expecting one of those names the rel
    type it cares about rather than asserting this is empty.
    """
    pattern = (
        f"MATCH (n:{NodeLabel.ENTITY} {{project_name: $p, file_path: $f}})-[r]->()"
        if file_path
        else f"MATCH (n:{NodeLabel.ENTITY} {{project_name: $p}})-[r]->()"
    )
    records = await graph_client.execute(
        f"{pattern} WHERE r.atl151_marker IS NULL RETURN n.file_path AS fp, type(r) AS t",
        {"p": project, "f": file_path},
    )
    return sorted(f"{r['fp']}:{r['t']}" for r in records)


async def _overrides(graph_client, project: str) -> int:
    """How many OVERRIDES edges the detector currently has standing."""
    records = await graph_client.execute(
        f"MATCH (:{NodeLabel.CALLABLE} {{project_name: $p}})-[r:{RelType.OVERRIDES}]->() RETURN count(r) AS c",
        {"p": project},
    )
    return records[0]["c"]


async def _calls(graph_client, project: str) -> list[tuple[str, str]]:
    """``(caller name, callee name)`` for every resolved CALLS edge, sorted."""
    records = await graph_client.execute(
        f"MATCH (a {{project_name: $p}})-[:{RelType.CALLS}]->(b) RETURN a.name AS a, b.name AS b",
        {"p": project},
    )
    return sorted((r["a"], r["b"]) for r in records)


class TestTheRelationshipRewriteIsSkippedWhenNothingMoved:
    """ATL-151, at the altitude a user meets it: ``atlas index --full``.

    ``--full`` is the run the story exists for. Entities were already diffed by
    ``content_hash`` and skipped when unchanged; relationships had no equivalent, so a
    re-check that found nothing changed still deleted and recreated every edge for every
    file — 4,878 files' worth on the production graph — to arrive back where it started.

    Every project here is deliberately **not** a git repo, so ``_decide_delta_mode``
    returns ``full`` on every run and enumeration is never the variable. What separates
    the runs is the ``--full`` flag alone: axis B, whether the ``file_hash`` gate is
    trusted. Without it an unchanged file is never even parsed and nothing below would
    reach the fingerprint at all.
    """

    @pytest.fixture
    def project_dir(self, tmp_path):
        """Two plain modules, one importing and calling the other.

        No ``__init__.py``, deliberately, and this is measured rather than assumed: add a
        non-empty one and ``test_a_full_recheck_of_an_unchanged_project_writes_no_relationships``
        fails with "1 delete(s) over 1 file(s)". A package marker with content re-reports
        an entity as *added* on every single re-index, so the classification refuses its
        skip forever — a pre-existing signal with nothing to do with this story, which
        would make the zero below unreachable for the wrong reason. (An *empty* marker is
        harmless: it lands in ``new_file_paths`` each run but has no relationships to
        rewrite, and the test still passes with one present.)
        """
        _write(tmp_path, "alpha.py", "def widen(value):\n    return value + 1\n")
        _write(tmp_path, "beta.py", "from alpha import widen\n\n\ndef run():\n    return widen(1)\n")
        return tmp_path

    async def test_a_full_recheck_of_an_unchanged_project_writes_no_relationships(
        self, project_dir, graph_client, event_bus, parse_calls
    ):
        """The headline claim: re-checking everything costs the parse and nothing else.

        The spy is the direct evidence and the marker is the corroborating kind — an
        edge that came back is a *different* edge, and no count can tell you that.
        """
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert await _mark_rels(graph_client, project) > 0, "the first index has to write edges"
        before = await _rel_census(graph_client, project)

        parse_calls.clear()
        with _RelWriteSpy(graph_client) as spy:
            result = await index_project(
                settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
            )

        assert result.mode == "full"
        # Non-vacuous: a run that trusted the gate and skipped every file would report
        # the same zero, for entirely the wrong reason.
        assert {"alpha.py", "beta.py"} <= set(parse_calls), "--full must distrust the gate and re-parse"
        assert spy.total == 0, f"a no-op re-check still issued {spy}"
        assert await _rel_census(graph_client, project) == before
        assert await _rewritten_rels(graph_client, project) == []

    async def test_a_file_that_gained_an_edge_is_rewritten_and_its_neighbour_is_not(
        self, project_dir, graph_client, event_bus, parse_calls
    ):
        """The selectivity, which is the part neither half proves alone.

        Both files are re-parsed — ``--full`` guarantees that — and exactly one of them
        reaches the database. Note what "gained an edge" has to mean here: a gained
        *call* would be the wrong example, because CALLS is resolved post-batch from the
        replay buffer and is not part of what this path writes at all. A gained
        definition is, and it moves the file's DEFINES set.
        """
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        await _mark_rels(graph_client, project)

        _write(
            project_dir,
            "alpha.py",
            "def widen(value):\n    return value + 1\n\n\ndef narrow(value):\n    return value - 1\n",
        )

        parse_calls.clear()
        with _RelWriteSpy(graph_client) as spy:
            await index_project(
                settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S
            )

        assert {"alpha.py", "beta.py"} <= set(parse_calls)
        assert spy.deleted_file_paths == 1, f"exactly one file should have been rewritten, got {spy}"
        assert await _rewritten_rels(graph_client, project, "beta.py") == []
        assert "alpha.py:DEFINES" in await _rewritten_rels(graph_client, project, "alpha.py")
        names = await graph_client.execute(
            f"MATCH (m:{NodeLabel.MODULE} {{project_name: $p, file_path: 'alpha.py'}})"
            f"-[:{RelType.DEFINES}]->(c) RETURN c.name AS name",
            {"p": project},
        )
        assert sorted(r["name"] for r in names) == ["narrow", "widen"]

    async def test_a_new_module_is_linked_from_a_file_the_recheck_did_not_rewrite(
        self, tmp_path, graph_client, event_bus
    ):
        """The story's real risk, and the reason the buffer was kept when the write was not.

        The story also asked for a skipped file's relationships not to be buffered for the
        resolution flush. That buffer is exactly what ADR-0026 added to fix a measured
        loss — resolution reads the graph as it stands at the flush, so a callee upserted
        later was never linked: CALLS 9,058 -> 9,713, cross-file 4,066 -> 4,720,
        ``find_dead_code`` on src/ 27 -> 15. Skipping the buffer for an unchanged file
        reintroduces precisely that, and worse, because ``--full`` is the run that repairs
        the hole — the repair would be the thing broken.

        So the write is skipped and the buffer is kept, and this test is what says so.
        Delete the buffering for skipped files and it fails with no edge at all: the
        caller's ``file_hash`` is stored the same run, so nothing would ever revisit it.
        """
        _write(tmp_path, "caller.py", "def run():\n    return compute()\n")
        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(tmp_path)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert await _calls(graph_client, project) == [], "precondition: nothing defines compute yet"
        await _mark_rels(graph_client, project, "caller.py")

        _write(tmp_path, "helper_lib.py", "def compute():\n    return 1\n")
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        # The skip really did fire for the caller — otherwise the buffer is not under
        # test. Named by rel type rather than asserted empty: the CALLS edge below is
        # itself new and unmarked, which is the whole point.
        assert "caller.py:DEFINES" not in await _rewritten_rels(graph_client, project, "caller.py"), (
            "the caller's rewrite was not skipped, so this proves nothing about the buffer"
        )
        assert await _calls(graph_client, project) == [("run", "compute")]

    async def test_a_detector_edge_that_stopped_firing_is_revoked_by_the_recheck(
        self, tmp_path, graph_client, event_bus, parse_calls
    ):
        """The one shape where the fingerprint, and not the classification, is deciding.

        For parser-emitted edges the entity classification is nearly sufficient on its
        own: adding or removing an edge almost always means adding or editing the entity
        it runs from, so a fingerprint forced to a constant changes nothing. What
        genuinely moves a file's written rel set while every one of its own entities
        stands still is **detector output**, which is derived from the rest of the graph.

        Here ``Base.run`` disappears and ``child_mod.py`` is untouched on disk, so it
        stops emitting OVERRIDES. Step 3's rewrite is the only thing that can revoke it —
        step 4b touches only files whose detectors emit something in the CURRENT run — and
        the only reason ``child_mod.py`` gets that rewrite is that its stored fingerprint
        covers the merged parser+detector set while the compare is made pre-detector, so
        a file that carried detector edges can never match. Force ``_rels_hash`` to a
        constant and this is the test in this file that fails, with the stale edge
        surviving every future re-check.
        """
        _write(tmp_path, "base_mod.py", "class Base:\n    def run(self):\n        return 1\n")
        _write(
            tmp_path,
            "child_mod.py",
            "from base_mod import Base\n\n\nclass Child(Base):\n    def run(self):\n        return 2\n",
        )
        settings = AtlasSettings(project_root=tmp_path, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(tmp_path)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        assert await _overrides(graph_client, project) == 1, "precondition: the detector fired"

        _write(tmp_path, "base_mod.py", "class Base:\n    def other(self):\n        return 1\n")
        parse_calls.clear()
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert "child_mod.py" in parse_calls, "the untouched file has to be re-parsed, or this pins nothing"
        assert await _overrides(graph_client, project) == 0, "a detector edge that stopped firing survived the re-check"

    async def test_repeated_full_rechecks_leave_the_relationship_census_alone(
        self, project_dir, graph_client, event_bus
    ):
        """Skipping the rewrite must not let anything accumulate in its place.

        The per-file delete used to be the idempotence of every edge the flush re-derives,
        and one resolver was leaning on it: ``resolve_doc_links`` used ``CREATE`` where
        the others MERGE, and duplicated every heuristic DOCUMENTS edge on every re-check
        — 213 -> 432 over three runs, unbounded. Nothing in the graph looked broken; the
        count simply grew.

        Runs two and three are the comparison, not one and two. A first index resolves
        against a partial graph and ADR-0026's replay MERGEs rather than retracts, so a
        single legitimate settling step between them is expected and is not what this is
        watching for.
        """
        _write(project_dir, "guide.md", "# Guide\n\n## Using widen\n\nCall `widen` to widen a value.\n")
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)

        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        settled = await _rel_census(graph_client, project)
        assert settled.get(RelType.DOCUMENTS.value, 0) > 0, "precondition: a doc edge exists to duplicate"

        await index_project(settings, graph_client, event_bus, full_reindex=True, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert await _rel_census(graph_client, project) == settled

    async def test_an_interrupted_run_leaves_both_gates_open(
        self, project_dir, graph_client, event_bus, parse_calls, monkeypatch
    ):
        """DESIGN J: ``rels_hash`` is withheld until the deferred flush, like ``file_hash``.

        Both are written by the same block of ``_flush_deferred_resolution`` and for the
        same reason, sharpened: a fingerprint stored before the flush describes a rel set
        whose deferred half is not resolved yet, so a crash in between produces a file
        that visibly re-parses on every run (its ``file_hash`` correctly unset) while
        declining to rewrite relationships whose other half never landed. That is
        undiagnosable from the outside — the gate you would inspect looks right.

        The kill is the flush never happening, which is what a process death between the
        upsert and the flush amounts to: the entity and relationship transactions are
        committed, the resolution is not, and neither gate property may be on disk.
        """
        settings = AtlasSettings(project_root=project_dir, embeddings=NO_EMBED)
        await graph_client.ensure_schema()
        project = derive_project_name(project_dir)
        files = ["alpha.py", "beta.py"]

        async def _never_flushes(self, *, final: bool = False) -> None:
            return None

        # Restored by hand rather than with monkeypatch.undo(): the parse_calls fixture
        # patches through the SAME function-scoped monkeypatch instance, so undo() would
        # silently take the parse recorder down with it and every assertion below would
        # read an empty list as "nothing was parsed".
        real_flush = ASTConsumer._flush_deferred_resolution
        monkeypatch.setattr(ASTConsumer, "_flush_deferred_resolution", _never_flushes)
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)
        monkeypatch.setattr(ASTConsumer, "_flush_deferred_resolution", real_flush)

        assert set(files) <= set(parse_calls), "precondition: the interrupted run did parse and upsert"
        assert await _calls(graph_client, project) == [], "precondition: its deferred half never resolved"
        assert await graph_client.get_batch_file_hashes(project, files) == dict.fromkeys(files)
        assert await graph_client.get_batch_rels_hashes(project, files) == dict.fromkeys(files)

        # Recovery, on the ordinary path: both gates being open is what makes the next
        # run re-read the files AND rewrite their relationships rather than trust either.
        parse_calls.clear()
        await index_project(settings, graph_client, event_bus, drain_timeout_s=TEST_DRAIN_TIMEOUT_S)

        assert set(files) <= set(parse_calls)
        assert await _calls(graph_client, project) == [("run", "widen")]
        assert all((await graph_client.get_batch_file_hashes(project, files)).values())
        assert all((await graph_client.get_batch_rels_hashes(project, files)).values())
