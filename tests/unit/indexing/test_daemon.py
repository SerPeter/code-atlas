"""Unit tests for DaemonManager supervision, startup catch-up, and shutdown ordering."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

import pytest
import typer

from code_atlas.indexing import daemon as daemon_module
from code_atlas.indexing.daemon import DaemonManager
from code_atlas.settings import AtlasSettings, ExtraVaultSettings

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeConsumer:
    """TierConsumer stand-in: crashes *crashes* times, then runs until stop()."""

    def __init__(self, name: str = "fake-0", crashes: int = 0) -> None:
        self.consumer_name = name
        self.runs = 0
        self._crashes = crashes
        self._stop = False
        self.running = asyncio.Event()

    @property
    def stopped(self) -> bool:
        return self._stop

    def stop(self) -> None:
        self._stop = True

    async def run(self) -> None:
        self.runs += 1
        if self.runs <= self._crashes:
            raise RuntimeError(f"boom {self.runs}")
        self.running.set()
        while not self._stop:
            await asyncio.sleep(0.01)


class FakeWatcher:
    """FileWatcher stand-in: crashes *crashes* times, drains only on clean exit."""

    def __init__(self, crashes: int = 0) -> None:
        self.runs = 0
        self._crashes = crashes
        self._stop_event = asyncio.Event()
        self.running = asyncio.Event()
        self.drained = False

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        self.runs += 1
        if self.runs <= self._crashes:
            raise RuntimeError("watch boom")
        self.running.set()
        await self._stop_event.wait()
        # Shutdown drain — only reached when stop() lets run() finish
        await asyncio.sleep(0)
        self.drained = True


class FakeBus:
    """EventBus stand-in that always pings OK, with an in-memory indexer lease."""

    def __init__(self, settings: object, *, project_name: str = "") -> None:
        self.project_name = project_name
        self.group_info: dict[str, dict[str, int | None]] = {}
        self._lease_holder: str | None = None

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def stream_group_info_multi(self, queries: list[tuple[Any, str]]) -> list[dict[str, int | None]]:
        return [self.group_info.get(group, {"pending": 0, "lag": 0}) for _topic, group in queries]

    async def acquire_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        if self._lease_holder is not None:
            return False
        self._lease_holder = owner
        return True

    async def renew_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        return self._lease_holder == owner

    async def release_indexer_lease(self, owner: str) -> bool:
        if self._lease_holder != owner:
            return False
        self._lease_holder = None
        return True

    async def read_indexer_lease(self) -> str | None:
        return self._lease_holder


# ---------------------------------------------------------------------------
# Supervision
# ---------------------------------------------------------------------------


class TestConsumerSupervision:
    """Consumer tasks are supervised: crash → recorded + backoff restart."""

    async def test_consumer_restarts_after_crash(self) -> None:
        manager = DaemonManager()
        consumer = FakeConsumer(crashes=1)

        task = asyncio.create_task(manager._run_consumer(consumer))  # type: ignore[arg-type]
        # First run crashes; supervision restarts after ~1s backoff
        await asyncio.wait_for(consumer.running.wait(), timeout=5.0)

        assert consumer.runs == 2
        status = manager.status()
        assert status["crash_counts"] == {"fake-0": 1}
        assert "boom 1" in status["last_crash"]["fake-0"]

        consumer.stop()
        await asyncio.wait_for(task, timeout=5.0)

    async def test_consumer_clean_exit_not_recorded_as_crash(self) -> None:
        manager = DaemonManager()
        consumer = FakeConsumer()

        task = asyncio.create_task(manager._run_consumer(consumer))  # type: ignore[arg-type]
        await asyncio.wait_for(consumer.running.wait(), timeout=2.0)
        consumer.stop()
        await asyncio.wait_for(task, timeout=2.0)

        assert consumer.runs == 1
        assert manager.status()["crash_counts"] == {}


class TestWatcherSupervision:
    """The watcher task is supervised with the same restart loop."""

    async def test_watcher_restarts_after_crash(self) -> None:
        manager = DaemonManager()
        watcher = FakeWatcher(crashes=1)
        manager._watcher = watcher  # type: ignore[assignment]

        task = asyncio.create_task(manager._run_watcher())
        await asyncio.wait_for(watcher.running.wait(), timeout=5.0)

        assert watcher.runs == 2
        status = manager.status()
        assert status["crash_counts"] == {"watcher": 1}
        assert "watch boom" in status["last_crash"]["watcher"]

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)
        assert watcher.drained


class TestStatus:
    """DaemonManager.status() exposes task liveness and crash state."""

    async def test_status_counts_running_tasks(self) -> None:
        manager = DaemonManager()
        consumer = FakeConsumer()
        task = asyncio.create_task(manager._run_consumer(consumer))  # type: ignore[arg-type]
        manager._tasks.append(task)
        await asyncio.wait_for(consumer.running.wait(), timeout=2.0)

        status = manager.status()
        assert status["tasks_running"] == 1
        assert status["tasks_total"] == 1

        consumer.stop()
        await asyncio.wait_for(task, timeout=2.0)
        assert manager.status()["tasks_running"] == 0


class TestPendingEventCounts:
    """pending_event_counts() surfaces backlog size, or None when not running."""

    async def test_returns_none_when_bus_not_set(self) -> None:
        manager = DaemonManager()
        assert await manager.pending_event_counts() is None

    async def test_returns_counts_from_bus(self) -> None:
        manager = DaemonManager()
        bus = FakeBus(object())
        bus.group_info = {"ast": {"pending": 2, "lag": 3}, "embed": {"pending": 1, "lag": 0}}
        manager._bus = bus  # type: ignore[assignment]
        manager._embed = object()  # type: ignore[assignment]  # non-None signals embed consumer active

        counts = await manager.pending_event_counts()
        assert counts == {"file-changed": 5, "embed-dirty": 1}

    async def test_unknown_lag_falls_back_to_pending_only(self) -> None:
        manager = DaemonManager()
        bus = FakeBus(object())
        bus.group_info = {"ast": {"pending": 4, "lag": None}}
        manager._bus = bus  # type: ignore[assignment]

        counts = await manager.pending_event_counts()
        assert counts == {"file-changed": 4}


# ---------------------------------------------------------------------------
# Shutdown ordering
# ---------------------------------------------------------------------------


class TestStopOrdering:
    """stop() lets tasks observe stop flags (watcher drain) before cancelling."""

    async def test_stop_lets_watcher_drain_pending(self) -> None:
        manager = DaemonManager()
        watcher = FakeWatcher()
        manager._watcher = watcher  # type: ignore[assignment]
        manager._tasks.append(asyncio.get_running_loop().create_task(manager._run_watcher()))
        await asyncio.wait_for(watcher.running.wait(), timeout=2.0)

        await manager.stop()

        assert watcher.drained


# ---------------------------------------------------------------------------
# Startup catch-up
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path) -> AtlasSettings:
    settings = AtlasSettings(project_root=tmp_path)
    settings.embeddings.enabled = False
    return settings


class TestStartupCatchup:
    """start(catchup=True) runs one delta index pass before consumers start."""

    @pytest.fixture
    def patched_daemon(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Patch DaemonManager collaborators; record call order in ``order``."""
        state: dict[str, Any] = {"order": [], "monorepo": False, "fail_catchup": False}

        class OrderedConsumer(FakeConsumer):
            async def run(self) -> None:
                state["order"].append("consumer-run")
                await super().run()

        async def fake_index_project(settings: object, graph: object, bus: object, **kwargs: Any) -> None:
            state["drain_timeout_s"] = kwargs.get("drain_timeout_s")
            if state["fail_catchup"]:
                raise RuntimeError("catch-up exploded")
            state["order"].append("catchup-project")

        async def fake_index_monorepo(settings: object, graph: object, bus: object, **kwargs: Any) -> None:
            state["drain_timeout_s"] = kwargs.get("drain_timeout_s")
            state["order"].append("catchup-monorepo")

        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr(
            daemon_module, "ASTConsumer", lambda bus, graph, settings, **kw: OrderedConsumer(name="ast-0")
        )
        monkeypatch.setattr(daemon_module, "index_project", fake_index_project)
        monkeypatch.setattr(daemon_module, "index_monorepo", fake_index_monorepo)
        monkeypatch.setattr(
            daemon_module, "detect_sub_projects", lambda root, mono: ["sub"] if state["monorepo"] else []
        )
        return state

    async def test_catchup_runs_before_consumers(self, tmp_path: Path, patched_daemon: dict[str, Any]) -> None:
        manager = DaemonManager()
        started = await manager.start(_make_settings(tmp_path), object(), include_watcher=False)  # type: ignore[arg-type]
        assert started is True
        await asyncio.sleep(0.05)

        order = patched_daemon["order"]
        assert order[0] == "catchup-project"
        assert "consumer-run" in order

        await manager.stop()

    async def test_catchup_threads_configured_drain_timeout(
        self, tmp_path: Path, patched_daemon: dict[str, Any]
    ) -> None:
        """settings.index.drain_timeout_s must reach index_project/index_monorepo, not the
        hardcoded 600s default -- a workload too large to drain in the default window can
        never advance its git_hash checkpoint, so every retry republishes everything."""
        settings = _make_settings(tmp_path)
        settings.index.drain_timeout_s = 3600.0

        manager = DaemonManager()
        started = await manager.start(settings, object(), include_watcher=False)  # type: ignore[arg-type]
        assert started is True
        await asyncio.sleep(0.05)

        assert patched_daemon["drain_timeout_s"] == 3600.0

        await manager.stop()

    async def test_catchup_false_skips_index(self, tmp_path: Path, patched_daemon: dict[str, Any]) -> None:
        manager = DaemonManager()
        started = await manager.start(_make_settings(tmp_path), object(), include_watcher=False, catchup=False)  # type: ignore[arg-type]
        assert started is True
        await asyncio.sleep(0.05)

        assert patched_daemon["order"] == ["consumer-run"]

        await manager.stop()

    async def test_catchup_routes_monorepo(self, tmp_path: Path, patched_daemon: dict[str, Any]) -> None:
        patched_daemon["monorepo"] = True
        manager = DaemonManager()
        started = await manager.start(_make_settings(tmp_path), object(), include_watcher=False)  # type: ignore[arg-type]
        assert started is True

        assert patched_daemon["order"][0] == "catchup-monorepo"

        await manager.stop()

    async def test_catchup_failure_is_non_fatal(self, tmp_path: Path, patched_daemon: dict[str, Any]) -> None:
        patched_daemon["fail_catchup"] = True
        manager = DaemonManager()
        started = await manager.start(_make_settings(tmp_path), object(), include_watcher=False)  # type: ignore[arg-type]
        assert started is True
        await asyncio.sleep(0.05)

        # Consumers still started despite the catch-up failure
        assert "consumer-run" in patched_daemon["order"]

        await manager.stop()

    async def test_catchup_holds_and_releases_lease(self, tmp_path: Path, patched_daemon: dict[str, Any]) -> None:
        """A successful catch-up must not leave the lease held afterwards."""
        manager = DaemonManager()
        started = await manager.start(_make_settings(tmp_path), object(), include_watcher=False)  # type: ignore[arg-type]
        assert started is True
        await asyncio.sleep(0.05)

        assert "catchup-project" in patched_daemon["order"]
        assert manager.bus._lease_holder is None  # type: ignore[union-attr]

        await manager.stop()

    async def test_catchup_skips_when_lease_held_by_another_process(
        self, tmp_path: Path, patched_daemon: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A foreign indexer lease (peer daemon or a running CLI index) must stop this
        catch-up rather than race it straight into Memgraph -- the exact multi-session
        collision the lease exists to prevent (see hold_indexer_lease's docstring)."""

        class BusyFakeBus(FakeBus):
            def __init__(self, settings: object, *, project_name: str = "") -> None:
                super().__init__(settings, project_name=project_name)
                self._lease_holder = "peer-host:999:deadbeef"

        monkeypatch.setattr("code_atlas.backends.EventBus", BusyFakeBus)

        # Zero, so the skip is immediate. With the default this test waited out the whole
        # lease budget -- it still passed, and the unit suite silently went from ~60s to
        # ~620s. A test that waits for a timeout is not testing the timeout; the bound
        # itself is asserted separately below.
        settings = _make_settings(tmp_path)
        settings.index.lease_wait_s = 0

        manager = DaemonManager()
        # Bounded at 5s, not left to the suite's 300s cap. `start()` is the call that
        # blocks, and this file's other eleven timeouts are all on the things its author
        # expected to block -- consumer/watcher readiness, task completion. So when
        # catch-up gained a lease wait, this test quietly took the entire budget and
        # still passed. A tight bound turns that into a failure in seconds.
        started = await asyncio.wait_for(
            manager.start(settings, object(), include_watcher=False),  # type: ignore[arg-type]
            timeout=5.0,
        )
        assert started is True
        await asyncio.sleep(0.05)

        order = patched_daemon["order"]
        assert "catchup-project" not in order
        assert "catchup-monorepo" not in order
        # Live-event consumption still starts -- it'll pick up work once the peer
        # holding the lease finishes, via the stream rather than this inline pass.
        assert "consumer-run" in order

    async def test_the_catchup_wait_is_bounded_far_below_the_cli_budget(
        self, tmp_path: Path, patched_daemon: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`start()` does not return until catch-up finishes, so whatever it waits for,
        the caller waits for too -- for the MCP server that is first_index_ready staying
        clear the whole time.

        The only reason to wait at all is a holder that is already dead, and its lease
        expires within the 60s TTL. Anything past that is a live indexer doing the same
        work, so waiting longer buys nothing and costs startup. Asserted against the
        constant rather than a sleep, because a test that actually waits 90s is how the
        suite grew tenfold in the first place.
        """
        from code_atlas.events import INDEXER_LEASE_TTL_MS
        from code_atlas.indexing.daemon import _CATCHUP_LEASE_WAIT_S

        settings = _make_settings(tmp_path)
        assert _CATCHUP_LEASE_WAIT_S > INDEXER_LEASE_TTL_MS / 1000, "must outlast a dead holder's lease"
        assert settings.index.lease_wait_s > _CATCHUP_LEASE_WAIT_S, "must be far below the foreground budget"

        captured: dict[str, float] = {}

        @contextlib.asynccontextmanager
        async def spy_lease(_bus, *, wait_s: float = 0.0, **_kw):
            captured["wait_s"] = wait_s
            yield "owner"

        monkeypatch.setattr("code_atlas.indexing.daemon.hold_indexer_lease", spy_lease)

        manager = DaemonManager()
        await manager.start(settings, object(), include_watcher=False)  # type: ignore[arg-type]
        await manager.stop()

        assert captured["wait_s"] == _CATCHUP_LEASE_WAIT_S

        await manager.stop()


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestWatcherScopeScan:
    """The watcher's FileScope must be scan()ned before it starts filtering.

    FileScope only discovers nested .gitignore files as a side effect of
    scan() (they're recorded while walking). Building it and handing it
    straight to the watcher without scanning means nested-.gitignore'd
    files are never excluded from live watching, even though a full/delta
    ``atlas index`` run (which does call scan()) excludes them correctly.
    """

    async def test_scope_scanned_and_known_files_passed_to_watcher(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state: dict[str, Any] = {"scanned": False}

        class FakeScope:
            def __init__(self, root: object, settings: object) -> None:
                pass

            def scan(self) -> list[str]:
                state["scanned"] = True
                return ["a.py", "b.py"]

            def is_included(self, rel_path: str) -> bool:
                return True

        captured: dict[str, Any] = {}

        class FakeWatcher:
            def __init__(
                self,
                root: object,
                bus: object,
                scope: object,
                settings: object,
                *,
                sub_projects: object = None,
                root_name: str = "",
                known_files: list[str] | None = None,
            ) -> None:
                captured["known_files"] = known_files
                self._stop = False

            @property
            def stopped(self) -> bool:
                return self._stop

            def stop(self) -> None:
                self._stop = True

            async def run(self) -> None:
                return  # clean exit — no crash, nothing to supervise

        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr(daemon_module, "FileScope", FakeScope)
        monkeypatch.setattr(daemon_module, "FileWatcher", FakeWatcher)
        monkeypatch.setattr(daemon_module, "detect_sub_projects", lambda root, mono: [])
        monkeypatch.setattr(daemon_module, "ASTConsumer", lambda bus, graph, settings, **kw: FakeConsumer(name="ast-0"))

        manager = DaemonManager()
        started = await manager.start(
            _make_settings(tmp_path),
            object(),  # type: ignore[arg-type]
            include_watcher=True,
            catchup=False,
        )
        assert started is True
        await asyncio.sleep(0.05)

        assert state["scanned"] is True
        assert captured["known_files"] == ["a.py", "b.py"]

        await manager.stop()


class TestVaultCatchupAndWatching:
    """Extra vaults (global vault, harness memory dir) get a one-time catch-up
    scan plus their own live FileWatcher instance (multi-root watching, Phase 5)."""

    async def test_catchup_vault_swallows_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        manager = DaemonManager()
        manager._bus = FakeBus(object())  # type: ignore[assignment]

        async def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(daemon_module, "publish_project_changes", boom)

        # Must not raise — a failed vault catch-up shouldn't take down startup.
        await manager._catchup_vault("test-vault", tmp_path, [], _make_settings(tmp_path), object())  # type: ignore[arg-type]

    async def test_vault_watcher_restarts_after_crash(self) -> None:
        manager = DaemonManager()
        watcher = FakeWatcher(crashes=1)

        task = asyncio.create_task(manager._run_vault_watcher("test-vault", watcher))  # type: ignore[arg-type]
        await asyncio.wait_for(watcher.running.wait(), timeout=5.0)

        assert watcher.runs == 2
        status = manager.status()
        assert status["crash_counts"] == {"vault:test-vault": 1}
        assert "watch boom" in status["last_crash"]["vault:test-vault"]

        watcher.stop()
        await asyncio.wait_for(task, timeout=2.0)
        assert watcher.drained


class TestVaultTaskSpawning:
    """start() spawns a catch-up pass + a live watcher task per configured extra vault."""

    async def test_start_spawns_watcher_per_extra_vault(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        catchup_calls: list[str] = []

        async def fake_catchup_vault(
            self: DaemonManager,
            project_name: str,
            vault_root: Path,
            files: list[str],
            settings: object,
            graph: object,
        ) -> None:
            catchup_calls.append(project_name)

        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr(daemon_module, "ASTConsumer", lambda bus, graph, settings, **kw: FakeConsumer(name="ast-0"))
        monkeypatch.setattr(DaemonManager, "_catchup_vault", fake_catchup_vault)

        vault_dir = tmp_path / "vault"
        vault_dir.mkdir()
        settings = _make_settings(tmp_path)
        settings.knowledge.extra_vaults = [ExtraVaultSettings(path=str(vault_dir), project_name="test-vault")]

        manager = DaemonManager()
        started = await manager.start(settings, object(), include_watcher=False, catchup=True)  # type: ignore[arg-type]
        assert started is True
        await asyncio.sleep(0.05)

        assert catchup_calls == ["test-vault"]
        assert len(manager._vault_watchers) == 1
        assert manager._vault_watchers[0]._root_name == "test-vault"
        await manager.stop()

    async def test_missing_vault_path_is_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr(daemon_module, "ASTConsumer", lambda bus, graph, settings, **kw: FakeConsumer(name="ast-0"))

        settings = _make_settings(tmp_path)
        settings.knowledge.extra_vaults = [
            ExtraVaultSettings(path=str(tmp_path / "does-not-exist"), project_name="ghost-vault")
        ]

        manager = DaemonManager()
        started = await manager.start(settings, object(), include_watcher=False, catchup=True)  # type: ignore[arg-type]
        assert started is True
        assert manager._vault_watchers == []
        await manager.stop()


class TestVaultStartupIsolation:
    """A failing vault must not abort startup or the remaining vaults (Bug A regression).

    Mirrors the GC sweep's ``try/except Exception: logger.exception(...)`` pattern
    already used elsewhere in start() — a bad vault (malformed pathspec, permission
    error, bad path) should be logged and skipped, not propagate out of start().
    """

    async def test_one_bad_vault_does_not_abort_others(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bad_dir = tmp_path / "bad-vault"
        bad_dir.mkdir()
        good_dir = tmp_path / "good-vault"
        good_dir.mkdir()
        bad_resolved = bad_dir.resolve()

        class FakeScope:
            def __init__(self, root: Path, settings: object) -> None:
                self._root = root

            def scan(self) -> list[str]:
                if self._root == bad_resolved:
                    raise RuntimeError("malformed pathspec")
                return []

            def is_included(self, rel_path: str) -> bool:
                return True

        monkeypatch.setattr("code_atlas.backends.EventBus", FakeBus)
        monkeypatch.setattr(daemon_module, "ASTConsumer", lambda bus, graph, settings, **kw: FakeConsumer(name="ast-0"))
        monkeypatch.setattr(daemon_module, "FileScope", FakeScope)

        settings = _make_settings(tmp_path)
        settings.knowledge.extra_vaults = [
            ExtraVaultSettings(path=str(bad_dir), project_name="bad-vault"),
            ExtraVaultSettings(path=str(good_dir), project_name="good-vault"),
        ]

        manager = DaemonManager()
        started = await manager.start(settings, object(), include_watcher=False, catchup=True)  # type: ignore[arg-type]
        assert started is True

        # The bad vault must not have produced a watcher; the good vault still gets one.
        assert len(manager._vault_watchers) == 1
        assert manager._vault_watchers[0]._root_name == "good-vault"

        await manager.stop()


class TestDaemonCliWiring:
    """`atlas daemon start` must start the file watcher (the pipeline's producer)."""

    async def test_daemon_start_includes_watcher(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from code_atlas import cli

        captured: dict[str, object] = {}

        class FakeGraph:
            async def ping(self) -> bool:
                return True

            async def ensure_schema(self) -> None:
                return None

            async def close(self) -> None:
                return None

        class FakeDaemon:
            async def start(self, settings: object, graph: object, **kwargs: object) -> bool:
                captured.update(kwargs)
                return False  # short-circuit _run_daemon after capturing

        monkeypatch.setattr("code_atlas.graph.client.GraphClient", lambda settings: FakeGraph())
        monkeypatch.setattr("code_atlas.indexing.daemon.DaemonManager", FakeDaemon)
        monkeypatch.setattr(cli, "_load_settings", lambda: _make_settings(tmp_path))

        with pytest.raises(typer.Exit):
            await cli._run_daemon(no_embed=True)

        assert captured.get("include_watcher") is True
