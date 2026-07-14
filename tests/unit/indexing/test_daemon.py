"""Unit tests for DaemonManager supervision, startup catch-up, and shutdown ordering."""

from __future__ import annotations

import asyncio
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
    """EventBus stand-in that always pings OK."""

    def __init__(self, settings: object, *, project_name: str = "") -> None:
        self.project_name = project_name

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


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

        async def fake_index_project(settings: object, graph: object, bus: object) -> None:
            if state["fail_catchup"]:
                raise RuntimeError("catch-up exploded")
            state["order"].append("catchup-project")

        async def fake_index_monorepo(settings: object, graph: object, bus: object) -> None:
            state["order"].append("catchup-monorepo")

        monkeypatch.setattr(daemon_module, "EventBus", FakeBus)
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

        monkeypatch.setattr(daemon_module, "EventBus", FakeBus)
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

        monkeypatch.setattr(daemon_module, "EventBus", FakeBus)
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
        monkeypatch.setattr(daemon_module, "EventBus", FakeBus)
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
