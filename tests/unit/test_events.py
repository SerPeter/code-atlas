"""Unit tests for EventBus durability behavior (no infrastructure needed)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.events import EventBus, FileChanged, Topic
from code_atlas.settings import RedisSettings

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakePipeline:
    """Records commands and replays canned results on execute()."""

    def __init__(self, fake: FakeRedis) -> None:
        self._fake = fake
        self._commands: list[tuple[str, str]] = []

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def xadd(
        self, key: str, fields: dict[bytes, bytes], *, maxlen: int | None = None, approximate: bool = True
    ) -> None:
        self._fake.xadds.append((key, maxlen))
        self._commands.append(("xadd", key))

    def xtrim(self, key: str, maxlen: int, approximate: bool = True) -> None:
        self._fake.xtrims.append((key, maxlen))
        self._commands.append(("xtrim", key))

    def xinfo_groups(self, key: str) -> None:
        self._commands.append(("xinfo_groups", key))

    async def execute(self) -> list[Any]:
        out: list[Any] = []
        for cmd, key in self._commands:
            if cmd == "xadd":
                out.append(b"1-0")
            elif cmd == "xinfo_groups":
                out.append(self._fake.groups_for(key))
            else:
                out.append(0)
        return out


class FakeRedis:
    """Minimal stand-in for redis.asyncio.Redis recording stream commands."""

    def __init__(self, *, groups: dict[str, list[dict[bytes, Any]]] | None = None) -> None:
        self.xadds: list[tuple[str, int | None]] = []
        self.xtrims: list[tuple[str, int]] = []
        self.destroyed: list[tuple[str, str]] = []
        self._groups = groups or {}

    def groups_for(self, key: str) -> list[dict[bytes, Any]]:
        return self._groups.get(key, [])

    async def xadd(
        self, key: str, fields: dict[bytes, bytes], *, maxlen: int | None = None, approximate: bool = True
    ) -> bytes:
        self.xadds.append((key, maxlen))
        return b"1-0"

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        return FakePipeline(self)

    async def xinfo_groups(self, key: str) -> list[dict[bytes, Any]]:
        return self.groups_for(key)

    async def xgroup_destroy(self, key: str, name: str) -> int:
        self.destroyed.append((key, name))
        return 1


def _make_bus(fake: FakeRedis, *, stream_maxlen: int = 1_000_000) -> EventBus:
    bus = EventBus(RedisSettings(stream_maxlen=stream_maxlen))
    bus._redis = fake
    return bus


def _event(path: str = "a.py") -> FileChanged:
    return FileChanged(path=path, change_type="modified")


def _maxlens(xadds: Sequence[tuple[str, int | None]]) -> list[int | None]:
    return [maxlen for _key, maxlen in xadds]


# ---------------------------------------------------------------------------
# Publish trim ceiling (S7 item d / contract #6)
# ---------------------------------------------------------------------------


class TestPublishMaxlen:
    async def test_publish_trims_to_settings_stream_maxlen(self) -> None:
        fake = FakeRedis()
        bus = _make_bus(fake, stream_maxlen=1_000_000)
        await bus.publish(Topic.FILE_CHANGED, _event())
        assert _maxlens(fake.xadds) == [1_000_000]

    async def test_publish_many_trims_to_settings_stream_maxlen(self) -> None:
        fake = FakeRedis()
        bus = _make_bus(fake, stream_maxlen=1_000_000)
        await bus.publish_many(Topic.FILE_CHANGED, [_event("a.py"), _event("b.py"), _event("c.py")])
        assert _maxlens(fake.xadds) == [1_000_000, 1_000_000, 1_000_000]

    async def test_stream_maxlen_zero_disables_trimming(self) -> None:
        fake = FakeRedis()
        bus = _make_bus(fake, stream_maxlen=0)
        await bus.publish(Topic.FILE_CHANGED, _event())
        await bus.publish_many(Topic.EMBED_DIRTY, [_event("b.py")])
        assert _maxlens(fake.xadds) == [None, None]

    async def test_callers_cannot_pass_their_own_maxlen(self) -> None:
        bus = _make_bus(FakeRedis())
        with pytest.raises(TypeError):
            await bus.publish(Topic.FILE_CHANGED, _event(), maxlen=10)  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            await bus.publish_many(Topic.FILE_CHANGED, [_event()], maxlen=10)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# NULL lag = unknown, never coerced to 0 (S7 item d / contract #4)
# ---------------------------------------------------------------------------


class TestStreamGroupInfoLag:
    async def test_null_lag_reported_as_none(self) -> None:
        key = "atlas:file-changed"
        fake = FakeRedis(groups={key: [{b"name": b"ast", b"pending": 3, b"lag": None}]})
        bus = _make_bus(fake)
        info = await bus.stream_group_info(Topic.FILE_CHANGED, "ast")
        assert info == {"pending": 3, "lag": None}

    async def test_null_lag_multi_reported_as_none(self) -> None:
        fake = FakeRedis(
            groups={
                "atlas:file-changed": [{b"name": b"ast", b"pending": 2, b"lag": None}],
                "atlas:embed-dirty": [{b"name": b"embed", b"pending": 0, b"lag": 7}],
            }
        )
        bus = _make_bus(fake)
        infos = await bus.stream_group_info_multi([(Topic.FILE_CHANGED, "ast"), (Topic.EMBED_DIRTY, "embed")])
        assert infos == [{"pending": 2, "lag": None}, {"pending": 0, "lag": 7}]

    async def test_integer_lag_stays_integer(self) -> None:
        key = "atlas:file-changed"
        fake = FakeRedis(groups={key: [{b"name": b"ast", b"pending": 1, b"lag": 4}]})
        bus = _make_bus(fake)
        info = await bus.stream_group_info(Topic.FILE_CHANGED, "ast")
        assert info == {"pending": 1, "lag": 4}

    async def test_missing_group_reports_zero(self) -> None:
        fake = FakeRedis(groups={"atlas:file-changed": [{b"name": b"other", b"pending": 9, b"lag": 9}]})
        bus = _make_bus(fake)
        assert await bus.stream_group_info(Topic.FILE_CHANGED, "ast") == {"pending": 0, "lag": 0}
        assert await bus.stream_group_info_multi([(Topic.FILE_CHANGED, "ast")]) == [{"pending": 0, "lag": 0}]


# ---------------------------------------------------------------------------
# Group-preserving flush (S7 item e / contract #3)
# ---------------------------------------------------------------------------


class TestFlushPreservesGroups:
    async def test_flush_trims_but_never_destroys_groups(self) -> None:
        fake = FakeRedis(
            groups={
                "atlas:file-changed": [{b"name": b"ast", b"pending": 5, b"lag": 10}],
                "atlas:embed-dirty": [{b"name": b"embed", b"pending": 1, b"lag": 2}],
            }
        )
        bus = _make_bus(fake)
        await bus.flush()
        assert fake.destroyed == []
        assert sorted(fake.xtrims) == [("atlas:embed-dirty", 0), ("atlas:file-changed", 0)]


# ---------------------------------------------------------------------------
# Indexer lease
# ---------------------------------------------------------------------------


class FakeLeaseBus:
    """A bus whose lease is held by someone else for the first *busy_for* attempts."""

    def __init__(self, busy_for: float = 0, holder: str = "host:999:abc") -> None:
        self.busy_for = busy_for
        self.holder = holder
        self.attempts = 0
        self.released = False

    async def acquire_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        self.attempts += 1
        return self.attempts > self.busy_for

    async def read_indexer_lease(self) -> str:
        return self.holder

    async def renew_indexer_lease(self, owner: str, ttl_ms: int) -> bool:
        return True

    async def release_indexer_lease(self, owner: str) -> bool:
        self.released = True
        return True


class FakeClock:
    """Drives the lease deadline from the sleeps the code asks for.

    Without this the test either spins for `wait_s` of real time or -- worse -- never
    reaches the deadline at all, because instant sleeps let it retry millions of times
    and eventually satisfy any finite `busy_for`. Both were mistakes in the first draft
    of these tests.
    """

    def __init__(self) -> None:
        self.now = 1000.0
        self.delays: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    from code_atlas import events

    c = FakeClock()
    monkeypatch.setattr(events.time, "monotonic", c.monotonic)
    monkeypatch.setattr("asyncio.sleep", c.sleep)
    return c


class TestIndexerLeaseWaiting:
    """Standing down on the first refusal assumes the holder will do the work.

    That assumption broke in the field: a reconnecting MCP server collided with the
    lease of a process that was then killed, so the holder never finished, its lease
    expired unnoticed, and nothing re-triggered the catch-up. The index sat idle with
    30k embeddings outstanding and no error anywhere.
    """

    async def test_a_waiter_takes_the_lease_once_it_frees(self, clock) -> None:
        from code_atlas.events import hold_indexer_lease

        bus = FakeLeaseBus(busy_for=3)

        async with hold_indexer_lease(bus, wait_s=600) as owner:
            assert owner

        assert bus.attempts == 4, "should have retried until the holder released"
        assert bus.released

    async def test_zero_wait_is_exactly_one_attempt(self, clock) -> None:
        """The previous behaviour has to stay reachable: a human at a terminal may want
        the refusal rather than a wait."""
        from code_atlas.events import IndexerBusyError, hold_indexer_lease

        bus = FakeLeaseBus(busy_for=1)

        with pytest.raises(IndexerBusyError):
            async with hold_indexer_lease(bus, wait_s=0):
                pass

        assert bus.attempts == 1
        assert clock.delays == [], "fail-fast must not sleep at all"

    async def test_giving_up_names_the_holder(self, clock) -> None:
        from code_atlas.events import IndexerBusyError, hold_indexer_lease

        bus = FakeLeaseBus(busy_for=float("inf"), holder="host:123:deadbeef")

        with pytest.raises(IndexerBusyError) as exc:
            async with hold_indexer_lease(bus, wait_s=30):
                pass

        assert exc.value.holder == "host:123:deadbeef"
        assert bus.attempts > 1, "should have retried before giving up"

    async def test_retries_are_jittered(self, clock) -> None:
        """Several MCP sessions in one worktree start within milliseconds of each other
        -- an agent client spawns them together. Retrying on a fixed interval keeps them
        in lockstep, and the same one tends to win every round."""
        from code_atlas import events
        from code_atlas.events import hold_indexer_lease

        bus = FakeLeaseBus(busy_for=6)

        async with hold_indexer_lease(bus, wait_s=600):
            pass

        assert len(clock.delays) == 6
        assert len(set(clock.delays)) > 1, "every retry waited the same -- no jitter"
        low = events._LEASE_POLL_S * (1 - events._LEASE_POLL_JITTER)
        high = events._LEASE_POLL_S * (1 + events._LEASE_POLL_JITTER)
        assert all(low <= d <= high for d in clock.delays), clock.delays

    async def test_no_sleep_overshoots_the_budget(self, clock) -> None:
        """A caller that asked for half a second must not be held for two."""
        from code_atlas.events import IndexerBusyError, hold_indexer_lease

        bus = FakeLeaseBus(busy_for=float("inf"))

        with pytest.raises(IndexerBusyError):
            async with hold_indexer_lease(bus, wait_s=0.5):
                pass

        assert clock.delays, "should have slept at least once"
        assert all(d <= 0.5 for d in clock.delays), clock.delays
        assert clock.now <= 1000.5
