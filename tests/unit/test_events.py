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
