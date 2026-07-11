"""Unit tests for consumer dedup and cooldown path identity (no infrastructure needed)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from code_atlas.events import FileChanged, Topic, encode_event
from code_atlas.indexing.consumers import _MAX_BATCH_FAILURES, ASTConsumer, BatchPolicy, TierConsumer
from code_atlas.parsing.ast import ParsedFile, ParsedRelationship
from code_atlas.schema import RelType
from code_atlas.settings import AtlasSettings

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.events import Event


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class RecordingBus:
    """Fake EventBus that records ACKs."""

    def __init__(self) -> None:
        self.acked: list[bytes] = []

    async def ack(self, topic: Topic, group: str, *msg_ids: bytes) -> int:
        self.acked.extend(msg_ids)
        return len(msg_ids)

    async def publish_many(self, topic: Topic, events: list[Event]) -> list[bytes]:
        return []


class StubGraph:
    """Minimal GraphClient substitute for deleted-file and resolution paths."""

    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []
        self.member_calls: list[tuple[str, list[ParsedRelationship]]] = []

    async def delete_file_entities(self, project_name: str, file_path: str) -> list[str]:
        self.deleted.append((project_name, file_path))
        return []

    async def build_resolution_lookup(self, project_name: str) -> tuple[Any, dict]:
        return object(), {}

    async def resolve_member_defines(
        self,
        project_name: str,
        member_rels: list[ParsedRelationship],
        *,
        lookup: Any = None,
        name_to_typedefs: dict | None = None,
    ) -> None:
        self.member_calls.append((project_name, list(member_rels)))


class FakeStreamBus:
    """In-memory stand-in for EventBus stream semantics (single topic/group)."""

    def __init__(self) -> None:
        self.stream: list[tuple[bytes, dict[bytes, bytes]]] = []
        self.pel: dict[bytes, dict[bytes, bytes]] = {}
        self.acked: list[bytes] = []
        self._next_id = 1

    def add(self, event: FileChanged) -> None:
        self.stream.append((f"{self._next_id}-0".encode(), encode_event(event)))
        self._next_id += 1

    async def ensure_group(self, topic: Topic, group: str) -> None:
        pass

    async def read_batch(
        self, topic: Topic, group: str, consumer: str, *, count: int, block_ms: int
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        await asyncio.sleep(0.01)
        batch = self.stream[:count]
        del self.stream[:count]
        self.pel.update(dict(batch))
        return batch

    async def read_pending(
        self, topic: Topic, group: str, consumer: str, *, count: int
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        return list(self.pel.items())[:count]

    async def ack(self, topic: Topic, group: str, *msg_ids: bytes) -> int:
        for mid in msg_ids:
            self.pel.pop(mid, None)
            self.acked.append(mid)
        return len(msg_ids)

    async def publish_many(self, topic: Topic, events: list[Event]) -> list[bytes]:
        return []


def _make_consumer(tmp_path: Path, *, cooldown_s: float = 0.0) -> ASTConsumer:
    return ASTConsumer(RecordingBus(), StubGraph(), AtlasSettings(project_root=tmp_path), cooldown_s=cooldown_s)  # type: ignore[arg-type]


def _event(path: str, project_name: str, project_root: str = "", change_type: str = "modified") -> FileChanged:
    return FileChanged(path=path, change_type=change_type, project_name=project_name, project_root=project_root)


# ---------------------------------------------------------------------------
# Dedup identity (S4) + PEL self-ACK guard (S7b)
# ---------------------------------------------------------------------------


async def test_dedup_key_scoped_by_project(tmp_path: Path) -> None:
    """Identical relative paths from different sub-projects must not collide."""
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev_a = _event("src/main.py", "mono/a")
    ev_b = _event("src/main.py", "mono/b")

    await consumer._dedup_put(pending, consumer.dedup_key(ev_a), b"1-0", ev_a)
    await consumer._dedup_put(pending, consumer.dedup_key(ev_b), b"2-0", ev_b)

    assert len(pending) == 2
    assert consumer.bus.acked == []  # type: ignore[attr-defined]


async def test_dedup_same_project_supersedes(tmp_path: Path) -> None:
    """Same path AND same project still dedups — first msg_id ACKed."""
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev1 = _event("src/main.py", "mono/a")
    ev2 = _event("src/main.py", "mono/a")

    await consumer._dedup_put(pending, consumer.dedup_key(ev1), b"1-0", ev1)
    await consumer._dedup_put(pending, consumer.dedup_key(ev2), b"2-0", ev2)

    assert len(pending) == 1
    assert consumer.bus.acked == [b"1-0"]  # type: ignore[attr-defined]


async def test_dedup_pel_reread_does_not_self_ack(tmp_path: Path) -> None:
    """A byte-identical msg_id (PEL re-read) must never ACK the retained message."""
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev = _event("src/main.py", "p")
    key = consumer.dedup_key(ev)

    await consumer._dedup_put(pending, key, b"1-0", ev)
    await consumer._dedup_put(pending, key, b"1-0", ev)

    assert consumer.bus.acked == []  # type: ignore[attr-defined]
    assert pending[key] == (b"1-0", ev)


async def test_dedup_reclaim_never_acks_newer_pending_message(tmp_path: Path) -> None:
    """Reclaim feeding an OLDER same-key PEL message must ACK only that older one.

    Retain-last-fed flip-flopped: feed m1 -> ACK m2, retain m1; feed m2 ->
    ACK m1, retain m2 — both XACKed while the retained pending entry had zero
    PEL coverage. Keep-newest must ACK m1 exactly once and keep m2 in the PEL.
    """
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev1 = _event("src/main.py", "p")
    ev2 = _event("src/main.py", "p")
    key = consumer.dedup_key(ev2)
    pending[key] = (b"5-0", ev2)  # newer message already held from a fresh read

    # PEL reclaim re-feeds every un-ACKed message in id order
    await consumer._dedup_put(pending, key, b"3-0", ev1)
    await consumer._dedup_put(pending, key, b"5-0", ev2)

    assert consumer.bus.acked == [b"3-0"]  # type: ignore[attr-defined]
    assert pending[key] == (b"5-0", ev2)


async def test_dedup_compares_stream_ids_numerically(tmp_path: Path) -> None:
    """b'9-0' is OLDER than b'10-0' despite lexicographic byte order."""
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev1 = _event("src/main.py", "p")
    ev2 = _event("src/main.py", "p")
    key = consumer.dedup_key(ev2)
    pending[key] = (b"10-0", ev2)

    await consumer._dedup_put(pending, key, b"9-0", ev1)

    assert consumer.bus.acked == [b"9-0"]  # type: ignore[attr-defined]
    assert pending[key] == (b"10-0", ev2)


async def test_dedup_supersession_prunes_fail_count(tmp_path: Path) -> None:
    """Every ACK path drops poison-tracking state — superseded msg_ids must not leak in _fail_counts."""
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev1 = _event("src/main.py", "p")
    ev2 = _event("src/main.py", "p")
    consumer._note_batch_failure([b"1-0"])
    assert consumer._fail_counts == {b"1-0": 1}

    await consumer._dedup_put(pending, consumer.dedup_key(ev1), b"1-0", ev1)
    await consumer._dedup_put(pending, consumer.dedup_key(ev2), b"2-0", ev2)

    assert consumer.bus.acked == [b"1-0"]  # type: ignore[attr-defined]
    assert consumer._fail_counts == {}


# ---------------------------------------------------------------------------
# ACK ordering / deferral (S7c)
# ---------------------------------------------------------------------------


async def test_ack_processed_only_acks_non_deferred(tmp_path: Path) -> None:
    consumer = _make_consumer(tmp_path)
    ev1 = _event("a.py", "p")
    ev2 = _event("b.py", "p")

    await consumer._ack_processed([ev1, ev2], [b"1-0", b"2-0"], {"p:b.py"})

    assert consumer.bus.acked == [b"1-0"]  # type: ignore[attr-defined]
    assert consumer._pel_dirty is True


async def test_dispatch_batch_retains_deferred_in_pel(tmp_path: Path) -> None:
    """A cooldown-deferred event must NOT be ACKed — it stays in the PEL."""
    consumer = _make_consumer(tmp_path, cooldown_s=60.0)
    root = str(tmp_path)

    await consumer._dispatch_batch([_event("src/x.py", "p", root, "deleted")], [b"1-0"], "b1")
    assert consumer.bus.acked == [b"1-0"]  # type: ignore[attr-defined]

    await consumer._dispatch_batch([_event("src/x.py", "p", root, "deleted")], [b"2-0"], "b2")
    assert consumer.bus.acked == [b"1-0"]  # type: ignore[attr-defined]
    assert consumer._pel_dirty is True


# ---------------------------------------------------------------------------
# Cooldown identity (S4)
# ---------------------------------------------------------------------------


async def test_cooldown_scoped_by_project(tmp_path: Path) -> None:
    """Project A's cooldown must not defer project B's same-relative-path event."""
    consumer = _make_consumer(tmp_path, cooldown_s=60.0)
    root = str(tmp_path)

    deferred = await consumer.process_batch([_event("src/main.py", "mono/a", root, "deleted")], "b1")
    assert deferred == set()

    deferred = await consumer.process_batch([_event("src/main.py", "mono/b", root, "deleted")], "b2")
    assert deferred == set()

    assert consumer.graph.deleted == [("mono/a", "src/main.py"), ("mono/b", "src/main.py")]  # type: ignore[attr-defined]


async def test_cooldown_defers_same_project(tmp_path: Path) -> None:
    """Re-sending the SAME project's file within the cooldown window still defers it."""
    consumer = _make_consumer(tmp_path, cooldown_s=60.0)
    root = str(tmp_path)

    assert await consumer.process_batch([_event("src/main.py", "mono/a", root, "deleted")], "b1") == set()

    deferred = await consumer.process_batch([_event("src/main.py", "mono/a", root, "deleted")], "b2")
    assert deferred == {"mono/a:src/main.py"}
    assert consumer.graph.deleted == [("mono/a", "src/main.py")]  # type: ignore[attr-defined]
    assert consumer.stats.files_deferred == 1


# ---------------------------------------------------------------------------
# Poison cap + PEL crash-recovery through run() (S7)
# ---------------------------------------------------------------------------


class FailingConsumer(TierConsumer):
    """process_batch always raises — every message is poison."""

    def __init__(self, bus: FakeStreamBus) -> None:
        super().__init__(
            bus=bus,  # type: ignore[arg-type]
            input_topic=Topic.FILE_CHANGED,
            group="ast",
            consumer_name="ast-0",
            policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=10),
        )
        self.attempts = 0

    async def process_batch(self, events: list[Event], batch_id: str) -> set[str] | None:
        self.attempts += 1
        raise RuntimeError("poison")


class FlakyConsumer(TierConsumer):
    """Fails the first process_batch call, succeeds afterwards."""

    def __init__(self, bus: FakeStreamBus) -> None:
        super().__init__(
            bus=bus,  # type: ignore[arg-type]
            input_topic=Topic.FILE_CHANGED,
            group="ast",
            consumer_name="ast-0",
            policy=BatchPolicy(time_window_s=0.3, max_batch_size=10, block_ms=10),
        )
        self.processed: list[Event] = []
        self._calls = 0

    async def process_batch(self, events: list[Event], batch_id: str) -> set[str] | None:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("first flush fails")
        self.processed.extend(events)
        return None


async def test_poison_message_parked_after_failure_cap() -> None:
    """A deterministically-failing message is parked (ACKed) after the cap, not retried forever."""
    bus = FakeStreamBus()
    bus.add(_event("poison.py", "p"))
    consumer = FailingConsumer(bus)

    task = asyncio.create_task(consumer.run())
    try:
        async with asyncio.timeout(5.0):
            while not bus.acked:
                await asyncio.sleep(0.05)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=5.0)

    assert bus.pel == {}
    assert consumer.attempts == _MAX_BATCH_FAILURES
    assert consumer._fail_counts == {}


async def test_pel_reclaimed_messages_survive_failed_first_flush() -> None:
    """Crash-recovery messages re-read from the PEL must not be self-ACKed before processing."""
    bus = FakeStreamBus()
    bus.add(_event("a.py", "p"))
    bus.add(_event("b.py", "p"))
    # Simulate a crashed prior run: messages delivered but never ACKed
    delivered = await bus.read_batch(Topic.FILE_CHANGED, "ast", "ast-0", count=10, block_ms=0)
    assert len(delivered) == 2
    assert len(bus.pel) == 2

    consumer = FlakyConsumer(bus)
    task = asyncio.create_task(consumer.run())
    try:
        async with asyncio.timeout(5.0):
            while len(consumer.processed) < 2:
                await asyncio.sleep(0.05)
    finally:
        consumer.stop()
        await asyncio.wait_for(task, timeout=5.0)

    assert {e.path for e in consumer.processed} == {"a.py", "b.py"}  # type: ignore[union-attr]
    assert bus.pel == {}


# ---------------------------------------------------------------------------
# Member-DEFINES routing (S5)
# ---------------------------------------------------------------------------


async def test_parse_file_partitions_member_rels(tmp_path: Path, monkeypatch) -> None:
    """DEFINES rels carrying parent_type_name go to member_rels, not non_import_rels."""
    consumer = _make_consumer(tmp_path)
    member = ParsedRelationship(
        from_qualified_name="p:pkg.routes",
        rel_type=RelType.DEFINES,
        to_name="p:pkg.routes.Server.Routes",
        properties={"parent_type_name": "Server", "parent_scope": "package"},
    )
    plain = ParsedRelationship(
        from_qualified_name="p:pkg.routes",
        rel_type=RelType.DEFINES,
        to_name="p:pkg.routes.helper",
    )
    fake = ParsedFile(file_path="pkg/routes.go", language="go", entities=[], relationships=[member, plain])
    monkeypatch.setattr("code_atlas.indexing.consumers.parse_file", lambda *a, **k: fake)

    pfd = await consumer._parse_file("p", "pkg/routes.go", source=b"")

    assert pfd is not None
    assert pfd.member_rels == [member]
    assert plain in pfd.non_import_rels
    assert member not in pfd.non_import_rels


async def test_flush_routes_member_rels_to_resolve_member_defines(tmp_path: Path) -> None:
    """Accumulated member rels are routed to GraphClient.resolve_member_defines on flush."""
    consumer = _make_consumer(tmp_path)
    rel = ParsedRelationship(
        from_qualified_name="proj:internal.server.routes",
        rel_type=RelType.DEFINES,
        to_name="proj:internal.server.routes.Server.Routes",
        properties={"parent_type_name": "Server", "parent_scope": "package"},
    )
    consumer._pending_member_rels.append(rel)
    consumer._pending_project_names.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.member_calls == [("proj", [rel])]  # type: ignore[attr-defined]
    assert consumer._pending_member_rels == []
