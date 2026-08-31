"""Unit tests for consumer dedup and cooldown path identity (no infrastructure needed)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from code_atlas.events import FileChanged, Topic, encode_event
from code_atlas.graph.client import UpsertResult
from code_atlas.indexing.consumers import (
    _MAX_BATCH_FAILURES,
    _RESOLVE_DUTY_RATIO,
    _RESOLVE_MAX_GAP_S,
    ASTConsumer,
    BatchPolicy,
    TierConsumer,
    _compute_file_hash,
    _next_resolve_gap,
    _rels_hash,
    _retry_key,
)
from code_atlas.parsing.ast import ParsedEntity, ParsedFile, ParsedRelationship
from code_atlas.schema import RelType
from code_atlas.settings import AtlasSettings, EmbeddingSettings, IndexSettings

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
        self.config_calls: list[tuple[str, list[ParsedRelationship]]] = []
        self.citation_calls: list[tuple[str, dict[str, list[str]], set[str] | None, bool]] = []
        self.hash_writes: list[tuple[str, dict[str, str]]] = []
        self.rels_hash_writes: list[tuple[str, dict[str, str]]] = []
        self.gc_calls: int = 0
        self.embed_chunk_gc_calls: int = 0

    async def delete_file_entities(self, project_name: str, file_path: str) -> list[str]:
        self.deleted.append((project_name, file_path))
        return []

    async def get_batch_file_hashes(self, project_name: str, file_paths: list[str]) -> dict[str, str]:
        return {}

    async def set_batch_file_hashes(self, project_name: str, hashes: dict[str, str]) -> None:
        self.hash_writes.append((project_name, dict(hashes)))

    async def get_batch_rels_hashes(self, project_name: str, file_paths: list[str]) -> dict[str, str | None]:
        return {}

    async def set_batch_rels_hashes(self, project_name: str, rels_hashes: dict[str, str]) -> None:
        self.rels_hash_writes.append((project_name, dict(rels_hashes)))

    async def upsert_batch_entities(
        self,
        project_name: str,
        file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]],
        *,
        rels_only: bool = False,
        rels_unchanged: Any = (),
    ) -> dict[str, UpsertResult]:
        # ``added`` carries qualified_names with the project prefix stripped,
        # the way the real delta classifier reports them.
        return {
            fp: UpsertResult(added=[e.qualified_name.split(":", 1)[1] for e in entities])
            for fp, (entities, _rels) in file_data.items()
        }

    async def invalidate_stale_anchors(self, changed_uids: set[str]) -> int:
        return 0

    async def resolve_config_refs(self, project_name: str, ref_rels: list[ParsedRelationship]) -> None:
        self.config_calls.append((project_name, list(ref_rels)))

    async def gc_orphaned_reference_nodes(self) -> int:
        self.gc_calls += 1
        return 0

    async def gc_orphaned_embed_chunks(self, project_name: str = "") -> int:
        self.embed_chunk_gc_calls += 1
        return 0

    async def resolve_citations(
        self,
        project_name: str,
        citations_by_uid: dict[str, list[str]],
        *,
        file_paths: Any = None,
        lookup: Any = None,
        retry_unresolved: bool = False,
    ) -> None:
        scope = set(file_paths) if file_paths is not None else None
        self.citation_calls.append((project_name, dict(citations_by_uid), scope, retry_unresolved))

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

    async def reclaim_abandoned(
        self, topic: Topic, group: str, consumer: str, *, min_idle_ms: int, count: int
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """No abandoned entries in a single-consumer fake — this fake IS the only consumer.

        Present because consumer names now carry a process identity, so the loop reclaims
        another process's orphaned PEL after draining its own.
        """
        return []

    async def ack(self, topic: Topic, group: str, *msg_ids: bytes) -> int:
        for mid in msg_ids:
            self.pel.pop(mid, None)
            self.acked.append(mid)
        return len(msg_ids)

    async def publish_many(self, topic: Topic, events: list[Event]) -> list[bytes]:
        return []


def _make_consumer(tmp_path: Path, *, cooldown_s: float = 0.0) -> ASTConsumer:
    return ASTConsumer(RecordingBus(), StubGraph(), AtlasSettings(project_root=tmp_path), cooldown_s=cooldown_s)  # ty: ignore[invalid-argument-type]


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
    assert consumer.bus.acked == []  # ty: ignore[unresolved-attribute]


async def test_dedup_same_project_supersedes(tmp_path: Path) -> None:
    """Same path AND same project still dedups — first msg_id ACKed."""
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev1 = _event("src/main.py", "mono/a")
    ev2 = _event("src/main.py", "mono/a")

    await consumer._dedup_put(pending, consumer.dedup_key(ev1), b"1-0", ev1)
    await consumer._dedup_put(pending, consumer.dedup_key(ev2), b"2-0", ev2)

    assert len(pending) == 1
    assert consumer.bus.acked == [b"1-0"]  # ty: ignore[unresolved-attribute]


async def test_dedup_pel_reread_does_not_self_ack(tmp_path: Path) -> None:
    """A byte-identical msg_id (PEL re-read) must never ACK the retained message."""
    consumer = _make_consumer(tmp_path)
    pending: dict[str, tuple[bytes, Event]] = {}
    ev = _event("src/main.py", "p")
    key = consumer.dedup_key(ev)

    await consumer._dedup_put(pending, key, b"1-0", ev)
    await consumer._dedup_put(pending, key, b"1-0", ev)

    assert consumer.bus.acked == []  # ty: ignore[unresolved-attribute]
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

    assert consumer.bus.acked == [b"3-0"]  # ty: ignore[unresolved-attribute]
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

    assert consumer.bus.acked == [b"9-0"]  # ty: ignore[unresolved-attribute]
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

    assert consumer.bus.acked == [b"1-0"]  # ty: ignore[unresolved-attribute]
    assert consumer._fail_counts == {}


# ---------------------------------------------------------------------------
# ACK ordering / deferral (S7c)
# ---------------------------------------------------------------------------


async def test_ack_processed_only_acks_non_deferred(tmp_path: Path) -> None:
    consumer = _make_consumer(tmp_path)
    ev1 = _event("a.py", "p")
    ev2 = _event("b.py", "p")

    await consumer._ack_processed([ev1, ev2], [b"1-0", b"2-0"], {"p:b.py"})

    assert consumer.bus.acked == [b"1-0"]  # ty: ignore[unresolved-attribute]
    assert consumer._pel_dirty is True


async def test_dispatch_batch_retains_deferred_in_pel(tmp_path: Path) -> None:
    """A cooldown-deferred event must NOT be ACKed — it stays in the PEL."""
    consumer = _make_consumer(tmp_path, cooldown_s=60.0)
    root = str(tmp_path)

    await consumer._dispatch_batch([_event("src/x.py", "p", root, "deleted")], [b"1-0"], "b1")
    assert consumer.bus.acked == [b"1-0"]  # ty: ignore[unresolved-attribute]

    await consumer._dispatch_batch([_event("src/x.py", "p", root, "deleted")], [b"2-0"], "b2")
    assert consumer.bus.acked == [b"1-0"]  # ty: ignore[unresolved-attribute]
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

    assert consumer.graph.deleted == [("mono/a", "src/main.py"), ("mono/b", "src/main.py")]  # ty: ignore[unresolved-attribute]


async def test_cooldown_defers_same_project(tmp_path: Path) -> None:
    """Re-sending the SAME project's file within the cooldown window still defers it."""
    consumer = _make_consumer(tmp_path, cooldown_s=60.0)
    root = str(tmp_path)

    assert await consumer.process_batch([_event("src/main.py", "mono/a", root, "deleted")], "b1") == set()

    deferred = await consumer.process_batch([_event("src/main.py", "mono/a", root, "deleted")], "b2")
    assert deferred == {"mono/a:src/main.py"}
    assert consumer.graph.deleted == [("mono/a", "src/main.py")]  # ty: ignore[unresolved-attribute]
    assert consumer.stats.files_deferred == 1


# ---------------------------------------------------------------------------
# Poison cap + PEL crash-recovery through run() (S7)
# ---------------------------------------------------------------------------


class FailingConsumer(TierConsumer):
    """process_batch always raises — every message is poison."""

    def __init__(self, bus: FakeStreamBus) -> None:
        super().__init__(
            bus=bus,  # ty: ignore[invalid-argument-type]
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
            bus=bus,  # ty: ignore[invalid-argument-type]
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

    assert {e.path for e in consumer.processed} == {"a.py", "b.py"}  # ty: ignore[unresolved-attribute]
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

    assert consumer.graph.member_calls == [("proj", [rel])]  # ty: ignore[unresolved-attribute]
    assert consumer._pending_member_rels == []


async def test_parse_file_partitions_config_rels(tmp_path: Path, monkeypatch) -> None:
    """READS_ENV/REFERENCES_FILE are deferred like imports — their target node
    does not exist until post-batch resolution MERGEs it, so they must never
    reach the immediate relationship-creation path.
    """
    consumer = _make_consumer(tmp_path)
    env = ParsedRelationship(from_qualified_name="p:conf.load", rel_type=RelType.READS_ENV, to_name="DATABASE_URL")
    res = ParsedRelationship(
        from_qualified_name="p:conf.load", rel_type=RelType.REFERENCES_FILE, to_name="data/fixtures.json"
    )
    plain = ParsedRelationship(from_qualified_name="p:conf", rel_type=RelType.DEFINES, to_name="p:conf.load")
    fake = ParsedFile(file_path="conf.py", language="python", entities=[], relationships=[env, res, plain])
    monkeypatch.setattr("code_atlas.indexing.consumers.parse_file", lambda *a, **k: fake)

    pfd = await consumer._parse_file("p", "conf.py", source=b"")

    assert pfd is not None
    assert pfd.config_rels == [env, res]
    assert pfd.non_import_rels == [plain]


async def test_parse_file_drops_references_to_directories_but_keeps_missing_files(tmp_path: Path, monkeypatch) -> None:
    """A directory path in a string literal is indistinguishable from a file path to
    the parser, which is pure and does no I/O. ``.atlas`` (a Path-typed settings
    default) became a ResourceFile node this way. A path that simply does not exist is
    kept — an unresolved reference to a data file is what this node type is for.
    """
    (tmp_path / "somedir").mkdir()
    consumer = _make_consumer(tmp_path)
    a_dir = ParsedRelationship(from_qualified_name="p:conf.load", rel_type=RelType.REFERENCES_FILE, to_name="somedir")
    absent = ParsedRelationship(
        from_qualified_name="p:conf.load", rel_type=RelType.REFERENCES_FILE, to_name="data/generated.json"
    )
    env = ParsedRelationship(from_qualified_name="p:conf.load", rel_type=RelType.READS_ENV, to_name="DATABASE_URL")
    fake = ParsedFile(file_path="conf.py", language="python", entities=[], relationships=[a_dir, absent, env])
    monkeypatch.setattr("code_atlas.indexing.consumers.parse_file", lambda *a, **k: fake)

    pfd = await consumer._parse_file("p", "conf.py", source=b"")

    assert pfd is not None
    assert pfd.config_rels == [absent, env]


async def test_flush_resolves_config_refs_then_runs_gc(tmp_path: Path) -> None:
    """Order matters: the sweep deletes anything at zero incoming edges, so it
    must run only after this flush has re-created its references.
    """
    consumer = _make_consumer(tmp_path)
    rel = ParsedRelationship(from_qualified_name="proj:conf.load", rel_type=RelType.READS_ENV, to_name="DATABASE_URL")
    consumer._pending_config_rels.append(rel)
    consumer._pending_project_names.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.config_calls == [("proj", [rel])]  # ty: ignore[unresolved-attribute]
    assert consumer.graph.gc_calls == 1  # ty: ignore[unresolved-attribute]
    assert consumer._pending_config_rels == []


async def test_flush_runs_gc_even_with_no_config_rels(tmp_path: Path) -> None:
    """A file whose LAST os.getenv() call was just deleted produces no config
    rels at all — and that is exactly the case that orphans a node. Gating the
    sweep on config rels being present would never collect it.
    """
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.config_calls == []  # ty: ignore[unresolved-attribute]
    assert consumer.graph.gc_calls == 1  # ty: ignore[unresolved-attribute]


async def test_flush_skips_gc_when_nothing_was_processed(tmp_path: Path) -> None:
    consumer = _make_consumer(tmp_path)

    await consumer._flush_deferred_resolution()

    assert consumer.graph.gc_calls == 0  # ty: ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Citation retry sweep
# ---------------------------------------------------------------------------


async def test_indexing_a_document_retries_that_projects_unresolved_citations(tmp_path: Path) -> None:
    """The live trigger. A daemon indexes the ADR long after the code that
    cites it; without re-attempting on that event the citation stays broken
    until the process restarts (the end-of-run sweep is shutdown-only).
    """
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")
    consumer._citation_retry_projects.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.citation_calls == [("proj", {}, None, True)]  # ty: ignore[unresolved-attribute]
    assert consumer._citation_retry_projects == set()


async def test_new_citations_and_a_document_change_resolve_in_one_pass(tmp_path: Path) -> None:
    """The batch's own citations ride along with the retry scan rather than
    costing a second project-wide pass."""
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")
    consumer._pending_citations["proj"] = {"proj:src.mod.f": ["ADR-0014"]}
    consumer._pending_citation_files["proj"] = {"src/mod.py"}
    consumer._citation_retry_projects.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.citation_calls == [  # ty: ignore[unresolved-attribute]
        ("proj", {"proj:src.mod.f": ["ADR-0014"]}, {"src/mod.py"}, True)
    ]


async def test_flush_without_documents_or_citations_does_not_sweep(tmp_path: Path) -> None:
    """Steady-state daemon flushes must not pay for a project-wide scan."""
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.citation_calls == []  # ty: ignore[unresolved-attribute]


async def test_a_reparsed_file_with_no_citations_still_reaches_the_resolver(tmp_path: Path) -> None:
    """The removal signal. A file whose last `see ADR-14` comment was deleted
    contributes nothing to _pending_citations, so gating the call on that dict
    is exactly why the stale edge used to survive — the file scope has to carry
    the call on its own."""
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")
    consumer._pending_citation_files["proj"] = {"src/mod.py"}

    await consumer._flush_deferred_resolution()

    assert consumer.graph.citation_calls == [("proj", {}, {"src/mod.py"}, False)]  # ty: ignore[unresolved-attribute]
    assert consumer._pending_citation_files == {}


async def test_process_batch_scopes_every_parsed_file_not_just_citing_ones(tmp_path: Path) -> None:
    """The scope is built from the parse, not from the citations it produced."""
    settings = AtlasSettings(project_root=tmp_path, embeddings=EmbeddingSettings(enabled=False))
    consumer = ASTConsumer(RecordingBus(), StubGraph(), settings)  # ty: ignore[invalid-argument-type]
    (tmp_path / "cited.py").write_text("# WHY: see ADR-0014\ndef f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    events = [_event("cited.py", "proj", str(tmp_path)), _event("plain.py", "proj", str(tmp_path))]
    await consumer.process_batch(events, "b1")  # ty: ignore[invalid-argument-type]  # a list of one event subtype, which the signature widens

    # The batch's own flush already drained the buffers, so assert on what
    # actually reached the resolver.
    assert consumer.graph.citation_calls == [  # ty: ignore[unresolved-attribute]
        ("proj", {"proj:cited.f": ["ADR-0014"]}, {"cited.py", "plain.py"}, False)
    ]


async def test_final_flush_sweeps_every_project_that_saw_a_citation(tmp_path: Path) -> None:
    """Backstop for the cold index: the document may have been in the graph
    before this run started, so no document-change event ever fires."""
    consumer = _make_consumer(tmp_path)
    consumer._citation_projects.add("proj")

    await consumer._flush_deferred_resolution(final=True)

    assert consumer.graph.citation_calls == [("proj", {}, None, True)]  # ty: ignore[unresolved-attribute]
    assert consumer._citation_projects == set()


# ---------------------------------------------------------------------------
# File-hash withholding vs. the deferred citation revoke (ATL-090)
# ---------------------------------------------------------------------------


class _UnchangedGraph(StubGraph):
    """Upserts that classify every entity as ``unchanged``.

    The shape of a file the hash gate let through whose entities all matched
    their stored ``content_hash`` — the file's bytes moved, nothing semantic
    did. Citations are part of ``content_hash``, so this is provably a file
    with no citation to revoke or recreate.
    """

    async def upsert_batch_entities(
        self,
        project_name: str,
        file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]],
        *,
        rels_only: bool = False,
        rels_unchanged: Any = (),
    ) -> dict[str, UpsertResult]:
        return {
            fp: UpsertResult(unchanged=[e.qualified_name.split(":", 1)[1] for e in entities])
            for fp, (entities, _rels) in file_data.items()
        }


def _reindex_consumer(tmp_path: Path, graph: StubGraph) -> ASTConsumer:
    """Consumer on the reindex policy, warmed past its spurious first flush.

    ``time_window_s=0`` sets ``_resolve_batch_interval=5``, so a single batch
    does NOT flush — which is what makes ``_pending_file_hashes`` observable.
    """
    return ASTConsumer(
        RecordingBus(),  # ty: ignore[invalid-argument-type]
        graph,  # ty: ignore[invalid-argument-type]
        AtlasSettings(project_root=tmp_path, embeddings=EmbeddingSettings(enabled=False)),
        policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
    )


async def test_a_file_whose_only_deferred_work_is_the_revoke_withholds_its_hash(tmp_path: Path) -> None:
    """A file with no imports/calls/type-refs and no citations left still has
    deferred work — it is in the revoke scope handed to resolve_citations.
    Writing its hash in process_batch would let a crash before the flush strand
    the stale edge behind a hash gate that now believes the file current.
    """
    graph = StubGraph()
    consumer = _reindex_consumer(tmp_path, graph)
    (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    await consumer.process_batch([], "warmup")  # absorbs the spurious first-batch flush
    await consumer.process_batch([_event("plain.py", "proj", str(tmp_path))], "b1")

    assert "plain.py" in consumer._pending_file_hashes.get("proj", {})
    assert graph.hash_writes == []


async def test_an_unchanged_file_still_withholds_its_hash_until_the_flush(tmp_path: Path) -> None:
    """Inverted from the assertion this replaces, which pinned the bug.

    The old behaviour wrote the hash immediately when every entity came back
    unchanged, on the argument that unchanged entities have "provably identical
    citations" so the revoke is a no-op. That holds only if the previous run
    FINISHED. Step 3's upsert has already advanced the stored content_hash, so
    after an interrupted run the delta compares against a partially-applied
    state: the file reports unchanged, takes the immediate path, and a second
    crash strands its citation edge permanently (ATL-090).

    Withholding unconditionally is strictly more conservative — recovery
    re-parses more, never less — and costs no extra write, because the flush
    already issues one set_batch_file_hashes per project.
    """
    graph = _UnchangedGraph()
    consumer = _reindex_consumer(tmp_path, graph)
    (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    await consumer.process_batch([], "warmup")
    await consumer.process_batch([_event("plain.py", "proj", str(tmp_path))], "b1")

    assert "plain.py" in consumer._pending_file_hashes.get("proj", {})
    assert graph.hash_writes == [], "no hash may be written before the deferred flush"


async def test_hash_survives_only_after_the_flush_so_a_crash_loop_cannot_strand_work(tmp_path: Path) -> None:
    """The crash-loop case: interrupted run, then an unchanged-looking re-parse.

    Simulates run N crashing after process_batch but before the flush, then run
    N+1 seeing every entity as unchanged (because run N's upsert advanced the
    stored hash). Run N+1 must STILL withhold, or a second crash gates the file
    out forever.
    """
    graph = _UnchangedGraph()
    consumer = _reindex_consumer(tmp_path, graph)
    (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")
    event = _event("plain.py", "proj", str(tmp_path))

    await consumer.process_batch([], "warmup")

    # Run N: processed, then "crash" — the flush never runs.
    await consumer.process_batch([event], "b1")
    assert graph.hash_writes == []
    consumer._pending_file_hashes.clear()  # the crash loses in-memory state

    # Run N+1: entities now look unchanged. It must not shortcut to a hash write.
    await consumer.process_batch([event], "b2")
    assert graph.hash_writes == [], "an unchanged-looking recovery run wrote the hash before its flush"
    assert "plain.py" in consumer._pending_file_hashes.get("proj", {})


class TestTheRelationshipFingerprint:
    """ATL-151: what a skipped relationship rewrite is allowed to trust.

    ``_rels_hash`` decides whether TX2's delete-then-recreate for a file can be left
    out entirely, so anything it cannot see is an edge that silently stops being
    updated.
    """

    @staticmethod
    def _rel(to_name: str, **props: object) -> ParsedRelationship:
        return ParsedRelationship(
            from_qualified_name="proj:mod.caller",
            rel_type=RelType.DEFINES,
            to_name=to_name,
            properties=dict(props),
        )

    def test_it_is_order_independent(self) -> None:
        """A parser that emits the same edges in a different order has not changed the
        graph, and must not trigger a rewrite of every edge in the file."""
        a, b, c = self._rel("x"), self._rel("y"), self._rel("z")

        assert _rels_hash([a, b, c]) == _rels_hash([c, a, b])

    def test_it_covers_properties_that_retry_key_ignores(self) -> None:
        """The reason this is not ``_retry_key``. That key is for deduplicating the
        replay buffer and reads only ``receiver``/``receiver_type``; a fingerprint
        narrower than the write lets an edge keep a stale property forever."""
        weight_5 = self._rel("x", weight=5)
        weight_9 = self._rel("x", weight=9)

        assert _retry_key(weight_5) == _retry_key(weight_9), "precondition: _retry_key cannot see this"
        assert _rels_hash([weight_5]) != _rels_hash([weight_9])

    def test_a_rel_type_change_alone_moves_it(self) -> None:
        same_endpoints = ParsedRelationship(
            from_qualified_name="proj:mod.caller",
            rel_type=RelType.CONTAINS,
            to_name="x",
        )

        assert _rels_hash([self._rel("x")]) != _rels_hash([same_endpoints])

    def test_the_endpoints_are_separated_rather_than_concatenated(self) -> None:
        """Guards the digest's framing: ``("ab", "c")`` and ``("a", "bc")`` describe
        different edges and must not collide."""
        assert _rels_hash([self._rel("x")]) != _rels_hash(
            [
                ParsedRelationship(
                    from_qualified_name="proj:mod.calle",
                    rel_type=RelType.DEFINES,
                    to_name="rx",
                )
            ]
        )


async def test_the_rels_hash_is_withheld_until_the_flush_like_the_file_hash(tmp_path: Path) -> None:
    """DEVIATION from a naive reading of ATL-151, and a sharper hazard than the file
    hash's own (ATL-090).

    A ``rels_hash`` written in ``process_batch`` describes a rel set whose deferred half
    is not resolved yet. After a crash in between, the next run re-parses the file --
    its ``file_hash`` is correctly unset -- and then declines to rewrite its
    relationships because the fingerprint matches. That is a file which visibly
    re-parses on every run while permanently missing edges, which is undiagnosable from
    outside. Both hashes therefore ride the same schedule.
    """
    graph = _UnchangedGraph()
    consumer = _reindex_consumer(tmp_path, graph)
    (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    await consumer.process_batch([], "warmup")
    await consumer.process_batch([_event("plain.py", "proj", str(tmp_path))], "b1")

    assert "plain.py" in consumer._pending_rels_hashes.get("proj", {})
    assert graph.rels_hash_writes == [], "no relationship fingerprint may be written before the deferred flush"

    await consumer._flush_deferred_resolution()
    assert [p for p, _ in graph.rels_hash_writes] == ["proj"]
    assert "plain.py" in graph.rels_hash_writes[0][1]


class TestTheGateKeyCoversExtraction:
    """ATL-152: the stored file_hash keys on the extraction contract, not on bytes alone.

    Hashing bytes alone made the gate's key narrower than the thing it gates. Raising
    ``index.max_source_chars`` on 2026-08-31 left 2,169 entities holding truncated source
    while the gate insisted every file was current, because no file's bytes had moved.
    """

    def test_a_moved_extraction_key_moves_the_file_hash(self) -> None:
        """The sabotage anchor: stop folding the key in and this is what fails."""
        source = b"def g():\n    return 2\n"

        assert _compute_file_hash(source, extraction_key="k1") == _compute_file_hash(source, extraction_key="k1")
        assert _compute_file_hash(source, extraction_key="k1") != _compute_file_hash(source, extraction_key="k2")

    def test_an_empty_key_is_a_no_op_rather_than_a_distinct_input(self) -> None:
        """So the function still means "the hash of these bytes" for a caller that does not
        gate on extraction. ``extraction_key`` never returns the empty string, so this
        cannot silently disable the gate's coverage in production.
        """
        source = b"def g():\n    return 2\n"

        assert _compute_file_hash(source, extraction_key="") == _compute_file_hash(source)
        assert _compute_file_hash(source, strip_whitespace=False, extraction_key="") == _compute_file_hash(
            source, strip_whitespace=False
        )

    async def test_the_consumer_hashes_under_its_own_extraction_key(self, tmp_path: Path) -> None:
        """The wiring, not just the signature.

        Two consumers differing only in an extraction-affecting setting must record
        different hashes for byte-identical input — otherwise the config half of the key
        is computed and then dropped on the floor.
        """
        (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")
        event = _event("plain.py", "proj", str(tmp_path))

        def _consumer(index: IndexSettings) -> ASTConsumer:
            return ASTConsumer(
                RecordingBus(),  # ty: ignore[invalid-argument-type]
                StubGraph(),  # ty: ignore[invalid-argument-type]
                AtlasSettings(project_root=tmp_path, index=index, embeddings=EmbeddingSettings(enabled=False)),
                policy=BatchPolicy(time_window_s=0, max_batch_size=10, block_ms=50),
            )

        narrow = _consumer(IndexSettings(max_source_chars=2000))
        wide = _consumer(IndexSettings(max_source_chars=48_000))
        for consumer in (narrow, wide):
            await consumer.process_batch([], "warmup")
            await consumer.process_batch([event], "b1")

        assert narrow._pending_file_hashes["proj"]["plain.py"] != wide._pending_file_hashes["proj"]["plain.py"]


class TestAdaptiveResolveCadence:
    """A resolution flush costs O(project size), and the number of flushes grows with
    the project too, so a fixed every-5-batches cadence spends a growing share of a
    reindex on resolution. The gap after each flush is therefore scaled by what that
    flush actually cost.
    """

    @staticmethod
    def _reindex(tmp_path: Path) -> ASTConsumer:
        # Reuses the existing reindex-policy helper: time_window_s=0 is what sets
        # _resolve_batch_interval=5, and therefore what makes _resolve_adaptive true.
        return _reindex_consumer(tmp_path, StubGraph())

    def test_adaptive_only_in_reindex_mode(self, tmp_path: Path):
        """Watch mode keeps its fixed cadence: `final` there only arrives at shutdown,
        so a stretched gap would delay one edited file's edges by that gap."""
        assert self._reindex(tmp_path)._resolve_adaptive is True
        assert _make_consumer(tmp_path)._resolve_adaptive is False

    async def test_a_cheap_flush_sets_no_meaningful_gap(self, tmp_path: Path):
        """Small projects must be unaffected — a fast flush yields a gap far below the
        batch cadence that triggers it, so the controller never engages."""
        consumer = self._reindex(tmp_path)
        await consumer._flush_deferred_resolution()
        assert consumer._resolve_min_gap_s < 1.0

    def test_an_expensive_flush_stretches_the_gap(self):
        """The gap is a multiple of the measured duration, so resolution stays roughly
        1/(1+ratio) of wall time however large the project gets."""
        assert _next_resolve_gap(3.0) == pytest.approx(3.0 * _RESOLVE_DUTY_RATIO)
        assert _next_resolve_gap(0.05) == pytest.approx(0.05 * _RESOLVE_DUTY_RATIO)

    def test_the_gap_is_capped(self):
        """A pathologically slow flush must not stall resolution indefinitely."""
        assert _next_resolve_gap(10_000.0) == _RESOLVE_MAX_GAP_S

    def test_a_negative_duration_cannot_produce_a_negative_gap(self):
        """The event loop clock is monotonic, but a negative gap would silently disable
        the controller rather than fail, so it is clamped rather than assumed."""
        assert _next_resolve_gap(-5.0) == 0.0

    def test_pending_count_covers_every_buffer(self, tmp_path: Path):
        """The ceiling that overrides the gap is only a memory guard if it counts every
        buffer a deferred flush accumulates."""
        consumer = self._reindex(tmp_path)
        buffers = [
            consumer._pending_import_rels,
            consumer._pending_call_rels,
            consumer._pending_type_rels,
            consumer._pending_inherit_rels,
            consumer._pending_ref_rels,
            consumer._pending_member_rels,
            consumer._pending_anchor_rels,
            consumer._pending_config_rels,
        ]
        assert consumer._pending_rel_count() == 0
        for i, buf in enumerate(buffers):
            buf.append(object())  # ty: ignore[invalid-argument-type]
            assert consumer._pending_rel_count() == i + 1, "a buffer is missing from the count"


class _LeaseBus:
    """Just enough bus for the stand-down check."""

    def __init__(self, holder: str | None) -> None:
        self.holder = holder
        self.reads = 0

    async def read_indexer_lease(self) -> str | None:
        self.reads += 1
        return self.holder


class TestStandDownForAForeignLease:
    """`_lease_owner` was initialised to None and never assigned by anything.

    It read as harmless because the daemon released its lease the moment catch-up
    finished, so every holder a consumer saw afterwards genuinely was foreign. A
    persistent indexer (`atlas index --watch`) keeps its lease for the whole session,
    and against that the daemon's own consumers would have stood down for their own
    caller -- idling forever with a full backlog, which is the exact symptom the lease
    exists to prevent, produced by the guard against it.
    """

    @pytest.fixture(autouse=True)
    def _fast_poll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shrink the stand-down poll interval so these tests do not sit through it.

        `_defer_to_foreign_lease` sleeps `_LEASE_POLL_S` (1s) between checks, and it reads
        the module global each time round, so patching the attribute is enough. Without
        this the two resume-path tests cost 1.00s and 1.01s of doing nothing -- measured,
        not assumed. What is under test is *that* the consumer stands down and resumes,
        never how long it naps between looks, so a smaller number cannot make these
        vacuous.
        """
        monkeypatch.setattr("code_atlas.indexing.consumers._LEASE_POLL_S", 0.01)

    @staticmethod
    def _consumer(bus, **kwargs):
        class _Probe(TierConsumer):
            async def process_batch(self, events, batch_id):  # pragma: no cover - never reached
                return None

        return _Probe(
            bus,
            Topic.FILE_CHANGED,
            group="g",
            consumer_name="probe",
            policy=BatchPolicy(max_batch_size=1, time_window_s=0),
            **kwargs,
        )

    async def test_a_consumer_does_not_stand_down_for_its_own_lease(self):
        bus = _LeaseBus(holder="host:1:mine")
        consumer = self._consumer(bus, defer_to_lease=True, lease_owner="host:1:mine")

        await asyncio.wait_for(consumer._defer_to_foreign_lease(), timeout=2)

        assert bus.reads == 1

    async def test_a_consumer_stands_down_for_someone_else(self):
        bus = _LeaseBus(holder="host:2:theirs")
        consumer = self._consumer(bus, defer_to_lease=True, lease_owner="host:1:mine")

        task = asyncio.create_task(consumer._defer_to_foreign_lease())
        await asyncio.sleep(0.05)
        assert not task.done(), "should still be waiting for a foreign lease"

        bus.holder = None  # the other indexer finished
        await asyncio.wait_for(task, timeout=5)

    async def test_no_owner_means_every_lease_is_foreign(self):
        """The previous behaviour, still correct for a consumer that holds no lease."""
        bus = _LeaseBus(holder="host:2:theirs")
        consumer = self._consumer(bus, defer_to_lease=True)

        task = asyncio.create_task(consumer._defer_to_foreign_lease())
        await asyncio.sleep(0.05)
        assert not task.done()

        consumer._stop = True
        await asyncio.wait_for(task, timeout=5)

    async def test_a_consumer_that_does_not_defer_never_asks(self):
        bus = _LeaseBus(holder="host:2:theirs")
        consumer = self._consumer(bus, defer_to_lease=False)

        await asyncio.wait_for(consumer._defer_to_foreign_lease(), timeout=2)

        assert bus.reads == 0
