"""Unit tests for consumer dedup and cooldown path identity (no infrastructure needed)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from code_atlas.events import FileChanged, Topic, encode_event
from code_atlas.graph.client import UpsertResult
from code_atlas.indexing.consumers import _MAX_BATCH_FAILURES, ASTConsumer, BatchPolicy, TierConsumer
from code_atlas.parsing.ast import ParsedEntity, ParsedFile, ParsedRelationship
from code_atlas.schema import RelType
from code_atlas.settings import AtlasSettings, EmbeddingSettings

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
        self.gc_calls: int = 0

    async def delete_file_entities(self, project_name: str, file_path: str) -> list[str]:
        self.deleted.append((project_name, file_path))
        return []

    async def get_batch_file_hashes(self, project_name: str, file_paths: list[str]) -> dict[str, str]:
        return {}

    async def set_batch_file_hashes(self, project_name: str, hashes: dict[str, str]) -> None:
        self.hash_writes.append((project_name, dict(hashes)))

    async def upsert_batch_entities(
        self, project_name: str, file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]]
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


async def test_flush_resolves_config_refs_then_runs_gc(tmp_path: Path) -> None:
    """Order matters: the sweep deletes anything at zero incoming edges, so it
    must run only after this flush has re-created its references.
    """
    consumer = _make_consumer(tmp_path)
    rel = ParsedRelationship(from_qualified_name="proj:conf.load", rel_type=RelType.READS_ENV, to_name="DATABASE_URL")
    consumer._pending_config_rels.append(rel)
    consumer._pending_project_names.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.config_calls == [("proj", [rel])]  # type: ignore[attr-defined]
    assert consumer.graph.gc_calls == 1  # type: ignore[attr-defined]
    assert consumer._pending_config_rels == []


async def test_flush_runs_gc_even_with_no_config_rels(tmp_path: Path) -> None:
    """A file whose LAST os.getenv() call was just deleted produces no config
    rels at all — and that is exactly the case that orphans a node. Gating the
    sweep on config rels being present would never collect it.
    """
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.config_calls == []  # type: ignore[attr-defined]
    assert consumer.graph.gc_calls == 1  # type: ignore[attr-defined]


async def test_flush_skips_gc_when_nothing_was_processed(tmp_path: Path) -> None:
    consumer = _make_consumer(tmp_path)

    await consumer._flush_deferred_resolution()

    assert consumer.graph.gc_calls == 0  # type: ignore[attr-defined]


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

    assert consumer.graph.citation_calls == [("proj", {}, None, True)]  # type: ignore[attr-defined]
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

    assert consumer.graph.citation_calls == [  # type: ignore[attr-defined]
        ("proj", {"proj:src.mod.f": ["ADR-0014"]}, {"src/mod.py"}, True)
    ]


async def test_flush_without_documents_or_citations_does_not_sweep(tmp_path: Path) -> None:
    """Steady-state daemon flushes must not pay for a project-wide scan."""
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")

    await consumer._flush_deferred_resolution()

    assert consumer.graph.citation_calls == []  # type: ignore[attr-defined]


async def test_a_reparsed_file_with_no_citations_still_reaches_the_resolver(tmp_path: Path) -> None:
    """The removal signal. A file whose last `see ADR-14` comment was deleted
    contributes nothing to _pending_citations, so gating the call on that dict
    is exactly why the stale edge used to survive — the file scope has to carry
    the call on its own."""
    consumer = _make_consumer(tmp_path)
    consumer._pending_project_names.add("proj")
    consumer._pending_citation_files["proj"] = {"src/mod.py"}

    await consumer._flush_deferred_resolution()

    assert consumer.graph.citation_calls == [("proj", {}, {"src/mod.py"}, False)]  # type: ignore[attr-defined]
    assert consumer._pending_citation_files == {}


async def test_process_batch_scopes_every_parsed_file_not_just_citing_ones(tmp_path: Path) -> None:
    """The scope is built from the parse, not from the citations it produced."""
    settings = AtlasSettings(project_root=tmp_path, embeddings=EmbeddingSettings(enabled=False))
    consumer = ASTConsumer(RecordingBus(), StubGraph(), settings)  # type: ignore[arg-type]
    (tmp_path / "cited.py").write_text("# WHY: see ADR-0014\ndef f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    events = [_event("cited.py", "proj", str(tmp_path)), _event("plain.py", "proj", str(tmp_path))]
    await consumer.process_batch(events, "b1")

    # The batch's own flush already drained the buffers, so assert on what
    # actually reached the resolver.
    assert consumer.graph.citation_calls == [  # type: ignore[attr-defined]
        ("proj", {"proj:cited.f": ["ADR-0014"]}, {"cited.py", "plain.py"}, False)
    ]


async def test_final_flush_sweeps_every_project_that_saw_a_citation(tmp_path: Path) -> None:
    """Backstop for the cold index: the document may have been in the graph
    before this run started, so no document-change event ever fires."""
    consumer = _make_consumer(tmp_path)
    consumer._citation_projects.add("proj")

    await consumer._flush_deferred_resolution(final=True)

    assert consumer.graph.citation_calls == [("proj", {}, None, True)]  # type: ignore[attr-defined]
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
        self, project_name: str, file_data: dict[str, tuple[list[ParsedEntity], list[ParsedRelationship]]]
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
        RecordingBus(),  # type: ignore[arg-type]
        graph,  # type: ignore[arg-type]
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


async def test_a_file_with_no_entity_change_still_takes_the_immediate_hash_path(tmp_path: Path) -> None:
    """The other half of the withholding condition: ``immediate_hashes`` stays
    reachable. When every entity comes back unchanged, the file's stored
    citations are byte-identical to the ones this parse produced, so the
    revoke can neither delete nor recreate anything for it.
    """
    graph = _UnchangedGraph()
    consumer = _reindex_consumer(tmp_path, graph)
    (tmp_path / "plain.py").write_text("def g():\n    return 2\n", encoding="utf-8")

    await consumer.process_batch([], "warmup")
    await consumer.process_batch([_event("plain.py", "proj", str(tmp_path))], "b1")

    assert consumer._pending_file_hashes == {}
    assert [fp for _, hashes in graph.hash_writes for fp in hashes] == ["plain.py"]
