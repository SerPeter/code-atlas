"""Two indexers must not share an identity — measured against a real Valkey.

Every claim these tests pin was measured live before the fix, on a stream shared by two
connections using the same group AND the same consumer name:

- `XREADGROUP ... 0` ("read my history") returned the OTHER connection's in-flight
  messages, so both processed them.
- Either connection's `XACK` of the other's in-flight message returned 1 and removed it
  from the PEL, deleting the peer's crash-recovery net.
- `XINFO GROUPS` still reported `consumers=1`, so nothing could see the second process.

A single 183-file index was split between two processes running different code versions
because of this, and the resulting Memgraph MVCC conflicts exhaust a 4-attempt retry into
a poison-park that ACK-drops the file.
"""

from __future__ import annotations

import pytest

from code_atlas.events import FileChanged, IndexerBusyError, Topic, hold_indexer_lease

pytestmark = [pytest.mark.integration]


async def test_two_consumers_never_see_or_ack_each_others_pending(event_bus):
    """The PEL cross-talk vector. Distinct consumer names are what separate the two."""
    topic = Topic.FILE_CHANGED
    group = "lease-test"
    await event_bus.ensure_group(topic, group)

    for i in range(4):
        await event_bus.publish(topic, FileChanged(path=f"m{i}.py", change_type="modified", project_name="p"))

    a_msgs = await event_bus.read_batch(topic, group, "worker-a", count=2, block_ms=200)
    b_msgs = await event_bus.read_batch(topic, group, "worker-b", count=2, block_ms=200)
    assert a_msgs
    assert b_msgs

    # Each consumer's history is its own. Under the old shared name this returned the
    # peer's in-flight ids as well.
    a_pending = {mid for mid, _ in await event_bus.read_pending(topic, group, "worker-a", count=20)}
    b_ids = {mid for mid, _ in b_msgs}
    assert a_pending.isdisjoint(b_ids)


async def test_abandoned_pending_is_reclaimable_by_another_consumer(event_bus):
    """The other half of unique names: a killed process's PEL must not be orphaned.

    `read_pending` only ever returns the caller's own deliveries, so without an explicit
    reclaim path a uniquely-named dead consumer's work is stranded forever.
    """
    topic = Topic.FILE_CHANGED
    group = "lease-reclaim"
    await event_bus.ensure_group(topic, group)
    await event_bus.publish(topic, FileChanged(path="orphan.py", change_type="modified", project_name="p"))

    dead = await event_bus.read_batch(topic, group, "worker-dead", count=1, block_ms=200)
    assert len(dead) == 1

    # Nothing is idle yet, so a live peer's work is left alone.
    assert await event_bus.reclaim_abandoned(topic, group, "worker-live", min_idle_ms=60_000, count=10) == []

    # Past the idle threshold it is adoptable.
    adopted = await event_bus.reclaim_abandoned(topic, group, "worker-live", min_idle_ms=0, count=10)
    assert {mid for mid, _ in adopted} == {mid for mid, _ in dead}


async def test_lease_is_exclusive_and_names_its_holder(event_bus):
    async with hold_indexer_lease(event_bus) as owner:
        assert await event_bus.read_indexer_lease() == owner
        with pytest.raises(IndexerBusyError) as excinfo:
            async with hold_indexer_lease(event_bus):
                pass
        assert excinfo.value.holder == owner

    # Released on exit, so the next indexer is not blocked by a finished one.
    assert await event_bus.read_indexer_lease() is None


async def test_lease_release_cannot_free_someone_elses(event_bus):
    """Compare-and-delete: a process that stalled past its TTL and lost the lease must not
    be able to free the lease that has since passed to another indexer."""
    async with hold_indexer_lease(event_bus) as owner:
        assert await event_bus.release_indexer_lease("some-other-process") is False
        assert await event_bus.read_indexer_lease() == owner


async def test_abandoned_work_is_swept_up_after_the_owner_dies(event_bus):
    """The adopt path must keep running, not only during a consumer's initial drain.

    Observed live: an indexer exited holding 96 file-changed and 1037 embed messages, and
    a healthy consumer polled beside them for minutes without ever looking — because
    `pel_drained` had already flipped True, and the reclaim was gated behind it.
    """
    topic = Topic.FILE_CHANGED
    group = "sweep-test"
    await event_bus.ensure_group(topic, group)
    await event_bus.publish(topic, FileChanged(path="stranded.py", change_type="modified", project_name="p"))

    dead = await event_bus.read_batch(topic, group, "worker-dead", count=1, block_ms=200)
    assert len(dead) == 1

    # A live consumer that has already drained its own PEL must still adopt this.
    adopted = await event_bus.reclaim_abandoned(topic, group, "worker-live", min_idle_ms=0, count=10)
    assert {mid for mid, _ in adopted} == {mid for mid, _ in dead}

    # And once adopted it is that consumer's own pending work, so a normal drain clears it.
    own = await event_bus.read_pending(topic, group, "worker-live", count=10)
    assert {mid for mid, _ in own} == {mid for mid, _ in dead}
