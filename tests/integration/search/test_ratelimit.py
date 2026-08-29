"""Integration tests for the Valkey-backed embedding rate limiter.

What matters here and cannot be tested without Valkey: that the budget and the AIMD
scale factor are genuinely *shared*, so a second process sees the first one's spend and
its backoff. A per-process limiter would pass every unit test and still let a daemon and
an MCP session issue double the configured rate.
"""

from __future__ import annotations

import time

import pytest
import redis.asyncio as aioredis

from code_atlas.search.ratelimit import ConcurrencyGate, RateLimiter

pytestmark = pytest.mark.integration


def _keys(settings, model: str) -> tuple[str, str, str]:
    prefix = f"{settings.redis.stream_prefix}:rl:{model}"
    return f"{prefix}:req", f"{prefix}:tok", f"{prefix}:scale"


async def _raw(settings):
    url = f"redis://{settings.redis.host}:{settings.redis.port}/{settings.redis.db}"
    return aioredis.from_url(url, decode_responses=True)


async def _server_now_ms(r) -> int:
    """Bucket timestamps must come from the same clock the Lua reads.

    The script uses ``redis.call('TIME')`` deliberately -- on this project's own dev
    machine the Valkey container's clock was measured drifting 16s ahead of the host
    inside a minute, which as client-supplied time would have handed every caller 16
    seconds of phantom refill. Seeding a bucket from ``time.time()`` reproduces exactly
    that bug in the test rather than in the limiter.
    """
    secs, micros = await r.time()
    return secs * 1000 + micros // 1000


@pytest.fixture
async def clean_keys(settings):
    """Drop limiter state before and after — buckets persist across tests otherwise."""
    model = f"test-rl-{time.time_ns()}"
    r = await _raw(settings)
    await r.delete(*_keys(settings, model))
    yield model
    await r.delete(*_keys(settings, model))
    await r.aclose()


async def test_unlimited_never_blocks(settings, clean_keys):
    """rpm/tpm of 0 mean unlimited: the buckets are skipped entirely."""
    async with RateLimiter(settings.redis, model=clean_keys, rpm=0, tpm=0) as lim:
        start = time.monotonic()
        for _ in range(20):
            await lim.acquire(tokens=10_000)
        assert time.monotonic() - start < 1.0


async def test_drained_bucket_makes_the_caller_wait(settings, clean_keys):
    """A bucket with no budget left must delay the call rather than let it through."""
    req_key, _, _ = _keys(settings, clean_keys)
    async with RateLimiter(settings.redis, model=clean_keys, rpm=120, tpm=0) as lim:
        r = await _raw(settings)
        # Simulate a drained request bucket: 120/min refills 1 token per 500ms.
        await r.hset(req_key, mapping={"t": "0", "ts": str(await _server_now_ms(r))})

        start = time.monotonic()
        await lim.acquire()
        waited = time.monotonic() - start
        assert waited >= 0.3, f"expected to wait for refill, returned in {waited:.3f}s"
        await r.aclose()


async def test_budget_is_shared_between_limiters(settings, clean_keys):
    """Two limiters on one model are two processes sharing one quota.

    This is the property the whole design exists for: without it, N daemons issue N
    times the configured rate and the limit is decorative.
    """
    async with (
        RateLimiter(settings.redis, model=clean_keys, rpm=10, tpm=0) as a,
        RateLimiter(settings.redis, model=clean_keys, rpm=10, tpm=0) as b,
    ):
        # Capacity is 10; spend it all through `a`.
        for _ in range(10):
            await a.acquire()

        r = await _raw(settings)
        req_key = _keys(settings, clean_keys)[0]
        level_after_a = float((await r.hget(req_key, "t")) or 0)
        assert level_after_a < 1.0, "spending through a should have drained the shared bucket"

        # b must observe a's spend, not start from a full bucket of its own.
        start = time.monotonic()
        await b.acquire()
        assert time.monotonic() - start >= 1.0, "second limiter did not see the first's spend"
        await r.aclose()


async def test_penalize_is_visible_to_another_limiter(settings, clean_keys):
    """One process's 429 has to slow every process, or the others keep pushing at the
    rate that just failed."""
    async with (
        RateLimiter(settings.redis, model=clean_keys, rpm=600, tpm=0) as a,
        RateLimiter(settings.redis, model=clean_keys, rpm=600, tpm=0) as b,
    ):
        scale = await a.penalize()
        assert scale == pytest.approx(0.5)

        await b.acquire()  # b reads the shared factor on its next acquire
        assert b.scale == pytest.approx(0.5)


async def test_penalty_cooldown_collapses_a_burst(settings, clean_keys):
    """A saturated batch produces one 429 per chunk. Those are one overload, not N, and
    must not compound into a collapse to the floor."""
    async with RateLimiter(settings.redis, model=clean_keys, rpm=600, tpm=0) as lim:
        first = await lim.penalize()
        second = await lim.penalize()
        third = await lim.penalize()
        assert first == pytest.approx(0.5)
        assert second == pytest.approx(0.5), "burst penalties compounded despite the cooldown"
        assert third == pytest.approx(0.5)


async def test_penalize_damps_the_concurrency_gate(settings, clean_keys):
    """When no rate limit is configured — the case for most models — the gate is the
    only thing AIMD can move."""
    gate = ConcurrencyGate(8)
    async with RateLimiter(settings.redis, model=clean_keys, rpm=0, tpm=0, gate=gate) as lim:
        assert gate.limit == 8
        await lim.penalize()
        assert gate.limit == 4


async def test_unreachable_valkey_degrades_instead_of_failing(settings, clean_keys):
    """Embedding must not fail because the limiter cannot reach its coordination store."""
    from code_atlas.settings import RedisSettings

    dead = RedisSettings(host="127.0.0.1", port=1)  # nothing listens here
    async with RateLimiter(dead, model=clean_keys, rpm=10, tpm=10) as lim:
        start = time.monotonic()
        await lim.acquire(tokens=5)  # must return, not raise, not hang
        assert time.monotonic() - start < 5.0
        assert await lim.penalize() == pytest.approx(0.5), "local backoff should still apply"


async def test_token_budget_is_metered_separately(settings, clean_keys):
    """A tokens-per-minute limit must bind even when the request count is well inside
    its own budget — a single call can carry thousands of tokens."""
    _, tok_key, _ = _keys(settings, clean_keys)
    async with RateLimiter(settings.redis, model=clean_keys, rpm=0, tpm=6000) as lim:
        r = await _raw(settings)
        await r.hset(tok_key, mapping={"t": "0", "ts": str(await _server_now_ms(r))})

        start = time.monotonic()
        await lim.acquire(tokens=100)  # 6000/min = 100 tokens per second
        assert time.monotonic() - start >= 0.5
        await r.aclose()
