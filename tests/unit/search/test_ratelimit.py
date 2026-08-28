"""Unit tests for the embedding rate limiter's pure/local behaviour.

The Valkey-backed bucket arithmetic is exercised in
``tests/integration/search/test_ratelimit.py``; what is testable without infra is the
concurrency gate (which is what AIMD actually moves when no rate limit is configured,
the common case) and the resolution order for rpm/tpm.
"""

from __future__ import annotations

import asyncio

import pytest

from code_atlas.search.embeddings import EmbedClient
from code_atlas.search.ratelimit import ConcurrencyGate
from code_atlas.settings import EmbeddingSettings


class TestConcurrencyGate:
    """A ceiling that AIMD can move while callers are queued on it."""

    async def test_admits_up_to_the_limit(self):
        gate = ConcurrencyGate(3)
        held: list[int] = []

        async def hold(i: int) -> None:
            async with gate:
                held.append(i)
                await asyncio.sleep(0.05)

        task = asyncio.gather(*(hold(i) for i in range(5)))
        await asyncio.sleep(0.01)
        assert len(held) == 3, "gate admitted more than its ceiling"
        await task
        assert len(held) == 5

    async def test_scale_shrinks_the_ceiling(self):
        gate = ConcurrencyGate(8)
        await gate.set_scale(0.5)
        assert gate.limit == 4

    async def test_ceiling_never_reaches_zero(self):
        """A 429 storm must slow the pipeline, never stop it. At the AIMD floor the
        ceiling still has to admit one caller or the embed queue strands forever."""
        gate = ConcurrencyGate(4)
        await gate.set_scale(0.01)
        assert gate.limit == 1
        async with gate:
            pass  # a caller still gets through

    async def test_raising_the_ceiling_wakes_waiters(self):
        """Recovery must not wait for an in-flight call to finish. AIMD's additive
        increase happens on the limiter's own clock, so a waiter blocked at the old
        ceiling has to be woken by the change itself."""
        gate = ConcurrencyGate(2)
        await gate.set_scale(0.5)  # ceiling 1
        entered = asyncio.Event()
        released = asyncio.Event()

        async def first() -> None:
            async with gate:
                entered.set()
                await released.wait()

        async def second() -> None:
            async with gate:
                pass

        t1 = asyncio.create_task(first())
        await entered.wait()
        t2 = asyncio.create_task(second())
        await asyncio.sleep(0.02)
        assert not t2.done(), "second caller should be blocked at ceiling 1"

        await gate.set_scale(1.0)  # ceiling back to 2 — must admit without a release
        await asyncio.wait_for(t2, timeout=1.0)

        released.set()
        await t1


class TestRateLimitResolution:
    """rpm/tpm resolve as: explicit config, then litellm's registry, then provider default."""

    def _client(self, **kw) -> EmbedClient:
        return EmbedClient(EmbeddingSettings(**kw))

    def test_explicit_config_wins(self):
        c = self._client(provider="litellm", model="gemini/gemini-embedding-001", rpm=42, tpm=99)
        assert (c._rpm, c._tpm) == (42, 99)

    def test_explicit_zero_means_unlimited_and_beats_the_registry(self):
        """0 is a value, not 'unset'. A model the registry *does* know must still be
        overridable to unlimited, or the config cannot express 'my quota is higher'."""
        c = self._client(provider="litellm", model="gemini/gemini-embedding-001", rpm=0, tpm=0)
        assert (c._rpm, c._tpm) == (0, 0)

    def test_registry_supplies_known_limits(self):
        """gemini/gemini-embedding-001 is one of the few embedding models litellm
        publishes limits for — the whole reason the registry is in the chain."""
        c = self._client(provider="litellm", model="gemini/gemini-embedding-001")
        assert c._rpm > 0
        assert c._tpm > 0

    def test_unknown_model_falls_back_to_provider_default(self):
        """get_model_info raises for unmapped models; that must resolve to the provider
        default rather than propagate."""
        c = self._client(provider="litellm", model="not-a-real-vendor/not-a-real-model")
        assert (c._rpm, c._tpm) == (0, 0)

    def test_local_provider_is_unlimited_by_default(self):
        """TEI's constraint is GPU concurrency, not a per-minute quota."""
        c = self._client(provider="tei", model="nomic-ai/nomic-embed-code")
        assert (c._rpm, c._tpm) == (0, 0)

    def test_no_limiter_without_redis_settings(self):
        """The limiter is opt-in per call site — health checks construct the client
        without one so a drained bucket cannot make the provider look unreachable."""
        c = self._client(provider="tei", model="nomic-ai/nomic-embed-code")
        assert c._limiter is None

    def test_gate_is_per_client_not_per_call(self):
        """The bug this replaced: embed_batch built a fresh Semaphore(max_concurrency)
        on every call while the consumer ran max_concurrency calls at once, so the real
        ceiling was max_concurrency squared."""
        c = self._client(provider="litellm", model="gemini/gemini-embedding-001")
        assert c._gate.limit == c.max_concurrency


class TestTokenCounting:
    """The tokens-per-minute budget needs a count per chunk, not a guess."""

    def test_truncate_returns_counts_alongside_texts(self):
        c = EmbedClient(EmbeddingSettings(provider="litellm", model="gemini/gemini-embedding-001"))
        texts, counts = c._truncate_texts(["hello world", "a somewhat longer piece of text here"])
        assert len(texts) == len(counts) == 2
        assert all(n > 0 for n in counts)
        assert counts[1] > counts[0]

    def test_counts_fall_back_when_model_limit_unknown(self):
        c = EmbedClient(EmbeddingSettings(provider="litellm", model="not-a-real-vendor/not-a-real-model"))
        assert c._max_input_tokens is None
        texts, counts = c._truncate_texts(["x" * 400])
        assert texts == ["x" * 400]
        assert counts == [100]  # ~4 chars/token


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.25, 0.05])
async def test_gate_scale_is_monotonic(scale: float):
    gate = ConcurrencyGate(16)
    await gate.set_scale(scale)
    assert 1 <= gate.limit <= 16
