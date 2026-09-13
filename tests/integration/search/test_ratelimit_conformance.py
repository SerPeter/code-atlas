"""The two limiters must pace the same way, because nothing makes them.

`RateLimiter` runs its arithmetic in Lua inside Valkey; `SqliteRateLimiter` runs it in
Python. Lua cannot call Python, so the refill, the clamp and the AIMD ladder exist twice
and will drift the moment one side is edited alone. That is the drift hazard ATL-158
documented in another form, and the answer is the same: a test that fails when they stop
agreeing.

The two clocks are genuinely different — the Lua reads `redis.call('TIME')` on purpose,
because a container clock measured drifting 16 seconds from this host would otherwise
hand every caller phantom refill. So this cannot compare absolute `wait_ms`. It compares
what a caller actually experiences: **which** calls are admitted, which block, and how
the scale factor moves. Those are the properties the pipeline depends on; the millisecond
is not.

Every case carries a non-vacuity guard. Both limiters return immediately when their store
is unreachable, so "they agree" is the default answer of two broken limiters — and a
Valkey-less run would report conformance it never tested.
"""

from __future__ import annotations

import time

import pytest

from code_atlas.search.ratelimit import (
    _AIMD_DECREASE,
    ConcurrencyGate,
    RateLimiter,
    SqliteRateLimiter,
    make_rate_limiter,
)
from code_atlas.settings import QueueBackendSettings, RedisSettings, SqliteBackendSettings

pytestmark = pytest.mark.integration


def _model() -> str:
    """A fresh key namespace per test — buckets outlive the process that filled them."""
    return f"conformance-{time.time_ns()}"


def _wait_spy(limiter, waits: list[int]):
    """Record each `wait_ms` the limiter computes, without changing what it does.

    Both implementations already produce the number: the Lua script returns
    `[wait_ms, level, cap, refill]`, and `SqliteRateLimiter._try_acquire` returns
    `(wait_ms, scale)`. Wrapping each one is asymmetric code reading a symmetric fact --
    the wrapper passes the real result straight through, so both limiters still block for
    exactly as long as they would have.
    """
    if isinstance(limiter, RateLimiter):
        inner = limiter._acquire_script

        async def spy(**kwargs):
            result = await inner(**kwargs)
            waits.append(int(result[0]))
            return result
    else:
        inner = limiter._try_acquire

        async def spy(tokens: int):
            wait_ms, scale = await inner(tokens)
            waits.append(int(wait_ms))
            return wait_ms, scale

    return spy


async def _admissions(limiter, *, calls: int, tokens: int = 0) -> list[bool]:
    """Whether each of *calls* acquires was admitted without blocking.

    Read from the limiter's own arithmetic, not inferred from elapsed time (ATL-193).

    This used to time each call and threshold it at 50 ms, on the reasoning that a
    refused bucket waits "orders of magnitude above scheduler noise". True of scheduler
    noise; false of a cold Valkey round trip, and false again of a shared CI runner under
    load. It failed 2 of 12 CI runs and blocked a release, in both directions:

        valkey=[False, True, True, True, False, False]   <- cold round trip on call 1
        sqlite=[True,  True, True, True, False, False]

    `acquire()` loops -- it sleeps and retries until it fits -- so a single call can
    compute several waits. The one that answers "was this admitted" is the **first**.
    """
    waits: list[int] = []
    spy = _wait_spy(limiter, waits)
    attr = "_acquire_script" if isinstance(limiter, RateLimiter) else "_try_acquire"
    original = getattr(limiter, attr)
    setattr(limiter, attr, spy)
    try:
        out: list[bool] = []
        for _ in range(calls):
            waits.clear()
            await limiter.acquire(tokens=tokens)
            # No wait at all means the limiter returned early -- degraded, or inside a
            # failure cooldown. That is not an admission, and the caller's `_degraded`
            # guard is what reports it.
            out.append(bool(waits) and waits[0] <= 0)
        return out
    finally:
        setattr(limiter, attr, original)


class TestAdmissionPattern:
    """The shape a caller sees: n through, then blocking."""

    @pytest.mark.parametrize(("rpm", "tpm", "tokens"), [(4, 0, 0), (0, 400, 100), (4, 400, 100)])
    async def test_both_admit_the_same_prefix(self, settings, tmp_path, rpm, tpm, tokens):
        """The burst allowance is the bucket capacity, and it must be the same number.

        A bucket starts full, so the first `cap` calls pass and the next one waits for
        refill. If the two implementations disagreed about `cap` — the scale multiply, the
        `cap < 1` clamp, the per-ms refill divisor — this is where it shows, and nowhere
        else would: a steady-state test just measures both against the same wall clock.
        """
        model = _model()
        async with (
            RateLimiter(settings.redis, model=model, rpm=rpm, tpm=tpm) as valkey,
            SqliteRateLimiter(tmp_path / "rl.sqlite3", model=model, rpm=rpm, tpm=tpm) as sqlite,
        ):
            v = await _admissions(valkey, calls=6, tokens=tokens)
            s = await _admissions(sqlite, calls=6, tokens=tokens)

            assert not valkey._degraded, (
                "the Valkey limiter degraded — this run compared a working SQLite limiter "
                "against a no-op, which agrees with anything"
            )
            assert not all(v), "no call blocked: the budget was never exhausted, so nothing was compared"
            assert v == s, f"admission patterns diverged: valkey={v} sqlite={s}"


class TestScaleTrajectory:
    """AIMD moves the same way on both, including the parts that deliberately do nothing."""

    async def test_penalize_halves_once_then_holds(self, settings, tmp_path):
        """A burst of 429s from one overloaded provider is one event, not many.

        Both sides floor the repeat inside a cooldown, and getting that wrong is invisible
        in normal operation — throughput just collapses to the AIMD floor under a
        transient. Asserting the second call is a no-op is the only way it surfaces.
        """
        model = _model()
        async with (
            RateLimiter(settings.redis, model=model, rpm=10, tpm=0) as valkey,
            SqliteRateLimiter(tmp_path / "rl.sqlite3", model=model, rpm=10, tpm=0) as sqlite,
        ):
            v_first, s_first = await valkey.penalize(), await sqlite.penalize()
            v_second, s_second = await valkey.penalize(), await sqlite.penalize()

            assert v_first == pytest.approx(_AIMD_DECREASE), "Valkey limiter did not halve — store unreachable?"
            assert s_first == pytest.approx(v_first)
            assert v_second == pytest.approx(v_first), "the cooldown did not hold on the Valkey side"
            assert s_second == pytest.approx(v_second), (
                f"cooldowns diverged: valkey held at {v_second}, sqlite moved to {s_second}"
            )

    async def test_the_gate_follows_the_scale_on_both(self, settings, tmp_path):
        """The factor is only useful because it damps concurrency.

        For most models no rpm/tpm is published at all, so the gate is the *only* thing a
        429 can act on. A limiter that computed the right scale and never applied it would
        pass every assertion above.
        """
        model = _model()
        v_gate, s_gate = ConcurrencyGate(8), ConcurrencyGate(8)
        async with (
            RateLimiter(settings.redis, model=model, rpm=0, tpm=0, gate=v_gate) as valkey,
            SqliteRateLimiter(tmp_path / "rl.sqlite3", model=model, rpm=0, tpm=0, gate=s_gate) as sqlite,
        ):
            await valkey.penalize()
            await sqlite.penalize()

            assert v_gate.limit < 8, "the Valkey limiter left the gate untouched — nothing was compared"
            assert v_gate.limit == s_gate.limit, f"gate ceilings diverged: valkey={v_gate.limit} sqlite={s_gate.limit}"


class TestSharedAcrossProcesses:
    """The whole reason this is not an in-memory counter."""

    async def test_a_second_sqlite_limiter_sees_the_first_one_s_spend(self, tmp_path):
        """Two daemons on one machine draw from one budget.

        An in-process limiter would pass every test above and still let two processes
        issue double the configured rate — the exact failure the Valkey buckets were built
        to prevent, reintroduced by the fallback.
        """
        model = _model()
        path = tmp_path / "rl.sqlite3"
        async with (
            SqliteRateLimiter(path, model=model, rpm=4, tpm=0) as first,
            SqliteRateLimiter(path, model=model, rpm=4, tpm=0) as second,
        ):
            spent = await _admissions(first, calls=4)
            assert all(spent), "the first limiter blocked inside its own allowance"

            after = await _admissions(second, calls=1)
            assert after == [False], (
                "a second limiter on the same file was admitted immediately — the budget is "
                "per-process, not shared, and two daemons would double the configured rate"
            )


class TestSelection:
    """`make_rate_limiter` picks on the queue backend, and must not probe to do it."""

    def test_sqlite_queue_gets_the_sqlite_limiter(self, settings, tmp_path):
        cfg = settings.model_copy(deep=True)
        cfg.backend.queue = QueueBackendSettings(sqlite=SqliteBackendSettings())
        cfg.project_root = tmp_path
        limiter = make_rate_limiter(cfg, model="m", rpm=1, tpm=1)
        assert isinstance(limiter, SqliteRateLimiter)

    @pytest.mark.parametrize("declared", [True, False], ids=["valkey", "undeclared"])
    def test_everything_else_gets_valkey(self, settings, declared):
        """An UNDECLARED queue backend resolves to Valkey deliberately.

        Choosing SQLite here would pace against a private file while the rest of the fleet
        paced against Valkey, and the shared budget would silently stop being shared —
        worse than the connect timeout the selection exists to avoid. Note this is the one
        place where "not configured" does NOT mean "probe": the bus factory already probed
        at startup, and a second probe here would reintroduce that timeout.
        """
        cfg = settings.model_copy(deep=True)
        cfg.backend.queue = QueueBackendSettings(valkey=RedisSettings() if declared else None)
        assert isinstance(make_rate_limiter(cfg, model="m", rpm=1, tpm=1), RateLimiter)
