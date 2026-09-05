"""Cross-process rate limiting for embedding provider calls.

Two mechanisms, both needed, because neither covers the other's case:

**Token buckets (Valkey).** Providers enforce *requests per minute* and *tokens per
minute*, not concurrency. A semaphore of size N is 60·N/latency requests per minute --
a number nobody configured and that changes with provider latency. When the limits are
known, two Valkey-backed buckets pace the calls exactly, shared by every process
pointed at the same Valkey: a daemon, an MCP session, and a second daemon for another
project all draw from one budget instead of three.

**AIMD scaling (also Valkey).** The limits usually are *not* known -- litellm's registry
carries rpm/tpm for 4 of its 134 embedding models -- and even a correct limit is wrong
when the quota is shared with something outside this process. So a 429 halves a shared
scale factor that damps both the buckets and the local concurrency gate, and recovers
additively. Because the factor lives in Valkey, one process hitting a 429 slows every
process down; because it damps concurrency too, it still works when no rate limit is
configured at all.

Both degrade to in-process behaviour if Valkey is unreachable -- embedding must not
fail because the limiter cannot reach its coordination store.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import TYPE_CHECKING, NamedTuple, Self

import aiosqlite
import redis.asyncio as aioredis
from loguru import logger

from code_atlas.settings import ensure_sqlite_data_dir

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.settings import AtlasSettings, RedisSettings

# AIMD constants. The decrease must be sharp (a 429 means we are already over) and the
# recovery slow enough that a fleet of processes does not re-converge on the limit in
# lockstep and trip it again.
_AIMD_DECREASE = 0.5
_AIMD_INCREASE = 0.1
_AIMD_INCREASE_INTERVAL_MS = 10_000
_AIMD_FLOOR = 0.05
# One decrease per burst, not one per rejected chunk.
_AIMD_PENALTY_COOLDOWN_MS = 5_000

# Bucket state is worthless once idle for longer than the window it meters.
_DEGRADED_RETRY_MS = 30_000
"""Cooldown before retrying a coordination store that just failed.

Long enough that a missing Valkey costs a couple of timeouts across a whole index, short
enough that a restarted one is picked up within a batch or two. Pacing is unavailable
for that window either way -- the alternative was paying for the discovery every batch.
"""

_BUCKET_TTL_MS = 300_000


# Both buckets are checked and debited in one script so a call cannot consume its
# request budget and then block on tokens -- that leaks budget on every retry and, with
# several processes, live-locks. Server time (redis.call('TIME')) rather than a
# client-supplied clock, so processes on different machines cannot disagree about
# refill. Valkey replicates script *effects*, so a non-deterministic read is safe here.
_ACQUIRE_LUA = """
local now_t = redis.call('TIME')
local now = (tonumber(now_t[1]) * 1000) + math.floor(tonumber(now_t[2]) / 1000)

local rpm   = tonumber(ARGV[1])
local tpm   = tonumber(ARGV[2])
local rcost = tonumber(ARGV[3])
local tcost = tonumber(ARGV[4])
local ttl   = tonumber(ARGV[5])
local step  = tonumber(ARGV[6])
local ivl   = tonumber(ARGV[7])

-- AIMD: additive increase back toward 1.0, at most once per interval.
local raw = redis.call('HMGET', KEYS[3], 'v', 'next')
local scale = tonumber(raw[1])
local nxt = tonumber(raw[2]) or 0
if scale == nil then
  scale = 1.0
elseif scale < 1.0 and now >= nxt then
  scale = math.min(1.0, scale + step)
  redis.call('HSET', KEYS[3], 'v', scale, 'next', now + ivl)
  redis.call('PEXPIRE', KEYS[3], ttl)
end

-- Returns {wait_ms, level, cap, refill_per_ms}; wait_ms 0 means admissible now.
local function inspect(key, limit, cost)
  if limit <= 0 then return {0, 0, 0, 0} end
  local cap = limit * scale
  if cap < 1 then cap = 1 end
  local refill = cap / 60000.0
  local st = redis.call('HMGET', key, 't', 'ts')
  local level = tonumber(st[1])
  local ts = tonumber(st[2])
  if level == nil then
    level = cap
    ts = now
  end
  level = math.min(cap, level + ((now - ts) * refill))
  -- A cost larger than the whole bucket would never admit; clamp so one oversized
  -- chunk drains the bucket instead of waiting forever.
  if cost > cap then cost = cap end
  if level >= cost then return {0, level, cap, refill} end
  return {math.ceil((cost - level) / refill), level, cap, refill}
end

local r = inspect(KEYS[1], rpm, rcost)
local t = inspect(KEYS[2], tpm, tcost)
local wait = math.max(r[1], t[1])

if wait > 0 then
  -- Persist the refilled levels without debiting, so the next caller's arithmetic
  -- starts from now rather than replaying the same elapsed window.
  if rpm > 0 then
    redis.call('HSET', KEYS[1], 't', r[2], 'ts', now)
    redis.call('PEXPIRE', KEYS[1], ttl)
  end
  if tpm > 0 then
    redis.call('HSET', KEYS[2], 't', t[2], 'ts', now)
    redis.call('PEXPIRE', KEYS[2], ttl)
  end
  return {wait, tostring(scale)}
end

if rpm > 0 then
  local c = rcost
  if c > r[3] then c = r[3] end
  redis.call('HSET', KEYS[1], 't', r[2] - c, 'ts', now)
  redis.call('PEXPIRE', KEYS[1], ttl)
end
if tpm > 0 then
  local c = tcost
  if c > t[3] then c = t[3] end
  redis.call('HSET', KEYS[2], 't', t[2] - c, 'ts', now)
  redis.call('PEXPIRE', KEYS[2], ttl)
end
return {0, tostring(scale)}
"""

# Multiplicative decrease. Floored so a sustained 429 storm cannot drive throughput to
# zero and strand the queue; the floor still admits ~1 call per bucket window.
_PENALIZE_LUA = """
local now_t = redis.call('TIME')
local now = (tonumber(now_t[1]) * 1000) + math.floor(tonumber(now_t[2]) / 1000)
local factor = tonumber(ARGV[1])
local floor_v = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local ivl = tonumber(ARGV[4])

local cooldown = tonumber(ARGV[5])
local cur_h = redis.call('HMGET', KEYS[1], 'v', 'pen')
local cur = tonumber(cur_h[1]) or 1.0
local last_pen = tonumber(cur_h[2]) or 0

-- One in-flight batch produces one 429 per chunk. Without this cooldown a burst of
-- max_concurrency rejections would halve the scale that many times over and slam it
-- into the floor, when the provider was reporting a single overload.
if now < (last_pen + cooldown) then
  return tostring(cur)
end

local nv = math.max(floor_v, cur * factor)
redis.call('HSET', KEYS[1], 'v', nv, 'next', now + ivl, 'pen', now)
redis.call('PEXPIRE', KEYS[1], ttl)
return tostring(nv)
"""


class ConcurrencyGate:
    """A semaphore whose ceiling can move while callers are waiting on it.

    ``asyncio.Semaphore`` is fixed at construction, so it cannot express "back off to
    half the workers because the provider is rejecting us" -- which is the only lever
    AIMD has when no rate limit is configured, the common case. Waiters are woken on
    every release and on every ceiling change, so a raised ceiling admits immediately
    rather than at the next release.
    """

    def __init__(self, limit: int) -> None:
        self._max = max(1, limit)
        self._limit = self._max
        self._active = 0
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    async def set_scale(self, scale: float) -> None:
        """Set the ceiling to *scale* of the configured maximum (never below 1).

        The unchanged case is checked before taking the lock, not after. This runs on
        every acquire -- the limiter re-reads the shared scale factor each time -- and
        that lock is the same one admission waits on, so taking it just to discover the
        ceiling is identical would make the steady state contend with itself.
        """
        new_limit = max(1, math.floor(self._max * scale))
        if new_limit == self._limit:
            return
        async with self._cond:
            if new_limit == self._limit:
                return
            self._limit = new_limit
            self._cond.notify_all()

    async def __aenter__(self) -> ConcurrencyGate:
        async with self._cond:
            await self._cond.wait_for(lambda: self._active < self._limit)
            self._active += 1
        return self

    async def __aexit__(self, *_exc: object) -> None:
        async with self._cond:
            self._active -= 1
            self._cond.notify_all()


class RateLimiter:
    """Valkey-backed request/token buckets plus a shared AIMD scale factor.

    *rpm* / *tpm* of ``0`` mean unlimited: that bucket is skipped entirely and costs no
    round trip. With both unlimited the limiter still runs, because the AIMD factor it
    maintains is what damps the concurrency gate -- which is the only protection
    available for the majority of models, whose limits are not published anywhere.
    """

    def __init__(
        self,
        redis_settings: RedisSettings,
        *,
        model: str,
        rpm: int,
        tpm: int,
        gate: ConcurrencyGate | None = None,
    ) -> None:
        url = f"redis://{redis_settings.host}:{redis_settings.port}/{redis_settings.db}"
        if redis_settings.password:
            url = f"redis://:{redis_settings.password}@{redis_settings.host}:{redis_settings.port}/{redis_settings.db}"
        self._redis = aioredis.from_url(url, decode_responses=True)
        self._acquire_script = self._redis.register_script(_ACQUIRE_LUA)
        self._penalize_script = self._redis.register_script(_PENALIZE_LUA)

        prefix = f"{redis_settings.stream_prefix}:rl:{model}"
        self._keys = [f"{prefix}:req", f"{prefix}:tok", f"{prefix}:scale"]
        self._rpm = max(0, rpm)
        self._tpm = max(0, tpm)
        self._gate = gate
        self._scale = 1.0
        # One failure log per limiter, not one per call: if Valkey is down the embed
        # path still works and the noise would bury the errors that matter.
        self._degraded = False
        # Wall-clock (ms) before the next attempt after a failure. Suppressing only the
        # *log* was not enough: every embed batch still opened a socket to a Valkey that
        # was not there and paid the connect timeout. Measured on a 10-file corpus with
        # the provider stubbed, that was 4.08s of a 14s index -- 29% of the run spent
        # dialling a host the first call already proved absent.
        self._retry_at_ms = 0.0

    @property
    def rpm(self) -> int:
        return self._rpm

    @property
    def tpm(self) -> int:
        return self._tpm

    @property
    def scale(self) -> float:
        """Last observed AIMD factor (1.0 = unthrottled)."""
        return self._scale

    async def acquire(self, *, tokens: int = 0) -> None:
        """Block until one request of *tokens* tokens fits within both budgets.

        Never raises on Valkey failure -- an unreachable coordination store degrades to
        no pacing, which is exactly the behaviour before this existed.
        """
        while True:
            if self._degraded and time.time() * 1000 < self._retry_at_ms:
                # Still inside the cooldown from the last failure. Skipping the call
                # rather than the log is the point: an unreachable store must cost one
                # timeout per window, not one per batch.
                return
            try:
                result = await self._acquire_script(
                    keys=self._keys,
                    args=[
                        self._rpm,
                        self._tpm,
                        1,
                        max(0, tokens),
                        _BUCKET_TTL_MS,
                        _AIMD_INCREASE,
                        _AIMD_INCREASE_INTERVAL_MS,
                    ],
                )
            except Exception:
                if not self._degraded:
                    self._degraded = True
                    logger.opt(exception=True).warning(
                        "Rate limiter cannot reach Valkey; embedding continues without cross-process "
                        "pacing (AIMD backoff still applies within this process)"
                    )
                self._retry_at_ms = time.time() * 1000 + _DEGRADED_RETRY_MS
                return

            self._degraded = False
            wait_ms = int(result[0])
            await self._apply_scale(float(result[1]))
            if wait_ms <= 0:
                return
            await asyncio.sleep(wait_ms / 1000.0)

    async def penalize(self) -> float:
        """Halve the shared scale factor after a provider rate-limit rejection."""
        try:
            raw = await self._penalize_script(
                keys=[self._keys[2]],
                args=[
                    _AIMD_DECREASE,
                    _AIMD_FLOOR,
                    _BUCKET_TTL_MS,
                    _AIMD_INCREASE_INTERVAL_MS,
                    _AIMD_PENALTY_COOLDOWN_MS,
                ],
            )
            scale = float(raw)
        except Exception:
            # Local-only decrease so a Valkey outage does not also disable backoff.
            scale = max(_AIMD_FLOOR, self._scale * _AIMD_DECREASE)
        await self._apply_scale(scale)
        logger.warning("Provider rate limit hit; embedding throughput scaled to {:.0%}", scale)
        return scale

    async def _apply_scale(self, scale: float) -> None:
        if scale == self._scale:
            return
        self._scale = scale
        if self._gate is not None:
            await self._gate.set_scale(scale)

    async def close(self) -> None:
        await self._redis.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Close on the way out, the same contract GraphClient and EventBus carry.

        The limiter was the one client left without it, so its ten tests each closed by
        hand on their last line -- and skipped the close whenever an assertion above it
        failed.
        """
        await self.close()


# ---------------------------------------------------------------------------
# Shared bucket arithmetic
# ---------------------------------------------------------------------------


class BucketDecision(NamedTuple):
    """What one bucket says about admitting a call of a given cost."""

    wait_ms: int
    """0 means admissible now."""
    level: float
    """The refilled level, before any debit."""
    cap: float
    debit: float
    """What to subtract when the call is admitted — the cost, clamped to the cap."""


def bucket_decision(*, limit: int, scale: float, cost: float, level: float | None, ts: int, now: int) -> BucketDecision:
    """Refill a leaky bucket to *now* and say whether *cost* fits.

    Extracted so the SQLite limiter and the Valkey Lua script are demonstrably the same
    arithmetic rather than two independent readings of the same intent. They remain two
    implementations — Lua cannot call this — so
    `tests/integration/search/test_ratelimit_conformance.py` runs both against one
    sequence and asserts they agree. That is the same shape the backend conformance
    suite uses, and for the same reason: a hand-written mirror drifts silently.

    A *limit* of 0 is unlimited and costs nothing to evaluate.
    """
    if limit <= 0:
        return BucketDecision(0, 0.0, 0.0, 0.0)

    cap = max(1.0, limit * scale)
    refill = cap / 60000.0
    current = cap if level is None else min(cap, level + ((now - ts) * refill))
    # A cost larger than the whole bucket would never admit; clamp so one oversized
    # chunk drains the bucket instead of waiting forever.
    want = min(cost, cap)
    if current >= want:
        return BucketDecision(0, current, cap, want)
    return BucketDecision(math.ceil((want - current) / refill), current, cap, want)


def aimd_increase(scale: float, next_at: int, now: int) -> tuple[float, int] | None:
    """Additive recovery toward 1.0, at most once per interval.

    Returns the new ``(scale, next_at)`` or None when nothing changed, so a caller can
    skip a write it does not need — the common case, since recovery is deliberately
    slower than the call rate.
    """
    if scale >= 1.0 or now < next_at:
        return None
    return min(1.0, scale + _AIMD_INCREASE), now + _AIMD_INCREASE_INTERVAL_MS


def aimd_penalty(scale: float, next_at: int, now: int) -> tuple[float, int] | None:
    """Multiplicative decrease, floored, at most once per cooldown.

    Returns None inside the cooldown: one decrease per burst, not one per rejected
    chunk, or a batch of 128 rejections would drive the scale to the floor in one go.
    """
    if now < next_at:
        return None
    return max(_AIMD_FLOOR, scale * _AIMD_DECREASE), now + _AIMD_PENALTY_COOLDOWN_MS


# ---------------------------------------------------------------------------
# SQLite-backed limiter
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rate_bucket (
    key   TEXT PRIMARY KEY,
    level REAL NOT NULL,
    ts    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rate_scale (
    key     TEXT PRIMARY KEY,
    v       REAL NOT NULL,
    next_at INTEGER NOT NULL
);
"""


class SqliteRateLimiter:
    """The same buckets and AIMD factor, coordinated through SQLite instead of Valkey.

    The embedded backend exists to run without Valkey, and until this existed it still
    built a Valkey-backed limiter: every embed batch dialled a server that was not there
    and paid a connection timeout before degrading. Measured on a 10-file corpus that was
    **4.08 seconds of a 14 second index**, with the provider itself stubbed to return
    instantly.

    Deleting the limiter on that path would have been the smaller change and the wrong
    one. The pacing is not Valkey's feature, it is the pipeline's, and an embedded
    deployment calling a cloud provider needs it exactly as much. So the coordination
    store follows the backend choice, the way ``SqliteEventBus`` and ``SqliteGraphClient``
    already do.

    **Cross-process, not merely in-process.** SQLite in WAL mode with ``BEGIN IMMEDIATE``
    serialises writers across processes, which is what the Lua script's atomicity buys on
    the Valkey side. Two daemons on one machine share a budget; an in-memory limiter would
    silently give each of them its own, which is the bug the buckets exist to prevent.
    """

    def __init__(
        self,
        path: Path,
        *,
        model: str,
        rpm: int,
        tpm: int,
        gate: ConcurrencyGate | None = None,
    ) -> None:
        self._path = path
        self._req_key = f"rl:{model}:req"
        self._tok_key = f"rl:{model}:tok"
        self._scale_key = f"rl:{model}:scale"
        self._rpm = max(0, rpm)
        self._tpm = max(0, tpm)
        self._gate = gate
        self._scale = 1.0
        self._degraded = False
        self._retry_at_ms = 0.0
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def rpm(self) -> int:
        return self._rpm

    @property
    def tpm(self) -> int:
        return self._tpm

    @property
    def scale(self) -> float:
        """Last observed AIMD factor (1.0 = unthrottled)."""
        return self._scale

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self._path)
            # WAL so a reader never blocks the writer, and a busy timeout so a contended
            # BEGIN IMMEDIATE waits rather than raising -- under contention is exactly
            # when the limiter matters most.
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")
            await conn.executescript(_SCHEMA_SQL)
            await conn.commit()
            self._conn = conn
        return self._conn

    async def _read_bucket(self, conn: aiosqlite.Connection, key: str) -> tuple[float | None, int]:
        cur = await conn.execute("SELECT level, ts FROM rate_bucket WHERE key = ?", (key,))
        row = await cur.fetchone()
        await cur.close()
        if row is None:
            return None, 0
        return float(row[0]), int(row[1])

    async def _write_bucket(self, conn: aiosqlite.Connection, key: str, level: float, ts: int) -> None:
        await conn.execute(
            "INSERT INTO rate_bucket(key, level, ts) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET level = excluded.level, ts = excluded.ts",
            (key, level, ts),
        )

    async def _read_scale(self, conn: aiosqlite.Connection) -> tuple[float, int]:
        cur = await conn.execute("SELECT v, next_at FROM rate_scale WHERE key = ?", (self._scale_key,))
        row = await cur.fetchone()
        await cur.close()
        return (1.0, 0) if row is None else (float(row[0]), int(row[1]))

    async def _write_scale(self, conn: aiosqlite.Connection, value: float, next_at: int) -> None:
        await conn.execute(
            "INSERT INTO rate_scale(key, v, next_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET v = excluded.v, next_at = excluded.next_at",
            (self._scale_key, value, next_at),
        )

    async def acquire(self, *, tokens: int = 0) -> None:
        """Block until one request of *tokens* tokens fits within both budgets.

        Never raises: an unusable coordination store degrades to no pacing, the same
        contract the Valkey limiter carries.
        """
        while True:
            if self._degraded and time.time() * 1000 < self._retry_at_ms:
                return
            try:
                wait_ms, scale = await self._try_acquire(max(0, tokens))
            except Exception:
                if not self._degraded:
                    self._degraded = True
                    logger.opt(exception=True).warning(
                        "Rate limiter cannot use its SQLite store; embedding continues without "
                        "cross-process pacing (AIMD backoff still applies within this process)"
                    )
                self._retry_at_ms = time.time() * 1000 + _DEGRADED_RETRY_MS
                return

            self._degraded = False
            await self._apply_scale(scale)
            if wait_ms <= 0:
                return
            await asyncio.sleep(wait_ms / 1000.0)

    async def _try_acquire(self, tokens: int) -> tuple[int, float]:
        """One atomic check-and-debit of both buckets.

        Both in a single transaction for the reason the Lua script gives: a call that
        consumed its request budget and then blocked on tokens leaks budget on every
        retry, and with several processes live-locks.
        """
        conn = await self._get_conn()
        now = int(time.time() * 1000)

        # The asyncio lock serialises coroutines inside this process; BEGIN IMMEDIATE
        # serialises processes. Both are needed -- aiosqlite funnels one connection
        # through a worker thread, so without the lock two coroutines interleave inside
        # one transaction.
        async with self._lock:
            await conn.execute("BEGIN IMMEDIATE")
            try:
                scale, next_at = await self._read_scale(conn)
                bumped = aimd_increase(scale, next_at, now)
                if bumped is not None:
                    scale, next_at = bumped
                    await self._write_scale(conn, scale, next_at)

                req_level, req_ts = await self._read_bucket(conn, self._req_key)
                tok_level, tok_ts = await self._read_bucket(conn, self._tok_key)
                req = bucket_decision(limit=self._rpm, scale=scale, cost=1, level=req_level, ts=req_ts, now=now)
                tok = bucket_decision(limit=self._tpm, scale=scale, cost=tokens, level=tok_level, ts=tok_ts, now=now)
                wait = max(req.wait_ms, tok.wait_ms)

                # Persist the refilled levels even when waiting, so the next caller's
                # arithmetic starts from now rather than replaying the same window.
                debit_req = 0.0 if wait > 0 else req.debit
                debit_tok = 0.0 if wait > 0 else tok.debit
                if self._rpm > 0:
                    await self._write_bucket(conn, self._req_key, req.level - debit_req, now)
                if self._tpm > 0:
                    await self._write_bucket(conn, self._tok_key, tok.level - debit_tok, now)
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return wait, scale

    async def penalize(self) -> float:
        """Halve the shared scale factor after a provider rate-limit rejection."""
        now = int(time.time() * 1000)
        try:
            conn = await self._get_conn()
            async with self._lock:
                await conn.execute("BEGIN IMMEDIATE")
                try:
                    scale, next_at = await self._read_scale(conn)
                    hit = aimd_penalty(scale, next_at, now)
                    if hit is not None:
                        scale, next_at = hit
                        await self._write_scale(conn, scale, next_at)
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
        except Exception:
            # Local-only decrease, so a broken store does not also disable backoff.
            scale = max(_AIMD_FLOOR, self._scale * _AIMD_DECREASE)

        await self._apply_scale(scale)
        logger.warning("Provider rate limit hit; embedding throughput scaled to {:.0%}", scale)
        return scale

    async def _apply_scale(self, scale: float) -> None:
        if scale == self._scale:
            return
        self._scale = scale
        if self._gate is not None:
            await self._gate.set_scale(scale)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def make_rate_limiter(
    settings: AtlasSettings,
    *,
    model: str,
    rpm: int,
    tpm: int,
    gate: ConcurrencyGate | None = None,
) -> RateLimiter | SqliteRateLimiter:
    """The limiter for whichever queue backend this project is configured for.

    Keyed on ``backend.queue`` rather than on a probe of its own. The bus factory
    already probes once at startup, and a second probe here would reintroduce exactly
    the connect timeout this selection exists to remove.

    ``"auto"`` therefore resolves to Valkey, not to a guess: the Valkey limiter degrades
    on its own when the store is absent, and since it now holds a retry cooldown that
    costs a couple of timeouts per index rather than one per batch. Choosing SQLite for
    ``"auto"`` would be worse -- it would pace against a private file while the rest of
    the fleet paced against Valkey, and the shared budget the buckets exist to enforce
    would quietly stop being shared.
    """
    if settings.backend.queue == "sqlite":
        path = ensure_sqlite_data_dir(settings) / "ratelimit.sqlite3"
        return SqliteRateLimiter(path, model=model, rpm=rpm, tpm=tpm, gate=gate)
    return RateLimiter(settings.redis, model=model, rpm=rpm, tpm=tpm, gate=gate)
