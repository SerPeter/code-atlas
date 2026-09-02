"""Telemetry plumbing for the embedded backends.

The Memgraph client attributes every round-trip from one chokepoint: `_timed_query`
wraps the four `execute`/`execute_write` paths in `graph/client.py` and labels the
sample with the `GraphClient` method that issued it. The SQLite backend has no such
chokepoint — it calls `conn.execute(...)` directly at 213 sites — so the equivalent
seam is the *connection object*, wrapped once in `_get_conn` and passed through
unchanged everywhere else.

That choice is what keeps this file small and keeps `sqlite_graph.py` untouched below
the connection: a statement added tomorrow is instrumented without anyone remembering
to instrument it, which is the property the 213 call sites make essential.

Everything here records onto the **same** instrument the Memgraph side uses,
`atlas_graph_query_seconds{op, kind}`, because the point of instrumenting this backend
at all is that the two become comparable. A new metric name would have defeated it.
"""

from __future__ import annotations

import time
from collections.abc import Sized
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from code_atlas.telemetry import caller_name, get_metrics
from code_atlas.telemetry import is_enabled as telemetry_enabled

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Generator, Iterable
    from typing import Protocol

    import aiosqlite

    class SqlConnection(Protocol):
        """What `SqliteGraphClient` actually needs from a connection.

        The client took `aiosqlite.Connection` everywhere, which named a concrete class
        for a dependency it only ever uses four methods of — so an instrumented or
        faked connection could not be typed as one without lying. This is the interface
        those 213 call sites were really written against; `aiosqlite.Connection`,
        `TimedConnection` and a test double all satisfy it structurally.

        TYPE_CHECKING-only, like `GraphBackend` in `graph/protocol.py`: it exists for
        static structural typing and never for a runtime `isinstance` check.
        """

        # Declared as returning an awaitable rather than as `async def`, because
        # `aiosqlite.Connection.execute` is not a coroutine function: it returns its own
        # `Result` wrapper, which is both awaitable and an async context manager. A
        # coroutine signature here would exclude the very class this abstracts.
        def execute(self, sql: str, parameters: Iterable[Any] | None = ...) -> Awaitable[aiosqlite.Cursor]: ...

        def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> Awaitable[aiosqlite.Cursor]: ...

        def executescript(self, sql_script: str) -> Awaitable[aiosqlite.Cursor]: ...

        def commit(self) -> Awaitable[None]: ...


# This proxy's own pass-through methods plus the timing helper, so `caller_name` walks
# past them to the `SqliteGraphClient` method that actually issued the statement. Same
# role as `_QUERY_PLUMBING` in `graph/client.py`; the names differ because the plumbing
# differs.
_SQL_PLUMBING = frozenset(
    {
        "execute",
        "executemany",
        "executescript",
        "_timed_sql",
        "_get_conn",
        "_record",
    }
)

# Statement verbs that read. Anything else is treated as a write, which is the safe
# direction to be wrong in: a mislabelled read inflates the write count and is visible,
# where a mislabelled write would quietly under-report the thing being watched.
_READ_VERBS = frozenset({"SELECT", "WITH", "EXPLAIN", "PRAGMA", "VALUES"})

# True while a multi-statement logical operation holds the client's write lock. This is
# what makes `kind` comparable with Memgraph's `read_tx`/`write_tx`, which mark
# statements running inside a managed transaction. SQLite's analogue is not a driver
# transaction but the `_write_lock` block that batches a read-classify-write sequence
# into one `commit()`, and that is the boundary a benchmark actually wants to see.
IN_TX: ContextVar[bool] = ContextVar("atlas_sqlite_in_tx", default=False)


@asynccontextmanager
async def in_transaction() -> AsyncIterator[None]:
    """Mark statements issued inside this block as transactional.

    Async so it composes on one line with the client's write lock
    (``async with self._write_lock, _txn():``) rather than forcing another level of
    indentation through eight multi-statement blocks.
    """
    token = IN_TX.set(True)
    try:
        yield
    finally:
        IN_TX.reset(token)


def statement_kind(sql: str) -> str:
    """The `kind` label for *sql* — one of read/write/read_tx/write_tx."""
    stripped = sql.lstrip()
    verb = stripped.split(None, 1)[0].upper() if stripped else ""
    base = "read" if verb in _READ_VERBS else "write"
    return f"{base}_tx" if IN_TX.get() else base


@contextmanager
def _timed_sql(kind: str) -> Generator[None]:
    """Record one SQLite statement against the method that issued it.

    Mirror of `graph/client.py:_timed_query`, including the guard: when telemetry is
    off there is no stack walk and no clock read, so an uninstrumented deployment pays
    one boolean check per statement rather than a frame scan per statement — which
    matters more here than on the Memgraph side, because these statements are local and
    fast rather than a network round-trip.
    """
    if not telemetry_enabled():
        yield
        return
    op = caller_name(_SQL_PLUMBING)
    started = time.perf_counter()
    try:
        yield
    finally:
        get_metrics().graph_query_seconds.record(time.perf_counter() - started, {"op": op, "kind": kind})


@dataclass
class StatementLog:
    """What one captured region asked of SQLite, from two different vantage points.

    `app` is what *we* issued, recorded in the proxy where the calling method is still
    on the stack. `traced` is what SQLite actually ran, recorded from its own trace
    callback on the worker thread where our stack is not available.

    Both are needed and neither substitutes for the other: the gap between them is the
    work SQLite generates on its own behalf — overwhelmingly FTS5 shadow-table
    maintenance — and a report that folds the two together cannot tell "our query
    regressed" from "the full-text index is merging segments".
    """

    app: list[tuple[str, str, int]] = field(default_factory=list)
    """(op, sql, parameter count) per statement we issued.

    The count is kept because `EXPLAIN QUERY PLAN` refuses a statement whose bindings
    are missing -- sqlite3 validates the count before SQLite ever sees the text, so a
    plan cannot be taken from the SQL alone.
    """
    traced: list[str] = field(default_factory=list)
    """Every statement SQLite executed, ours and its own."""

    @property
    def internal(self) -> list[str]:
        """Statements SQLite generated itself.

        Identified by the leading `-- ` its trace callback prefixes them with. That is a
        documented SQLite behaviour for statements it expands on your behalf, and it is
        the only signal available — the callback carries no other provenance.
        """
        return [s for s in self.traced if s.lstrip().startswith("--")]

    def summary(self) -> dict[str, int]:
        internal = len(self.internal)
        return {
            "app_statements": len(self.app),
            "traced_statements": len(self.traced),
            "internal_statements": internal,
        }


@dataclass(frozen=True)
class ScanReport:
    """Which statements SQLite planned as a full scan of `nodes`, and what that costs.

    Every node index in this schema is partial (`... WHERE labels = '<Label>'`) and
    there is no unqualified index on `nodes`, so a predicate that does not name a single
    label cannot use any of them. This report exists to put a number on that rather than
    an argument: `amplification` is row visits, which grows linearly with the corpus and
    so makes the shape self-evident across a size sweep.
    """

    node_rows: int
    scanning: tuple[tuple[str, str, int], ...]
    """(op, sql, executions) for each distinct statement that plans as a scan."""
    non_scanning: int
    """Distinct statements that plan as an index seek — reported so a detector that
    flags everything is visibly wrong rather than quietly alarming."""

    @property
    def scanning_executions(self) -> int:
        return sum(n for _, _, n in self.scanning)

    @property
    def amplification(self) -> int:
        """Row visits: scanning executions times the rows each must walk."""
        return self.scanning_executions * self.node_rows

    def summary(self) -> dict[str, int]:
        return {
            "node_rows": self.node_rows,
            "scanning_statements": len(self.scanning),
            "non_scanning_statements": self.non_scanning,
            "scanning_executions": self.scanning_executions,
            "scan_amplification": self.amplification,
        }


class TimedConnection:
    """`aiosqlite.Connection` proxy that times and attributes every statement.

    Wraps rather than subclasses because the object being wrapped may be a test fake:
    `SqliteGraphClient` accepts an injected connection and uses it as-is, and a subclass
    would have refused those.

    Only the four statement-issuing methods are overridden; everything else
    (`commit`, `close`, `total_changes`, extension loading) falls through untouched, so
    this stays correct as aiosqlite's surface changes.
    """

    def __init__(self, conn: aiosqlite.Connection, log: StatementLog | None = None) -> None:
        self._conn = conn
        self._log = log

    @property
    def raw(self) -> aiosqlite.Connection:
        """The wrapped connection, for the few callers that need the real object."""
        return self._conn

    def _record(self, sql: str, parameters: Iterable[Any] | None = None) -> None:
        if self._log is not None:
            # `caller_name` is cheap but not free, and this path only runs under an
            # explicit capture, so the attribution is worth paying for here even when
            # telemetry itself is off.
            n = len(parameters) if isinstance(parameters, Sized) else 0
            self._log.app.append((caller_name(_SQL_PLUMBING), sql, n))

    async def execute(self, sql: str, parameters: Iterable[Any] | None = None) -> aiosqlite.Cursor:
        self._record(sql, parameters)
        with _timed_sql(statement_kind(sql)):
            return await self._conn.execute(sql, parameters)

    async def executemany(self, sql: str, parameters: Iterable[Iterable[Any]]) -> aiosqlite.Cursor:
        self._record(sql)
        with _timed_sql(statement_kind(sql)):
            return await self._conn.executemany(sql, parameters)

    async def executescript(self, sql_script: str) -> aiosqlite.Cursor:
        self._record(sql_script)
        with _timed_sql("write"):
            return await self._conn.executescript(sql_script)

    async def commit(self) -> None:
        """Timed like a statement, because it is the one that reaches the disk.

        Declared rather than left to ``__getattr__`` for two reasons: a Protocol check
        cannot see a dynamically forwarded member, and the commit is where a write
        actually reaches the disk.

        It lands under ``op="commit"`` rather than under the method that called it,
        because ``commit`` is not in the plumbing skip set. That is deliberate: it keeps
        fsync cost as its own line instead of blending it into the statements that
        preceded it, which is the difference between "the write was slow" and "the
        flush was".
        """
        with _timed_sql("write_tx" if IN_TX.get() else "write"):
            await self._conn.commit()

    async def close(self) -> None:
        """Explicit so lifecycle stays visible on the proxy (ADR-0038)."""
        await self._conn.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


@asynccontextmanager
async def capture_statements(conn: Any) -> AsyncIterator[StatementLog]:
    """Record every statement issued inside this block, ours and SQLite's own.

    A diagnostic mode, not something ordinary operation turns on: the trace callback
    fires per statement on the worker thread and would itself distort the measurement it
    is meant to inform.

    Accepts either a `TimedConnection` or a bare connection; only the former can
    attribute statements to a calling method, and the trace half works for both.
    """
    log = StatementLog()
    timed = conn if isinstance(conn, TimedConnection) else None
    if timed is not None:
        timed._log = log  # noqa: SLF001 — this is the owning module
    raw = timed.raw if timed is not None else conn
    await raw.set_trace_callback(log.traced.append)
    try:
        yield log
    finally:
        await raw.set_trace_callback(None)
        if timed is not None:
            timed._log = None  # noqa: SLF001


async def analyse_scans(conn: Any, log: StatementLog) -> ScanReport:
    """Re-plan each captured statement and report which ones scan `nodes`.

    Offline by necessity: Python's `sqlite3` does not expose
    `sqlite3_stmt_status(SQLITE_STMTSTATUS_FULLSCAN_STEP)`, so there is no in-process
    counter for "this execution walked the table". `EXPLAIN QUERY PLAN` answers the same
    question from the plan instead, which is the stable half anyway — the plan is what a
    schema or predicate change moves.

    Bound with NULLs rather than left unbound: `sqlite3` validates the binding count
    before SQLite plans anything, so a statement cannot be planned from its text alone.
    The values do not steer the plan for these queries -- none is a LIKE-prefix or a
    range whose bounds pick an index.
    """
    raw = conn.raw if isinstance(conn, TimedConnection) else conn

    counts: dict[str, int] = {}
    ops: dict[str, str] = {}
    nparams: dict[str, int] = {}
    for op, sql, n in log.app:
        counts[sql] = counts.get(sql, 0) + 1
        ops.setdefault(sql, op)
        nparams[sql] = n

    scanning: list[tuple[str, str, int]] = []
    non_scanning = 0
    for sql, executions in counts.items():
        # Multi-statement scripts and DDL are not planned; skip rather than guess.
        if ";" in sql.strip().rstrip(";") or not sql.lstrip()[:6].upper().startswith(
            ("SELECT", "WITH", "UPDATE", "DELETE", "INSERT")
        ):
            continue
        try:
            cur = await raw.execute(f"EXPLAIN QUERY PLAN {sql}", [None] * nparams.get(sql, 0))
            rows = await cur.fetchall()
            await cur.close()
        except Exception:
            continue
        detail = " ".join(str(r[-1]) for r in rows)
        if "SCAN nodes" in detail:
            scanning.append((ops[sql], sql, executions))
        else:
            non_scanning += 1

    cur = await raw.execute("SELECT count(*) FROM nodes")
    row = await cur.fetchone()
    await cur.close()

    return ScanReport(
        node_rows=int(row[0]) if row else 0,
        scanning=tuple(scanning),
        non_scanning=non_scanning,
    )
