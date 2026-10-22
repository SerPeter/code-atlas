"""Audit the query plans of every Cypher statement the client sends, against a real graph.

On demand, not in CI: a plan only means something on a graph of realistic size, because
Memgraph plans by cost and on a test graph of a few nodes a full scan is the right plan. Both
plan regressions of 2026-10 (a reclaim that timed out at 120 s, a reclaim that never finished)
returned correct results on every test graph and were invisible to the bench, whose verdicts
come from round-trip counters a scan and an index lookup share.

Companion to ``scripts/profile_query.py``: that one runs a fixed set of MCP read tools and
times them; this one plans every statement the integration suite reaches -- indexing, the
resolvers, the GC, migrations -- and says which will not survive a larger graph.

Two steps:

    CYPHER_RECORD_OUT=statements.jsonl uv run --no-sync python -m pytest \
        -p tests.support.cypher_recorder -m integration tests/integration
    uv run --no-sync python scripts/audit_query_plans.py statements.jsonl [--time] [--uri bolt://host:7687]

**Read-only.** Every statement is only EXPLAINed. ``--time`` additionally *runs* the flagged
statements that contain no write clause, with their recorded params, and terminates any that
outlive ``--budget`` on the server (a client-side timeout leaves the query running). Writes are
never executed. The connection comes from the repository's settings unless ``--uri`` is given.

Severity, from what actually broke:

- **high** -- cost multiplies per row: a full ``ScanAll`` fed by an ``Unwind`` (one scan of the
  whole graph per list item), or a chain of nested ``Unwind`` steps (a list multiplied by
  itself). Exit status 1 when any are found.
- **medium** -- one full ``ScanAll`` per call: cost grows with the whole database, not with the
  project. Matters on hot paths (every search, every indexing batch).
- **info** -- a ``Cartesian`` product; usually two indexed single-node lookups joined.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

_DDL = re.compile(
    # EXPLAIN/PROFILE: validate_cypher already plans user Cypher; a second EXPLAIN is a syntax error.
    r"^\s*(CREATE\s+(INDEX|CONSTRAINT|VECTOR|TEXT|EDGE)|DROP\s|ANALYZE|SHOW\s|STORAGE\s|FREE\s|TERMINATE\s"
    r"|EXPLAIN\s|PROFILE\s)",
    re.IGNORECASE,
)
_WRITE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH|DROP|FOREACH|CALL)\b", re.IGNORECASE)
_SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


@dataclass
class Finding:
    caller: str
    count: int
    query: str
    params: dict[str, Any]
    issues: list[tuple[str, str]] = field(default_factory=list)
    seconds: float | None = None

    @property
    def severity(self) -> str:
        return min((s for s, _ in self.issues), key=_SEVERITY_ORDER.__getitem__)


def _steps(plan: list[str]) -> list[tuple[int, str]]:
    """``(branch depth, operator)`` per plan line; depth is the number of ``|`` before the ``*``."""
    out = []
    for line in plan:
        head, _, op = line.partition("*")
        out.append((head.count("|"), op.strip()))
    return out


def classify(plan: list[str]) -> list[tuple[str, str]]:
    steps = _steps(plan)
    issues: list[tuple[str, str]] = []
    for i, (depth, op) in enumerate(steps):
        if not re.match(r"ScanAll\s*\(", op):
            continue
        # Memgraph prints a plan top-down: what feeds an operator sits below it in the same
        # branch, which ends where the depth drops back.
        fed_by_unwind = False
        for d, o in steps[i + 1 :]:
            if d < depth:
                break
            if d == depth and o.startswith("Unwind"):
                fed_by_unwind = True
                break
        issues.append(("high", f"full scan per list row: {op}") if fed_by_unwind else ("medium", f"full scan: {op}"))
    run = worst = 0
    for _, op in steps:
        run = run + 1 if op.startswith("Unwind") else 0
        worst = max(worst, run)
    if worst >= 2:
        issues.append(("high", f"{worst} nested Unwinds"))
    if any(op.startswith("Cartesian") for _, op in steps):
        issues.append(("info", "Cartesian product"))
    return issues


def _connect(uri: str | None) -> Any:
    if uri:
        return GraphDatabase.driver(uri, auth=("", ""))
    from code_atlas.settings import AtlasSettings

    mg = AtlasSettings().memgraph
    return GraphDatabase.driver(f"bolt://{mg.host}:{mg.port}", auth=(mg.username, mg.password))


def _timed(driver: Any, finding: Finding, budget_s: float) -> float:
    """Run a read-only statement; terminate it server-side past *budget_s*. ``inf`` when cut."""
    marker = f"/* audit {time.time()} */ "
    done = threading.Event()
    cut = threading.Event()

    def guard() -> None:
        if done.wait(budget_s):
            return
        with driver.session() as s:
            for tx in s.run("SHOW TRANSACTIONS").data():
                if any(marker in q for q in tx["query"]):
                    s.run(f'TERMINATE TRANSACTIONS "{tx["transaction_id"]}"').consume()
                    cut.set()

    threading.Thread(target=guard, daemon=True).start()
    start = time.perf_counter()
    try:
        with driver.session() as s:
            s.run(marker + finding.query, finding.params).consume()
    except Exception:
        if not cut.is_set():
            raise
    finally:
        done.set()
    return float("inf") if cut.is_set() else time.perf_counter() - start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("statements", type=Path, help="JSONL written by tests/support/cypher_recorder.py")
    ap.add_argument("--uri", help="bolt URI; default: this repository's Memgraph settings")
    ap.add_argument("--time", action="store_true", help="also run flagged read-only statements and time them")
    ap.add_argument("--budget", type=float, default=30.0, help="seconds before a timed statement is terminated")
    ap.add_argument("--min-severity", choices=list(_SEVERITY_ORDER), default="medium")
    args = ap.parse_args()

    records = [json.loads(line) for line in args.statements.read_text(encoding="utf-8").splitlines() if line.strip()]
    driver = _connect(args.uri)
    findings: list[Finding] = []
    explained = 0
    errors: list[str] = []
    with driver.session() as s:
        for rec in records:
            if rec["caller"] == "?" or _DDL.match(rec["query"]):
                continue
            try:
                plan = [r["QUERY PLAN"] for r in s.run("EXPLAIN " + rec["query"], rec["params"])]
            except Exception as exc:  # a params sample the planner rejects; report, keep going
                errors.append(f"{rec['caller']}: {str(exc)[:120]}")
                continue
            explained += 1
            issues = classify(plan)
            if issues:
                findings.append(Finding(rec["caller"], rec["count"], rec["query"], rec["params"], issues))

    limit = _SEVERITY_ORDER[args.min_severity]
    shown = [f for f in findings if _SEVERITY_ORDER[f.severity] <= limit]
    if args.time:
        for f in shown:
            if not _WRITE.search(f.query):
                f.seconds = _timed(driver, f, args.budget)
    driver.close()

    shown.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], -(f.seconds or 0), f.caller))
    print(f"{len(records)} statements recorded, {explained} explained, {len(findings)} flagged\n")
    for f in shown:
        timing = (
            ""
            if f.seconds is None
            else (f"  >{args.budget:.0f}s" if f.seconds == float("inf") else f"  {f.seconds:.3f}s")
        )
        print(f"[{f.severity}] {f.caller}  (sent x{f.count}){timing}")
        for sev, text in f.issues:
            print(f"    {sev}: {text}")
        print(f"    {' '.join(f.query.split())[:200]}")
    for e in errors:
        print(f"[explain failed] {e}")
    return 1 if any(f.severity == "high" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
