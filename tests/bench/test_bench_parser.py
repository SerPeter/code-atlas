"""Parser throughput benchmark.

Pure CPU benchmark — no I/O, no graph. Calls parse_file() on all
synthetic files and measures files/sec and entities/sec.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from code_atlas.parsing.ast import parse_file

pytestmark = [pytest.mark.bench, pytest.mark.slow]


def test_parser_throughput_small(bench_small: tuple[Path, list[str]]):
    """Parse all files in the small codebase and report throughput."""
    root, rel_paths = bench_small
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]

    total_entities = 0
    start = time.perf_counter()

    for rel_path in py_paths:
        abs_path = root / rel_path
        source = abs_path.read_bytes()
        result = parse_file(rel_path, source, project_name="bench")
        if result is not None:
            total_entities += len(result.entities)

    elapsed = time.perf_counter() - start
    files_per_sec = len(py_paths) / elapsed if elapsed > 0 else 0
    entities_per_sec = total_entities / elapsed if elapsed > 0 else 0

    report = {
        "benchmark": "parser_throughput_small",
        "files": len(py_paths),
        "entities": total_entities,
        "elapsed_s": round(elapsed, 3),
        "files_per_sec": round(files_per_sec, 1),
        "entities_per_sec": round(entities_per_sec, 1),
    }
    print(f"\n{json.dumps(report, indent=2)}")

    # Regression guard
    assert files_per_sec > 100, f"Parser too slow: {files_per_sec:.1f} files/sec (expected >100)"


def test_parser_throughput_medium(bench_medium: tuple[Path, list[str]]):
    """Parse all files in the medium codebase and report throughput."""
    root, rel_paths = bench_medium
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]

    total_entities = 0
    start = time.perf_counter()

    for rel_path in py_paths:
        abs_path = root / rel_path
        source = abs_path.read_bytes()
        result = parse_file(rel_path, source, project_name="bench")
        if result is not None:
            total_entities += len(result.entities)

    elapsed = time.perf_counter() - start
    files_per_sec = len(py_paths) / elapsed if elapsed > 0 else 0
    entities_per_sec = total_entities / elapsed if elapsed > 0 else 0

    report = {
        "benchmark": "parser_throughput_medium",
        "files": len(py_paths),
        "entities": total_entities,
        "elapsed_s": round(elapsed, 3),
        "files_per_sec": round(files_per_sec, 1),
        "entities_per_sec": round(entities_per_sec, 1),
    }
    print(f"\n{json.dumps(report, indent=2)}")

    assert files_per_sec > 100, f"Parser too slow: {files_per_sec:.1f} files/sec (expected >100)"


def test_parse_medium_corpus_instruction_count(benchmark, bench_medium: tuple[Path, list[str]]):
    """The same parse loop, measured in instructions instead of seconds.

    The two throughput tests above are a floor -- "not catastrophically slow" -- and
    wall-clock is all they can be, because they are read by a human on one machine. They
    cannot detect a 15% regression, and on a shared CI runner they cannot detect a 50%
    one either.

    Instruction counts do not care what else the machine is doing, so this is the one
    that can gate a pull request. Parsing is the right place to spend it: it is pure CPU
    with no I/O, and it is where the time actually goes -- a profiled index of this
    repo's own parsing package spent 0.957s in the parse phase, 0.600s of that in our
    handlers against 0.129s in tree-sitter itself.

    Deliberately NOT applied to test_vector_search_latency, which is the test that
    prompted reaching for this. That one is dominated by a network round-trip to
    Memgraph, and counting instructions in the Python process would measure the driver
    rather than the query. Nothing here fixes measuring a remote server over a shared
    connection; its wide wall-clock budget stays the honest answer.

    Without `--codspeed` this is one ordinary pass over the corpus, so it costs a normal
    run nothing beyond the parse it was already doing.
    """
    root, rel_paths = bench_medium
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]
    sources = [(p, (root / p).read_bytes()) for p in py_paths]

    def parse_all() -> int:
        return sum(
            len(result.entities)
            for rel_path, source in sources
            if (result := parse_file(rel_path, source, project_name="bench")) is not None
        )

    # File reads are hoisted out of the measured callable on purpose: they are I/O, and
    # including them would make the count depend on the page cache.
    entities = benchmark(parse_all)

    assert entities > 0, "measured a parse that produced nothing"
