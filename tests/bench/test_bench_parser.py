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

from code_atlas.parser import parse_file

pytestmark = [pytest.mark.bench, pytest.mark.slow]


def test_parser_throughput_small(bench_small: tuple[Path, list[str]]):
    """Parse all files in the small codebase and report throughput."""
    root, rel_paths = bench_small
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]

    total_entities = 0
    start = time.perf_counter()

    for rel_path in py_paths:
        abs_path = root / rel_path.replace("/", "\\")
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
        abs_path = root / rel_path.replace("/", "\\")
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
