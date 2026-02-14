"""Memory benchmark.

Uses process RSS (via psutil if available, otherwise tracemalloc as fallback)
to measure actual memory consumption during parsing, including C extensions
like tree-sitter that tracemalloc cannot track.

Results are accumulated to simulate a real indexing pipeline that holds all
ParsedFile objects before writing to the graph.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from code_atlas.parser import ParsedFile, parse_file

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.bench, pytest.mark.slow]


def _get_rss_mb() -> float:
    """Return current process RSS in MB, or -1 if unavailable."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def _run_memory_bench(root: Path, rel_paths: list[str], label: str) -> None:
    """Parse all files, accumulate results, and report memory usage."""
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]

    rss_before = _get_rss_mb()

    # Accumulate all results like a real indexing pipeline would
    all_results: list[ParsedFile] = []
    total_entities = 0
    for rel_path in py_paths:
        abs_path = root / rel_path.replace("/", "\\")
        source = abs_path.read_bytes()
        result = parse_file(rel_path, source, project_name="bench")
        if result is not None:
            total_entities += len(result.entities)
            all_results.append(result)

    rss_after = _get_rss_mb()

    report: dict[str, object] = {
        "benchmark": f"parser_memory_{label}",
        "files_parsed": len(all_results),
        "entities": total_entities,
    }
    if rss_before >= 0:
        report["rss_before_mb"] = round(rss_before, 1)
        report["rss_after_mb"] = round(rss_after, 1)
        report["rss_delta_mb"] = round(rss_after - rss_before, 1)
    else:
        report["note"] = "install psutil for RSS measurement"

    print(f"\n{json.dumps(report, indent=2)}")

    # Keep reference alive until after measurement
    del all_results


def test_parser_memory_small(bench_small: tuple[Path, list[str]]):
    """Measure memory during parsing of the small codebase (results accumulated)."""
    root, rel_paths = bench_small
    _run_memory_bench(root, rel_paths, "small")


def test_parser_memory_medium(bench_medium: tuple[Path, list[str]]):
    """Measure memory during parsing of the medium codebase (results accumulated)."""
    root, rel_paths = bench_medium
    _run_memory_bench(root, rel_paths, "medium")
