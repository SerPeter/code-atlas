"""Concurrent query stress test.

Fires N concurrent graph_search queries and measures QPS and error rate.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.parser import parse_file

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph import GraphClient

pytestmark = [pytest.mark.bench, pytest.mark.integration]

_QUERIES = ["Class0_0", "method_1", "module_func_5", "Class1_10", "MODULE_CONST_42", "method_3"]


@pytest.fixture
async def seeded_graph(graph_client: GraphClient, bench_small: tuple[Path, list[str]]) -> GraphClient:
    """Seed graph with small codebase entities for concurrent testing."""
    await graph_client.ensure_schema()

    root, rel_paths = bench_small
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]
    project_name = "bench"
    await graph_client.merge_project_node(project_name)

    for rel_path in py_paths[:30]:
        abs_path = root / rel_path.replace("/", "\\")
        source = abs_path.read_bytes()
        result = parse_file(rel_path, source, project_name=project_name)
        if result is not None:
            for entity in result.entities:
                entity_dict = {
                    "uid": entity.qualified_name,
                    "project_name": project_name,
                    "name": entity.name,
                    "qualified_name": entity.qualified_name.split(":", 1)[1]
                    if ":" in entity.qualified_name
                    else entity.qualified_name,
                    "file_path": entity.file_path,
                    "kind": entity.kind,
                    "line_start": entity.line_start,
                    "line_end": entity.line_end,
                    "visibility": entity.visibility,
                    "content_hash": entity.content_hash,
                }
                label = entity.label.value
                set_clause = ", ".join(f"n.{k} = ${k}" for k in entity_dict)
                await graph_client.execute_write(
                    f"MERGE (n:{label} {{uid: $uid}}) SET {set_clause}",
                    entity_dict,
                )

    return graph_client


async def _run_concurrent_queries(graph: GraphClient, concurrency: int, total: int) -> dict:
    """Fire *total* graph_search queries at *concurrency* parallelism."""
    sem = asyncio.Semaphore(concurrency)
    errors = 0
    latencies: list[float] = []

    async def _query():
        nonlocal errors
        async with sem:
            name = random.choice(_QUERIES)
            t0 = time.perf_counter()
            try:
                await graph.graph_search(name, limit=5)
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1

    start = time.perf_counter()
    await asyncio.gather(*[_query() for _ in range(total)])
    wall = time.perf_counter() - start

    qps = total / wall if wall > 0 else 0
    return {
        "concurrency": concurrency,
        "total_queries": total,
        "errors": errors,
        "wall_s": round(wall, 3),
        "qps": round(qps, 1),
        "error_rate": round(errors / total, 4) if total else 0,
    }


@pytest.mark.parametrize("concurrency", [10, 50])
async def test_concurrent_graph_search(seeded_graph: GraphClient, concurrency: int):
    """Measure QPS at different concurrency levels."""
    total = concurrency * 5  # 5 queries per slot
    result = await _run_concurrent_queries(seeded_graph, concurrency, total)
    report = {"benchmark": f"concurrent_graph_search_c{concurrency}", **result}
    print(f"\n{json.dumps(report, indent=2)}")

    assert result["error_rate"] < 0.05, f"Error rate too high: {result['error_rate']}"
    if concurrency >= 50:
        assert result["qps"] > 50, f"QPS too low at c={concurrency}: {result['qps']}"
