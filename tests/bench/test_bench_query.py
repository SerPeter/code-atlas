"""Query latency benchmark.

Pre-seeds graph with small codebase + dummy embeddings, then measures
p50/p95/p99/max latency for each search type.
"""

from __future__ import annotations

import json
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

from code_atlas.parser import parse_file

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph import GraphClient

pytestmark = [pytest.mark.bench, pytest.mark.integration]

# Queries exercised across benchmarks
_SEARCH_NAMES = ["Class0_0", "method_1", "module_func_5", "Class1_10", "MODULE_CONST_42"]


def _percentiles(values: list[float]) -> dict[str, float]:
    """Compute p50/p95/p99/max from a list of floats (ms)."""
    if not values:
        return {"p50": 0, "p95": 0, "p99": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "p50": round(s[int(n * 0.50)], 2),
        "p95": round(s[int(n * 0.95)], 2),
        "p99": round(s[int(n * 0.99)], 2),
        "max": round(s[-1], 2),
    }


@pytest.fixture
async def seeded_graph(graph_client: GraphClient, bench_small: tuple[Path, list[str]]) -> GraphClient:
    """Seed graph with small codebase entities + dummy embeddings."""
    await graph_client.ensure_schema()

    root, rel_paths = bench_small
    py_paths = [p for p in rel_paths if p.endswith(".py") and "__init__" not in p]

    project_name = "bench"
    await graph_client.merge_project_node(project_name)

    # Parse and upsert a subset (first 50 files for speed)
    for rel_path in py_paths[:50]:
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
                    "docstring": entity.docstring or "",
                    "signature": entity.signature or "",
                    "content_hash": entity.content_hash,
                }
                label = entity.label.value
                set_clause = ", ".join(f"n.{k} = ${k}" for k in entity_dict)
                await graph_client.execute_write(
                    f"MERGE (n:{label} {{uid: $uid}}) SET {set_clause}",
                    entity_dict,
                )

    # Write dummy embeddings
    dim = graph_client._dimension
    rng = random.Random(42)
    records = await graph_client.execute(
        "MATCH (n {project_name: $p}) WHERE n.uid IS NOT NULL RETURN n.uid AS uid LIMIT 500",
        {"p": project_name},
    )
    items = [(r["uid"], [rng.gauss(0, 1) for _ in range(dim)]) for r in records]
    if items:
        await graph_client.write_embeddings(items)

    return graph_client


async def test_graph_search_latency(seeded_graph: GraphClient):
    """Measure graph_search p50/p95/p99/max over 50 iterations."""
    latencies: list[float] = []
    for _ in range(50):
        name = random.choice(_SEARCH_NAMES)
        t0 = time.perf_counter()
        await seeded_graph.graph_search(name, limit=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = _percentiles(latencies)
    report = {"benchmark": "graph_search_latency", "iterations": 50, **stats}
    print(f"\n{json.dumps(report, indent=2)}")
    assert stats["p95"] < 500, f"graph_search p95 too high: {stats['p95']}ms"


async def test_text_search_latency(seeded_graph: GraphClient):
    """Measure text_search (BM25) p50/p95/p99/max over 50 iterations."""
    latencies: list[float] = []
    for _ in range(50):
        query = random.choice(["Class", "method", "module_func", "Docstring", "process"])
        t0 = time.perf_counter()
        await seeded_graph.text_search(query, limit=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = _percentiles(latencies)
    report = {"benchmark": "text_search_latency", "iterations": 50, **stats}
    print(f"\n{json.dumps(report, indent=2)}")
    assert stats["p95"] < 200, f"text_search p95 too high: {stats['p95']}ms"


async def test_vector_search_latency(seeded_graph: GraphClient):
    """Measure vector_search p50/p95/p99/max over 50 iterations."""
    dim = seeded_graph._dimension
    rng = np.random.default_rng(42)
    latencies: list[float] = []

    for _ in range(50):
        vec = rng.standard_normal(dim).tolist()
        t0 = time.perf_counter()
        await seeded_graph.vector_search(vec, limit=10)
        latencies.append((time.perf_counter() - t0) * 1000)

    stats = _percentiles(latencies)
    report = {"benchmark": "vector_search_latency", "iterations": 50, **stats}
    print(f"\n{json.dumps(report, indent=2)}")
    assert stats["p95"] < 200, f"vector_search p95 too high: {stats['p95']}ms"
