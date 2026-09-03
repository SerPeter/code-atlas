"""Per-arm query cost on the embedded backend, and the guard that a broken index is slow news.

Three retrieval arms run concurrently inside `hybrid_search` and nothing times them
individually: they are `asyncio.create_task`'d and awaited in a bare loop, and
`channel_status` records only ok/error. What does separate them is the `op` label on
`atlas_graph_query_seconds`, so this measures each arm on its own by calling it directly.

**The failure this file exists to prevent.** `text_search._one` catches
`aiosqlite.OperationalError`, logs a warning and returns `[]` (`sqlite_graph.py:2799`).
A broken FTS index therefore produces *no results, very quickly* — which on a latency
benchmark is indistinguishable from a fast one, and better-looking than a working index.
Any query benchmark that measures time without asserting results is measuring exactly the
wrong thing when it matters most.

Runs on the embedded backend: no Docker, no network.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.bench import capture, stub_provider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.bench, pytest.mark.slow]

_FILES = 24
_QUERY = "helper0"


def _write_corpus(root: Path) -> None:
    pkg = root / "src"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for i in range(_FILES):
        (pkg / f"mod{i}.py").write_text(
            f'"""Module {i} with a helper."""\n\n\n'
            f'class Holder{i}:\n    """Holds things."""\n\n'
            f"    def run(self, x: int) -> int:\n        return helper{i}(x)\n\n\n"
            f'def helper{i}(x: int) -> int:\n    """A helper that helps."""\n    return x + {i}\n',
            encoding="utf-8",
        )
    (root / "README.md").write_text("# Corpus\n\nA helper is documented here.\n", encoding="utf-8")


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory):
    """One indexed SQLite graph, reused by every test here.

    Module-scoped because indexing costs ~15s and none of these tests mutate the graph —
    except the broken-index test, which drops a table and is therefore ordered to run in
    its own database.
    """
    import asyncio

    base = tmp_path_factory.mktemp("queryarms")
    root = base / "corpus"
    root.mkdir()
    _write_corpus(root)

    async def _build():
        from code_atlas.backends import use_backends
        from code_atlas.indexing.orchestrator import index_project
        from code_atlas.settings import AtlasSettings

        settings = AtlasSettings(
            project_root=root,
            backend={"graph": "sqlite", "queue": "sqlite", "sqlite_data_dir": str(base / "db")},
            embeddings={"dimension": 16},
        )
        with stub_provider(16):
            async with use_backends(settings, with_bus=True) as backends:
                # `index_project` does not create the schema -- the CLI calls `ensure_schema`
                # separately. Without it every FTS5 and vec0 write is swallowed by
                # `_safe_exec` as "no such table", so the measured system has no text or
                # vector index at all and its write cost is understated.
                await backends.graph.ensure_schema()
                await index_project(
                    settings,
                    backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                    backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                    full_reindex=True,
                    project_name="bench-arms",
                )
        return settings

    return asyncio.run(_build()), base


async def _open(settings):
    from code_atlas.backends import use_backends

    return use_backends(settings, with_bus=False)


class TestEachArmIsAttributed:
    async def test_arms_report_results_and_round_trips(self, indexed):
        """Each arm measured alone, because concurrently they cannot be told apart.

        Results are asserted, not just timings. An arm returning nothing is the cheapest
        possible arm, and a benchmark that only times it rewards the failure.
        """
        settings, _base = indexed
        cap = capture()
        rows: list[dict[str, int | str]] = []

        async with await _open(settings) as backends:
            graph = backends.graph
            dimension = 16
            for arm, call in (
                ("graph", lambda: graph.graph_search(_QUERY, limit=10)),
                ("bm25", lambda: graph.text_search(_QUERY, limit=10)),
                ("vector", lambda: graph.vector_search([0.1] * dimension, limit=10)),
            ):
                cap.clear()
                before = sum(cap.round_trips().values())
                results = await call()
                after = sum(cap.round_trips().values())
                rows.append({"arm": arm, "results": len(results), "round_trips": after - before})

        print(f"\n{json.dumps({'arms': rows}, indent=2)}")
        hits = {str(r["arm"]): int(r["results"]) for r in rows}
        assert hits["graph"] > 0, "the graph arm matched nothing — the corpus or query is wrong"
        assert hits["bm25"] > 0, (
            "the BM25 arm matched nothing. Two causes worth checking in order: the schema was "
            "never created (every FTS write is swallowed by `_safe_exec`), or the probe query "
            "is not a token the tokenizer produced — `helper` does not match `helper0`."
        )
        # The vector arm is queried with an arbitrary vector, so it returns nearest
        # neighbours rather than matches; non-empty is the only meaningful claim.
        assert hits["vector"] > 0, "the vector arm returned nothing at all"

    async def test_vector_search_cost_is_vectors_compared(self, indexed):
        """sqlite-vec's `vec0` is brute force, so cost is vectors compared, not time.

        A deterministic number that grows with the corpus, which is what makes it worth
        reporting instead of a millisecond figure that depends on the machine.
        """
        settings, _base = indexed
        async with await _open(settings) as backends:
            info = await backends.graph.get_vector_index_info()

        total = sum(int(entry.get("size", 0) or 0) for entry in info)
        print(f"\nvector index sizes: {info}  -> vectors compared per brute-force search: {total}")
        assert total > 0, "no vectors indexed, so the vector arm above searched nothing"


class TestABrokenIndexIsNotAFastOne:
    """The sharpest failure a latency-only query benchmark cannot see."""

    async def test_dropping_an_fts_table_yields_no_results_and_no_error(self, tmp_path):
        """Demonstrates the hazard before asserting the guard.

        `_one` catches `OperationalError`, logs a warning and returns `[]`, so the call
        succeeds, returns fast, and reports nothing. On a p95 latency chart that is an
        improvement.
        """
        from code_atlas.backends import use_backends
        from code_atlas.indexing.orchestrator import index_project
        from code_atlas.settings import AtlasSettings

        root = tmp_path / "corpus"
        root.mkdir()
        _write_corpus(root)
        settings = AtlasSettings(
            project_root=root,
            backend={"graph": "sqlite", "queue": "sqlite", "sqlite_data_dir": str(tmp_path / "db")},
            embeddings={"dimension": 16},
        )

        with stub_provider(16):
            async with use_backends(settings, with_bus=True) as backends:
                # `index_project` does not create the schema -- the CLI calls `ensure_schema`
                # separately. Without it every FTS5 and vec0 write is swallowed by
                # `_safe_exec` as "no such table", so the measured system has no text or
                # vector index at all and its write cost is understated.
                await backends.graph.ensure_schema()
                await index_project(
                    settings,
                    backends.graph,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                    backends.bus,  # ty: ignore[invalid-argument-type]  # index_project accepts either backend
                    full_reindex=True,
                    project_name="bench-broken",
                )

        async with use_backends(settings, with_bus=False) as backends:
            graph = backends.graph
            healthy = await graph.text_search(_QUERY, limit=10)
            assert healthy, "the corpus does not match the probe query, so the sabotage proves nothing"

            # Narrowed rather than ignored: this sabotage is SQLite-specific, and if the
            # file is ever pointed at Memgraph it should stop rather than skip silently.
            assert isinstance(graph, SqliteGraphClient), "the FTS sabotage is SQLite-only"
            conn = await graph._get_conn()
            for table in ("text_callable", "text_typedef", "text_module"):
                await conn.execute(f"DROP TABLE IF EXISTS {table}")
            await conn.commit()

            broken = await graph.text_search(_QUERY, limit=10)

        assert broken == [], (
            "dropping every FTS table should make text search return nothing — if it still "
            "returns results the swallow has moved and this guard needs rewriting"
        )
