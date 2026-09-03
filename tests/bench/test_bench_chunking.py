"""The chunking regression, re-created: `max_source_chars` can make `EmbedChunk` unreachable.

One of the four defects this epic exists to catch. `index.max_source_chars` truncates an
entity's stored source *before* the embed stage ever sees it, so setting it below the
model's input cap means no text is ever long enough to split — `EmbedChunk` is not
merely rare, it is impossible, and every measurement of the chunking path silently
measures nothing.

The old bench could not see this for two reasons, both structural: its `EmbedClient` was
an `AsyncMock` returning a single vector for any batch (so chunking never ran at all),
and its synthetic corpus contained no entity remotely near a model cap.

This runs on the embedded backend and the in-process provider stub, so it needs no Docker
and no network.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from code_atlas.bench import capture, profile_corpus, stub_provider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = [pytest.mark.bench, pytest.mark.slow]

# The cap and the truncation are not in the same currency, and that is the whole trap:
# `max_input_tokens` counts TOKENS while `max_source_chars` counts CHARACTERS, at roughly
# four chars per token.
#
# Chunking is unreachable exactly when `max_source_chars < max_input_tokens * ~4`. The
# first attempt at this test used a 64-token cap (~256 chars) against a 2000-char
# truncation, and both arms chunked identically — 2000 chars is still ~500 tokens, far
# over 64. Getting that relationship backwards makes the regression arm pass for the
# wrong reason, which is exactly the failure mode this file exists to catch elsewhere.
#
# 2000 tokens is ~8000 chars, so a 2000-CHAR truncation lands well under it.
_INPUT_TOKEN_CAP = 2000
_TRUNCATED_CHARS = 2000
_GENEROUS_CHARS = 200_000


def _write_oversized_corpus(root: Path) -> None:
    """One function far past the token cap, plus ordinary ones that fit.

    The body is distinct lines rather than a repeated one so the splitter has real
    boundaries to cut on, and so the chunks differ from each other — identical chunks
    would collapse under the graph dedup layer and the count would be a different fact
    than the one being asserted.
    """
    pkg = root / "src"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")

    body = "\n".join(f"    step_{i} = compute({i}) + offset_{i} * factor_{i}" for i in range(400))
    (pkg / "huge.py").write_text(
        f'"""A module with one very large function."""\n\n\n'
        f"def compute(n: int) -> int:\n    return n + 1\n\n\n"
        f'def enormous() -> int:\n    """Far past any sane model input cap."""\n{body}\n    return 0\n',
        encoding="utf-8",
    )
    (pkg / "small.py").write_text(
        '"""Ordinary module."""\n\n\ndef tiny(x: int) -> int:\n    return x + 1\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text("# Corpus\n\nMarkdown so the doc path runs.\n", encoding="utf-8")


async def _index(root: Path, db_dir: Path, project: str, *, max_source_chars: int) -> dict[str, int]:
    """Index *root* and return the corpus label counts."""
    from code_atlas.backends import use_backends
    from code_atlas.indexing.orchestrator import index_project
    from code_atlas.settings import AtlasSettings

    settings = AtlasSettings(
        project_root=root,
        backend={"graph": "sqlite", "queue": "sqlite", "sqlite_data_dir": str(db_dir)},
        # An explicit cap is required, not optional: litellm's registry has no entry for
        # a TEI- or OpenRouter-prefixed model, and an unknown cap means no chunking and
        # no truncation at all (ADR-0040). Leaving it unset would make this test pass
        # vacuously by never splitting anything.
        embeddings={"dimension": 32, "max_input_tokens": _INPUT_TOKEN_CAP},
        index={"max_source_chars": max_source_chars},
    )
    cap = capture()
    cap.clear()
    with stub_provider(32):
        async with use_backends(settings, with_bus=True) as backends:
            # `index_project` does not create the schema -- the CLI calls `ensure_schema`
            # separately. Without it every FTS5 and vec0 write is swallowed by
            # `_safe_exec` as "no such table", so the measured system has no text or
            # vector index at all and its write cost is understated.
            await backends.graph.ensure_schema()
            await index_project(
                settings,
                backends.graph,  # ty: ignore[invalid-argument-type]
                backends.bus,  # ty: ignore[invalid-argument-type]
                full_reindex=True,
                project_name=project,
            )
            profile = await profile_corpus(backends.graph, project)
    return profile.labels


@pytest.fixture
def oversized_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    _write_oversized_corpus(root)
    return root


class TestChunkingIsReachable:
    async def test_a_generous_source_cap_produces_embed_chunks(self, oversized_corpus: Path, tmp_path: Path):
        """The control arm. Without it the regression arm proves nothing.

        If this ever stops producing chunks, the test below starts passing for the wrong
        reason — it would be asserting zero against zero.
        """
        labels = await _index(oversized_corpus, tmp_path / "db_ok", "bench-chunk-ok", max_source_chars=_GENEROUS_CHARS)
        assert labels.get("EmbedChunk", 0) > 0, (
            f"the oversized entity produced no EmbedChunk at all, so this corpus cannot detect the regression: {labels}"
        )

    async def test_truncating_below_the_model_cap_makes_chunking_unreachable(
        self, oversized_corpus: Path, tmp_path: Path
    ):
        """The regression itself, re-created.

        `max_source_chars = 2000` is the value that shipped. It truncates the entity's
        source before the embed stage sees it, so nothing is ever long enough to split
        and `EmbedChunk` becomes unreachable — not rare, impossible.

        The failure mode this guards is quiet: chunking measurements keep reporting, they
        just report zero, and zero reads as "this corpus has no oversized entities"
        rather than as "the cap made them invisible".
        """
        labels = await _index(
            oversized_corpus, tmp_path / "db_bad", "bench-chunk-bad", max_source_chars=_TRUNCATED_CHARS
        )
        assert labels.get("EmbedChunk", 0) == 0, (
            "truncation below the model cap should make chunking unreachable — if this "
            "now produces chunks, the interaction has changed and the guard needs rewriting"
        )

    async def test_the_two_arms_disagree(self, oversized_corpus: Path, tmp_path: Path):
        """The whole point, stated as one comparison.

        Same corpus, same model cap, one setting different — and the chunking path either
        runs or does not. A single-arm assertion could not tell that apart from a corpus
        that simply has nothing large in it.
        """
        generous = await _index(oversized_corpus, tmp_path / "db_a", "bench-chunk-a", max_source_chars=_GENEROUS_CHARS)
        truncated = await _index(
            oversized_corpus, tmp_path / "db_b", "bench-chunk-b", max_source_chars=_TRUNCATED_CHARS
        )

        assert generous.get("EmbedChunk", 0) > truncated.get("EmbedChunk", 0), (
            f"max_source_chars did not change chunk reachability: "
            f"{generous.get('EmbedChunk', 0)} vs {truncated.get('EmbedChunk', 0)}"
        )
        # The entities themselves must survive either way. If truncation also dropped
        # entities, the comparison above would be measuring the wrong difference.
        assert generous.get("Callable", 0) == truncated.get("Callable", 0), (
            "truncation changed the entity count, so the chunk comparison is not isolating chunking"
        )


class TestTimingIsReported:
    async def test_the_run_reports_a_chunking_cost(self, oversized_corpus: Path, tmp_path: Path):
        """An oversized entity should make the tokenizer work visibly harder.

        `split_text` re-encodes the same text repeatedly walking down the border ladder,
        so a corpus with something to split costs materially more tokenizer calls than
        one without. Reported rather than budgeted — the count is the deliverable.
        """
        started = time.perf_counter()
        with stub_provider(32) as stats:
            labels = await _index(
                oversized_corpus, tmp_path / "db_t", "bench-chunk-timing", max_source_chars=_GENEROUS_CHARS
            )
        elapsed = time.perf_counter() - started

        print(
            f"\nchunking run: {labels.get('EmbedChunk', 0)} EmbedChunk, "
            f"{stats.tokenizer_calls} tokenizer calls, {elapsed:.2f}s"
        )
        assert labels.get("EmbedChunk", 0) > 0
