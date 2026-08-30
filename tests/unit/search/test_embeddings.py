"""Unit tests for the embeddings module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from code_atlas.chunking import CHARS_PER_TOKEN_FALLBACK, split_embed_text
from code_atlas.events import EmbedDirty, EntityRef
from code_atlas.indexing.consumers import EmbedConsumer
from code_atlas.search.embeddings import EmbedClient, EmbeddingError, build_embed_text, hash_text
from code_atlas.settings import EmbeddingSettings

# ---------------------------------------------------------------------------
# build_embed_text tests
# ---------------------------------------------------------------------------


class TestBuildEmbedText:
    def test_callable_method(self):
        props = {
            "_label": "Callable",
            "qualified_name": "myapp.parser.Parser.process",
            "kind": "method",
            "signature": "Parser.process(self, source: str) -> ParsedFile",
            "docstring": "Parse source code into a structured representation.",
        }
        text = build_embed_text(props)
        assert "Module: myapp.parser" in text
        assert "Class: Parser" in text
        assert "Method: Parser.process(self, source: str) -> ParsedFile" in text
        assert '"""Parse source code into a structured representation."""' in text

    def test_callable_function(self):
        props = {
            "_label": "Callable",
            "qualified_name": "myapp.utils.format_path",
            "kind": "function",
            "signature": "format_path(path: Path) -> str",
            "docstring": "Format a path for display.",
        }
        text = build_embed_text(props)
        assert "Module: myapp.utils" in text
        assert "Function: format_path(path: Path) -> str" in text
        assert '"""Format a path for display."""' in text
        # Should NOT have Class line for top-level function
        assert "Class:" not in text

    def test_typedef_class(self):
        props = {
            "_label": "TypeDef",
            "qualified_name": "myapp.parser.Parser",
            "kind": "class",
            "signature": "",
            "docstring": "AST parser using tree-sitter.",
        }
        text = build_embed_text(props)
        assert "Module: myapp.parser" in text
        assert "Class: Parser" in text
        assert '"""AST parser using tree-sitter."""' in text

    def test_value_constant(self):
        props = {
            "_label": "Value",
            "qualified_name": "myapp.settings.DEFAULT_TIMEOUT",
            "kind": "constant",
            "signature": "",
            "docstring": "",
        }
        text = build_embed_text(props)
        assert "Module: myapp.settings" in text
        assert "Constant: DEFAULT_TIMEOUT" in text

    def test_module(self):
        props = {
            "_label": "Module",
            "qualified_name": "myapp.parser",
            "kind": "",
            "signature": "",
            "docstring": "AST parser using tree-sitter for Python source files.",
        }
        text = build_embed_text(props)
        assert "Module: myapp.parser" in text
        assert '"""AST parser using tree-sitter for Python source files."""' in text

    def test_doc_section(self):
        props = {
            "_label": "DocSection",
            "qualified_name": "docs/architecture.md > Architecture > Event Pipeline > AST Stage",
            "kind": "",
            "signature": "",
            "docstring": "The AST stage processes file changes...",
        }
        text = build_embed_text(props)
        assert "File: docs/architecture.md" in text
        assert "Section: Architecture > Event Pipeline > AST Stage" in text
        assert '"""The AST stage processes file changes..."""' in text

    def test_empty_qualified_name_returns_empty(self):
        props = {"_label": "Callable", "qualified_name": "", "kind": "function"}
        assert build_embed_text(props) == ""

    def test_missing_props_graceful(self):
        props = {"_label": "Callable", "qualified_name": "foo.bar", "kind": "function"}
        text = build_embed_text(props)
        assert "Module: foo" in text
        assert "Function: bar" in text

    def test_no_docstring_omitted(self):
        props = {
            "_label": "Callable",
            "qualified_name": "foo.bar",
            "kind": "function",
            "signature": "bar()",
            "docstring": "",
        }
        text = build_embed_text(props)
        assert '"""' not in text

    def test_source_in_embed_text(self):
        """Source appears in embed text after docstring."""
        source_code = (
            "def retry(fn, max_attempts=3):\n"
            "    for i in range(max_attempts):\n"
            "        try:\n"
            "            return fn()\n"
            "        except Exception:\n"
            "            time.sleep(2**i)"
        )
        props = {
            "_label": "Callable",
            "qualified_name": "myapp.utils.retry",
            "kind": "function",
            "signature": "retry(fn, max_attempts=3)",
            "docstring": "Retry with backoff.",
            "source": source_code,
        }
        text = build_embed_text(props)
        assert '"""Retry with backoff."""' in text
        assert "def retry(fn, max_attempts=3):" in text
        assert "time.sleep(2**i)" in text

    def test_source_without_docstring(self):
        """Source still included when docstring is empty."""
        props = {
            "_label": "Callable",
            "qualified_name": "foo.bar",
            "kind": "function",
            "signature": "bar(x)",
            "docstring": "",
            "source": "def bar(x):\n    return x + 1",
        }
        text = build_embed_text(props)
        assert "def bar(x):" in text
        assert "return x + 1" in text


# ---------------------------------------------------------------------------
# EmbedClient tests (mocked litellm)
# ---------------------------------------------------------------------------


def _make_settings(**kwargs: Any) -> EmbeddingSettings:
    defaults: dict[str, Any] = {
        "model": "nomic-ai/nomic-embed-code",
        "base_url": "http://localhost:8080",
        "dimension": 768,
        "batch_size": 32,
        "timeout_s": 30.0,
    }
    defaults.update(kwargs)
    return EmbeddingSettings(**defaults)


@dataclass
class FakeEmbeddingItem:
    embedding: list[float]


@dataclass
class FakeEmbeddingResponse:
    data: list[FakeEmbeddingItem]


class TestEmbedClient:
    def test_model_string_with_base_url(self):
        client = EmbedClient(_make_settings())
        assert client._model == "openai/nomic-ai/nomic-embed-code"
        assert client._api_base == "http://localhost:8080"
        assert client._api_key == "unused"

    def test_model_string_already_prefixed(self):
        client = EmbedClient(_make_settings(model="openai/my-model"))
        assert client._model == "openai/my-model"

    def test_model_string_cloud_provider(self):
        client = EmbedClient(_make_settings(provider="litellm", base_url=""))
        assert client._model == "nomic-ai/nomic-embed-code"
        assert client._api_base is None
        assert client._api_key is None

    def test_model_string_ollama_provider(self):
        client = EmbedClient(_make_settings(provider="ollama", base_url="http://localhost:11434"))
        assert client._model == "nomic-ai/nomic-embed-code"
        assert client._api_base == "http://localhost:11434"
        assert client._api_key is None

    def test_dimensions_forwarded_for_litellm_provider(self):
        client = EmbedClient(_make_settings(provider="litellm", base_url="", dimension=1536))
        assert client._build_kwargs(["hello"])["dimensions"] == 1536

    def test_dimensions_omitted_when_unset(self):
        client = EmbedClient(_make_settings(provider="litellm", base_url="", dimension=None))
        assert "dimensions" not in client._build_kwargs(["hello"])

    def test_dimensions_not_forwarded_for_tei_provider(self):
        client = EmbedClient(_make_settings(provider="tei", dimension=768))
        assert "dimensions" not in client._build_kwargs(["hello"])

    async def test_embed_one(self):
        client = EmbedClient(_make_settings())
        fake_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1, 0.2, 0.3])])

        patch_target = "code_atlas.search.embeddings.litellm.aembedding"
        with patch(patch_target, new_callable=AsyncMock, return_value=fake_response):
            result = await client.embed_one("hello")

        assert result == [0.1, 0.2, 0.3]

    async def test_embed_batch_single_chunk(self):
        client = EmbedClient(_make_settings(batch_size=32))
        texts = ["text1", "text2", "text3"]
        fake_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[float(i)]) for i in range(3)])

        with patch(
            "code_atlas.search.embeddings.litellm.aembedding", new_callable=AsyncMock, return_value=fake_response
        ) as mock_embed:
            result = await client.embed_batch(texts)

        assert len(result) == 3
        mock_embed.assert_called_once()

    async def test_embed_batch_multiple_chunks(self):
        client = EmbedClient(_make_settings(batch_size=3))
        texts = [f"text{i}" for i in range(10)]

        call_count = 0

        async def fake_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            nonlocal call_count
            call_count += 1
            n = len(kwargs["input"])
            return FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[float(call_count)]) for _ in range(n)])

        with patch("code_atlas.search.embeddings.litellm.aembedding", side_effect=fake_aembedding):
            result = await client.embed_batch(texts)

        assert len(result) == 10
        # 10 texts / batch_size 3 = 4 calls (3 + 3 + 3 + 1)
        assert call_count == 4

    async def test_embed_batch_empty(self):
        client = EmbedClient(_make_settings())
        result = await client.embed_batch([])
        assert result == []

    async def test_embed_error_propagation(self):
        client = EmbedClient(_make_settings())

        with (
            patch(
                "code_atlas.search.embeddings.litellm.aembedding",
                new_callable=AsyncMock,
                side_effect=Exception("Connection refused"),
            ),
            pytest.raises(EmbeddingError, match="Connection refused"),
        ):
            await client.embed_one("test")

    async def test_embed_transient_error_retries_then_succeeds(self, no_retry_backoff):
        """A transient provider error (rate limit) is retried, not raised immediately.

        The backoff is zeroed, not the policy: two real retries at the shipped
        wait_exponential(min=1) cost 3.02s of sleeping, and the attempt count and retry
        predicate -- the parts actually under test -- are untouched by dropping it.
        """
        no_retry_backoff(EmbedClient._embed_call)
        client = EmbedClient(_make_settings())
        fake_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1, 0.2, 0.3])])

        call_count = 0

        async def flaky_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise litellm.RateLimitError(message="rate limited", llm_provider="openai", model="test-model")
            return fake_response

        with patch("code_atlas.search.embeddings.litellm.aembedding", side_effect=flaky_aembedding):
            result = await client.embed_one("hello")

        assert result == [0.1, 0.2, 0.3]
        assert call_count == 3

    async def test_embed_non_retryable_error_fails_immediately(self):
        """A non-retryable error (e.g. bad request) is not retried."""
        client = EmbedClient(_make_settings())

        call_count = 0

        async def failing_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            nonlocal call_count
            call_count += 1
            raise litellm.BadRequestError(message="invalid input", llm_provider="openai", model="test-model")

        with (
            patch("code_atlas.search.embeddings.litellm.aembedding", side_effect=failing_aembedding),
            pytest.raises(EmbeddingError),
        ):
            await client.embed_one("hello")

        assert call_count == 1

    async def test_concurrent_ordering_preserved(self):
        """Results maintain correct ordering despite concurrent execution with varying delays."""
        client = EmbedClient(_make_settings(batch_size=2, max_concurrency=4))
        texts = [f"text{i}" for i in range(6)]

        # Simulate varying response times — later chunks respond faster
        async def fake_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            inputs = kwargs["input"]
            # Return vectors that encode chunk identity via values
            vecs = [FakeEmbeddingItem(embedding=[float(ord(t[-1]))]) for t in inputs]
            return FakeEmbeddingResponse(data=vecs)

        with patch("code_atlas.search.embeddings.litellm.aembedding", side_effect=fake_aembedding):
            result = await client.embed_batch(texts)

        assert len(result) == 6
        # Each vector should correspond to the original text's last char
        for i, vec in enumerate(result):
            expected = float(ord(str(i)))
            assert vec == [expected], f"result[{i}] ordering mismatch"

    async def test_concurrent_partial_failure(self):
        """When one chunk fails during concurrent execution, the error propagates."""
        client = EmbedClient(_make_settings(batch_size=2, max_concurrency=4))
        texts = [f"text{i}" for i in range(6)]

        call_count = 0

        async def fake_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise EmbeddingError("Chunk 2 failed")
            n = len(kwargs["input"])
            return FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[1.0]) for _ in range(n)])

        with (
            patch("code_atlas.search.embeddings.litellm.aembedding", side_effect=fake_aembedding),
            pytest.raises(EmbeddingError, match="Chunk 2 failed"),
        ):
            await client.embed_batch(texts)

    async def test_concurrency_limited_by_semaphore(self):
        """Semaphore limits the number of concurrent API calls to max_concurrency."""
        client = EmbedClient(_make_settings(batch_size=1, max_concurrency=2))
        texts = [f"text{i}" for i in range(5)]

        peak_concurrent = 0
        current_concurrent = 0

        async def fake_aembedding(**kwargs: Any) -> FakeEmbeddingResponse:
            nonlocal peak_concurrent, current_concurrent
            current_concurrent += 1
            peak_concurrent = max(peak_concurrent, current_concurrent)
            await asyncio.sleep(0.01)  # Yield to let other tasks run
            current_concurrent -= 1
            n = len(kwargs["input"])
            return FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[1.0]) for _ in range(n)])

        with patch("code_atlas.search.embeddings.litellm.aembedding", side_effect=fake_aembedding):
            result = await client.embed_batch(texts)

        assert len(result) == 5
        assert peak_concurrent <= 2, f"Peak concurrent calls ({peak_concurrent}) exceeded max_concurrency (2)"

    async def test_health_check_success(self):
        client = EmbedClient(_make_settings())
        fake_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1])])

        patch_target = "code_atlas.search.embeddings.litellm.aembedding"
        with patch(patch_target, new_callable=AsyncMock, return_value=fake_response):
            assert await client.health_check() is True

    async def test_health_check_failure(self):
        client = EmbedClient(_make_settings())

        with patch(
            "code_atlas.search.embeddings.litellm.aembedding",
            new_callable=AsyncMock,
            side_effect=Exception("down"),
        ):
            assert await client.health_check() is False


# ---------------------------------------------------------------------------
# Embed consumer cache integration tests (mocked graph + embed + cache)
# ---------------------------------------------------------------------------


class TestEmbedDedupLookup:
    """The embed stage's two-level lookup, with mocks (ADR-0036).

    1. unchanged -- the node's own embed_hash still matches, nothing to do
    2. dedup     -- some node somewhere already has a vector for this exact text
    3. API       -- genuinely new text

    Level 2 used to be a Valkey cache holding a copy of what level 1 reads out of the
    graph. Measured on the production instance, 98.8% of its 32,385 keys were already
    on a graph node, it shared a `noeviction` instance with the event streams and the
    indexer lease, and it filled that instance and failed their writes.
    """

    @staticmethod
    def _make_entity_ref(qn: str) -> EntityRef:
        return EntityRef(qualified_name=qn, node_type="Callable", file_path="f.py")

    @staticmethod
    def _make_embed_dirty(entity: EntityRef) -> EmbedDirty:
        return EmbedDirty(entity=entity, significance="HIGH")

    @staticmethod
    def _mock_embed() -> AsyncMock:
        """Build an EmbedClient mock with required properties."""
        mock = AsyncMock()
        mock.batch_size = 32
        mock.max_concurrency = 4
        # Not a MagicMock: the consumer stamps this onto every vector it writes, and a
        # MagicMock reaches the Bolt driver as an unserialisable parameter.
        mock.configured_model = "test-model"
        return mock

    @staticmethod
    def _props(text_hash: str | None, has_embedding: bool) -> dict:
        return {
            "uid": "foo.bar",
            "qualified_name": "foo.bar",
            "name": "bar",
            "signature": "",
            "docstring": "",
            "kind": "function",
            "_label": "Callable",
            "embed_hash": text_hash,
            "has_embedding": has_embedding,
        }

    async def test_unchanged_entity_skips_everything(self):
        """Level 1: the node's own hash still matches, so nothing downstream runs."""
        bus, graph, embed = AsyncMock(), AsyncMock(), self._mock_embed()
        text_hash = hash_text("Module: foo\nFunction: bar")
        graph.read_entity_texts = AsyncMock(return_value=[self._props(text_hash, True)])
        graph.find_embeddings_by_hash = AsyncMock(return_value={})

        consumer = EmbedConsumer(bus, graph, embed)
        await consumer.process_batch([self._make_embed_dirty(self._make_entity_ref("foo.bar"))], "t01")

        embed.embed_batch.assert_not_called()
        graph.find_embeddings_by_hash.assert_not_called()
        graph.write_embeddings_and_hashes.assert_not_called()

    async def test_duplicate_text_is_copied_from_the_graph(self):
        """Level 2: another node already has a vector for this text -- no API call.

        This is the moved-file / second-worktree / copied-helper case, and it is the
        one thing the deleted Valkey cache genuinely did that level 1 does not.
        """
        bus, graph, embed = AsyncMock(), AsyncMock(), self._mock_embed()
        text_hash = hash_text("Module: foo\nFunction: bar")
        existing_vec = [0.1, 0.2, 0.3]
        graph.read_entity_texts = AsyncMock(return_value=[self._props(None, False)])
        graph.find_embeddings_by_hash = AsyncMock(return_value={text_hash: existing_vec})

        consumer = EmbedConsumer(bus, graph, embed)
        await consumer.process_batch([self._make_embed_dirty(self._make_entity_ref("foo.bar"))], "t02")

        embed.embed_batch.assert_not_called()
        graph.find_embeddings_by_hash.assert_called_once()
        assert graph.find_embeddings_by_hash.call_args[0][1] == "test-model"
        graph.write_embeddings_and_hashes.assert_called_once()
        assert graph.write_embeddings_and_hashes.call_args[0][0][0] == ("foo.bar", existing_vec, text_hash)

    async def test_genuinely_new_text_calls_the_api(self):
        bus, graph, embed = AsyncMock(), AsyncMock(), self._mock_embed()
        text_hash = hash_text("Module: foo\nFunction: bar")
        api_vec = [0.5, 0.6, 0.7]
        graph.read_entity_texts = AsyncMock(return_value=[self._props(None, False)])
        graph.find_embeddings_by_hash = AsyncMock(return_value={})
        embed.embed_batch = AsyncMock(return_value=[api_vec])

        consumer = EmbedConsumer(bus, graph, embed)
        await consumer.process_batch([self._make_embed_dirty(self._make_entity_ref("foo.bar"))], "t03")

        embed.embed_batch.assert_called_once()
        graph.write_embeddings_and_hashes.assert_called_once()
        assert graph.write_embeddings_and_hashes.call_args[0][0][0] == ("foo.bar", api_vec, text_hash)

    async def test_the_lookup_is_asked_for_this_projects_model(self):
        """A vector only means anything inside its own model's space.

        The backend filters on the model; this pins that the consumer actually passes
        the configured one rather than letting the predicate default away. Two models
        were measured coexisting at 1536d in one database (ADR-0035), so an unfiltered
        copy mixes spaces with no dimension error to catch it.
        """
        bus, graph, embed = AsyncMock(), AsyncMock(), self._mock_embed()
        embed.configured_model = "model-b"
        graph.read_entity_texts = AsyncMock(return_value=[self._props(None, False)])
        graph.find_embeddings_by_hash = AsyncMock(return_value={})
        embed.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])

        consumer = EmbedConsumer(bus, graph, embed)
        await consumer.process_batch([self._make_embed_dirty(self._make_entity_ref("foo.bar"))], "t04")

        assert graph.find_embeddings_by_hash.call_args[0][1] == "model-b"
        assert graph.write_embeddings_and_hashes.call_args[1]["model"] == "model-b"

    async def test_identical_texts_in_one_batch_embed_once(self):
        """Two entities sharing a text pay the provider once, not twice.

        The one hit class the Valkey cache could never serve: `--full` cleared it
        before the run, so within-run duplicates always missed it.
        """
        bus, graph, embed = AsyncMock(), AsyncMock(), self._mock_embed()
        shared = [0.9, 0.8]
        graph.read_entity_texts = AsyncMock(
            return_value=[
                {**self._props(None, False), "uid": "a", "qualified_name": "foo.bar"},
                {**self._props(None, False), "uid": "b", "qualified_name": "foo.bar"},
            ]
        )
        graph.find_embeddings_by_hash = AsyncMock(return_value={})
        embed.embed_batch = AsyncMock(return_value=[shared])

        consumer = EmbedConsumer(bus, graph, embed)
        await consumer.process_batch(
            [
                self._make_embed_dirty(self._make_entity_ref("a")),
                self._make_embed_dirty(self._make_entity_ref("b")),
            ],
            "t05",
        )

        # One text sent to the provider...
        assert len(embed.embed_batch.call_args[0][0]) == 1
        # ...and both entities written, from that one vector.
        written = graph.write_embeddings_and_hashes.call_args[0][0]
        assert len(written) == 2
        assert all(w[1] == shared for w in written)


# ---------------------------------------------------------------------------
# split_embed_text tests
# ---------------------------------------------------------------------------


def _chars(text: str) -> int:
    """Measure in characters — one token per character makes the limits readable."""
    return len(text)


class TestSplitEmbedText:
    def test_short_text_is_one_chunk(self):
        chunks, hard = split_embed_text("hello", limit=100, measure=_chars)
        assert chunks == ["hello"]
        assert hard is False

    def test_empty_text_yields_no_chunks(self):
        assert split_embed_text("", limit=100, measure=_chars) == ([], False)

    def test_unknown_limit_returns_text_whole(self):
        """limit<=0 means the registry had no answer; guessing one is worse than not."""
        text = "x" * 5000
        assert split_embed_text(text, limit=0, measure=_chars) == ([text], False)

    def test_splits_on_paragraph_border_before_line_border(self):
        text = "a\nb\n\nc\nd"
        chunks, hard = split_embed_text(text, limit=4, measure=_chars)
        assert chunks == ["a\nb", "c\nd"]
        assert hard is False

    def test_packs_greedily_rather_than_one_chunk_per_line(self):
        """The ladder cuts at a border, not at every border."""
        text = "\n".join("line" for _ in range(6))  # 6 * 4 + 5 = 29 chars
        chunks, _ = split_embed_text(text, limit=14, measure=_chars)
        assert chunks == ["line\nline\nline", "line\nline\nline"]

    def test_every_chunk_is_within_the_limit(self):
        text = "\n\n".join("para " * 20 for _ in range(10))
        chunks, _ = split_embed_text(text, limit=200, measure=_chars, max_chunks=99)
        assert chunks
        assert all(_chars(c) <= 200 for c in chunks)

    def test_hard_split_when_no_border_exists(self):
        """A base64 blob or minified bundle has no border to cut at."""
        chunks, hard = split_embed_text("x" * 250, limit=100, measure=_chars, max_chunks=99)
        assert hard is True
        assert [len(c) for c in chunks] == [100, 100, 50]
        assert "".join(chunks) == "x" * 250

    def test_hard_split_is_not_reported_for_a_clean_border_split(self):
        chunks, hard = split_embed_text("aaaa\n\nbbbb", limit=5, measure=_chars)
        assert chunks == ["aaaa", "bbbb"]
        assert hard is False

    def test_max_chunks_caps_the_tail(self):
        chunks, _ = split_embed_text("x" * 1000, limit=10, measure=_chars, max_chunks=3)
        assert len(chunks) == 3

    def test_dense_tokenizer_still_terminates(self):
        """A measure that reports far more units than characters must converge."""
        chunks, hard = split_embed_text("y" * 300, limit=10, measure=lambda t: len(t) * 7, max_chunks=400)
        assert hard is True
        assert all(len(c) * 7 <= 10 for c in chunks)
        assert "".join(chunks) == "y" * 300


class TestEmbedClientSplitText:
    def test_unmapped_model_has_no_limit_without_an_override(self):
        """This is the production failure: no cap, no chunking, no truncation."""
        client = EmbedClient(_make_settings(model="nomic-ai/nomic-embed-code"))
        assert client._max_input_tokens is None
        chunks, hard = client.split_text("word " * 20_000)
        assert len(chunks) == 1
        assert hard is False

    def test_explicit_max_input_tokens_restores_the_cap(self):
        client = EmbedClient(_make_settings(max_input_tokens=2048, truncate_ratio=1.0))
        assert client._max_input_tokens == 2048
        chunks, _ = client.split_text("word " * 20_000)
        assert len(chunks) > 1
        assert all(client.count_tokens(c) <= 2048 for c in chunks)

    def test_override_beats_the_registry(self):
        """A registry answer of 8191 must not override a deliberate 512."""
        client = EmbedClient(
            _make_settings(provider="litellm", base_url="", model="text-embedding-3-small", max_input_tokens=512)
        )
        assert client._max_input_tokens == int(512 * 0.9)

    def test_truncate_ratio_applies_to_the_override(self):
        client = EmbedClient(_make_settings(max_input_tokens=1000, truncate_ratio=0.9))
        assert client._max_input_tokens == 900

    def test_max_chunks_setting_bounds_the_split(self):
        client = EmbedClient(_make_settings(max_input_tokens=64, truncate_ratio=1.0, max_chunks=2))
        chunks, _ = client.split_text("word " * 5_000)
        assert len(chunks) == 2

    def test_count_tokens_falls_back_when_no_tokenizer_is_reachable(self):
        client = EmbedClient(_make_settings())
        with patch("code_atlas.search.embeddings.litellm.encode", side_effect=Exception("not mapped")):
            first = client.count_tokens("x" * 300)
            assert first == 300 // CHARS_PER_TOKEN_FALLBACK + 1
            assert client._encode_ok is False
            # Remembered: the splitter measures a text many times on the way down.
            client.count_tokens("x" * 300)
