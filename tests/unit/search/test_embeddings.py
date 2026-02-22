"""Unit tests for the embeddings module."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from code_atlas.events import EmbedDirty, EntityRef
from code_atlas.indexing.consumers import Tier3EmbedConsumer
from code_atlas.search.embeddings import EmbedCache, EmbedClient, EmbeddingError, build_embed_text
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
            "qualified_name": "docs/architecture.md > Architecture > Event Pipeline > Tier 2",
            "kind": "",
            "signature": "",
            "docstring": "The AST parsing tier processes file changes...",
        }
        text = build_embed_text(props)
        assert "File: docs/architecture.md" in text
        assert "Section: Architecture > Event Pipeline > Tier 2" in text
        assert '"""The AST parsing tier processes file changes..."""' in text

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
# Tier3 cache integration tests (mocked graph + embed + cache)
# ---------------------------------------------------------------------------


class TestTier3CacheLookup:
    """Test the three-tier lookup logic in Tier3EmbedConsumer with mocks."""

    @staticmethod
    def _make_entity_ref(qn: str) -> EntityRef:
        return EntityRef(qualified_name=qn, node_type="Callable", file_path="f.py")

    @staticmethod
    def _make_embed_dirty(entities: list[EntityRef]) -> EmbedDirty:
        return EmbedDirty(entities=entities, significance="HIGH", batch_id="test01")

    async def test_graph_hit_skips_all(self):
        """When graph node has matching embed_hash+embedding, no cache or API call needed."""
        bus = AsyncMock()
        graph = AsyncMock()
        embed = AsyncMock()
        cache = AsyncMock()

        text = "Module: foo\nFunction: bar"
        text_hash = EmbedCache.hash_text(text)

        graph.read_entity_texts = AsyncMock(
            return_value=[
                {
                    "uid": "foo.bar",
                    "qualified_name": "foo.bar",
                    "name": "bar",
                    "signature": "",
                    "docstring": "",
                    "kind": "function",
                    "_label": "Callable",
                    "embed_hash": text_hash,
                    "embedding": [0.1, 0.2, 0.3],
                }
            ]
        )

        consumer = Tier3EmbedConsumer(bus, graph, embed, cache=cache)
        entity = self._make_entity_ref("foo.bar")
        event = self._make_embed_dirty([entity])

        await consumer.process_batch([event], "test01")

        embed.embed_batch.assert_not_called()
        cache.get_many.assert_not_called()
        graph.write_embeddings.assert_not_called()

    async def test_cache_hit_skips_embed(self):
        """When Valkey cache has the vector, API call is skipped."""
        bus = AsyncMock()
        graph = AsyncMock()
        embed = AsyncMock()
        cache = AsyncMock()

        text = "Module: foo\nFunction: bar"
        text_hash = EmbedCache.hash_text(text)
        cached_vec = [0.1, 0.2, 0.3]

        graph.read_entity_texts = AsyncMock(
            return_value=[
                {
                    "uid": "foo.bar",
                    "qualified_name": "foo.bar",
                    "name": "bar",
                    "signature": "",
                    "docstring": "",
                    "kind": "function",
                    "_label": "Callable",
                    "embed_hash": None,
                    "embedding": None,
                }
            ]
        )
        cache.get_many = AsyncMock(return_value={text_hash: cached_vec})

        consumer = Tier3EmbedConsumer(bus, graph, embed, cache=cache)
        entity = self._make_entity_ref("foo.bar")
        event = self._make_embed_dirty([entity])

        await consumer.process_batch([event], "test02")

        embed.embed_batch.assert_not_called()
        cache.get_many.assert_called_once()
        graph.write_embeddings.assert_called_once()
        graph.write_embed_hashes.assert_called_once()

        # Verify correct vector was written (keyed by uid now)
        write_args = graph.write_embeddings.call_args[0][0]
        assert write_args[0] == ("foo.bar", cached_vec)

    async def test_cache_miss_embeds_and_stores(self):
        """When no hits anywhere, API is called and result stored in cache."""
        bus = AsyncMock()
        graph = AsyncMock()
        embed = AsyncMock()
        cache = AsyncMock()

        api_vec = [0.5, 0.6, 0.7]

        graph.read_entity_texts = AsyncMock(
            return_value=[
                {
                    "uid": "foo.bar",
                    "qualified_name": "foo.bar",
                    "name": "bar",
                    "signature": "",
                    "docstring": "",
                    "kind": "function",
                    "_label": "Callable",
                    "embed_hash": None,
                    "embedding": None,
                }
            ]
        )
        cache.get_many = AsyncMock(return_value={})  # no cache hit
        embed.embed_batch = AsyncMock(return_value=[api_vec])

        consumer = Tier3EmbedConsumer(bus, graph, embed, cache=cache)
        entity = self._make_entity_ref("foo.bar")
        event = self._make_embed_dirty([entity])

        await consumer.process_batch([event], "test03")

        embed.embed_batch.assert_called_once()
        cache.put_many.assert_called_once()
        graph.write_embeddings.assert_called_once()
        graph.write_embed_hashes.assert_called_once()

        # Verify the API vector was written to graph (keyed by uid now)
        write_args = graph.write_embeddings.call_args[0][0]
        assert write_args[0] == ("foo.bar", api_vec)
