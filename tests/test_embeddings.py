"""Tests for the embeddings module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from code_atlas.embeddings import EmbedClient, EmbeddingError, build_embed_text
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

    def test_model_string_no_base_url(self):
        client = EmbedClient(_make_settings(base_url=""))
        assert client._model == "nomic-ai/nomic-embed-code"
        assert client._api_base is None
        assert client._api_key is None

    async def test_embed_one(self):
        client = EmbedClient(_make_settings())
        fake_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1, 0.2, 0.3])])

        with patch("code_atlas.embeddings.litellm.aembedding", new_callable=AsyncMock, return_value=fake_response):
            result = await client.embed_one("hello")

        assert result == [0.1, 0.2, 0.3]

    async def test_embed_batch_single_chunk(self):
        client = EmbedClient(_make_settings(batch_size=32))
        texts = ["text1", "text2", "text3"]
        fake_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[float(i)]) for i in range(3)])

        with patch(
            "code_atlas.embeddings.litellm.aembedding", new_callable=AsyncMock, return_value=fake_response
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

        with patch("code_atlas.embeddings.litellm.aembedding", side_effect=fake_aembedding):
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
                "code_atlas.embeddings.litellm.aembedding",
                new_callable=AsyncMock,
                side_effect=Exception("Connection refused"),
            ),
            pytest.raises(EmbeddingError, match="Connection refused"),
        ):
            await client.embed_one("test")

    async def test_health_check_success(self):
        client = EmbedClient(_make_settings())
        fake_response = FakeEmbeddingResponse(data=[FakeEmbeddingItem(embedding=[0.1])])

        with patch("code_atlas.embeddings.litellm.aembedding", new_callable=AsyncMock, return_value=fake_response):
            assert await client.health_check() is True

    async def test_health_check_failure(self):
        client = EmbedClient(_make_settings())

        with patch(
            "code_atlas.embeddings.litellm.aembedding",
            new_callable=AsyncMock,
            side_effect=Exception("down"),
        ):
            assert await client.health_check() is False


# ---------------------------------------------------------------------------
# Model lock tests (integration-level, using graph_client fixture)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestModelLock:
    async def test_first_run_sets_config(self, graph_client):
        await graph_client.ensure_schema()

        result = await graph_client.get_embedding_config()
        assert result is None

        await graph_client.set_embedding_config("nomic-ai/nomic-embed-code", 768)
        result = await graph_client.get_embedding_config()
        assert result == ("nomic-ai/nomic-embed-code", 768)

    async def test_model_mismatch_detected(self, graph_client):
        await graph_client.ensure_schema()
        await graph_client.set_embedding_config("old-model", 768)

        stored = await graph_client.get_embedding_config()
        assert stored is not None
        stored_model, _ = stored
        assert stored_model == "old-model"
        assert stored_model != "new-model"

    async def test_clear_embeddings(self, graph_client):
        await graph_client.ensure_schema()
        # Create a test node with an embedding
        await graph_client.execute_write(
            "CREATE (n:Module {uid: 'test:mod', qualified_name: 'mod', project_name: 'test', embedding: [0.1, 0.2]})"
        )
        # Clear all embeddings
        await graph_client.clear_all_embeddings()
        records = await graph_client.execute("MATCH (n {uid: 'test:mod'}) RETURN n.embedding AS emb")
        assert records[0]["emb"] is None
