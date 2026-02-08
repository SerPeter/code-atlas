"""Embedding client and text builder for Code Atlas.

Uses litellm to route embedding requests to any OpenAI-compatible endpoint
(self-hosted TEI, OpenAI, Cohere, etc.) via a single code path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import litellm
from loguru import logger

if TYPE_CHECKING:
    from code_atlas.settings import EmbeddingSettings


class EmbeddingError(Exception):
    """Raised when an embedding operation fails."""


class EmbedClient:
    """Async embedding client backed by litellm.

    Routes to any OpenAI-compatible endpoint. When ``base_url`` is set
    (e.g. self-hosted TEI), the model is prefixed with ``openai/`` so
    litellm treats it as an OpenAI-compatible API.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        self._settings = settings
        self._batch_size = settings.batch_size
        self._timeout = settings.timeout_s

        # Compute the litellm model string
        if settings.base_url:
            # Self-hosted endpoint — prefix with openai/ unless already prefixed
            model = settings.model
            if not model.startswith("openai/"):
                model = f"openai/{model}"
            self._model = model
            self._api_base = settings.base_url
            self._api_key = "unused"  # TEI ignores key, but OpenAI SDK requires one
        else:
            # Cloud provider — litellm routes by prefix (e.g. "text-embedding-3-small")
            self._model = settings.model
            self._api_base = None
            self._api_key = None

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, chunking by ``batch_size``.

        Returns a flat list of vectors in the same order as *texts*.
        Raises ``EmbeddingError`` on failure.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "input": chunk,
                    "timeout": self._timeout,
                }
                if self._api_base:
                    kwargs["api_base"] = self._api_base
                if self._api_key:
                    kwargs["api_key"] = self._api_key

                response = await litellm.aembedding(**kwargs)
                vectors = [item.embedding for item in response.data]
                all_vectors.extend(vectors)
            except Exception as exc:
                msg = f"Embedding failed for batch [{i}:{i + len(chunk)}]: {exc}"
                logger.error(msg)
                raise EmbeddingError(msg) from exc

        return all_vectors

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text. Convenience wrapper around :meth:`embed_batch`."""
        result = await self.embed_batch([text])
        return result[0]

    async def health_check(self) -> bool:
        """Check if the embedding service is reachable.

        For endpoints with ``base_url``, tries a small embedding call.
        Returns True if successful, False otherwise.
        """
        try:
            await self.embed_one("health check")
        except EmbeddingError:
            return False
        else:
            return True


# ---------------------------------------------------------------------------
# Embed text builder
# ---------------------------------------------------------------------------

# Node labels that represent code entities (used for template selection)
_CODE_ENTITY_LABELS = frozenset({"Callable", "TypeDef", "Value", "Module"})


def build_embed_text(props: dict[str, Any]) -> str:
    """Build embeddable text from graph node properties.

    Enriches with hierarchical context derived from ``qualified_name``
    (e.g. ``myapp.parser.Parser.process`` → Module / Class / Method).

    Args:
        props: Node properties dict with keys like ``qualified_name``,
               ``signature``, ``docstring``, ``kind``, ``_label``.

    Returns:
        A text string suitable for embedding. Returns empty string if
        the node has insufficient data.
    """
    label = props.get("_label", "")
    qualified_name = props.get("qualified_name", "")
    kind = props.get("kind", "")
    signature = props.get("signature", "")
    docstring = props.get("docstring", "")

    if not qualified_name:
        return ""

    if label == "DocSection":
        return _build_doc_section_text(qualified_name, docstring)
    if label in _CODE_ENTITY_LABELS:
        return _build_code_entity_text(label, kind, qualified_name, signature, docstring)

    # Fallback for unknown labels: just use qualified_name + docstring
    parts = [qualified_name]
    if docstring:
        parts.append(docstring)
    return "\n".join(parts)


def _build_code_entity_text(label: str, kind: str, qualified_name: str, signature: str, docstring: str) -> str:
    """Build embed text for code entities (Callable, TypeDef, Value, Module)."""
    parts = qualified_name.split(".")
    lines: list[str] = []

    if label == "Module":
        lines.append(f"Module: {qualified_name}")
        if docstring:
            lines.append(f'"""{docstring}"""')
        return "\n".join(lines)

    # Reconstruct hierarchy from qualified_name parts
    # e.g. myapp.parser.Parser.process → Module: myapp.parser, Class: Parser, Method: process
    if len(parts) >= 2:
        # Module is everything up to the entity (or its class)
        _method_kinds = ("method", "constructor", "destructor", "static_method", "class_method", "property")
        if label == "Callable" and kind in _method_kinds:
            # Method — parent is a class, grandparent is the module
            if len(parts) >= 3:
                module_name = ".".join(parts[:-2])
                class_name = parts[-2]
                lines.append(f"Module: {module_name}")
                lines.append(f"Class: {class_name}")
            else:
                module_name = ".".join(parts[:-1])
                lines.append(f"Module: {module_name}")
        elif label == "TypeDef":
            module_name = ".".join(parts[:-1])
            lines.append(f"Module: {module_name}")
        else:
            # Top-level function, value, etc.
            module_name = ".".join(parts[:-1])
            lines.append(f"Module: {module_name}")

    # Entity line with kind label
    display_kind = _kind_display(label, kind)
    if signature:
        lines.append(f"{display_kind}: {signature}")
    else:
        lines.append(f"{display_kind}: {parts[-1] if parts else qualified_name}")

    if docstring:
        lines.append(f'"""{docstring}"""')

    return "\n".join(lines)


def _build_doc_section_text(qualified_name: str, docstring: str) -> str:
    """Build embed text for DocSection nodes.

    The ``qualified_name`` encodes the header breadcrumb
    (e.g. ``docs/architecture.md > Architecture > Event Pipeline > Tier 2``).
    """
    # Split on " > " to get file path and section headers
    breadcrumb_parts = qualified_name.split(" > ")
    lines: list[str] = []

    if breadcrumb_parts:
        lines.append(f"File: {breadcrumb_parts[0]}")
        if len(breadcrumb_parts) > 1:
            lines.append(f"Section: {' > '.join(breadcrumb_parts[1:])}")

    if docstring:
        lines.append(f'"""{docstring}"""')

    return "\n".join(lines)


def _kind_display(label: str, kind: str) -> str:
    """Map label+kind to a human-readable display name for the embed text."""
    if label == "Callable":
        return {
            "function": "Function",
            "method": "Method",
            "constructor": "Constructor",
            "destructor": "Destructor",
            "static_method": "StaticMethod",
            "class_method": "ClassMethod",
            "property": "Property",
            "closure": "Closure",
        }.get(kind, "Function")
    if label == "TypeDef":
        return {
            "class": "Class",
            "struct": "Struct",
            "interface": "Interface",
            "trait": "Trait",
            "enum": "Enum",
            "union": "Union",
            "type_alias": "TypeAlias",
            "protocol": "Protocol",
            "record": "Record",
            "data_type": "DataType",
            "typeclass": "Typeclass",
            "annotation": "Annotation",
        }.get(kind, "Class")
    if label == "Value":
        return {
            "variable": "Variable",
            "constant": "Constant",
            "field": "Field",
            "enum_member": "EnumMember",
        }.get(kind, "Value")
    return label
