"""Embedding client and text builder for Code Atlas.

Uses litellm to route embedding requests to any OpenAI-compatible endpoint
(self-hosted TEI, OpenAI, Cohere, etc.) via a single code path.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import litellm
import pathspec
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from code_atlas.chunking import CHARS_PER_TOKEN_FALLBACK, SplitResult, split_embed_text
from code_atlas.search.ratelimit import ConcurrencyGate, RateLimiter, SqliteRateLimiter, make_rate_limiter
from code_atlas.settings import _PROVIDER_DEFAULTS
from code_atlas.telemetry import get_tracer

if TYPE_CHECKING:
    from code_atlas.settings import AtlasSettings, EmbeddingSettings

_tracer = get_tracer(__name__)

# Transient provider errors worth retrying — rate limits, connection issues,
# timeouts, 5xx. Deliberately excludes non-retryable errors (bad request, auth
# failure) so those still fail fast.
_RETRYABLE_ERRORS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
)


_TOKEN_CACHE_SIZE = 512
"""Texts whose token count is remembered.

Sized for one node's descent down the border ladder plus the batch around it, not for a
whole corpus: the reuse this exploits is within a single `split_text` call, where the same
chunk is measured to decide whether it fits, again to decide whether to keep descending,
and again in the final pass.
"""


class EmbeddingError(Exception):
    """Raised when an embedding operation fails."""


class EmbedClient:
    """Async embedding client backed by litellm.

    Routes to any OpenAI-compatible endpoint. When ``base_url`` is set
    (e.g. self-hosted TEI), the model is prefixed with ``openai/`` so
    litellm treats it as an OpenAI-compatible API.
    """

    def __init__(self, settings: EmbeddingSettings, atlas_settings: AtlasSettings | None = None) -> None:
        self._settings = settings
        # batch_size and max_concurrency are guaranteed non-None after the
        # _apply_provider_defaults model validator runs on EmbeddingSettings.
        if settings.batch_size is None or settings.max_concurrency is None:
            msg = "EmbeddingSettings.batch_size and .max_concurrency must be set (provider defaults should apply)"
            raise ValueError(msg)
        # The raw configured model, before the provider prefixing below. This is the
        # value the SchemaVersion lock stores and the value stamped onto vectors --
        # `self._model` is litellm-shaped ("openai/..." for TEI) and comparing the two
        # produces a mismatch that never resolves.
        self.configured_model: str = settings.model
        self._batch_size: int = settings.batch_size
        self._max_concurrency: int = settings.max_concurrency
        self._timeout = settings.timeout_s
        # Token counts for texts the splitter has already measured. Bounded: chunk
        # texts run to kilobytes, so an unbounded cache is a slow leak in a daemon that
        # indexes for hours.
        self._token_cache: OrderedDict[str, int] = OrderedDict()
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._query_cache_size = settings.query_cache_size

        # Compute the litellm model string based on provider
        if settings.provider == "tei":
            # Self-hosted TEI endpoint — prefix with openai/ for litellm compat
            model = settings.model
            if not model.startswith("openai/"):
                model = f"openai/{model}"
            self._model = model
            self._api_base = settings.base_url
            self._api_key = "unused"  # TEI ignores key, but OpenAI SDK requires one
        elif settings.provider == "ollama":
            # Local Ollama — use base_url but let litellm handle routing
            self._model = settings.model
            self._api_base = settings.base_url
            self._api_key = None
        else:
            # Cloud provider via litellm (openai/, gemini/, voyage/, cohere/, etc.)
            self._model = settings.model
            self._api_base = None
            self._api_key = None

        # Infer max input tokens from litellm's model registry
        self._max_input_tokens = self._resolve_max_input_tokens()
        # None = not yet tried; False = this model has no reachable tokenizer.
        self._encode_ok: bool | None = None

        # One gate for the whole client, not one per embed_batch call. A per-call
        # semaphore bounded the chunks of a single call while the embed consumer ran
        # max_concurrency of those calls at once, so the real ceiling was
        # max_concurrency squared -- 64 requests in flight for a configured 8.
        self._gate = ConcurrencyGate(self._max_concurrency)
        self._rpm = self._resolve_rate_limit(settings.rpm, "rpm")
        self._tpm = self._resolve_rate_limit(settings.tpm, "tpm")
        # Whole settings rather than `settings.redis`, because which coordination store
        # paces this client is a *backend* choice now, not a Redis detail. An embedded
        # deployment used to build a Valkey limiter regardless and pay a connect timeout
        # per batch for a host it had already been told was not there.
        self._limiter: RateLimiter | SqliteRateLimiter | None = None
        if atlas_settings is not None:
            self._limiter = make_rate_limiter(
                atlas_settings, model=self._model, rpm=self._rpm, tpm=self._tpm, gate=self._gate
            )
            logger.debug(
                "Embedding rate limits for '{}': rpm={} tpm={} (0 = unlimited)",
                self._model,
                self._rpm,
                self._tpm,
            )

    async def close(self) -> None:
        """Release the rate limiter's connection pool or SQLite handle.

        Only a client given *atlas_settings* has a limiter -- pass none and this is a
        no-op -- so every call site can close unconditionally.
        """
        if self._limiter is not None:
            await self._limiter.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def batch_size(self) -> int:
        """Resolved batch_size (after provider defaults)."""
        return self._batch_size

    @property
    def max_concurrency(self) -> int:
        """Resolved max_concurrency (after provider defaults)."""
        return self._max_concurrency

    def _resolve_rate_limit(self, configured: int | None, field: str) -> int:
        """Resolve a per-minute budget: explicit config, then litellm's registry, then
        the provider default. 0 means unlimited at every level.

        The registry sits in the middle because it is authoritative when it has an
        answer and absent otherwise -- it publishes rpm/tpm for 4 of its 134 embedding
        models. A model it does not know falls through to the provider default rather
        than to a guess, and ``get_model_info`` raises outright for unmapped models, so
        the lookup is exception-guarded exactly like _resolve_max_input_tokens below.
        """
        if configured is not None:
            return configured
        try:
            value = litellm.get_model_info(self._model).get(field)
        except Exception:
            value = None
        if value:
            return int(value)
        defaults = _PROVIDER_DEFAULTS.get(self._settings.provider, _PROVIDER_DEFAULTS["tei"])
        return defaults.get(field, 0)

    def _resolve_max_input_tokens(self) -> int | None:
        """Resolve the model's max input token limit, registry or explicit override.

        Applies ``truncate_ratio`` as a safety margin (e.g. 0.9 * 8192 = 7372).

        The configured override wins because the registry's answer for a routed model
        is *no answer*: ``openrouter/google/gemini-embedding-001`` and
        ``openai/nomic-ai/nomic-embed-code`` both raise "isn't mapped yet", which used
        to mean no cap at all -- so nothing chunked, nothing truncated, and a single
        over-length node took down the whole 128-text provider call it was batched into.
        """
        if self._settings.max_input_tokens is not None:
            return int(self._settings.max_input_tokens * self._settings.truncate_ratio)
        try:
            info = litellm.get_model_info(self._model)
            limit = info.get("max_input_tokens")
            if limit and limit > 0:
                effective = int(limit * self._settings.truncate_ratio)
                logger.debug("Embedding model max input tokens: {} (effective: {})", limit, effective)
                return effective
        except Exception:
            self._warn_unknown_cap()
            return None
        self._warn_unknown_cap()
        return None

    def _warn_unknown_cap(self) -> None:
        """Say out loud that neither chunking nor truncation will happen.

        At warning level, not debug, because the consequence grew: entity source is
        capped at index.max_source_chars, which is 48,000 characters — roughly 12,000
        tokens. With no cap resolved, a text that size is sent whole, and one
        over-length text fails the entire provider call it was batched into, taking
        127 unrelated texts down with it. This used to be a debug line back when the
        source cap made every code entity ~500 tokens and nothing could exceed a limit.
        """
        logger.warning(
            "No input-token cap for embedding model '{}' — litellm's registry does not know it, "
            "so neither chunking nor truncation will run. Set [embeddings] max_input_tokens "
            "(gemini-embedding-001 is 2048, text-embedding-3-small is 8191).",
            self._model,
        )

    def count_tokens(self, text: str) -> int:
        """Token count for *text* under this model's tokenizer, memoised.

        Falls back to :data:`CHARS_PER_TOKEN_FALLBACK` when the model has no
        tokenizer litellm can reach. The first failure is remembered -- ``encode``
        raises per call for an unmapped model, and the splitter measures a text many
        times on its way down the border ladder.

        **Memoised because the splitter asks the same question repeatedly.** Descending
        the ladder, `split_embed_text` measures every chunk to decide whether it fits,
        measures them all again to decide whether to stop descending, and measures them
        once more in the final pass. Identical strings, three encodes. The cache turns
        the repeats into dictionary hits, and it is bounded because chunk texts are large
        enough that an unbounded one would be a slow leak in a long-lived daemon.

        Keyed on the text alone: the model is fixed for the life of the client, and a
        client for another model has its own cache.
        """
        cached = self._token_cache.get(text)
        if cached is not None:
            self._token_cache.move_to_end(text)
            return cached

        if self._encode_ok is not False:
            try:
                count = len(litellm.encode(model=self._model, text=text))
            except Exception:
                self._encode_ok = False
            else:
                self._encode_ok = True
                self._remember_tokens(text, count)
                return count
        fallback = len(text) // CHARS_PER_TOKEN_FALLBACK + 1
        self._remember_tokens(text, fallback)
        return fallback

    def _remember_tokens(self, text: str, count: int) -> None:
        self._token_cache[text] = count
        if len(self._token_cache) > _TOKEN_CACHE_SIZE:
            self._token_cache.popitem(last=False)

    def split_text(self, text: str) -> SplitResult:
        """Split *text* into chunks this model will accept.

        A text already under the cap comes back as a single chunk, which is the case
        for all but a fraction of a percent of nodes. When the cap is unknown the text
        is returned whole -- see ``EmbeddingSettings.max_input_tokens``.
        """
        return split_embed_text(
            text,
            limit=self._max_input_tokens or 0,
            measure=self.count_tokens,
            max_chunks=self._settings.max_chunks,
        )

    def _truncate_texts(self, texts: list[str]) -> tuple[list[str], list[int]]:
        """Truncate texts over the model's input limit; return them with token counts.

        The counts are a by-product: every text is encoded here anyway to decide whether
        it needs truncating, so exact accounting for the tokens-per-minute budget costs
        nothing. When the model's limit is unknown nothing is encoded, and the counts
        fall back to the same ~4-chars-per-token approximation the truncation path uses.
        """
        if self._max_input_tokens is None:
            return texts, [len(t) // 4 for t in texts]
        limit = self._max_input_tokens
        result: list[str] = []
        counts: list[int] = []
        for text in texts:
            try:
                tokens = litellm.encode(model=self._model, text=text)
            except Exception:
                result.append(text)
                counts.append(len(text) // 4)
                continue
            if len(tokens) <= limit:
                result.append(text)
                counts.append(len(tokens))
            else:
                try:
                    truncated = litellm.decode(model=self._model, tokens=tokens[:limit])
                except Exception:
                    truncated = text[: limit * 4]  # ~4 chars/token fallback
                last_nl = truncated.rfind("\n")
                if last_nl > 0:
                    truncated = truncated[:last_nl]
                logger.warning("Truncated embed text from {} to ~{} tokens", len(tokens), limit)
                result.append(truncated)
                counts.append(limit)
        return result, counts

    @retry(
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        before_sleep=lambda rs: logger.warning(
            "Embedding call transient error, retrying in {:.1f}s (attempt {}): {}",
            rs.next_action.sleep,  # ty: ignore[unresolved-attribute]  # tenacity's RetryCallState is loosely typed in its stubs
            rs.attempt_number,
            rs.outcome.exception(),  # ty: ignore[unresolved-attribute]  # tenacity's RetryCallState is loosely typed in its stubs
        ),
        reraise=True,
    )
    async def _embed_call(self, kwargs: dict[str, Any]) -> Any:
        """``litellm.aembedding`` with retry on transient provider errors (rate limits,
        connection issues, timeouts, 5xx) — a burst of embedding calls during a large
        index/catchup is exactly the pattern that trips cloud provider rate limits.
        """
        try:
            return await litellm.aembedding(**kwargs)
        except litellm.RateLimitError:
            # Tell every process, not just this one: the scale factor lives in Valkey, so
            # a single 429 damps the whole fleet's buckets and concurrency before the
            # retry above sleeps. Without it the sibling processes keep pushing at the
            # rate that just failed, and the backoff here is wasted.
            if self._limiter is not None:
                await self._limiter.penalize()
            raise

    def _build_kwargs(self, chunk: list[str]) -> dict[str, Any]:
        """Build keyword arguments for a single litellm.aembedding call."""
        kwargs: dict[str, Any] = {
            "model": self._model,
            "input": chunk,
            "timeout": self._timeout,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
            # TEI rejects encoding_format=null; explicitly request floats
            kwargs["encoding_format"] = "float"
        if self._api_key:
            kwargs["api_key"] = self._api_key
        # Request provider-side dimension reduction (e.g. OpenAI/Gemini MRL
        # truncation) so the returned vector matches the configured Memgraph
        # index size. Only meaningful for litellm-routed cloud providers —
        # TEI/Ollama models have a fixed dimension set by the hosted model.
        if self._settings.provider == "litellm" and self._settings.dimension is not None:
            kwargs["dimensions"] = self._settings.dimension
        return kwargs

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, chunking by ``batch_size``.

        Fires chunks concurrently (up to ``max_concurrency``) for throughput.
        Returns a flat list of vectors in the same order as *texts*.
        Raises ``EmbeddingError`` on failure.
        """
        if not texts:
            return []

        texts, token_counts = self._truncate_texts(texts)

        with _tracer.start_as_current_span(
            "embed.embed_batch", attributes={"batch_size": len(texts), "model": self._model}
        ):

            async def _do_chunk(chunk_idx: int, chunk: list[str], chunk_tokens: int) -> tuple[int, list[list[float]]]:
                # Gate first, budget second. The gate bounds sockets and costs nothing to
                # hold; taking rate budget first would debit it and then block on the gate,
                # leaving the scarcer resource reserved and unused.
                async with self._gate:
                    if self._limiter is not None:
                        await self._limiter.acquire(tokens=chunk_tokens)
                    try:
                        kwargs = self._build_kwargs(chunk)
                        response = await self._embed_call(kwargs)
                    except Exception as exc:
                        offset = chunk_idx * self._batch_size
                        msg = f"Embedding failed for batch [{offset}:{offset + len(chunk)}]: {exc}"
                        logger.error(msg)
                        raise EmbeddingError(msg) from exc
                    else:
                        vectors = [
                            item["embedding"] if isinstance(item, dict) else item.embedding for item in response.data
                        ]
                        return chunk_idx, vectors

            tasks = [
                _do_chunk(
                    i,
                    texts[i * self._batch_size : (i + 1) * self._batch_size],
                    sum(token_counts[i * self._batch_size : (i + 1) * self._batch_size]),
                )
                for i in range((len(texts) + self._batch_size - 1) // self._batch_size)
            ]
            results = await asyncio.gather(*tasks)

            all_vectors: list[list[float]] = []
            for _, vecs in sorted(results):
                all_vectors.extend(vecs)
            return all_vectors

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text with LRU caching for repeated queries."""
        with _tracer.start_as_current_span("embed.embed_one") as span:
            if text in self._query_cache:
                self._query_cache.move_to_end(text)
                span.set_attribute("cache_hit", True)
                return self._query_cache[text]

            span.set_attribute("cache_hit", False)
            result = await self.embed_batch([text])
            vector = result[0]

            self._query_cache[text] = vector
            if len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)

            return vector

    async def detect_dimension(self) -> int:
        """Probe the embedding service and return the vector dimension."""
        vectors = await self.embed_batch(["dimension probe"])
        return len(vectors[0])

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
# Embedding policy
# ---------------------------------------------------------------------------

DEFAULT_EXCLUDE_KINDS: tuple[str, ...] = ("config_setting", "config_section", "xml_setting", "xml_element")
"""Kinds that carry no vector unless the user asks for one.

These two are emitted by exactly one code path -- ``config.py``'s generic structural
fallback (``_GENERIC_SECTION_KIND`` / ``_GENERIC_SETTING_KIND``), the handler that runs
when no dialect recognised the file. So "structured data nobody could make sense of" is
not a path pattern to be guessed at; it is a kind, and excluding the kind needs no
per-repo tuning. A file a dialect *did* recognise gets ``k8s_resource``,
``compose_service``, ``ci_job``, ``dbt_source`` and so on, and keeps its vector.

``xml_element`` and ``xml_setting`` are the XML branch's twins of those two and are
excluded for the same reason. They were held back while ATL-144 was pending, on the
grounds that excluding them would pre-empt a story trying to extract *more* from
Salesforce metadata. ATL-144 settled it the other way: a recognised type gets a
Salesforce kind (``sf_flow``, ``sobject``, ``sobject_field``, ...) and keeps its vector,
so these two now mean precisely "XML no handler recognised". One real permission set
contributed 1,944 of them, all named ``fieldPermissions``, all with an empty ``source``,
and every one was embedded.

``permission_set`` and ``profile`` were considered and NOT excluded. The volume argument
that motivated it is already answered -- ATL-181 made one 427 KB permission set a single
node instead of 1,947 -- and the node that remains carries its ``label`` and
``description`` as its docstring, which is real prose an admin wrote. Its grants live in
properties and edges, not in its text, so there is nothing table-shaped to exclude. The
completeness test below is what caught this: every kind excluded here must come from the
structural fallback, and a kind a handler mints deliberately does not qualify.

``config_file`` and ``xml_document`` are deliberately absent: the file-level node is named
after the file and answers "what is this config for", which is a fair semantic target. It
is one node per file, so it is not what floods anything.

Excluded is not invisible -- the node keeps its name, its edges and its FTS document, and
``_floor_excluded_in_vector_channel`` admits it at the tail of the vector list wherever a
query gates on vector similarity (ADR-0047).
"""


@dataclass(frozen=True)
class EmbedPolicy:
    """Which entities are allowed to carry a vector.

    Recomputed from settings wherever it is needed rather than stamped onto nodes. A node
    property would mean a ``SCHEMA_VERSION`` bump, and a bump drops the vector indices
    unconditionally while recreating them only conditionally (ADR-0024). It would also
    freeze the policy at index time, so changing it would need a reindex -- where this
    takes effect on the next pass, and at query time immediately.

    Cheap by construction: one set lookup and at most two pathspec matches, on inputs the
    caller already holds.
    """

    exclude_kinds: frozenset[str]
    exclude_paths: pathspec.PathSpec | None
    include_paths: pathspec.PathSpec | None

    @classmethod
    def from_settings(cls, settings: EmbeddingSettings) -> EmbedPolicy:
        """Compile *settings* into a policy."""
        kinds = settings.exclude_kinds if settings.exclude_kinds is not None else list(DEFAULT_EXCLUDE_KINDS)
        return cls(
            exclude_kinds=frozenset(kinds),
            exclude_paths=pathspec.PathSpec.from_lines("gitignore", settings.exclude) if settings.exclude else None,
            include_paths=pathspec.PathSpec.from_lines("gitignore", settings.include) if settings.include else None,
        )

    @property
    def is_permissive(self) -> bool:
        """True when nothing is excluded, so callers can skip the check entirely."""
        return not self.exclude_kinds and self.exclude_paths is None

    def allows(self, kind: str, file_path: str) -> bool:
        """Whether an entity of *kind* at *file_path* may be embedded.

        ``include`` is checked first and wins outright: the default is broad on purpose,
        and one line should be able to bring a directory back without having to restate
        the kind list it collides with.
        """
        if self.include_paths is not None and file_path and self.include_paths.match_file(file_path):
            return True
        if kind in self.exclude_kinds:
            return False
        return not (self.exclude_paths is not None and file_path and self.exclude_paths.match_file(file_path))


# ---------------------------------------------------------------------------
# Embed text builder
# ---------------------------------------------------------------------------

# Node labels that represent code entities (used for template selection)
_CODE_ENTITY_LABELS = frozenset({"Callable", "TypeDef", "Value", "Module"})


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of *text*.

    Lives next to :func:`build_embed_text` because the pair is a contract, not two
    utilities: a node's ``embed_hash`` is exactly ``hash_text(build_embed_text(props))``,
    and the embed stage's short-circuit compares against it. It was a static method on
    the Valkey cache until ATL-127 deleted that class; the hash outlived the cache
    because the graph is what stores it.
    """
    return hashlib.sha256(text.encode()).hexdigest()


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
    source = props.get("source", "")

    if not qualified_name:
        return ""

    if label == "Note":
        return _build_note_text(props.get("name", ""), props.get("tags") or [], docstring)
    if label == "DocSection":
        return _build_doc_section_text(qualified_name, docstring)
    if label in _CODE_ENTITY_LABELS:
        return _build_code_entity_text(label, kind, qualified_name, signature, docstring, source)

    # Fallback for unknown labels: just use qualified_name + docstring
    parts = [qualified_name]
    if docstring:
        parts.append(docstring)
    return "\n".join(parts)


def _build_code_entity_text(
    label: str, kind: str, qualified_name: str, signature: str, docstring: str, source: str = ""
) -> str:
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

    if source:
        lines.append("")
        lines.append(source)

    return "\n".join(lines)


def _build_doc_section_text(qualified_name: str, docstring: str) -> str:
    """Build embed text for DocSection nodes.

    The ``qualified_name`` encodes the header breadcrumb
    (e.g. ``wiki/architecture.md > Architecture > Event Pipeline > AST Stage``).
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


def _build_note_text(name: str, tags: list[str], docstring: str) -> str:
    """Build embed text for Note nodes (title + tags + full body)."""
    lines: list[str] = []
    if name:
        lines.append(f"Note: {name}")
    if tags:
        lines.append(f"Tags: {', '.join(tags)}")
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
