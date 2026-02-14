"""Daemon manager — reusable watcher + pipeline lifecycle.

Encapsulates the EventBus, FileWatcher, EmbedClient, EmbedCache,
and Tier 1/2/3 consumers.  Used by both the CLI (``atlas watch``,
``atlas daemon start``) and the MCP server for auto-indexing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from code_atlas.events import EventBus
from code_atlas.indexing.consumers import Tier1GraphConsumer, Tier2ASTConsumer, Tier3EmbedConsumer
from code_atlas.indexing.orchestrator import FileScope
from code_atlas.indexing.watcher import FileWatcher
from code_atlas.search.embeddings import EmbedCache, EmbedClient

if TYPE_CHECKING:
    from code_atlas.graph.client import GraphClient
    from code_atlas.settings import AtlasSettings


@dataclass
class DaemonManager:
    """Manages watcher + pipeline lifecycle.  Reusable across CLI and MCP."""

    _bus: EventBus | None = field(default=None, repr=False)
    _watcher: FileWatcher | None = field(default=None, repr=False)
    _consumers: list[Tier1GraphConsumer | Tier2ASTConsumer | Tier3EmbedConsumer] = field(
        default_factory=list, repr=False
    )
    _tasks: list[asyncio.Task[None]] = field(default_factory=list, repr=False)
    _cache: EmbedCache | None = field(default=None, repr=False)
    _embed: EmbedClient | None = field(default=None, repr=False)

    async def start(
        self,
        settings: AtlasSettings,
        graph: GraphClient,
        *,
        include_watcher: bool = True,
    ) -> bool:
        """Try to start watcher + pipeline.

        Returns ``False`` if Valkey is unreachable (graceful degradation).
        The caller-owned *graph* is shared — **not** closed by this manager.

        Parameters
        ----------
        settings:
            Full atlas settings (redis, embeddings, watcher, scope, …).
        graph:
            An already-connected :class:`GraphClient`.
        include_watcher:
            If ``False``, only start the tier consumers (no filesystem watcher).
            Useful for ``atlas daemon start`` which has no watch root.
        """
        bus = EventBus(settings.redis)
        try:
            await bus.ping()
        except Exception:
            logger.warning("Valkey unavailable — running without auto-indexing")
            await bus.close()
            return False

        self._bus = bus

        embed = EmbedClient(settings.embeddings)
        self._embed = embed

        cache: EmbedCache | None = None
        if settings.embeddings.cache_ttl_days > 0:
            cache = EmbedCache(settings.redis, settings.embeddings)
        self._cache = cache

        self._consumers = [
            Tier1GraphConsumer(bus, graph, settings),
            Tier2ASTConsumer(bus, graph, settings),
            Tier3EmbedConsumer(bus, graph, embed, cache=cache),
        ]

        if include_watcher:
            scope = FileScope(settings.project_root, settings)
            self._watcher = FileWatcher(settings.project_root, bus, scope, settings.watcher)

        # Spawn background tasks
        if self._watcher is not None:
            self._tasks.append(asyncio.get_running_loop().create_task(self._run_watcher()))
        for consumer in self._consumers:
            self._tasks.append(asyncio.get_running_loop().create_task(self._run_consumer(consumer)))

        return True

    async def wait(self) -> None:
        """Block until all background tasks finish (or are cancelled)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Graceful shutdown: stop watcher, consumers, close connections."""
        if self._watcher is not None:
            self._watcher.stop()

        for consumer in self._consumers:
            consumer.stop()

        # Cancel tasks and wait for them to finish
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._cache is not None:
            await self._cache.close()

        if self._bus is not None:
            await self._bus.close()

        logger.debug("DaemonManager stopped")

    async def _run_watcher(self) -> None:
        """Run the file watcher, logging crashes instead of propagating."""
        try:
            await self._watcher.run()  # type: ignore[union-attr]
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("File watcher crashed")

    @staticmethod
    async def _run_consumer(consumer: Tier1GraphConsumer | Tier2ASTConsumer | Tier3EmbedConsumer) -> None:
        """Run a consumer, catching exceptions so one failure doesn't crash the rest."""
        try:
            await consumer.run()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Consumer {} crashed", consumer.consumer_name)
