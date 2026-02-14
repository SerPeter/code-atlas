"""Indexing package — orchestrator, event consumers, watcher, and daemon."""

from __future__ import annotations

from code_atlas.indexing.consumers import (
    BatchPolicy,
    Tier1GraphConsumer,
    Tier2ASTConsumer,
    Tier3EmbedConsumer,
    TierConsumer,
)
from code_atlas.indexing.daemon import DaemonManager
from code_atlas.indexing.orchestrator import (
    DeltaStats,
    DetectedProject,
    FileScope,
    IndexResult,
    StalenessChecker,
    StalenessInfo,
    classify_file_project,
    detect_sub_projects,
    index_monorepo,
    index_project,
)
from code_atlas.indexing.watcher import FileWatcher

__all__ = [
    "BatchPolicy",
    "DaemonManager",
    "DeltaStats",
    "DetectedProject",
    "FileScope",
    "FileWatcher",
    "IndexResult",
    "StalenessChecker",
    "StalenessInfo",
    "Tier1GraphConsumer",
    "Tier2ASTConsumer",
    "Tier3EmbedConsumer",
    "TierConsumer",
    "classify_file_project",
    "detect_sub_projects",
    "index_monorepo",
    "index_project",
]
