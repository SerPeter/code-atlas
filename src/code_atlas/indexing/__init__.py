"""Indexing package — orchestrator, event consumers, watcher, and daemon."""

from __future__ import annotations

from code_atlas.indexing.consumers import (
    ASTConsumer,
    BatchPolicy,
    EmbedConsumer,
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
    "ASTConsumer",
    "BatchPolicy",
    "DaemonManager",
    "DeltaStats",
    "DetectedProject",
    "EmbedConsumer",
    "FileScope",
    "FileWatcher",
    "IndexResult",
    "StalenessChecker",
    "StalenessInfo",
    "TierConsumer",
    "classify_file_project",
    "detect_sub_projects",
    "index_monorepo",
    "index_project",
]
