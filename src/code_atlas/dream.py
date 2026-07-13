"""Deterministic dream-mode report — the lint half of knowledge consolidation.

Computes inbox digest, orphan notes, dangling links, duplicate-id conflicts,
and cross-note similarity (including cross-project promotion candidates)
across every configured vault (repo ``docs/`` + any ``[knowledge] extra_vaults``),
regardless of which project a Note lives in. The disposition step
(KEEP/MERGE/PROMOTE/DROP) is agent-side — see the ``dream-mode`` command —
this module only produces the deterministic inputs it consumes.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from code_atlas.parsing.ast import parse_file
from code_atlas.schema import NodeLabel, NoteKind, RelType

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph.client import GraphClient

# LINKS_TO/DERIVED_FROM/SUPERSEDES targets are deterministic note uids (see
# markdown.py's _resolve_note_ref) — an exact-uid miss means the link is
# genuinely dangling. DOCUMENTS targets resolve heuristically by name instead,
# so an unresolved one is an expected heuristic-miss, not a lint finding.
_LINK_REL_TYPES = frozenset({RelType.LINKS_TO, RelType.DERIVED_FROM, RelType.SUPERSEDES})

_DEFAULT_SIMILARITY_THRESHOLD = 0.92


@dataclass(frozen=True)
class VaultRoot:
    """A vault directory to scan for filesystem-only lint checks."""

    path: Path
    project_name: str


@dataclass(frozen=True)
class DuplicateIdConflict:
    """Two or more files in the same vault resolve to the same note uid."""

    qualified_name: str
    project_name: str
    file_paths: list[str]


@dataclass(frozen=True)
class DanglingLink:
    """A LINKS_TO/DERIVED_FROM/SUPERSEDES reference whose target doesn't exist."""

    from_uid: str
    rel_type: str
    target_uid: str


@dataclass(frozen=True)
class OrphanNote:
    """A Note with no LINKS_TO edges in or out — disconnected from the note graph."""

    uid: str
    name: str
    project_name: str
    file_path: str


@dataclass(frozen=True)
class SimilarPair:
    """Two notes whose embeddings are highly similar — merge/dup candidates."""

    uid_a: str
    uid_b: str
    project_a: str
    project_b: str
    similarity: float


@dataclass(frozen=True)
class DreamReport:
    """The full deterministic dream-mode lint report."""

    inbox_count: int
    inbox_paths: list[str]
    orphan_notes: list[OrphanNote]
    duplicate_ids: list[DuplicateIdConflict]
    dangling_links: list[DanglingLink]
    similar_pairs: list[SimilarPair]
    promotion_candidates: list[SimilarPair]
    memory_index_issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Filesystem scan: duplicate ids + link targets (reuses the real parser so
# resolution logic never drifts from what indexing actually does)
# ---------------------------------------------------------------------------


def _scan_vault_for_notes(vault: VaultRoot) -> tuple[dict[str, list[str]], list[tuple[str, str, str]]]:
    """Walk *vault*, parsing every markdown file.

    Returns ``(qualified_name -> file_paths, link_targets)`` where
    ``link_targets`` is ``(from_uid, rel_type_value, target_uid)`` for
    LINKS_TO/DERIVED_FROM/SUPERSEDES relationships only.
    """
    by_qn: dict[str, list[str]] = {}
    links: list[tuple[str, str, str]] = []
    if not vault.path.is_dir():
        return by_qn, links

    for md_file in sorted(vault.path.rglob("*.md")):
        rel_path = md_file.relative_to(vault.path).as_posix()
        try:
            source = md_file.read_bytes()
        except OSError:
            logger.warning("dream: cannot read {}", md_file)
            continue
        parsed = parse_file(rel_path, source, vault.project_name)
        if parsed is None:
            continue
        for entity in parsed.entities:
            if entity.label != NodeLabel.NOTE:
                continue
            by_qn.setdefault(entity.qualified_name, []).append(rel_path)
        links.extend(
            (rel.from_qualified_name, rel.rel_type.value, rel.to_name)
            for rel in parsed.relationships
            if rel.rel_type in _LINK_REL_TYPES
        )
    return by_qn, links


_MEMORY_LINK_RE = re.compile(r"\]\(([\w.\-/]+\.md)\)")


def _check_memory_index(vault: VaultRoot) -> list[str]:
    """Best-effort: compare a vault's MEMORY.md links against files on disk.

    Only meaningful for a harness memory-dir vault (has a MEMORY.md index) —
    silently returns nothing for vaults that don't have one.
    """
    memory_md = vault.path / "MEMORY.md"
    if not memory_md.is_file():
        return []
    try:
        content = memory_md.read_text(encoding="utf-8")
    except OSError:
        return []

    referenced = set(_MEMORY_LINK_RE.findall(content))
    on_disk = {p.name for p in vault.path.glob("*.md") if p.name != "MEMORY.md"}

    issues: list[str] = []
    missing_on_disk = sorted(referenced - on_disk)
    missing_from_index = sorted(on_disk - referenced)
    if missing_on_disk:
        issues.append(f"{vault.project_name}: MEMORY.md references missing file(s): {missing_on_disk}")
    if missing_from_index:
        issues.append(f"{vault.project_name}: file(s) on disk not listed in MEMORY.md: {missing_from_index}")
    return issues


# ---------------------------------------------------------------------------
# Graph-based checks
# ---------------------------------------------------------------------------


async def _find_dangling_links(graph: GraphClient, links: list[tuple[str, str, str]]) -> list[DanglingLink]:
    if not links:
        return []
    target_uids = sorted({to_uid for _, _, to_uid in links})
    rows = await graph.execute("UNWIND $uids AS uid MATCH (n {uid: uid}) RETURN uid", {"uids": target_uids})
    existing = {r["uid"] for r in rows}
    return [
        DanglingLink(from_uid=from_uid, rel_type=rel_type, target_uid=to_uid)
        for from_uid, rel_type, to_uid in links
        if to_uid not in existing
    ]


async def _find_orphan_notes(graph: GraphClient) -> list[OrphanNote]:
    rows = await graph.execute(
        f"MATCH (n:{NodeLabel.NOTE}) WHERE NOT (n)-[:{RelType.LINKS_TO}]-() "
        "RETURN n.uid AS uid, n.name AS name, n.project_name AS project_name, n.file_path AS file_path"
    )
    return [OrphanNote(**row) for row in rows]


async def _find_inbox_notes(graph: GraphClient) -> tuple[int, list[str]]:
    rows = await graph.execute(
        f"MATCH (n:{NodeLabel.NOTE}) WHERE n.kind = $draft OR n.file_path CONTAINS '/inbox/' "
        "RETURN n.file_path AS file_path ORDER BY file_path",
        {"draft": NoteKind.DRAFT.value},
    )
    paths = [r["file_path"] for r in rows]
    return len(paths), paths


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _find_similar_pairs(graph: GraphClient, threshold: float) -> list[SimilarPair]:
    """All-pairs cosine similarity over Note embeddings.

    O(N^2) in note count — acceptable for a periodic lint report over a
    knowledge vault (not a hot query path); Memgraph's vector_search is a
    KNN-for-one-query-vector primitive, not an all-pairs one, so pulling
    every embedding once and comparing in Python is the simpler v1 approach.
    """
    rows = await graph.execute(
        f"MATCH (n:{NodeLabel.NOTE}) WHERE n.embedding IS NOT NULL "
        "RETURN n.uid AS uid, n.project_name AS project_name, n.embedding AS embedding"
    )
    pairs: list[SimilarPair] = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            similarity = _cosine_similarity(a["embedding"], b["embedding"])
            if similarity >= threshold:
                pairs.append(
                    SimilarPair(
                        uid_a=a["uid"],
                        uid_b=b["uid"],
                        project_a=a["project_name"],
                        project_b=b["project_name"],
                        similarity=round(similarity, 4),
                    )
                )
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def build_dream_report(
    graph: GraphClient,
    vault_roots: list[VaultRoot],
    *,
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
) -> DreamReport:
    """Compute the deterministic dream-mode lint report.

    *vault_roots* are scanned on disk for duplicate ids and link targets;
    everything else is computed from the graph, which already spans every
    indexed vault regardless of source.
    """
    duplicate_ids: list[DuplicateIdConflict] = []
    link_targets: list[tuple[str, str, str]] = []
    memory_index_issues: list[str] = []

    for vault in vault_roots:
        by_qn, links = _scan_vault_for_notes(vault)
        duplicate_ids.extend(
            DuplicateIdConflict(qualified_name=qn, project_name=vault.project_name, file_paths=paths)
            for qn, paths in by_qn.items()
            if len(paths) > 1
        )
        link_targets.extend(links)
        memory_index_issues.extend(_check_memory_index(vault))

    dangling_links = await _find_dangling_links(graph, link_targets)
    orphan_notes = await _find_orphan_notes(graph)
    inbox_count, inbox_paths = await _find_inbox_notes(graph)
    similar_pairs = await _find_similar_pairs(graph, similarity_threshold)
    promotion_candidates = [p for p in similar_pairs if p.project_a != p.project_b]

    return DreamReport(
        inbox_count=inbox_count,
        inbox_paths=inbox_paths,
        orphan_notes=orphan_notes,
        duplicate_ids=duplicate_ids,
        dangling_links=dangling_links,
        similar_pairs=similar_pairs,
        promotion_candidates=promotion_candidates,
        memory_index_issues=memory_index_issues,
    )


# ---------------------------------------------------------------------------
# HOME.md landing page rendering
# ---------------------------------------------------------------------------


def _render_list(items: list[str], *, empty: str = "_(none)_") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def render_home_md(report: DreamReport) -> str:
    """Render the vault landing page — inbox digest, lint findings, hubs.

    Produced only by ``atlas dream`` (no daemon timer — avoids a
    write->watch->index feedback loop and vault git churn).
    """
    sections: list[str] = [
        "# Knowledge Vault — Home",
        "",
        "_Generated by `atlas dream` — do not edit directly._",
        "",
        f"## Inbox ({report.inbox_count})",
        "",
        _render_list(report.inbox_paths, empty="_(empty)_"),
        "",
        f"## Orphan notes ({len(report.orphan_notes)})",
        "",
        _render_list([f"{n.uid} ({n.file_path})" for n in report.orphan_notes]),
        "",
        f"## Dangling links ({len(report.dangling_links)})",
        "",
        _render_list([f"{d.from_uid} --{d.rel_type}--> {d.target_uid}" for d in report.dangling_links]),
        "",
        f"## Duplicate ids ({len(report.duplicate_ids)})",
        "",
        _render_list([f"{d.qualified_name}: {d.file_paths}" for d in report.duplicate_ids]),
        "",
        f"## Similar note pairs ({len(report.similar_pairs)})",
        "",
        _render_list([f"{p.uid_a} ~ {p.uid_b} ({p.similarity})" for p in report.similar_pairs]),
        "",
        f"## Promotion candidates ({len(report.promotion_candidates)})",
        "",
        _render_list([f"{p.uid_a} ~ {p.uid_b} ({p.similarity})" for p in report.promotion_candidates]),
        "",
        "## MEMORY.md consistency",
        "",
        _render_list(report.memory_index_issues, empty="_(no issues found)_"),
        "",
    ]
    return "\n".join(sections)


def report_to_dict(report: DreamReport) -> dict[str, Any]:
    """Plain-dict view of *report* for JSON output (CLI ``--json`` / MCP tool)."""
    return {
        "inbox_count": report.inbox_count,
        "inbox_paths": report.inbox_paths,
        "orphan_notes": [vars(n) for n in report.orphan_notes],
        "duplicate_ids": [vars(d) for d in report.duplicate_ids],
        "dangling_links": [vars(d) for d in report.dangling_links],
        "similar_pairs": [vars(p) for p in report.similar_pairs],
        "promotion_candidates": [vars(p) for p in report.promotion_candidates],
        "memory_index_issues": report.memory_index_issues,
    }
