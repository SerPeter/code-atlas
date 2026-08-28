"""Deterministic dream-mode report — the lint half of knowledge consolidation.

Computes inbox digest, orphan notes, dangling links, duplicate-id conflicts,
and cross-note similarity (including cross-project promotion candidates)
across every configured vault (this project's own vault, see ``[knowledge] vault_path``,
plus any ``[knowledge] extra_vaults``), regardless of which project a Note lives in. The disposition step
(KEEP/MERGE/PROMOTE/DROP) is agent-side — see the ``dream-mode`` command —
this module only produces the deterministic inputs it consumes.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from code_atlas.parsing.ast import parse_file
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from pathlib import Path

    from code_atlas.graph.protocol import GraphBackend

# LINKS_TO/DERIVED_FROM/SUPERSEDES targets are deterministic note uids (see
# markdown.py's _resolve_note_ref) — an exact-uid miss means the link is
# genuinely dangling. DOCUMENTS targets resolve heuristically by name instead,
# so an unresolved one is an expected heuristic-miss, not a lint finding.
_LINK_REL_TYPES = frozenset({RelType.LINKS_TO, RelType.DERIVED_FROM, RelType.SUPERSEDES, RelType.CONTRADICTS})

# Two thresholds, because one cannot serve two decisions. 0.92 is tuned to avoid
# false merge candidates -- which means genuinely fragmented concepts sitting at
# 0.80-0.92 (same topic, different wording) never surfaced at all. Lowering the
# single threshold instead would flood the disposition table with noise and spend
# LLM adjudication on obvious non-pairs. So: a high-confidence merge band, and a
# gray zone that is the only thing needing judgement.
_DEFAULT_SIMILARITY_MERGE = 0.92
_DEFAULT_SIMILARITY_REVIEW = 0.80


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
class BrokenAnchor:
    """A Note whose explicit ``anchors:`` reference is broken (deleted target) or unresolved."""

    uid: str
    name: str
    project_name: str
    file_path: str
    unresolved_anchors: list[str]


@dataclass(frozen=True)
class SimilarPair:
    """Two notes that may be the same concept — merge/dup candidates."""

    uid_a: str
    uid_b: str
    project_a: str
    project_b: str
    similarity: float
    # "merge" -- high confidence, defaults to MERGE with a human confirming.
    # "review" -- the gray zone, and the only band worth spending judgement on.
    band: str = "merge"
    # "embedding" or "title". A title collision is not a similarity score, and
    # reporting 1.0 for one would claim a measurement that was never made.
    match: str = "embedding"


@dataclass(frozen=True)
class Fragmentation:
    """How splintered the vault is: N notes forming M distinct concepts.

    The leading indicator. If resolution is poor everything downstream inherits it,
    and a pair list alone never says whether the vault is getting better or worse.
    """

    notes: int
    concepts: int
    clustered_notes: int
    multi_note_clusters: int

    @property
    def ratio(self) -> float:
        """Concepts per note. 1.0 is perfectly resolved; lower means more splintering."""
        return round(self.concepts / self.notes, 4) if self.notes else 1.0


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
    fragmentation: Fragmentation = field(default_factory=lambda: Fragmentation(0, 0, 0, 0))
    broken_anchors: list[BrokenAnchor] = field(default_factory=list)
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


async def _find_dangling_links(graph: GraphBackend, links: list[tuple[str, str, str]]) -> list[DanglingLink]:
    if not links:
        return []
    target_uids = sorted({to_uid for _, _, to_uid in links})
    existing = await graph.get_existing_uids(target_uids)
    return [
        DanglingLink(from_uid=from_uid, rel_type=rel_type, target_uid=to_uid)
        for from_uid, rel_type, to_uid in links
        if to_uid not in existing
    ]


async def _find_orphan_notes(graph: GraphBackend) -> list[OrphanNote]:
    rows = await graph.get_orphan_notes()
    return [OrphanNote(**row) for row in rows]


async def _find_broken_anchors(graph: GraphBackend) -> list[BrokenAnchor]:
    """Notes whose explicit ``anchors:`` are broken (deleted target) or unresolved (no match)."""
    rows = await graph.get_broken_anchor_notes()
    return [
        BrokenAnchor(
            uid=row["uid"],
            name=row["name"],
            project_name=row["project_name"],
            file_path=row["file_path"],
            unresolved_anchors=row["unresolved_anchors"] or [],
        )
        for row in rows
    ]


async def _find_inbox_notes(graph: GraphBackend) -> tuple[int, list[str]]:
    paths = await graph.get_inbox_note_paths()
    return len(paths), paths


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


_TITLE_NOISE = re.compile(r"[^a-z0-9]+")


def _normalize_title(name: str) -> str:
    """Case- and punctuation-folded title, for exact-collision blocking."""
    return _TITLE_NOISE.sub("-", name.strip().lower()).strip("-")


def _title_collisions(rows: list[dict[str, Any]]) -> list[SimilarPair]:
    """Notes sharing a normalized title, grouped by hash rather than compared pairwise.

    Cheap on purpose: this must not add a second O(N^2) pass. It also runs on notes
    that have **no vector**, which is the whole point — an unembedded note is invisible
    to the cosine scan, so the parallel-worktree duplicate (same note discovered twice,
    one copy never embedded) could not be found at all before.
    """
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        title = _normalize_title(str(row.get("name") or ""))
        if title:
            by_title[title].append(row)

    pairs: list[SimilarPair] = []
    for group in by_title.values():
        if len(group) < 2:
            continue
        ordered = sorted(group, key=lambda r: r["uid"])
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                pairs.append(
                    SimilarPair(
                        uid_a=a["uid"],
                        uid_b=b["uid"],
                        project_a=a["project_name"],
                        project_b=b["project_name"],
                        # Not a similarity score. An exact title collision is a
                        # different kind of evidence, and reporting 1.0 would claim a
                        # measurement nobody made.
                        similarity=1.0,
                        band="merge",
                        match="title",
                    )
                )
    return pairs


def _fragmentation(note_count: int, merge_pairs: list[SimilarPair]) -> Fragmentation:
    """Union-find over the merge band: N notes collapse into M concepts."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair in merge_pairs:
        union(pair.uid_a, pair.uid_b)

    clusters: dict[str, set[str]] = defaultdict(set)
    for uid in parent:
        clusters[find(uid)].add(uid)
    multi = [c for c in clusters.values() if len(c) > 1]
    clustered = sum(len(c) for c in multi)
    # Every note outside a multi-note cluster is its own concept.
    concepts = (note_count - clustered) + len(multi)
    return Fragmentation(
        notes=note_count,
        concepts=concepts,
        clustered_notes=clustered,
        multi_note_clusters=len(multi),
    )


async def _find_similar_pairs(
    graph: GraphBackend, merge_threshold: float, review_threshold: float
) -> list[SimilarPair]:
    """Candidate duplicate pairs, split into a merge band and a review band.

    Two sources of evidence. Cosine similarity over stored vectors, banded by the two
    thresholds; and exact normalized-title collisions, which land in the merge band and
    work without vectors at all.

    O(N^2) over the embedded notes — acceptable for a periodic lint report, not a hot
    query path; Memgraph's vector_search is a KNN-for-one-vector primitive, not an
    all-pairs one, so pulling every vector once and comparing in Python stays simpler.
    The title pass is hash-grouped and adds no second quadratic sweep.
    """
    rows = await graph.get_notes_for_dedup()
    embedded = [r for r in rows if r.get("embedding")]

    seen: set[tuple[str, str]] = set()
    pairs: list[SimilarPair] = []
    for pair in _title_collisions(rows):
        seen.add((pair.uid_a, pair.uid_b))
        pairs.append(pair)

    for i in range(len(embedded)):
        for j in range(i + 1, len(embedded)):
            a, b = embedded[i], embedded[j]
            key = (a["uid"], b["uid"]) if a["uid"] < b["uid"] else (b["uid"], a["uid"])
            if key in seen:
                continue  # already a title collision; that evidence is stronger
            similarity = _cosine_similarity(a["embedding"], b["embedding"])
            if similarity < review_threshold:
                continue
            seen.add(key)
            pairs.append(
                SimilarPair(
                    uid_a=a["uid"],
                    uid_b=b["uid"],
                    project_a=a["project_name"],
                    project_b=b["project_name"],
                    similarity=round(similarity, 4),
                    band="merge" if similarity >= merge_threshold else "review",
                    match="embedding",
                )
            )
    pairs.sort(key=lambda p: (p.band != "merge", -p.similarity))
    return pairs


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def build_dream_report(
    graph: GraphBackend,
    vault_roots: list[VaultRoot],
    *,
    similarity_merge: float = _DEFAULT_SIMILARITY_MERGE,
    similarity_review: float = _DEFAULT_SIMILARITY_REVIEW,
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
    broken_anchors = await _find_broken_anchors(graph)
    inbox_count, inbox_paths = await _find_inbox_notes(graph)
    similar_pairs = await _find_similar_pairs(graph, similarity_merge, similarity_review)
    promotion_candidates = [p for p in similar_pairs if p.project_a != p.project_b]
    # Fragmentation counts only the merge band: a gray-zone pair is a question, and
    # collapsing questions into concepts would report a resolution nobody made.
    all_notes = await graph.get_notes_for_dedup()
    fragmentation = _fragmentation(len(all_notes), [p for p in similar_pairs if p.band == "merge"])

    return DreamReport(
        inbox_count=inbox_count,
        inbox_paths=inbox_paths,
        orphan_notes=orphan_notes,
        duplicate_ids=duplicate_ids,
        dangling_links=dangling_links,
        similar_pairs=similar_pairs,
        promotion_candidates=promotion_candidates,
        fragmentation=fragmentation,
        broken_anchors=broken_anchors,
        memory_index_issues=memory_index_issues,
    )


# ---------------------------------------------------------------------------
# HOME.md landing page rendering
# ---------------------------------------------------------------------------


def _render_list(items: list[str], *, empty: str = "_(none)_") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _band(report: DreamReport, band: str) -> list[SimilarPair]:
    return [p for p in report.similar_pairs if p.band == band]


def _render_pair(pair: SimilarPair) -> str:
    """A pair line that says what kind of evidence put it there."""
    if pair.match == "title":
        return f"{pair.uid_a} ~ {pair.uid_b} (identical title)"
    return f"{pair.uid_a} ~ {pair.uid_b} ({pair.similarity})"


def render_fragmentation(frag: Fragmentation) -> str:
    """The headline number: how splintered the vault is, in one line.

    The trend across cycles is the point, not the absolute value — a pair list never
    says whether things are getting better or worse.
    """
    if not frag.notes:
        return "Fragmentation: no notes indexed"
    detail = (
        f" ({frag.clustered_notes} notes in {frag.multi_note_clusters} multi-note clusters)"
        if frag.multi_note_clusters
        else " (no duplicate clusters)"
    )
    return f"Fragmentation: {frag.notes} notes, {frag.concepts} concepts{detail}"


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
        # Leads the report on purpose: resolution is the leading indicator. If the
        # vault is splintering, everything downstream of retrieval inherits it, and a
        # list of pairs never says whether that is getting better or worse.
        f"**{render_fragmentation(report.fragmentation)}**",
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
        f"## Broken anchors ({len(report.broken_anchors)})",
        "",
        _render_list(
            [f"{b.uid} ({b.file_path}): {b.unresolved_anchors or 'target deleted'}" for b in report.broken_anchors]
        ),
        "",
        f"## Duplicate ids ({len(report.duplicate_ids)})",
        "",
        _render_list([f"{d.qualified_name}: {d.file_paths}" for d in report.duplicate_ids]),
        "",
        f"## Similar note pairs ({len(report.similar_pairs)})",
        "",
        # Merge band first: it is the actionable half. The review band is where
        # judgement is actually needed, and separating them is what keeps the
        # adjudication cost proportional to the ambiguity rather than the volume.
        f"### Merge band ({len(_band(report, 'merge'))}) — high confidence",
        "",
        _render_list([_render_pair(p) for p in _band(report, "merge")]),
        "",
        f"### Review band ({len(_band(report, 'review'))}) — needs judgement",
        "",
        _render_list([_render_pair(p) for p in _band(report, "review")]),
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
        "broken_anchors": [vars(b) for b in report.broken_anchors],
        # `vars` already carries `band` and `match`, so a consumer can tell a
        # high-confidence merge from a gray-zone question without re-deriving it.
        "similar_pairs": [vars(p) for p in report.similar_pairs],
        "merge_band": [vars(p) for p in report.similar_pairs if p.band == "merge"],
        "review_band": [vars(p) for p in report.similar_pairs if p.band == "review"],
        "promotion_candidates": [vars(p) for p in report.promotion_candidates],
        "fragmentation": {**vars(report.fragmentation), "ratio": report.fragmentation.ratio},
        "memory_index_issues": report.memory_index_issues,
    }
