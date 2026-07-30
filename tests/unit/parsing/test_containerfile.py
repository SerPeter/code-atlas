"""Tests for Dockerfile / Containerfile parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_containerfile", reason="tree-sitter-containerfile not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import NodeLabel, RelType

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "Dockerfile") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity_by_name(parsed: ParsedFile, name: str):
    matches = [e for e in parsed.entities if e.name == name]
    assert len(matches) == 1, (
        f"Expected 1 entity named {name!r}, got {len(matches)}: {[e.name for e in parsed.entities]}"
    )
    return matches[0]


def _stages(parsed: ParsedFile):
    return [e for e in parsed.entities if e.kind == "docker_stage"]


def _rels_from(parsed: ParsedFile, from_qn_suffix: str, rel_type: RelType):
    return [
        r for r in parsed.relationships if r.from_qualified_name.endswith(from_qn_suffix) and r.rel_type == rel_type
    ]


def _imports(parsed: ParsedFile, from_qn_suffix: str) -> list[str]:
    return [r.to_name for r in _rels_from(parsed, from_qn_suffix, RelType.IMPORTS)]


# ---------------------------------------------------------------------------
# 1. Language detection — BOTH dispatch routes (basename and extension)
# ---------------------------------------------------------------------------


def test_language_detection_by_basename():
    for path in ("Dockerfile", "docker/Dockerfile", "Containerfile", "deploy/Containerfile", "dockerfile"):
        cfg = get_language_for_file(path)
        assert cfg is not None, path
        assert cfg.name == "containerfile", path


def test_language_detection_by_extension():
    for path in ("build/api.dockerfile", "x.containerfile", "build/API.Dockerfile"):
        cfg = get_language_for_file(path)
        assert cfg is not None, path
        assert cfg.name == "containerfile", path


def test_language_detection_not_containerfile():
    # Whole-basename match only: the basename here is "dockerfile.txt".
    assert get_language_for_file("dockerfile.txt") is None
    assert get_language_for_file("data.csv") is None
    assert get_language_for_file("readme.txt") is None


def test_parse_file_reached_via_basename_route():
    parsed = _parse("FROM alpine AS base\n", path="docker/Dockerfile")
    assert parsed.language == "containerfile"
    assert [e.name for e in _stages(parsed)] == ["base"]


def test_parse_file_reached_via_extension_route():
    parsed = _parse("FROM alpine AS base\n", path="docker/api.dockerfile")
    assert parsed.language == "containerfile"
    assert [e.name for e in _stages(parsed)] == ["base"]
    module = _entity_by_name(parsed, "api.dockerfile")
    assert module.qualified_name == f"{PROJECT}:docker.api_dockerfile"


# ---------------------------------------------------------------------------
# 2. Module entity
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("FROM alpine\n", path="docker/Dockerfile")
    module = _entity_by_name(parsed, "Dockerfile")
    assert module.label == NodeLabel.MODULE
    assert module.kind == "containerfile"
    assert module.qualified_name == f"{PROJECT}:docker.Dockerfile"
    assert module.file_path == "docker/Dockerfile"
    assert module.line_start == 1


# ---------------------------------------------------------------------------
# 3. Build stages as entities
# ---------------------------------------------------------------------------


def test_stage_entities_one_per_from():
    parsed = _parse(
        "# syntax=docker/dockerfile:1\n"
        "ARG PY=3.14\n"
        "FROM python:3.14-slim AS base\n"
        "WORKDIR /app\n"
        "\n"
        "FROM base AS builder\n"
        "RUN uv build\n"
        "\n"
        "FROM base\n"
        'CMD ["app"]\n'
    )
    stages = _stages(parsed)
    assert [e.name for e in stages] == ["base", "builder", "stage2"]
    assert all(e.label == NodeLabel.TYPE_DEF for e in stages)
    assert [e.qualified_name for e in stages] == [
        f"{PROJECT}:Dockerfile.base",
        f"{PROJECT}:Dockerfile.builder",
        f"{PROJECT}:Dockerfile.stage2",
    ]
    assert [e.extra_properties["stage_index"] for e in stages] == [0, 1, 2]
    assert [e.extra_properties["base_image"] for e in stages] == ["python:3.14-slim", "base", "base"]


def test_stage_span_excludes_trailing_comment_and_blank_lines():
    parsed = _parse("FROM alpine AS base\nRUN echo hi\n\n# NOTE: the runtime stage\nFROM base AS final\nRUN echo bye\n")
    base = _entity_by_name(parsed, "base")
    assert (base.line_start, base.line_end) == (1, 2)
    final = _entity_by_name(parsed, "final")
    assert (final.line_start, final.line_end) == (5, 6)
    # Excluding the trailing comment is what keeps the note off the *preceding*
    # stage. It settles on the file-level module rather than on `final` because
    # the grammar's comment node spills onto the next line, which puts `final`
    # outside extract_rationale's "following declaration" window.
    assert base.rationale is None
    assert _entity_by_name(parsed, "Dockerfile").rationale == "NOTE: the runtime stage"


def test_stage_signature_and_source():
    parsed = _parse("FROM python:3.14-slim AS base\nRUN uv sync\n")
    base = _entity_by_name(parsed, "base")
    assert base.signature == "FROM python:3.14-slim AS base"
    assert base.source == "FROM python:3.14-slim AS base\nRUN uv sync"


def test_module_defines_every_stage():
    parsed = _parse("FROM alpine AS base\nFROM base AS final\n")
    defines = _rels_from(parsed, "Dockerfile", RelType.DEFINES)
    assert sorted(r.to_name for r in defines) == [
        f"{PROJECT}:Dockerfile.base",
        f"{PROJECT}:Dockerfile.final",
    ]


def test_duplicate_stage_alias_gets_distinct_uid():
    parsed = _parse("FROM alpine AS base\nFROM alpine AS BASE\n")
    stages = _stages(parsed)
    assert len(stages) == 2
    assert len({e.qualified_name for e in stages}) == 2


# ---------------------------------------------------------------------------
# 4. Base images become external dependency edges
# ---------------------------------------------------------------------------


def test_base_image_import_strips_tag_and_digest():
    parsed = _parse(
        "FROM python:3.14-slim AS a\n"
        "FROM node:20@sha256:abc AS b\n"
        "FROM ghcr.io/astral-sh/uv:0.5 AS c\n"
        "FROM localhost:5000/app AS d\n"
        "FROM alpine AS e\n"
    )
    assert _imports(parsed, "Dockerfile.a") == ["python"]
    assert _imports(parsed, "Dockerfile.b") == ["node"]
    assert _imports(parsed, "Dockerfile.c") == ["ghcr.io/astral-sh/uv"]
    assert _imports(parsed, "Dockerfile.d") == ["localhost:5000/app"]
    assert _imports(parsed, "Dockerfile.e") == ["alpine"]


def test_interpolated_base_image_emits_no_import():
    parsed = _parse("ARG BASE=alpine\nFROM $BASE AS dyn\n")
    dyn = _entity_by_name(parsed, "dyn")
    assert dyn.extra_properties["base_image"] == "$BASE"
    assert _imports(parsed, "Dockerfile.dyn") == []


# ---------------------------------------------------------------------------
# 5. Intra-file FROM references resolve to the earlier stage's uid
# ---------------------------------------------------------------------------


def test_from_stage_alias_targets_stage_qualified_name():
    parsed = _parse("FROM alpine AS base\nFROM base AS final\n")
    # Unprefixed qualified_name — that is what resolve_imports matches on.
    assert _imports(parsed, "Dockerfile.final") == ["Dockerfile.base"]


def test_from_stage_alias_is_case_insensitive():
    parsed = _parse("FROM alpine AS Base\nFROM base AS final\n")
    assert _imports(parsed, "Dockerfile.final") == ["Dockerfile.Base"]


def test_tagged_reference_is_an_image_not_a_stage():
    parsed = _parse("FROM alpine AS base\nFROM base:latest AS final\n")
    assert _imports(parsed, "Dockerfile.final") == ["base"]


def test_forward_stage_reference_is_treated_as_an_image():
    parsed = _parse("FROM final AS first\nFROM alpine AS final\n")
    assert _imports(parsed, "Dockerfile.first") == ["final"]


# ---------------------------------------------------------------------------
# 6. COPY --from is the multi-stage edge
# ---------------------------------------------------------------------------


def test_copy_from_stage_alias():
    parsed = _parse(
        "FROM alpine AS builder\nRUN make\nFROM alpine AS final\nCOPY --from=builder --chown=1:1 /out /out\n"
    )
    assert _imports(parsed, "Dockerfile.final") == ["alpine", "Dockerfile.builder"]


def test_copy_from_numeric_stage_index():
    parsed = _parse("FROM alpine\nFROM alpine AS final\nCOPY --from=0 /out /out\n")
    assert _imports(parsed, "Dockerfile.final") == ["alpine", "Dockerfile.stage0"]


def test_copy_from_out_of_range_index_emits_no_edge():
    parsed = _parse("FROM alpine AS only\nCOPY --from=7 /out /out\n")
    assert _imports(parsed, "Dockerfile.only") == ["alpine"]


def test_copy_from_external_image():
    parsed = _parse("FROM alpine AS final\nCOPY --from=nginx:alpine /etc/nginx /etc/nginx\n")
    assert _imports(parsed, "Dockerfile.final") == ["alpine", "nginx"]


def test_repeated_copy_from_is_deduped():
    parsed = _parse(
        "FROM alpine AS builder\nFROM alpine AS final\nCOPY --from=builder /a /a\nCOPY --from=builder /b /b\n"
    )
    assert _imports(parsed, "Dockerfile.final") == ["alpine", "Dockerfile.builder"]


def test_add_from_stage_alias():
    parsed = _parse("FROM alpine AS builder\nFROM alpine AS final\nADD --from=builder /a /a\n")
    assert _imports(parsed, "Dockerfile.final") == ["alpine", "Dockerfile.builder"]


# ---------------------------------------------------------------------------
# 7. COPY/ADD build-context sources
# ---------------------------------------------------------------------------


def test_copy_sources_are_normalized_repo_relative_paths():
    parsed = _parse(
        "FROM alpine AS base\n"
        "COPY pyproject.toml uv.lock ./\n"
        "COPY ./src ./src\n"
        "COPY ./scripts/../config/app.yml /etc/app.yml\n"
        "ADD scripts/entrypoint.sh /entrypoint.sh\n"
    )
    base = _entity_by_name(parsed, "base")
    assert base.extra_properties["copy_sources"] == [
        "config/app.yml",
        "pyproject.toml",
        "scripts/entrypoint.sh",
        "src",
        "uv.lock",
    ]


def test_copy_sources_skip_unresolvable_and_stage_scoped_paths():
    parsed = _parse(
        "FROM alpine AS builder\n"
        "FROM alpine AS base\n"
        "COPY --from=builder /app/dist /dist\n"
        "ADD https://example.com/x.tar.gz /tmp/\n"
        "COPY $VERSION/x /y\n"
        "COPY /abs/path /dst\n"
        "COPY ../outside/x /dst\n"
    )
    base = _entity_by_name(parsed, "base")
    assert "copy_sources" not in base.extra_properties


def test_copy_sources_survive_line_continuations():
    parsed = _parse("FROM alpine AS base\nCOPY \\\n  a.txt \\\n  b.txt \\\n  /dst/\n")
    base = _entity_by_name(parsed, "base")
    assert base.extra_properties["copy_sources"] == ["a.txt", "b.txt"]


def test_copy_sources_keep_globs():
    parsed = _parse("FROM alpine AS base\nCOPY requirements*.txt ./\n")
    base = _entity_by_name(parsed, "base")
    assert base.extra_properties["copy_sources"] == ["requirements*.txt"]


# ---------------------------------------------------------------------------
# 8. Content hash
# ---------------------------------------------------------------------------


def test_content_hash_tracks_stage_body_and_copy_sources():
    before = _entity_by_name(_parse("FROM alpine AS base\nRUN make one\n"), "base")
    body_changed = _entity_by_name(_parse("FROM alpine AS base\nRUN make two\n"), "base")
    copies_changed = _entity_by_name(_parse("FROM alpine AS base\nRUN make one\nCOPY a.txt /a\n"), "base")
    assert before.content_hash
    assert before.content_hash != body_changed.content_hash
    assert before.content_hash != copies_changed.content_hash


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------


def test_empty_file_produces_no_entities():
    parsed = _parse("")
    assert parsed.entities == []
    assert parsed.relationships == []


def test_whitespace_only_file_produces_no_entities():
    parsed = _parse("\n   \n")
    assert parsed.entities == []


def test_file_without_from_yields_module_only():
    parsed = _parse("# just a fragment\nARG VERSION=1\n")
    assert [e.kind for e in parsed.entities] == ["containerfile"]
    assert parsed.relationships == []


def test_trailing_garbage_does_not_lose_earlier_stages():
    parsed = _parse("FROM alpine AS ok\nRUN make\n!!! nonsense ###\n")
    assert [e.name for e in _stages(parsed)] == ["ok"]


def test_syntax_errors_do_not_crash():
    # tree-sitter recovery can swallow later instructions into an earlier one;
    # the contract here is only "returns a ParsedFile without raising".
    parsed = _parse("FROM\nCOPY\n!!! nonsense ###\nFROM alpine AS ok\n")
    assert _entity_by_name(parsed, "Dockerfile").kind == "containerfile"


def test_undecodable_bytes_do_not_crash():
    result = parse_file("Dockerfile", b"\x00\x01 junk \xff\nFROM alpine\n", PROJECT)
    assert result is not None
    assert [e.name for e in _stages(result)] == ["stage0"]
