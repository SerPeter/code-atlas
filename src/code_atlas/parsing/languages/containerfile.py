"""Container build support — tree-sitter parser for Dockerfiles / Containerfiles.

Emits one Module entity per file plus one ``docker_stage`` TypeDef per ``FROM``,
wired up with:

- ``Module -DEFINES-> stage``.
- ``stage -IMPORTS-> stage`` for an intra-file ``FROM <alias>`` or
  ``COPY --from=<alias>`` (the multi-stage edge). ``to_name`` is the target
  stage's *unprefixed* qualified_name, which ``GraphClient.resolve_imports``
  matches exactly against its internal qualified_name map — no name guessing,
  no cross-file ambiguity.
- ``stage -IMPORTS-> ExternalPackage`` for a base image or a
  ``COPY --from=<image>``. Parsers never emit ExternalPackage/ExternalSymbol
  nodes themselves: ``resolve_imports`` mints them, keyed
  ``{project}:ext/{top_level}`` where ``top_level`` is the part of ``to_name``
  before the first dot. So ``to_name`` carries the image *repository* with tag
  and digest stripped — a tag would leave ``ext/python:3`` as the package name.
  Registry-qualified images inherit the same dot split that Go import paths
  already live with (``ghcr.io/astral-sh/uv`` → package ``ext/ghcr``, symbol
  ``ext/ghcr.io/astral-sh/uv``).

COPY/ADD build-context sources are recorded on the stage as a ``copy_sources``
property rather than as edges, for two reasons. The build context is chosen by
the *caller* (``docker build <ctx>``, compose ``build.context``) and never
stated in the Dockerfile, so these paths are only repo-relative under the usual
root-context convention. And nothing in GraphClient routes a relationship by
file path from a non-doc node — the one path-suffix resolver
(``_create_doc_links``) requires a DocSection/Note source. Keeping them as
normalized paths leaves the join available (``MATCH (m:Module) WHERE
m.file_path IN stage.copy_sources``) without minting a bogus ExternalSymbol for
every path that happens not to match.

Stages are TypeDefs — the label for a named declarative object, which makes them
embeddable and text-searchable. Caveat: TypeDef is what the *name*-matched
resolvers target (INHERITS/IMPLEMENTS, resolve_type_refs,
resolve_member_defines), so a stage alias exactly equal to a code type name can
perturb those. Stage aliases are lowercase by convention while type names are
usually capitalized, which keeps the collision surface small.

Grammar notes (measured, tree-sitter-containerfile ABI 15):
  - Root is ``source_file``; instructions are its direct children.
  - ``from_instruction``: field ``as`` → ``image_alias``, plus an ``image_spec``
    child with fields ``name``/``tag``/``digest`` (``tag`` and ``digest`` text
    keep their leading ``:``/``@``).
  - ``copy_instruction``/``add_instruction``: ``param`` children hold flags
    (``--from=builder``) as raw text with no named name/value children;
    ``path`` children are the sources followed by the destination.
  - ``$VAR`` interpolation shows up as an ``expansion`` child of ``path`` and
    ``image_name``; those values are build-time inputs, so they are dropped.
  - The package is ``tree-sitter-containerfile``, NOT ``tree-sitter-dockerfile``
    — the latter publishes no Windows wheels and must not be used.

This is the one language dispatched by *filename* rather than suffix: the
canonical file is called ``Dockerfile`` with no extension at all. See
``LanguageConfig.filenames``.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from tree_sitter import Node

_EXTENSIONS = frozenset({".dockerfile", ".containerfile"})

# Lowercase: get_language_for_file lowercases the basename before lookup.
_FILENAMES = frozenset({"dockerfile", "containerfile"})

_STAGE_KIND = "docker_stage"

_COPY_INSTRUCTIONS = frozenset({"copy_instruction", "add_instruction"})

# ADD also accepts remote sources; none of them are build-context paths.
_REMOTE_PREFIXES = ("http://", "https://", "ftp://", "ssh://", "git://", "git@")


def _module_qualified_name(file_path: str) -> str:
    """Convert a file path to a dotted qualified name, extension folded in.

    ``build/Dockerfile`` -> ``build.Dockerfile``
    ``build/api.dockerfile`` -> ``build.api_dockerfile``

    Unlike the code-language modules, any extension is *preserved* (its dot
    replaced) rather than stripped, because ``qualified_name`` IS the graph uid
    and stripping would let two differently-named files claim one node.
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    parts = list(p.parts)
    if parts:
        parts[-1] = parts[-1].replace(".", "_")
    return ".".join(parts)


def _split_stages(root: Node) -> list[tuple[Node, list[Node]]]:
    """Split the file into ``(from_instruction, following instructions)`` pairs.

    Instructions before the first ``FROM`` (global ``ARG``s, the ``# syntax``
    directive) belong to no stage and are dropped.
    """
    stages: list[tuple[Node, list[Node]]] = []
    for child in root.children:
        if child.type == "from_instruction":
            stages.append((child, []))
        elif stages:
            stages[-1][1].append(child)
    return stages


def _stage_keys(stages: list[tuple[Node, list[Node]]]) -> list[str]:
    """Name each stage by its ``AS`` alias, falling back to ``stage{index}``.

    Duplicate keys get a ``#index`` suffix: ``qualified_name`` is the graph uid,
    so two stages collapsing onto one key would silently overwrite each other's
    node. Docker rejects duplicate stage names, but a broken file still must not
    corrupt the graph.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for index, (from_node, _) in enumerate(stages):
        alias = from_node.child_by_field_name("as")
        key = node_text(alias) if alias is not None else f"stage{index}"
        while key.lower() in seen:
            key = f"{key}#{index}"
        seen.add(key.lower())
        keys.append(key)
    return keys


def _image_spec(from_node: Node) -> str:
    """Raw text of a ``FROM``'s image spec — ``python:3.14-slim``, ``base``, ``$IMG``."""
    spec = next((c for c in from_node.children if c.type == "image_spec"), None)
    return node_text(spec) if spec is not None else ""


def _repository(image_ref: str) -> str | None:
    """Strip tag and digest from an image reference, leaving the repository.

    ``nginx:1.27-alpine`` -> ``nginx``; ``ghcr.io/x/y@sha256:...`` -> ``ghcr.io/x/y``.
    A colon before a ``/`` is a registry port, not a tag, so
    ``localhost:5000/app`` survives intact. Returns None for anything
    interpolating a build ARG — an ExternalPackage named ``$BASE_IMAGE`` is
    worse than no edge at all.
    """
    if "$" in image_ref:
        return None
    ref = image_ref.split("@", 1)[0]
    head, sep, tail = ref.rpartition(":")
    if sep and "/" not in tail:
        ref = head
    return ref or None


def _copy_from_ref(node: Node) -> str | None:
    """The ``--from=`` value of a COPY/ADD instruction, if it has one.

    The grammar exposes flags as a ``param`` node whose name and value are plain
    text with no named children, so this reads the node's text.
    """
    for child in node.children:
        if child.type != "param":
            continue
        key, sep, value = node_text(child).removeprefix("--").partition("=")
        if sep and key.lower() == "from":
            return value
    return None


def _dependency_target(ref: str, before: int, keys_by_name: dict[str, int], stage_qns: list[str]) -> str | None:
    """Resolve a ``FROM``/``--from=`` reference into an IMPORTS ``to_name``.

    An earlier stage — by case-insensitive alias or by numeric index — resolves
    to that stage's unprefixed qualified_name, which resolve_imports matches
    exactly. Anything else is an image reference and becomes an external package
    name. Docker only allows references to stages declared *earlier*, and stage
    names never carry a tag, so ``base:latest`` is an image even when a stage
    called ``base`` exists.
    """
    if ref.isdigit():
        index = int(ref)
        # An out-of-range index is a broken file, not an image called "7".
        return stage_qns[index] if index < before else None
    index = keys_by_name.get(ref.lower(), -1)
    if 0 <= index < before:
        return stage_qns[index]
    return _repository(ref)


def _context_path(raw: str) -> str | None:
    """Normalize a COPY/ADD source into a repo-root-relative path, or None.

    Assumes the build context is the repository root (the ``docker build .``
    default) — the Dockerfile itself never states its context. Drops
    container-absolute paths, remote ADD sources, heredoc bodies and anything
    interpolating a variable. Globs are kept verbatim.
    """
    if not raw or raw.startswith(("/", "<<")) or "$" in raw:
        return None
    if raw.lower().startswith(_REMOTE_PREFIXES):
        return None
    parts: list[str] = []
    for part in raw.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if not parts:
                # Escapes the context root — not resolvable against the repo.
                return None
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts) or None


def _context_sources(node: Node) -> list[str]:
    """Build-context sources of a COPY/ADD — every ``path`` child but the last."""
    paths = [c for c in node.children if c.type == "path"]
    return [p for p in (_context_path(node_text(n)) for n in paths[:-1]) if p is not None]


def _parse_containerfile(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile:
    """Extract the module entity, its build stages, and their dependency edges."""
    norm_path = path.replace("\\", "/")
    language = "containerfile"

    if not source.strip():
        # No Module node for an empty file — it would be an unsearchable stub
        # that still costs an embedding.
        return ParsedFile(file_path=norm_path, language=language, entities=[], relationships=[])

    module_qn = _module_qualified_name(norm_path)
    module_uid = f"{project_name}:{module_qn}"
    entities = [
        ParsedEntity(
            name=PurePosixPath(norm_path).name,
            qualified_name=module_uid,
            label=NodeLabel.MODULE,
            kind="containerfile",
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=norm_path,
        )
    ]
    relationships: list[ParsedRelationship] = []

    stages = _split_stages(root)
    keys = _stage_keys(stages)
    stage_qns = [f"{module_qn}.{key}" for key in keys]
    keys_by_name = {key.lower(): index for index, key in enumerate(keys)}

    for index, (from_node, instructions) in enumerate(stages):
        stage_uid = f"{project_name}:{stage_qns[index]}"
        spec = _image_spec(from_node)
        sources: list[str] = []
        targets = [_dependency_target(spec, index, keys_by_name, stage_qns)]

        for instruction in instructions:
            if instruction.type not in _COPY_INSTRUCTIONS:
                continue
            copy_from = _copy_from_ref(instruction)
            if copy_from is None:
                # Without --from, the sources come from the build context.
                sources.extend(_context_sources(instruction))
            else:
                targets.append(_dependency_target(copy_from, index, keys_by_name, stage_qns))

        # Trailing comments are excluded from the span so that a `# NOTE:` block
        # sitting above the next FROM is attributed to that stage, not this one.
        body = [n for n in instructions if n.type != "comment"]
        last = body[-1] if body else from_node
        extra: dict[str, Any] = {"stage_index": index}
        if spec:
            extra["base_image"] = spec
        if sources:
            extra["copy_sources"] = sorted(set(sources))

        entities.append(
            ParsedEntity(
                name=keys[index],
                qualified_name=stage_uid,
                label=NodeLabel.TYPE_DEF,
                kind=_STAGE_KIND,
                line_start=from_node.start_point[0] + 1,
                line_end=last.end_point[0] + 1,
                file_path=norm_path,
                signature=" ".join(node_text(from_node).split()),
                source=source[from_node.start_byte : last.end_byte].decode("utf-8", errors="replace"),
                extra_properties=extra,
            )
        )
        relationships.append(
            ParsedRelationship(from_qualified_name=module_uid, rel_type=RelType.DEFINES, to_name=stage_uid)
        )
        relationships.extend(
            ParsedRelationship(from_qualified_name=stage_uid, rel_type=RelType.IMPORTS, to_name=target)
            for target in dict.fromkeys(t for t in targets if t is not None)
        )

    return ParsedFile(file_path=norm_path, language=language, entities=entities, relationships=relationships)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

try:
    import tree_sitter_containerfile as _ts_containerfile
    from tree_sitter import Language, Query

    _CONTAINERFILE_LANGUAGE = Language(_ts_containerfile.language())
    _CONTAINERFILE_QUERY = Query(_CONTAINERFILE_LANGUAGE, "(source_file) @root")

    register_language(
        LanguageConfig(
            name="containerfile",
            extensions=_EXTENSIONS,
            language=_CONTAINERFILE_LANGUAGE,
            query=_CONTAINERFILE_QUERY,
            parse_func=_parse_containerfile,
            filenames=_FILENAMES,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass
