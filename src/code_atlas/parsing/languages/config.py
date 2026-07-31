"""Structured config / data support — YAML, JSON, TOML and XML.

Extraction is **dialect-aware**, and that is the whole point of this module.
Four grammars are registered here (the precedent is ``typescript.py``, which
registers typescript/tsx/javascript together), each in its own ``try`` block so
that a single missing or ABI-mismatched wheel costs only its own format instead
of taking the other three down with it.

Grammar notes (all measured):
  - tree-sitter-yaml ABI 14, root ``stream`` (``stream > document``).
  - tree-sitter-json ABI 14, root ``document``.
  - tree-sitter-toml ABI 14, root ``document``.
  - tree-sitter-xml ABI 14, root ``document``. **The module exposes
    ``language_xml()`` and ``language_dtd()`` — there is no ``language()``.**
    Calling ``language()`` raises AttributeError, which the ``except
    ImportError`` guard does NOT catch, so the format would vanish with nothing
    but a warning in the log.

Dispatch note: extension alone does not identify a config *dialect*. A ``.yml``
file may be a GitHub Actions workflow, a k8s manifest, a compose file or an
Ansible playbook, and those want very different entities. ``get_language_for_file``
resolves an extension to exactly one handler, so the branching *must* happen
inside the handler — which is fine, because ``parse_func`` already receives the
path and the raw bytes. ``markdown.py`` is the precedent: it branches on YAML
frontmatter to pick Note-mode versus DocFile-mode inside a single registration.
The XML branch goes one step further and hands off: ``_parse_xml`` offers every
document to ``salesforce.parse_salesforce_metadata`` first (SFDX metadata is all
plain ``.xml``, and its dialects need real per-type handling), and only parses
structurally when that declines.

Why PyYAML rather than the tree-sitter tree
-------------------------------------------
Dialect detection and reference extraction need *semantic* structure — "is the
top level a mapping with ``apiVersion``", "is ``services`` a mapping of
mappings", "is ``depends_on`` the short list form or the long map form". Walking
tree-sitter's ``block_mapping_pair`` spine to rebuild that is a reimplementation
of a YAML loader. What tree-sitter offers over ``yaml.safe_load`` is byte
offsets, and PyYAML's composer already carries those on every node's
``start_mark``/``end_mark``. So this module composes each document with
``yaml.SafeLoader`` (structure *and* marks in one pass) and uses the tree-sitter
parse only as the carrier the framework requires — plus, for free, comment nodes
for rationale extraction. XML is the opposite case: there is no stdlib XML
parser that reports line numbers, and ``xml.etree`` additionally carries
entity-expansion risk on untrusted input, so XML genuinely uses its grammar.

Generic config versus data
--------------------------
A YAML/JSON/TOML file no dialect claims is still *configuration* far more often
than not — ``atlas.toml``, ``.eslintrc.json``, ``group_vars/all.yml`` — and the
questions asked of it ("where is ``line-length`` set?") are answerable from
nothing more than its key tree. So an unrecognised file falls back to
``_parse_generic``: a Module node plus a bounded tree of section/setting nodes.

The one thing that must NOT be indexed is a **data** file. A 4 MB fixture of
10 000 identically shaped records would mint a node and an embedding per record
and answer nothing, so ``_looks_like_data`` triages first and returns ``None``
for anything that smells like a record dump. See its docstring for the ladder.
``parse_file`` normalises that ``None`` into an empty ``ParsedFile``, so a
rejected file still passes through the content-hash gate and is not re-read on
every pass.

Schema mapping (no new NodeLabels, no new RelTypes — both have import-time
validators that RuntimeError, and a new label additionally needs unique
constraints, existence constraints, label-property indices and a label group):
  - the file           -> ``Module``   kind ``k8s_manifest`` / ``compose_file`` /
                          ``github_workflow`` / ``ansible_playbook`` /
                          ``ansible_tasks`` / ``ansible_handlers`` /
                          ``xml_document`` / ``config_file``
  - a declared object  -> ``TypeDef``  kind ``k8s_resource`` / ``compose_service`` /
                          ``ansible_role`` / ``xml_element`` / ``config_section``
  - an invokable unit  -> ``Callable`` kind ``ci_job`` / ``ansible_play`` /
                          ``ansible_task`` / ``ansible_handler``
  - a leaf setting     -> ``Value``    kind ``xml_setting`` / ``config_setting``

Edge mapping is constrained by ``GraphClient``'s routing registries:
  - ``DEFINES``/``CONTAINS`` are uid-routed, so both endpoints must be entities
    this parser emitted for *this* file. Used for file -> object and
    object -> child.
  - ``USES_TYPE`` is post-batch and name-routed to ``TypeDef`` nodes, preferring
    a same-file match and falling back to a project-wide unique one
    (``resolve_type_refs``). That is exactly the resolution profile config
    cross-references need, so every object -> object reference (k8s name refs and
    selector matches, compose ``depends_on``, Ansible role includes) is a
    ``USES_TYPE`` edge. It is why k8s resources are named ``Kind/name``: the
    reference sites almost always know the target's kind, and ``Kind/name``
    makes the name unambiguous where a bare ``web`` would match both a Service
    and a Deployment.
  - ``CALLS`` is post-batch and name-routed to ``Callable`` nodes, with the same
    same-file-then-project-wide cascade plus ADR-0014 ambiguity fan-out. Used
    for GitHub Actions ``needs:`` and Ansible ``notify:`` -> handler.
  - ``IMPORTS`` is post-batch and matches the target's ``qualified_name``,
    minting ``ExternalPackage``/``ExternalSymbol`` stubs when nothing matches
    (``resolve_imports``). Used for in-repo file references (compose
    ``build.dockerfile``, Ansible task includes) and for external references
    (container images, ``uses:`` actions).
  - ``DEPENDS_ON`` is out-of-band (project-to-project) and must never be emitted
    from a parser, which is why compose ``depends_on`` becomes ``USES_TYPE``.

Known limitation: entity uids are derived from the file path, so a reference
whose target lives in another file resolves only through ``USES_TYPE``/``CALLS``
name matching, never by uid. Content-derived uids would give cross-file
uid-routing but would collide whenever two files declare the same object — which
is precisely what a kustomize overlay patch does. Path-derived uids plus
name-routed references is the trade this module makes.
"""

from __future__ import annotations

import json
import re
import tomllib
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import yaml
from tree_sitter import Language, Query

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.parsing.languages.salesforce import (
    parse_salesforce_metadata,
    xml_child_elements,
    xml_tag,
    xml_text,
)
from code_atlas.schema import NodeLabel, RelType

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tree_sitter import Node

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})

# Module `kind` per recognised dialect. A file matching nothing here falls back
# to the generic key-tree parse — see the module docstring.
_MODULE_KINDS: dict[str, str] = {
    "k8s": "k8s_manifest",
    "compose": "compose_file",
    "github_workflow": "github_workflow",
    "ansible_playbook": "ansible_playbook",
    "ansible_tasks": "ansible_tasks",
    "ansible_handlers": "ansible_handlers",
}

_QN_UNSAFE_RE = re.compile(r"[^0-9A-Za-z_-]+")

_MAX_WALK_DEPTH = 64
"""Depth ceiling for every walk over a *constructed* config object.

Belt to the cycle detectors' braces: an acyclic but pathologically nested
document (YAML flow style nests ~490 deep before the interpreter stack goes) is
bounded here instead of being left to ``parse_file``'s RecursionError catch,
which throws away the whole file. Real manifests nest well under 20.
"""


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert a file path to a dotted qualified name, dots folded in every segment.

    ``deploy/app.yaml``          -> ``deploy.app_yaml``
    ``charts/app.v2/values.yaml`` -> ``charts.app_v2.values_yaml``

    Two rules, both load-bearing, because ``qualified_name`` IS the graph uid and
    a uid collision means the later upsert silently overwrites the earlier node:

    1. The extension is *preserved* (its dot replaced) rather than stripped. One
       stem under several config formats is completely routine — ``openapi.json``
       beside ``openapi.yaml``, ``config.toml`` beside ``config.yml`` — and
       stripping would make them all claim ``{project}:openapi``.
    2. The dot is folded out of **every** path segment, not just the basename.
       ``.`` is this qualified name's separator, so a dot surviving inside a
       directory name fakes a separator: fold only the basename and ``a.b/X`` and
       ``a/b/X`` both render ``a.b.X``. Directories with dots in them are
       everywhere in real repos (``.github``, ``app.v2``, ``com.acme``), so this
       is a collision that actually happens.

    Folding is not injective — ``a.b/X`` and ``a_b/X`` still meet at ``a_b.X`` —
    and making it injective would need an escape scheme (``_`` -> ``__``) that
    disfigures every ordinary name for a collision far rarer than the one above.
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    return ".".join(part.replace(".", "_") for part in p.parts)


def _qn_segment(text: str) -> str:
    """Fold arbitrary config text into one safe dotted-qualified-name segment.

    Config object names are far freer than identifiers — ``restart nginx``,
    ``ghcr.io/acme/api``, ``Deployment/web``. A raw dot would fake a nesting
    level in the qualified name, so everything outside ``[0-9A-Za-z_-]``
    collapses to ``_``.
    """
    cleaned = _QN_UNSAFE_RE.sub("_", text.strip()).strip("_")
    return cleaned or "unnamed"


def _join_relative(base_dir: PurePosixPath, rel: str) -> str | None:
    """Resolve *rel* against *base_dir*, collapsing ``.`` and ``..``.

    Returns ``None`` for an absolute path or one that climbs out of the project
    root: a file outside the indexed tree has no node to point at, and emitting
    an IMPORTS edge for it would only mint a junk ExternalPackage stub.
    """
    if rel.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", rel):
        return None
    parts: list[str] = list(base_dir.parts)
    for segment in PurePosixPath(rel.replace("\\", "/")).parts:
        if segment in {"", "."}:
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(segment)
    return "/".join(parts) if parts else None


def _image_package(image: str) -> str | None:
    """Strip tag and digest from a container image reference.

    ``ghcr.io/acme/api:1.2.3`` -> ``ghcr.io/acme/api``. Returns ``None`` when the
    reference is templated (``${TAG}``, ``{{ .Values.image }}``) and therefore
    names nothing resolvable.
    """
    ref = image.strip()
    if not ref or "${" in ref or "{{" in ref:
        return None
    ref = ref.split("@", 1)[0]
    # A colon after the last slash is a tag; before it, it is a registry port.
    head, sep, tail = ref.rpartition("/")
    tail = tail.split(":", 1)[0]
    return f"{head}{sep}{tail}" if tail else None


# ---------------------------------------------------------------------------
# Emission accumulator
# ---------------------------------------------------------------------------


@dataclass
class _Out:
    """Entity/relationship accumulator with per-file qualified-name deduping.

    Duplicate qualified names inside one file silently collide on the same uid
    (a config file with two identically named tasks is ordinary), so the ``#N``
    suffix convention from ``markdown.py``'s section dedupe is applied here too.
    """

    file_path: str
    project_name: str
    text: str
    entities: list[ParsedEntity] = field(default_factory=list)
    relationships: list[ParsedRelationship] = field(default_factory=list)
    seen_qns: set[str] = field(default_factory=set)
    seen_rels: set[tuple[str, str, str]] = field(default_factory=set)
    newlines: list[int] = field(default_factory=list, init=False, repr=False)
    """Byte-offset of every ``\\n`` in ``text``, ascending — see ``end_line``."""

    def __post_init__(self) -> None:
        # One C-speed scan of the file, not one per entity. `str.find` in a loop
        # beats a per-character comprehension by an order of magnitude.
        offsets: list[int] = []
        pos = self.text.find("\n")
        while pos != -1:
            offsets.append(pos)
            pos = self.text.find("\n", pos + 1)
        self.newlines = offsets

    def add(
        self,
        *,
        name: str,
        qn_suffix: str,
        label: NodeLabel,
        kind: str,
        line_start: int,
        line_end: int,
        source: str | None = None,
        docstring: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Append an entity; returns its uid (the project-prefixed qualified name)."""
        qn = f"{self.project_name}:{qn_suffix}"
        if qn in self.seen_qns:
            counter = 2
            while f"{qn}#{counter}" in self.seen_qns:
                counter += 1
            qn = f"{qn}#{counter}"
        self.seen_qns.add(qn)
        self.entities.append(
            ParsedEntity(
                name=name,
                qualified_name=qn,
                label=label,
                kind=kind,
                line_start=line_start,
                line_end=max(line_start, line_end),
                file_path=self.file_path,
                docstring=docstring,
                source=source,
                extra_properties=extra or {},
            )
        )
        return qn

    def rel(self, from_uid: str, rel_type: RelType, to_name: str) -> None:
        """Append a relationship, ignoring exact duplicates."""
        key = (from_uid, rel_type.value, to_name)
        if key in self.seen_rels:
            return
        self.seen_rels.add(key)
        self.relationships.append(ParsedRelationship(from_qualified_name=from_uid, rel_type=rel_type, to_name=to_name))

    def lines(self, start: yaml.Node, end: yaml.Node | None = None) -> tuple[int, int]:
        """1-based inclusive line span covering *start* through *end*."""
        end_node = start if end is None else end
        return start.start_mark.line + 1, self.end_line(end_node)

    def end_line(self, node: yaml.Node) -> int:
        """1-based inclusive last line of *node*.

        PyYAML's end mark sits at the first character *after* the node, which for
        a block collection is the indentation of the *following* sibling — so the
        raw mark line overshoots by one whenever the node ends a line. Backing up
        over trailing whitespace lands on the node's real last character.

        The line number then comes from a binary search of the precomputed
        newline index. It used to come from ``text.count("\\n", 0, end)``, which
        rescans the file from byte zero on *every* entity — O(entities x
        filesize), and a config file's entity count grows with its size, so that
        term is quadratic in the one dimension that matters.
        """
        end = node.end_mark.index
        while end > node.start_mark.index and self.text[end - 1].isspace():
            end -= 1
        return bisect_right(self.newlines, end - 1) + 1

    def slice(self, start: yaml.Node, end: yaml.Node | None = None) -> str:
        """Raw source text covering *start* through *end*."""
        end_node = start if end is None else end
        return self.text[start.start_mark.index : end_node.end_mark.index].rstrip()


def _suffix(uid: str) -> str:
    """The qualified name inside a uid, i.e. the uid minus its ``project:`` prefix."""
    return uid.split(":", 1)[1] if ":" in uid else uid


# ---------------------------------------------------------------------------
# YAML loading (structure + line marks in one pass)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Doc:
    """One document of a YAML stream: the constructed value plus its composer node."""

    data: Any
    node: yaml.Node


def _load_yaml_documents(text: str) -> list[_Doc] | None:
    """Load every document in a YAML stream, keeping composer nodes for line marks.

    ``yaml.safe_load_all`` throws the nodes away, and the nodes are the only
    place line/byte marks live — hence the explicit loader. ``check_node`` /
    ``get_node`` / ``construct_document`` / ``dispose`` are precisely the calls
    ``yaml.load_all`` itself makes; this only interposes to keep the node.

    Returns ``None`` for a stream that is not valid YAML at all — the caller
    declines the file rather than indexing a partial parse.

    The loader is constructed **inside** the ``try``. ``yaml.SafeLoader(text)``
    runs ``Reader.check_printable`` in its own ``__init__`` and raises
    ``ReaderError`` — a ``YAMLError`` subclass — for a single non-printable
    character anywhere in the file. Built outside the guard, that escapes
    ``parse_file`` and kills the whole AST batch over one stray control byte in
    one file.
    """
    docs: list[_Doc] = []
    loader: yaml.SafeLoader | None = None
    try:
        loader = yaml.SafeLoader(text)
        while loader.check_node():
            node = loader.get_node()
            if node is None:
                continue
            docs.append(_Doc(data=loader.construct_document(node), node=node))
    except yaml.YAMLError:
        return None
    finally:
        if loader is not None:
            loader.dispose()
    return docs


def _map_entries(node: yaml.Node) -> list[tuple[yaml.Node, yaml.Node]]:
    """The ``(key, value)`` node pairs of a mapping node, or ``[]``."""
    return list(node.value) if isinstance(node, yaml.MappingNode) else []


def _entry_node(node: yaml.Node, key: str) -> tuple[yaml.Node, yaml.Node] | None:
    """The ``(key, value)`` node pair for *key*, matched on the raw scalar text.

    Raw text, not the constructed value: PyYAML implements YAML 1.1, so a bare
    ``on`` key constructs as the boolean ``True`` while its scalar node still
    reads ``"on"``.
    """
    for key_node, value_node in _map_entries(node):
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return key_node, value_node
    return None


def _entry_nodes_by_key(node: yaml.Node | None) -> dict[str, tuple[yaml.Node, yaml.Node]]:
    """Index a mapping node's entries by raw key text, for line lookups."""
    if node is None:
        return {}
    return {k.value: (k, v) for k, v in _map_entries(node) if isinstance(k, yaml.ScalarNode)}


def _seq_nodes(node: yaml.Node | None) -> list[yaml.Node]:
    """The item nodes of a sequence node, or ``[]``."""
    if node is None or not isinstance(node, yaml.SequenceNode):
        return []
    return list(node.value)


def _dig(data: Any, path: tuple[str, ...]) -> Any:
    """Follow a chain of mapping keys, returning ``None`` at the first miss."""
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _as_str_list(value: Any) -> list[str]:
    """Normalise the ubiquitous ``scalar | [scalar, ...]`` config shape."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


# ---------------------------------------------------------------------------
# Dialect detection
#
# Reliable structural signals first, path convention only as a secondary guard.
# ---------------------------------------------------------------------------

_WORKFLOW_DIRS = frozenset({".github", ".gitea", ".forgejo"})


def _is_k8s(data: Any) -> bool:
    """``apiVersion`` + ``kind`` + a named ``metadata`` — the one path-free marker.

    Kubernetes is the asymmetric case in config detection: it has no filename
    convention at all (SchemaStore has no ``fileMatch`` for it), and it does not
    need one. ``metadata.name`` is required on top of the two obvious keys
    because it is the node's identity — without it there is nothing to build a
    uid from, and things like ``kustomization.yaml`` (apiVersion + kind, no
    metadata) correctly fall through to "unrecognised".
    """
    if not isinstance(data, dict):
        return False
    metadata = data.get("metadata")
    return (
        isinstance(data.get("apiVersion"), str)
        and isinstance(data.get("kind"), str)
        and isinstance(metadata, dict)
        and isinstance(metadata.get("name"), str)
    )


def _is_compose(data: Any) -> bool:
    """Top-level ``services`` mapping, corroborated by ``version`` or a real service."""
    if not isinstance(data, dict):
        return False
    services = data.get("services")
    if not isinstance(services, dict) or not services:
        return False
    if "version" in data:
        return True
    return any(isinstance(spec, dict) and ("image" in spec or "build" in spec) for spec in services.values())


def _in_workflows_dir(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return any(parts[i] in _WORKFLOW_DIRS and parts[i + 1] == "workflows" for i in range(len(parts) - 1))


def _is_workflow(path: str, data: Any) -> bool:
    """Top-level ``jobs`` mapping plus either the workflows path or a trigger key.

    ``on:`` is the trigger key, and PyYAML implements YAML 1.1 — a bare ``on``
    constructs as the boolean ``True``, so both spellings are probed.
    """
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), dict):
        return False
    return _in_workflows_dir(path) or True in data or "on" in data


def _is_ansible_playbook(data: Any) -> bool:
    """Top-level sequence with a ``hosts`` mapping in it.

    ``hosts:`` is the only reliable playbook marker. ``tasks:`` alone is not
    (a tasks *file* has no such key), and ``roles:`` alone is not either.
    """
    return isinstance(data, list) and any(isinstance(item, dict) and "hosts" in item for item in data)


def _ansible_task_file_kind(path: str) -> str | None:
    """``tasks/`` or ``handlers/`` parent directory — the load-bearing secondary signal.

    An Ansible tasks file carries no self-identifying marker whatsoever; it is a
    plain sequence of mappings, structurally identical to any other list-of-
    objects YAML in any repo. Without the directory constraint this rule
    false-positives on arbitrary data files, so a miss is preferred over a false
    positive: no ``tasks/`` or ``handlers/`` parent, no Ansible interpretation.
    """
    parent = PurePosixPath(path).parent.name
    if parent == "tasks":
        return "ansible_tasks"
    if parent == "handlers":
        return "ansible_handlers"
    return None


def _document_dialect(path: str, data: Any) -> str | None:
    """Classify one document. Order matters: content markers before conventions."""
    if _is_k8s(data):
        return "k8s"
    if _is_compose(data):
        return "compose"
    if _is_workflow(path, data):
        return "github_workflow"
    if _is_ansible_playbook(data):
        return "ansible_playbook"
    if isinstance(data, list) and data and any(isinstance(item, dict) for item in data):
        return _ansible_task_file_kind(path)
    return None


def _is_opaque(path: str, source: bytes) -> bool:
    """Content that must never reach the YAML loader at all.

    Vault-encrypted files are ciphertext, and Helm templates under
    ``templates/`` are Go templates that are routinely invalid YAML before
    rendering (conditionally emitted list items, unbalanced indentation). This
    gate has to run *before* the load, not as an exception handler around it:
    a mis-parse that happens to succeed is worse than a skip.
    """
    if source.lstrip().startswith(b"$ANSIBLE_VAULT;"):
        return True
    return "templates" in PurePosixPath(path).parts and b"{{" in source


# ---------------------------------------------------------------------------
# Kubernetes
# ---------------------------------------------------------------------------

# Reference fields that are a nested mapping: key -> (candidate name keys,
# target kind, or None to read `kind` from the mapping itself).
_K8S_MAPPING_REFS: dict[str, tuple[tuple[str, ...], str | None]] = {
    "configMap": (("name",), "ConfigMap"),
    "configMapRef": (("name",), "ConfigMap"),
    "configMapKeyRef": (("name",), "ConfigMap"),
    # Volume form is `secret.secretName`, projected-source form is `secret.name`.
    "secret": (("secretName", "name"), "Secret"),
    "secretRef": (("name",), "Secret"),
    "secretKeyRef": (("name",), "Secret"),
    "persistentVolumeClaim": (("claimName",), "PersistentVolumeClaim"),
    "roleRef": (("name",), None),
    "scaleTargetRef": (("name",), None),
    # Ingress backend: spec.rules[].http.paths[].backend.service.name
    "service": (("name",), "Service"),
}

# Reference fields that are a bare scalar: key -> target kind.
_K8S_SCALAR_REFS: dict[str, str] = {
    "serviceAccountName": "ServiceAccount",
    "secretName": "Secret",
    "storageClassName": "StorageClass",
    "ingressClassName": "IngressClass",
}

# Reference fields that are a list of mappings: key -> (name key, target kind).
_K8S_LIST_REFS: dict[str, tuple[str, str | None]] = {
    "imagePullSecrets": ("name", "Secret"),
    "subjects": ("name", None),
}

# Where a workload's pod-template labels live, most specific first.
_K8S_POD_LABEL_PATHS: tuple[tuple[str, ...], ...] = (
    ("spec", "template", "metadata", "labels"),
    ("spec", "jobTemplate", "spec", "template", "metadata", "labels"),
)

# Only these kinds select *other* objects. A Deployment's own
# `spec.selector.matchLabels` targets the pods it owns, so including workload
# kinds here would emit a self-edge for every Deployment in the repo.
_K8S_SELECTOR_KINDS = frozenset({"Service", "NetworkPolicy", "PodDisruptionBudget"})
_K8S_SELECTOR_PATHS: tuple[tuple[str, ...], ...] = (
    ("spec", "selector"),
    ("spec", "selector", "matchLabels"),
    ("spec", "podSelector", "matchLabels"),
)


@dataclass(frozen=True)
class _K8sResource:
    """A k8s resource entity plus the label sets needed for the selector join."""

    uid: str
    name: str
    pod_labels: dict[str, str]
    selector_labels: dict[str, str]


def _flat_labels(value: Any) -> dict[str, str]:
    """A mapping of scalars rendered as ``str``, or ``{}`` if it is not flat.

    Non-flatness is itself a signal: ``Service.spec.selector`` is flat, whereas
    ``Deployment.spec.selector`` is ``{matchLabels: {...}}`` and correctly yields
    nothing here.
    """
    if not isinstance(value, dict) or not value:
        return {}
    labels: dict[str, str] = {}
    for key, val in value.items():
        if not isinstance(key, str) or isinstance(val, (dict, list)) or val is None:
            return {}
        labels[key] = str(val)
    return labels


def _pod_labels(data: dict[str, Any]) -> dict[str, str]:
    for path in _K8S_POD_LABEL_PATHS:
        labels = _flat_labels(_dig(data, path))
        if labels:
            return labels
    if data.get("kind") == "Pod":
        return _flat_labels(_dig(data, ("metadata", "labels")))
    return {}


def _selector_labels(data: dict[str, Any]) -> dict[str, str]:
    if data.get("kind") not in _K8S_SELECTOR_KINDS:
        return {}
    for path in _K8S_SELECTOR_PATHS:
        labels = _flat_labels(_dig(data, path))
        if labels:
            return labels
    return {}


def _ref_from_mapping(mapping: dict[str, Any], name_keys: tuple[str, ...], kind: str | None) -> str | None:
    target_kind = kind if kind is not None else mapping.get("kind")
    if not isinstance(target_kind, str) or not target_kind:
        return None
    for name_key in name_keys:
        name = mapping.get(name_key)
        if isinstance(name, str) and name:
            return f"{target_kind}/{name}"
    return None


def _k8s_key_refs(key: str, value: Any, refs: list[str]) -> None:
    """Apply the three reference-field tables to one mapping entry."""
    scalar_kind = _K8S_SCALAR_REFS.get(key)
    if scalar_kind is not None and isinstance(value, str) and value:
        refs.append(f"{scalar_kind}/{value}")
    mapping_rule = _K8S_MAPPING_REFS.get(key)
    if mapping_rule is not None and isinstance(value, dict):
        ref = _ref_from_mapping(value, *mapping_rule)
        if ref is not None:
            refs.append(ref)
    list_rule = _K8S_LIST_REFS.get(key)
    if list_rule is not None and isinstance(value, list):
        for item in value:
            ref = _ref_from_mapping(item, (list_rule[0],), list_rule[1]) if isinstance(item, dict) else None
            if ref is not None:
                refs.append(ref)


def _walk_k8s(node: Any, refs: list[str], images: list[str], *, seen: set[int] | None = None, depth: int = 0) -> None:
    """Collect ``Kind/name`` references and container images from a resource body.

    ``seen``/``depth`` are not defensive padding. ``spec: &s {child: *s}`` is
    legal YAML, and ``SafeLoader`` genuinely constructs the resulting dict with
    itself inside it (verified: ``data["spec"]["child"] is data["spec"]``), so an
    unguarded walk of the constructed object never terminates.

    ``seen`` holds every container visited by *this* walk, not just the ones on
    the current path. A path-scoped set would still terminate, but a diamond of
    aliases would then be re-walked once per path — exponential in depth. Global
    is safe here because an alias is the *same object*: re-walking it can only
    re-collect refs the ``_Out.rel`` dedupe would drop anyway. Nothing in the
    constructed document can be freed mid-walk, so ``id()`` cannot be recycled.
    """
    if seen is None:
        seen = set()
    if depth >= _MAX_WALK_DEPTH:
        return
    if isinstance(node, (dict, list)):
        if id(node) in seen:
            return
        seen.add(id(node))
    if isinstance(node, list):
        for item in node:
            _walk_k8s(item, refs, images, seen=seen, depth=depth + 1)
        return
    if not isinstance(node, dict):
        return
    image = node.get("image")
    # The `name` sibling is what keeps a ConfigMap `data:` payload that happens
    # to carry an `image` key from being read as a container spec.
    if isinstance(image, str) and isinstance(node.get("name"), str):
        images.append(image)
    for key, value in node.items():
        if isinstance(key, str):
            _k8s_key_refs(key, value, refs)
        _walk_k8s(value, refs, images, seen=seen, depth=depth + 1)


def _extract_k8s(
    out: _Out,
    data: dict[str, Any],
    *,
    module_uid: str,
    line_start: int,
    line_end: int,
    source: str,
) -> _K8sResource:
    """Emit one k8s resource entity plus its name-reference and image edges."""
    kind = data["kind"]
    metadata = data["metadata"]
    resource_name = metadata["name"]
    name = f"{kind}/{resource_name}"
    namespace = metadata.get("namespace")

    extra: dict[str, Any] = {
        "api_version": data["apiVersion"],
        "k8s_kind": kind,
        "resource_name": resource_name,
    }
    if isinstance(namespace, str) and namespace:
        extra["namespace"] = namespace

    uid = out.add(
        name=name,
        qn_suffix=f"{_suffix(module_uid)}.{_qn_segment(kind)}_{_qn_segment(resource_name)}",
        label=NodeLabel.TYPE_DEF,
        kind="k8s_resource",
        line_start=line_start,
        line_end=line_end,
        source=source,
        extra=extra,
    )
    out.rel(module_uid, RelType.DEFINES, uid)

    refs: list[str] = []
    images: list[str] = []
    _walk_k8s(data, refs, images)
    for ref in refs:
        if ref != name:
            out.rel(uid, RelType.USES_TYPE, ref)
    for image in images:
        package = _image_package(image)
        if package is not None:
            out.rel(uid, RelType.IMPORTS, package)

    return _K8sResource(
        uid=uid,
        name=name,
        pod_labels=_pod_labels(data),
        selector_labels=_selector_labels(data),
    )


def _link_k8s_selectors(out: _Out, resources: list[_K8sResource]) -> None:
    """Join label selectors against pod-template labels, within this file.

    Set containment, not a name reference — the signature Kubernetes edge. Every
    match is emitted rather than one winner, mirroring ADR-0014's ambiguous-CALLS
    decision. Resolution is intra-file only: a cross-file join needs every
    document in the batch and therefore a post-batch resolver, which would mean
    a new RelType.
    """
    workloads = [r for r in resources if r.pod_labels]
    for source in resources:
        if not source.selector_labels:
            continue
        for workload in workloads:
            if workload.uid == source.uid:
                continue
            if all(workload.pod_labels.get(key) == value for key, value in source.selector_labels.items()):
                out.rel(source.uid, RelType.USES_TYPE, workload.name)


# ---------------------------------------------------------------------------
# docker-compose
# ---------------------------------------------------------------------------


def _compose_depends_on(spec: dict[str, Any]) -> list[str]:
    """Both ``depends_on`` forms: the short list and the long condition map."""
    value = spec.get("depends_on")
    if isinstance(value, dict):
        return [key for key in value if isinstance(key, str)]
    return _as_str_list(value)


def _compose_build_target(base_dir: PurePosixPath, build: Any) -> str | None:
    """Resolve ``build`` to the Containerfile path it names, relative to the repo.

    ``build.dockerfile`` is relative to ``build.context``, which is itself
    relative to the compose file. A remote (git/URL) context names nothing in
    this repo.
    """
    if isinstance(build, str):
        context, dockerfile = build, "Dockerfile"
    elif isinstance(build, dict):
        context = build.get("context", ".")
        dockerfile = build.get("dockerfile", "Dockerfile")
        if not isinstance(context, str) or not isinstance(dockerfile, str):
            return None
    else:
        return None
    if "://" in context or context.startswith("git@"):
        return None
    return _join_relative(base_dir, f"{context}/{dockerfile}")


def _extract_compose(out: _Out, doc: _Doc, module_uid: str) -> None:
    """Emit one entity per service, with depends_on / image / build edges."""
    services = doc.data["services"]
    service_nodes = _entry_nodes_by_key(_pair_value(_entry_node(doc.node, "services")))
    base_dir = PurePosixPath(out.file_path).parent

    for service_name, spec in services.items():
        if not isinstance(service_name, str):
            continue
        line_start, line_end, source = _entry_position(out, service_nodes.get(service_name), doc)
        uid = out.add(
            name=service_name,
            qn_suffix=f"{_suffix(module_uid)}.{_qn_segment(service_name)}",
            label=NodeLabel.TYPE_DEF,
            kind="compose_service",
            line_start=line_start,
            line_end=line_end,
            source=source,
        )
        out.rel(module_uid, RelType.DEFINES, uid)
        if not isinstance(spec, dict):
            continue
        for dependency in _compose_depends_on(spec):
            out.rel(uid, RelType.USES_TYPE, dependency)
        image = spec.get("image")
        if isinstance(image, str):
            package = _image_package(image)
            # With `build:` also present, `image:` is the *output* tag rather
            # than an input, but it is still the name other manifests reference.
            if package is not None:
                out.rel(uid, RelType.IMPORTS, package)
        target = _compose_build_target(base_dir, spec.get("build"))
        if target is not None:
            out.rel(uid, RelType.IMPORTS, _module_qualified_name(target))


# ---------------------------------------------------------------------------
# GitHub Actions
# ---------------------------------------------------------------------------


def _workflow_action_refs(spec: dict[str, Any]) -> list[str]:
    """External ``uses:`` references on the job and on each of its steps.

    ``./local-action`` and ``docker://`` forms are skipped: the former names a
    directory rather than a file (so there is no qualified name to match), and
    the latter is not an action at all.
    """
    candidates = [spec.get("uses")]
    steps = spec.get("steps")
    if isinstance(steps, list):
        candidates += [step.get("uses") for step in steps if isinstance(step, dict)]
    refs: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        ref = candidate.split("@", 1)[0].strip()
        if "/" not in ref or ref.startswith(".") or "://" in ref:
            continue
        refs.append(ref)
    return refs


def _extract_workflow(out: _Out, doc: _Doc, module_uid: str) -> None:
    """Emit one Callable per job, with ``needs:`` and ``uses:`` edges."""
    jobs = doc.data["jobs"]
    job_nodes = _entry_nodes_by_key(_pair_value(_entry_node(doc.node, "jobs")))

    for job_id, spec in jobs.items():
        if not isinstance(job_id, str):
            continue
        line_start, line_end, source = _entry_position(out, job_nodes.get(job_id), doc)
        display_name = spec.get("name") if isinstance(spec, dict) else None
        uid = out.add(
            name=job_id,
            qn_suffix=f"{_suffix(module_uid)}.{_qn_segment(job_id)}",
            label=NodeLabel.CALLABLE,
            kind="ci_job",
            line_start=line_start,
            line_end=line_end,
            source=source,
            docstring=display_name if isinstance(display_name, str) else None,
        )
        out.rel(module_uid, RelType.DEFINES, uid)
        if not isinstance(spec, dict):
            continue
        for need in _as_str_list(spec.get("needs")):
            out.rel(uid, RelType.CALLS, need)
        for action in _workflow_action_refs(spec):
            out.rel(uid, RelType.IMPORTS, action)


# ---------------------------------------------------------------------------
# Ansible
# ---------------------------------------------------------------------------

_ANSIBLE_TASK_LISTS: tuple[str, ...] = ("pre_tasks", "tasks", "post_tasks", "handlers")
_ANSIBLE_BLOCK_KEYS: tuple[str, ...] = ("block", "rescue", "always")
_ANSIBLE_INCLUDE_KEYS = frozenset({"include_tasks", "import_tasks", "include"})
_ANSIBLE_ROLE_KEYS = frozenset({"include_role", "import_role"})
_ROLE_ENTRY_FILES = frozenset({"main.yml", "main.yaml"})


def _role_dir(path: str) -> str | None:
    """The role directory for ``roles/<name>/tasks/main.yml``, else ``None``.

    Only the role's tasks entry point mints the role node. Every other file in
    the role (``handlers/main.yml``, ``defaults/main.yml``, ...) would claim the
    same uid, and the last one upserted would silently win.
    """
    p = PurePosixPath(path)
    if p.name not in _ROLE_ENTRY_FILES:
        return None
    parts = p.parts
    if len(parts) < 4 or parts[-2] != "tasks" or parts[-4] != "roles":
        return None
    return "/".join(parts[:-2])


def _ansible_role_names(value: Any) -> list[str]:
    """Role names from a play's ``roles:`` list (bare names and ``{role: x}`` maps)."""
    names: list[str] = []
    if not isinstance(value, list):
        return names
    for item in value:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            for key in ("role", "name"):
                candidate = item.get(key)
                if isinstance(candidate, str):
                    names.append(candidate)
                    break
    return names


def _task_directives(task: dict[str, Any], names: frozenset[str]) -> list[Any]:
    """Values of a task's *names* keys, accepting the fully-qualified spelling.

    A task's module key is written either short (``include_tasks:``) or as an
    FQCN (``ansible.builtin.include_tasks:``), and both are ubiquitous in real
    playbooks — ansible-lint's ``fqcn`` rule actively pushes projects toward the
    long form. Matching on the last dotted segment covers both without a
    collection lookup table.
    """
    return [value for key, value in task.items() if isinstance(key, str) and key.rsplit(".", 1)[-1] in names]


def _task_include_target(task: dict[str, Any]) -> str | None:
    """The task file named by ``include_tasks`` / ``import_tasks`` / legacy ``include``."""
    for value in _task_directives(task, _ANSIBLE_INCLUDE_KEYS):
        if isinstance(value, str):
            return value
        if isinstance(value, dict) and isinstance(value.get("file"), str):
            return value["file"]
    return None


def _task_role_names(task: dict[str, Any]) -> list[str]:
    """Role names from ``include_role`` / ``import_role``.

    Note that ``tasks_from:`` retargets the include at a non-``main`` task file
    inside the role; the role node stays the right anchor either way, and
    modelling the retarget would need a node this parser does not mint.
    """
    names: list[str] = []
    for value in _task_directives(task, _ANSIBLE_ROLE_KEYS):
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, dict) and isinstance(value.get("name"), str):
            names.append(value["name"])
    return names


def _extract_tasks(
    out: _Out,
    tasks: Any,
    seq_node: yaml.Node | None,
    *,
    parent_uid: str,
    rel_type: RelType,
    kind: str,
) -> None:
    """Emit one Callable per *named* task, plus notify / include / role edges.

    Unnamed tasks get no node: their only available identity would be positional,
    and a positional uid churns the whole file's graph every time a task is
    inserted above it. Their nested ``block``/``rescue``/``always`` children are
    still visited, since those may well be named.
    """
    if not isinstance(tasks, list):
        return
    item_nodes = _seq_nodes(seq_node)
    base_dir = PurePosixPath(out.file_path).parent

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        node = item_nodes[index] if index < len(item_nodes) else None
        name = task.get("name")
        if not isinstance(name, str) or not name.strip():
            _extract_task_blocks(out, task, node, parent_uid=parent_uid, rel_type=rel_type, kind=kind)
            continue

        line_start, line_end = out.lines(node) if node is not None else (1, 1)
        uid = out.add(
            name=name,
            qn_suffix=f"{_suffix(parent_uid)}.{_qn_segment(name)}",
            label=NodeLabel.CALLABLE,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            source=out.slice(node) if node is not None else None,
        )
        out.rel(parent_uid, rel_type, uid)

        for handler in _as_str_list(task.get("notify")):
            out.rel(uid, RelType.CALLS, handler)
        for role in _task_role_names(task):
            out.rel(uid, RelType.USES_TYPE, role)
        include = _task_include_target(task)
        if include is not None and "{{" not in include:
            target = _join_relative(base_dir, include)
            if target is not None:
                out.rel(uid, RelType.IMPORTS, _module_qualified_name(target))
        _extract_task_blocks(out, task, node, parent_uid=uid, rel_type=RelType.DEFINES, kind=kind)


def _extract_task_blocks(
    out: _Out,
    task: dict[str, Any],
    node: yaml.Node | None,
    *,
    parent_uid: str,
    rel_type: RelType,
    kind: str,
) -> None:
    """Recurse into a task's ``block`` / ``rescue`` / ``always`` sub-lists."""
    for key in _ANSIBLE_BLOCK_KEYS:
        if key not in task:
            continue
        sub_node = _pair_value(_entry_node(node, key)) if node is not None else None
        _extract_tasks(out, task[key], sub_node, parent_uid=parent_uid, rel_type=rel_type, kind=kind)


def _extract_play(out: _Out, play: dict[str, Any], node: yaml.Node | None, module_uid: str) -> None:
    """Emit one Callable per play, its role references and its task lists."""
    hosts = play.get("hosts")
    display_name = play.get("name")
    name = display_name if isinstance(display_name, str) and display_name.strip() else f"hosts:{hosts}"
    line_start, line_end = out.lines(node) if node is not None else (1, 1)

    extra: dict[str, Any] = {}
    if isinstance(hosts, (str, int, bool)):
        extra["hosts"] = str(hosts)

    uid = out.add(
        name=name,
        qn_suffix=f"{_suffix(module_uid)}.{_qn_segment(name)}",
        label=NodeLabel.CALLABLE,
        kind="ansible_play",
        line_start=line_start,
        line_end=line_end,
        source=out.slice(node) if node is not None else None,
        extra=extra,
    )
    out.rel(module_uid, RelType.DEFINES, uid)

    for role in _ansible_role_names(play.get("roles")):
        out.rel(uid, RelType.USES_TYPE, role)
    for list_key in _ANSIBLE_TASK_LISTS:
        if list_key not in play:
            continue
        seq_node = _pair_value(_entry_node(node, list_key)) if node is not None else None
        _extract_tasks(
            out,
            play[list_key],
            seq_node,
            parent_uid=uid,
            rel_type=RelType.DEFINES,
            kind="ansible_handler" if list_key == "handlers" else "ansible_task",
        )


def _extract_playbook(out: _Out, doc: _Doc, module_uid: str) -> None:
    """Emit plays and ``import_playbook`` edges from a playbook document."""
    item_nodes = _seq_nodes(doc.node)
    base_dir = PurePosixPath(out.file_path).parent

    for index, item in enumerate(doc.data):
        if not isinstance(item, dict):
            continue
        node = item_nodes[index] if index < len(item_nodes) else None
        if "hosts" in item:
            _extract_play(out, item, node, module_uid)
        elif isinstance(item.get("import_playbook"), str):
            target = _join_relative(base_dir, item["import_playbook"])
            if target is not None:
                out.rel(module_uid, RelType.IMPORTS, _module_qualified_name(target))


def _extract_task_file(out: _Out, doc: _Doc, module_uid: str, kind: str) -> None:
    """Emit tasks (or handlers) from a ``tasks/``-or-``handlers/`` file."""
    _extract_tasks(
        out,
        doc.data,
        doc.node,
        parent_uid=module_uid,
        rel_type=RelType.DEFINES,
        kind="ansible_handler" if kind == "ansible_handlers" else "ansible_task",
    )


def _emit_role(out: _Out, module_uid: str, line_end: int) -> None:
    """Mint the ``ansible_role`` node for a role's ``tasks/main.yml``.

    The role is a *directory*, so its uid keys on the directory rather than the
    file. It exists to be the target of ``roles:`` / ``include_role`` references,
    which resolve by name through ``resolve_type_refs``.
    """
    role_dir = _role_dir(out.file_path)
    if role_dir is None:
        return
    role_uid = out.add(
        name=PurePosixPath(role_dir).name,
        qn_suffix=_module_qualified_name(role_dir),
        label=NodeLabel.TYPE_DEF,
        kind="ansible_role",
        line_start=1,
        line_end=line_end,
    )
    out.rel(role_uid, RelType.CONTAINS, module_uid)


# ---------------------------------------------------------------------------
# Shared node/line plumbing
# ---------------------------------------------------------------------------


def _pair_value(pair: tuple[yaml.Node, yaml.Node] | None) -> yaml.Node | None:
    return pair[1] if pair is not None else None


def _entry_position(out: _Out, pair: tuple[yaml.Node, yaml.Node] | None, doc: _Doc) -> tuple[int, int, str]:
    """Line span and source text for a mapping entry, keyed from its *key* node.

    Falls back to the whole document when the composer pair is missing (a merge
    key or an anchor can put a constructed key in the data with no node of its
    own).
    """
    if pair is None:
        line_start, line_end = out.lines(doc.node)
        return line_start, line_end, out.slice(doc.node)
    key_node, value_node = pair
    line_start, line_end = out.lines(key_node, value_node)
    return line_start, line_end, out.slice(key_node, value_node)


# ---------------------------------------------------------------------------
# XML
# ---------------------------------------------------------------------------


def _parse_xml(path: str, root: Node, project_name: str) -> ParsedFile | None:
    """Salesforce metadata if the document is any, otherwise a minimal structural parse.

    ``salesforce.parse_salesforce_metadata`` claims the SFDX metadata types it
    models (CustomObject, CustomField, Flow, CustomLabels, CustomMetadata) and
    declines everything else, which lands here.

    The fallback is deliberately one level deep: the root element is the
    document type and its direct children are the settings worth searching for
    (``<status>``, ``<label>``, ``<isExposed>``), whereas walking the whole tree
    of a large FlexiPage or permission set would mint hundreds of nodes per file
    for no additional answerable question.  The element-tree primitives it uses
    live in ``salesforce.py`` so that the import between the two modules runs in
    exactly one direction.
    """
    parsed = parse_salesforce_metadata(path, root, project_name)
    if parsed is not None:
        return parsed

    element = next((child for child in root.children if child.type == "element"), None)
    tag = xml_tag(element) if element is not None else None
    if element is None or tag is None:
        return None

    out = _Out(file_path=path, project_name=project_name, text="")
    module_uid = out.add(
        name=PurePosixPath(path).name,
        qn_suffix=_module_qualified_name(path),
        label=NodeLabel.MODULE,
        kind="xml_document",
        line_start=1,
        line_end=root.end_point[0] + 1,
    )
    root_uid = out.add(
        name=tag,
        qn_suffix=f"{_suffix(module_uid)}.{_qn_segment(tag)}",
        label=NodeLabel.TYPE_DEF,
        kind="xml_element",
        line_start=element.start_point[0] + 1,
        line_end=element.end_point[0] + 1,
    )
    out.rel(module_uid, RelType.DEFINES, root_uid)

    for child in xml_child_elements(element):
        child_tag = xml_tag(child)
        if child_tag is None:
            continue
        has_children = bool(xml_child_elements(child))
        child_uid = out.add(
            name=child_tag,
            qn_suffix=f"{_suffix(root_uid)}.{_qn_segment(child_tag)}",
            label=NodeLabel.TYPE_DEF if has_children else NodeLabel.VALUE,
            kind="xml_element" if has_children else "xml_setting",
            line_start=child.start_point[0] + 1,
            line_end=child.end_point[0] + 1,
            source=None if has_children else xml_text(child),
        )
        out.rel(root_uid, RelType.DEFINES, child_uid)

    return ParsedFile(file_path=path, language="xml", entities=out.entities, relationships=out.relationships)


# ---------------------------------------------------------------------------
# Generic config — the fallback for anything no dialect claims
# ---------------------------------------------------------------------------

MAX_GENERIC_CONFIG_BYTES = 256 * 1024
"""Size ceiling for the generic fallback; above it a file is read as data.

Configuration people write by hand does not reach a quarter of a megabyte. What
does is generated: lockfiles, fixtures, API dumps, translation bundles. This is
the cheapest rung of ``_looks_like_data`` (a ``len()``) and the one that keeps
the expensive rungs off the biggest inputs, so it is checked before them.

Module-level rather than a ``Settings`` field: wiring it into ``atlas.toml``
means editing ``settings.py``, which this module does not own. Override it by
assignment if a repo genuinely needs to.
"""

_DATA_SUFFIXES = frozenset({".jsonl", ".ndjson"})
"""Extensions that are record streams by definition — rejected on the name alone.

Deliberately *not* in the JSON registration below. Registering an extension puts
it in the indexer's scan scope (``_DEFAULT_INCLUDE`` in ``indexing/orchestrator``
has a test asserting the two registries agree), so registering these would make
the indexer read every log file in the repo purely to decline it. Leaving them
unregistered is the stronger rejection: such files are never opened at all.

The check stays here because it is the format's contract, not the registry's: a
handler asked about a record stream must say no whichever route the path took.
"""

# `_looks_like_data` thresholds — see its docstring for what each one catches.
_DATA_LIST_LEN_MAX = 50
_DATA_LIST_HOMOGENEOUS_MIN = 10
_DATA_KEY_TOTAL_MIN = 200
_DATA_KEY_DISTINCT_RATIO = 0.15

_GENERIC_MAX_DEPTH = 3
"""Key-tree levels emitted below the Module node.

Three covers the shapes people actually search for (``[tool.ruff.lint]
select``, ``logging.handlers.console.level``) and stops a deeply nested blob
from minting a node per leaf.
"""

_GENERIC_MAX_ENTITIES = 200
"""Hard cap on nodes minted for one generic file, Module node included."""

_GENERIC_MODULE_KIND = "config_file"
_GENERIC_SECTION_KIND = "config_section"
_GENERIC_SETTING_KIND = "config_setting"


def _key_stats(data: Any) -> tuple[int, int]:
    """``(total mapping-key occurrences, distinct key names)`` over the whole tree.

    Iterative with a visited set, for the same reason ``_walk_k8s`` has one: a
    self-referential YAML anchor constructs a cyclic object, and this function
    runs on documents no dialect vouched for.
    """
    total = 0
    distinct: set[str] = set()
    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(data, 0)]
    while stack:
        node, depth = stack.pop()
        if depth >= _MAX_WALK_DEPTH or not isinstance(node, (dict, list)) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str):
                    total += 1
                    distinct.add(key)
                stack.append((value, depth + 1))
        else:
            stack.extend((item, depth + 1) for item in node)
    return total, len(distinct)


def _looks_like_data(data: Any) -> bool:
    """Is this a record dump rather than configuration?

    The ladder, cheapest and most certain first (the size and parseability rungs
    live in the callers, which hold the bytes):

    1. A top-level **array**. ``>= 50`` items is a dump outright. Below that,
       ``>= 10`` items that are *all* mappings and share at most
       ``max(2, n / 20)`` distinct key sets is a table: real config lists are
       short, or heterogeneous, or scalars. Anything else is config.
    2. A top-level **scalar** — a document with no structure to extract.
    3. A **mapping** with ``>= 200`` key occurrences of which fewer than 15% are
       distinct. This is the strongest signal in the ladder and the only one that
       sees *nested* data: ``{"rows": [...10k records...]}`` has a top level of
       one key and would sail past every array check, but its key census is the
       same handful of field names repeated ten thousand times. Hand-written
       config is the opposite shape — almost every key occurs once.
    """
    if isinstance(data, list):
        count = len(data)
        if count >= _DATA_LIST_LEN_MAX:
            return True
        if count >= _DATA_LIST_HOMOGENEOUS_MIN and all(isinstance(item, dict) for item in data):
            shapes = {frozenset(key for key in item if isinstance(key, str)) for item in data}
            return len(shapes) <= max(2, count // 20)
        return False
    if not isinstance(data, dict):
        return True
    total, distinct = _key_stats(data)
    return total >= _DATA_KEY_TOTAL_MIN and distinct / total < _DATA_KEY_DISTINCT_RATIO


def _accepts_generic(source: bytes, data: Any) -> bool:
    """Size rung plus structural rungs — may this document be indexed generically?"""
    return len(source) <= MAX_GENERIC_CONFIG_BYTES and not _looks_like_data(data)


@dataclass(frozen=True)
class _KeyNode:
    """One key of a generic config file: its span, its leaf text and its children.

    The three formats reach this shape by different routes — YAML through
    PyYAML's composer nodes (which carry marks), JSON and TOML through their
    tree-sitter trees (``json.loads``/``tomllib`` report no positions at all) —
    so the emitter below is format-agnostic.
    """

    key: str
    line_start: int
    line_end: int
    text: str | None
    """Raw ``key: value`` source for a leaf; ``None`` for a node with children."""
    children: tuple[_KeyNode, ...] = ()


def _emit_generic(out: _Out, keys: Sequence[_KeyNode], parent_uid: str) -> None:
    """Emit a key tree under *parent_uid*, stopping at ``_GENERIC_MAX_ENTITIES``."""
    for key_node in keys:
        if len(out.entities) >= _GENERIC_MAX_ENTITIES:
            return
        has_children = bool(key_node.children)
        uid = out.add(
            name=key_node.key,
            qn_suffix=f"{_suffix(parent_uid)}.{_qn_segment(key_node.key)}",
            label=NodeLabel.TYPE_DEF if has_children else NodeLabel.VALUE,
            kind=_GENERIC_SECTION_KIND if has_children else _GENERIC_SETTING_KIND,
            line_start=key_node.line_start,
            line_end=key_node.line_end,
            source=key_node.text,
        )
        out.rel(parent_uid, RelType.DEFINES, uid)
        _emit_generic(out, key_node.children, uid)


def _generic_module(out: _Out, total_lines: int) -> str:
    """Mint the Module node for a generic config file; returns its uid.

    Added before any key node so that it, and not a key, owns the undeduplicated
    file-level qualified name. Carries the file text as ``source`` (the framework
    truncates it) — for a file whose keys the parser cannot interpret, full-text
    search over the raw content is most of the value on offer.
    """
    return out.add(
        name=PurePosixPath(out.file_path).name,
        qn_suffix=_module_qualified_name(out.file_path),
        label=NodeLabel.MODULE,
        kind=_GENERIC_MODULE_KIND,
        line_start=1,
        line_end=total_lines,
        source=out.text,
    )


def _generic_yaml_keys(out: _Out, node: yaml.Node, depth: int, on_path: set[int]) -> list[_KeyNode]:
    """Build the key tree of a YAML mapping node, using composer marks for lines.

    ``on_path`` is scoped to the current branch, not the whole walk: unlike
    ``_walk_k8s``, two siblings pointing at one anchored mapping should both get
    their subtree, and ``_GENERIC_MAX_DEPTH`` already bounds the blow-up a
    path-scoped set would otherwise allow.
    """
    if depth >= _GENERIC_MAX_DEPTH or not isinstance(node, yaml.MappingNode) or id(node) in on_path:
        return []
    on_path.add(id(node))
    keys: list[_KeyNode] = []
    for key_node, value_node in node.value:
        if not isinstance(key_node, yaml.ScalarNode):
            continue
        children = tuple(_generic_yaml_keys(out, value_node, depth + 1, on_path))
        line_start, line_end = out.lines(key_node, value_node)
        keys.append(
            _KeyNode(
                key=str(key_node.value),
                line_start=line_start,
                line_end=line_end,
                text=None if children else out.slice(key_node, value_node),
                children=children,
            )
        )
    on_path.discard(id(node))
    return keys


def _json_key_text(node: Node) -> str:
    """The unquoted text of a JSON ``string`` node used as an object key."""
    for child in node.named_children:
        if child.type == "string_content":
            return node_text(child)
    return node_text(node).strip('"')


def _generic_json_keys(node: Node, depth: int) -> list[_KeyNode]:
    """Build the key tree of a tree-sitter JSON ``object`` node.

    No cycle guard: a tree-sitter tree is a tree.
    """
    if depth >= _GENERIC_MAX_DEPTH or node.type != "object":
        return []
    keys: list[_KeyNode] = []
    for pair in node.named_children:
        if pair.type != "pair":
            continue
        key_node = pair.child_by_field_name("key")
        value_node = pair.child_by_field_name("value")
        if key_node is None or value_node is None:
            continue
        children = tuple(_generic_json_keys(value_node, depth + 1))
        keys.append(
            _KeyNode(
                key=_json_key_text(key_node),
                line_start=pair.start_point[0] + 1,
                line_end=pair.end_point[0] + 1,
                text=None if children else node_text(pair),
                children=children,
            )
        )
    return keys


_TOML_KEY_NODES = frozenset({"bare_key", "quoted_key", "dotted_key"})
_TOML_TABLE_NODES = frozenset({"table", "table_array_element"})


def _toml_key_text(node: Node) -> str:
    """The text of a TOML key node, quotes stripped.

    A ``dotted_key`` is returned whole (``tool.ruff.lint``); ``_qn_segment``
    folds its dots, so it becomes one segment rather than faking three levels of
    nesting that the TOML grammar does not actually have.
    """
    return node_text(node).strip('"').strip("'")


def _toml_pair(pair: Node) -> _KeyNode | None:
    """One ``key = value`` line as a leaf key node."""
    named = pair.named_children
    if not named or named[0].type not in _TOML_KEY_NODES:
        return None
    return _KeyNode(
        key=_toml_key_text(named[0]),
        line_start=pair.start_point[0] + 1,
        line_end=pair.end_point[0] + 1,
        text=node_text(pair),
    )


def _generic_toml_keys(root: Node) -> list[_KeyNode]:
    """Top-level pairs plus one section per ``[table]`` / ``[[table]]``.

    TOML tables are flat in the grammar — ``[tool.ruff.lint]`` is a single node
    with a dotted header, not three nested ones — so this two-level shape is the
    document's real shape rather than a truncation at ``_GENERIC_MAX_DEPTH``.
    """
    keys: list[_KeyNode] = []
    for child in root.named_children:
        if child.type == "pair":
            entry = _toml_pair(child)
            if entry is not None:
                keys.append(entry)
            continue
        if child.type not in _TOML_TABLE_NODES:
            continue
        named = child.named_children
        if not named or named[0].type not in _TOML_KEY_NODES:
            continue
        children = tuple(entry for entry in map(_toml_pair, named[1:]) if entry is not None)
        keys.append(
            _KeyNode(
                key=_toml_key_text(named[0]),
                line_start=child.start_point[0] + 1,
                # The table node's own end point runs to the start of the next
                # table, blank lines included; its last child is where it really
                # ends.
                line_end=named[-1].end_point[0] + 1,
                text=None if children else node_text(child).rstrip(),
                children=children,
            )
        )
    return keys


def _parse_toml(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile | None:
    """Generic key-tree parse of a TOML file — there is no TOML dialect table.

    Before this existed ``.toml`` was registered and dispatched to nothing at
    all: every TOML file in every repo parsed successfully and produced zero
    entities, silently.
    """
    try:
        # TOMLDecodeError and UnicodeDecodeError are both ValueError subclasses.
        # Either one means "not a single well-formed TOML document" -> data.
        text = source.decode("utf-8")
        data = tomllib.loads(text)
    except ValueError:
        return None
    if not _accepts_generic(source, data):
        return None

    out = _Out(file_path=path, project_name=project_name, text=text)
    module_uid = _generic_module(out, text.count("\n") + 1)
    _emit_generic(out, _generic_toml_keys(root), module_uid)
    return ParsedFile(file_path=path, language="toml", entities=out.entities, relationships=out.relationships)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _extract_document(out: _Out, doc: _Doc, dialect: str, module_uid: str) -> _K8sResource | None:
    """Route one classified document to its dialect handler."""
    if dialect == "k8s":
        line_start, line_end = out.lines(doc.node)
        return _extract_k8s(
            out,
            doc.data,
            module_uid=module_uid,
            line_start=line_start,
            line_end=line_end,
            source=out.slice(doc.node),
        )
    if dialect == "compose":
        _extract_compose(out, doc, module_uid)
    elif dialect == "github_workflow":
        _extract_workflow(out, doc, module_uid)
    elif dialect == "ansible_playbook":
        _extract_playbook(out, doc, module_uid)
    else:
        _extract_task_file(out, doc, module_uid, dialect)
    return None


def _parse_yaml(path: str, source: bytes, project_name: str) -> ParsedFile | None:
    """Classify each document of a YAML stream and extract the recognised ones."""
    if _is_opaque(path, source):
        return None
    text = source.decode("utf-8", errors="replace")
    docs = _load_yaml_documents(text)
    if not docs:
        return None

    classified = [(doc, _document_dialect(path, doc.data)) for doc in docs]
    recognised = [(doc, dialect) for doc, dialect in classified if dialect is not None]
    if not recognised:
        return _parse_generic_yaml(path, source, text, docs, project_name)

    out = _Out(file_path=path, project_name=project_name, text=text)
    # The Module entity must be added first so that it, and not some later
    # entity, owns the undeduplicated file-level qualified name.
    total_lines = text.count("\n") + 1
    module_uid = out.add(
        name=PurePosixPath(path).name,
        qn_suffix=_module_qualified_name(path),
        label=NodeLabel.MODULE,
        kind=_MODULE_KINDS[recognised[0][1]],
        line_start=1,
        line_end=total_lines,
    )
    if recognised[0][1] in {"ansible_tasks", "ansible_handlers"}:
        _emit_role(out, module_uid, total_lines)

    resources = [_extract_document(out, doc, dialect, module_uid) for doc, dialect in recognised]
    _link_k8s_selectors(out, [r for r in resources if r is not None])

    return ParsedFile(file_path=path, language="yaml", entities=out.entities, relationships=out.relationships)


def _parse_generic_yaml(
    path: str,
    source: bytes,
    text: str,
    docs: list[_Doc],
    project_name: str,
) -> ParsedFile | None:
    """Generic key-tree parse of a YAML stream no dialect claimed.

    A stream carrying more than one document is the parseability rung of the
    data ladder: ``yaml.safe_load`` would refuse it outright, and a *generic*
    multi-document stream is a record dump (the dialects that legitimately use
    ``---`` — k8s above all — were already claimed above).
    """
    if len(docs) != 1 or not _accepts_generic(source, docs[0].data):
        return None
    out = _Out(file_path=path, project_name=project_name, text=text)
    module_uid = _generic_module(out, text.count("\n") + 1)
    _emit_generic(out, _generic_yaml_keys(out, docs[0].node, 0, set()), module_uid)
    return ParsedFile(file_path=path, language="yaml", entities=out.entities, relationships=out.relationships)


def _parse_json(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile | None:
    """Extract a JSON document: k8s-as-JSON if it is one, otherwise a key tree.

    The k8s predicate is content-only, so it applies to ``.json`` unchanged.
    Everything else (package.json, tsconfig, .eslintrc) falls through to the
    generic parse unless the data ladder rejects it.
    """
    try:
        data = json.loads(source)
    except ValueError, UnicodeDecodeError:
        # Not one well-formed document — NDJSON, a truncated dump, a log. Data.
        return None

    text = source.decode("utf-8", errors="replace")
    total_lines = text.count("\n") + 1

    if not _is_k8s(data):
        if not _accepts_generic(source, data):
            return None
        out = _Out(file_path=path, project_name=project_name, text=text)
        module_uid = _generic_module(out, total_lines)
        document = next((child for child in root.named_children if child.type == "object"), None)
        keys = _generic_json_keys(document, 0) if document is not None else []
        _emit_generic(out, keys, module_uid)
        return ParsedFile(file_path=path, language="json", entities=out.entities, relationships=out.relationships)

    out = _Out(file_path=path, project_name=project_name, text=text)
    module_uid = out.add(
        name=PurePosixPath(path).name,
        qn_suffix=_module_qualified_name(path),
        label=NodeLabel.MODULE,
        kind=_MODULE_KINDS["k8s"],
        line_start=1,
        line_end=total_lines,
    )
    resource = _extract_k8s(
        out,
        data,
        module_uid=module_uid,
        line_start=1,
        line_end=total_lines,
        source=text,
    )
    _link_k8s_selectors(out, [resource])
    return ParsedFile(file_path=path, language="json", entities=out.entities, relationships=out.relationships)


def _parse_config(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile | None:
    """Extract entities from a structured config file, by format then by dialect.

    Returns ``None`` — no nodes, no embeddings — only for an empty file, a data
    file (see ``_looks_like_data``) or an XML document with no root element.
    Anything else that parses gets at least a Module node and its key tree.
    ``parse_file`` turns the ``None`` into an empty ``ParsedFile``, so a rejected
    file is still hashed and not re-parsed on the next pass.
    """
    norm_path = path.replace("\\", "/")
    suffix = PurePosixPath(norm_path).suffix.lower()
    if not source.strip() or suffix in _DATA_SUFFIXES:
        return None
    if suffix == ".xml":
        return _parse_xml(norm_path, root, project_name)
    if suffix == ".json":
        return _parse_json(norm_path, source, root, project_name)
    if suffix == ".toml":
        return _parse_toml(norm_path, source, root, project_name)
    if suffix in _YAML_SUFFIXES:
        return _parse_yaml(norm_path, source, project_name)
    return None


# ---------------------------------------------------------------------------
# Language registration — one try per grammar, deliberately not one shared try
# ---------------------------------------------------------------------------

try:
    import tree_sitter_yaml as _ts_yaml

    _YAML_LANGUAGE = Language(_ts_yaml.language())
    register_language(
        LanguageConfig(
            name="yaml",
            extensions=frozenset({".yaml", ".yml"}),
            language=_YAML_LANGUAGE,
            query=Query(_YAML_LANGUAGE, "(stream) @root"),
            parse_func=_parse_config,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass

try:
    import tree_sitter_json as _ts_json

    _JSON_LANGUAGE = Language(_ts_json.language())
    register_language(
        LanguageConfig(
            name="json",
            extensions=frozenset({".json"}),
            language=_JSON_LANGUAGE,
            query=Query(_JSON_LANGUAGE, "(document) @root"),
            parse_func=_parse_config,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass

try:
    import tree_sitter_toml as _ts_toml

    _TOML_LANGUAGE = Language(_ts_toml.language())
    register_language(
        LanguageConfig(
            name="toml",
            extensions=frozenset({".toml"}),
            language=_TOML_LANGUAGE,
            query=Query(_TOML_LANGUAGE, "(document) @root"),
            parse_func=_parse_config,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass

try:
    import tree_sitter_xml as _ts_xml

    # language_xml(), NOT language() — see the module docstring. There is no
    # language() attribute on this module, and AttributeError would escape the
    # ImportError guard below.
    _XML_LANGUAGE = Language(_ts_xml.language_xml())
    register_language(
        LanguageConfig(
            name="xml",
            extensions=frozenset({".xml"}),
            language=_XML_LANGUAGE,
            query=Query(_XML_LANGUAGE, "(document) @root"),
            parse_func=_parse_config,
            # Capital C — the XML grammar names this node "Comment".
            comment_node_types=frozenset({"Comment"}),
        )
    )
except ImportError:
    pass
