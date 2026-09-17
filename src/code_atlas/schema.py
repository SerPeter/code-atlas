"""Graph schema definitions for Code Atlas.

Defines node labels, relationship types, kind discriminators, constraint/index
specs, and pure DDL generation functions.  Import-time validation ensures every
NodeLabel is covered by constraint and index registries.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum

# Schema version — bump on every schema change that requires migration.
SCHEMA_VERSION: int = 20

# Sentinel ``project_name`` for nodes that are shared across every project.
#
# ``project_name`` carries an existence constraint on every entity label, so a
# genuinely global node cannot simply omit it (Memgraph reports the violation
# at COMMIT, not at statement time, so the offending write appears to succeed
# and the whole transaction dies later).  A reserved sentinel is the only way
# to express "belongs to no single project" — see NodeLabel.ENV_VAR.
#
# Anything that filters by project must treat this value as a member of every
# project, and anything that refuses to touch non-test data must allowlist it
# (see tests/integration/conftest.py::_assert_disposable_db).
GLOBAL_PROJECT: str = "_global"

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------


class NodeLabel(StrEnum):
    # Containers
    PROJECT = "Project"
    PACKAGE = "Package"
    MODULE = "Module"
    # Code entities (discriminated by `kind`)
    TYPE_DEF = "TypeDef"
    CALLABLE = "Callable"
    VALUE = "Value"
    # Documentation
    DOC_FILE = "DocFile"
    DOC_SECTION = "DocSection"
    # Knowledge vault (zettelkasten notes — code-atlas + memory-dir dialects)
    NOTE = "Note"
    # External dependencies
    EXTERNAL_PACKAGE = "ExternalPackage"
    EXTERNAL_SYMBOL = "ExternalSymbol"
    # Referenced-but-not-defined runtime surface (see _EXTERNAL_LABELS)
    ENV_VAR = "EnvVar"
    RESOURCE_FILE = "ResourceFile"
    # Marker: stamped onto every _ENTITY_LABELS node alongside its primary
    # label (see _MARKER_LABELS) so a uid-only lookup that doesn't know which
    # of the 12 entity labels it's after can use one shared index instead of
    # an unindexed ScanAll over the whole graph.
    ENTITY = "Entity"
    # Overflow vectors: chunks 2..N of a node whose embed text exceeds the model's
    # input cap. Chunk 1 stays on the node itself, so the common case -- a node that
    # fits -- creates none of these. See _EMBED_CHUNK_LABELS.
    EMBED_CHUNK = "EmbedChunk"
    # Meta
    SCHEMA_VERSION = "SchemaVersion"


# ---------------------------------------------------------------------------
# Relationship types
# ---------------------------------------------------------------------------


class RelType(StrEnum):
    # Structural
    CONTAINS = "CONTAINS"
    DEFINES = "DEFINES"
    # Type hierarchy
    INHERITS = "INHERITS"
    IMPLEMENTS = "IMPLEMENTS"
    # Call / data flow
    CALLS = "CALLS"
    # Names a callable as a VALUE without invoking it: passed as an argument, held in a
    # dispatch table, handed to a decorator. Deliberately not CALLS — blast_radius,
    # find_dead_code, the outline and Leiden all read CALLS as "executes", and a callback
    # that is registered today and invoked by a framework tomorrow did not execute here.
    REFERENCES = "REFERENCES"
    IMPORTS = "IMPORTS"
    USES_TYPE = "USES_TYPE"
    OVERRIDES = "OVERRIDES"
    # Runtime configuration surface (code -> EnvVar / ResourceFile)
    READS_ENV = "READS_ENV"
    REFERENCES_FILE = "REFERENCES_FILE"
    # Dependencies
    DEPENDS_ON = "DEPENDS_ON"
    # A pipeline model produces the warehouse object a BI table loads from -- the one
    # edge that carries impact across the warehouse boundary, so `blast_radius` on a dbt
    # model reaches the reports built on it. Its own type rather than DEPENDS_ON, which
    # already carries two endpoint shapes (project-to-project, and Project ->
    # ExternalPackage with a version); a third would make the deletion queries harder to
    # keep correct than one more name is worth.
    FEEDS = "FEEDS"
    # Documentation
    DOCUMENTS = "DOCUMENTS"
    # Similarity
    SIMILAR_TO = "SIMILAR_TO"
    EXPORTS = "EXPORTS"
    # Pattern-detected
    HANDLES_ROUTE = "HANDLES_ROUTE"
    HANDLES_EVENT = "HANDLES_EVENT"
    REGISTERED_BY = "REGISTERED_BY"
    INJECTED_INTO = "INJECTED_INTO"
    TESTS = "TESTS"
    HANDLES_COMMAND = "HANDLES_COMMAND"
    # Knowledge vault (Note <-> Note/DocSection)
    LINKS_TO = "LINKS_TO"
    DERIVED_FROM = "DERIVED_FROM"
    SUPERSEDES = "SUPERSEDES"
    # Symmetric: two notes that disagree and nobody has resolved which is right.
    # Written only from frontmatter -- there is no automated contradiction detection,
    # and inventing one would be a machine asserting a dispute a human never made.
    CONTRADICTS = "CONTRADICTS"
    # Git-derived signals (Module <-> Module, out-of-band — see git_signals.py)
    CO_CHANGES_WITH = "CO_CHANGES_WITH"


# ---------------------------------------------------------------------------
# Kind discriminators
# ---------------------------------------------------------------------------


class TypeDefKind(StrEnum):
    CLASS = "class"
    STRUCT = "struct"
    INTERFACE = "interface"
    TRAIT = "trait"
    ENUM = "enum"
    UNION = "union"
    TYPE_ALIAS = "type_alias"
    PROTOCOL = "protocol"
    RECORD = "record"
    DATA_TYPE = "data_type"
    TYPECLASS = "typeclass"
    ANNOTATION = "annotation"


class CallableKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor"
    STATIC_METHOD = "static_method"
    CLASS_METHOD = "class_method"
    PROPERTY = "property"
    CLOSURE = "closure"


class ValueKind(StrEnum):
    VARIABLE = "variable"
    CONSTANT = "constant"
    FIELD = "field"
    ENUM_MEMBER = "enum_member"
    # A long string literal lifted out of the code around it: a prompt template, an
    # embedded query, a help screen. Content someone searches for by what it says,
    # which inside a function body reaches the graph no other way.
    TEXT_BLOCK = "text_block"


class NoteKind(StrEnum):
    """Note lifecycle stage — stored in the shared ``kind`` property, like CallableKind."""

    DRAFT = "draft"
    NOTE = "note"
    DECISION = "decision"


class Visibility(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    PROTECTED = "protected"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Label groupings
# ---------------------------------------------------------------------------

_CODE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.PROJECT,
        NodeLabel.PACKAGE,
        NodeLabel.MODULE,
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
    }
)

_DOC_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.DOC_FILE,
        NodeLabel.DOC_SECTION,
        NodeLabel.NOTE,
    }
)

# "Exists only because something referenced it" — no source location of its
# own, MERGEd during post-batch resolution, never produced by a parser as a
# ParsedEntity.  EnvVar/ResourceFile share exactly that lifecycle with
# ExternalPackage/ExternalSymbol, which is why they live here rather than in
# _CODE_LABELS: joining this group is what gives them uid uniqueness, the
# uid+project_name existence constraints, and the property/composite indices.
_EXTERNAL_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.EXTERNAL_PACKAGE,
        NodeLabel.EXTERNAL_SYMBOL,
        NodeLabel.ENV_VAR,
        NodeLabel.RESOURCE_FILE,
    }
)

# Reference-counted labels: a node exists exactly as long as something points
# at it, so "zero incoming edges" means "unreferenced" and the node can be
# swept (see GraphBackend.gc_orphaned_reference_nodes).  ExternalPackage and
# ExternalSymbol are deliberately NOT here — an ExternalPackage receives a
# structural CONTAINS edge from its package and, since v18, a structural
# incoming ``Project -[DEPENDS_ON]->`` carrying the manifest-declared version,
# so incoming-edge count is not a reference count for them.  (Before v18 the
# version was a node property, which was the other half of the same argument;
# moving it onto an edge did not weaken the exclusion, it added a second
# structural edge to it.)
#
# INVARIANT: never give these labels a structural incoming edge (a Project or
# Package CONTAINS, say).  It would make every node permanently referenced and
# silently disable the sweep.  DEPENDS_ON on ExternalPackage is exactly that
# kind of edge, which is why the label must stay out of this set.
_REFERENCE_COUNTED_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.ENV_VAR,
        NodeLabel.RESOURCE_FILE,
    }
)

# Deliberately outside _ENTITY_LABELS: an EmbedChunk is not a thing in the codebase,
# it is a second vector for one. It carries no :Entity marker (so no uid-only lookup,
# relationship linking or marker sweep ever reaches one), no content_hash, no
# file_path, and no text index -- only enough to be found by vector search and
# resolved back to its parent. It is in _EMBEDDABLE_LABELS purely to get a vector
# index and be queried alongside the rest.
_EMBED_CHUNK_LABELS: frozenset[NodeLabel] = frozenset({NodeLabel.EMBED_CHUNK})

_EMBEDDABLE_LABELS: frozenset[NodeLabel] = _EMBED_CHUNK_LABELS | frozenset(
    {
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
        NodeLabel.MODULE,
        NodeLabel.DOC_SECTION,
        NodeLabel.NOTE,
        # Since v19 (ATL-191 P5). A stub-indexed ExternalSymbol carries the signature and
        # docstring of a library entrypoint, which is what makes "how do I set a timeout"
        # reach `httpx.Client.request` rather than only the repo's own wrapper. Gated by
        # `[libraries] embed_stubs`, so a symbol with no stub read stays a bare name with
        # nothing to embed. A library's public API also changes far less often than the
        # code calling it, so these vectors are re-bought rarely.
        NodeLabel.EXTERNAL_SYMBOL,
    }
)

# EnvVar/ResourceFile are text-searchable but NOT embeddable: an agent asking
# "where is DATABASE_URL read?" needs the keyword hit, and there is nothing to
# embed — the node is a bare identifier, and by the names-only invariant it
# never holds the variable's value or a default.
_TEXT_SEARCHABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.TYPE_DEF,
        NodeLabel.CALLABLE,
        NodeLabel.VALUE,
        NodeLabel.MODULE,
        NodeLabel.DOC_SECTION,
        NodeLabel.NOTE,
        NodeLabel.ENV_VAR,
        NodeLabel.RESOURCE_FILE,
    }
)

# Stamped alongside a primary entity label on every node in this union, purely
# so a caller holding only a uid (not knowing which of the other 12 labels it
# belongs to) can MATCH (n:Entity {uid: ...}) against one shared index instead
# of an unindexed ScanAll over the whole graph. See NodeLabel.ENTITY.
#
# Deliberately NOT folded into _ENTITY_LABELS: that set drives the constraint and
# index registries, and a marker sits on *every node in the graph*, so each
# registry it joins is a cost every write pays. uid/project_name existence and
# uniqueness are already enforced by the primary label underneath it, and only
# two of the seven indexed properties are ever reached through the marker --
# see _MARKER_INDEX_PROPERTIES.
_MARKER_LABELS: frozenset[NodeLabel] = frozenset({NodeLabel.ENTITY})

# All non-meta labels (must have uid + project_name)
_ENTITY_LABELS: frozenset[NodeLabel] = _CODE_LABELS | _DOC_LABELS | _EXTERNAL_LABELS

# The labels that carry ``file_hash`` — the per-file gate that lets an unchanged file
# skip parsing entirely. A label qualifies by having exactly one node per file, written
# on every parse of it, which is what makes the stored hash mean "the content this file
# was last indexed at".
#
# DocFile is here because markdown produces no Module node, so without it every ``.md``
# file re-parsed on every event forever — the gate read back None and could never
# record one. Measured on a 3,254-file repo whose 2,651 markdown files were all
# ungated: coverage was 18%, and adding DocFile takes it to 99.7%.
#
# A migration that clears hashes to force a re-parse must clear ALL of these, which is
# what generate_clear_file_hashes_ddl is for — the two that predate it named Module and
# Package literally, and a third written from that template would silently leave every
# document gated.
FILE_HASH_LABELS: tuple[NodeLabel, ...] = (NodeLabel.MODULE, NodeLabel.PACKAGE, NodeLabel.DOC_FILE)


STDLIB_MODULE_NAMES: frozenset[str] = frozenset(sys.stdlib_module_names)
"""Top-level module names shipped with this Python. Used to tell a dependency from a batteries-included import.

An `ExternalPackage` means only "outside this project", so `ext/hashlib` and `ext/litellm`
are the same kind of node -- and on a real corpus the first kind dominates: of the 90
package names shared by two or more projects here, **59 are stdlib**, and the five most
widely shared are `json`, `pathlib`, `time`, `collections` and `datetime`. A dependency
report that leads with those is answering a question nobody asked.

**Python's list is the only one available in-process, so this is Python-shaped**, and
deliberately not hidden behind a name that suggests otherwise. A Ruby or Node package
named after a Python stdlib module is mislabelled by it -- Ruby's `json` and `time` are
in fact also stdlib, so the two most common collisions happen to land right, which is
luck rather than design. Recording the importer's language on the node is the honest fix
and is a larger change: it has to reach the graph through the parse contract, because the
graph layer cannot import the language registry without pulling tree-sitter into every
consumer of a graph client.

Kept here rather than in either backend so there is one list. `server/analysis.py` had
its own copy for `_mark_external`; two frozensets of the same thing is the drift this
module exists to prevent.
"""


def split_image_reference(reference: str) -> tuple[str, str] | None:
    """Split a container image reference into ``(repository, version)``.

    ``ghcr.io/acme/api:1.2.3`` -> ``("ghcr.io/acme/api", "1.2.3")``;
    ``python:3.14-slim`` -> ``("python", "3.14-slim")``;
    ``redis@sha256:ab...`` -> ``("redis", "sha256:ab...")``;
    ``nginx`` -> ``("nginx", "")``.

    A colon *after* the last slash is a tag; before it, it is a registry port, so
    ``localhost:5000/app`` keeps its port and has no version. A digest wins over a tag
    when both are present: it is the stronger pin, and it is what the build actually
    resolved to. An untagged image yields an empty version rather than an invented
    ``latest`` -- the file did not say ``latest``, and a DEPENDS_ON edge with no version
    still records the declaration.

    Returns ``None`` for a templated reference (``${TAG}``, ``{{ .Values.image }}``,
    ``$BASE_IMAGE``): those name nothing resolvable, and an ExternalPackage called
    ``$BASE_IMAGE`` is worse than no edge at all.

    One copy on purpose. The repository half becomes an ExternalPackage *name* (via the
    parsers, which mark it :data:`IMPORT_ATOMIC_NAME`) and the version half becomes the
    ``DEPENDS_ON`` version (via the manifest parsers in ``indexing/orchestrator.py``).
    Those two join on the name, so a second implementation drifting by one character
    silently stops writing dependency edges for every image.
    """
    ref = reference.strip()
    if not ref or "${" in ref or "{{" in ref or "$" in ref:
        return None
    body, _, digest = ref.partition("@")
    head, slash, tail = body.rpartition("/")
    name, _, tag = tail.partition(":")
    if not name:
        return None
    return f"{head}{slash}{name}", digest or tag


IMPORT_ECOSYSTEM = "ecosystem"
"""`ParsedRelationship.properties` key: which packaging ecosystem this import names (ATL-194).

`resolve_imports` mints one `ExternalPackage` per *name*, so a name claimed by two
ecosystems was one node. `redis` is the worked example: the Python client
(`dependencies = ["redis"]`) and the server image (`image: redis:7`) are unrelated
artifacts with unrelated versions, and both minted `ext/redis`. The manifest merge then
saw two version claims for one node and dropped both -- honest, and useless.

The ecosystem is a *parser*-side fact for the reason `IMPORT_ATOMIC_NAME` is: the resolver
sees a string, and nothing about `redis` says which one it is. Unlike the atomic-name
marker this one is stamped centrally, by `parse_file`, from the language that produced the
file -- so a language gets its ecosystem for free and only a parser emitting imports from
*another* ecosystem has to say so. `config.py` is the whole of that: a compose `image:` and
a Kubernetes container image are `docker`, a workflow `uses:` is `actions`, all from a YAML
file whose own ecosystem is nothing in particular.
"""

ECOSYSTEM_PYPI = "pypi"
ECOSYSTEM_DOCKER = "docker"
ECOSYSTEM_ACTIONS = "actions"
ECOSYSTEM_UNKNOWN = "unknown"

LANGUAGE_ECOSYSTEM: dict[str, str] = {
    "python": "pypi",
    "typescript": "npm",
    "tsx": "npm",
    "javascript": "npm",
    "rust": "crates",
    "go": "go",
    "java": "maven",
    "csharp": "nuget",
    "php": "packagist",
    "ruby": "rubygems",
    "apex": "salesforce",
    "salesforce": "salesforce",
    "sql": "warehouse",
    # A TMDL partition references the same warehouse tables a SQL file defines, and the
    # two have to land on one node or the whole point of the shared `warehouse.` namespace
    # is lost. So TMDL's ecosystem is where its imports *point*, not what it is written in.
    "tmdl": "warehouse",
}
"""Language name -> the registry its imports name, where one exists.

Only languages whose import statements resolve against a *package registry* get an entry.
A language absent here falls back to its own name (`cpp`, `hcl`, `shell`), which is
honest: a C++ `#include` names a header on a path, not a package anybody publishes, and
inventing a registry for it would claim a fact.

`sql` maps to `warehouse` because that is what its externals already were -- a schema-wide
table reference, minted `ext/warehouse.<name>` since long before this existed.
"""


def ecosystem_for_language(language: str) -> str:
    """The ecosystem a language's imports belong to. Never empty."""
    return LANGUAGE_ECOSYSTEM.get(language) or language or ECOSYSTEM_UNKNOWN


def external_qualified_name(ecosystem: str, name: str) -> str:
    """`ext/{ecosystem}/{name}` -- the qualified_name of an external node (ATL-194).

    The separator is `/` and the ecosystem segment can never contain one, so a name that
    *does* (`ghcr.io/acme/api`, `actions/checkout`, `localhost:5000/app`) survives whole:
    split once, from the left.
    """
    return f"ext/{ecosystem}/{name}"


def split_external_qualified_name(qualified_name: str) -> tuple[str, str] | None:
    """Inverse of :func:`external_qualified_name`, or None if this is not an external name.

    Returns None for a pre-ATL-194 `ext/{name}` too, which is what lets a reader tell a
    migrated node from one it has not reached yet.
    """
    rest = qualified_name.removeprefix("ext/")
    if rest == qualified_name:
        return None
    ecosystem, separator, name = rest.partition("/")
    return (ecosystem, name) if separator and name else None


IMPORT_ATOMIC_NAME = "atomic_name"
"""`ParsedRelationship.properties` key: `to_name` is already the package name (ATL-191 P2).

`resolve_imports` derives an `ExternalPackage` from an unresolved import by taking the part
before the first dot, because for the languages that rule was written for -- Python, Java,
Go -- a dotted import name is a *module path* whose first segment is the distribution.

A container image reference is not a module path. Applying the rule to
`ghcr.io/huggingface/text-embeddings-inference` truncates a registry hostname and produces a
package called `ghcr`, which then collects every unrelated image on that registry, and a
sibling `ExternalSymbol` carrying the real name. A parser that knows its `to_name` is an
opaque identifier sets this, and the resolver uses the name whole.

It is a parser-side marker rather than a resolver-side heuristic on purpose: nothing about
the *string* distinguishes `ghcr.io/acme/api` from `os.path`, and a resolver guessing from
shape would have to be re-guessed for every ecosystem added later. The parser already knows
which it emitted.

Go module paths (`github.com/spf13/cobra`) have exactly the same defect and deliberately do
not set this yet -- adopting it would change the uid of every Go external package, which is
a migration, not a fix. See the manifest-parsing header in `indexing/orchestrator.py`.
"""


PROVENANCE_DECLARED = "declared"
PROVENANCE_STDLIB = "stdlib"
PROVENANCE_UNDECLARED = "undeclared"
EXTERNAL_PROVENANCE: tuple[str, ...] = (PROVENANCE_DECLARED, PROVENANCE_STDLIB, PROVENANCE_UNDECLARED)
"""Where an `ExternalPackage` came from, most deliberate first (ATL-191).

Ranking wants to know how much a package was *chosen*. A name in a manifest is a decision
somebody made; the standard library is a given; anything else arrived because something
else needed it. `[search.importance]` turns that into a multiplier, so dilution is handled
by ranking rather than by refusing to index — which is what the original design did, and
why it capped the graph at bare names.

All three are read off the graph with no new parsing and no new pass over source:

* **declared** -- a `Project -[DEPENDS_ON]-> ExternalPackage` edge exists. That edge is
  written from a manifest and from nothing else, so its presence *is* the declaration.
* **stdlib** -- the name is in :data:`STDLIB_MODULE_NAMES`. Python-shaped, like everything
  else keyed on that set.
* **undeclared** -- neither. Imported by this project's code, declared by nobody in it.

  Named for what the check tests, not for a mechanism it cannot see. "Transitive" would
  be a claim about *why* the package is here, and on the reference graph that claim is
  mostly false: the tier's largest members are `valkey/valkey`, `memgraph/memgraph-mage`,
  `ghcr`, `actions/setup-python` and `astral-sh/setup-uv` -- docker images and GitHub
  Actions, which arrived through no dependency resolver at all. Splitting a genuine
  transitive dependency from a foreign-ecosystem name needs ecosystem identity on the
  node, which does not exist yet.

The ordering of the tuple is the ranking order and is load-bearing; it is not alphabetical.

Since ATL-191 P2, `declared` also covers `[project.optional-dependencies]` and
`[dependency-groups]` (gated on the distribution actually being installed, since an extra
nobody selected is not a dependency of this checkout), and container images declared by a
`Dockerfile` or a compose file, which carry their tag as the version.

One imprecision remains, narrowing rather than wrong: a non-Python ecosystem with no
manifest parser here reaches `undeclared` by absence rather than by evidence. Ruby `require`
paths, Go module paths and GitHub Actions `uses:` references are all in that set -- the
Actions in particular are declared, and pinned, in `.github/workflows/`, which is not a
root manifest and so is not probed.
"""


_GRANT_ONLY_KINDS: frozenset[str] = frozenset({"permission_set", "profile"})
"""Entity kinds whose edges say "who may see this", not "what depends on this".

Both come from ``salesforce.py``'s ``_parse_permission_document``, which gives one
node an ``IMPORTS`` edge per grant — 1,900 from one measured 427 KB file. Every one
lands at depth 1 with a perfect ``confidence_score``, because ``IMPORTS`` carries no
``weight`` property, so without a demotion they outrank every transitive code
dependent under ``blast_radius``'s depth-first ordering.

Held here rather than in ``graph/client.py`` because ``server/analysis.py`` — the
one place that ranks on it — already imports ``schema`` and takes no dependency on
the graph layer. Literals rather than an import from ``parsing.languages``, the
same rule every other cross-layer kind constant follows; a test pins the two
spellings together so they cannot drift silently.
"""

CUSTOM_COMPONENT_PREFIX: str = "cmp."
"""Target prefix for a custom component whose KIND one file cannot determine.

Salesforce shares one ``c`` namespace between Aura and LWC and forbids the two from
holding the same name — "A custom Lightning web component and a custom Aura
component in the same namespace can't have the same name" (LWC Dev Guide, Component
Namespaces). So ``<c:foo>`` in Aura markup, and a FlexiPage ``componentName``, each
name exactly one component identity — but nothing in the *file* says whether it is
an Aura bundle or an LWC one.

Nothing ever mints a ``cmp.`` node. The parsers emit this kind-agnostic target and
``resolve_imports`` widens it across :data:`COMPONENT_ALIAS_PREFIXES`, so the one
real node is found and no false stub is minted for the kind that does not exist.
Emitting both targets instead — which is what shipped first — resolved the true
edge and left an ``ext/`` stub asserting a component that was never referenced.

Lives here rather than in ``parsing/`` because both graph backends need it and
neither may import a parser module; ``schema.py`` is the only module all four share.
"""

COMPONENT_ALIAS_PREFIXES: tuple[str, ...] = ("aura.", "lwc.")
"""The namespaces a :data:`CUSTOM_COMPONENT_PREFIX` target may resolve into.

Order is fixed but arbitrary: it decides which node wins if a repo somehow holds
both, and such a repo cannot deploy. Aura first because Aura-in-Aura is the
majority — 94 of 132 references in the measured corpus.
"""


# Bumped BY HAND when extraction output changes for a reason no setting captures: a
# parser fix, a new language handler or grammar, a changed uid scheme, a different
# _compute_content_hash formula, a detector implementation change.
#
# Part of the file-hash gate's contract, which is why it lives beside FILE_HASH_LABELS.
# settings.extraction_key folds it in alongside the extraction-affecting configuration,
# and consumers._compute_file_hash folds that into the stored file_hash -- so bumping it
# invalidates every stored hash and the next run re-parses (ADR-0042 decision 5). It is
# the *primary* half of that key: the config half can only see three caps, the detector
# list and the rationale settings, and everything else about extraction lives in code.
#
# Deliberately NOT the package version -- a docs-only release would then force a global
# re-parse of every project. Deliberately NOT SCHEMA_VERSION either: an extraction change
# and a schema change are different events, and coupling them makes each pay the other's
# cost, most sharply the vector-index drop and rebuild every schema migration performs.
EXTRACTION_EPOCH: int = 11


# ---------------------------------------------------------------------------
# uid construction for reference-counted nodes
#
# Shared by both graph backends (and by the parsers that emit the references)
# so the two can never drift on the key that identifies the node.
# ---------------------------------------------------------------------------

ENV_VAR_PREFIX: str = "env/"
RESOURCE_FILE_PREFIX: str = "res/"


def env_var_uid(name: str) -> str:
    """uid for an environment variable — GLOBAL, deliberately unprefixed by project.

    Breaks the usual ``{project_name}:{qualified_name}`` uid format on purpose:
    an env var means the same thing in every repo, has no version dimension and
    carries no per-project attributes, so every callsite everywhere converges on
    one node.  That is what makes "who reads DATABASE_URL across all my repos" a
    single-node lookup instead of a name-join.  The node still needs a
    ``project_name`` for the existence constraint — it gets ``GLOBAL_PROJECT``.
    """
    return f"{ENV_VAR_PREFIX}{name}"


def resource_file_uid(project_name: str, path: str) -> str:
    """uid for a referenced (not indexed) file — PROJECT-SCOPED, unlike env vars.

    The asymmetry with :func:`env_var_uid` is intentional: a path is only
    meaningful relative to a project root, so ``data/fixtures.json`` in two
    repos is two different files and must not collapse into one node.
    """
    return f"{project_name}:{RESOURCE_FILE_PREFIX}{path}"


# ---------------------------------------------------------------------------
# Spec dataclasses (frozen, for generating DDL)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniqueConstraintSpec:
    label: NodeLabel
    property: str


@dataclass(frozen=True)
class ExistenceConstraintSpec:
    label: NodeLabel
    property: str


@dataclass(frozen=True)
class IndexSpec:
    label: NodeLabel
    property: str


@dataclass(frozen=True)
class CompositeIndexSpec:
    label: NodeLabel
    properties: tuple[str, ...]


@dataclass(frozen=True)
class VectorIndexSpec:
    name: str
    label: NodeLabel
    property: str
    dimension: int
    capacity: int
    metric: str = "cos"


@dataclass(frozen=True)
class TextIndexSpec:
    name: str
    label: NodeLabel


# ---------------------------------------------------------------------------
# Constraint and index registries
# ---------------------------------------------------------------------------

# uid uniqueness on all entity labels; version on SchemaVersion
UNIQUE_CONSTRAINTS: tuple[UniqueConstraintSpec, ...] = (
    *[UniqueConstraintSpec(label=lbl, property="uid") for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)],
    *[UniqueConstraintSpec(label=lbl, property="uid") for lbl in sorted(_EMBED_CHUNK_LABELS, key=str)],
    UniqueConstraintSpec(label=NodeLabel.SCHEMA_VERSION, property="version"),
)

# uid + project_name existence on all entity labels; version on SchemaVersion
EXISTENCE_CONSTRAINTS: tuple[ExistenceConstraintSpec, ...] = (
    *[
        spec
        for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)
        for spec in (
            ExistenceConstraintSpec(label=lbl, property="uid"),
            ExistenceConstraintSpec(label=lbl, property="project_name"),
        )
    ],
    *[
        spec
        for lbl in sorted(_EMBED_CHUNK_LABELS, key=str)
        for spec in (
            ExistenceConstraintSpec(label=lbl, property="uid"),
            ExistenceConstraintSpec(label=lbl, property="project_name"),
        )
    ],
    ExistenceConstraintSpec(label=NodeLabel.SCHEMA_VERSION, property="version"),
)

# Property indices for fast lookups
_INDEX_PROPERTIES: tuple[str, ...] = (
    "uid",
    "qualified_name",
    "file_path",
    "name",
    "project_name",
    "kind",
    "content_hash",
)

# The only properties a marker label is looked up by: uid, for the uid-only
# MATCHes the marker exists to index, and (project_name, name) for cross-project
# import resolution. Everything else is reached through a primary label, which
# has its own index -- indexing it on the marker too would buy no query anything
# and cost every single node write.
# ``embed_hash`` is here rather than in _INDEX_PROPERTIES on purpose. The
# embedding dedup lookup (ATL-127) asks "does ANY node anywhere already carry a
# vector for this text?" -- it is cross-label and cross-project by nature, so one
# index on the marker answers it in a single seek where six per-label indices
# would need six. It is also nearly free: a label-property index only holds nodes
# that HAVE the property, and only embeddable nodes ever carry embed_hash.
#
# Adding it to _INDEX_PROPERTIES instead would make generate_drop_redundant_marker_ddl
# emit a DROP for it, since that subtracts this tuple from that one.
_MARKER_INDEX_PROPERTIES: tuple[str, ...] = ("uid", "embed_hash")

# ``parent_uid`` is the one that earns its keep: every read of a chunk is "which node
# does this vector belong to", and every write of one is "drop this node's old chunks".
# ``embed_hash`` earns its place the same way it does on the marker: it is the key the
# dedup lookup seeks on. A chunk is an unusually good dedup source, because its
# embed_hash is the hash of its OWN text -- where a split parent's is the hash of the
# whole text, which no single vector corresponds to.
_EMBED_CHUNK_INDEX_PROPERTIES: tuple[str, ...] = ("uid", "parent_uid", "project_name", "embed_hash")

LABEL_PROPERTY_INDICES: tuple[IndexSpec, ...] = (
    *(
        IndexSpec(label=lbl, property=prop)
        for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)
        for prop in _INDEX_PROPERTIES
    ),
    *(
        IndexSpec(label=lbl, property=prop)
        for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value)
        for prop in _MARKER_INDEX_PROPERTIES
    ),
    *(
        IndexSpec(label=lbl, property=prop)
        for lbl in sorted(_EMBED_CHUNK_LABELS, key=lambda lbl: lbl.value)
        for prop in _EMBED_CHUNK_INDEX_PROPERTIES
    ),
)

# Composite property indices for multi-property lookups
_COMPOSITE_PROPERTIES: tuple[tuple[str, ...], ...] = (
    ("project_name", "file_path"),
    ("project_name", "name"),
)

_MARKER_COMPOSITE_PROPERTIES: tuple[tuple[str, ...], ...] = (("project_name", "name"),)

COMPOSITE_INDICES: tuple[CompositeIndexSpec, ...] = (
    *(
        CompositeIndexSpec(label=lbl, properties=props)
        for lbl in sorted(_ENTITY_LABELS, key=lambda lbl: lbl.value)
        for props in _COMPOSITE_PROPERTIES
    ),
    *(
        CompositeIndexSpec(label=lbl, properties=props)
        for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value)
        for props in _MARKER_COMPOSITE_PROPERTIES
    ),
)

# Text (BM25) indices — one per searchable label
TEXT_INDICES: tuple[TextIndexSpec, ...] = tuple(
    TextIndexSpec(name=f"text_{lbl.value.lower()}", label=lbl)
    for lbl in sorted(_TEXT_SEARCHABLE_LABELS, key=lambda lbl: lbl.value)
)


def build_vector_index_specs(dimension: int, capacity: int = 50_000) -> tuple[VectorIndexSpec, ...]:
    """Build vector index specs from settings (dimension is runtime config)."""
    return tuple(
        VectorIndexSpec(
            name=f"vec_{lbl.value.lower()}",
            label=lbl,
            property="embedding",
            dimension=dimension,
            capacity=capacity,
        )
        for lbl in sorted(_EMBEDDABLE_LABELS, key=lambda lbl: lbl.value)
    )


# ---------------------------------------------------------------------------
# Cypher fragments
# ---------------------------------------------------------------------------


def primary_label_expr(var: str = "n") -> str:
    """Cypher expression for *var*'s own label, ignoring any marker label.

    Every entity node carries a marker (:Entity) alongside the label saying what
    it actually is, so ``labels(n)[0]`` is not reliably that label any more.
    Memgraph returns labels in write order, which makes "primary label first" a
    convention the write sites happen to keep, not something the database
    promises -- and a node written ``:Entity:Callable`` would silently report its
    type as ``Entity`` in every search result, diagram and impact row.

    Filtering markers out instead survives a marker being added, reordered, or
    missing altogether (a node written before the marker existed still yields its
    real label rather than null).
    """
    markers = ", ".join(f"'{lbl.value}'" for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value))
    return f"[_lbl IN labels({var}) WHERE NOT _lbl IN [{markers}]][0]"


# ---------------------------------------------------------------------------
# DDL generation (pure functions, no I/O)
# ---------------------------------------------------------------------------


def generate_unique_constraint_ddl() -> list[str]:
    """Generate CREATE CONSTRAINT statements for unique properties."""
    return [
        f"CREATE CONSTRAINT ON (n:{spec.label.value}) ASSERT n.{spec.property} IS UNIQUE;"
        for spec in UNIQUE_CONSTRAINTS
    ]


def generate_existence_constraint_ddl() -> list[str]:
    """Generate CREATE CONSTRAINT statements for mandatory properties."""
    return [
        f"CREATE CONSTRAINT ON (n:{spec.label.value}) ASSERT EXISTS (n.{spec.property});"
        for spec in EXISTENCE_CONSTRAINTS
    ]


def generate_index_ddl() -> list[str]:
    """Generate CREATE INDEX statements for label-property pairs."""
    return [f"CREATE INDEX ON :{spec.label.value}({spec.property});" for spec in LABEL_PROPERTY_INDICES]


def generate_composite_index_ddl() -> list[str]:
    """Generate CREATE INDEX statements for composite (multi-property) indices."""
    return [f"CREATE INDEX ON :{spec.label.value}({', '.join(spec.properties)});" for spec in COMPOSITE_INDICES]


def generate_vector_index_ddl(dimension: int, capacity: int = 50_000) -> list[str]:
    """Generate CREATE VECTOR INDEX statements for embeddable labels (Memgraph 3.7+ DDL)."""
    specs = build_vector_index_specs(dimension, capacity)
    return [
        (
            f"CREATE VECTOR INDEX {spec.name} ON :{spec.label.value}({spec.property})"
            f' WITH CONFIG {{"dimension": {spec.dimension}, "capacity": {spec.capacity}, "metric": "{spec.metric}"}};'
        )
        for spec in specs
    ]


def generate_text_index_ddl() -> list[str]:
    """Generate CREATE TEXT INDEX statements for BM25 searchable labels (Memgraph 3.7+ DDL)."""
    return [f"CREATE TEXT INDEX {spec.name} ON :{spec.label.value};" for spec in TEXT_INDICES]


def generate_drop_vector_index_ddl() -> list[str]:
    """Generate DROP statements for all vector indices (Memgraph 3.7+ DDL)."""
    return [
        f"DROP VECTOR INDEX {spec.name};"
        for spec in build_vector_index_specs(0)  # dimension irrelevant for drops
    ]


def generate_drop_text_index_ddl() -> list[str]:
    """Generate DROP statements for all text indices (Memgraph 3.7+ DDL)."""
    return [f"DROP TEXT INDEX {spec.name};" for spec in TEXT_INDICES]


def generate_clear_file_hashes_ddl() -> str:
    """Cypher clearing every stored gate property, returning a file to "never indexed".

    Both of them: ``file_hash`` (does this file need re-parsing?) and ``rels_hash``
    (does its relationship set need rewriting?). One statement rather than two, because
    a caller that opened one gate and left the other shut would re-parse a file and then
    decline to rewrite its edges -- the single place that has to cover both, or a future
    migration heals one gate and not the other. The predicate matches on *either* being
    present for the same reason: a node carrying only a ``rels_hash`` must not be missed.

    No longer what migrations use. Twelve of them called this to force a re-parse until
    ATL-152 gave the gate an ``EXTRACTION_EPOCH``: a migration whose new data can only
    come from re-reading the source now bumps the epoch, which invalidates every stored
    hash by changing the key rather than by deleting 845 values. What remains is the
    manual lever -- staging the "no stored hash" state by hand, or in a test that needs
    it -- and it is still derived from FILE_HASH_LABELS so a label added to the gate is
    covered without anyone remembering to add it here.
    """
    predicate = " OR ".join(f"n:{label.value}" for label in FILE_HASH_LABELS)
    return (
        f"MATCH (n) WHERE ({predicate}) AND (n.file_hash IS NOT NULL OR n.rels_hash IS NOT NULL) "
        "REMOVE n.file_hash, n.rels_hash"
    )


def generate_drop_redundant_marker_ddl() -> list[str]:
    """Generate DROP statements for marker-label indices and constraints nothing queries.

    Schema v13 introduced :Entity by folding _MARKER_LABELS into _ENTITY_LABELS --
    the set that drives every constraint and index registry -- so the marker picked
    up all seven property indices, both composites, and uid/project_name uniqueness
    and existence. A marker sits on *every node in the graph*, so that is nine index
    structures and three constraint checks paid by every single write, from a label
    whose entire purpose was to make writes cheaper.

    Only _MARKER_INDEX_PROPERTIES and _MARKER_COMPOSITE_PROPERTIES are ever reached
    through the marker. Everything else is reached through a primary label that has
    its own index, and uid/project_name are already unique and mandatory there.

    Schema application is additive -- _apply_full_schema/_migrate_indices only ever
    CREATE -- so a database that ran v13 keeps all nine until these statements run.
    """
    stmts: list[str] = []
    for lbl in sorted(_MARKER_LABELS, key=lambda lbl: lbl.value):
        stmts.extend(
            f"DROP INDEX ON :{lbl.value}({prop});"
            for prop in sorted(set(_INDEX_PROPERTIES) - set(_MARKER_INDEX_PROPERTIES))
        )
        stmts.extend(
            f"DROP INDEX ON :{lbl.value}({', '.join(props)});"
            for props in sorted(set(_COMPOSITE_PROPERTIES) - set(_MARKER_COMPOSITE_PROPERTIES))
        )
        stmts.append(f"DROP CONSTRAINT ON (n:{lbl.value}) ASSERT n.uid IS UNIQUE;")
        stmts.extend(
            f"DROP CONSTRAINT ON (n:{lbl.value}) ASSERT EXISTS (n.{prop});" for prop in ("project_name", "uid")
        )
    return stmts


# ---------------------------------------------------------------------------
# Import-time validation
# ---------------------------------------------------------------------------


def _validate_schema_completeness() -> None:
    """Ensure every NodeLabel is covered by constraint/index registries.

    Raises RuntimeError at import time if any label is missing — prevents
    the silent-drop bug (competitor insight P0).
    """
    all_labels = set(NodeLabel)
    # A marker never carries uid/project_name of its own -- it is stamped onto a
    # node that already has both, constrained by its primary label. Constraining
    # them again on the marker would re-check every write for nothing.
    constrained_labels = all_labels - _MARKER_LABELS

    # Unique constraints: every non-marker label must appear
    unique_labels = {spec.label for spec in UNIQUE_CONSTRAINTS}
    missing_unique = constrained_labels - unique_labels
    if missing_unique:
        raise RuntimeError(f"NodeLabels missing from UNIQUE_CONSTRAINTS: {missing_unique}")

    # Existence constraints: every non-marker label must appear
    existence_labels = {spec.label for spec in EXISTENCE_CONSTRAINTS}
    missing_existence = constrained_labels - existence_labels
    if missing_existence:
        raise RuntimeError(f"NodeLabels missing from EXISTENCE_CONSTRAINTS: {missing_existence}")

    # Label property indices: entity and marker labels must appear (SchemaVersion exempt).
    # Markers are here on purpose -- being indexed is the only reason one exists.
    index_labels = {spec.label for spec in LABEL_PROPERTY_INDICES}
    missing_index = (_ENTITY_LABELS | _MARKER_LABELS) - index_labels
    if missing_index:
        raise RuntimeError(f"Entity labels missing from LABEL_PROPERTY_INDICES: {missing_index}")

    # Label groupings must cover all non-meta labels
    grouped = (
        _CODE_LABELS
        | _DOC_LABELS
        | _EXTERNAL_LABELS
        | _MARKER_LABELS
        | _EMBED_CHUNK_LABELS
        | {NodeLabel.SCHEMA_VERSION}
    )
    missing_group = all_labels - grouped
    if missing_group:
        raise RuntimeError(f"NodeLabels not in any label group: {missing_group}")


_validate_schema_completeness()
