"""Shell support — tree-sitter parser for Bash/sh/Zsh scripts.

Emits one ``Module`` per script, one ``Callable`` per ``function_definition``,
``CALLS`` between functions defined in the same file, and ``IMPORTS`` for
sourced scripts plus a curated set of infrastructure CLIs.

Design decisions
----------------
**Flat function namespace.** Shell has no lexical scoping: once a
``function_definition`` *executes*, the function is callable from anywhere,
including from a function defined earlier in the file. So a function nested in a
conditional, a loop, or another function still gets a top-level
``{module_qn}.{name}`` qualified name; only a ``conditional``/``nested`` tag
records where it was written.

**CALLS is same-file only.** A bare ``command_name`` becomes a ``CALLS`` edge
only when it matches a function defined in *this* file, so ``resolve_calls``
resolves it via its same-file strategy. Emitting every unmatched command name as
a bare-name CALLS would hand ``resolve_calls`` its project-wide strategy and
mint cross-language false positives (a shell script running ``build`` linking to
some Python ``build()``).

**Sourced scripts only when the path is literal.** ``source ./lib/x.sh`` is
resolved against the *sourcing script's own directory* — bash actually resolves
it against ``$PWD``, but the script's own directory is the near-universal intent
and is what ShellCheck's ``source=`` directive assumes. Interpolated paths
(``source "$SCRIPT_DIR/x.sh"``) are deliberately NOT guessed at; the only escape
hatch is an authoritative ``# shellcheck source=<path>`` directive on the line
above, which is machine-readable by design and costs no heuristics. Absolute
paths (``/etc/profile.d/x.sh``) are skipped: they can never be repo files, and
``resolve_imports`` would mint a meaningless ``ExternalPackage`` for ``etc``.

**External commands: curated allowlist, not a builtin denylist.** Every invoked
command *is* a fact, but a denylist still emits one ``ExternalPackage`` per
coreutil (``sed``, ``cut``, ``date``, ``wc``, ...) — hundreds of nodes of no
query value. The allowlist below keeps the queries that motivate the feature
("what touches ``kubectl``/``terraform``/``aws``") at a bounded, reviewable node
count. Edges hang off the ``Module``, not each ``Callable``, so the count is
O(distinct tools per file) rather than O(invocations), and ``IMPORTS`` ->
``ExternalPackage`` is exactly how ``import os`` is already modelled.

**Invoked in-repo scripts are NOT linked.** ``./scripts/other.sh`` resolves
against the runtime ``$PWD``, which is genuinely unknowable statically (unlike
``source``, which has ShellCheck's script-relative convention behind it). A
guess here produces either wrong edges or ``ExternalPackage`` noise.

Grammar notes (measured, tree-sitter-bash ABI 15):
  - Root is ``program``. ``function_definition`` exposes ``name`` (always a
    ``word``) and ``body`` fields for all three spellings — ``foo() {}``,
    ``function foo {}``, ``function foo() {}`` — and ``body`` is a
    ``compound_statement`` or, for ``foo() ( ... )``, a ``subshell``.
  - Heredocs are safe by construction: the payload is one opaque
    ``heredoc_body`` leaf, so text that looks like a definition or a call inside
    a heredoc is never tokenised into ``function_definition``/``command``.
  - Zsh-only syntax degrades to ``ERROR`` nodes rather than failing the parse,
    and definitions elsewhere in the file still extract normally.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from code_atlas.parsing.ast import (
    LanguageConfig,
    ParsedEntity,
    ParsedFile,
    ParsedRelationship,
    node_text,
    register_language,
)
from code_atlas.schema import CallableKind, NodeLabel, RelType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Node

_EXTENSIONS = frozenset({".sh", ".bash", ".zsh"})

_SOURCE_COMMANDS = frozenset({"source", "."})

# Prefixes that stand in front of the command actually being run. Unwrapping
# them is what keeps `sudo docker ...` from reading as an invocation of `sudo`.
_COMMAND_WRAPPERS = frozenset({"sudo", "command", "exec", "nohup", "time", "env", "xargs"})

# Node types whose presence in a string means its value depends on runtime
# state, so the string is not a usable literal path.
_INTERPOLATION_TYPES = frozenset(
    {
        "expansion",
        "simple_expansion",
        "command_substitution",
        "arithmetic_expansion",
        "process_substitution",
    }
)

_FUNCTION_BODY_TYPES = frozenset({"compound_statement", "subshell"})

# ShellCheck's own inline annotation for a source it cannot resolve either.
# `\bsource=` deliberately does not match the sibling `source-path=` directive.
_SHELLCHECK_SOURCE_RE = re.compile(r"shellcheck\b[^\n]*?\bsource=(\S+)")

# Infrastructure/ops tooling whose repo-wide usage is a question people actually
# ask. Deliberately excludes coreutils, text processing, linters and test
# runners — see the module docstring for why this is an allowlist.
_TRACKED_COMMANDS = frozenset(
    {
        # Containers and orchestration
        "docker",
        "docker-compose",
        "podman",
        "nerdctl",
        "kubectl",
        "helm",
        "kustomize",
        "skaffold",
        # Infrastructure as code / configuration management
        "terraform",
        "terragrunt",
        "tofu",
        "pulumi",
        "ansible",
        "ansible-playbook",
        "packer",
        "vagrant",
        # Cloud CLIs
        "aws",
        "az",
        "gcloud",
        "gsutil",
        "doctl",
        "flyctl",
        "heroku",
        # Databases
        "psql",
        "pg_dump",
        "pg_restore",
        "mysql",
        "mysqldump",
        "mongosh",
        "redis-cli",
        "valkey-cli",
        "sqlite3",
        # Source hosting CLIs
        "git",
        "gh",
        "glab",
        # Build and package managers
        "make",
        "cmake",
        "ninja",
        "bazel",
        "just",
        "task",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "pip",
        "pip3",
        "uv",
        "uvx",
        "poetry",
        "pipx",
        "cargo",
        "go",
        "mvn",
        "gradle",
        "bundle",
        "gem",
        "composer",
        "dotnet",
        # Service management and remote execution
        "systemctl",
        "journalctl",
        "supervisorctl",
        "ssh",
        "scp",
        "rsync",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_qualified_name(file_path: str) -> str:
    """Convert a file path to a dotted qualified name, extension folded in.

    ``scripts/deploy.sh`` -> ``scripts.deploy_sh``;  ``a.b/x.sh`` -> ``a_b.x_sh``

    Unlike the code-language modules, the extension is *preserved* (its dot
    replaced) rather than stripped. ``qualified_name`` IS the graph uid, and
    ``build.sh`` sitting beside ``build.py`` is ordinary — stripping would make
    both files claim ``{project}:build`` and the later upsert would silently
    overwrite the earlier one.

    Dots are folded in *every* segment, not just the basename, for that same
    reason: ``.`` is the separator being built here, so a directory named
    ``a.b`` would fake a nesting level and make ``a.b/x.sh`` and ``a/b/x.sh``
    claim one uid.
    """
    p = PurePosixPath(file_path.replace("\\", "/"))
    return ".".join(part.replace(".", "_") for part in p.parts)


# Both walks below are iterative on purpose. Shell nesting is input-controlled
# and invisible to the framework's pre-parse block-depth guard, which measures
# *indentation*: a single unindented line of `a && b && c && ...` left-nests one
# `list` node per operator, and a generated or minified script can carry
# thousands. A recursive walk raises RecursionError past ~1000 levels, and
# ``parse_file`` turns that into "refuse the whole file" — so one long chain
# would cost every function in the script. An explicit stack is bounded by heap
# rather than by the C stack, which is why raising ``sys.setrecursionlimit``
# would not be an equivalent fix. ``reversed`` preserves document order.


def _iter_descendants(node: Node) -> Iterator[Node]:
    """Yield every descendant of *node* in document order."""
    stack = list(reversed(node.children))
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(current.children))


def _iter_commands(node: Node) -> Iterator[Node]:
    """Yield ``command`` nodes under *node*, never crossing into a nested function.

    Descends through pipelines, redirections, conditionals, case arms and
    command substitutions, so ``x=$(build foo)`` and ``! deploy`` both surface.
    ``function_definition`` subtrees are skipped so a nested function's calls are
    attributed to the nested function rather than to its enclosing scope.
    """
    stack = [child for child in reversed(node.children) if child.type != "function_definition"]
    while stack:
        current = stack.pop()
        if current.type == "command":
            yield current
        stack.extend(child for child in reversed(current.children) if child.type != "function_definition")


def _literal_text(node: Node) -> str | None:
    """Return *node*'s value when it is a compile-time literal, else ``None``."""
    if node.type == "word":
        return node_text(node)
    if node.type == "raw_string":
        return node_text(node).strip("'")
    if node.type == "string":
        if any(child.type in _INTERPOLATION_TYPES for child in node.children):
            return None
        return "".join(node_text(child) for child in node.children if child.type == "string_content")
    return None


def _command_name_of(command: Node) -> str | None:
    """Return the literal text of a ``command``'s ``command_name``, if literal."""
    for child in command.children:
        if child.type != "command_name":
            continue
        for inner in child.children:
            text = _literal_text(inner)
            if text:
                return text
        return None
    return None


def _wrapped_operands(command: Node) -> list[str]:
    """Literal operand tokens of a wrapper command, e.g. ``sudo -u app kubectl get``.

    *Every* operand is returned rather than just the first, because a wrapper's
    own value-taking flags (``sudo -u <user>``, ``xargs -n <count>``) are
    indistinguishable from the wrapped command without modelling each wrapper's
    option grammar. That is safe only because the sole caller filters against
    ``_TRACKED_COMMANDS``, which no flag value or subcommand realistically
    satisfies — which is also why this is *not* used for CALLS resolution, where
    a bare subcommand token could collide with a local function name.

    Scanning stops at the first non-literal argument rather than guessing past an
    interpolation.
    """
    operands: list[str] = []
    for child in command.children:
        if child.type == "command_name":
            continue
        text = _literal_text(child)
        if text is None:
            break
        if not text or text.startswith("-") or "=" in text:
            # A wrapper flag (`-u`) or an inline env assignment (`FOO=1`).
            continue
        operands.append(text)
    return operands


def _shellcheck_source_directives(root: Node, source: bytes) -> dict[int, str]:
    """Map 1-based comment line -> the path asserted by ``# shellcheck source=``."""
    if b"shellcheck" not in source:
        return {}
    directives: dict[int, str] = {}
    for node in _iter_descendants(root):
        if node.type != "comment":
            continue
        match = _SHELLCHECK_SOURCE_RE.search(node_text(node))
        if match is not None:
            directives[node.start_point[0] + 1] = match.group(1)
    return directives


def _resolve_script_path(from_dir: str, raw: str) -> str | None:
    """Normalise a literal relative path against *from_dir*, or ``None``.

    Returns ``None`` for absolute paths, home-relative paths, flags, anything
    still carrying shell syntax, and paths that climb above the project root.
    """
    if not raw or raw.startswith(("/", "~", "-")) or any(ch in raw for ch in "$`*?"):
        return None
    parts = [segment for segment in from_dir.split("/") if segment]
    for segment in raw.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(segment)
    if not parts:
        return None
    return "/".join(parts)


def _source_target(command: Node, from_dir: str, directives: dict[int, str]) -> str | None:
    """Resolve the script a ``source``/``.`` command pulls in, or ``None``."""
    raw: str | None = None
    for child in command.children:
        if child.type == "command_name":
            continue
        raw = _literal_text(child)
        break
    if not raw:
        # Interpolated or missing operand — only an explicit ShellCheck
        # directive on the line immediately above can speak for it.
        raw = directives.get(command.start_point[0])
    if not raw:
        return None
    return _resolve_script_path(from_dir, raw)


def _is_directive_comment(text: str) -> bool:
    """True for tooling directives that must not be mistaken for documentation."""
    return text.lstrip("#").strip().lower().startswith("shellcheck")


def _doc_comment_above(node: Node) -> str | None:
    """Collect the contiguous ``#`` comment run immediately preceding *node*."""
    lines: list[str] = []
    prev = node.prev_sibling
    expected = node.start_point[0] - 1
    while prev is not None and prev.type == "comment" and prev.end_point[0] == expected:
        text = node_text(prev)
        if text.startswith("#!") or _is_directive_comment(text):
            break
        lines.append(text.lstrip("#").strip())
        expected = prev.start_point[0] - 1
        prev = prev.prev_sibling
    if not lines:
        return None
    lines.reverse()
    return "\n".join(lines)


def _module_header(root: Node, source: bytes) -> tuple[str | None, str | None]:
    """Split the file preamble into ``(shebang, docstring)``."""
    leading: list[Node] = []
    following: Node | None = None
    for child in root.children:
        if child.type == "comment":
            leading.append(child)
            continue
        following = child
        break

    shebang: str | None = None
    if leading and source.startswith(b"#!") and leading[0].start_point[0] == 0:
        shebang = node_text(leading[0]).strip()
        leading = leading[1:]

    if following is not None and following.type == "function_definition":
        # A comment run touching the first function documents *that function*,
        # not the file — drop it so it is not claimed twice.
        expected = following.start_point[0] - 1
        while leading and leading[-1].end_point[0] == expected:
            expected = leading[-1].start_point[0] - 1
            leading.pop()

    body = [node_text(node).lstrip("#").strip() for node in leading]
    docstring = "\n".join(line for line in body if not _is_directive_comment(line))
    return shebang, docstring.strip() or None


def _definition_scope_tag(node: Node) -> str | None:
    """Tag recording where a ``function_definition`` was written.

    ``nested`` inside another function, ``conditional`` inside any other
    construct (``if``, ``case``, loop, subshell), ``None`` at file top level.
    """
    parent = node.parent
    if parent is None or parent.type == "program":
        return None
    while parent is not None and parent.type != "program":
        if parent.type == "function_definition":
            return "nested"
        parent = parent.parent
    return "conditional"


def _declaration_signature(node: Node, source: bytes) -> str | None:
    """The declaration text up to the body, preserving the spelling used.

    ``foo() {`` -> ``foo()``;  ``function foo {`` -> ``function foo``.
    """
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    return source[node.start_byte : end].decode("utf-8", errors="replace").strip() or None


def _named_functions(root: Node) -> list[tuple[Node, str]]:
    """Every ``function_definition`` with a usable name, in document order."""
    found: list[tuple[Node, str]] = []
    for node in _iter_descendants(root):
        if node.type != "function_definition":
            continue
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue
        name = node_text(name_node)
        if name:
            found.append((node, name))
    return found


def _collect_dependencies(
    commands: list[Node],
    from_dir: str,
    directives: dict[int, str],
    defined_names: set[str],
    targets: list[str],
) -> None:
    """Append sourced-script and tracked-tool IMPORTS targets found in *commands*."""
    for command in commands:
        primary = _command_name_of(command)
        if primary is None:
            continue
        if primary in _SOURCE_COMMANDS:
            resolved = _source_target(command, from_dir, directives)
            if resolved is not None:
                targets.append(_module_qualified_name(resolved))
            continue
        candidates = [primary]
        if primary in _COMMAND_WRAPPERS:
            candidates.extend(_wrapped_operands(command))
        # A local function shadows any same-named tool, so it is not a dependency.
        # This guard is also load-bearing for CALLS resolution: resolve_calls keys
        # its import_map on the *target node's* name and tries that strategy
        # first, so an IMPORTS -> ExternalPackage("task") emitted alongside a
        # local `task()` would make every call to `task` resolve to the external
        # package (which the Callable-scoped edge Cypher then silently drops)
        # instead of falling through to the same-file strategy.
        targets.extend(name for name in candidates if name in _TRACKED_COMMANDS and name not in defined_names)


def _parse_shell(path: str, source: bytes, root: Node, project_name: str) -> ParsedFile:
    """Extract entities and relationships from a shell parse tree."""
    norm_path = path.replace("\\", "/")
    language = "shell"

    if not source.strip():
        # No Module node for an empty file — it would be an unsearchable stub
        # that still costs an embedding.
        return ParsedFile(file_path=norm_path, language=language, entities=[], relationships=[])

    module_qn = _module_qualified_name(norm_path)
    module_uid = f"{project_name}:{module_qn}"
    parent_dir = str(PurePosixPath(norm_path).parent)
    from_dir = "" if parent_dir == "." else parent_dir

    directives = _shellcheck_source_directives(root, source)
    functions = _named_functions(root)
    defined_names = {name for _node, name in functions}

    # Commands outside every function body — the script's own top-level program.
    top_level_commands = list(_iter_commands(root))
    invoked_at_top_level = {name for command in top_level_commands if (name := _command_name_of(command))}

    shebang, module_doc = _module_header(root, source)
    entities: list[ParsedEntity] = [
        ParsedEntity(
            name=PurePosixPath(norm_path).name,
            qualified_name=module_uid,
            label=NodeLabel.MODULE,
            kind="shell_script",
            line_start=1,
            line_end=root.end_point[0] + 1,
            file_path=norm_path,
            docstring=module_doc,
            signature=shebang,
            # Deliberate deviation from the other language modules, which leave
            # Module.source unset: a shell script's top-level statements belong
            # to no child entity, so without this a function-less script (an
            # install or CI script) would be invisible to vector/BM25 search.
            source=node_text(root),
        )
    ]
    relationships: list[ParsedRelationship] = []
    import_targets: list[str] = []

    occurrences: dict[str, int] = {}
    for node, name in functions:
        occurrences[name] = occurrences.get(name, 0) + 1
        seen = occurrences[name]
        # Redefinition in one file is a bug, but it must not silently collapse
        # two Callables onto one uid.
        qn = f"{module_qn}.{name}" if seen == 1 else f"{module_qn}.{name}#{seen}"
        uid = f"{project_name}:{qn}"

        tags: list[str] = []
        scope_tag = _definition_scope_tag(node)
        if scope_tag is not None:
            tags.append(scope_tag)
        if name in invoked_at_top_level:
            tags.append("entry_point")

        entities.append(
            ParsedEntity(
                name=name,
                qualified_name=uid,
                label=NodeLabel.CALLABLE,
                kind=CallableKind.FUNCTION,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                file_path=norm_path,
                docstring=_doc_comment_above(node),
                signature=_declaration_signature(node, source),
                source=node_text(node),
                tags=tags,
            )
        )
        relationships.append(ParsedRelationship(from_qualified_name=module_uid, rel_type=RelType.DEFINES, to_name=uid))

        body = node.child_by_field_name("body")
        if body is None or body.type not in _FUNCTION_BODY_TYPES:
            continue
        body_commands = list(_iter_commands(body))
        called: set[str] = set()
        for command in body_commands:
            callee = _command_name_of(command)
            # Only the command name itself can name a function: `sudo`/`env`/
            # `command`/`xargs` exec a binary and cannot reach a shell function at
            # all, so their operands are never CALLS candidates. Self-calls are
            # dropped because resolve_calls excludes the caller from its own
            # candidate set — the edge could only land on a same-named
            # redefinition.
            if callee is None or callee == name or callee in called or callee not in defined_names:
                continue
            called.add(callee)
            relationships.append(ParsedRelationship(from_qualified_name=uid, rel_type=RelType.CALLS, to_name=callee))
        _collect_dependencies(body_commands, from_dir, directives, defined_names, import_targets)

    _collect_dependencies(top_level_commands, from_dir, directives, defined_names, import_targets)
    relationships.extend(
        ParsedRelationship(from_qualified_name=module_uid, rel_type=RelType.IMPORTS, to_name=target)
        for target in dict.fromkeys(import_targets)
    )

    return ParsedFile(file_path=norm_path, language=language, entities=entities, relationships=relationships)


# ---------------------------------------------------------------------------
# Language registration
# ---------------------------------------------------------------------------

try:
    import tree_sitter_bash as _ts_bash
    from tree_sitter import Language, Query

    _BASH_LANGUAGE = Language(_ts_bash.language())
    _BASH_QUERY = Query(_BASH_LANGUAGE, "(program) @root")

    register_language(
        LanguageConfig(
            name="shell",
            extensions=_EXTENSIONS,
            language=_BASH_LANGUAGE,
            query=_BASH_QUERY,
            parse_func=_parse_shell,
            comment_node_types=frozenset({"comment"}),
        )
    )
except ImportError:
    pass
