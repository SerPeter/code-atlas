"""Tests for shell (Bash/sh/Zsh) parser."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("tree_sitter_bash", reason="tree-sitter-bash not installed")

from code_atlas.parsing.ast import ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import CallableKind, NodeLabel, RelType

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "scripts/deploy.sh") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity_by_name(parsed: ParsedFile, name: str):
    matches = [e for e in parsed.entities if e.name == name]
    assert len(matches) == 1, (
        f"Expected 1 entity named {name!r}, got {len(matches)}: {[e.name for e in parsed.entities]}"
    )
    return matches[0]


def _rels_from(parsed: ParsedFile, from_qn_suffix: str, rel_type: RelType):
    return [
        r for r in parsed.relationships if r.from_qualified_name.endswith(from_qn_suffix) and r.rel_type == rel_type
    ]


def _call_targets(parsed: ParsedFile, from_qn_suffix: str) -> set[str]:
    return {r.to_name for r in _rels_from(parsed, from_qn_suffix, RelType.CALLS)}


def _import_targets(parsed: ParsedFile) -> list[str]:
    return [r.to_name for r in parsed.relationships if r.rel_type == RelType.IMPORTS]


def _run_isolated(body: str) -> subprocess.CompletedProcess[str]:
    """Run *body* in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["scripts/deploy.sh", "lib/util.bash", "dotfiles/aliases.zsh", "Build.SH"])
def test_language_detection_shell(path: str):
    cfg = get_language_for_file(path)
    assert cfg is not None
    assert cfg.name == "shell"


def test_language_detection_not_shell():
    assert get_language_for_file("data.csv") is None
    assert get_language_for_file("readme.txt") is None


# ---------------------------------------------------------------------------
# 2. Module entity
# ---------------------------------------------------------------------------


def test_module_entity():
    parsed = _parse("echo hi\n", path="scripts/deploy.sh")
    mod = _entity_by_name(parsed, "deploy.sh")
    assert mod.label == NodeLabel.MODULE
    assert mod.kind == "shell_script"
    # The extension is folded in, not stripped: deploy.sh must not collide with
    # a sibling deploy.py on the same uid.
    assert mod.qualified_name == f"{PROJECT}:scripts.deploy_sh"
    assert mod.line_start == 1


def test_module_source_is_populated():
    # Top-level statements belong to no child entity, so the Module carries them.
    parsed = _parse("echo hi\ntouch /tmp/marker\n")
    assert "touch /tmp/marker" in (_entity_by_name(parsed, "deploy.sh").source or "")


def test_empty_file_produces_no_entities():
    parsed = _parse("")
    assert parsed.entities == []
    assert parsed.relationships == []
    assert _parse("   \n\n").entities == []


# ---------------------------------------------------------------------------
# 3. Shebang and preamble handling
# ---------------------------------------------------------------------------


def test_shebang_becomes_module_signature():
    parsed = _parse("#!/usr/bin/env bash\nset -euo pipefail\necho hi\n")
    mod = _entity_by_name(parsed, "deploy.sh")
    assert mod.signature == "#!/usr/bin/env bash"


def test_module_docstring_from_header_comments():
    parsed = _parse("#!/bin/bash\n# Deploy the service.\n# Two lines.\nset -euo pipefail\n")
    mod = _entity_by_name(parsed, "deploy.sh")
    assert mod.docstring == "Deploy the service.\nTwo lines."


def test_header_comment_touching_first_function_belongs_to_the_function():
    parsed = _parse("#!/bin/bash\n# Log a line.\nlog() {\n  echo hi\n}\n")
    assert _entity_by_name(parsed, "deploy.sh").docstring is None
    assert _entity_by_name(parsed, "log").docstring == "Log a line."


def test_set_euo_pipefail_produces_no_entities_or_edges():
    parsed = _parse("#!/bin/bash\nset -euo pipefail\nshopt -s nullglob\n")
    assert [e.name for e in parsed.entities] == ["deploy.sh"]
    assert parsed.relationships == []


# ---------------------------------------------------------------------------
# 4. Function definitions — all three spellings
# ---------------------------------------------------------------------------


SOURCE_THREE_FORMS = """\
#!/bin/bash
posix_form() {
  echo a
}

function keyword_form {
  echo b
}

function keyword_parens() {
  echo c
}
"""


@pytest.mark.parametrize(
    ("name", "signature"),
    [
        ("posix_form", "posix_form()"),
        ("keyword_form", "function keyword_form"),
        ("keyword_parens", "function keyword_parens()"),
    ],
)
def test_function_forms(name: str, signature: str):
    parsed = _parse(SOURCE_THREE_FORMS)
    fn = _entity_by_name(parsed, name)
    assert fn.label == NodeLabel.CALLABLE
    assert fn.kind == CallableKind.FUNCTION
    assert fn.qualified_name == f"{PROJECT}:scripts.deploy_sh.{name}"
    assert fn.signature == signature


def test_function_lines_docstring_and_source():
    parsed = _parse("#!/bin/bash\n# Build the image.\n# Tag comes from $1.\nbuild() {\n  echo hi\n}\n")
    fn = _entity_by_name(parsed, "build")
    assert (fn.line_start, fn.line_end) == (4, 6)
    assert fn.docstring == "Build the image.\nTag comes from $1."
    assert fn.source is not None
    assert fn.source.startswith("build() {")


def test_shellcheck_directive_is_not_a_docstring():
    parsed = _parse("# shellcheck disable=SC2086\nrun() {\n  echo hi\n}\n")
    assert _entity_by_name(parsed, "run").docstring is None


def test_subshell_body_function():
    parsed = _parse("isolated() (\n  echo hi\n)\n")
    fn = _entity_by_name(parsed, "isolated")
    assert fn.signature == "isolated()"


def test_dashed_and_namespaced_names():
    parsed = _parse("my-func() {\n  echo a\n}\npkg::helper() {\n  echo b\n}\n")
    assert _entity_by_name(parsed, "my-func").qualified_name.endswith(".my-func")
    assert _entity_by_name(parsed, "pkg::helper").qualified_name.endswith(".pkg::helper")


# ---------------------------------------------------------------------------
# 5. Flat namespace: functions in conditionals and inside other functions
# ---------------------------------------------------------------------------


NESTED_SOURCE = """\
#!/bin/bash
top_level() {
  echo a
}

if [[ -n "${CI:-}" ]]; then
  ci_only() {
    echo b
  }
fi

outer() {
  inner() {
    echo c
  }
  inner
}
"""


def test_conditionally_defined_function_is_extracted_flat():
    parsed = _parse(NESTED_SOURCE)
    fn = _entity_by_name(parsed, "ci_only")
    # Shell has no lexical scoping — a conditional definition is still global.
    assert fn.qualified_name == f"{PROJECT}:scripts.deploy_sh.ci_only"
    assert fn.tags == ["conditional"]


def test_function_nested_in_function_is_extracted_flat():
    parsed = _parse(NESTED_SOURCE)
    fn = _entity_by_name(parsed, "inner")
    assert fn.qualified_name == f"{PROJECT}:scripts.deploy_sh.inner"
    assert fn.tags == ["nested"]


def test_top_level_function_has_no_scope_tag():
    assert _entity_by_name(_parse(NESTED_SOURCE), "top_level").tags == []


def test_nested_function_body_is_not_attributed_to_the_outer_function():
    parsed = _parse(NESTED_SOURCE)
    # `inner` is called by `outer`; `echo c` belongs to `inner`, not `outer`.
    assert _call_targets(parsed, ".outer") == {"inner"}


def test_duplicate_function_names_get_distinct_uids():
    parsed = _parse("foo() { echo 1; }\nfoo() { echo 2; }\n")
    uids = sorted(e.qualified_name for e in parsed.entities if e.label == NodeLabel.CALLABLE)
    assert uids == [f"{PROJECT}:scripts.deploy_sh.foo", f"{PROJECT}:scripts.deploy_sh.foo#2"]


# ---------------------------------------------------------------------------
# 6. DEFINES
# ---------------------------------------------------------------------------


def test_defines_module_to_every_function():
    parsed = _parse(NESTED_SOURCE)
    defines = _rels_from(parsed, ":scripts.deploy_sh", RelType.DEFINES)
    assert {r.to_name for r in defines} == {
        f"{PROJECT}:scripts.deploy_sh.{n}" for n in ("top_level", "ci_only", "outer", "inner")
    }


# ---------------------------------------------------------------------------
# 7. CALLS between functions in the same file
# ---------------------------------------------------------------------------


CALLS_SOURCE = """\
#!/bin/bash
log_info() {
  printf '%s\\n' "$*"
}

build() {
  log_info "building"
  docker build -t x .
}

main() {
  build
  x=$(build extra)
  echo hi | log_info
  ! build
  case "$1" in
    a) log_info a ;;
  esac
}

main "$@"
"""


def test_calls_between_functions():
    parsed = _parse(CALLS_SOURCE)
    assert _call_targets(parsed, ".build") == {"log_info"}
    assert _call_targets(parsed, ".main") == {"build", "log_info"}


def test_calls_reach_into_substitutions_pipelines_and_case_arms():
    parsed = _parse(CALLS_SOURCE)
    # `x=$(build extra)`, `echo hi | log_info`, `! build` and the case arm are all
    # nested constructs — one CALLS per distinct callee, de-duplicated.
    assert len(_rels_from(parsed, ".main", RelType.CALLS)) == 2


def test_external_commands_are_not_calls():
    parsed = _parse(CALLS_SOURCE)
    assert "docker" not in _call_targets(parsed, ".build")
    assert "printf" not in _call_targets(parsed, ".log_info")


def test_no_calls_emitted_from_module_scope():
    parsed = _parse(CALLS_SOURCE)
    # `main "$@"` at the bottom cannot be a CALLS edge (Module is not a Callable);
    # it surfaces as the entry_point tag instead.
    assert _rels_from(parsed, ":scripts.deploy_sh", RelType.CALLS) == []
    assert "entry_point" in _entity_by_name(parsed, "main").tags
    assert "entry_point" not in _entity_by_name(parsed, "build").tags


def test_self_recursive_call_is_not_emitted():
    parsed = _parse("recurse() {\n  recurse\n}\n")
    assert _rels_from(parsed, ".recurse", RelType.CALLS) == []


# ---------------------------------------------------------------------------
# 8. IMPORTS — sourced scripts
# ---------------------------------------------------------------------------


def test_source_literal_relative_paths():
    parsed = _parse(
        "source lib/log.sh\n. ./lib/util.sh\nsource '../common.sh'\nsource \"helpers.sh\"\n",
    )
    assert _import_targets(parsed) == [
        "scripts.lib.log_sh",
        "scripts.lib.util_sh",
        "common_sh",
        "scripts.helpers_sh",
    ]


def test_source_from_repo_root_script():
    parsed = _parse("source lib/x.sh\n", path="deploy.sh")
    assert _import_targets(parsed) == ["lib.x_sh"]


def test_source_inside_a_function_still_belongs_to_the_module():
    parsed = _parse("boot() {\n  source ./lib/log.sh\n}\n")
    rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert [(r.from_qualified_name, r.to_name) for r in rels] == [
        (f"{PROJECT}:scripts.deploy_sh", "scripts.lib.log_sh")
    ]


def test_interpolated_source_paths_are_not_guessed():
    parsed = _parse(
        'source "$SCRIPT_DIR/lib/log.sh"\n'
        'source "${BASH_SOURCE[0]%/*}/x.sh"\n'
        'source "$(dirname "$0")/y.sh"\n'
        "source $CONFIG\n"
        "source\n",
    )
    assert _import_targets(parsed) == []


def test_absolute_and_home_source_paths_are_skipped():
    parsed = _parse("source /etc/profile.d/x.sh\n. ~/.bashrc\n")
    assert _import_targets(parsed) == []


def test_source_path_escaping_the_project_root_is_skipped():
    parsed = _parse("source ../../outside.sh\n", path="scripts/deploy.sh")
    assert _import_targets(parsed) == []


def test_shellcheck_source_directive_resolves_a_dynamic_source():
    parsed = _parse(
        '# shellcheck source=./lib/log.sh\nsource "$SCRIPT_DIR/lib/log.sh"\n',
    )
    assert _import_targets(parsed) == ["scripts.lib.log_sh"]


def test_shellcheck_source_directive_alongside_disable():
    parsed = _parse(
        '# shellcheck disable=SC1091 source=lib/util.sh\n. "$LIB/util.sh"\n',
    )
    assert _import_targets(parsed) == ["scripts.lib.util_sh"]


def test_shellcheck_source_dev_null_yields_nothing():
    parsed = _parse('# shellcheck source=/dev/null\nsource "$LIB/x.sh"\n')
    assert _import_targets(parsed) == []


def test_shellcheck_source_path_directive_is_not_mistaken_for_source():
    parsed = _parse('# shellcheck source-path=SCRIPTDIR\nsource "$LIB/x.sh"\n')
    assert _import_targets(parsed) == []


def test_imports_are_deduplicated():
    parsed = _parse("source ./lib/log.sh\ndocker ps\nsource lib/log.sh\ndocker build .\n")
    assert _import_targets(parsed) == ["scripts.lib.log_sh", "docker"]


# ---------------------------------------------------------------------------
# 9. IMPORTS — tracked external commands (curated allowlist)
# ---------------------------------------------------------------------------


def test_tracked_commands_become_module_imports():
    parsed = _parse("kubectl apply -f x.yaml\nterraform plan\n")
    rels = [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS]
    assert all(r.from_qualified_name == f"{PROJECT}:scripts.deploy_sh" for r in rels)
    assert [r.to_name for r in rels] == ["kubectl", "terraform"]


def test_untracked_commands_are_ignored():
    parsed = _parse("sed -i s/a/b/ f\nawk '{print}' f\ncut -d, -f1 f\nwc -l f\ndate\n")
    assert _import_targets(parsed) == []


def test_tracked_command_inside_a_function_and_a_pipeline():
    parsed = _parse("dump() {\n  pg_dump db | gzip > out.gz\n}\n")
    assert _import_targets(parsed) == ["pg_dump"]


def test_wrapper_prefixes_are_unwrapped():
    parsed = _parse("sudo docker ps\nsudo -u app kubectl get pods\nenv FOO=1 terraform init\n")
    assert _import_targets(parsed) == ["docker", "kubectl", "terraform"]


def test_wrapper_operands_are_not_call_candidates():
    # `sudo docker build` must not read as a call to the local `build()` — sudo
    # execs a binary and cannot reach a shell function.
    parsed = _parse("build() {\n  echo hi\n}\nmain() {\n  sudo docker build .\n}\n")
    assert _call_targets(parsed, ".main") == set()
    assert _import_targets(parsed) == ["docker"]


def test_local_function_shadows_a_tracked_command_name():
    parsed = _parse("task() {\n  echo hi\n}\nmain() {\n  task build\n}\n")
    assert _import_targets(parsed) == []
    assert _call_targets(parsed, ".main") == {"task"}


def test_interpolated_command_name_is_ignored():
    parsed = _parse('"$SCRIPT_DIR/nested.sh" arg\n"$TOOL" ps\n')
    assert _import_targets(parsed) == []


def test_invoked_repo_script_is_not_linked():
    # Relative command paths resolve against the runtime $PWD, which is not
    # statically knowable — see the module docstring.
    parsed = _parse("./scripts/other.sh\nbash lib/x.sh\n")
    assert _import_targets(parsed) == []


# ---------------------------------------------------------------------------
# 10. Robustness — heredocs, broken syntax, Zsh, CRLF
# ---------------------------------------------------------------------------


HEREDOC_SOURCE = """\
#!/bin/bash
real_fn() {
  echo hi
}

apply() {
  kubectl apply -f - <<EOF
fake_fn() {
  echo not-a-function
}
real_fn
source ./lib/not-imported.sh
docker ps
EOF
  real_fn
}
"""


def test_heredoc_body_does_not_leak_entities_or_edges():
    parsed = _parse(HEREDOC_SOURCE)
    assert sorted(e.name for e in parsed.entities) == ["apply", "deploy.sh", "real_fn"]
    assert _call_targets(parsed, ".apply") == {"real_fn"}
    # `source ./lib/not-imported.sh` and `docker ps` live inside the heredoc; only
    # the real `kubectl` invocation is a dependency.
    assert _import_targets(parsed) == ["kubectl"]


def test_quoted_and_indented_heredoc():
    parsed = _parse("f() {\ncat <<-'EOF'\ng() { echo x; }\nf arg\nEOF\n}\n")
    assert sorted(e.name for e in parsed.entities) == ["deploy.sh", "f"]
    assert _rels_from(parsed, ".f", RelType.CALLS) == []


def test_unbalanced_braces_do_not_crash():
    parsed = _parse("broken() {\n  echo hi\n")
    # An unterminated body is an ERROR node, not a function_definition — the file
    # still parses to a Module rather than raising.
    assert [e.name for e in parsed.entities] == ["deploy.sh"]


def test_zsh_specific_syntax_still_yields_functions():
    parsed = _parse(
        "typeset -A map\nmap[k]=v\nemit() {\n  print -r -- $map[k]\n}\nemit\n",
        path="dotfiles/aliases.zsh",
    )
    fn = _entity_by_name(parsed, "emit")
    assert fn.label == NodeLabel.CALLABLE
    assert "entry_point" in fn.tags


def test_crlf_line_endings():
    parsed = _parse("#!/bin/bash\r\nbuild() {\r\n  echo hi\r\n}\r\nbuild\r\n")
    fn = _entity_by_name(parsed, "build")
    assert (fn.line_start, fn.line_end) == (2, 4)
    assert "entry_point" in fn.tags


def test_comment_only_file():
    parsed = _parse("#!/bin/bash\n# Nothing but notes.\n")
    mod = _entity_by_name(parsed, "deploy.sh")
    assert mod.signature == "#!/bin/bash"
    assert mod.docstring == "Nothing but notes."


# ---------------------------------------------------------------------------
# 11. Content hash
# ---------------------------------------------------------------------------


def test_content_hash_is_set_and_position_independent():
    a = _parse("build() {\n  echo hi\n}\n")
    b = _parse("# leading comment\n\nbuild() {\n  echo hi\n}\n")
    fn_a, fn_b = _entity_by_name(a, "build"), _entity_by_name(b, "build")
    assert fn_a.content_hash
    assert fn_a.content_hash == fn_b.content_hash


def test_content_hash_changes_with_body():
    a = _entity_by_name(_parse("build() {\n  echo hi\n}\n"), "build")
    b = _entity_by_name(_parse("build() {\n  echo bye\n}\n"), "build")
    assert a.content_hash != b.content_hash


# ---------------------------------------------------------------------------
# 12. Uid collisions and hostile nesting
# ---------------------------------------------------------------------------


def test_a_dotted_directory_does_not_collide_with_a_nested_one():
    """``.`` is the qualified-name separator, so it must be folded in every segment."""
    dotted = _parse("build() {\n  echo hi\n}\n", path="a.b/x.sh")
    nested = _parse("build() {\n  echo hi\n}\n", path="a/b/x.sh")
    assert _entity_by_name(dotted, "x.sh").qualified_name == f"{PROJECT}:a_b.x_sh"
    assert _entity_by_name(nested, "x.sh").qualified_name == f"{PROJECT}:a.b.x_sh"
    assert _entity_by_name(dotted, "build").qualified_name == f"{PROJECT}:a_b.x_sh.build"
    assert _entity_by_name(nested, "build").qualified_name == f"{PROJECT}:a.b.x_sh.build"


def test_long_operator_chain_does_not_exhaust_the_stack():
    """One unindented line of 5000 ``&&`` left-nests 5000 ``list`` nodes.

    The framework's pre-parse guard measures *indentation* depth, so it sees a
    file one line deep and passes this straight through to the handler. Run
    out-of-process: if either tree walk regressed to recursion the failure is a
    stack overflow, which is not always a catchable ``RecursionError`` —
    in-process it could take the test session down with it.
    """
    proc = _run_isolated(
        """
        from code_atlas.parsing.ast import parse_file

        src = "build() {\\n  echo hi\\n}\\n" + " && ".join(["build"] * 5000) + "\\n"
        result = parse_file("scripts/deep.sh", src.encode(), "p")
        assert result is not None, "parse_file refused the file outright"
        names = sorted(e.name for e in result.entities)
        assert names == ["build", "deep.sh"], names
        # The chain is top-level, so `build` is reachable as an entry point —
        # which is only knowable by walking every node of the chain.
        build = next(e for e in result.entities if e.name == "build")
        assert "entry_point" in build.tags, build.tags
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_deeply_nested_subshells_do_not_exhaust_the_stack():
    """Same hazard from grammar nesting rather than a flat chain.

    ``( ( ( ... ) ) )`` needs the spaces: ``(((run)))`` is one
    ``parenthesized_expression``, not three subshells.
    """
    proc = _run_isolated(
        """
        from code_atlas.parsing.ast import parse_file

        depth = 3000
        src = "run() {\\n  echo hi\\n}\\n" + "( " * depth + "run" + " )" * depth + "\\n"
        result = parse_file("scripts/nested.sh", src.encode(), "p")
        assert result is not None, "parse_file refused the file outright"
        run = next(e for e in result.entities if e.name == "run")
        assert "entry_point" in run.tags, run.tags
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
