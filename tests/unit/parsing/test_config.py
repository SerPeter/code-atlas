"""Tests for structured config parsing — YAML/JSON/TOML/XML dialect detection."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

pytest.importorskip("tree_sitter_yaml", reason="tree-sitter-yaml not installed")
pytest.importorskip("tree_sitter_json", reason="tree-sitter-json not installed")
pytest.importorskip("tree_sitter_toml", reason="tree-sitter-toml not installed")
pytest.importorskip("tree_sitter_xml", reason="tree-sitter-xml not installed")

from tree_sitter import Parser

from code_atlas.parsing.ast import ParsedEntity, ParsedFile, get_language_for_file, parse_file
from code_atlas.parsing.languages.config import (
    MAX_GENERIC_CONFIG_BYTES,
    _load_yaml_documents,
    _Out,
    _parse_config,
)
from code_atlas.schema import NodeLabel, RelType

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str) -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity_by_name(parsed: ParsedFile, name: str) -> ParsedEntity:
    matches = [e for e in parsed.entities if e.name == name]
    assert len(matches) == 1, (
        f"Expected 1 entity named {name!r}, got {len(matches)}: {[e.name for e in parsed.entities]}"
    )
    return matches[0]


def _rels_from(parsed: ParsedFile, from_qn_suffix: str, rel_type: RelType):
    return [
        r for r in parsed.relationships if r.from_qualified_name.endswith(from_qn_suffix) and r.rel_type == rel_type
    ]


def _targets(parsed: ParsedFile, from_qn_suffix: str, rel_type: RelType) -> set[str]:
    return {r.to_name for r in _rels_from(parsed, from_qn_suffix, rel_type)}


def _decline(source: str, path: str) -> None:
    """Assert the handler declines *path* outright — no entities, no edges.

    Checked at both levels: ``_parse_config`` returns ``None`` (the contract that
    keeps *data* out of the graph — unrecognised config now falls back to a
    generic key tree instead) and ``parse_file`` normalises that into an empty
    ``ParsedFile`` (so the file is still hashed and not re-parsed on the next
    indexing pass).
    """
    config = get_language_for_file(path)
    assert config is not None, f"no language registered for {path}"
    raw = source.encode("utf-8")
    root = Parser(config.language).parse(raw).root_node
    assert _parse_config(path, raw, root, PROJECT) is None

    normalized = parse_file(path, raw, PROJECT)
    assert normalized is not None
    assert normalized.entities == []
    assert normalized.relationships == []


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("deploy/app.yaml", "yaml"),
        ("deploy/app.yml", "yaml"),
        ("manifests/pod.json", "json"),
        ("pyproject.toml", "toml"),
        ("force-app/Flow.flow-meta.xml", "xml"),
    ],
)
def test_language_detection_config(path: str, expected: str) -> None:
    config = get_language_for_file(path)
    assert config is not None
    assert config.name == expected


def test_language_detection_not_config() -> None:
    assert get_language_for_file("data.csv") is None
    assert get_language_for_file("readme.txt") is None


# ---------------------------------------------------------------------------
# 2. Kubernetes — apiVersion + kind + metadata.name is the only marker
# ---------------------------------------------------------------------------

K8S_DEPLOYMENT = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  namespace: prod
spec:
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      serviceAccountName: web-sa
      imagePullSecrets:
        - name: regcred
      volumes:
        - name: cfg
          configMap:
            name: app-config
        - name: tls
          secret:
            secretName: web-tls
      containers:
        - name: api
          image: ghcr.io/acme/api:1.2.3
          envFrom:
            - configMapRef:
                name: env-config
          env:
            - name: PW
              valueFrom:
                secretKeyRef:
                  name: db-password
                  key: pw
"""


def test_k8s_resource_entity() -> None:
    parsed = _parse(K8S_DEPLOYMENT, "deploy/app.yaml")

    module = _entity_by_name(parsed, "app.yaml")
    assert module.label == NodeLabel.MODULE
    assert module.kind == "k8s_manifest"

    # Named `Kind/name`, not a bare `web`: a Service and a Deployment sharing one
    # name is the single most common k8s naming pattern, and the reference edges
    # resolve by name.
    resource = _entity_by_name(parsed, "Deployment/web")
    assert resource.label == NodeLabel.TYPE_DEF
    assert resource.kind == "k8s_resource"
    assert resource.extra_properties["k8s_kind"] == "Deployment"
    assert resource.extra_properties["api_version"] == "apps/v1"
    assert resource.extra_properties["resource_name"] == "web"
    assert resource.extra_properties["namespace"] == "prod"
    assert resource.line_start == 1

    assert _targets(parsed, ":deploy.app_yaml", RelType.DEFINES) == {f"{PROJECT}:deploy.app_yaml.Deployment_web"}


def test_k8s_reference_edges() -> None:
    parsed = _parse(K8S_DEPLOYMENT, "deploy/app.yaml")

    assert _targets(parsed, ".Deployment_web", RelType.USES_TYPE) == {
        "ServiceAccount/web-sa",
        "Secret/regcred",
        "ConfigMap/app-config",
        "Secret/web-tls",
        "ConfigMap/env-config",
        "Secret/db-password",
    }


def test_k8s_image_ref_is_external_import() -> None:
    parsed = _parse(K8S_DEPLOYMENT, "deploy/app.yaml")

    # Tag stripped; the registry host survives so two registries never collide.
    assert _targets(parsed, ".Deployment_web", RelType.IMPORTS) == {"ghcr.io/acme/api"}


def test_k8s_image_key_without_container_name_is_not_an_image() -> None:
    source = """\
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  image: this-is-a-config-value
"""
    parsed = _parse(source, "deploy/cm.yaml")
    assert _targets(parsed, ".ConfigMap_app-config", RelType.IMPORTS) == set()


def test_k8s_multi_document_yields_one_entity_per_document() -> None:
    source = f"""\
{K8S_DEPLOYMENT}---
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  selector:
    app: web
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  key: value
"""
    parsed = _parse(source, "deploy/app.yaml")

    resources = [e for e in parsed.entities if e.kind == "k8s_resource"]
    assert [e.name for e in resources] == ["Deployment/web", "Service/web", "ConfigMap/app-config"]
    # Distinct uids despite Deployment and Service both being called `web`.
    assert len({e.qualified_name for e in resources}) == 3
    # Later documents get their real line offsets, not the file's.
    assert _entity_by_name(parsed, "Service/web").line_start > _entity_by_name(parsed, "Deployment/web").line_end


def test_k8s_service_selector_matches_workload_labels() -> None:
    source = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
        tier: front
    spec:
      containers:
        - name: api
          image: nginx
---
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  selector:
    app: web
---
apiVersion: v1
kind: Service
metadata:
  name: other
spec:
  selector:
    app: elsewhere
"""
    parsed = _parse(source, "deploy/app.yaml")

    # Subset match, so `app: web` selects a pod template labelled app+tier.
    assert _targets(parsed, ".Service_web", RelType.USES_TYPE) == {"Deployment/web"}
    # Non-matching selector emits nothing rather than a guess.
    assert _targets(parsed, ".Service_other", RelType.USES_TYPE) == set()
    # A Deployment's own selector targets the pods it owns — never a self-edge.
    assert _targets(parsed, ".Deployment_web", RelType.USES_TYPE) == set()


def test_k8s_as_json() -> None:
    source = (
        '{"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"},\n'
        '"spec": {"containers": [{"name": "c", "image": "nginx:1.25"}]}}'
    )
    parsed = _parse(source, "manifests/pod.json")

    assert _entity_by_name(parsed, "pod.json").kind == "k8s_manifest"
    assert _entity_by_name(parsed, "Pod/p").kind == "k8s_resource"
    assert _targets(parsed, ".Pod_p", RelType.IMPORTS) == {"nginx"}


# ---------------------------------------------------------------------------
# 3. docker-compose — top-level `services`
# ---------------------------------------------------------------------------

COMPOSE = """\
services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile.prod
    image: acme/api:dev
    depends_on:
      db:
        condition: service_healthy
  worker:
    build: ./worker
    depends_on:
      - api
      - db
  db:
    image: postgres:16
"""


def test_compose_services_are_entities() -> None:
    parsed = _parse(COMPOSE, "docker-compose.yml")

    module = _entity_by_name(parsed, "docker-compose.yml")
    assert module.kind == "compose_file"

    api = _entity_by_name(parsed, "api")
    assert api.label == NodeLabel.TYPE_DEF
    assert api.kind == "compose_service"
    assert api.line_start == 2

    assert _targets(parsed, ":docker-compose_yml", RelType.DEFINES) == {
        f"{PROJECT}:docker-compose_yml.api",
        f"{PROJECT}:docker-compose_yml.worker",
        f"{PROJECT}:docker-compose_yml.db",
    }


def test_compose_depends_on_both_forms() -> None:
    parsed = _parse(COMPOSE, "docker-compose.yml")

    # DEPENDS_ON is out-of-band (project-to-project) in GraphClient, so service
    # dependencies ride USES_TYPE, which resolves same-file names first.
    assert _targets(parsed, ":docker-compose_yml.api", RelType.USES_TYPE) == {"db"}
    assert _targets(parsed, ":docker-compose_yml.worker", RelType.USES_TYPE) == {"api", "db"}


def test_compose_build_points_at_a_containerfile() -> None:
    parsed = _parse(COMPOSE, "docker-compose.yml")

    # `dockerfile` is relative to `context`, which is relative to the compose
    # file; the target is the qualified name containerfile.py mints.
    assert "api.Dockerfile_prod" in _targets(parsed, ":docker-compose_yml.api", RelType.IMPORTS)
    assert _targets(parsed, ":docker-compose_yml.worker", RelType.IMPORTS) == {"worker.Dockerfile"}


def test_compose_image_refs_are_external() -> None:
    parsed = _parse(COMPOSE, "docker-compose.yml")

    assert "acme/api" in _targets(parsed, ":docker-compose_yml.api", RelType.IMPORTS)
    assert _targets(parsed, ":docker-compose_yml.db", RelType.IMPORTS) == {"postgres"}


def test_compose_needs_more_than_a_services_key() -> None:
    # `services:` alone, with nothing image/build/version shaped under it, is not
    # enough — plenty of unrelated config uses the word. It is still config, so
    # it lands in the generic parse rather than being dropped.
    parsed = _parse("services:\n  a: 1\n  b: 2\n", "config/registry.yml")

    assert _entity_by_name(parsed, "registry.yml").kind == "config_file"
    assert [e.kind for e in parsed.entities if e.kind.startswith("compose")] == []


# ---------------------------------------------------------------------------
# 4. GitHub Actions — .github/workflows/*.yml with top-level `jobs`
# ---------------------------------------------------------------------------

WORKFLOW = """\
name: CI
on:
  push:
    branches: [main]
jobs:
  build:
    name: Build the thing
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
  publish:
    needs: [build, test]
    uses: ./.github/workflows/release.yml
"""


def test_workflow_jobs_are_callables() -> None:
    parsed = _parse(WORKFLOW, ".github/workflows/ci.yml")

    module = _entity_by_name(parsed, "ci.yml")
    assert module.kind == "github_workflow"

    build = _entity_by_name(parsed, "build")
    assert build.label == NodeLabel.CALLABLE
    assert build.kind == "ci_job"
    assert build.docstring == "Build the thing"


def test_workflow_needs_edges() -> None:
    parsed = _parse(WORKFLOW, ".github/workflows/ci.yml")

    assert _targets(parsed, ".ci_yml.test", RelType.CALLS) == {"build"}
    assert _targets(parsed, ".ci_yml.publish", RelType.CALLS) == {"build", "test"}


def test_workflow_uses_edges() -> None:
    parsed = _parse(WORKFLOW, ".github/workflows/ci.yml")

    assert _targets(parsed, ".ci_yml.build", RelType.IMPORTS) == {"actions/checkout", "astral-sh/setup-uv"}
    # A local `./…` reusable workflow names a path, not an action — skipped
    # rather than turned into a junk ExternalPackage stub.
    assert _targets(parsed, ".ci_yml.publish", RelType.IMPORTS) == set()


# ---------------------------------------------------------------------------
# 5. Ansible — playbooks (a list whose items carry `hosts`) vs task files
# ---------------------------------------------------------------------------

PLAYBOOK = """\
- name: Configure web tier
  hosts: webservers
  roles:
    - common
    - role: nginx
      tags: [web]
  tasks:
    - name: Install nginx
      ansible.builtin.package:
        name: nginx
      notify: restart nginx
    - name: Include extras
      ansible.builtin.include_tasks: extras.yml
    - name: Apply db role
      include_role:
        name: database
    - block:
        - name: Guarded step
          ansible.builtin.command: /bin/true
  handlers:
    - name: restart nginx
      ansible.builtin.service:
        name: nginx
        state: restarted

- import_playbook: extra.yml
"""


def test_ansible_playbook_plays_and_tasks() -> None:
    parsed = _parse(PLAYBOOK, "playbooks/site.yml")

    module = _entity_by_name(parsed, "site.yml")
    assert module.kind == "ansible_playbook"

    play = _entity_by_name(parsed, "Configure web tier")
    assert play.label == NodeLabel.CALLABLE
    assert play.kind == "ansible_play"
    assert play.extra_properties["hosts"] == "webservers"

    task = _entity_by_name(parsed, "Install nginx")
    assert task.kind == "ansible_task"
    handler = _entity_by_name(parsed, "restart nginx")
    assert handler.kind == "ansible_handler"
    # A named task nested in a `block:` is still reachable.
    assert _entity_by_name(parsed, "Guarded step").kind == "ansible_task"


def test_ansible_role_and_include_edges() -> None:
    parsed = _parse(PLAYBOOK, "playbooks/site.yml")

    assert _targets(parsed, ".site_yml.Configure_web_tier", RelType.USES_TYPE) == {"common", "nginx"}
    # Both the FQCN spelling (`ansible.builtin.include_role`) and the short one.
    assert _targets(parsed, ".Apply_db_role", RelType.USES_TYPE) == {"database"}
    assert _targets(parsed, ".Include_extras", RelType.IMPORTS) == {"playbooks.extras_yml"}
    assert _targets(parsed, ":playbooks.site_yml", RelType.IMPORTS) == {"playbooks.extra_yml"}


def test_ansible_notify_targets_handler_by_name() -> None:
    parsed = _parse(PLAYBOOK, "playbooks/site.yml")

    assert _targets(parsed, ".Install_nginx", RelType.CALLS) == {"restart nginx"}


def test_ansible_tasks_file_mints_the_role_node() -> None:
    source = """\
- name: Install packages
  ansible.builtin.package:
    name: nginx
  notify: reload nginx

- name: Run extras
  ansible.builtin.include_tasks: extras.yml
"""
    parsed = _parse(source, "roles/web/tasks/main.yml")

    module = _entity_by_name(parsed, "main.yml")
    assert module.kind == "ansible_tasks"

    # The role is a directory, so its uid keys on the directory — and only
    # tasks/main.yml mints it, or every file in the role would claim the uid.
    role = _entity_by_name(parsed, "web")
    assert role.label == NodeLabel.TYPE_DEF
    assert role.kind == "ansible_role"
    assert role.qualified_name == f"{PROJECT}:roles.web"
    assert _targets(parsed, ":roles.web", RelType.CONTAINS) == {f"{PROJECT}:roles.web.tasks.main_yml"}

    assert _targets(parsed, ".Install_packages", RelType.CALLS) == {"reload nginx"}
    assert _targets(parsed, ".Run_extras", RelType.IMPORTS) == {"roles.web.tasks.extras_yml"}


def test_ansible_handlers_file() -> None:
    source = """\
- name: reload nginx
  ansible.builtin.service:
    name: nginx
    state: reloaded
"""
    parsed = _parse(source, "roles/web/handlers/main.yml")

    assert _entity_by_name(parsed, "main.yml").kind == "ansible_handlers"
    assert _entity_by_name(parsed, "reload nginx").kind == "ansible_handler"
    # Not the role entry point — no role node from a handlers file.
    assert [e.name for e in parsed.entities if e.kind == "ansible_role"] == []


def test_ansible_task_shape_outside_tasks_dir_gets_no_tasks() -> None:
    # Structurally identical to a tasks file. Without the `tasks/`/`handlers/`
    # directory the Ansible reading is a guess, and a miss beats a false
    # positive that pollutes the graph with fake tasks. A two-item list is too
    # short to read as data either, so it ends up a bare generic Module node.
    parsed = _parse("- name: one\n  value: 1\n- name: two\n  value: 2\n", "data/items.yml")

    assert [e.kind for e in parsed.entities] == ["config_file"]


def test_ansible_unnamed_task_gets_no_node() -> None:
    # A positional uid would churn the whole file's graph on every insertion.
    source = """\
- ansible.builtin.ping:
- name: Named one
  ansible.builtin.ping:
"""
    parsed = _parse(source, "roles/web/tasks/main.yml")
    assert [e.name for e in parsed.entities if e.kind == "ansible_task"] == ["Named one"]


def test_ansible_duplicate_task_names_get_distinct_uids() -> None:
    source = """\
- name: Converge
  ansible.builtin.ping:
- name: Converge
  ansible.builtin.ping:
"""
    parsed = _parse(source, "roles/web/tasks/main.yml")

    tasks = [e for e in parsed.entities if e.kind == "ansible_task"]
    assert len(tasks) == 2
    assert len({e.qualified_name for e in tasks}) == 2


# ---------------------------------------------------------------------------
# 6. Unrecognised config falls back to a generic key tree
# ---------------------------------------------------------------------------

GENERIC_YAML = """\
retry_count: 3
timeout: 30
database:
  host: localhost
  pool:
    size: 5
    overflow: 10
"""


def test_generic_yaml_emits_a_key_tree() -> None:
    parsed = _parse(GENERIC_YAML, "config/settings.yml")

    module = _entity_by_name(parsed, "settings.yml")
    assert module.label == NodeLabel.MODULE
    assert module.kind == "config_file"
    # The raw text rides along as `source`: for a file whose keys nothing can
    # interpret, full-text search over the content is most of the value on offer.
    assert "retry_count: 3" in (module.source or "")

    retry = _entity_by_name(parsed, "retry_count")
    assert retry.label == NodeLabel.VALUE
    assert retry.kind == "config_setting"
    assert retry.line_start == 1
    assert retry.source == "retry_count: 3"

    database = _entity_by_name(parsed, "database")
    assert database.label == NodeLabel.TYPE_DEF
    assert database.kind == "config_section"
    assert (database.line_start, database.line_end) == (3, 7)

    assert _targets(parsed, ":config.settings_yml", RelType.DEFINES) == {
        f"{PROJECT}:config.settings_yml.retry_count",
        f"{PROJECT}:config.settings_yml.timeout",
        f"{PROJECT}:config.settings_yml.database",
    }
    # Three levels below the Module node, so `database.pool.size` is reachable.
    assert _targets(parsed, ":config.settings_yml.database.pool", RelType.DEFINES) == {
        f"{PROJECT}:config.settings_yml.database.pool.size",
        f"{PROJECT}:config.settings_yml.database.pool.overflow",
    }


def test_generic_json_emits_a_key_tree() -> None:
    parsed = _parse('{"name": "thing", "scripts": {"build": "tsc"}}', "package.json")

    assert _entity_by_name(parsed, "package.json").kind == "config_file"
    assert _entity_by_name(parsed, "name").kind == "config_setting"
    assert _entity_by_name(parsed, "scripts").kind == "config_section"
    assert _targets(parsed, ":package_json.scripts", RelType.DEFINES) == {f"{PROJECT}:package_json.scripts.build"}


TOML_CONFIG = """\
line-length = 120

[project]
name = "thing"
dependencies = ["a", "b"]

[tool.ruff.lint]
select = ["E", "F"]
"""


def test_toml_produces_entities() -> None:
    """`.toml` was registered and dispatched to nothing — always zero entities."""
    parsed = _parse(TOML_CONFIG, "pyproject.toml")

    assert _entity_by_name(parsed, "pyproject.toml").kind == "config_file"

    top_level = _entity_by_name(parsed, "line-length")
    assert top_level.label == NodeLabel.VALUE
    assert top_level.line_start == 1

    project = _entity_by_name(parsed, "project")
    assert project.label == NodeLabel.TYPE_DEF
    assert project.kind == "config_section"
    # The table node itself runs to the start of the next table, blank lines
    # included; the span reported is where its last pair actually ends.
    assert (project.line_start, project.line_end) == (3, 5)

    # A dotted header is ONE section — the TOML grammar has no nesting there —
    # and its dots fold, so they cannot fake qualified-name levels either.
    lint = _entity_by_name(parsed, "tool.ruff.lint")
    assert lint.qualified_name == f"{PROJECT}:pyproject_toml.tool_ruff_lint"
    assert _targets(parsed, ":pyproject_toml.tool_ruff_lint", RelType.DEFINES) == {
        f"{PROJECT}:pyproject_toml.tool_ruff_lint.select"
    }


def test_kustomization_falls_back_to_generic() -> None:
    # apiVersion + kind but no metadata.name: nothing to build a resource uid
    # from, and kustomize composition is a different graph from the resource
    # graph — but the file is still config.
    parsed = _parse(
        "apiVersion: kustomize.config.k8s.io/v1beta1\nkind: Kustomization\nresources:\n  - deployment.yaml\n",
        "overlays/prod/kustomization.yaml",
    )

    assert _entity_by_name(parsed, "kustomization.yaml").kind == "config_file"
    assert [e for e in parsed.entities if e.kind == "k8s_resource"] == []


# ---------------------------------------------------------------------------
# 6b. …but data files are still rejected outright
# ---------------------------------------------------------------------------


def test_record_stream_extension_is_rejected_by_name() -> None:
    """A record stream is never config, and its content is never looked at.

    Asserted against the handler directly rather than through ``parse_file``:
    these extensions are deliberately unregistered, so the indexer never opens
    such a file in the first place. The handler still has to refuse one, because
    the registry is not the only thing that can route a path here.
    """
    assert get_language_for_file("logs/events.jsonl") is None
    assert get_language_for_file("logs/events.ndjson") is None

    raw = b'{"a": 1}\n{"a": 2}\n'
    json_config = get_language_for_file("logs/events.json")
    assert json_config is not None
    root = Parser(json_config.language).parse(raw).root_node
    assert _parse_config("logs/events.jsonl", raw, root, PROJECT) is None
    assert _parse_config("logs/events.ndjson", raw, root, PROJECT) is None


def test_oversized_config_is_data() -> None:
    _decline(f"note: {'x' * (MAX_GENERIC_CONFIG_BYTES + 1)}\n", "config/big.yml")


def test_multi_document_generic_stream_is_data() -> None:
    # `yaml.safe_load` would refuse this outright; the dialects that legitimately
    # use `---` were all claimed before the fallback ran.
    _decline("a: 1\n---\nb: 2\n", "fixtures/stream.yml")


def test_top_level_array_of_records_is_data() -> None:
    _decline("\n".join(f"- id: {i}\n  value: v{i}" for i in range(12)) + "\n", "fixtures/rows.yml")


def test_long_top_level_array_is_data() -> None:
    _decline("\n".join(f"- item-{i}" for i in range(60)) + "\n", "fixtures/list.yml")


def test_top_level_scalar_is_data() -> None:
    _decline("just a string\n", "config/motd.yml")


def test_nested_record_dump_is_data() -> None:
    # `{"rows": [...]}` has one top-level key, so every array-shaped check misses
    # it. The key census does not: 7 field names repeated 40 times is 281 key
    # occurrences over 8 distinct names.
    rows = [{"id": i, "name": "n", "email": "e", "age": 1, "city": "c", "zip": "z", "tag": "t"} for i in range(40)]
    _decline(json.dumps({"rows": rows}), "fixtures/people.json")


def test_wide_hand_written_config_survives_the_key_census() -> None:
    # Same key volume as the dump above, but config repeats almost nothing.
    parsed = _parse("\n".join(f"setting_{i}: {i}" for i in range(250)) + "\n", "config/many.yml")

    assert _entity_by_name(parsed, "setting_0").kind == "config_setting"
    # …and the per-file node budget still caps what one file can mint.
    assert len(parsed.entities) <= 200


# ---------------------------------------------------------------------------
# 7. XML — the deliberate exception (Salesforce metadata needs it)
# ---------------------------------------------------------------------------


def test_xml_structural_parse() -> None:
    source = """\
<?xml version="1.0" encoding="UTF-8"?>
<Flow xmlns="http://soap.sforce.com/2006/04/metadata">
    <apiVersion>59.0</apiVersion>
    <label>My Flow</label>
    <decisions>
        <name>Check</name>
    </decisions>
    <status>Active</status>
</Flow>
"""
    parsed = _parse(source, "force-app/flows/MyFlow.flow-meta.xml")

    module = _entity_by_name(parsed, "MyFlow.flow-meta.xml")
    assert module.kind == "xml_document"

    root = _entity_by_name(parsed, "Flow")
    assert root.label == NodeLabel.TYPE_DEF
    assert root.kind == "xml_element"
    assert root.line_start == 2

    label = _entity_by_name(parsed, "label")
    assert label.label == NodeLabel.VALUE
    assert label.kind == "xml_setting"
    assert label.source == "My Flow"

    # A child with element children stays a container, not a setting.
    assert _entity_by_name(parsed, "decisions").label == NodeLabel.TYPE_DEF

    assert _targets(parsed, ".MyFlow_flow-meta_xml", RelType.DEFINES) == {
        f"{PROJECT}:force-app.flows.MyFlow_flow-meta_xml.Flow"
    }
    assert len(_rels_from(parsed, ".MyFlow_flow-meta_xml.Flow", RelType.DEFINES)) == 4


def test_xml_without_an_element_declines() -> None:
    _decline('<?xml version="1.0"?>\n', "force-app/broken.xml")


# ---------------------------------------------------------------------------
# 8. Robustness — a config parser must never crash the indexing pipeline
# ---------------------------------------------------------------------------


def test_empty_file_produces_nothing() -> None:
    _decline("", "deploy/app.yaml")
    _decline("   \n\n", "deploy/app.yaml")


def test_malformed_yaml_does_not_raise() -> None:
    _decline("apiVersion: v1\nkind: [unclosed\n", "deploy/app.yaml")


def test_malformed_json_does_not_raise() -> None:
    _decline('{"apiVersion": "v1", ', "manifests/pod.json")


def test_helm_template_is_never_yaml_loaded() -> None:
    # Go-template YAML is routinely invalid before rendering; the gate has to run
    # before the load, not as an exception handler around it.
    _decline(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {{ .Release.Name }}\n",
        "charts/app/templates/deployment.yaml",
    )


def test_vault_encrypted_file_is_opaque() -> None:
    _decline("$ANSIBLE_VAULT;1.1;AES256\n36613864363...\n", "group_vars/all/vault.yml")


def test_non_printable_character_does_not_crash_the_batch() -> None:
    # PyYAML raises ReaderError from `SafeLoader.__init__` (via check_printable),
    # i.e. before any of the calls the loader's try block wraps. Constructed
    # outside that block, one stray control byte in one file escaped parse_file
    # and took down the whole AST batch.
    _decline("apiVersion: v1\nkind: Pod\nmetadata:\n  name: \x07p\n", "deploy/bell.yaml")


_CYCLIC_ANCHOR_SCRIPT = """
from code_atlas.parsing.ast import parse_file

SOURCE = b'''apiVersion: v1
kind: Pod
metadata:
  name: p
spec: &s
  child: *s
'''
result = parse_file("deploy/cycle.yaml", SOURCE, "test_project")
assert result is not None, "parse_file bailed out instead of walking the cycle"
names = [e.name for e in result.entities]
assert "Pod/p" in names, names
print("WALK-TERMINATED")
"""


def test_self_referential_anchor_terminates() -> None:
    """A self-referential anchor is legal YAML and SafeLoader builds a cyclic dict.

    Verified out of process: the failure under test is unbounded recursion over
    the constructed object, and an interpreter dying that way can take the test
    runner with it rather than raising something pytest can report.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _CYCLIC_ANCHOR_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, f"exit {completed.returncode}\n{completed.stderr}"
    assert "WALK-TERMINATED" in completed.stdout


def test_end_line_matches_a_naive_newline_count() -> None:
    """The line index replaced ``text.count("\\n", 0, end)`` on every entity.

    The old expression was correct, just O(entities x filesize) — and a config
    file's entity count grows with its size, so that product is quadratic. This
    pins the bisect against the implementation it replaced.
    """
    source = "\n".join(f"key_{i}:\n  nested_{i}: {i}" for i in range(200)) + "\n"
    docs = _load_yaml_documents(source)
    assert docs is not None
    assert len(docs) == 1
    out = _Out(file_path="config/wide.yml", project_name=PROJECT, text=source)

    for key_node, value_node in docs[0].node.value:
        for node in (key_node, value_node):
            end = node.end_mark.index
            while end > node.start_mark.index and source[end - 1].isspace():
                end -= 1
            assert out.end_line(node) == source.count("\n", 0, end) + 1


def test_content_hash_is_set_on_every_entity() -> None:
    parsed = _parse(K8S_DEPLOYMENT, "deploy/app.yaml")
    assert all(e.content_hash for e in parsed.entities)


def test_formats_sharing_a_stem_get_distinct_uids() -> None:
    """``app.json`` beside ``app.yaml`` is routine, and qualified_name IS the uid.

    Stripping the extension the way the code-language modules do would make both
    files claim one node, and the later upsert would silently overwrite the other.
    """
    as_json = _parse('{"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "p"}}', "deploy/app.json")
    as_yaml = _parse("apiVersion: v1\nkind: Pod\nmetadata:\n  name: p\n", "deploy/app.yaml")

    assert as_json.entities[0].qualified_name != as_yaml.entities[0].qualified_name
    assert {e.qualified_name for e in as_json.entities}.isdisjoint({e.qualified_name for e in as_yaml.entities})


def test_dotted_directory_does_not_collide_with_real_nesting() -> None:
    """``a.b/x`` and ``a/b/x`` used to render one qualified name, i.e. one uid.

    Folding the dot only in the basename leaves the directory's dot standing in
    for the qualified name's own separator, so the two paths met at
    ``charts.app.v2.pod_yaml`` and whichever file was upserted second silently
    overwrote the first. Dots fold in *every* segment.
    """
    document = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: p\n"
    dotted = _parse(document, "charts/app.v2/pod.yaml")
    nested = _parse(document, "charts/app/v2/pod.yaml")

    assert dotted.entities[0].qualified_name == f"{PROJECT}:charts.app_v2.pod_yaml"
    assert nested.entities[0].qualified_name == f"{PROJECT}:charts.app.v2.pod_yaml"
    assert {e.qualified_name for e in dotted.entities}.isdisjoint({e.qualified_name for e in nested.entities})
