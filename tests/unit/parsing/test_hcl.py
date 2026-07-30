"""Tests for HCL / Terraform parser."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("tree_sitter_hcl", reason="tree-sitter-hcl not installed")

from code_atlas.parsing.ast import ParsedEntity, ParsedFile, get_language_for_file, parse_file
from code_atlas.schema import NodeLabel, RelType

PROJECT = "test_project"

# Node property names the graph client writes from ParsedEntity fields before it
# applies ``SET n += e.extra_properties`` (GraphClient._batch_create_entities /
# _batch_update_entities). An extra_properties key matching one of these
# silently overwrites it.
_RESERVED_NODE_PROPERTIES = frozenset(ParsedEntity.__dataclass_fields__) | {"uid", "project_name"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "infra/main.tf") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _run_isolated(body: str) -> subprocess.CompletedProcess[str]:
    """Run *body* in a fresh interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


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


def _refs_from(parsed: ParsedFile, from_qn_suffix: str) -> list[str]:
    """USES_TYPE targets of one entity — the Terraform addresses it references."""
    return [r.to_name for r in _rels_from(parsed, from_qn_suffix, RelType.USES_TYPE)]


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------


def test_language_detection_hcl():
    for path in ("infra/main.tf", "infra/prod.tfvars", "cfg/nomad.hcl"):
        cfg = get_language_for_file(path)
        assert cfg is not None, path
        assert cfg.name == "hcl"


def test_language_detection_not_hcl():
    assert get_language_for_file("data.csv") is None
    assert get_language_for_file("readme.txt") is None


# ---------------------------------------------------------------------------
# 2. File-level Module entity
# ---------------------------------------------------------------------------


def test_file_module_entity_kinds():
    kinds = {}
    for path in ("infra/main.tf", "infra/prod.tfvars", "cfg/nomad.hcl"):
        parsed = _parse('variable "a" {}\n', path=path)
        module = parsed.entities[0]
        assert module.label == NodeLabel.MODULE
        kinds[path] = module.kind
    assert kinds == {
        "infra/main.tf": "terraform_file",
        "infra/prod.tfvars": "terraform_vars",
        "cfg/nomad.hcl": "hcl_file",
    }


def test_file_module_qualified_name_keeps_extension():
    parsed = _parse('variable "a" {}\n', path="infra/main.tf")
    assert parsed.entities[0].qualified_name == f"{PROJECT}:infra.main_tf"


def test_empty_file_emits_nothing():
    parsed = _parse("   \n\n", path="infra/main.tf")
    assert parsed.entities == []
    assert parsed.relationships == []


# ---------------------------------------------------------------------------
# 3. Block declarations -> TypeDef, named by Terraform address
# ---------------------------------------------------------------------------

_DECLARATIONS = """
resource "aws_s3_bucket" "logs" {
  bucket = "logs"
}

data "aws_ami" "ubuntu" {
  most_recent = true
}

variable "region" {
  description = "AWS region"
  type        = string
}

output "bucket" {
  value = aws_s3_bucket.logs.id
}

locals {
  name_prefix = "app"
  suffix      = "v1"
}

module "vpc" {
  source = "./modules/vpc"
}

provider "aws" {
  region = "eu-west-1"
}
"""


def test_declarations_are_typedefs_named_by_address():
    parsed = _parse(_DECLARATIONS)
    found = {e.name: e.kind for e in parsed.entities if e.label == NodeLabel.TYPE_DEF}
    assert found == {
        "aws_s3_bucket.logs": "terraform_resource",
        "data.aws_ami.ubuntu": "terraform_data",
        "var.region": "terraform_variable",
        "output.bucket": "terraform_output",
        "local.name_prefix": "terraform_local",
        "local.suffix": "terraform_local",
        "module.vpc": "terraform_module_call",
        "provider.aws": "terraform_provider",
    }


def test_declaration_uids_are_namespaced_by_directory():
    parsed = _parse(_DECLARATIONS, path="infra/prod/main.tf")
    assert _entity_by_name(parsed, "aws_s3_bucket.logs").qualified_name == f"{PROJECT}:infra.prod.aws_s3_bucket.logs"
    assert _entity_by_name(parsed, "var.region").qualified_name == f"{PROJECT}:infra.prod.var.region"


def test_declaration_uids_survive_a_move_between_files_in_one_module():
    """The directory is the Terraform module — the file within it is not part of the identity."""
    a = _entity_by_name(_parse(_DECLARATIONS, path="infra/main.tf"), "var.region")
    b = _entity_by_name(_parse(_DECLARATIONS, path="infra/variables.tf"), "var.region")
    assert a.qualified_name == b.qualified_name
    assert a.content_hash == b.content_hash


def test_root_level_file_has_no_namespace_prefix():
    parsed = _parse(_DECLARATIONS, path="main.tf")
    assert _entity_by_name(parsed, "var.region").qualified_name == f"{PROJECT}:var.region"


def test_locals_are_one_entity_per_attribute():
    parsed = _parse("locals {\n  a = 1\n  b = 2\n}\n")
    locals_ = [e for e in parsed.entities if e.kind == "terraform_local"]
    assert [e.name for e in locals_] == ["local.a", "local.b"]
    assert locals_[0].line_start == 2
    assert locals_[1].line_start == 3


def test_provider_alias_is_part_of_the_address():
    parsed = _parse('provider "aws" {\n  alias = "west"\n}\n')
    entity = _entity_by_name(parsed, "provider.aws.west")
    assert entity.kind == "terraform_provider"


def test_block_without_its_labels_is_skipped():
    parsed = _parse('resource "aws_s3_bucket" {\n  bucket = "x"\n}\n')
    assert [e.label for e in parsed.entities] == [NodeLabel.MODULE]


# ---------------------------------------------------------------------------
# 4. Entity metadata (docstring / signature / properties / lines)
# ---------------------------------------------------------------------------


def test_description_becomes_docstring():
    parsed = _parse(_DECLARATIONS)
    assert _entity_by_name(parsed, "var.region").docstring == "AWS region"
    assert _entity_by_name(parsed, "aws_s3_bucket.logs").docstring is None


def test_block_header_becomes_signature():
    parsed = _parse(_DECLARATIONS)
    assert _entity_by_name(parsed, "aws_s3_bucket.logs").signature == 'resource "aws_s3_bucket" "logs"'
    assert _entity_by_name(parsed, "data.aws_ami.ubuntu").signature == 'data "aws_ami" "ubuntu"'


def test_resource_type_recorded_as_property():
    parsed = _parse(_DECLARATIONS)
    assert _entity_by_name(parsed, "aws_s3_bucket.logs").extra_properties["resource_type"] == "aws_s3_bucket"
    assert _entity_by_name(parsed, "data.aws_ami.ubuntu").extra_properties["resource_type"] == "aws_ami"


def test_block_source_and_line_span():
    parsed = _parse(_DECLARATIONS)
    entity = _entity_by_name(parsed, "aws_s3_bucket.logs")
    assert entity.line_start == 2
    assert entity.line_end == 4
    assert entity.source is not None
    assert entity.source.startswith('resource "aws_s3_bucket" "logs" {')


# ---------------------------------------------------------------------------
# 5. DEFINES — file Module owns every block it declares
# ---------------------------------------------------------------------------


def test_defines_from_file_module():
    parsed = _parse(_DECLARATIONS)
    defines = _rels_from(parsed, "infra.main_tf", RelType.DEFINES)
    assert {r.to_name for r in defines} == {e.qualified_name for e in parsed.entities if e.label == NodeLabel.TYPE_DEF}


# ---------------------------------------------------------------------------
# 6. Reference extraction -> USES_TYPE on the bare address
# ---------------------------------------------------------------------------


def test_reference_heads_var_local_data_module_resource():
    source = """
resource "aws_instance" "web" {
  ami       = data.aws_ami.ubuntu.id
  subnet    = module.vpc.public_subnet_ids[0]
  size      = var.instance_type
  tags      = local.tags
  bucket    = aws_s3_bucket.logs.id
}
"""
    parsed = _parse(source)
    assert set(_refs_from(parsed, "aws_instance.web")) == {
        "data.aws_ami.ubuntu",
        "module.vpc",
        "var.instance_type",
        "local.tags",
        "aws_s3_bucket.logs",
    }


def test_references_inside_interpolation_and_heredoc():
    source = """
resource "aws_s3_bucket" "logs" {
  bucket = "${local.name_prefix}-logs"
  policy = <<-EOT
    arn: ${aws_iam_role.app.arn}
    env: ${var.env}
  EOT
}
"""
    parsed = _parse(source)
    assert set(_refs_from(parsed, "aws_s3_bucket.logs")) == {
        "local.name_prefix",
        "aws_iam_role.app",
        "var.env",
    }


def test_references_inside_functions_objects_and_operators():
    source = """
resource "aws_instance" "web" {
  tags  = merge(local.tags, { Name = "${var.env}-web" })
  total = var.a + local.b
  count = var.enabled ? 1 : 0
  keys  = { for k, v in var.subnets : k => v }
}
"""
    parsed = _parse(source)
    assert set(_refs_from(parsed, "aws_instance.web")) == {
        "local.tags",
        "var.env",
        "var.a",
        "local.b",
        "var.enabled",
        "var.subnets",
    }


def test_references_inside_nested_blocks_belong_to_the_declaration():
    source = """
resource "aws_security_group" "app" {
  dynamic "ingress" {
    for_each = var.rules
    content {
      cidr_blocks = var.cidrs
    }
  }
}
"""
    parsed = _parse(source)
    assert set(_refs_from(parsed, "aws_security_group.app")) == {"var.rules", "var.cidrs"}


def test_depends_on_addresses_become_references():
    source = """
resource "aws_instance" "web" {
  depends_on = [aws_s3_bucket.logs, module.vpc]
}
"""
    parsed = _parse(source)
    assert set(_refs_from(parsed, "aws_instance.web")) == {"aws_s3_bucket.logs", "module.vpc"}


def test_splat_and_index_tails_are_dropped_from_the_address():
    source = """
output "ids" {
  value = aws_subnet.private[*].id
}

output "alt" {
  value = aws_subnet.public.*.id
}

output "one" {
  value = aws_subnet.other[0].id
}
"""
    parsed = _parse(source)
    assert _refs_from(parsed, "output.ids") == ["aws_subnet.private"]
    assert _refs_from(parsed, "output.alt") == ["aws_subnet.public"]
    assert _refs_from(parsed, "output.one") == ["aws_subnet.other"]


def test_references_are_deduped_per_declaration():
    source = """
output "dup" {
  value = "${aws_s3_bucket.logs.id}-${aws_s3_bucket.logs.arn}"
}
"""
    parsed = _parse(source)
    assert _refs_from(parsed, "output.dup") == ["aws_s3_bucket.logs"]


# ---------------------------------------------------------------------------
# 7. Reference guards — what must NOT become an edge
# ---------------------------------------------------------------------------


def test_runtime_heads_are_not_references():
    source = """
resource "aws_instance" "web" {
  name     = each.value.name
  index    = count.index
  ip       = self.private_ip
  script   = "${path.module}/init.sh"
  ws       = terraform.workspace
  keep     = var.env
}
"""
    parsed = _parse(source)
    assert _refs_from(parsed, "aws_instance.web") == ["var.env"]


def test_type_keywords_and_object_keys_are_not_references():
    source = """
variable "region" {
  type    = string
  default = "eu-west-1"
}

resource "aws_instance" "web" {
  tags = {
    Env  = "prod"
    Name = "web"
  }
}
"""
    parsed = _parse(source)
    assert _refs_from(parsed, "var.region") == []
    assert _refs_from(parsed, "aws_instance.web") == []


def test_self_reference_is_not_emitted():
    source = """
variable "region" {
  validation {
    condition     = length(var.region) > 0
    error_message = "required"
  }
}
"""
    parsed = _parse(source)
    assert _refs_from(parsed, "var.region") == []


def test_provider_argument_resolves_to_a_provider_address():
    source = """
resource "aws_s3_bucket" "logs" {
  provider = aws.west
}

resource "aws_s3_bucket" "backup" {
  provider = google
}
"""
    parsed = _parse(source)
    assert _refs_from(parsed, "aws_s3_bucket.logs") == ["provider.aws.west"]
    assert _refs_from(parsed, "aws_s3_bucket.backup") == ["provider.google"]


# ---------------------------------------------------------------------------
# 8. Module sources
# ---------------------------------------------------------------------------


def test_local_module_source_is_normalized_against_the_calling_directory():
    parsed = _parse('module "vpc" {\n  source = "../modules/vpc"\n}\n', path="infra/prod/main.tf")
    props = _entity_by_name(parsed, "module.vpc").extra_properties
    assert props["module_source"] == "../modules/vpc"
    assert props["module_source_type"] == "local"
    assert props["module_source_path"] == "infra/modules/vpc"
    assert _rels_from(parsed, "module.vpc", RelType.IMPORTS) == []


def test_registry_module_source_becomes_an_import():
    parsed = _parse('module "vpc" {\n  source = "terraform-aws-modules/vpc/aws"\n}\n')
    props = _entity_by_name(parsed, "module.vpc").extra_properties
    assert props["module_source_type"] == "registry"
    assert "module_source_path" not in props
    imports = _rels_from(parsed, "module.vpc", RelType.IMPORTS)
    assert [r.to_name for r in imports] == ["terraform-aws-modules/vpc/aws"]


def test_remote_module_source_records_no_import():
    source = """
module "a" {
  source = "git::https://github.com/acme/tf.git//modules/vpc?ref=v1"
}

module "b" {
  source = "app.terraform.io/acme/vpc/aws"
}
"""
    parsed = _parse(source)
    assert _entity_by_name(parsed, "module.a").extra_properties["module_source_type"] == "remote"
    assert _entity_by_name(parsed, "module.b").extra_properties["module_source_type"] == "remote"
    assert [r for r in parsed.relationships if r.rel_type == RelType.IMPORTS] == []


def test_module_source_properties_do_not_shadow_reserved_node_properties():
    """``SET n += extra_properties`` runs last, so a colliding key overwrites the real one.

    A key named ``source`` here would replace the node's ``source`` property —
    the block's own text, which feeds BM25 and vector search — with a Terraform
    module address, silently. See ``GraphClient._batch_create_entities``.
    """
    source = """
module "vpc" {
  source = "../modules/vpc"
}

resource "aws_s3_bucket" "logs" {
  bucket = "logs"
}
"""
    parsed = _parse(source)
    for entity in parsed.entities:
        assert not (entity.extra_properties.keys() & _RESERVED_NODE_PROPERTIES), entity.name
    module_call = _entity_by_name(parsed, "module.vpc")
    # The reserved property still carries the HCL block, not the Terraform source.
    assert module_call.source is not None
    assert module_call.source.startswith('module "vpc"')


def test_required_providers_become_imports_from_the_file():
    source = """
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    corp = {
      source = "app.terraform.io/acme/corp"
    }
  }
  backend "s3" {
    bucket = "tfstate"
  }
}
"""
    parsed = _parse(source)
    imports = _rels_from(parsed, "infra.main_tf", RelType.IMPORTS)
    assert [r.to_name for r in imports] == ["hashicorp/aws"]
    # The terraform block itself is not addressable, so it gets no entity.
    assert [e.label for e in parsed.entities] == [NodeLabel.MODULE]


# ---------------------------------------------------------------------------
# 9. .tfvars — assignments, not declarations
# ---------------------------------------------------------------------------

_TFVARS = """
region      = "eu-west-1"
tags        = { Env = "prod" }
db_password = "hunter2"
"""


def test_tfvars_assignments_are_values_linked_to_their_variable():
    parsed = _parse(_TFVARS, path="infra/dev.tfvars")
    values = [e for e in parsed.entities if e.label == NodeLabel.VALUE]
    assert [e.name for e in values] == ["region", "tags", "db_password"]
    assert {e.kind for e in values} == {"terraform_var_value"}
    assert _refs_from(parsed, "dev_tfvars.region") == ["var.region"]
    assert _rels_from(parsed, "infra.dev_tfvars", RelType.DEFINES)


def test_tfvars_uids_are_file_scoped():
    """Two tfvars files in one directory legitimately set the same key."""
    dev = _entity_by_name(_parse(_TFVARS, path="infra/dev.tfvars"), "region")
    prod = _entity_by_name(_parse(_TFVARS, path="infra/prod.tfvars"), "region")
    assert dev.qualified_name == f"{PROJECT}:infra.dev_tfvars.region"
    assert prod.qualified_name == f"{PROJECT}:infra.prod_tfvars.region"


def test_tfvars_values_are_not_copied_into_the_graph():
    """.tfvars routinely holds credentials — only the key, file and line are kept."""
    parsed = _parse(_TFVARS, path="infra/dev.tfvars")
    for entity in parsed.entities:
        assert entity.source is None
        assert entity.signature is None
    assert "hunter2" not in repr(parsed)


def test_tfvars_blocks_are_ignored_rather_than_erroring():
    parsed = _parse('region = "eu-west-1"\nvariable "x" {}\n', path="infra/dev.tfvars")
    assert [e.name for e in parsed.entities] == ["dev.tfvars", "region"]


# ---------------------------------------------------------------------------
# 10. Plain .hcl and edge cases
# ---------------------------------------------------------------------------


def test_plain_hcl_gets_only_the_file_entity():
    """Nomad/Packer/Terragrunt share HCL syntax but not Terraform block semantics."""
    source = """
job "batch" {
  group "web" {
    count = 2
  }
}
"""
    parsed = _parse(source, path="cfg/nomad.hcl")
    assert [(e.label, e.kind) for e in parsed.entities] == [(NodeLabel.MODULE, "hcl_file")]
    assert parsed.relationships == []


def test_syntax_errors_do_not_crash_the_parser():
    parsed = _parse("not valid @@@ hcl {{{ \n = = =\n")
    assert [e.label for e in parsed.entities] == [NodeLabel.MODULE]


def test_unterminated_block_still_yields_its_declaration():
    parsed = _parse('variable "region" {\n  type = string\n')
    assert _entity_by_name(parsed, "var.region").kind == "terraform_variable"


def test_content_hash_is_set_and_position_independent():
    a = _entity_by_name(_parse(_DECLARATIONS), "aws_s3_bucket.logs")
    b = _entity_by_name(_parse("\n\n" + _DECLARATIONS), "aws_s3_bucket.logs")
    assert a.content_hash
    assert a.content_hash == b.content_hash


# ---------------------------------------------------------------------------
# 11. Uid collisions and hostile nesting
# ---------------------------------------------------------------------------


def test_a_dotted_directory_does_not_collide_with_a_nested_one():
    """``.`` is the qualified-name separator, so it must be folded in every segment."""
    dotted = _parse('variable "x" {}\n', path="a.b/main.tf")
    nested = _parse('variable "x" {}\n', path="a/b/main.tf")
    assert dotted.entities[0].qualified_name == f"{PROJECT}:a_b.main_tf"
    assert nested.entities[0].qualified_name == f"{PROJECT}:a.b.main_tf"
    assert dotted.entities[0].qualified_name != nested.entities[0].qualified_name
    # The block namespace inside the file has to agree with its module uid.
    assert _entity_by_name(dotted, "var.x").qualified_name == f"{PROJECT}:a_b.var.x"
    assert _entity_by_name(nested, "var.x").qualified_name == f"{PROJECT}:a.b.var.x"


def test_deeply_nested_expression_does_not_exhaust_the_stack():
    """A 5000-deep expression must extract normally, not cost the whole file.

    Run out-of-process: if the reference walk regressed to recursion the failure
    is a stack overflow, which is not always a catchable ``RecursionError`` —
    in-process it could take the test session down with it.
    """
    proc = _run_isolated(
        """
        from code_atlas.parsing.ast import parse_file

        depth = 5000
        src = "locals {\\n  x = " + "(" * depth + "var.a" + ")" * depth + "\\n}\\n"
        result = parse_file("infra/deep.tf", src.encode(), "p")
        assert result is not None, "parse_file refused the file outright"
        refs = [r.to_name for r in result.relationships if r.rel_type == "USES_TYPE"]
        assert refs == ["var.a"], refs
        print("OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
