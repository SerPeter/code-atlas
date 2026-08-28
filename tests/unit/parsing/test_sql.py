"""Tests for SQL parser."""

from __future__ import annotations

import pytest

pytest.importorskip("tree_sitter_sql", reason="tree-sitter-sql not installed")

from code_atlas.parsing.ast import (
    DEFAULT_MAX_PARSE_BYTES,
    ParsedFile,
    get_language_for_file,
    parse_file,
)
from code_atlas.schema import NodeLabel, RelType

PROJECT = "test_project"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(source: str, path: str = "db/schema.sql") -> ParsedFile:
    result = parse_file(path, source.encode("utf-8"), PROJECT)
    assert result is not None
    return result


def _entity_by_name(parsed: ParsedFile, name: str):
    matches = [e for e in parsed.entities if e.name == name]
    assert len(matches) == 1, (
        f"Expected 1 entity named {name!r}, got {len(matches)}: {[e.name for e in parsed.entities]}"
    )
    return matches[0]


def _entity_by_qn(parsed: ParsedFile, qualified_name: str):
    matches = [e for e in parsed.entities if e.qualified_name == qualified_name]
    assert len(matches) == 1, (
        f"Expected 1 entity with qn {qualified_name!r}, got {len(matches)}: "
        f"{[e.qualified_name for e in parsed.entities]}"
    )
    return matches[0]


def _kinds(parsed: ParsedFile, kind: str) -> list[str]:
    return [e.name for e in parsed.entities if e.kind == kind]


def _rels_from(parsed: ParsedFile, from_qn_suffix: str, rel_type: RelType):
    return [
        r for r in parsed.relationships if r.from_qualified_name.endswith(from_qn_suffix) and r.rel_type == rel_type
    ]


def _uses_targets(parsed: ParsedFile, from_qn_suffix: str) -> set[str]:
    return {r.to_name for r in _rels_from(parsed, from_qn_suffix, RelType.USES_TYPE)}


# ---------------------------------------------------------------------------
# 1. Language detection
# ---------------------------------------------------------------------------


def test_language_detection_sql():
    cfg = get_language_for_file("db/schema.sql")
    assert cfg is not None
    assert cfg.name == "sql"


def test_language_detection_not_sql():
    assert get_language_for_file("data.csv") is None
    assert get_language_for_file("readme.txt") is None


# ---------------------------------------------------------------------------
# 2. Module entity — the per-file node the hash gate depends on
# ---------------------------------------------------------------------------


def test_module_entity_keeps_the_extension_in_its_uid():
    parsed = _parse("CREATE TABLE t (id INT);")
    module = _entity_by_name(parsed, "schema.sql")
    assert module.label is NodeLabel.MODULE
    assert module.kind == "sql_file"
    # `.sql` folded in rather than stripped: a schema.py in the same directory
    # must not claim the same uid.
    assert module.qualified_name == f"{PROJECT}:db.schema_sql"
    assert module.file_path == "db/schema.sql"


def test_backslash_paths_are_normalised():
    parsed = _parse("CREATE TABLE t (id INT);", path="db\\migrations\\001_init.sql")
    assert parsed.file_path == "db/migrations/001_init.sql"
    assert all(e.file_path == "db/migrations/001_init.sql" for e in parsed.entities)


# ---------------------------------------------------------------------------
# 3. CREATE TABLE -> TypeDef with its columns as members
# ---------------------------------------------------------------------------

_USERS = """
CREATE TABLE IF NOT EXISTS public.users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    tenant_id INTEGER NOT NULL REFERENCES tenants (id) ON DELETE CASCADE
);
"""


def test_create_table_emits_typedef():
    parsed = _parse(_USERS)
    table = _entity_by_name(parsed, "users")
    assert table.label is NodeLabel.TYPE_DEF
    assert table.kind == "sql_table"
    # Schema-namespaced, NOT file-scoped — an ALTER TABLE in another migration
    # has to be able to derive this exact uid.
    assert table.qualified_name == f"{PROJECT}:sql.table.public.users"
    assert table.line_start == 2
    assert table.source is not None
    assert "tenant_id" in table.source


def test_columns_are_values_defined_by_their_table():
    parsed = _parse(_USERS)
    table_qn = f"{PROJECT}:sql.table.public.users"
    assert _kinds(parsed, "sql_column") == ["id", "email", "tenant_id"]

    email = _entity_by_qn(parsed, f"{table_qn}.email")
    assert email.label is NodeLabel.VALUE
    assert email.kind == "sql_column"
    assert email.signature == "email VARCHAR(255) NOT NULL UNIQUE"

    defines = {r.to_name for r in _rels_from(parsed, ":sql.table.public.users", RelType.DEFINES)}
    assert defines == {f"{table_qn}.id", f"{table_qn}.email", f"{table_qn}.tenant_id"}


def test_module_defines_every_top_level_object():
    parsed = _parse(_USERS + "\nCREATE VIEW v AS SELECT 1;\n")
    defines = {r.to_name for r in _rels_from(parsed, ":db.schema_sql", RelType.DEFINES)}
    assert defines == {f"{PROJECT}:sql.table.public.users", f"{PROJECT}:sql.view.v"}


def test_primary_key_is_tagged_inline_and_table_level():
    parsed = _parse("""
CREATE TABLE a (id INT PRIMARY KEY, other INT);
CREATE TABLE b (x INT, y INT, PRIMARY KEY (x));
""")
    assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.a.id").tags == ["primary_key"]
    assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.a.other").tags == []
    assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.b.x").tags == ["primary_key"]
    assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.b.y").tags == []


def test_identifiers_are_case_folded_so_references_resolve():
    # Unquoted SQL identifiers are case-insensitive; USES_TYPE resolves on the
    # node's `name` verbatim, so both sides must be folded or the edge vanishes.
    parsed = _parse("CREATE TABLE Users (Id INT);\nCREATE VIEW v AS SELECT * FROM USERS;")
    assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.users").name == "users"
    assert _uses_targets(parsed, ":sql.view.v") == {"users"}


def test_create_table_as_select_has_no_columns():
    parsed = _parse("CREATE TABLE snapshot AS SELECT * FROM users;")
    assert _kinds(parsed, "sql_table") == ["snapshot"]
    assert _kinds(parsed, "sql_column") == []


# ---------------------------------------------------------------------------
# 4. FOREIGN KEY — the highest-value edge in this format
# ---------------------------------------------------------------------------


def test_inline_references_emits_column_and_table_edges():
    parsed = _parse(_USERS)
    table_qn = f"{PROJECT}:sql.table.public.users"
    # Recorded twice on purpose: the column-level edge says WHICH column, the
    # table-level one answers "what references tenants" without a join.
    assert _uses_targets(parsed, ":sql.table.public.users.tenant_id") == {"tenants"}
    assert _uses_targets(parsed, table_qn) == {"tenants"}
    # ...and on the node, so the FK survives even when `tenants` is not indexed.
    assert _entity_by_qn(parsed, f"{table_qn}.tenant_id").extra_properties == {"sql_references": "tenants(id)"}


def test_named_table_level_foreign_key():
    # `CONSTRAINT <name> FOREIGN` is itself an ERROR node in this grammar — the
    # surviving `constraint` node starts at `KEY`, so detection cannot key on
    # the FOREIGN keyword.
    parsed = _parse("""
CREATE TABLE orders (
    id INT PRIMARY KEY,
    user_id INT,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users (id)
);
""")
    assert _uses_targets(parsed, ":sql.table.orders") == {"users"}
    assert _uses_targets(parsed, ":sql.table.orders.user_id") == {"users"}


def test_unnamed_table_level_foreign_key():
    parsed = _parse("""
CREATE TABLE orders (
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES public.users (id)
);
""")
    # Schema qualifier dropped from the edge target: the referenced node's
    # `name` is the bare table name.
    assert _uses_targets(parsed, ":sql.table.orders.user_id") == {"users"}


def test_primary_key_constraint_is_not_a_foreign_key():
    parsed = _parse("CREATE TABLE t (a INT, b INT, PRIMARY KEY (a, b), UNIQUE (b));")
    assert [r.rel_type for r in parsed.relationships] == [RelType.DEFINES] * 3
    assert _uses_targets(parsed, ":sql.table.t") == set()


def test_alter_table_add_constraint_targets_the_created_table_node():
    # The dominant FK form in migration-per-file repos. No entity is emitted —
    # the edge has to land on the node the CREATE migration made.
    parsed = _parse(
        "ALTER TABLE orders ADD CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users (id);",
        path="db/migrations/003_add_fk.sql",
    )
    assert _kinds(parsed, "sql_table") == []
    assert _uses_targets(parsed, f"{PROJECT}:sql.table.orders") == {"users"}
    assert _uses_targets(parsed, f"{PROJECT}:sql.table.orders.user_id") == {"users"}


def test_alter_table_add_column_is_ignored():
    # Folding migrations into a current schema is explicitly out of scope.
    parsed = _parse("ALTER TABLE orders ADD COLUMN note TEXT;", path="db/migrations/004.sql")
    assert _kinds(parsed, "sql_column") == []
    assert parsed.relationships == []


def test_self_referencing_foreign_key():
    parsed = _parse("CREATE TABLE employees (id INT PRIMARY KEY, manager_id INT REFERENCES employees (id));")
    assert _uses_targets(parsed, ":sql.table.employees") == {"employees"}


# ---------------------------------------------------------------------------
# 5. CREATE VIEW -> entity + a dependency edge per table it selects from
# ---------------------------------------------------------------------------


def test_view_depends_on_every_relation_it_reads():
    parsed = _parse("""
CREATE OR REPLACE VIEW active_users AS
SELECT u.id, o.total
FROM users u
JOIN orders o ON o.user_id = u.id
LEFT JOIN public.payments p ON p.order_id = o.id
WHERE u.active = TRUE;
""")
    view = _entity_by_name(parsed, "active_users")
    assert view.label is NodeLabel.TYPE_DEF
    assert view.kind == "sql_view"
    assert view.qualified_name == f"{PROJECT}:sql.view.active_users"
    assert _uses_targets(parsed, ":sql.view.active_users") == {"users", "orders", "payments"}


def test_materialized_view_is_tagged_and_excludes_cte_names():
    parsed = _parse("""
CREATE MATERIALIZED VIEW rollup AS
WITH recent AS (SELECT * FROM events)
SELECT count(*) FROM recent JOIN dims ON dims.id = recent.dim_id;
""")
    assert _entity_by_name(parsed, "rollup").tags == ["materialized"]
    # `recent` is a CTE, not a table — an edge to it would dangle forever.
    assert _uses_targets(parsed, ":sql.view.rollup") == {"events", "dims"}


def test_view_reads_tables_inside_subqueries():
    parsed = _parse("CREATE VIEW v AS SELECT * FROM (SELECT id FROM inner_tbl) s, other_tbl;")
    assert _uses_targets(parsed, ":sql.view.v") == {"inner_tbl", "other_tbl"}


# ---------------------------------------------------------------------------
# 6. CREATE INDEX -> edge to its table
# ---------------------------------------------------------------------------


def test_named_index_edges_to_its_table():
    parsed = _parse("CREATE UNIQUE INDEX idx_users_email ON users (email, tenant_id);")
    index = _entity_by_name(parsed, "idx_users_email")
    assert index.label is NodeLabel.VALUE
    assert index.kind == "sql_index"
    # Table in the uid: index names are only per-table unique in MySQL.
    assert index.qualified_name == f"{PROJECT}:sql.index.users.idx_users_email"
    assert index.signature == "CREATE UNIQUE INDEX idx_users_email ON users (email, tenant_id)"
    assert _uses_targets(parsed, ":sql.index.users.idx_users_email") == {"users"}


def test_unnamed_index_gets_a_deterministic_name():
    parsed = _parse("CREATE INDEX ON public.orders USING btree (created_at);")
    index = _entity_by_name(parsed, "orders_created_at_idx")
    assert index.qualified_name == f"{PROJECT}:sql.index.public.orders.orders_created_at_idx"
    assert _uses_targets(parsed, index.qualified_name) == {"orders"}


# ---------------------------------------------------------------------------
# 7. CREATE FUNCTION / PROCEDURE -> Callable
# ---------------------------------------------------------------------------


def test_create_function_with_dollar_quoted_body():
    parsed = _parse("""
CREATE OR REPLACE FUNCTION add_nums(a INT, b INT) RETURNS INT AS $$
BEGIN
  RETURN a + b;
END;
$$ LANGUAGE plpgsql;
""")
    fn = _entity_by_name(parsed, "add_nums")
    assert fn.label is NodeLabel.CALLABLE
    assert fn.kind == "sql_function"
    assert fn.qualified_name == f"{PROJECT}:sql.function.add_nums"
    assert fn.signature == "add_nums(a INT, b INT)"
    assert fn.tags == []
    assert _rels_from(parsed, ":db.schema_sql", RelType.DEFINES)[0].to_name == fn.qualified_name


def test_create_procedure_is_recovered_from_the_error_region():
    # The grammar has no `create_procedure` rule at all: the whole statement
    # becomes an ERROR node and even the name token is dropped from the tree.
    parsed = _parse("CREATE PROCEDURE refresh_all() LANGUAGE SQL AS $$ SELECT 1; $$;")
    proc = _entity_by_name(parsed, "refresh_all")
    assert proc.label is NodeLabel.CALLABLE
    assert proc.kind == "sql_procedure"
    assert proc.qualified_name == f"{PROJECT}:sql.procedure.refresh_all"
    # Tagged so the degraded extraction is visible in the graph, not silent.
    assert proc.tags == ["recovered"]


def test_recovered_procedure_is_not_duplicated_per_nested_error():
    parsed = _parse("""
CREATE PROCEDURE dbo.GetUsers @Id INT
AS
BEGIN
    SELECT * FROM [dbo].[Users] WHERE Id = @Id;
END
GO
""")
    procs = [e for e in parsed.entities if e.kind == "sql_procedure"]
    assert [e.qualified_name for e in procs] == [f"{PROJECT}:sql.procedure.dbo.getusers"]
    assert procs[0].name == "getusers"
    assert procs[0].line_start == 2


# ---------------------------------------------------------------------------
# 8. Robustness — dialects this grammar cannot parse must degrade, never raise
# ---------------------------------------------------------------------------


def test_tsql_bracket_quoting_degrades_gracefully():
    # Measured: this grammar fails on T-SQL bracket quoting and produces a tree
    # full of ERROR nodes. Extraction must keep whatever survived; some
    # entities or none is fine, an exception is not.
    parsed = _parse("""
SELECT TOP 10 [a] FROM [dbo].[t] WITH (NOLOCK);
CREATE TABLE [dbo].[Users] ([Id] INT NOT NULL, [Email] NVARCHAR(255));
""")
    # The file still yields its Module node, so the hash gate still works.
    assert _entity_by_name(parsed, "schema.sql").label is NodeLabel.MODULE
    # ...and the DDL that partially survived is still extracted, with the
    # bracket quoting stripped off the identifiers.
    table = _entity_by_qn(parsed, f"{PROJECT}:sql.table.dbo.users")
    assert table.name == "users"
    assert _kinds(parsed, "sql_column") == ["id", "email"]


def test_tsql_query_only_file_raises_nothing():
    parsed = _parse("SELECT TOP 10 [a] FROM [dbo].[t] WITH (NOLOCK);")
    assert _kinds(parsed, "sql_table") == []
    assert parsed.relationships == []


def test_plsql_procedure_raises_nothing():
    parsed = _parse("CREATE OR REPLACE PROCEDURE p IS BEGIN NULL; END;")
    assert _entity_by_name(parsed, "p").kind == "sql_procedure"


def test_mysql_backtick_quoting():
    parsed = _parse("""
CREATE TABLE `posts` (
  `id` int NOT NULL AUTO_INCREMENT,
  `author_id` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `posts_author` FOREIGN KEY (`author_id`) REFERENCES `authors` (`id`)
) ENGINE=InnoDB;
""")
    assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.posts").name == "posts"
    assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.posts.id").tags == ["primary_key"]
    assert _uses_targets(parsed, ":sql.table.posts.author_id") == {"authors"}


def test_ansi_quoted_identifiers():
    parsed = _parse('CREATE TABLE "public"."Thing" ("Id" INT REFERENCES "other" ("x"));')
    thing = _entity_by_name(parsed, "thing")
    assert thing.qualified_name == f"{PROJECT}:sql.table.public.thing"
    # A *quoted* column name parses as `literal` rather than `identifier`.
    assert _entity_by_qn(parsed, f"{thing.qualified_name}.id").extra_properties == {"sql_references": "other(x)"}


def test_pure_garbage_raises_nothing():
    parsed = _parse("}}}} not sql at all ;;; CREATE CREATE ((( 42")
    assert _entity_by_name(parsed, "schema.sql").label is NodeLabel.MODULE


# ---------------------------------------------------------------------------
# 9. Migrations — parsed independently, never ordered
# ---------------------------------------------------------------------------


def test_migration_files_are_plain_files_with_stable_object_uids():
    # Two numbered migrations, parsed in isolation: the CREATE in one and the
    # ALTER in the other must agree on the table's uid, because that is the
    # only thing that makes the cross-file FK edge land.
    init = _parse("CREATE TABLE orders (id INT, user_id INT);", path="db/migrations/001_init.sql")
    add_fk = _parse(
        "ALTER TABLE orders ADD CONSTRAINT fk FOREIGN KEY (user_id) REFERENCES users (id);",
        path="db/migrations/002_fk.sql",
    )
    table_qn = f"{PROJECT}:sql.table.orders"
    assert _entity_by_qn(init, table_qn).file_path == "db/migrations/001_init.sql"
    assert {r.from_qualified_name for r in add_fk.relationships} == {table_qn, f"{table_qn}.user_id"}
    # No ordering/precedence relationship is invented between the two files.
    assert [r for r in init.relationships if r.rel_type is not RelType.DEFINES] == []


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------


def test_empty_file_yields_nothing():
    parsed = _parse("")
    assert parsed.entities == []
    assert parsed.relationships == []


def test_whitespace_only_file_yields_nothing():
    assert _parse("\n\n   \t\n").entities == []


def test_comment_only_file_yields_just_the_module():
    parsed = _parse("-- nothing but a note\n")
    assert [e.kind for e in parsed.entities] == ["sql_file"]


def test_duplicate_object_names_in_one_file_do_not_collide():
    parsed = _parse("DROP TABLE IF EXISTS t; CREATE TABLE t (id INT); CREATE TABLE t (id INT);")
    tables = [e.qualified_name for e in parsed.entities if e.kind == "sql_table"]
    assert tables == [f"{PROJECT}:sql.table.t", f"{PROJECT}:sql.table.t#2"]


def test_content_hash_is_set_and_position_independent():
    first = _parse("CREATE TABLE t (id INT);")
    shifted = _parse("\n\n-- moved down\nCREATE TABLE t (id INT);")
    table_a = _entity_by_name(first, "t")
    table_b = _entity_by_name(shifted, "t")
    assert table_a.content_hash
    assert table_a.content_hash == table_b.content_hash
    assert table_a.line_start != table_b.line_start


def test_block_comment_markers_attach_as_rationale():
    # `/* */` is `marginalia`, not `comment`, in this grammar — both node types
    # are registered or SQL would opt out of rationale extraction.
    parsed = _parse("/* WHY: denormalised on purpose, see ADR-0007 */\nCREATE TABLE wide (id INT);")
    table = _entity_by_name(parsed, "wide")
    assert table.rationale == "WHY: denormalised on purpose, see ADR-0007"
    assert table.citations == ["ADR-0007"]


# ---------------------------------------------------------------------------
# 11. Uid collisions and oversized dumps
# ---------------------------------------------------------------------------


def test_a_dotted_directory_does_not_collide_with_a_nested_one():
    """``.`` is the qualified-name separator, so it must be folded in every segment."""
    dotted = _parse("CREATE TABLE t (id INT);", path="a.b/x.sql")
    nested = _parse("CREATE TABLE t (id INT);", path="a/b/x.sql")
    assert _entity_by_name(dotted, "x.sql").qualified_name == f"{PROJECT}:a_b.x_sql"
    assert _entity_by_name(nested, "x.sql").qualified_name == f"{PROJECT}:a.b.x_sql"


def test_an_oversized_dump_is_refused_before_it_reaches_the_grammar():
    """tree-sitter-sql error recovery is superlinear on the all-ERROR trees T-SQL produces.

    The only place that can stop it is the framework's *pre-parse* ceiling —
    ``_parse_sql`` is handed an already-built tree, so a check here would run
    after the expensive recovery. This asserts the coupling from the SQL side so
    that lowering ``DEFAULT_MAX_PARSE_BYTES``' reach (or exempting ``.sql``)
    cannot happen unnoticed. Cheap: the guard is a length check, and the bytes
    are never handed to the parser.
    """
    unit = "CREATE PROCEDURE [dbo].[P] AS BEGIN SELECT [a] FROM [dbo].[T]; END\nGO\n"
    dump = (unit * (DEFAULT_MAX_PARSE_BYTES // len(unit) + 1)).encode()
    assert len(dump) > DEFAULT_MAX_PARSE_BYTES
    assert parse_file("db/dump.sql", dump, PROJECT) is None
    # Just under the ceiling the same content still parses — the guard is about
    # size, not about refusing T-SQL.
    assert parse_file("db/small.sql", (unit * 4).encode(), PROJECT) is not None


# ---------------------------------------------------------------------------
# dbt mode (ATL-132)
# ---------------------------------------------------------------------------


class TestDbtModels:
    """A dbt model file is a bare SELECT wrapped in Jinja: the DDL walker finds no
    entities in it, and the ref() calls carrying the whole dependency structure live
    inside exactly the spans the grammar chokes on."""

    def test_a_model_file_becomes_a_model_entity(self):
        parsed = _parse(
            "select * from {{ ref('stg_orders') }}",
            path="models/marts/orders.sql",
        )
        model = _entity_by_qn(parsed, f"{PROJECT}:dbt.model.orders")
        assert model.label is NodeLabel.TYPE_DEF
        assert model.kind == "dbt_model"

    def test_ref_becomes_a_resolvable_edge(self):
        parsed = _parse("select * from {{ ref('stg_orders') }}", path="models/orders.sql")
        rels = [r for r in parsed.relationships if r.rel_type is RelType.USES_TYPE]
        assert [r.to_name for r in rels] == ["stg_orders"]
        assert rels[0].from_qualified_name == f"{PROJECT}:dbt.model.orders"

    def test_two_arg_ref_uses_the_model_name_not_the_package(self):
        """`ref('pkg', 'model')` names the package first. Taking group 1 would emit an
        edge to the package and the model DAG would be quietly wrong."""
        parsed = _parse("select * from {{ ref('jaffle', 'stg_customers') }}", path="models/orders.sql")
        names = [r.to_name for r in parsed.relationships if r.rel_type is RelType.USES_TYPE]
        assert names == ["stg_customers"]

    def test_source_edges_use_the_dotted_name(self):
        """resolve_type_refs matches TypeDefs by `name`, and the source node declared in
        schema.yml is named `shop.raw_orders` — a bare `raw_orders` never resolves."""
        parsed = _parse("select * from {{ source('shop', 'raw_orders') }}", path="models/stg.sql")
        names = [r.to_name for r in parsed.relationships if r.rel_type is RelType.USES_TYPE]
        assert names == ["shop.raw_orders"]

    def test_a_macro_call_becomes_a_calls_edge(self):
        parsed = _parse(
            "select {{ cents_to_dollars('amount') }} as usd from {{ ref('stg') }}",
            path="models/orders.sql",
        )
        calls = [r.to_name for r in parsed.relationships if r.rel_type is RelType.CALLS]
        assert calls == ["cents_to_dollars"]

    def test_dbt_builtins_are_not_treated_as_macros(self):
        """ref/source/config/var are dbt's own, not macros anybody defined. Emitting
        CALLS edges to them would attach a guessed edge to every model in the project."""
        parsed = _parse(
            "{{ config(materialized='view') }} select {{ var('x') }} from {{ ref('a') }}",
            path="models/orders.sql",
        )
        assert [r.to_name for r in parsed.relationships if r.rel_type is RelType.CALLS] == []

    def test_materialized_config_is_recorded_as_a_tag(self):
        parsed = _parse("{{ config(materialized='incremental') }} select 1", path="models/orders.sql")
        model = _entity_by_qn(parsed, f"{PROJECT}:dbt.model.orders")
        assert "materialized:incremental" in model.tags

    def test_a_snapshot_block_names_the_snapshot_not_the_file(self):
        parsed = _parse(
            "{% snapshot orders_snapshot %}\nselect * from {{ ref('orders') }}\n{% endsnapshot %}",
            path="snapshots/whatever.sql",
        )
        snap = _entity_by_qn(parsed, f"{PROJECT}:dbt.model.orders_snapshot")
        assert snap.kind == "dbt_snapshot"

    def test_line_numbers_survive_neutralization(self):
        """The shim is length-preserving and never touches a newline, so a macro on
        line 4 must still report line 4 — that is the whole reason for the technique."""
        source = "\n\n\n{% macro late(x) %}\n  {{ x }}\n{% endmacro %}\n"
        parsed = _parse(source, path="macros/late.sql")
        macro = _entity_by_qn(parsed, f"{PROJECT}:dbt.macro.late")
        assert macro.line_start == 4

    def test_plain_sql_is_untouched_by_dbt_mode(self):
        """No Jinja, no dbt mode — the DDL path must behave exactly as before."""
        parsed = _parse("CREATE TABLE users (id INT PRIMARY KEY);")
        assert _entity_by_qn(parsed, f"{PROJECT}:sql.table.users").kind == "sql_table"
        assert not [e for e in parsed.entities if e.kind.startswith("dbt")]


class TestDbtMacros:
    def test_a_macro_definition_becomes_a_callable_with_its_signature(self):
        parsed = _parse(
            "{% macro cents_to_dollars(column_name, scale=2) %}\n  {{ column_name }} / 100\n{% endmacro %}",
            path="macros/cents.sql",
        )
        macro = _entity_by_qn(parsed, f"{PROJECT}:dbt.macro.cents_to_dollars")
        assert macro.label is NodeLabel.CALLABLE
        assert macro.kind == "dbt_macro"
        assert macro.signature == "cents_to_dollars(column_name, scale=2)"

    def test_a_macro_file_does_not_also_become_a_model(self):
        """Macros are not models. Emitting both would put a phantom `cents` model in
        every project's DAG, named after a file that selects nothing."""
        parsed = _parse("{% macro cents(x) %}{{ x }}{% endmacro %}", path="macros/cents.sql")
        assert not [e for e in parsed.entities if e.kind == "dbt_model"]

    def test_a_macro_does_not_call_itself(self):
        """The macro body references its own parameters inside {{ }}; a naive scan
        turns the definition into a self-call."""
        parsed = _parse(
            "{% macro cents(x) %}{{ cents(x) }}{% endmacro %}",
            path="macros/cents.sql",
        )
        assert [r.to_name for r in parsed.relationships if r.rel_type is RelType.CALLS] == []
