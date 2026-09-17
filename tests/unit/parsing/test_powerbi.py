"""TMDL: a Power BI semantic model parsed as entities.

Every fixture below is written from the shapes in Microsoft's own TMDL documentation.
That matters more than it sounds: the first version of this parser was written from a
mental model of the format and tested against a fixture written from the *same* mental
model, so the fixture agreed with the parser about things they were both wrong about.
Four of the tests here failed against that version.

The traps, all of which fail silently rather than loudly:

* **A multi-line expression is indented DEEPER than the properties that follow it.**
  A naive "everything nested under this object" read swallows `formatString` into the DAX.
* **Property values may be double-quoted**, with `""` escaping inside.
* **A bare property name means true.** `isHidden` and `isHidden: true` are the same thing.
* **Names are quoted only when they have to be**, so the same measure is written two ways
  depending on where it appears; get that wrong and the CALLS edges just miss.
* **Read is case-insensitive** even though the serializer writes camelCase.
"""

from __future__ import annotations

import json

import pytest

from code_atlas.parsing.ast import ParsedEntity, parse_file
from code_atlas.parsing.languages.powerbi import (
    dax_references,
    model_name_for,
    split_dotted_reference,
    unquote,
    warehouse_objects,
)
from code_atlas.schema import IMPORT_ECOSYSTEM, NodeLabel, RelType

PATH = "Contoso.SemanticModel/definition/tables/Sales.tmdl"


def _parse(text: str, path: str = PATH):
    result = parse_file(path, text.encode("utf-8"), "proj")
    assert result is not None, f"{path} was declined"
    return result


def _by_name(result) -> dict[str, ParsedEntity]:
    return {e.name: e for e in result.entities}


def _kinds(result) -> dict[str, str]:
    return {e.name: e.kind for e in result.entities}


def _source(entity: ParsedEntity) -> str:
    """The entity's expression text. A None here is a failure, not a shape to tolerate --
    every caller below is asserting about DAX the parser was supposed to capture."""
    assert entity.source is not None, f"{entity.name} captured no expression at all"
    return entity.source


def _rels(result, rel_type: RelType) -> list[tuple[str, str]]:
    return [(r.from_qualified_name, r.to_name) for r in result.relationships if r.rel_type == rel_type]


TABLE = """\
/// Sales fact table.
table Sales
\tlineageTag: 9a48bea0-e5fb-40fa-9e81-f61288e31a02
\tisHidden

\tcolumn 'Net Price'
\t\tdataType: int64
\t\tisHidden
\t\tisAvailableInMdx: false
\t\tsourceColumn: "Net Price"

\tcolumn Quantity
\t\tdataType: int64

\t/// This is the Measure Description
\t/// One more line
\tmeasure 'Sales Amount' = SUMX('Sales', 'Sales'[Quantity] * 'Sales'[Net Price])
\t\tformatString: $ #,##0

\tmeasure 'Sales Amount YTD' =
\t\t\tvar result = TOTALYTD([Sales Amount], 'Calendar'[Date])
\t\t\treturn result
\t\tformatString: $ #,##0
\t\tdisplayFolder: " My ""Amazing"" Measures"

\tpartition Sales-Partition = m
\t\tmode: import
\t\tsource =
\t\t\tlet
\t\t\t\tSource = Sql.Database(Server, Database)
\t\t\tin
\t\t\t\tSource
"""


class TestEntities:
    def test_the_model_objects_become_entities(self):
        """Pairs, not a name-keyed dict: the file and the table are both called `Sales`
        here, which is the normal case for a `tables/<Name>.tmdl`."""
        assert [(e.kind, e.name) for e in _parse(TABLE).entities] == [
            ("pbi_definition", "Sales"),
            ("pbi_table", "Sales"),
            ("pbi_column", "Net Price"),
            ("pbi_column", "Quantity"),
            ("pbi_measure", "Sales Amount"),
            ("pbi_measure", "Sales Amount YTD"),
        ]

    def test_the_table_is_a_typedef_and_the_measures_are_callables(self):
        """Deliberate label choices. Measures call each other, and every existing tool
        reads CALLS as "executes" -- so a measure chain answers "what breaks if I change
        this" through machinery that already exists."""
        labels = {(e.kind, e.label) for e in _parse(TABLE).entities}
        assert (("pbi_table"), NodeLabel.TYPE_DEF) in labels
        assert (("pbi_measure"), NodeLabel.CALLABLE) in labels
        assert (("pbi_column"), NodeLabel.VALUE) in labels

    def test_a_measure_carries_its_dax(self):
        measure = _by_name(_parse(TABLE))["Sales Amount"]
        assert _source(measure) == "SUMX('Sales', 'Sales'[Quantity] * 'Sales'[Net Price])"

    def test_the_lineage_tag_is_a_property_and_never_the_identifier(self):
        """A lineageTag is stable and useless to search for. The uid is the name the
        report author typed."""
        table = next(e for e in _parse(TABLE).entities if e.kind == "pbi_table")
        assert table.extra_properties["lineageTag"] == "9a48bea0-e5fb-40fa-9e81-f61288e31a02"
        assert "9a48bea0" not in table.qualified_name

    def test_the_uid_carries_the_model_the_table_belongs_to(self):
        table = next(e for e in _parse(TABLE).entities if e.kind == "pbi_table")
        assert table.qualified_name == "proj:pbi.model.Contoso.table.Sales"

    def test_a_partition_is_folded_onto_the_table_rather_than_given_a_node(self):
        result = _parse(TABLE)
        assert "Sales-Partition" not in _by_name(result)
        table = next(e for e in result.entities if e.kind == "pbi_table")
        assert table.extra_properties["mode"] == "import"


class TestTheMultiLineExpressionBoundary:
    """The trap that a self-written fixture cannot catch. TMDL indents a multi-line
    expression one level DEEPER than the object's properties, and the properties come
    after it -- so the expression ends where the indentation steps back out, not at the
    end of the block."""

    def test_the_expression_stops_before_the_properties_that_follow_it(self):
        dax = _source(_by_name(_parse(TABLE))["Sales Amount YTD"])
        assert "formatString" not in dax, f"a property leaked into the DAX: {dax!r}"
        assert dax == "var result = TOTALYTD([Sales Amount], 'Calendar'[Date])\nreturn result"

    def test_those_properties_are_still_read_as_properties(self):
        """Non-vacuity: an expression that stopped too early would also satisfy the test
        above, by dropping the properties entirely."""
        measure = _by_name(_parse(TABLE))["Sales Amount YTD"]
        assert measure.extra_properties["formatString"] == "$ #,##0"

    def test_the_expression_is_dedented(self):
        """TMDL strips outer indentation beyond the parent's level on read, so the stored
        DAX is what a person would paste, not what the file's nesting made of it."""
        assert not _source(_by_name(_parse(TABLE))["Sales Amount YTD"]).startswith((" ", "\t"))

    def test_a_backtick_fenced_expression_keeps_its_body_and_drops_the_fence(self):
        text = (
            "table Sales\n"
            "\tmeasure Fenced = ```\n"
            "\t\t\tvar myVar = TODAY()\n"
            "\t\t\treturn myVar\n"
            "\t\t```\n"
            "\t\tformatString: #,##0\n"
        )
        dax = _source(_by_name(_parse(text))["Fenced"])
        assert "```" not in dax
        assert "formatString" not in dax
        assert "var myVar = TODAY()" in dax


class TestProperties:
    def test_a_double_quoted_value_is_unwrapped(self):
        """Quotes around a property value are optional and stripped on serialization.
        Kept, nothing searching for the bare name matches."""
        assert _by_name(_parse(TABLE))["Net Price"].extra_properties["sourceColumn"] == "Net Price"

    def test_a_doubled_quote_inside_one_is_unescaped(self):
        measure = _by_name(_parse(TABLE))["Sales Amount YTD"]
        assert measure.extra_properties["displayFolder"] == ' My "Amazing" Measures'

    def test_a_bare_property_name_means_true(self):
        assert _by_name(_parse(TABLE))["Net Price"].extra_properties["isHidden"] == "true"

    def test_an_explicit_false_is_not_read_as_true(self):
        """The shorthand is "present implies true", which is easy to over-apply into
        "present at all implies true"."""
        column = _by_name(_parse(TABLE))["Net Price"]
        assert column.extra_properties.get("isAvailableInMdx", "false") == "false"

    def test_a_multi_line_description_is_kept_whole(self):
        assert _by_name(_parse(TABLE))["Sales Amount"].docstring == "This is the Measure Description\nOne more line"

    def test_a_description_does_not_leap_a_blank_line(self):
        """The spec allows no whitespace between a description block and the object type
        token, so a stray comment must not become the description of whatever follows."""
        text = "/// orphaned\n\ntable Sales\n\tcolumn Quantity\n\t\tdataType: int64\n"
        table = next(e for e in _parse(text).entities if e.kind == "pbi_table")
        assert table.docstring == ""


class TestNameQuoting:
    """A name is quoted only when it contains a dot, equals, colon, single quote or
    whitespace -- so the same object is written two ways depending on where it appears.
    Normalising is what makes those one node."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("'Sales Amount'", "Sales Amount"),
            ("Quantity", "Quantity"),
            ("'It''s Fine'", "It's Fine"),
            ("''", ""),
            ("'a.b'", "a.b"),
        ],
    )
    def test_unquote(self, raw: str, expected: str):
        assert unquote(raw) == expected

    def test_a_quoted_definition_and_a_bare_reference_are_the_same_target(self):
        text = "table Sales\n\tmeasure 'Sales Amount' = SUM(Sales[Quantity])\n\tmeasure Derived = [Sales Amount] * 2\n"
        result = _parse(text)
        assert ("proj:pbi.model.Contoso.table.Sales.measure.Derived", "Sales Amount") in _rels(result, RelType.CALLS)

    def test_a_name_needing_quotes_is_still_a_single_uid_segment(self):
        """A colon in a name would split `{project}:{qualified_name}` for every consumer
        downstream; a dot would split the segment boundary. Both fold, and the fold earns
        a suffix because it is lossy -- see TestUidsAreDistinct."""
        text = "table 'A.B: C'\n\tcolumn Quantity\n\t\tdataType: int64\n"
        table = next(e for e in _parse(text).entities if e.kind == "pbi_table")
        assert table.qualified_name.startswith("proj:pbi.model.Contoso.table.A_B_ C~")
        assert table.name == "A.B: C", "the display name keeps what the author typed"

    def test_a_name_needing_no_folding_stays_readable(self):
        """Only `.` and `:` fold, because only those two break something downstream. A
        space is safe everywhere the qualified name is consumed, and folding it would make
        every uid in a Power BI model less legible for nothing."""
        text = "table 'Sales Amount'\n\tcolumn Quantity\n\t\tdataType: int64\n"
        table = next(e for e in _parse(text).entities if e.kind == "pbi_table")
        assert table.qualified_name == "proj:pbi.model.Contoso.table.Sales Amount"


class TestDaxReferences:
    def test_a_bare_bracket_is_a_measure_reference(self):
        measures, columns = dax_references("[Sales Amount] + 1")
        assert measures == {"Sales Amount"}
        assert columns == set()

    def test_a_qualified_column_is_not_read_as_a_measure(self):
        """`'Sales'[Amount]` also matches the bare-reference pattern. Read naively, every
        column reference mints a phantom measure named after the column."""
        measures, columns = dax_references("SUM('Sales'[Amount])")
        assert measures == set()
        assert columns == {("Sales", "Amount")}

    def test_an_unquoted_table_qualifier_is_recognised(self):
        _measures, columns = dax_references("SUM(Sales[Amount])")
        assert columns == {("Sales", "Amount")}

    def test_both_shapes_in_one_expression(self):
        measures, columns = dax_references("CALCULATE([Base], 'Calendar'[Date] > TODAY())")
        assert measures == {"Base"}
        assert columns == {("Calendar", "Date")}

    def test_a_measure_chain_becomes_call_edges(self):
        """Three deep, which is what makes blast_radius answer "what breaks if I change
        this" for a model."""
        text = "table Sales\n\tmeasure A = SUM(Sales[Quantity])\n\tmeasure B = [A] * 2\n\tmeasure C = [B] + [A]\n"
        calls = {(frm.rsplit(".", 1)[-1], to) for frm, to in _rels(_parse(text), RelType.CALLS)}
        assert calls == {("B", "A"), ("C", "B"), ("C", "A")}

    def test_a_column_reference_lands_on_the_table(self):
        """USES_TYPE resolves against TypeDef and a column is a Value, so a
        column-targeted edge would resolve to nothing at all. The column rides in the
        edge's properties instead."""
        text = "table Sales\n\tmeasure A = SUM('Calendar'[Date])\n"
        uses = [r for r in _parse(text).relationships if r.rel_type == RelType.USES_TYPE]
        assert [(r.to_name, r.properties["column"]) for r in uses] == [("Calendar", "Date")]


class TestTheFileRoot:
    """Every language emits exactly one Module for the file, and it is not a formality:
    that node carries `file_hash`, and the per-file gate is what lets an unchanged file
    skip parsing. A language emitting none re-parses every one of its files on every
    indexing pass, forever, with no error anywhere."""

    def test_every_file_has_exactly_one_module(self):
        roots = [e for e in _parse(TABLE).entities if e.label == NodeLabel.MODULE]
        assert len(roots) == 1
        assert roots[0].line_start == 1

    def test_the_root_spans_the_file_whatever_line_the_declaration_is_on(self):
        root = next(e for e in _parse(TABLE).entities if e.label == NodeLabel.MODULE)
        assert root.line_start == 1
        assert root.line_end == len(TABLE.splitlines())

    def test_a_model_file_is_its_own_root(self):
        text = "model Model\n\tculture: en-US\n\nref table Sales\n"
        result = _parse(text, "Contoso.SemanticModel/definition/model.tmdl")
        roots = [e for e in result.entities if e.label == NodeLabel.MODULE]
        assert [(r.kind, r.name) for r in roots] == [("pbi_model", "Model")]
        assert roots[0].extra_properties["culture"] == "en-US"

    def test_tables_hang_off_their_own_file(self):
        """Structural edges stay inside the file that states them. A cross-file
        `DEFINES` cannot be written at upsert time -- uid-routed edges MATCH both
        endpoints and nothing orders the files within a batch."""
        result = _parse(TABLE)
        root = next(e for e in result.entities if e.label == NodeLabel.MODULE)
        table = next(e for e in result.entities if e.kind == "pbi_table")
        assert (root.qualified_name, table.qualified_name) in _rels(result, RelType.DEFINES)


class TestModelName:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("Contoso.SemanticModel/definition/tables/Sales.tmdl", "Contoso"),
            ("a/b/Sales Model.SemanticModel/definition/model.tmdl", "Sales Model"),
            ("some/Model/definition/tables/Sales.tmdl", "Model"),
            ("tables/Sales.tmdl", "model"),
        ],
    )
    def test_derived_from_the_path(self, path: str, expected: str):
        """A table file says `table Sales` and never names its model, and `parse_file` is
        a pure function over one file -- so the model cannot be read out of model.tmdl."""
        assert model_name_for(path) == expected

    def test_two_models_in_one_project_do_not_collide(self):
        a = _parse(TABLE, "A.SemanticModel/definition/tables/Sales.tmdl")
        b = _parse(TABLE, "B.SemanticModel/definition/tables/Sales.tmdl")
        a_table = next(e for e in a.entities if e.kind == "pbi_table")
        b_table = next(e for e in b.entities if e.kind == "pbi_table")
        assert a_table.qualified_name != b_table.qualified_name


class TestItReadsWhatItRecognisesAndIgnoresTheRest:
    """Microsoft versions TMDL. A keyword this parser has never seen must cost a
    property, never a table -- the failure mode of strictness here is an entire semantic
    model vanishing from the graph on a Power BI update."""

    def test_an_unknown_property_does_not_lose_the_table(self):
        text = (
            "table Sales\n"
            "\tsomeFutureProperty: whatever\n"
            "\tcolumn Quantity\n"
            "\t\tdataType: int64\n"
            "\t\tanotherNewThing: 7\n"
        )
        assert set(_kinds(_parse(text))) == {"Sales", "Quantity"}

    def test_an_unknown_child_object_does_not_lose_its_siblings(self):
        text = (
            "table Sales\n"
            "\thierarchy 'Product Hierarchy'\n"
            "\t\tlevel Category\n"
            "\t\t\tcolumn: Category\n"
            "\tmeasure A = SUM(Sales[Quantity])\n"
        )
        assert "A" in _kinds(_parse(text))

    def test_a_file_declaring_nothing_modelled_is_declined(self):
        """A culture, role or perspective file.

        Declining yields an EMPTY ParsedFile, not None: None means "unsupported
        language", and the AST consumer only records a file hash for files that produced
        a ParsedFile -- so None would put the file permanently outside the hash gate and
        force a re-read on every pass.
        """
        text = 'cultureInfo en-US\n\tlinguisticMetadata = {\n\t\t"Version": "1.0.0"\n\t}\n'
        result = parse_file("Contoso.SemanticModel/definition/cultures/en-US.tmdl", text.encode(), "proj")
        assert result is not None
        assert result.entities == []

    def test_keywords_are_matched_case_insensitively(self):
        """The API writes camelCase but reads case-insensitively, so a hand-edited
        `Table` is valid and must not silently drop the table."""
        text = "Table Sales\n\tMeasure A = SUM(Sales[Quantity])\n"
        assert "A" in _kinds(_parse(text))

    def test_an_empty_file_is_declined_rather_than_crashing(self):
        result = parse_file(PATH, b"", "proj")
        assert result is not None
        assert result.entities == []

    def test_a_file_of_only_comments_is_declined(self):
        result = parse_file(PATH, b"// nothing here\n/// not even this\n", "proj")
        assert result is not None
        assert result.entities == []


class TestCalculationGroups:
    """A calculation group is a model's time-intelligence layer -- YTD, MTD, PY applied
    across every measure -- so it is one of the highest-fan-out objects there is.

    This whole class exists because the dispatch branch for it was dead: the keyword was
    lowercased and then compared against a camelCase literal, so `_handle_calculation_group`
    was never called from anywhere. The table still parsed, so a calculation group landed
    in the graph as an *empty table* -- the answer came back confident and short.
    """

    GROUP = (
        "table 'Time Intelligence'\n"
        "\tcalculationGroup\n"
        "\t\tmultipleOrEmptySelectionExpression = SELECTEDMEASURE()\n"
        "\t\tnoSelectionExpression = CALCULATE(SELECTEDMEASURE(), 'Date'[IsCurrent] = TRUE())\n"
        "\n"
        "\t\tcalculationItem YTD = CALCULATE(SELECTEDMEASURE(), DATESYTD('Date'[Date]))\n"
        "\t\t\tordinal: 1\n"
        "\t\t\tformatStringDefinition = SELECTEDVALUE('Formats'[Format])\n"
    )

    def test_a_calculation_item_becomes_an_entity(self):
        assert _kinds(_parse(self.GROUP))["YTD"] == "pbi_calc_item"

    def test_it_carries_its_dax(self):
        assert _source(_by_name(_parse(self.GROUP))["YTD"]) == "CALCULATE(SELECTEDMEASURE(), DATESYTD('Date'[Date]))"

    def test_its_dax_produces_edges(self):
        uses = {(r.to_name, r.properties["column"]) for r in _parse(self.GROUP).relationships if r.properties}
        assert ("Date", "Date") in uses

    def test_the_group_level_selection_expressions_are_read(self):
        """`noSelectionExpression` is the default behaviour applied to every measure the
        group is viewed through, and usually the only place it names what it reads."""
        uses = {(r.to_name, r.properties["column"]) for r in _parse(self.GROUP).relationships if r.properties}
        assert ("Date", "IsCurrent") in uses

    def test_the_group_expressions_are_not_counted_twice(self):
        """They are reachable both from the table and from the group's own handler."""
        edges = [r for r in _parse(self.GROUP).relationships if (r.properties or {}).get("column") == "IsCurrent"]
        assert len(edges) == 1


class TestExpressionValuedChildren:
    """TMDL has two child syntaxes: `key: value` assigns a scalar, `key = expr` assigns an
    EXPRESSION. Only the first was implemented, so every expression-valued child was
    discarded -- one omission behind eight separate missing dependencies.

    Each is DAX that names real tables and columns. A dynamic format string in particular
    reaches into a lookup table nothing else in the model touches, so there is no second
    witness for that edge anywhere.
    """

    MEASURE = (
        "table Sales\n"
        "\tmeasure Amount = SUM(Sales[Total])\n"
        "\t\tformatStringDefinition = SELECTEDVALUE('Formats'[Format])\n"
        "\t\tdetailRowsDefinition = SELECTCOLUMNS('Employee', \"Id\", 'Employee'[EmployeeId])\n"
    )

    def _uses(self, text: str) -> set:
        return {(r.to_name, r.properties["column"]) for r in _parse(text).relationships if r.properties}

    def test_a_format_string_definition_produces_its_edge(self):
        assert ("Formats", "Format") in self._uses(self.MEASURE)

    def test_a_detail_rows_definition_produces_its_edge(self):
        assert ("Employee", "EmployeeId") in self._uses(self.MEASURE)

    def test_they_are_stored_but_not_folded_into_source(self):
        """`source` is the object's OWN expression. A measure whose source silently also
        contained its format string would read wrong to anyone who fetched it."""
        measure = _by_name(_parse(self.MEASURE))["Amount"]
        assert _source(measure) == "SUM(Sales[Total])"
        assert measure.extra_properties["formatStringDefinition"] == "SELECTEDVALUE('Formats'[Format])"

    def test_a_kpi_expression_produces_a_call_edge(self):
        """The canonical pattern is a hidden targets table referenced ONLY from a KPI. Miss
        it and those measures read as dead code while being load-bearing."""
        text = (
            "table Sales\n"
            "\tmeasure 'Total sales' = SUM(Sales[Amount])\n"
            "\t\tkpi\n"
            "\t\t\tstatusGraphic: Traffic Light\n"
            "\t\t\ttargetExpression = [Sales target]\n"
        )
        calls = {r.to_name for r in _parse(text).relationships if r.rel_type == RelType.CALLS}
        assert "Sales target" in calls

    def test_a_table_level_definition_produces_its_edge(self):
        text = "table Sales\n\tdefaultDetailRowsDefinition = SELECTCOLUMNS('Sales', \"N\", 'Sales'[OrderNumber])\n"
        assert ("Sales", "OrderNumber") in self._uses(text)

    def test_a_multi_line_measure_body_is_not_read_as_a_property(self):
        """`var x = ...` inside a DAX body is `key = value` shaped. It must not be mistaken
        for an expression-valued property."""
        text = "table Sales\n\tmeasure A =\n\t\t\tvar x = SUM(Sales[Amount])\n\t\t\treturn x\n"
        assert _by_name(_parse(text))["A"].extra_properties == {}


class TestBareReferencesInRowContext:
    """A bare `[X]` is a measure reference EXCEPT inside an iterator, where it is a column
    of the table being iterated. DAX makes them syntactically identical.

    Reading every `[X]` as a measure is not a missed edge but a wrong one, and measures are
    very often named after the column they aggregate -- so the phantom CALLS would land on
    a real measure and read as structural fact.
    """

    TEXT = "table Sales\n\tcolumn Quantity\n\t\tdataType: int64\n\tmeasure Revenue = SUMX('Sales', [Quantity] * 2)\n"

    def test_a_name_matching_a_column_of_this_table_is_not_a_measure_call(self):
        calls = {r.to_name for r in _parse(self.TEXT).relationships if r.rel_type == RelType.CALLS}
        assert "Quantity" not in calls

    def test_it_becomes_a_table_reference_instead(self):
        """Non-vacuity: dropping the reference entirely would also pass the test above."""
        uses = {(r.to_name, r.properties["column"]) for r in _parse(self.TEXT).relationships if r.properties}
        assert ("Sales", "Quantity") in uses

    def test_a_name_that_is_not_a_column_here_is_still_a_measure_call(self):
        text = "table Sales\n\tcolumn Quantity\n\t\tdataType: int64\n\tmeasure B = [Some Other Measure] * 2\n"
        calls = {r.to_name for r in _parse(text).relationships if r.rel_type == RelType.CALLS}
        assert calls == {"Some Other Measure"}

    def test_a_column_declared_after_the_measure_still_counts(self):
        """Columns are collected in a pass of their own, because TMDL lets children be
        declared in any order and intermingled."""
        text = "table Sales\n\tmeasure Revenue = SUMX('Sales', [Quantity])\n\tcolumn Quantity\n\t\tdataType: int64\n"
        calls = {r.to_name for r in _parse(text).relationships if r.rel_type == RelType.CALLS}
        assert "Quantity" not in calls


class TestCalculatedColumnEdges:
    def test_a_calculated_column_is_not_a_leaf(self):
        """Measures and calc items emitted edges; columns did not, so blast_radius treated
        every calculated column as a dead end."""
        text = "table Sales\n\tcolumn Margin = 'Sales'[Price] - 'Costs'[Amount]\n\t\tdataType: double\n"
        uses = {(r.to_name, r.properties["column"]) for r in _parse(text).relationships if r.properties}
        assert ("Costs", "Amount") in uses

    def test_a_plain_column_emits_nothing(self):
        text = "table Sales\n\tcolumn Quantity\n\t\tdataType: int64\n\t\tsourceColumn: QTY\n"
        assert [r for r in _parse(text).relationships if r.rel_type != RelType.DEFINES] == []


class TestRealWorldLexicalShapes:
    """Measured against a 322-file sample of public .tmdl: 3 carry a UTF-8 BOM, 28 use
    CRLF, and names contain characters that look like delimiters."""

    def test_a_byte_order_mark_does_not_lose_the_file(self):
        """Left in place the BOM prefixes line 1, so the file's only root declaration
        fails to match and the entire model file is dropped without a word."""
        text = "\ufefftable Sales\n\tcolumn Quantity\n\t\tdataType: int64\n"
        assert "Sales" in _kinds(_parse(text))

    def test_crlf_line_endings_parse(self):
        text = "table Sales\r\n\tcolumn Quantity\r\n\t\tdataType: int64\r\n"
        assert _kinds(_parse(text))["Quantity"] == "pbi_column"

    def test_a_percent_in_an_unquoted_name(self):
        text = "table T\n\tcalculationGroup\n\t\tcalculationItem YOY% = SELECTEDMEASURE()\n"
        assert "YOY%" in _kinds(_parse(text))

    def test_a_format_string_value_full_of_delimiters_survives(self):
        text = "table Sales\n\tmeasure A = SUM(Sales[X])\n\t\tformatString: #,##0.00 %;(#,##0.00 %)\n"
        assert _by_name(_parse(text))["A"].extra_properties["formatString"] == "#,##0.00 %;(#,##0.00 %)"


class TestPartitions:
    def test_every_partition_is_considered_not_just_the_first(self):
        """A table partitioned by year has one per year. Reading only partition #1 reports
        a mode the table may not uniformly have."""
        text = "table Sales\n\tpartition P2023 = m\n\t\tmode: import\n\tpartition P2024 = m\n\t\tmode: directQuery\n"
        table = next(e for e in _parse(text).entities if e.kind == "pbi_table")
        assert table.extra_properties["mode"] == "mixed"

    def test_a_uniform_mode_is_reported_as_itself(self):
        text = "table Sales\n\tpartition A = m\n\t\tmode: import\n\tpartition B = m\n\t\tmode: import\n"
        table = next(e for e in _parse(text).entities if e.kind == "pbi_table")
        assert table.extra_properties["mode"] == "import"


class TestUserDefinedFunctions:
    """`functions.tmdl` holds DAX UDFs -- top-level declarations, not nested under a
    table. They are called from measures, calculation items AND role filters, so a UDF is
    often the shared dependency that makes several parts of a model move together; miss it
    and each of them looks independent."""

    TEXT = (
        "/// Applies a rate.\n"
        "/// @param value The base value\n"
        "function 'RLS.Pct' =\n"
        "\t\t(value: DOUBLE) => value * [Rate]\n"
        "\tannotation DAXLIB_PackageId = acme\n"
    )
    PATH = "Contoso.SemanticModel/definition/functions.tmdl"

    def test_a_function_is_a_callable(self):
        result = _parse(self.TEXT, self.PATH)
        fn = next(e for e in result.entities if e.kind == "pbi_udf")
        assert fn.label == NodeLabel.CALLABLE
        assert fn.name == "RLS.Pct"

    def test_it_carries_its_body(self):
        fn = next(e for e in _parse(self.TEXT, self.PATH).entities if e.kind == "pbi_udf")
        assert _source(fn) == "(value: DOUBLE) => value * [Rate]"

    def test_its_body_produces_call_edges(self):
        calls = {r.to_name for r in _parse(self.TEXT, self.PATH).relationships if r.rel_type == RelType.CALLS}
        assert calls == {"Rate"}

    def test_the_jsdoc_style_description_is_the_docstring(self):
        """`/// @param` is the documented convention for documenting a UDF."""
        fn = next(e for e in _parse(self.TEXT, self.PATH).entities if e.kind == "pbi_udf")
        assert fn.docstring == "Applies a rate.\n@param value The base value"

    def test_a_name_with_a_dot_is_one_uid_segment(self):
        """`RLS.Pct` is a normal UDF naming convention, and an unfolded dot would fake a
        nesting level in the qualified name."""
        fn = next(e for e in _parse(self.TEXT, self.PATH).entities if e.kind == "pbi_udf")
        assert fn.qualified_name.startswith("proj:pbi.model.Contoso.function.RLS_Pct~")


class TestSecurityRoles:
    """Row-level security is DAX naming real tables and columns, in a file nothing else in
    the model references. So "what does this role depend on" -- and its mirror, "what
    breaks if I rename this column" -- have no other witness anywhere in the graph.

    Three spellings occur in the wild and the DOCUMENTED one is the rare one: measured over
    public repositories, the default-property form outnumbers an explicit
    `filterExpression` child roughly 28 to 1.
    """

    PATH = "Contoso.SemanticModel/definition/roles/R.tmdl"

    DEFAULT_FORM = "role Store1\n\tmodelPermission: read\n\n\ttablePermission Store = 'Store'[Code] IN {1,2}\n"
    CHILD_EQUALS = "role Region\n\ttablePermission Sales\n\t\tfilterExpression = 'Sales'[Region] = \"EU\"\n"
    CHILD_COLON = "role All\n\ttablePermission Sales\n\t\tfilterExpression: TRUE()\n"

    def _uses(self, text: str) -> set:
        return {(r.to_name, r.properties.get("column")) for r in _parse(text, self.PATH).relationships if r.properties}

    def test_a_role_becomes_an_entity(self):
        role = next(e for e in _parse(self.DEFAULT_FORM, self.PATH).entities if e.kind == "pbi_role")
        assert role.name == "Store1"
        assert role.extra_properties["modelPermission"] == "read"

    @pytest.mark.parametrize("form", ["DEFAULT_FORM", "CHILD_EQUALS", "CHILD_COLON"])
    def test_every_spelling_links_the_role_to_the_table_it_filters(self, form: str):
        """The permission names its table in the declaration, so this edge is certain even
        when the filter expression itself is one this parser cannot read."""
        assert any(t for t, _c in self._uses(getattr(self, form))), form

    def test_the_default_property_filter_is_read(self):
        assert ("Store", "Code") in self._uses(self.DEFAULT_FORM)

    def test_an_equals_assigned_child_filter_is_read(self):
        assert ("Sales", "Region") in self._uses(self.CHILD_EQUALS)

    def test_a_role_file_is_no_longer_declined(self):
        """Before roles were handled, a roles/*.tmdl file produced nothing and was hashed
        as empty -- so it was never revisited either."""
        assert _parse(self.CHILD_COLON, self.PATH).entities


class TestUidsAreDistinct:
    """A uid is the graph's identity. Two objects emitting the same one merge into a single
    node carrying an arbitrary winner's DAX and the union of both edge sets -- worse than a
    missing entity, because a missing entity is silence and a merged one is a confident
    wrong answer. For a *table* the merge takes every measure and column with it.

    Folding `.` and `:` is lossy, so it can do exactly that: `'A.B'` and `'A_B'` both fold
    to `A_B`. This is the same fold `config.py` performs and accepts; the difference is the
    stakes, so here a folded name earns a suffix.
    """

    TWO_TABLES = "table 'A.B'\n\tcolumn X\n\t\tdataType: int64\n\ntable 'A_B'\n\tcolumn Y\n\t\tdataType: int64\n"

    def test_two_names_that_fold_alike_do_not_collide(self):
        uids = [e.qualified_name for e in _parse(self.TWO_TABLES).entities]
        assert len(uids) == len(set(uids)), uids

    def test_children_hang_off_the_uid_their_parent_actually_got(self):
        """The subtle half. The table is disambiguated, but its columns are built from a
        qualified name computed before that -- so they would hang off the OTHER table."""
        result = _parse(self.TWO_TABLES)
        tables = {e.name: e.qualified_name for e in result.entities if e.kind == "pbi_table"}
        for column in (e for e in result.entities if e.kind == "pbi_column"):
            parent = column.qualified_name.rsplit(".column.", 1)[0]
            assert parent in tables.values(), f"{column.name} hangs off {parent}, which is no table here"

    def test_the_suffix_does_not_depend_on_declaration_order(self):
        """Derived from the name, not from arrival order. Order-dependence would mean
        reordering a file changes a uid and churns the whole subtree under it."""
        reversed_text = "table 'A_B'\n\tcolumn Y\n\t\tdataType: int64\n\ntable 'A.B'\n\tcolumn X\n\t\tdataType: int64\n"
        first = {e.name: e.qualified_name for e in _parse(self.TWO_TABLES).entities}
        second = {e.name: e.qualified_name for e in _parse(reversed_text).entities}
        assert first == second

    def test_an_unfolded_name_carries_no_suffix(self):
        """The overwhelming majority. A suffix on every name would be pure noise."""
        text = "table Sales\n\tmeasure 'Sales Amount' = SUM(Sales[X])\n"
        names = {e.name: e.qualified_name for e in _parse(text).entities}
        assert "~" not in names["Sales Amount"]


class TestSplitDottedReference:
    """TMDL references a fully qualified object with dot notation, quoting each half only
    when it must. Splitting on the first dot breaks the case where the dot is inside a
    quoted name -- which is the one that produces a silently wrong table."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Sales.Col", ("Sales", "Col")),
            ("Sales.'Product Key'", ("Sales", "Product Key")),
            ("'My Table'.'My Col'", ("My Table", "My Col")),
            ("'A.B'.Col", ("A.B", "Col")),
            ("", ("", "")),
            ("NoDotHere", ("", "")),
        ],
    )
    def test_split(self, text: str, expected: tuple[str, str]):
        assert split_dotted_reference(text) == expected


class TestRelationships:
    """`relationships.tmdl` is the model's join graph.

    Each relationship gets a node rather than being written as a direct table-to-table
    edge, and that is a lifetime decision. `_recreate_file_relationships` deletes a file's
    edges by their SOURCE node's file_path -- so a table-sourced edge declared in
    relationships.tmdl would never be deleted when that file changed, and WOULD be deleted,
    unrecoverably, when the table's own file was re-parsed.
    """

    PATH = "Contoso.SemanticModel/definition/relationships.tmdl"
    TEXT = (
        "relationship cdb6e6a9-c9d1-42b9-b9e0-484a1bc7e123\n"
        "\tfromColumn: Sales.'Product Key'\n"
        "\ttoColumn: Product.'Product Key'\n"
        "\ttoCardinality: one\n"
        "\tcrossFilteringBehavior: bothDirections\n"
        "\tisActive: false\n"
    )

    def test_a_relationship_becomes_a_node(self):
        rel = next(e for e in _parse(self.TEXT, self.PATH).entities if e.kind == "pbi_relationship")
        assert rel.name == "Sales -> Product"

    def test_it_links_both_tables(self):
        uses = {
            (r.to_name, r.properties["column"])
            for r in _parse(self.TEXT, self.PATH).relationships
            if r.rel_type == RelType.USES_TYPE
        }
        assert uses == {("Sales", "Product Key"), ("Product", "Product Key")}

    def test_the_join_metadata_rides_on_the_node(self):
        """Cardinality and cross-filter direction describe the relationship, not either
        table -- so they belong on it rather than becoming seven more nodes."""
        rel = next(e for e in _parse(self.TEXT, self.PATH).entities if e.kind == "pbi_relationship")
        assert rel.extra_properties["toCardinality"] == "one"
        assert rel.extra_properties["crossFilteringBehavior"] == "bothDirections"
        assert rel.extra_properties["isActive"] == "false"

    def test_the_guid_is_kept_but_is_not_the_identity(self):
        rel = next(e for e in _parse(self.TEXT, self.PATH).entities if e.kind == "pbi_relationship")
        assert rel.extra_properties["relationshipId"] == "cdb6e6a9-c9d1-42b9-b9e0-484a1bc7e123"
        assert "cdb6e6a9" not in rel.qualified_name

    def test_a_dot_inside_a_quoted_table_name_is_not_a_separator(self):
        text = "relationship r1\n\tfromColumn: 'My.Table'.Col\n\ttoColumn: Other.Col\n"
        uses = {r.to_name for r in _parse(text, self.PATH).relationships if r.rel_type == RelType.USES_TYPE}
        assert uses == {"My.Table", "Other"}

    def test_a_relationship_naming_no_endpoints_is_skipped(self):
        """Emitting it would put an edgeless node in the graph, which reads as an orphan
        rather than as a malformed declaration."""
        text = "relationship r1\n\tisActive: true\n"
        result = parse_file(self.PATH, text.encode("utf-8"), "proj")
        assert result is not None
        assert [e for e in result.entities if e.kind == "pbi_relationship"] == []

    def test_every_relationship_in_the_file_is_read(self):
        text = self.TEXT + "\nrelationship r2\n\tfromColumn: A.X\n\ttoColumn: B.Y\n"
        names = {e.name for e in _parse(text, self.PATH).entities if e.kind == "pbi_relationship"}
        assert names == {"Sales -> Product", "A -> B"}


class TestWarehouseObjects:
    """A partition names the warehouse object its table actually loads from, in plain text.
    That name is the one thread connecting a semantic model back to the pipeline producing
    its data -- and it is the input to the resolver that turns it into a real edge.

    Folded to lower case because that IS the join: warehouse identifiers are conventionally
    upper case and dbt model names lower case, so `FCT_ORDERS` and `fct_orders` are one
    object written by two tools. Conventionally, not by rule, which is why the resolver
    consuming this must grade an ambiguous match rather than pick one.
    """

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            # The navigation step every Snowflake-shaped connector emits.
            ('Schema{[Name="FCT_ORDERS",Kind="Table"]}[Data]', {"fct_orders"}),
            # A view is an object the warehouse produces too; which it is says nothing
            # about who produces it.
            ('Schema{[Name="DIM_CUSTOMER",Kind="View"]}[Data]', {"dim_customer"}),
            # The SQL Server shape of the same step.
            ('Sql.Database(s, d){[Schema="dbo",Item="Orders"]}[Data]', {"orders"}),
            # A native query names its objects in SQL rather than through the connector.
            ('Sql.Database(a, b, [Query="select * from analytics.marts.fct_sales"])', {"fct_sales"}),
            ('Value.NativeQuery(Source, "select 1 from RAW.EVENTS e join DIM_DATE d on 1=1")', {"events", "dim_date"}),
            # Nothing to find is a normal answer, not a failure.
            ("let x = 1 in x", set()),
            ("", set()),
        ],
    )
    def test_extraction(self, expression: str, expected: set[str]):
        assert warehouse_objects(expression) == expected

    def test_the_last_dotted_segment_wins_in_sql(self):
        """`analytics.marts.orders` and `orders` name one table with different amounts of
        qualification, and a dbt model is known by the bare name."""
        assert warehouse_objects('[Query="select * from analytics.marts.orders"]') == {"orders"}

    def test_a_partition_produces_an_import_edge(self):
        text = (
            "table Orders\n"
            "\tpartition Orders = m\n"
            "\t\tmode: import\n"
            "\t\tsource =\n"
            "\t\t\tlet\n"
            '\t\t\t\tSource = Schema{[Name="FCT_ORDERS",Kind="Table"]}[Data]\n'
            "\t\t\tin\n"
            "\t\t\t\tSource\n"
        )
        imports = [r for r in _parse(text).relationships if r.rel_type == RelType.IMPORTS]
        assert [r.to_name for r in imports] == ["warehouse.fct_orders"]
        # The edge itself still carries nothing a parser chose: `resolve_imports` builds it
        # from from_uid/to_uid alone. Properties on the *rel* are resolution inputs, not
        # edge data -- `ecosystem` decides which node the name mints (ATL-194), and TMDL
        # maps to `warehouse` so a partition and the SQL that defines the table land on one
        # node. Pinned in test_sqlite_warehouse.py.
        assert imports[0].properties == {IMPORT_ECOSYSTEM: "warehouse"}

    def test_a_direct_lake_partition_is_not_missed(self):
        """Direct Lake gives `source` children instead of an expression and names its object
        in `entityName:` -- so looking for `source =` misses every such model."""
        text = (
            "table Sales\n"
            "\tpartition Sales = entity\n"
            "\t\tmode: directLake\n"
            "\t\tsource\n"
            "\t\t\tentityName: FCT_SALES\n"
            "\t\t\tschemaName: dbo\n"
        )
        imports = {r.to_name for r in _parse(text).relationships if r.rel_type == RelType.IMPORTS}
        assert imports == {"warehouse.fct_sales"}

    def test_every_partition_of_a_table_contributes(self):
        """A table partitioned by year reads the same object several times; one partitioned
        by source reads several. Reading only the first would under-report either way."""
        text = (
            "table Sales\n"
            '\tpartition P1 = m\n\t\tsource = Schema{[Name="FCT_2023",Kind="Table"]}[Data]\n'
            '\tpartition P2 = m\n\t\tsource = Schema{[Name="FCT_2024",Kind="Table"]}[Data]\n'
        )
        imports = {r.to_name for r in _parse(text).relationships if r.rel_type == RelType.IMPORTS}
        assert imports == {"warehouse.fct_2023", "warehouse.fct_2024"}

    def test_a_table_reading_nothing_emits_nothing(self):
        """A calculated table has no partition source. An edge to a warehouse object that
        was never named would be invented, not inferred."""
        text = 'table Calc\n\tpartition Calc = calculated\n\t\tsource = ROW("a", 1)\n'
        assert [r for r in _parse(text).relationships if r.rel_type == RelType.IMPORTS] == []

    def test_two_tables_reading_one_object_converge_on_one_name(self):
        """The convergence the dotted namespace exists for: `resolve_imports` mints one stub
        per name, so both tables end up pointing at the same node."""
        text = (
            'table A\n\tpartition A = m\n\t\tsource = Schema{[Name="SHARED",Kind="Table"]}[Data]\n'
            '\ntable B\n\tpartition B = m\n\t\tsource = Schema{[Name="shared",Kind="Table"]}[Data]\n'
        )
        imports = {r.to_name for r in _parse(text).relationships if r.rel_type == RelType.IMPORTS}
        assert imports == {"warehouse.shared"}, "case folding is what makes them converge"


# ---------------------------------------------------------------------------
# PBIR — the report layer (ATL-170)
# ---------------------------------------------------------------------------

REPORT_PATH = "Sales.Report/definition/report.json"
PAGE_PATH = "Sales.Report/definition/pages/3cf1cedb01b04a3b132e/page.json"
VISUAL_PATH = "Sales.Report/definition/pages/3cf1cedb01b04a3b132e/visuals/19eb7a5feb78ab3943a8/visual.json"


def _schema(kind: str, version: str = "1.0.0") -> str:
    return f"https://developer.microsoft.com/json-schemas/fabric/item/report/definition/{kind}/{version}/schema.json"


def _pbir(path: str, document: dict):
    result = parse_file(path, json.dumps(document).encode("utf-8"), "proj")
    assert result is not None
    return result


VISUAL_DOC = {
    "$schema": _schema("visualContainer", "2.4.0"),
    "name": "19eb7a5feb78ab3943a8",
    "position": {"x": 0, "y": 0, "z": 0, "width": 100, "height": 100},
    "visual": {
        "visualType": "barChart",
        "query": {
            "queryState": {
                "Category": {
                    "projections": [
                        {
                            "field": {
                                "Column": {
                                    "Expression": {"SourceRef": {"Entity": "Product"}},
                                    "Property": "Brand",
                                }
                            },
                            "queryRef": "Product.Brand",
                            "nativeQueryRef": "Brand",
                        }
                    ]
                },
                "Y": {
                    "projections": [
                        {
                            "field": {
                                "Aggregation": {
                                    "Expression": {
                                        "Column": {
                                            "Expression": {"SourceRef": {"Entity": "Sales"}},
                                            "Property": "Amount",
                                        }
                                    },
                                    "Function": 0,
                                }
                            },
                            "queryRef": "Sum(Sales.Amount)",
                            "nativeQueryRef": "Total Amount",
                        }
                    ]
                },
            }
        },
    },
    "filterConfig": {
        "filters": [
            {
                "name": "f1",
                "field": {"Column": {"Expression": {"SourceRef": {"Entity": "Calendar"}}, "Property": "Year"}},
            }
        ]
    },
}


class TestThePbirRoute:
    """`.json` belongs to the config language; a PBIR file is claimed by what it declares."""

    def test_a_pbir_file_is_claimed(self):
        assert _pbir(VISUAL_PATH, VISUAL_DOC).entities[0].kind == "pbi_visual"

    def test_an_ordinary_json_file_is_untouched(self):
        """A theme, a layout, a lockfile — the generic handler keeps every one of them."""
        result = parse_file("StaticResources/theme.json", b'{"name": "Blue", "background": "#FFF"}', "proj")
        assert result is not None
        assert {e.kind for e in result.entities} == {"config_file", "config_setting"}

    @pytest.mark.parametrize("version", ["1.0.0", "2.4.0", "2.11.0", "99.0.0"])
    def test_the_sniff_does_not_pin_a_version(self, version: str):
        """Each kind is versioned independently and they move — files in the wild declare
        visualContainer 2.4.0, page 1.4.0, report 1.3.0. Pinning one dates the parser."""
        doc = {**VISUAL_DOC, "$schema": _schema("visualContainer", version)}
        assert _pbir(VISUAL_PATH, doc).entities[0].kind == "pbi_visual"

    @pytest.mark.parametrize("kind", ["pagesMetadata", "versionMetadata", "bookmark", "visualContainerMobileState"])
    def test_pbir_files_carrying_no_object_are_declined(self, kind: str):
        """Page order, a format version, a captured UI state, a phone layout. All real PBIR
        files, none of them a thing anyone searches for or an edge anyone traverses."""
        result = parse_file(
            f"Sales.Report/definition/{kind}.json", json.dumps({"$schema": _schema(kind)}).encode(), "proj"
        )
        assert result is not None
        assert result.entities == []


class TestPbirEntities:
    def test_a_report_is_named_by_its_folder(self):
        """`report.json` carries no name of its own — verified against real files."""
        entity = _pbir(REPORT_PATH, {"$schema": _schema("report", "1.3.0")}).entities[0]
        assert (entity.kind, entity.name) == ("pbi_report", "Sales")
        assert entity.qualified_name == "proj:pbi.report.Sales"

    def test_a_page_is_named_by_its_display_name(self):
        doc = {"$schema": _schema("page", "1.4.0"), "name": "3cf1cedb01b04a3b132e", "displayName": "Page 1"}
        entity = _pbir(PAGE_PATH, doc).entities[0]
        assert (entity.kind, entity.name) == ("pbi_page", "Page 1")
        assert entity.qualified_name == "proj:pbi.report.Sales.page.3cf1cedb01b04a3b132e"

    def test_a_visual_is_named_by_its_type_because_that_is_all_there_usually_is(self):
        """Real visuals carry no title, so the type IS the usual name rather than a
        fallback. The generated id keeps the uid unique."""
        entity = _pbir(VISUAL_PATH, VISUAL_DOC).entities[0]
        assert (entity.kind, entity.name) == ("pbi_visual", "barChart")
        assert entity.qualified_name.endswith(".page.3cf1cedb01b04a3b132e.visual.19eb7a5feb78ab3943a8")

    def test_two_visuals_of_one_type_on_one_page_stay_distinct(self):
        """They share a display name, which is fine — names repeat in code too. The uid
        must not, or the page's two bar charts merge into one node."""
        a = _pbir(VISUAL_PATH, VISUAL_DOC).entities[0]
        other = {**VISUAL_DOC, "name": "aaaa1111bbbb2222cccc"}
        b = _pbir(VISUAL_PATH.replace("19eb7a5feb78ab3943a8", "aaaa1111bbbb2222cccc"), other).entities[0]
        assert a.name == b.name
        assert a.qualified_name != b.qualified_name

    def test_every_file_has_exactly_one_module(self):
        """Only Module, Package and DocFile carry `file_hash`, so a report object modelled
        as anything else re-parses on every indexing pass forever."""
        for path, doc in ((REPORT_PATH, {"$schema": _schema("report")}), (VISUAL_PATH, VISUAL_DOC)):
            entities = _pbir(path, doc).entities
            assert [e.label for e in entities] == [NodeLabel.MODULE]
            assert entities[0].line_start == 1

    def test_the_field_wells_are_searchable_text(self):
        """A visual named `barChart` is unfindable by what it shows. The queryRefs are what
        a report author sees in the field well, so they go where BM25 can reach them."""
        source = _source(_pbir(VISUAL_PATH, VISUAL_DOC).entities[0])
        assert "Brand" in source
        assert "Total Amount" in source


class TestPbirFieldReferences:
    def _uses(self, path: str, doc: dict) -> set:
        return {(r.to_name, r.properties["column"]) for r in _pbir(path, doc).relationships if r.properties}

    def test_a_bare_column_projection(self):
        assert ("Product", "Brand") in self._uses(VISUAL_PATH, VISUAL_DOC)

    def test_an_aggregated_projection_is_not_missed(self):
        """`Aggregation` WRAPS the column. Matching one level deep finds the bare
        projections and misses every Sum(...) — which is most of them in a real report."""
        assert ("Sales", "Amount") in self._uses(VISUAL_PATH, VISUAL_DOC)

    def test_a_filter_outside_the_query_state_is_found(self):
        """The same container appears under filterConfig, sortDefinition and formatting
        rules. Walking only `queryState` under-reports what a visual depends on."""
        assert ("Calendar", "Year") in self._uses(VISUAL_PATH, VISUAL_DOC)

    def test_a_measure_reference_is_found(self):
        doc = {
            "$schema": _schema("visualContainer"),
            "name": "v1",
            "visual": {
                "visualType": "card",
                "query": {
                    "queryState": {
                        "Values": {
                            "projections": [
                                {
                                    "field": {
                                        "Measure": {
                                            "Expression": {"SourceRef": {"Entity": "Measures"}},
                                            "Property": "Total Sales",
                                        }
                                    }
                                }
                            ]
                        }
                    }
                },
            },
        }
        assert ("Measures", "Total Sales") in self._uses(VISUAL_PATH, doc)

    def test_a_source_alias_without_an_entity_produces_nothing(self):
        """`semanticQuery`'s own SourceRef documents only `Source`, a query alias into a
        `From` clause a visual has none of. Real files carry `Entity` instead; an alias
        cannot be resolved to a table here, and guessing would invent an edge."""
        doc = {
            "$schema": _schema("visualContainer"),
            "name": "v1",
            "visual": {
                "visualType": "card",
                "query": {
                    "queryState": {
                        "Values": {
                            "projections": [
                                {"field": {"Column": {"Expression": {"SourceRef": {"Source": "s"}}, "Property": "X"}}}
                            ]
                        }
                    }
                },
            },
        }
        assert self._uses(VISUAL_PATH, doc) == set()

    def test_a_page_level_filter_gives_the_page_its_edges(self):
        doc = {
            "$schema": _schema("page"),
            "name": "p1",
            "displayName": "Overview",
            "filterConfig": {
                "filters": [
                    {"field": {"Column": {"Expression": {"SourceRef": {"Entity": "Store"}}, "Property": "Region"}}}
                ]
            },
        }
        assert self._uses(PAGE_PATH, doc) == {("Store", "Region")}


class TestPbirRobustness:
    def test_malformed_json_is_declined_rather_than_raised(self):
        result = parse_file(VISUAL_PATH, b'{"$schema": "' + _schema("visualContainer").encode() + b'", broken', "proj")
        assert result is not None
        assert result.entities == []

    def test_a_json_array_at_the_top_level_is_declined(self):
        result = parse_file(VISUAL_PATH, b"[1, 2, 3]", "proj")
        assert result is not None

    def test_an_unknown_pbir_kind_degrades_to_nothing_rather_than_an_error(self):
        """Microsoft adds file kinds. A new one must cost its own node, never the report."""
        result = parse_file(
            "Sales.Report/definition/newThing.json", json.dumps({"$schema": _schema("newThing")}).encode(), "proj"
        )
        assert result is not None
        assert result.entities == []
