"""A YAML date in frontmatter must not take a whole batch down.

`yaml.safe_load` turns an unquoted `2026-08-30` into a `datetime.date`, and the markdown
parser's `_coerce_property_value` **keeps** it on purpose: Bolt has a Date type, so that
is the right value for the backend it was written against.

`json.dumps` does not have one. Before `_dumps`, writing such a node raised
`TypeError: Object of type date is not JSON serializable` — and not as a lost property.
The exception left `upsert_file_entities` and failed the whole **batch**, so every
unrelated file travelling with the dated one was retried five times and then parked as a
poison message. One dated ADR silently dropped a batch of this repo's own documentation,
and the only place it was visible was the benchmark's own stdout.

That is why this is a test and not a comment: the symptom appears on files that have
nothing wrong with them, which makes it close to undiagnosable from the outside.
"""

from __future__ import annotations

import datetime
import json
from typing import TYPE_CHECKING

import pytest

from code_atlas.backends.sqlite_graph import SqliteGraphClient, _dumps
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import NodeLabel, Visibility

if TYPE_CHECKING:
    from pathlib import Path


def _note(name: str, *, extra: dict) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=f"proj:note:{name}",
        label=NodeLabel.NOTE,
        kind="note",
        line_start=1,
        line_end=2,
        file_path=f"wiki/{name}.md",
        visibility=Visibility.PUBLIC,
        content_hash="h1",
        extra_properties=extra,
    )


class TestTemporalsSerialise:
    @pytest.mark.parametrize(
        "value",
        [
            datetime.date(2026, 8, 30),
            # Naive on purpose: `yaml.safe_load` produces a naive datetime for an
            # unquoted timestamp, so a tz-aware one would test a value this never sees.
            datetime.datetime(2026, 8, 30, 14, 5, 1),  # noqa: DTZ001
            datetime.time(9, 30),
        ],
    )
    def test_each_temporal_type_becomes_an_iso_string(self, value):
        """All three, because YAML produces all three and only `date` was ever seen."""
        assert json.loads(_dumps({"when": value}))["when"] == value.isoformat()

    def test_nesting_is_reached(self):
        """Frontmatter is stored as one nested map (ADR-0039), so the date is never at
        the top level in practice — it is under `frontmatter.<key>`."""
        payload = {"frontmatter": {"meta": {"created": datetime.date(2026, 8, 30)}, "tags": ["a"]}}
        assert json.loads(_dumps(payload))["frontmatter"]["meta"]["created"] == "2026-08-30"

    def test_ordinary_values_are_untouched(self):
        """Non-vacuity in the other direction: a `default=` that fired on everything
        would stringify every int and float in every props_json in the database."""
        payload = {"line_start": 1, "weight": 0.5, "ok": True, "name": "x", "none": None, "list": [1, "2"]}
        assert json.loads(_dumps(payload)) == payload


class TestTheBatchSurvives:
    async def test_a_dated_note_does_not_fail_the_files_beside_it(self, tmp_path: Path):
        """The actual failure mode, reproduced at the level it occurred.

        Asserting only that the dated node writes would miss the point: the damage was to
        its neighbours. So the batch carries one dated file and one ordinary one, and the
        ordinary one has to arrive.
        """
        async with SqliteGraphClient(tmp_path / "graph.sqlite3") as client:
            await client.ensure_schema()
            await client.merge_project_node("proj")

            await client.upsert_file_entities(
                "proj",
                "wiki/dated.md",
                [_note("dated", extra={"frontmatter": {"created": datetime.date(2026, 8, 30)}})],
                [],
            )
            await client.upsert_file_entities(
                "proj", "wiki/plain.md", [_note("plain", extra={"frontmatter": {"kind": "note"}})], []
            )

            assert await client.node_exists("proj:note:plain"), (
                "the undated note beside the dated one is missing — a serialisation error in one "
                "file is taking its whole batch down again"
            )
            dated = await client.get_entity_by_uid("proj:note:dated")
            assert dated is not None
            assert dated["frontmatter"]["created"] == "2026-08-30", (
                f"the date did not round-trip as an ISO string: {dated.get('frontmatter')}"
            )
