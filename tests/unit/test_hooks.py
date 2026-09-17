"""Claude Code routing hooks: what counts as a symbol search, what the agent is told, when it is blocked."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from code_atlas import hooks
from code_atlas.backends.sqlite_graph import SqliteGraphClient
from code_atlas.cli import app
from code_atlas.parsing.ast import ParsedEntity
from code_atlas.schema import SCHEMA_VERSION, NodeLabel, Visibility

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("def resolve_context", ["resolve_context"]),
        (r"\bGraphClient\b", ["GraphClient"]),
        (r"class EventBus\(", ["EventBus"]),
        (r"^\s*def index", ["index"]),
        (r"get_node\(", ["get_node"]),
        ("create_graph_client|ensure_schema", ["create_graph_client", "ensure_schema"]),
        ("self.graph.ping", ["ping"]),
        # Text searches the graph cannot answer -- a hint or a block here would be wrong.
        ("TODO", []),
        ("import", []),
        ("error: .*", []),
        ("foo.*bar", []),
        ("GraphClient|timed out", []),
    ],
)
def test_symbols_in_pattern(pattern: str, expected: list[str]) -> None:
    assert hooks.symbols_in_pattern(pattern) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('grep -rn "GraphClient" src', "GraphClient"),
        ("rtk grep -rn ensure_schema src | head", "ensure_schema"),
        ("git grep -e derive_project_name", "derive_project_name"),
        ("ls src && rg --type py 'class Foo' src", "class Foo"),
        ("rg -A 3 -g '*.py' EventBus", "EventBus"),
        ("echo grep something", None),
        ("uv run pytest tests/unit", None),
    ],
)
def test_search_pattern_from_bash(command: str, expected: str | None) -> None:
    assert hooks.search_pattern("Bash", {"command": command}) == expected


# ---------------------------------------------------------------------------
# Event handlers, with the graph stubbed at the two seams
# ---------------------------------------------------------------------------

_CTX = {"available": True, "project": "proj", "graph_project": "proj", "entities": 42, "backend": "sqlite"}
_ROW = {
    "name": "create_graph_client",
    "uid": "proj:backends.create_graph_client",
    "label": "Callable",
    "labels": "Callable",
    "file": "src/backends.py",
    "line": 196,
    "refs": 4,
    "callers": 3,
}


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Session state in tmp, and a graph that knows exactly one name."""
    monkeypatch.setattr(hooks, "_state_dir", lambda: tmp_path / "state")
    calls: dict[str, Any] = {"ctx": dict(_CTX), "lookups": 0}

    def fake_lookup(_ctx: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
        calls["lookups"] += 1
        return [_ROW] if _ROW["name"] in names else []

    monkeypatch.setattr(hooks, "resolve_context", lambda _cwd: calls["ctx"])
    monkeypatch.setattr(hooks, "lookup", fake_lookup)
    return calls


def _out(capsys: pytest.CaptureFixture[str]) -> dict[str, Any] | None:
    text = capsys.readouterr().out
    return json.loads(text)["hookSpecificOutput"] if text else None


def _grep(pattern: str, **extra: Any) -> dict[str, Any]:
    return {"session_id": "s1", "cwd": ".", "tool_name": "Grep", "tool_input": {"pattern": pattern}, **extra}


def test_start_card_only_when_indexed(graph: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    hooks.on_start({"session_id": "s1", "cwd": "."}, "SubagentStart")
    out = _out(capsys)
    assert out is not None
    assert out["hookEventName"] == "SubagentStart"
    assert "get_node" in out["additionalContext"]
    assert "42 code entities" in out["additionalContext"]

    graph["ctx"] = {"available": False, "reason": "not-indexed"}
    hooks.on_start({"session_id": "s2", "cwd": "."}, "SessionStart")
    assert _out(capsys) is None


def test_post_tool_hints_once_per_symbol_and_never_on_a_miss(
    graph: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    hooks.on_post_tool(_grep(r"\bcreate_graph_client\b"))
    out = _out(capsys)
    assert out is not None
    hint = out["additionalContext"]
    assert "src/backends.py:196" in hint
    assert "3 callers" in hint
    assert 'get_context("proj:backends.create_graph_client")' in hint

    hooks.on_post_tool(_grep("def create_graph_client"))
    assert _out(capsys) is None, "the same symbol is hinted once"

    hooks.on_post_tool(_grep("def unknown_function"))
    assert _out(capsys) is None, "absence from the index is not absence from the code"

    hooks.on_post_tool(_grep("TODO"))
    assert _out(capsys) is None
    assert graph["lookups"] == 2, "a text search never reaches the graph"


def test_no_hints_once_the_agent_uses_the_graph(graph: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    hooks.on_post_tool({"session_id": "s1", "tool_name": "mcp__code-atlas__get_node", "tool_input": {}})
    hooks.on_post_tool(_grep(r"\bcreate_graph_client\b"))
    assert _out(capsys) is None
    assert graph["lookups"] == 0


def test_hints_are_capped_per_agent(graph: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    for i in range(hooks._MAX_HINTS_PER_AGENT):
        hooks.on_post_tool(_grep(f"def other_name_{i}"))
    hooks.on_post_tool(_grep("def create_graph_client"))
    assert _out(capsys) is None
    assert graph["lookups"] == hooks._MAX_HINTS_PER_AGENT


def test_non_strict_pre_tool_never_blocks(graph: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    hooks.on_pre_tool(_grep("def create_graph_client"), strict=False)
    assert _out(capsys) is None
    assert graph["lookups"] == 0


def test_strict_blocks_an_answerable_search_once(graph: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    hooks.on_pre_tool(_grep("def unknown_function"), strict=True)
    assert _out(capsys) is None, "a search the graph cannot answer is never blocked"

    hooks.on_pre_tool(_grep("def create_graph_client"), strict=True)
    out = _out(capsys)
    assert out is not None
    assert out["permissionDecision"] == "deny"
    assert "src/backends.py:196" in out["permissionDecisionReason"]

    hooks.on_pre_tool(_grep("def create_graph_client"), strict=True)
    assert _out(capsys) is None, "the re-issued search is allowed"


def test_strict_block_is_per_agent_and_lifted_by_using_the_graph(
    graph: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    hooks.on_pre_tool(_grep("def create_graph_client"), strict=True)
    assert _out(capsys) is not None

    sub = {"agent_id": "a1", "agent_type": "Explore"}
    hooks.on_post_tool({"session_id": "s1", "tool_name": "mcp__code-atlas__get_node", "tool_input": {}, **sub})
    hooks.on_pre_tool(_grep("def create_graph_client", **sub), strict=True)
    assert _out(capsys) is None, "an agent that already used code-atlas is not blocked"


def test_strict_delegation_block(graph: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    def spawn(subagent_type: str) -> dict[str, Any]:
        return {"session_id": "s1", "tool_name": "Agent", "tool_input": {"subagent_type": subagent_type}}

    hooks.on_pre_tool(spawn("reviewer"), strict=True)
    assert _out(capsys) is None, "a reviewer is delegated for its work, not its search"

    hooks.on_pre_tool(spawn("Explore"), strict=True)
    out = _out(capsys)
    assert out is not None
    assert out["permissionDecision"] == "deny"

    hooks.on_pre_tool(spawn("Explore"), strict=True)
    assert _out(capsys) is None


def test_main_fails_open(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def boom(_payload: dict[str, Any]) -> None:
        raise RuntimeError("graph exploded")

    monkeypatch.setattr(hooks, "on_post_tool", boom)
    monkeypatch.setattr("sys.stdin", type("S", (), {"buffer": type("B", (), {"read": lambda _s: b"{}"})()})())
    assert hooks.main(["post-tool"]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "graph exploded" in captured.err


def test_mcp_availability_follows_claude_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, main, wt = tmp_path / "home", tmp_path / "repo", tmp_path / "repo" / ".claude" / "worktrees" / "w"
    for d in (home, wt):
        d.mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.delenv("ATLAS_HOOKS_MCP", raising=False)
    claude_json = home / ".claude.json"

    def config(project: dict[str, Any], top: dict[str, Any] | None = None) -> None:
        claude_json.write_text(json.dumps({"mcpServers": top or {}, "projects": {str(main): project}}), "utf-8")

    assert hooks.mcp_unavailable([wt, main]) == "mcp-not-configured"

    config({"mcpServers": {"code-atlas": {}}})
    assert hooks.mcp_unavailable([wt, main]) is None, "a worktree inherits the main repository's entry"

    config({"mcpServers": {"code-atlas": {}}, "disabledMcpServers": ["code-atlas"]})
    assert hooks.mcp_unavailable([wt, main]) == "mcp-disabled"

    config({}, top={"code-atlas": {}})
    assert hooks.mcp_unavailable([wt, main]) is None

    claude_json.unlink()
    (main / ".mcp.json").write_text('{"mcpServers": {"code-atlas": {}}}', encoding="utf-8")
    assert hooks.mcp_unavailable([wt, main]) is None

    (main / ".mcp.json").unlink()
    monkeypatch.setenv("ATLAS_HOOKS_MCP", "1")
    assert hooks.mcp_unavailable([wt, main]) is None


# ---------------------------------------------------------------------------
# Fail-fast health: problems found while resolving the session's graph
# ---------------------------------------------------------------------------


def _atlas_settings(tmp_path: Path, graph: str, queue: str) -> Any:
    from code_atlas.settings import AtlasSettings

    backend: dict[str, Any] = {}
    if graph != "auto":
        backend["graph"] = {graph: {}}
    if queue != "auto":
        backend["queue"] = {queue: {}}
    return AtlasSettings(project_root=tmp_path, backend=backend)


def _conclusion(result: Any) -> tuple[str, str, str, str]:
    """What a user acts on. `detail` is left out: it carries the transport's own error text."""
    return (result.name, str(result.status), result.message, result.suggestion)


def _sqlite_graph(ping: bool = True) -> Any:
    from unittest.mock import AsyncMock

    graph = SqliteGraphClient.__new__(SqliteGraphClient)
    graph.ping = AsyncMock(return_value=ping)
    return graph


def _sqlite_bus() -> Any:
    from unittest.mock import AsyncMock

    from code_atlas.backends.sqlite_queue import SqliteEventBus

    bus = SqliteEventBus.__new__(SqliteEventBus)
    bus.ping = AsyncMock(return_value=True)
    return bus


def _down() -> Any:
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.ping = AsyncMock(side_effect=ConnectionRefusedError("refused"))
    return client


def _up() -> Any:
    from unittest.mock import AsyncMock

    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    return client


# (graph config, queue config, what the hook's probes found, what atlas health's clients are)
_SCENARIOS = {
    "memgraph declared and down": (
        "memgraph",
        "valkey",
        {"reason": "unreachable", "backend": "memgraph"},
        {"valkey": True},
        _down,
        _up,
    ),
    "memgraph up, valkey declared and down": (
        "memgraph",
        "valkey",
        {"backend": "memgraph"},
        {"valkey": False},
        _up,
        _down,
    ),
    "undeclared, both fell back to sqlite": (
        "auto",
        "auto",
        {"backend": "sqlite"},
        {"valkey": False},
        _sqlite_graph,
        _sqlite_bus,
    ),
    "sqlite declared on both axes": ("sqlite", "sqlite", {"backend": "sqlite"}, {}, _sqlite_graph, _sqlite_bus),
    "everything up": ("memgraph", "valkey", {"backend": "memgraph"}, {"valkey": True}, _up, _up),
}


@pytest.mark.parametrize("scenario", list(_SCENARIOS))
async def test_the_hook_and_atlas_health_reach_the_same_verdicts(tmp_path: Path, scenario: str) -> None:
    """The contract behind sharing verdicts: they may probe differently, never conclude differently."""
    from code_atlas.server.health import check_memgraph, check_valkey

    graph_choice, queue_choice, probed, up, graph_client, bus_client = _SCENARIOS[scenario]
    settings = _atlas_settings(tmp_path, graph_choice, queue_choice)

    hook = {r.name: _conclusion(r) for r in hooks._backend_results({"project": "p", **probed}, settings, up)}
    health = {
        "memgraph": _conclusion(
            await check_memgraph(graph_client(), settings.memgraph, chosen=graph_choice == "sqlite")
        ),
        "valkey": _conclusion(await check_valkey(bus_client(), settings.redis, chosen=queue_choice == "sqlite")),
    }
    if queue_choice == "sqlite":
        health.pop("valkey")  # a declared SQLite queue is not probed by the hook: there is no server to reach
    assert hook == health


async def test_schema_verdicts_match(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock

    from code_atlas.server.health import check_schema

    settings = _atlas_settings(tmp_path, "sqlite", "sqlite")
    for stored in (None, SCHEMA_VERSION - 1, SCHEMA_VERSION, SCHEMA_VERSION + 1):
        graph = AsyncMock()
        graph.get_schema_version = AsyncMock(return_value=stored)
        hook = hooks._backend_results({"project": "p", "backend": "sqlite", "schema": stored}, settings, {})
        (hook_schema,) = [r for r in hook if r.name == "schema"]
        assert _conclusion(hook_schema) == _conclusion(await check_schema(graph)), stored


def test_the_hook_alone_asks_whether_this_repository_is_indexed(tmp_path: Path) -> None:
    settings = _atlas_settings(tmp_path, "memgraph", "valkey")
    results = hooks._backend_results(
        {"project": "proj", "backend": "memgraph", "reason": "not-indexed"}, settings, {"valkey": True}
    )
    problems = [hooks.format_problem(r) for r in results if r.status != "ok"]
    assert problems == [
        "index FAIL: This repository is not indexed (project proj). Fix: Run 'atlas index' in the repository."
    ]


@pytest.fixture
def embed_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    monkeypatch.setattr(hooks, "_state_dir", lambda: tmp_path / "state")
    calls: dict[str, Any] = {"n": 0, "ok": True}

    def fake_call(_settings: Any) -> bool:
        calls["n"] += 1
        return calls["ok"]

    monkeypatch.setattr(hooks, "_embedding_call_succeeds", fake_call)
    return calls


def test_an_embedding_success_is_cached_per_config_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embed_calls: dict[str, Any]
) -> None:
    import time_machine

    from code_atlas.settings import AtlasSettings

    monkeypatch.setenv("OPENAI_API_KEY", "key-one")
    settings = AtlasSettings(project_root=tmp_path, embeddings={"enabled": True, "provider": "litellm", "model": "m"})

    assert hooks.check_embeddings_cached(settings).status == "ok"
    assert hooks.check_embeddings_cached(settings).status == "ok"
    assert embed_calls["n"] == 1, "a recent success for the same config is not re-checked"

    monkeypatch.setenv("OPENAI_API_KEY", "key-two")
    hooks.check_embeddings_cached(settings)
    assert embed_calls["n"] == 2, "a rotated key is a different config"

    with time_machine.travel(time.time() + hooks._EMBEDDINGS_OK_TTL_S + 60):
        hooks.check_embeddings_cached(settings)
    assert embed_calls["n"] == 3, "a day-old success is re-checked"

    assert "key-two" not in (tmp_path / "state" / "embeddings-ok.json").read_text(encoding="utf-8")


def test_a_library_printing_to_stdout_cannot_corrupt_the_hook_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """litellm prints a banner to stdout on a failed call; Claude Code would discard the result."""

    def noisy_start(_payload: dict[str, Any], event: str) -> None:
        print("Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new")
        hooks._emit(event, additionalContext="card")

    monkeypatch.setattr(hooks, "on_start", noisy_start)
    monkeypatch.setattr("sys.stdin", type("S", (), {"buffer": type("B", (), {"read": lambda _s: b"{}"})()})())
    assert hooks.main(["session-start"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["hookSpecificOutput"]["additionalContext"] == "card"
    assert "Give Feedback" in captured.err


def test_a_slow_provider_is_not_reported_as_a_broken_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embed_calls: dict[str, Any]
) -> None:
    from code_atlas.settings import AtlasSettings

    def times_out(_settings: Any) -> bool:
        raise TimeoutError

    monkeypatch.setattr(hooks, "_embedding_call_succeeds", times_out)
    settings = AtlasSettings(project_root=tmp_path, embeddings={"enabled": True, "provider": "litellm", "model": "m"})
    result = hooks.check_embeddings_cached(settings)
    assert result.status == "warn"
    assert "timeout" in result.message
    assert "API key" not in result.suggestion
    assert not (tmp_path / "state" / "embeddings-ok.json").exists(), "a timeout is not a success"


def test_the_hook_builds_a_real_embedding_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the provider call is stubbed. Every other embedding test replaces the whole call,
    so a constructor change -- `limiter` became required -- surfaced only as a false
    "provider unreachable" at session start."""
    from code_atlas.search.embeddings import EmbedClient
    from code_atlas.settings import AtlasSettings

    async def answers(_self: Any) -> bool:
        return True

    monkeypatch.setattr(EmbedClient, "health_check", answers)
    settings = AtlasSettings(project_root=tmp_path, embeddings={"enabled": True, "provider": "litellm", "model": "m"})
    assert hooks._embedding_call_succeeds(settings) is True


def test_an_embedding_failure_is_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, embed_calls: dict[str, Any]
) -> None:
    from code_atlas.settings import AtlasSettings

    embed_calls["ok"] = False
    settings = AtlasSettings(project_root=tmp_path, embeddings={"enabled": True, "provider": "litellm", "model": "m"})
    for _ in range(2):
        result = hooks.check_embeddings_cached(settings)
        assert result.status == "warn"
        assert result.suggestion
    assert embed_calls["n"] == 2

    disabled = AtlasSettings(project_root=tmp_path, embeddings={"enabled": False})
    assert hooks.check_embeddings_cached(disabled).status == "ok"
    assert embed_calls["n"] == 2, "disabled embeddings are never called"


class _FakeSocket:
    """A socket whose connect to "slow" hangs past any timeout and to "fast" succeeds at once."""

    def __init__(self, family: Any, _kind: Any) -> None:
        self.family = family
        self.timeout = 0.0

    def __enter__(self) -> _FakeSocket:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def connect(self, address: tuple[str, int]) -> None:
        import time

        if address[0].startswith("slow"):
            time.sleep(self.timeout)
            raise TimeoutError


def test_ports_are_probed_concurrently_and_settled_by_the_first_address(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket
    import time

    def fake_getaddrinfo(host: str, port: int, **_kwargs: Any) -> list[Any]:
        # Every host resolves twice, like `localhost`; "mixed" has a hanging IPv6 and a live IPv4.
        v6 = "slow-v6" if host in ("mixed", "slow") else "fast-v6"
        v4 = "slow-v4" if host == "slow" else "fast-v4"
        return [(socket.AF_INET6, 0, 0, "", (v6, port)), (socket.AF_INET, 0, 0, "", (v4, port))]

    monkeypatch.setattr("socket.getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr("socket.socket", _FakeSocket)

    t0 = time.perf_counter()
    assert hooks.ports_open([("slow", 1), ("mixed", 2), ("slow", 3)], timeout_s=0.3) == [False, True, False]
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.55, f"{elapsed:.2f}s: targets or addresses were probed one after the other"

    t0 = time.perf_counter()
    assert hooks.ports_open([("mixed", 1)], timeout_s=2.0) == [True]
    assert time.perf_counter() - t0 < 0.5, "a target that answered on IPv4 waited out its hanging IPv6 address"


def test_the_real_probe_sees_a_listening_port() -> None:
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        assert hooks.ports_open([("127.0.0.1", listener.getsockname()[1])], timeout_s=1.0) == [True]


def test_connect_timeout_setting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic import ValidationError

    from code_atlas.settings import AtlasSettings

    monkeypatch.setenv("ATLAS_HOOKS_MCP", "1")  # the hook's own switch must not read as a settings key
    assert AtlasSettings(project_root=tmp_path).health.connect_timeout_s == 1.0

    monkeypatch.setenv("ATLAS_HEALTH__CONNECT_TIMEOUT_S", "5")
    assert AtlasSettings(project_root=tmp_path).health.connect_timeout_s == 5.0

    monkeypatch.delenv("ATLAS_HEALTH__CONNECT_TIMEOUT_S")
    (tmp_path / "atlas.local.toml").write_text("[health]\nconnect_timeout_s = 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        AtlasSettings(project_root=tmp_path)


def test_session_start_relays_problems_to_the_user_but_subagents_only_get_the_card(
    graph: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    graph["ctx"] = {**_CTX, "problems": ["Valkey at x:1 is unreachable. Fix: docker compose up -d valkey"]}
    hooks.on_start({"session_id": "s1", "cwd": "."}, "SessionStart")
    raw = json.loads(capsys.readouterr().out)
    assert raw["systemMessage"].startswith("code-atlas: Valkey at x:1")
    context = raw["hookSpecificOutput"]["additionalContext"]
    assert "get_node" in context, "a degraded index still gets the routing card"
    assert "Tell the user" in context

    hooks.on_start({"session_id": "s1", "cwd": ".", "agent_id": "a1"}, "SubagentStart")
    raw = json.loads(capsys.readouterr().out)
    assert "systemMessage" not in raw
    assert "Tell the user" not in raw["hookSpecificOutput"]["additionalContext"]


def test_session_start_speaks_when_the_graph_is_down(graph: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    graph["ctx"] = {"available": False, "reason": "unreachable", "problems": ["Memgraph at bolt://x is unreachable."]}
    hooks.on_start({"session_id": "s1", "cwd": "."}, "SessionStart")
    raw = json.loads(capsys.readouterr().out)
    assert "Memgraph at bolt://x" in raw["systemMessage"]
    assert "get_node" not in raw["hookSpecificOutput"]["additionalContext"], "no card for tools that cannot answer"


# ---------------------------------------------------------------------------
# The SQLite lookup query against a real schema
# ---------------------------------------------------------------------------


def _entity(name: str, label: NodeLabel = NodeLabel.CALLABLE) -> ParsedEntity:
    return ParsedEntity(
        name=name,
        qualified_name=f"proj:mod.{name}",
        label=label,
        kind="function" if label == NodeLabel.CALLABLE else "class",
        line_start=7,
        line_end=9,
        file_path="mod.py",
        docstring=None,
        signature=None,
        visibility=Visibility.PUBLIC,
        content_hash="h",
    )


async def test_sqlite_context_and_lookup(tmp_path: Path) -> None:
    path = tmp_path / "graph.sqlite3"
    async with SqliteGraphClient(path) as client:
        await client.ensure_schema()
        entities = [_entity("target"), _entity("caller_a"), _entity("caller_b"), _entity("Holder", NodeLabel.TYPE_DEF)]
        await client.upsert_file_entities("proj", "mod.py", entities, [])
        conn = await client._get_conn()
        await conn.executemany(
            "INSERT INTO edges(from_uid, to_uid, rel_type) VALUES (?, ?, ?)",
            [
                ("proj:mod.caller_a", "proj:mod.target", "CALLS"),
                ("proj:mod.caller_b", "proj:mod.target", "CALLS"),
                ("proj:mod.Holder", "proj:mod.target", "REFERENCES"),
            ],
        )
        await conn.commit()

    ctx: dict[str, Any] = {"project": "proj", "backend": "sqlite", "path": str(path)}
    picked = hooks._pick_project(ctx, hooks._run(ctx, hooks._count_query(ctx), projects=["proj", "other"]))
    assert picked["available"] is True
    assert picked["entities"] == 4
    assert picked["schema"] == SCHEMA_VERSION

    rows = hooks.lookup(picked, ["target", "missing"])
    assert len(rows) == 1
    row = rows[0]
    assert (row["uid"], row["label"], row["file"], row["line"]) == ("proj:mod.target", "Callable", "mod.py", 7)
    assert (row["callers"], row["refs"]) == (2, 3)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def _tool_env(tmp_path: Path, *, with_script: bool) -> tuple[Path, Path]:
    """A fake `code-atlas-mcp` uv tool environment: its interpreter, and a receipt naming the scripts
    uv installed into a separate bin directory, the way a real `uv tool install` lays them out."""
    env = tmp_path / "uv-tools" / "code-atlas-mcp"
    python = env / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    bin_dir = tmp_path / "bin with space"
    bin_dir.mkdir()
    entrypoints = [("atlas", bin_dir / "atlas.exe")]
    if with_script:
        entrypoints.append(("atlas-hook", bin_dir / "atlas-hook.exe"))
    lines = ["[tool]", "entrypoints = ["]
    for name, path in entrypoints:
        path.write_text("", encoding="utf-8")
        lines.append(f'    {{ name = "{name}", install-path = "{path.as_posix()}", from = "code-atlas-mcp" }},')
    lines.append("]")
    (env / "uv-receipt.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return python, bin_dir / "atlas-hook.exe"


def test_merge_keeps_foreign_hooks_and_replaces_ours() -> None:
    foreign = {"matcher": "Bash", "hooks": [{"type": "command", "command": "rtk hook claude"}]}
    settings: dict[str, Any] = {"model": "x", "hooks": {"PreToolUse": [foreign]}}
    script = '"C:/Users/me/.local/bin/atlas-hook.exe"'

    hooks.merge_settings(settings, hooks.hook_config(strict=True, launcher=script))
    hooks.merge_settings(settings, hooks.hook_config(strict=True, launcher=script))
    pre: list[dict[str, Any]] = settings["hooks"]["PreToolUse"]
    assert pre[0] == foreign
    assert len(pre) == 2, "re-installing replaces, never duplicates"
    assert pre[1]["hooks"][0]["command"] == f"{script} pre-tool --strict"

    hooks.merge_settings(settings, hooks.hook_config(strict=False))
    assert settings["hooks"]["PreToolUse"] == [foreign], "dropping --strict removes the blocking hook"

    hooks.merge_settings(settings, None)
    assert settings == {"model": "x", "hooks": {"PreToolUse": [foreign]}}


def test_an_upgrade_replaces_the_old_module_form() -> None:
    """Every install before `atlas-hook` wrote `"<python>" -m code_atlas.hooks`. Re-installing must
    recognise those entries as ours, or an upgrade leaves every hook running twice."""
    old = hooks.hook_config(strict=True, launcher='"C:/tools/code-atlas-mcp/Scripts/python.exe" -m code_atlas.hooks')
    settings: dict[str, Any] = {"hooks": old}

    hooks.merge_settings(settings, hooks.hook_config(strict=True, launcher='"C:/bin/atlas-hook.exe"'))

    commands = [h["command"] for groups in settings["hooks"].values() for g in groups for h in g["hooks"]]
    assert len(commands) == 4, commands
    assert all(c.startswith('"C:/bin/atlas-hook.exe" ') for c in commands), commands


def test_is_ours_is_not_fooled_by_a_similar_name() -> None:
    def group(command: str) -> dict[str, Any]:
        return {"hooks": [{"type": "command", "command": command}]}

    assert hooks._is_ours(group('"C:/bin/atlas-hook.exe" post-tool'))
    assert hooks._is_ours(group('"/home/me/.local/bin/atlas-hook" post-tool'))
    assert hooks._is_ours(group('"C:/py/python.exe" -m code_atlas.hooks post-tool'))
    assert not hooks._is_ours(group('"C:/bin/atlas-hooks-extra.exe" post-tool'))
    assert not hooks._is_ours(group("rtk hook claude"))


def test_cli_install_and_uninstall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    _python, script = _tool_env(tmp_path, with_script=True)
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "uv-tools"))
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir()
    settings.write_text('{"env": {"A": "1"}}', encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(app, ["hooks", "install", "--strict"])
    assert result.exit_code == 0, result.output
    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["env"] == {"A": "1"}
    assert set(data["hooks"]) == {"SessionStart", "SubagentStart", "PostToolUse", "PreToolUse"}
    commands = {h["command"] for groups in data["hooks"].values() for g in groups for h in g["hooks"]}
    assert all(c.startswith(f'"{script.as_posix()}" ') for c in commands), "hooks run as the tool's atlas-hook"
    assert (tmp_path / ".claude" / "settings.json.atlas-bak").exists()

    result = runner.invoke(app, ["hooks", "install", "--python", "C:/elsewhere/python.exe"])
    assert result.exit_code == 0, result.output
    data = json.loads(settings.read_text(encoding="utf-8"))
    commands = {h["command"] for groups in data["hooks"].values() for g in groups for h in g["hooks"]}
    assert len(commands) == 3, "the atlas-hook entries were replaced, not kept alongside"
    assert all(c.startswith('"C:/elsewhere/python.exe" -m code_atlas.hooks ') for c in commands), "--python wins"

    result = runner.invoke(app, ["hooks", "uninstall"])
    assert result.exit_code == 0, result.output
    assert json.loads(settings.read_text(encoding="utf-8")) == {"env": {"A": "1"}}


def test_hook_launcher_prefers_the_uv_tool_over_a_development_venv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning the venv `atlas hooks install` ran from tied every Claude Code session to a checkout's
    development venv, which `uv sync` rewrites and a running hook locks."""
    dev = tmp_path / "checkout"
    (dev / ".venv").mkdir(parents=True)
    (dev / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.setattr("sys.prefix", str(dev / ".venv"))
    monkeypatch.setattr("sys.executable", str(dev / ".venv" / "Scripts" / "python.exe"))
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "uv-tools"))

    python, script = _tool_env(tmp_path, with_script=True)
    assert hooks.hook_launcher() == (f'"{script.as_posix()}"', "the code-atlas uv tool install")
    assert hooks.hook_launcher("C:\\py\\python.exe") == ('"C:/py/python.exe" -m code_atlas.hooks', "--python")

    # A tool installed before atlas-hook existed: its interpreter, and a nudge to reinstall.
    script.unlink()
    launcher, why = hooks.hook_launcher()
    assert launcher == f'"{python.as_posix()}" -m code_atlas.hooks'
    assert "reinstall" in why

    python.unlink()
    launcher, why = hooks.hook_launcher()
    assert launcher == f'"{(dev / ".venv" / "Scripts" / "python.exe").as_posix()}" -m code_atlas.hooks'
    assert why.endswith("development venv"), "no tool: the venv is used, and named as one"


def test_install_refuses_a_settings_file_it_cannot_read(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        hooks.write_settings(path, hooks.hook_config(strict=False))
    assert path.read_text(encoding="utf-8") == "[1, 2]"
