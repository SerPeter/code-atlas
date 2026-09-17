"""Claude Code hooks that route structure questions to the graph instead of grep.

Agents reach for Grep, Read and subagents before an MCP tool: the built-ins are always
loaded, MCP tools are often deferred behind a search, and a tool description is not read
until the tool already is. So the routing has to arrive from outside the tool list:

* ``session-start`` / ``subagent-start`` -- a compact "which tool for which question" card,
  only when this project is actually indexed. SubagentStart matters on its own: the built-in
  Explore and Plan agents skip CLAUDE.md, so a rule written there never reaches them. At session
  start, a fail-fast health notice too (graph down, not indexed, schema skew, Valkey down), shown
  to the user and handed to the agent to relay.
* ``post-tool`` -- after a Grep (or a grep/rg run through Bash) for something identifier-shaped
  that the graph knows, one line saying what a single graph call would have returned. The
  grep output is never replaced: grep is exhaustive over literal text and the graph is not,
  so substituting one for the other would turn an index gap into a confident wrong answer.
* ``pre-tool --strict`` -- the opt-in block. Advisory context is routinely walked past
  mid-task, so strict mode denies the first symbol grep the graph can answer, and the first
  exploration subagent or workflow, for each agent that has not used code-atlas yet. Each
  kind blocks at most once per agent, so it can never strand one.

Every path fails open: an error, an unreachable graph, or an unindexed project prints
nothing and the tool call proceeds. This module keeps stdlib-only imports at the top because
it runs on every matched tool call -- ``code_atlas.graph.client`` alone takes ~4s to import.

Run as ``atlas-hook <event> [--strict]`` (or ``python -m code_atlas.hooks``); ``atlas hooks install``
writes it.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import sys
import tempfile
import threading
import time
import warnings
from contextlib import closing, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from code_atlas.health_verdicts import CheckResult, CheckStatus

if TYPE_CHECKING:
    from code_atlas.settings import AtlasSettings

MCP_SERVER = "code-atlas"
_MCP_PREFIX = f"mcp__{MCP_SERVER}__"

# One hint per symbol, and a ceiling per agent: the card is the lesson, the hints are the
# reminder, and a reminder on every grep of a long session is noise the agent learns to skip.
_MAX_HINTS_PER_AGENT = 5
_MAX_SYMBOLS_PER_SEARCH = 3
_STATE_TTL_S = 2 * 24 * 3600
_CODE_LABELS = ("Callable", "TypeDef", "Module", "Value")
_INBOUND = ("CALLS", "USES_TYPE", "INHERITS", "IMPLEMENTS", "OVERRIDES", "REFERENCES", "IMPORTS")
# Subagent types whose job is finding things -- the ones a graph call can replace. A reviewer
# or executor is delegated for its work, not its search, and is never blocked.
_EXPLORATION_AGENTS = frozenset({"", "Explore", "explore", "general-purpose", "Plan", "claude"})

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# Wrappers an agent puts around a name it is looking for: `\bFoo\b`, `def foo`, `class Foo\(`.
_DECL_PREFIX = re.compile(
    r"^(?:\^\s*|\\s[*+]\s*)*(?:(?:async\s+|export\s+|pub\s+)*(?:def|class|function|fn|func|interface|struct|type|enum)"
    r"(?:\\s[+*]|\s+))?"
)
_SEARCH_COMMANDS = frozenset({"grep", "egrep", "fgrep", "rg", "ag", "ack"})
# Flags whose value is the next token, so the pattern is not mistaken for it (`rg -t py Foo`).
_VALUE_FLAGS = frozenset(
    {"-t", "--type", "-T", "--type-not", "-g", "--glob", "-A", "-B", "-C", "-m", "--max-count", "-f", "--file"}
    | {"--include", "--exclude", "--exclude-dir", "-d", "--max-depth", "-j", "--threads", "-M", "--color"}
)


# ---------------------------------------------------------------------------
# Search-shape parsing (pure)
# ---------------------------------------------------------------------------


def symbols_in_pattern(pattern: str) -> list[str]:
    """Names a search pattern is looking for, or [] when it is not a symbol lookup.

    Deliberately narrow. `TODO`, `error:` or `\\d+ms` are text searches the graph cannot
    answer; a false negative costs one missed hint, a false positive costs a wrong block.
    """
    out: list[str] = []
    for raw in pattern.split("|"):
        part = raw.strip().replace("\\b", "").replace("\\<", "").replace("\\>", "")
        declared = _DECL_PREFIX.match(part)
        has_decl = declared is not None and bool(declared.group(0).strip(" ^"))
        part = _DECL_PREFIX.sub("", part, count=1)
        part = re.sub(r"(?:\\s\*)?\\?\($|\s*\($|\$$", "", part)
        if "." in part:  # `self.graph.ping` / `obj\.method`: the last segment is the name
            head, _, part = part.replace("\\.", ".").rpartition(".")
            has_decl = has_decl or bool(_IDENT.fullmatch(head.rsplit(".", 1)[-1]))
        if not _IDENT.fullmatch(part):
            return []
        # `TODO`, `import`, `error` are words, not names: a bare word needs `def`/`class` in front
        # of it to count. snake_case and camelCase/PascalCase stand on their own.
        named = "_" in part.strip("_") or (not part.isupper() and not part.islower())
        if not (named or has_decl):
            return []
        out.append(part)
    return list(dict.fromkeys(out))[:_MAX_SYMBOLS_PER_SEARCH]


def search_pattern(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """The pattern a Grep call or a grep/rg Bash command searches for, else None."""
    if tool_name == "Grep":
        pat = tool_input.get("pattern")
        return pat if isinstance(pat, str) else None
    if tool_name != "Bash":
        return None
    command = tool_input.get("command")
    if not isinstance(command, str):
        return None
    for segment in re.split(r"\|\||&&|[|;\n]", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            continue
        if not tokens:
            continue
        name, rest = Path(tokens[0]).name.removesuffix(".exe"), tokens[1:]
        if name == "rtk" and rest:  # the rtk proxy prefixes rewritten commands
            name, rest = rest[0], rest[1:]
        if name == "git" and rest[:1] == ["grep"]:
            name, rest = "grep", rest[1:]
        if name in _SEARCH_COMMANDS:
            return _pattern_argument(rest)
    return None


def _pattern_argument(args: list[str]) -> str | None:
    """The pattern among a grep-family command's arguments: `-e PAT`, else the first positional."""
    skip = False
    for i, tok in enumerate(args):
        if skip:
            skip = False
        elif tok in ("-e", "--regexp") and i + 1 < len(args):
            return args[i + 1]
        elif tok in _VALUE_FLAGS:
            skip = True
        elif not tok.startswith("-"):
            return tok
    return None


# ---------------------------------------------------------------------------
# Session state -- a JSON file per session in the temp dir
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    return Path(tempfile.gettempdir()) / "code-atlas-hooks"


def _state_path(session_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id or "nosession")
    return _state_dir() / f"{safe}.json"


def _load_state(session_id: str) -> dict[str, Any]:
    try:
        return json.loads(_state_path(session_id).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}


def _save_state(session_id: str, state: dict[str, Any]) -> None:
    path = _state_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    tmp.replace(path)  # parallel tool calls race; last writer wins, worst case a repeated hint


def _prune_state_dir() -> None:
    cutoff = time.time() - _STATE_TTL_S
    try:
        for p in _state_dir().glob("*.json"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except OSError:
        pass


def _agent(state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    key = payload.get("agent_id") or "main"
    return state.setdefault("agents", {}).setdefault(key, {"oriented": False, "blocked": [], "hinted": []})


# ---------------------------------------------------------------------------
# Graph access -- resolved once per session, then raw driver queries
# ---------------------------------------------------------------------------


def mcp_unavailable(dirs: list[Path]) -> str | None:
    """Why this session cannot call code-atlas's MCP tools, or None when it can.

    Advertising, or blocking in favour of, tools the agent does not have is worse than saying
    nothing. *dirs* is most-specific first (cwd, repo root, main checkout): Claude Code keeps a
    worktree's MCP settings under the main repository's entry. A server configured somewhere
    this cannot see (a plugin, managed settings, ``--mcp-config``) reads as unavailable -- silence
    is the safe miss -- so ``ATLAS_HOOKS_MCP=1`` asserts availability outright.
    """
    if os.environ.get("ATLAS_HOOKS_MCP") == "1":
        return None
    try:
        data = json.loads((Path.home() / ".claude.json").read_text(encoding="utf-8"))
    except OSError, ValueError:
        data = {}
    projects = {}
    for key, value in (data.get("projects") or {}).items():
        with suppress(OSError, ValueError):
            projects[Path(key).resolve()] = value
    configured = MCP_SERVER in (data.get("mcpServers") or {})
    for d in dirs:
        proj = projects.get(d.resolve())
        if proj is not None:
            if MCP_SERVER in (proj.get("disabledMcpServers") or []):
                return "mcp-disabled"
            configured = configured or MCP_SERVER in (proj.get("mcpServers") or {})
        mcp_json = d / ".mcp.json"
        if not configured and mcp_json.is_file():
            with suppress(OSError, ValueError):
                configured = MCP_SERVER in (json.loads(mcp_json.read_text(encoding="utf-8")).get("mcpServers") or {})
    return None if configured else "mcp-not-configured"


def resolve_context(cwd: str) -> dict[str, Any]:
    """Which graph answers for *cwd*, and whether it holds anything.

    Imports settings (~0.4s), so it runs once per session and the result is cached. A linked
    worktree whose own project was never indexed falls back to the main checkout's project:
    agents mostly work in worktrees, and a slightly stale answer beats none -- the hint says so.
    """
    from code_atlas.settings import AtlasSettings, derive_project_name, find_git_root

    # No .env is read, anywhere in Atlas: providing the environment is the caller's job (a shell
    # profile, direnv, Claude Code's settings `env`), and a tool loading a repo's .env loads what it should not.
    root = find_git_root(Path(cwd))
    if root is None:
        return {"available": False, "reason": "not-a-repo"}
    main_root = _main_checkout(root)
    reason = mcp_unavailable([Path(cwd), root, *([main_root] if main_root else [])])
    if reason is not None:
        return {"available": False, "reason": reason}
    # Name -> root, in preference order. A worktree is named `<dir>@<branch>` after its own
    # directory, so the fallback name has to be derived from the main checkout's root.
    candidates = {derive_project_name(r): r for r in [root, main_root] if r is not None}
    try:
        settings = AtlasSettings(project_root=root)
    except Exception as exc:  # code-atlas is configured here and its config does not load: say so
        failed = CheckResult("config", CheckStatus.FAIL, "Configuration does not load", detail=str(exc))
        return {"available": False, "reason": "config", "problems": [format_problem(failed)]}
    backend = settings.backend
    targets: dict[str, tuple[str, int]] = {}
    if backend.graph_choice in ("memgraph", "auto"):
        targets["memgraph"] = (settings.memgraph.host, settings.memgraph.port)
    if backend.queue_choice in ("valkey", "auto"):
        targets["valkey"] = (settings.redis.host, settings.redis.port)

    # Everything at once. The embedding check is the slow one when it is not cached (importing
    # litellm alone is ~3.4s), so it starts first and the backend probes run while it works.
    embeddings: list[CheckResult] = []
    worker = threading.Thread(target=lambda: embeddings.append(check_embeddings_cached(settings)), daemon=True)
    worker.start()
    up = dict(zip(targets, ports_open(list(targets.values()), settings.health.connect_timeout_s), strict=True))
    ctx = _probe_graph(settings, candidates, memgraph_up=up.get("memgraph", False))
    worker.join()

    results = [*_backend_results(ctx, settings, up), *embeddings]
    ctx["problems"] = [format_problem(r) for r in results if r.status != CheckStatus.OK]
    return ctx


def _backend_results(ctx: dict[str, Any], settings: AtlasSettings, up: dict[str, bool]) -> list[CheckResult]:
    """The same verdicts ``atlas health`` reaches, from what the hook's own probes found.

    Only *how* the backends were reached differs -- a bare TCP connect and the hook's own
    queries against ``atlas health``'s client pings. What that means, and the fix, is
    :mod:`code_atlas.health_verdicts` in both. One check is the hook's alone: whether *this*
    repository is indexed, where ``atlas health`` asks only whether any project is.
    """
    from code_atlas.health_verdicts import memgraph_verdict, schema_verdict, valkey_verdict
    from code_atlas.schema import SCHEMA_VERSION

    results: list[CheckResult] = []
    graph_choice, queue_choice = settings.backend.graph_choice, settings.backend.queue_choice
    mg, redis = settings.memgraph, settings.redis
    reason = ctx.get("reason")
    embedded = ctx.get("backend") == "sqlite"

    if reason in ("unreachable", "unreadable"):
        results.append(
            memgraph_verdict(
                f"{mg.host}:{mg.port}", reachable=False, embedded=embedded, detail=ctx.get("graph_error", "")
            )
        )
    elif ctx.get("backend"):
        results.append(
            memgraph_verdict(f"{mg.host}:{mg.port}", reachable=True, embedded=embedded, chosen=graph_choice == "sqlite")
        )
    if "schema" in ctx:
        results.append(schema_verdict(ctx["schema"], SCHEMA_VERSION))
    if reason in ("not-indexed", "no-index"):
        results.append(
            CheckResult(
                "index",
                CheckStatus.FAIL,
                f"This repository is not indexed (project {ctx['project']})",
                suggestion="Run 'atlas index' in the repository.",
            )
        )

    addr = f"{redis.host}:{redis.port}"
    if queue_choice == "valkey":
        results.append(valkey_verdict(addr, reachable=up.get("valkey", False)))
    elif queue_choice == "auto":
        # Undeclared: a Valkey that does not answer is the one `atlas` falls back from.
        results.append(valkey_verdict(addr, reachable=True, embedded=not up.get("valkey", False)))
    return results


def format_problem(result: CheckResult) -> str:
    """One line per non-OK check, in the terms ``atlas health`` prints."""
    line = f"{result.name} {result.status.upper()}: {result.message}"
    if result.detail:
        line += f" ({result.detail[:300]})"
    if result.suggestion:
        line += f". Fix: {result.suggestion}"
    return line


_EMBEDDINGS_OK_TTL_S = 24 * 3600


def embeddings_fingerprint(settings: AtlasSettings) -> str:
    """What an embedding check's success depends on: provider, model, endpoint, and the credentials.

    Credentials enter as a hash with everything else, never stored: litellm reads the provider's
    key from the environment (``OPENAI_API_KEY``, ``GEMINI_API_KEY``, ...), so any variable that
    looks like a key, a token or an API base is part of it. Rotating a key invalidates the cache.
    """
    import hashlib

    emb = settings.embeddings
    env = sorted((k, v) for k, v in os.environ.items() if "API_KEY" in k or "API_BASE" in k or k.endswith("_TOKEN"))
    raw = json.dumps([emb.provider, emb.model, emb.base_url, emb.dimension, env])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def check_embeddings_cached(settings: AtlasSettings) -> CheckResult:
    """``atlas health``'s embedding check -- one real embedding call -- skipped after a recent success.

    Only a success is cached, keyed by :func:`embeddings_fingerprint` for a day: config and
    credentials rarely change, and the check costs ~4-5s (litellm's import, then a billed
    round trip). A failure is never cached, so a broken token is re-checked, and reported, on
    every session start until it is fixed.
    """
    from code_atlas.health_verdicts import embeddings_verdict

    emb = settings.embeddings
    if not emb.enabled:
        return embeddings_verdict(emb, ok=True)
    cache = _state_dir() / "embeddings-ok.json"
    key, now = embeddings_fingerprint(settings), time.time()
    try:
        known = {
            k: t for k, t in json.loads(cache.read_text(encoding="utf-8")).items() if now - t < _EMBEDDINGS_OK_TTL_S
        }
    except OSError, ValueError, AttributeError:
        known = {}
    if key in known:
        return embeddings_verdict(emb, ok=True)
    try:
        ok, detail = _embedding_call_succeeds(settings), ""
    except TimeoutError:
        return embeddings_verdict(emb, ok=False, timed_out=True)
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    if ok:
        known[key] = now
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(known), encoding="utf-8")
        tmp.replace(cache)
    return embeddings_verdict(emb, ok=ok, detail=detail)


def _embedding_call_succeeds(settings: AtlasSettings) -> bool:
    import asyncio

    from loguru import logger

    logger.remove()  # the embedding client logs to stderr, which is not the hook's output
    from code_atlas.search.embeddings import EmbedClient
    from code_atlas.search.ratelimit import unpaced

    # Unpaced, as `atlas health` builds it: a probe must answer "does the provider accept this
    # config" now, not wait on a drained bucket and report a busy provider as a broken one.
    client = EmbedClient(settings.embeddings, limiter=unpaced())
    return asyncio.run(asyncio.wait_for(client.health_check(), settings.health.check_timeout_s))


def ports_open(targets: list[tuple[str, int]], timeout_s: float) -> list[bool]:
    """Bare TCP connects to every target concurrently, each bounded by ``[health] connect_timeout_s``.

    Before any driver is involved, and in parallel on two axes. Across targets, so the check costs
    the slowest backend rather than the sum. And across addresses: on Windows a refused connection
    to `localhost` is not instant, and trying IPv6 then IPv4 in turn doubled the wait. A target is
    settled by its first successful address, so an IPv4-only Memgraph does not wait out ``::1``.

    Threads, not asyncio: importing asyncio costs ~110 ms, paid by every session start.
    """
    import socket

    up = [False] * len(targets)
    attempts: list[tuple[int, socket.AddressFamily, Any]] = []
    for i, (host, port) in enumerate(targets):
        with suppress(OSError):
            attempts += [(i, info[0], info[4]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]
    pending = [sum(1 for a in attempts if a[0] == i) for i in range(len(targets))]
    settled = threading.Condition()

    def attempt(i: int, family: socket.AddressFamily, address: Any) -> None:
        ok = False
        with suppress(OSError), socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout_s)
            sock.connect(address)
            ok = True
        with settled:
            up[i] = up[i] or ok
            pending[i] -= 1
            settled.notify_all()

    for args in attempts:
        # Daemon threads: a connect still hanging when the answer is already known must not hold the hook open.
        threading.Thread(target=attempt, args=args, daemon=True).start()
    with settled:
        settled.wait_for(lambda: all(ok or left == 0 for ok, left in zip(up, pending, strict=True)), timeout_s)
        return list(up)


def _probe_graph(settings: AtlasSettings, candidates: dict[str, Path], *, memgraph_up: bool) -> dict[str, Any]:
    """The first backend -- in the order ``atlas`` itself would pick -- holding a candidate project."""
    ctx: dict[str, Any] = {"available": False, "project": next(iter(candidates))}
    choice = settings.backend.graph_choice
    memgraph_error = None
    if choice in ("memgraph", "auto"):
        mg = settings.memgraph
        ctx.update(
            backend="memgraph",
            uri=f"bolt://{mg.host}:{mg.port}",
            user=mg.username,
            auth=bool(mg.password),
            timeout_s=settings.health.connect_timeout_s,
        )
        try:
            if not memgraph_up:
                msg = f"no answer within {ctx['timeout_s']}s"
                raise ConnectionError(msg)  # noqa: TRY301 -- the same path as a driver failure
            return _pick_project(ctx, _run(ctx, _count_query(ctx), projects=list(candidates)))
        except Exception as exc:
            if choice == "memgraph":
                return {**ctx, "reason": "unreachable", "graph_error": str(exc)}
            memgraph_error = str(exc)
    for name, candidate_root in candidates.items():
        sqlite_path = candidate_root / settings.backend.sqlite_data_dir / "graph.sqlite3"
        if not sqlite_path.exists():
            continue
        ctx.update(backend="sqlite", path=str(sqlite_path))
        try:
            return _pick_project(ctx, _run(ctx, _count_query(ctx), projects=[name]))
        except Exception as exc:
            return {**ctx, "reason": "unreadable", "graph_error": str(exc)}
    # Under `auto`, a Memgraph that did not answer and no embedded index to fall back to is an
    # outage, not a repository nobody indexed.
    if memgraph_error is not None:
        return {**ctx, "backend": "memgraph", "reason": "unreachable", "graph_error": memgraph_error}
    return {**ctx, "backend": None, "reason": "no-index"}


def _main_checkout(root: Path) -> Path | None:
    """The main working tree behind a linked worktree (via its ``commondir``), else None."""
    from code_atlas.settings import resolve_git_dir

    git_dir = resolve_git_dir(root)
    if git_dir is None or not (git_dir / "commondir").is_file():
        return None
    try:
        common = (git_dir / "commondir").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return (git_dir / common).resolve().parent


def _pick_project(ctx: dict[str, Any], counts: dict[str, int]) -> dict[str, Any]:
    ctx = {**ctx, "schema": _run(ctx, _schema_query(ctx))}
    for name, n in sorted(counts.items(), key=lambda kv: kv[0] != ctx["project"]):
        if n:
            return {**ctx, "available": True, "graph_project": name, "entities": n}
    return {**ctx, "reason": "not-indexed"}


def _schema_query(ctx: dict[str, Any]) -> str:
    if ctx["backend"] == "memgraph":
        return "MATCH (sv:SchemaVersion) RETURN max(sv.version) AS version"
    return "SELECT CAST(value AS INTEGER) FROM meta WHERE key = 'schema_version'"


def _count_query(ctx: dict[str, Any]) -> str:
    if ctx["backend"] == "memgraph":
        return "UNWIND $projects AS p MATCH (n:Entity {project_name: p}) RETURN p AS project, count(n) AS n"
    marks = ",".join(f"'{label}'" for label in _CODE_LABELS)
    return f"SELECT project_name, count(*) FROM nodes WHERE labels IN ({marks}) AND project_name IN ({{ps}}) GROUP BY 1"


def _lookup_query(ctx: dict[str, Any]) -> str:
    rels = "|".join(_INBOUND)
    if ctx["backend"] == "memgraph":
        return (
            "UNWIND $names AS nm MATCH (n:Entity {project_name: $project, name: nm}) "
            f"OPTIONAL MATCH (c)-[r:{rels}]->(n) "
            "WITH n, count(DISTINCT c) AS refs, count(DISTINCT CASE WHEN type(r) = 'CALLS' THEN c END) AS callers "
            "RETURN n.name AS name, n.uid AS uid, labels(n) AS labels, n.file_path AS file, "
            "n.line_start AS line, refs, callers ORDER BY refs DESC LIMIT 12"
        )
    marks = ",".join(f"'{label}'" for label in _CODE_LABELS)
    rel_list = ",".join(f"'{r}'" for r in _INBOUND)
    return (
        "SELECT name, uid, labels, file_path, json_extract(props_json, '$.line_start'), "
        f"(SELECT count(DISTINCT from_uid) FROM edges WHERE to_uid = nodes.uid AND rel_type IN ({rel_list})), "
        "(SELECT count(DISTINCT from_uid) FROM edges WHERE to_uid = nodes.uid AND rel_type = 'CALLS') "
        f"FROM nodes WHERE labels IN ({marks}) AND project_name = ? AND name IN ({{ns}}) ORDER BY 6 DESC LIMIT 12"
    )


def _run(ctx: dict[str, Any], query: str, **params: Any) -> Any:
    if ctx["backend"] == "memgraph":
        return _run_memgraph(ctx, query, params)
    return _run_sqlite(ctx, query, params)


def _run_memgraph(ctx: dict[str, Any], query: str, params: dict[str, Any]) -> Any:
    from neo4j import GraphDatabase

    auth = None
    if ctx.get("user") or ctx.get("auth"):
        from code_atlas.settings import AtlasSettings

        # The password is never written to the temp-dir cache; re-read it when one is set.
        auth = (ctx.get("user") or "", AtlasSettings().memgraph.password if ctx.get("auth") else "")
    with (
        GraphDatabase.driver(ctx["uri"], auth=auth, connection_timeout=ctx["timeout_s"]) as driver,
        driver.session() as session,
    ):
        rows = session.run(query, params).data()  # ty: ignore[invalid-argument-type]
    if "projects" in params:
        return {r["project"]: r["n"] for r in rows}
    if not params:  # the schema version
        return rows[0]["version"] if rows else None
    return rows


def _run_sqlite(ctx: dict[str, Any], query: str, params: dict[str, Any]) -> Any:
    uri = Path(ctx["path"]).as_uri() + "?mode=ro"
    # closing(), not the connection's own `with`: that one commits and leaves the handle open.
    with closing(sqlite3.connect(uri, uri=True, timeout=1.0)) as conn:
        if not params:  # the schema version
            row = conn.execute(query).fetchone()
            return row[0] if row else None
        if "projects" in params:
            ps = params["projects"]
            rows = conn.execute(query.format(ps=",".join("?" * len(ps))), ps).fetchall()
            return dict(rows)
        ns = params["names"]
        rows = conn.execute(query.format(ns=",".join("?" * len(ns))), [params["project"], *ns]).fetchall()
    keys = ("name", "uid", "labels", "file", "line", "refs", "callers")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def _context(payload: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    ctx = state.get("ctx")
    if ctx is None:
        try:
            ctx = resolve_context(payload.get("cwd") or str(Path.cwd()))
        except Exception as exc:
            ctx = {"available": False, "reason": f"error: {type(exc).__name__}"}
        state["ctx"] = ctx
    return ctx


def lookup(ctx: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """Code entities named *names* in the session's project, most-referenced first."""
    rows = _run(ctx, _lookup_query(ctx), names=names, project=ctx["graph_project"])
    for row in rows:
        labels = row["labels"]
        if isinstance(labels, str):
            labels = [labels]
        row["label"] = next((lb for lb in labels if lb in _CODE_LABELS), labels[0] if labels else "?")
    return rows


# ---------------------------------------------------------------------------
# Output text
# ---------------------------------------------------------------------------


def routing_card(ctx: dict[str, Any]) -> str:
    where = ctx["graph_project"]
    if where != ctx["project"]:
        where += " (main checkout's index -- may lag this branch)"
    return (
        f"code-atlas has this repo indexed: {where}, {ctx['entities']:,} code entities. "
        f"Its MCP tools ({_MCP_PREFIX}*, load via ToolSearch if deferred) answer structure questions in one "
        "call -- use them before Grep/Read, and before delegating a lookup to a subagent:\n"
        '- where is X defined -> get_node("X")\n'
        "- who calls / uses X -> get_context(uid)   - what breaks if X changes -> blast_radius(uid)\n"
        '- what is in a module -> summarize_module("path")   - how does Y work -> hybrid_search("Y")\n'
        "Grep stays right for literal text (messages, config values) and exhaustive rename sweeps."
    )


def _loc(row: dict[str, Any]) -> str:
    return f"{row['file']}:{row['line']}" if row.get("line") else str(row.get("file") or "?")


def hint_line(rows: list[dict[str, Any]]) -> str:
    """One line per searched name: what a graph call returns that grep output does not."""
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(row["name"], []).append(row)
    parts = []
    for name, defs in by_name.items():
        if len(defs) == 1:
            parts.append(f"{name} = {_describe(defs[0])}")
        else:
            parts.append(f"{name} = {len(defs)} definitions: " + "; ".join(_describe(d) for d in defs[:3]))
    first = rows[0]
    return (
        f"code-atlas already knew: {' | '.join(parts)}. "
        f'Next time get_node("{first["name"]}") answers in one call, and '
        f'get_context("{first["uid"]}") names every dependent -- which grep cannot.'
    )


def _describe(row: dict[str, Any]) -> str:
    uses = f"{row['callers']} callers" if row["callers"] else f"{row['refs']} dependents"
    return f"{row['label']} {_loc(row)} ({uses})"


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


# The stream Claude Code parses as the hook's JSON. main() takes it and points sys.stdout at
# stderr: a library that prints -- litellm writes a "Give Feedback / Get Help" banner to stdout
# on a failed call -- would otherwise prepend text to the JSON and the whole result is discarded.
_hook_output: Any = None


def _emit(event: str, system_message: str | None = None, **fields: Any) -> None:
    out: dict[str, Any] = {"hookSpecificOutput": {"hookEventName": event, **fields}}
    if system_message:
        out["systemMessage"] = system_message
    (_hook_output or sys.stdout).write(json.dumps(out))


def health_notice(problems: list[str]) -> str:
    return (
        "code-atlas is not healthy in this session:\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\nTell the user about this at the start of your first reply, including the fix. Until it is fixed, "
        "code-atlas answers may be missing or stale: use Grep/Read, and do not treat an empty graph result "
        "as proof that something does not exist."
    )


def on_start(payload: dict[str, Any], event: str) -> None:
    state = _load_state(payload.get("session_id", ""))
    if event == "SessionStart":
        _prune_state_dir()
        state.pop("ctx", None)  # a new or resumed session re-checks; the index may have changed
    ctx = _context(payload, state)
    _save_state(payload.get("session_id", ""), state)
    parts = [routing_card(ctx)] if ctx.get("available") else []
    problems = ctx.get("problems") or []
    # The notice goes to the session that can relay it -- a subagent reports to its parent, not the user.
    if problems and event == "SessionStart":
        parts.append(health_notice(problems))
        _emit(event, system_message="code-atlas: " + " | ".join(problems), additionalContext="\n\n".join(parts))
    elif parts:
        _emit(event, additionalContext="\n\n".join(parts))


def on_post_tool(payload: dict[str, Any]) -> None:
    tool = payload.get("tool_name", "")
    session = payload.get("session_id", "")
    state = _load_state(session)
    agent = _agent(state, payload)
    if tool.startswith(_MCP_PREFIX):
        if not agent["oriented"]:
            agent["oriented"] = True
            _save_state(session, state)
        return
    if agent["oriented"]:  # it already knows the graph; a grep now is a deliberate text search
        return
    pattern = search_pattern(tool, payload.get("tool_input") or {})
    names = [n for n in symbols_in_pattern(pattern or "") if n not in agent["hinted"]]
    if not names or len(agent["hinted"]) >= _MAX_HINTS_PER_AGENT:
        return
    ctx = _context(payload, state)
    rows = lookup(ctx, names) if ctx.get("available") else []
    agent["hinted"].extend(names)
    _save_state(session, state)
    if rows:  # a miss says nothing: absence from an index is not absence from the code
        _emit("PostToolUse", additionalContext=hint_line(rows))


def on_pre_tool(payload: dict[str, Any], *, strict: bool) -> None:
    if not strict:
        return
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    session = payload.get("session_id", "")
    state = _load_state(session)
    agent = _agent(state, payload)
    if agent["oriented"]:
        return

    if tool in ("Agent", "Task", "Workflow"):
        kind = "delegate"
        if tool != "Workflow" and str(tool_input.get("subagent_type") or "") not in _EXPLORATION_AGENTS:
            return
        if kind in agent["blocked"] or not _context(payload, state).get("available"):
            _save_state(session, state)
            return
        reason = (
            "code-atlas strict mode: this repo is indexed, and most exploration a subagent or workflow would do "
            "is one graph call -- get_node / get_context / blast_radius / summarize_module / hybrid_search "
            f"({_MCP_PREFIX}*). Try that first. If you still need to delegate, re-issue this call: it will be "
            "allowed. This block fires once."
        )
    else:
        kind = "search"
        names = symbols_in_pattern(search_pattern(tool, tool_input) or "")
        if not names or kind in agent["blocked"]:
            return
        ctx = _context(payload, state)
        rows = lookup(ctx, names) if ctx.get("available") else []
        if not rows:  # only block a search the graph can actually answer
            _save_state(session, state)
            return
        reason = (
            "code-atlas strict mode: the graph already has this. "
            + hint_line(rows)
            + " Re-issue the search if you need the literal text matches: it will be allowed. This block fires once."
        )
    agent["blocked"].append(kind)
    _save_state(session, state)
    _emit("PreToolUse", permissionDecision="deny", permissionDecisionReason=reason)


# ---------------------------------------------------------------------------
# Installation -- merging into a Claude Code settings file
# ---------------------------------------------------------------------------

_MODULE = "code_atlas.hooks"


_SCRIPT = "atlas-hook"
"""The console script that runs :func:`main` -- see ``[project.scripts]``."""

_DISTRIBUTION = "code-atlas-mcp"


def _quoted(path: str | Path) -> str:
    # Forward slashes, because Claude Code runs hooks through Git Bash on Windows, which eats
    # backslashes; quoted, because a user profile path can contain spaces.
    return '"' + str(path).replace("\\", "/") + '"'


def _module_launcher(python: str) -> str:
    return f"{_quoted(python)} -m {_MODULE}"


def hook_launcher(python: str | None = None) -> tuple[str, str]:
    """The command prefix the hooks run with, and why that one.

    Pinned by absolute path, never a bare ``atlas-hook``: a checkout's development venv installs
    its own copy of the script, and it comes first on PATH whenever Claude Code starts from an
    activated venv -- which ties every session to a venv that ``uv sync`` rewrites and whose
    executables a running hook locks. The uv tool install is the one meant to serve other
    sessions, so it wins when it exists. Order:

    1. ``--python`` -- ``"<python>" -m code_atlas.hooks`` under that interpreter.
    2. The uv tool install's ``atlas-hook`` script, read from its receipt.
    3. The uv tool install's interpreter, for an install that predates ``atlas-hook``.
    4. The current interpreter.
    """
    if python:
        return _module_launcher(python), "--python"
    env = uv_tool_env()
    if env is not None:
        script = _tool_entrypoint(env, _SCRIPT)
        if script is not None:
            return _quoted(script), "the code-atlas uv tool install"
        for candidate in (env / "Scripts" / "python.exe", env / "bin" / "python"):
            if candidate.is_file():
                return (
                    _module_launcher(str(candidate)),
                    f"the code-atlas uv tool install, which predates {_SCRIPT} -- reinstall it for the short command",
                )
    if (Path(sys.prefix).parent / "pyproject.toml").is_file():
        return _module_launcher(sys.executable), "the current interpreter -- a project's development venv"
    return _module_launcher(sys.executable), "the current interpreter"


def uv_tool_env() -> Path | None:
    """The ``code-atlas-mcp`` uv tool environment, if one is installed."""
    import shutil
    import subprocess

    tool_dir = os.environ.get("UV_TOOL_DIR")
    if not tool_dir and (uv := shutil.which("uv")):
        with suppress(OSError, subprocess.SubprocessError):
            done = subprocess.run([uv, "tool", "dir"], capture_output=True, text=True, timeout=10, check=True)
            tool_dir = done.stdout.strip()
    if not tool_dir:
        return None
    env = Path(tool_dir) / _DISTRIBUTION
    return env if env.is_dir() else None


def _tool_entrypoint(env: Path, name: str) -> Path | None:
    """Where uv installed the tool's *name* script, per the environment's ``uv-receipt.toml``.

    The receipt rather than a guess at uv's bin directory: it names the exact file uv wrote for
    this tool, so a same-named script from some other install on PATH can never be picked up.
    """
    import tomllib

    try:
        receipt = tomllib.loads((env / "uv-receipt.toml").read_text(encoding="utf-8"))
    except OSError, tomllib.TOMLDecodeError:
        return None
    for entry in receipt.get("tool", {}).get("entrypoints", []):
        if entry.get("name") == name and (path := entry.get("install-path")) and Path(path).is_file():
            return Path(path)
    return None


def hook_config(*, strict: bool, launcher: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """The ``hooks`` entries to merge, keyed by event. *launcher* comes from :func:`hook_launcher`."""
    base = launcher or _module_launcher(sys.executable)

    def entry(matcher: str, args: str) -> dict[str, Any]:
        # A ceiling, not a budget: headroom for `[health] connect_timeout_s` at its maximum on a
        # session start that probes Memgraph and Valkey over two address families each.
        return {"matcher": matcher, "hooks": [{"type": "command", "command": f"{base} {args}", "timeout": 60}]}

    config = {
        "SessionStart": [entry("startup|resume|clear|compact", "session-start")],
        "SubagentStart": [entry("*", "subagent-start")],
        "PostToolUse": [entry(f"Grep|Bash|{_MCP_PREFIX}.*", "post-tool")],
    }
    if strict:
        config["PreToolUse"] = [entry("Grep|Bash|Agent|Task|Workflow", "pre-tool --strict")]
    return config


_OUR_COMMAND = re.compile(rf"{re.escape(_MODULE)}|[\\/\"]{re.escape(_SCRIPT)}(?:\.exe)?\"?\s")


def _is_ours(group: dict[str, Any]) -> bool:
    # Both shapes: `"<python>" -m code_atlas.hooks` (every install before atlas-hook, and
    # --python) and `"<bin>/atlas-hook.exe"`, so re-installing replaces rather than duplicates.
    return any(_OUR_COMMAND.search(str(h.get("command", ""))) for h in group.get("hooks") or [])


def merge_settings(settings: dict[str, Any], config: dict[str, list[dict[str, Any]]] | None) -> dict[str, Any]:
    """*settings* with every code-atlas hook replaced by *config* (None removes them).

    Other hooks -- rtk, a cwd guard -- are left exactly where they were.
    """
    hooks = settings.setdefault("hooks", {})
    for event in list(hooks):
        kept = [g for g in hooks[event] or [] if not _is_ours(g)]
        if kept:
            hooks[event] = kept
        else:
            del hooks[event]
    for event, groups in (config or {}).items():
        hooks.setdefault(event, []).extend(groups)
    if not hooks:
        del settings["hooks"]
    return settings


def write_settings(path: Path, config: dict[str, list[dict[str, Any]]] | None) -> None:
    """Merge *config* into the settings file at *path*, backing the original up first.

    Refuses a file that is not a JSON object rather than overwriting it -- replacing a
    settings file you failed to parse destroys the user's configuration.
    """
    settings: dict[str, Any] = {}
    if path.exists():
        raw = path.read_text(encoding="utf-8-sig")
        loaded = json.loads(raw) if raw.strip() else {}
        if not isinstance(loaded, dict):
            msg = f"{path} is not a JSON object; not modifying it"
            raise ValueError(msg)
        settings = loaded
        path.with_name(path.name + ".atlas-bak").write_text(raw, encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merge_settings(settings, config), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    global _hook_output
    args = sys.argv[1:] if argv is None else argv
    if not args:
        sys.stderr.write("usage: atlas-hook {session-start|subagent-start|pre-tool|post-tool} [--strict]\n")
        return 2
    try:
        warnings.simplefilter("ignore")  # stderr from a hook is noise in the transcript, never context
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace") or "{}")
        if not isinstance(payload, dict):
            return 0
        sys.stdout.reconfigure(encoding="utf-8")  # ty: ignore[unresolved-attribute]
        _hook_output, sys.stdout = sys.stdout, sys.stderr
        event = args[0]
        if event == "session-start":
            on_start(payload, "SessionStart")
        elif event == "subagent-start":
            on_start(payload, "SubagentStart")
        elif event == "post-tool":
            on_post_tool(payload)
        elif event == "pre-tool":
            on_pre_tool(payload, strict="--strict" in args or os.environ.get("ATLAS_HOOKS_STRICT") == "1")
    except Exception as exc:  # fail open: a hook bug must never block or break a tool call
        if os.environ.get("ATLAS_HOOKS_DEBUG"):
            raise
        sys.stderr.write(f"code-atlas hook: {type(exc).__name__}: {exc}\n")
    finally:
        if _hook_output is not None:
            _hook_output.flush()
            sys.stdout, _hook_output = _hook_output, None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
