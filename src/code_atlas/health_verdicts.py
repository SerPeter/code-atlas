"""What a health probe's result means, and what to do about it.

Shared by ``atlas health`` (:mod:`code_atlas.server.health`) and the Claude Code session-start
hook (:mod:`code_atlas.hooks`), so the two can differ in how they reach a backend -- a client
ping against a bare TCP connect -- but never in what they conclude or which fix they name.

Standard library only, on purpose: the hook imports this on every session start, and the
modules ``atlas health`` probes with cost seconds to import.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_atlas.settings import EmbeddingSettings

MEMGRAPH_FIX = "docker compose up -d memgraph"
VALKEY_FIX = "docker compose up -d valkey"


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    """Result of a single health check."""

    name: str
    status: CheckStatus
    message: str
    detail: str = ""
    suggestion: str = ""


def memgraph_verdict(
    addr: str, *, reachable: bool, embedded: bool = False, chosen: bool = False, detail: str = ""
) -> CheckResult:
    """The graph backend, honestly naming which engine answered.

    *embedded* means the SQLite graph is the one in use; *chosen* that the config declared it.
    Only an embedded graph nobody chose warns: an undeclared backend falls back to SQLite
    whenever Memgraph is unreachable, so on a machine without Docker running that is the
    *default* outcome, and ADR-0015 calls SQLite explicitly not a parity replacement (ATL-112).
    A repository that declares SQLite asked for exactly what it got.
    """
    name = "memgraph"
    backend = "SQLite (embedded)" if embedded else f"Memgraph ({addr})"
    if not reachable:
        return CheckResult(name, CheckStatus.FAIL, f"Unreachable — {backend}", detail=detail, suggestion=MEMGRAPH_FIX)
    if embedded and not chosen:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Connected — {backend}, NOT Memgraph",
            detail=(
                "The embedded fallback is active. Community detection is unavailable and "
                "some analyses differ from Memgraph; see the README."
            ),
            suggestion=f"Start Memgraph ({addr}) with: {MEMGRAPH_FIX}",
        )
    return CheckResult(name, CheckStatus.OK, f"Connected — {backend}")


def valkey_verdict(
    addr: str,
    *,
    reachable: bool,
    embedded: bool = False,
    chosen: bool = False,
    postgres: str | None = None,
    detail: str = "",
) -> CheckResult:
    """The event queue, on the same terms as :func:`memgraph_verdict`.

    A queue that is down is a WARN, not a FAIL: every query still answers, but the index stops
    following edits -- which is the part worth saying. *postgres* is the address of a
    ``[backend.queue.postgres]`` queue (ADR-0056); Postgres is only ever declared, never a fallback.
    """
    name = "valkey"
    if postgres is not None:
        backend, waits_for, fix = (
            f"Postgres ({postgres})",
            "Postgres is reachable",
            "Check [backend.queue.postgres] and that the Postgres server is running.",
        )
    elif embedded:
        backend, waits_for, fix = "SQLite (embedded)", "the embedded queue is available", VALKEY_FIX
    else:
        backend, waits_for, fix = f"Valkey ({addr})", "Valkey is reachable", VALKEY_FIX
    if not reachable:
        stalled = f"auto-indexing disabled — file changes will NOT be indexed until {waits_for}"
        return CheckResult(
            name, CheckStatus.WARN, f"Unreachable — {backend} — {stalled}", detail=detail, suggestion=fix
        )
    if embedded and not chosen:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Connected — {backend}, NOT Valkey",
            detail="Blocking reads are emulated by polling; throughput is lower than the Valkey path.",
            suggestion=f"Start Valkey ({addr}) with: {VALKEY_FIX}",
        )
    return CheckResult(name, CheckStatus.OK, f"Connected — {backend}")


def schema_verdict(stored: int | None, current: int, *, error: str = "") -> CheckResult:
    """The graph's schema version against the one this install writes."""
    name = "schema"
    if error:
        return CheckResult(name, CheckStatus.FAIL, "Cannot read schema version", detail=error)
    if stored is None:
        return CheckResult(
            name,
            CheckStatus.WARN,
            "No schema version found",
            detail="Database may be empty.",
            suggestion="Run 'atlas index' to initialize the schema.",
        )
    if stored == current:
        return CheckResult(name, CheckStatus.OK, f"Version {stored} (current)")
    if stored < current:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"Version {stored} (expected {current})",
            detail="Schema is outdated.",
            suggestion="Run 'atlas index' to migrate the schema.",
        )
    return CheckResult(
        name,
        CheckStatus.FAIL,
        f"Version {stored} > code {current}",
        detail="Database schema is newer than the installed code.",
        suggestion="Update your Code Atlas installation.",
    )


def embeddings_verdict(
    settings: EmbeddingSettings, *, ok: bool | None, detail: str = "", timed_out: bool = False
) -> CheckResult:
    """The embedding provider: config, model and credentials, proven by one real embedding call.

    *ok* is None when there is no client to ask. A provider that does not answer is a WARN:
    graph and keyword search still work, vector search and new embeddings do not. *timed_out*
    is kept apart from a refusal: a first call in a fresh process can outlast the check while the
    key is fine, and telling someone to check a working key sends them the wrong way.
    """
    name = "embeddings"
    if not settings.enabled:
        return CheckResult(name, CheckStatus.OK, "Disabled (lightweight mode)")
    if settings.provider == "tei":
        info, fix = f"tei @ {settings.base_url}", "docker compose --profile tei up -d"
    elif settings.provider == "ollama":
        info, fix = f"ollama @ {settings.base_url}", f"Start Ollama and pull the model: ollama pull {settings.model}"
    else:
        info = f"{settings.provider} ({settings.model})"
        fix = "Check the provider API key (e.g. OPENAI_API_KEY) is set where atlas runs, and network connectivity."
    if ok is None:
        return CheckResult(name, CheckStatus.WARN, f"No client ({info})", suggestion="Check embedding settings.")
    if ok:
        return CheckResult(name, CheckStatus.OK, f"Responding ({info})")
    if timed_out:
        return CheckResult(
            name,
            CheckStatus.WARN,
            f"No answer within the check timeout ({info})",
            detail="Slow, not proven broken: config and credentials were not rejected.",
            suggestion="Raise [health] check_timeout_s if the provider is merely slow; otherwise check connectivity.",
        )
    return CheckResult(name, CheckStatus.WARN, f"Unreachable ({info})", detail=detail, suggestion=fix)
