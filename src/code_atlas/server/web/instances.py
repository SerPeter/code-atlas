"""Where the running UIs are, and which port the next one may take.

`atlas ui` is a per-checkout tool run by hand, so several are live at once as a matter
of course -- one per worktree, sometimes two agent sessions in the same one. All of them
defaulted to port 8420, so the second invocation died on "address already in use" and
the user had to find a free port themselves.

Two ideas, and it matters which one is authoritative:

- **The bound socket is the claim.** The port is taken by binding it here and handing the
  socket to uvicorn, so two invocations racing for the same port cannot both win. A
  check-then-bind would leave exactly that gap.
- **The registry file is a report.** It says where the other UIs are so the CLI can print
  them. Nothing depends on it being correct, which is why it needs no locking, no pid
  liveness probe, and no cleanup on a hard kill: an entry whose port is free is stale by
  definition and is pruned on sight, by the same probe that decides availability.

`os.kill(pid, 0)` would be the usual liveness check and is deliberately not used: on
Windows `os.kill` does not implement signal 0 and terminates the process instead.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Per-user on Windows (%TEMP%), shared on POSIX. Shared is fine: the records are only a
#: report, and ports are a machine-wide resource anyway.
RUNTIME_DIR = Path(tempfile.gettempdir()) / "code-atlas-ui"

#: How many consecutive ports to try before giving up. Twenty is far past any plausible
#: number of concurrent UIs; failing after that means something else is wrong.
PORT_SPAN = 20


@dataclass(frozen=True)
class UiInstance:
    """A UI that was serving when its record was written."""

    pid: int
    host: str
    port: int
    project: str
    root: str
    started: float

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _bind(host: str, port: int) -> socket.socket | None:
    """Bind *port*, or ``None`` if something already holds it."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name != "nt":
        # Windows' SO_REUSEADDR lets a second socket steal a bound port, which would
        # make this probe answer "free" for a port that is very much in use.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError:
        sock.close()
        return None
    return sock


def port_is_free(host: str, port: int) -> bool:
    """Whether *port* can be bound right now."""
    sock = _bind(host, port)
    if sock is None:
        return False
    sock.close()
    return True


def claim_port(host: str, preferred: int, span: int = PORT_SPAN) -> tuple[socket.socket, int]:
    """Bind the first free port at or after *preferred* and return it with its socket.

    The socket is returned rather than closed so the caller can hand it to uvicorn:
    releasing it first would reopen the race this exists to close.
    """
    for port in range(preferred, preferred + span):
        sock = _bind(host, port)
        if sock is not None:
            return sock, port
    msg = f"No free port in {preferred}-{preferred + span - 1} on {host}"
    raise OSError(msg)


def live_instances() -> list[UiInstance]:
    """Every UI whose port is still held, pruning records whose port is not.

    A record is not evidence: the process may have been killed, or the port taken over
    by something unrelated. Only the port probe is.
    """
    if not RUNTIME_DIR.is_dir():
        return []
    live: list[UiInstance] = []
    for path in sorted(RUNTIME_DIR.glob("*.json")):
        try:
            record = UiInstance(**json.loads(path.read_text(encoding="utf-8")))
        except OSError, ValueError, TypeError:
            # Unreadable or written by an older shape — same treatment as stale.
            path.unlink(missing_ok=True)
            continue
        if port_is_free(record.host, record.port):
            path.unlink(missing_ok=True)
        else:
            live.append(record)
    return live


@contextlib.contextmanager
def registered(host: str, port: int, project: str, root: str) -> Iterator[None]:
    """Publish this UI's whereabouts for the duration of the block.

    Failures here are logged and swallowed: a UI that cannot write its record still
    serves perfectly well, and refusing to start over a missing temp directory would
    trade a working tool for a bookkeeping detail.
    """
    record = UiInstance(pid=os.getpid(), host=host, port=port, project=project, root=root, started=time.time())
    path = RUNTIME_DIR / f"{host}-{port}.json"
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(record)), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not record this UI instance ({}) — continuing", exc)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
