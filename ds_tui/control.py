"""Mutating operations for the DS Bridge TUI.

Philosophy (mirrors mantis): the TUI is read-only over all trading state.
The ONLY writes here are stopping a bot process — and callers MUST pass
through a confirm prompt (app.py::_confirm) before invoking these.

stop_process refuses PIDs that don't look like a DS bot process.
"""

from __future__ import annotations

import signal
import subprocess
from dataclasses import dataclass
from typing import Optional

from . import readers


@dataclass
class ActionResult:
    ok: bool
    message: str


def is_ds_bot_process(pid: int) -> bool:
    try:
        r = subprocess.run(
            ["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True, timeout=5
        )
        args = (r.stdout or "").strip()
        return ("multi_bot_aio.py" in args) or ("bot.py" in args)
    except Exception:
        return False


def stop_process(pid: int, timeout: float = 10.0) -> ActionResult:
    """SIGTERM a running DS bot process; verify it exited; escalate to SIGKILL."""
    if not isinstance(pid, int) or pid <= 1:
        return ActionResult(False, f"refusing to stop pid {pid!r}")
    if not is_ds_bot_process(pid):
        return ActionResult(False, f"pid {pid} does not look like a DS bot process")
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\x00", b" ").decode(errors="replace").strip()
    except OSError:
        return ActionResult(False, f"pid {pid} no longer exists")

    # refuse anything with shell metacharacters in the recorded cmdline
    if any(ch in cmd for ch in ";|&`$><"):
        return ActionResult(False, "refusing: cmdline contains shell metacharacters")

    try:
        import os
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return ActionResult(True, f"pid {pid} already gone")
    except PermissionError:
        return ActionResult(False, f"no permission to stop pid {pid}")
    except Exception as e:
        return ActionResult(False, f"failed to signal pid {pid}: {e}")

    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_ds_bot_process(pid):
            return ActionResult(True, f"stopped pid {pid}: {cmd[:80]}")
        time.sleep(0.3)

    try:
        import os
        os.kill(pid, signal.SIGKILL)
        return ActionResult(True, f"force-killed pid {pid}: {cmd[:80]}")
    except Exception as e:
        return ActionResult(False, f"pid {pid} ignored SIGTERM and kill failed: {e}")


def find_bot_by_query(query: str) -> Optional[dict]:
    """Loose-match a running bot by pid, strategy, exchange or symbol substring."""
    q = query.strip().lower()
    bots = readers.discover_running_bots()
    for b in bots:
        if q == str(b["pid"]):
            return b
    part = [
        b for b in bots
        if q in str(b["strategy"]).lower()
        or q in str(b["exchange"]).lower()
        or q in str(b["symbol"]).lower()
    ]
    return part[0] if len(part) == 1 else None
