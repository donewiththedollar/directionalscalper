"""Read-only data access for the DS Bridge TUI.

Everything here is read-only: process discovery (ps), log tailing and
classification, and redacted config viewing.  No exchange API calls, no
writes.  Degrades gracefully — missing data renders as '—', never a crash.
"""

from __future__ import annotations

import copy
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import CONFIGS_DIR, LOGS_DIR


# ── running-bot discovery ────────────────────────────────────────────────────

def _parse_bot_args(args_str: str) -> Dict[str, str]:
    """Extract --flag value pairs from a command line into a plain dict."""
    out: Dict[str, str] = {}
    parts = args_str.split()
    i = 0
    while i < len(parts):
        tok = parts[i]
        if tok.startswith("--") and "=" not in tok and i + 1 < len(parts):
            out[tok[2:]] = parts[i + 1]
            i += 2
        elif tok.startswith("--") and "=" in tok:
            k, _, v = tok[2:].partition("=")
            out[k] = v
            i += 1
        else:
            i += 1
    return out


def discover_running_bots() -> List[Dict[str, Any]]:
    """Find running DS bot processes (multi_bot_aio.py / bot.py) with their args."""
    bots: List[Dict[str, Any]] = []
    try:
        r = subprocess.run(
            ["ps", "-eo", "pid,etime,args"], capture_output=True, text=True, timeout=10
        )
        for line in (r.stdout or "").splitlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid_s, etime, args = parts
            if "multi_bot_aio.py" not in args and re.search(r"python3?\s+.*bot\.py\b", args) is None:
                continue
            flags = _parse_bot_args(args)
            bots.append({
                "pid": int(pid_s),
                "etime": etime,
                "kind": "multi" if "multi_bot_aio.py" in args else "single",
                "exchange": flags.get("exchange", "—"),
                "strategy": flags.get("strategy", "—"),
                "account": flags.get("account_name", "—"),
                "symbol": flags.get("symbol", "(rotator)" if "multi_bot_aio.py" in args else "—"),
                "config": flags.get("config", "—"),
                "_args": args,
            })
    except Exception:
        pass
    return sorted(bots, key=lambda b: b["pid"])


# ── logs ─────────────────────────────────────────────────────────────────────

def list_log_files(limit: int = 12) -> List[Dict[str, Any]]:
    """Newest log files in logs/ by mtime."""
    files = []
    try:
        for p in Path(LOGS_DIR).glob("*.log*"):
            try:
                files.append({
                    "name": p.name,
                    "path": p,
                    "mtime": p.stat().st_mtime,
                    "size": p.stat().st_size,
                })
            except OSError:
                continue
    except OSError:
        pass
    return sorted(files, key=lambda f: f["mtime"], reverse=True)[:limit]


def tail_log(path: Path, max_bytes: int = 64_000, max_lines: int = 40) -> List[str]:
    """Last lines of a log file, tolerating a truncated live-appended tail."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            raw = f.read().decode("utf-8", errors="replace")
        lines = raw.splitlines()
        return lines[-max_lines:]
    except OSError:
        return []


# ── alert classification (condor logwatch idea, DS patterns) ────────────────

ALERT_PATTERNS: List[tuple] = [
    (re.compile(r"Traceback|unhandled exception", re.I), ("TRACEBACK", "bold red")),
    (re.compile(r"\bCRITICAL\b"), ("CRITICAL", "bold red")),
    (re.compile(r"\bERROR\b"), ("ERROR", "red")),
    (re.compile(r"rejected|reject", re.I), ("REJECTED", "yellow")),
    (re.compile(r"liquidat", re.I), ("LIQUIDATION", "bold red")),
    (re.compile(r"insufficient (balance|margin)", re.I), ("MARGIN", "yellow")),
    (re.compile(r"kill.?switch|flatten.*halt|drawdown", re.I), ("KILL/DRAWDOWN", "bold yellow")),
    (re.compile(r"rate.?limit|429|-1003", re.I), ("RATE-LIMIT", "yellow")),
]


def classify_line(line: str) -> Optional[tuple]:
    """Return (label, style) for the first matching critical pattern, else None."""
    for rx, label in ALERT_PATTERNS:
        if rx.search(line):
            return label
    return None


def recent_alerts(max_lines: int = 14) -> List[Dict[str, Any]]:
    """Scan the newest logs for critical lines; return labelled alerts, newest last."""
    alerts: List[Dict[str, Any]] = []
    now = time.time()
    for f in list_log_files(6):
        for line in tail_log(f["path"], max_bytes=32_000, max_lines=400):
            hit = classify_line(line)
            if hit:
                alerts.append({
                    "file": f["name"],
                    "line": line.strip()[:180],
                    "label": hit[0],
                    "style": hit[1],
                    "age": now - f["mtime"],
                })
    return alerts[-max_lines:]


# ── config viewing with secret redaction ────────────────────────────────────

_SECRET_KEY_RX = re.compile(
    r"api_key|api_secret|secret|token|passphrase|password|private_key|mnemonic|seed",
    re.I,
)


def redact_config(obj: Any) -> Any:
    """Deep-copy obj masking every secret-looking field with '****'.

    Never mutates the input; never raises on ordinary containers.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_KEY_RX.search(k):
                out[k] = "****"
            else:
                out[k] = redact_config(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact_config(v) for v in obj]
    return obj


def load_config_redacted(name: str) -> Optional[Any]:
    """Load configs/<name> JSON and redact it; returns None if unreadable."""
    import json

    path = Path(CONFIGS_DIR) / name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    return redact_config(copy.deepcopy(data))


def list_config_files(limit: int = 12) -> List[str]:
    try:
        return sorted(p.name for p in Path(CONFIGS_DIR).glob("*.json"))[:limit]
    except OSError:
        return []
