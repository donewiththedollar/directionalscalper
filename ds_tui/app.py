"""DS Bridge — a Hummingbot-style terminal console for DirectionalScalper.

Layout (top to bottom):
  STATUS STRIP  one compact line: bots running · exchanges · alerts · uptime
  MAIN OUTPUT   left: running-bots table (or command output / log pager);
                right: live ALERTS pane (critical log lines, classified).
  INPUT         a persistent ``>>> `` command line pinned at the very bottom.

Read-only over all trading state. The ONLY write is stopping a bot process,
gated behind an explicit y/N confirm. API keys are never rendered — config
viewing goes through readers.redact_config. Degrades gracefully: missing
data shows '—', never a crash.
"""

from __future__ import annotations

import json
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from typing import Any, List, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import control, readers
from .paths import LOGS_DIR

REFRESH_S = 2.0

ACCENT = "grey39"
ACCENT_HI = "cyan"

_BANNER = "DS BRIDGE"


# ── shared state ──────────────────────────────────────────────────────────
@dataclass
class UIState:
    cmd_buffer: str = ""
    should_quit: bool = False
    started_at: float = field(default_factory=time.time)
    status_msg: str = ""
    view: str = "tables"          # "tables" | "output"
    output_lines: List[str] = field(default_factory=list)
    bots: list = field(default_factory=list)
    last_discovery: float = 0.0


def _fmt_uptime(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ── renderers ────────────────────────────────────────────────────────────────
def render_status(state: UIState) -> Panel:
    alerts = readers.recent_alerts(max_lines=50)
    n_crit = sum(1 for a in alerts if "red" in a["style"])
    exchanges = ",".join(sorted({b["exchange"] for b in state.bots})) or "—"
    t = Text()
    t.append(f" {_BANNER} ", style="bold white on dark_blue")
    t.append(f"  bots {len(state.bots)}", style="bold cyan")
    t.append(f"  ·  exchange(s) {exchanges}", style="white")
    alert_style = "bold red" if n_crit else "green"
    t.append(f"  ·  alerts {len(alerts)}", style=alert_style)
    if state.status_msg:
        t.append(f"  ·  {state.status_msg}", style="yellow")
    t.append(Text.assemble(("  ·  up ", "dim"), (_fmt_uptime(time.time() - state.started_at), "dim")))
    return Panel(t, border_style=ACCENT, padding=(0, 0))


def render_bots_table(state: UIState) -> Table:
    table = Table(title="Running Bots", box=None, title_style="bold", expand=True)
    for col, just in (("PID", "right"), ("KIND", "left"), ("EXCHANGE", "left"),
                      ("STRATEGY", "left"), ("ACCOUNT", "left"), ("SYMBOL", "left"),
                      ("UPTIME", "right")):
        table.add_column(col, justify=just, no_wrap=True)
    if not state.bots:
        table.add_row("—", "—", "—", "no running DS processes", "—", "—", "—")
    for b in state.bots:
        table.add_row(
            str(b["pid"]), b["kind"], str(b["exchange"]),
            str(b["strategy"])[:28], str(b["account"])[:16],
            str(b["symbol"])[:14], b["etime"],
        )
    return table


def render_log_files_table() -> Table:
    table = Table(title="Log Files (newest first)", box=None, title_style="bold", expand=True)
    table.add_column("NAME", no_wrap=True)
    table.add_column("SIZE", justify="right")
    table.add_column("AGE", justify="right")
    files = readers.list_log_files()
    if not files:
        table.add_row("—", "—", f"(no logs dir at {LOGS_DIR})")
    now = time.time()
    for f in files[:12]:
        age = max(0, int(now - f["mtime"]))
        table.add_row(f["name"][:40], f"{f['size']:,}B", _fmt_uptime(age))
    return table


def render_output(state: UIState) -> Panel:
    body = Text("\n".join(state.output_lines[-40:]) or "(no output)", style="white")
    return Panel(body, title="[bold]OUTPUT[/]", border_style=ACCENT_HI)


def render_alerts(max_lines: int = 18) -> Panel:
    lines = Text()
    alerts = readers.recent_alerts(max_lines=max_lines)
    if not alerts:
        lines.append("no critical lines in recent logs\n", style="dim green")
        lines.append("(scanning newest logs for Traceback/ERROR/reject/", style="dim")
        lines.append("liquidation/margin/rate-limit patterns)", style="dim")
    else:
        now = time.time()
        for a in alerts:
            age = _fmt_uptime(max(0, int(now - a["age"])))
            lines.append(f"[{age}]", style="dim")
            lines.append(f" {a['label']}", style=a["style"])
            lines.append(f" {a['line']}\n", style="white")
    return Panel(lines, title=f"[bold]LOG / ALERTS[/]", border_style=ACCENT)


def render_input(state: UIState) -> Panel:
    t = Text()
    t.append(">>> ", style="bold cyan")
    t.append(state.cmd_buffer)
    t.append("▌", style="blink bold")
    return Panel(t, border_style=ACCENT_HI, padding=(0, 1))


def build_layout(state: UIState, width: int, height: int) -> Layout:
    layout = Layout()
    layout.split_column(Layout(name="status", size=3), Layout(name="body"), Layout(name="input", size=3))
    layout["status"].update(render_status(state))
    layout["input"].update(render_input(state))

    main_view = render_bots_table(state) if state.view == "tables" else render_output(state)
    extra = Group(render_log_files_table(), main_view) if state.view == "tables" else main_view

    layout["body"].split_row(Layout(name="main", ratio=3), Layout(name="alerts", ratio=2))
    layout["main"].update(Panel(extra, border_style=ACCENT))
    layout["alerts"].update(render_alerts())
    return layout


# ── terminal plumbing ────────────────────────────────────────────────────────
class _RawKeys:
    """Context manager putting stdin in cbreak mode for single-key reads."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def _read_key(timeout: float) -> Optional[str]:
    import select
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        more, _, _ = select.select([sys.stdin], [], [], 0.001)
        if not more:
            return "ESC"
        seq = sys.stdin.read(2)
        return {"[A": "UP", "[B": "DOWN", "[C": "RIGHT", "[D": "LEFT"}.get(seq, "ESC")
    return ch


def _confirm(console: Console, prompt: str) -> bool:
    console.print(Text(f"\n{prompt} [y/N] ", style="bold yellow"), end="")
    sys.stdout.flush()
    key = _read_key(timeout=30.0)
    console.print("")
    return key is not None and key.lower() == "y"


_HELP = """DS Bridge commands
  status / bots       refresh the running-bot discovery
  logs                show newest log files
  log <name> [lines]  tail a log file from logs/ (e.g. log MultiBot)
  errors [n]          recent classified critical lines across all logs
  config [file.json]  view a configs/*.json file with ALL SECRETS REDACTED
  grep <pattern>      search the newest log for a pattern (fixed string)
  stop <pid|name>     stop a running DS bot process (SIGTERM) — confirmed
  clear               clear the output pane
  help                this help
  quit / q            exit DS Bridge
Type at the >>> prompt and press Enter. 'stop' matches loosely by strategy/
exchange/symbol substring, or exactly by pid. Trading state is READ-ONLY."""


def _pager(console: Console, live: Live, fn) -> None:
    """Drop the alt-screen, run fn (which prints), wait for a key, resume."""
    live.stop()
    try:
        fn()
        console.print("\n[dim]— press any key to return —[/dim]")
        _read_key(timeout=120.0)
    finally:
        live.start(refresh=True)


def _emit(state: UIState, text: str) -> None:
    state.view = "output"
    state.output_lines.extend(text.splitlines())


def _match_bot(state: UIState, query: str):
    q = query.strip().lower()
    exact_pid = [b for b in state.bots if q == str(b["pid"])]
    if exact_pid:
        return exact_pid[0]
    part = [
        b for b in state.bots
        if q in str(b["strategy"]).lower()
        or q in str(b["exchange"]).lower()
        or q in str(b["symbol"]).lower()
    ]
    return part[0] if len(part) == 1 else None


def _dispatch_command(console: Console, state: UIState, live: Live, cmd: str) -> None:
    parts = cmd.split()
    verb, args = parts[0].lower(), parts[1:]

    if verb in ("quit", "q", "exit"):
        state.should_quit = True
        return
    if verb in ("help", "h", "?"):
        _pager(console, live, lambda: console.print(_HELP))
        return
    if verb in ("status", "bots"):
        state.view = "tables"
        state.bots = readers.discover_running_bots()
        state.status_msg = f"{len(state.bots)} bot(s)"
        return
    if verb == "clear":
        state.output_lines.clear()
        state.view = "tables"
        state.status_msg = ""
        return
    if verb == "logs":
        state.bots = readers.discover_running_bots()
        _pager(console, live, lambda: console.print(render_log_files_table()))
        return
    if verb == "log":
        if not args:
            _emit(state, "usage: log <name-substring> [lines]")
            return
        files = readers.list_log_files(40)
        match = next((f for f in files if args[0].lower() in f["name"].lower()), None)
        if not match:
            _emit(state, f"no log matching '{args[0]}' — try 'logs'")
            return
        n = int(args[1]) if len(args) > 1 and args[1].isdigit() else 40
        lines = readers.tail_log(match["path"], max_bytes=200_000, max_lines=n)

        def show_log():
            console.print(f"[bold]{match['name']}[/] (last {len(lines)} lines)")
            for ln in lines:
                hit = readers.classify_line(ln)
                style = hit[1] if hit else "dim"
                console.print(Text(ln, style=style))
        _pager(console, live, show_log)
        return
    if verb == "errors":
        n = int(args[0]) if args and args[0].isdigit() else 20

        def show_errors():
            alerts = readers.recent_alerts(max_lines=n)
            if not alerts:
                console.print("[green]no critical lines found in recent logs[/]")
            for a in alerts:
                console.print(f"[dim]{a['file']}[/] [bold]{a['label']}[/] {a['line']}")
        _pager(console, live, show_errors)
        return
    if verb == "config":
        name = args[0] if args else None
        if not name:
            def show_cfg_list():
                console.print("[bold]configs/*.json:[/] " + ", ".join(readers.list_config_files()))
                console.print("[dim]secrets are redacted on display[/]")
            _pager(console, live, show_cfg_list)
            return
        data = readers.load_config_redacted(name)

        def show_cfg():
            if data is None:
                console.print(f"[red]cannot read configs/{name}[/]")
                return
            console.print_json(json.dumps(data, indent=2, default=str))
        _pager(console, live, show_cfg)
        return
    if verb == "grep":
        if not args:
            _emit(state, "usage: grep <pattern>")
            return
        pattern = args[0]
        files = readers.list_log_files(6)
        hits: List[str] = []
        for f in files:
            for ln in readers.tail_log(f["path"], max_bytes=200_000, max_lines=2000):
                if pattern.lower() in ln.lower():
                    hits.append(f"[{f['name']}] {ln.strip()[:160]}")
                    if len(hits) >= 80:
                        break
            if len(hits) >= 80:
                break
        _pager(console, live, lambda: console.print(
            "\n".join(hits) or f"[yellow]no matches for '{pattern}' in newest logs[/]"))
        return
    if verb == "stop":
        if not args:
            _emit(state, "usage: stop <pid|strategy|exchange>")
            return
        bot = _match_bot(state, args[0])
        if not bot:
            _emit(state, f"no unique bot matching '{args[0]}' — run 'bots' and use the PID")
            return
        ok = _confirm(console, f"STOP bot pid {bot['pid']} ({bot['kind']} {bot['exchange']} {bot['strategy']} {bot['symbol']})?")
        if not ok:
            state.status_msg = "stop cancelled"
            return
        result = control.stop_process(bot["pid"])
        state.bots = readers.discover_running_bots()
        state.status_msg = result.message
        return

    state.status_msg = f"unknown command '{verb}' — try 'help'"


def _handle_input(console: Console, state: UIState, key: str, live: Live) -> None:
    if key in ("\r", "\n"):
        cmd_text = state.cmd_buffer.strip()
        state.cmd_buffer = ""
        if cmd_text:
            state.status_msg = ""
            _dispatch_command(console, state, live, cmd_text)
        return
    if key == "ESC":
        state.cmd_buffer = ""
        state.status_msg = ""
        return
    if key in ("\x7f", "\x08"):
        state.cmd_buffer = state.cmd_buffer[:-1]
        return
    if key == ":" and not state.cmd_buffer:
        return  # compat shim, like mantis
    if key == "":  # ctrl-c
        state.should_quit = True
        return
    if key and len(key) == 1 and key.isprintable():
        state.cmd_buffer += key


def _refresh_discovery(state: UIState, force: bool = False) -> None:
    if force or (time.time() - state.last_discovery) > 15.0:
        state.bots = readers.discover_running_bots()
        state.last_discovery = time.time()


# ── main loop ────────────────────────────────────────────────────────────────
def run() -> int:
    console = Console()
    state = UIState()
    _refresh_discovery(state, force=True)

    if not sys.stdin.isatty():
        # piped / CI: render one frame and exit cleanly
        console.print(build_layout(state, console.width, console.height))
        return 0

    with _RawKeys():
        with Live(
            build_layout(state, console.width, console.height),
            console=console, screen=True, refresh_per_second=4, auto_refresh=False,
        ) as live:
            while True:
                key = _read_key(timeout=REFRESH_S)
                if key is not None:
                    _handle_input(console, state, key, live)
                    if state.should_quit:
                        break
                _refresh_discovery(state)
                live.update(build_layout(state, console.width, console.height), refresh=True)
    return 0
