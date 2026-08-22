"""Tests for DS Bridge TUI helpers (pure functions only — no terminal)."""

from directionalscalper.core.exchanges import *  # noqa: F401,F403  (repo import sanity)
from ds_tui import readers
from ds_tui.control import find_bot_by_query, is_ds_bot_process, stop_process


def test_parse_bot_args():
    args = readers._parse_bot_args(
        "python3 multi_bot_aio.py --exchange bybit --account_name account_1 "
        "--strategy qsgridob --config configs/config.json"
    )
    assert args["exchange"] == "bybit"
    assert args["strategy"] == "qsgridob"
    assert args["config"] == "configs/config.json"


def test_parse_bot_args_equals_form():
    args = readers._parse_bot_args("python3 bot.py --symbol=SUIUSDT --exchange=bybit")
    assert args == {"symbol": "SUIUSDT", "exchange": "bybit"}


def test_classify_line_patterns():
    assert readers.classify_line("2026-01-01 Traceback (most recent call last):") == ("TRACEBACK", "bold red")
    assert readers.classify_line("ERROR: order rejected by venue") in {
        ("ERROR", "red"), ("REJECTED", "yellow")
    }
    assert readers.classify_line("position updated normally") is None
    assert readers.classify_line("warning: rate limit hit, backing off")[0] == "RATE-LIMIT"


def test_redact_config_masks_secrets_and_never_mutates():
    import copy

    original = {
        "api_key": "REALKEY123",
        "nested": {"passphrase": "hunter2", "levels": [1, {"secret": "abc"}]},
        "symbol": "BTCUSDT",
    }
    snapshot = copy.deepcopy(original)
    red = readers.redact_config(original)
    # secrets masked at any depth
    assert red["api_key"] == "****"
    assert red["nested"]["passphrase"] == "****"
    assert red["nested"]["levels"][1]["secret"] == "****"
    # non-secret data survives
    assert red["symbol"] == "BTCUSDT"
    assert red["nested"]["levels"][0] == 1
    # input untouched
    assert original == snapshot
    assert original["api_key"] == "REALKEY123"


def test_load_config_redacted_missing_file_is_none():
    assert readers.load_config_redacted("definitely_not_a_real_file_9x.json") is None


def test_stop_process_refuses_non_bot_pid():
    ok = stop_process(1)
    assert not ok.ok  # never allowed to signal pid 1
    assert "refus" in ok.message.lower()


def test_is_ds_bot_process_current_python():
    # this test process's cmdline contains pytest, not bot.py — must be False
    import os
    assert not is_ds_bot_process(os.getpid())


def test_find_bot_by_query_no_crash_empty():
    # no bots running in CI; loose match must return None without raising
    assert find_bot_by_query("nonexistent-strategy-xyz") is None
