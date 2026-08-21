"""Smoke tests: verify the package imports and config models load on current deps.

These are intentionally lightweight (no network, no exchange calls) so they can
run in CI or inside a fresh Docker build to catch dependency breakage early.
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "config_example.json"
EXAMPLE_ACCOUNT = REPO_ROOT / "configs" / "account_example.json"


def test_config_module_imports():
    import config  # noqa: F401


def test_core_package_imports():
    import directionalscalper.core.exchanges as exchanges  # noqa: F401
    import directionalscalper.core.strategies.base_strategy as base_strategy  # noqa: F401


def test_entry_points_import():
    # Import-only check: __main__ blocks must not execute on import.
    import bot  # noqa: F401
    import multi_bot_aio  # noqa: F401
    import rate_limit  # noqa: F401


def test_example_config_and_account_merge():
    """load_config() must merge account credentials into the parsed Config."""
    import config as cfg

    raw = json.loads(EXAMPLE_CONFIG.read_text())
    account = json.loads(EXAMPLE_ACCOUNT.read_text())

    # sanity: every exchange in the example config has a matching account entry
    cfg_pairs = {(e["name"], e["account_name"]) for e in raw["exchanges"]}
    acct_pairs = {(e["name"], e["account_name"]) for e in account["exchanges"]}
    assert cfg_pairs <= acct_pairs, "example config has exchanges with no account entry"

    parsed = cfg.load_config(EXAMPLE_CONFIG, EXAMPLE_ACCOUNT)
    assert parsed.bot is not None
    assert isinstance(parsed.bot.linear_grid, dict)
    assert all(e.api_key and e.api_secret for e in parsed.exchanges)


def test_standardize_symbol():
    """Entry-point helper delegates to core.symbols (mantis convention:
    uppercase BASEQUOTE, strict validation)."""
    from multi_bot_aio import standardize_symbol

    assert standardize_symbol("BTC/USDT") == "BTCUSDT"
    assert standardize_symbol("SUI/USDT:SUI") == "SUIUSDT"
    assert standardize_symbol("ethusdt") == "ETHUSDT"


def test_symbols_canonical_and_venue_forms():
    from directionalscalper.core.symbols import (
        canonical_symbol,
        to_blofin_inst_id,
        to_bybit_linear,
    )

    for raw in ("BTC/USDT", "btc-usdt", "BTC/USDT:USDT", "BTCUSDT"):
        assert canonical_symbol(raw) == "BTCUSDT"
        assert to_bybit_linear(raw) == "BTCUSDT"
        assert to_blofin_inst_id(raw) == "BTC-USDT"

    # already-canonical venue forms pass through unchanged
    assert to_blofin_inst_id("BTC-USDT") == "BTC-USDT"

    import pytest

    for bad in ("", None, "BTC", "ETH/BTC", "NOT A SYMBOL"):
        with pytest.raises(ValueError):
            canonical_symbol(bad)
    with pytest.raises(ValueError):
        to_blofin_inst_id("ETH/BTC")
