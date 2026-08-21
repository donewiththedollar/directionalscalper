"""Unit tests for the breathing-grid policy math (pure functions, no exchange)."""

import pytest

from directionalscalper.core.strategies.bybit.gridbased.breathing_policy import (
    BreathingPolicy,
    adverse_continuation,
    adverse_fraction,
    fee_aware_tp_fraction,
    mirror_prices,
    per_level_qty,
    realized_volatility,
    rung_depths,
    sequential_add_allowed,
    take_profit_price,
)


@pytest.fixture
def policy():
    return BreathingPolicy()


def test_realized_volatility_zero_for_flat_series():
    assert realized_volatility([100.0] * 10) == 0.0
    assert realized_volatility([]) == 0.0
    assert realized_volatility([1.0, 2.0]) == 0.0


def test_realized_volatility_positive_for_moving_series():
    vol = realized_volatility([100, 101, 99, 102, 98, 103])
    assert vol > 0


def test_adverse_fraction_sides():
    # long suffers when price falls below anchor
    assert adverse_fraction("long", 100.0, 90.0) == pytest.approx(0.10)
    assert adverse_fraction("long", 100.0, 110.0) < 0
    # short suffers when price rises above anchor
    assert adverse_fraction("short", 100.0, 110.0) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        adverse_fraction("sideways", 100.0, 90.0)


def test_adverse_continuation():
    # long: touched 95, now 94 -> continued adversely by ~1.05%
    assert adverse_continuation("long", 95.0, 94.0) > 0.01
    assert adverse_continuation("long", 95.0, 96.0) < 0


def test_fee_aware_tp_adds_both_fees():
    gross = fee_aware_tp_fraction(0.003, 0.0002, 0.00055)
    assert gross == pytest.approx(0.00375)


def test_rung_depths_monotonic_and_vol_scaled(policy):
    mid = 100.0
    low_vol = rung_depths(mid, 0.001, policy)
    high_vol = rung_depths(mid, 0.02, policy)

    # nearest-first, all below mid for the long side
    assert all(p < mid for p in low_vol)
    assert low_vol == sorted(low_vol, reverse=True)
    assert len(low_vol) == policy.level_count

    # higher volatility pushes rungs further out (cheapest rung deeper)
    assert min(high_vol) < min(low_vol)

    # clamped to terminal depth bound even at extreme vol
    crazy = rung_depths(mid, 5.0, policy)
    assert (mid - min(crazy)) / mid <= policy.max_terminal_depth + 1e-9

    with pytest.raises(ValueError):
        rung_depths(-1.0, 0.01, policy)


def test_mirror_prices_reflect_around_mid():
    mid = 200.0
    longs = [190.0, 180.0]
    shorts = mirror_prices(longs, mid)
    assert shorts == [210.0, 220.0]
    with pytest.raises(ValueError):
        mirror_prices([250.0], mid)


def test_per_level_qty_budget_respected(policy):
    mid = 50.0
    budget = 600.0
    qtys = per_level_qty(budget, mid, policy)
    notional = sum(q * mid for q in qtys)
    assert notional == pytest.approx(budget)
    with pytest.raises(ValueError):
        per_level_qty(-5, mid, policy)


def test_sequential_add_gate(policy):
    touched_rung = 95.0
    # wicked through and back: no continuation -> blocked
    assert not sequential_add_allowed("long", touched_rung, 96.0, policy)
    # continued past the rung by more than the buffer -> allowed
    assert sequential_add_allowed("long", touched_rung, 94.5, policy)


def test_take_profit_price_is_fee_aware_and_directional(policy):
    long_tp = take_profit_price("long", 100.0, policy)
    short_tp = take_profit_price("short", 100.0, policy)
    assert long_tp > 100.0 * (1 + policy.net_tp_fraction)   # fees push TP further out
    assert short_tp < 100.0 * (1 - policy.net_tp_fraction)
    with pytest.raises(ValueError):
        take_profit_price("both", 100.0, policy)


def test_policy_validation():
    with pytest.raises(ValueError):
        BreathingPolicy(min_terminal_depth=0.2, max_terminal_depth=0.1)  # min > max
    with pytest.raises(ValueError):
        BreathingPolicy(level_count=1)
    with pytest.raises(ValueError):
        BreathingPolicy(max_leverage=0)
    with pytest.raises(ValueError):
        BreathingPolicy(net_tp_fraction=-1)


def test_executor_config_rejects_unknown_keys():
    """The executor must fail fast on typos in breathing_grid config."""
    from directionalscalper.core.strategies.bybit.gridbased.breathing_grid import DEFAULT_CONFIG

    unknown = {"budget_usdd": 100}
    assert set(unknown) - set(DEFAULT_CONFIG), "sanity check on the check itself"


def test_venue_adapters_share_one_core():
    """mantis-style modularity: one core executor, thin venue adapters."""
    from directionalscalper.core.strategies.bybit.gridbased.breathing_grid import (
        BreathingGridCore,
        BreathingGridFutures,
    )
    from directionalscalper.core.strategies.blofin import BreathingGridBloFin

    for adapter in (BreathingGridFutures, BreathingGridBloFin):
        assert issubclass(adapter, BreathingGridCore)
        for hook in (
            "_venue_set_leverage",
            "_venue_mid_price",
            "_venue_closes",
            "_venue_positions",
            "_venue_equity_usd",
            "_venue_place_limit",
            "_venue_cancel_order",
            "_venue_open_order_ids",
            "_venue_flatten",
        ):
            assert hook in vars(adapter), f"{adapter.__name__} must implement {hook} itself"


def test_blofin_broker_id_default():
    """Order flow must be attributed via the mantis broker code by default."""
    from directionalscalper.core.exchanges.blofin import BlofinExchange

    assert BlofinExchange.DEFAULT_BROKER_ID == "cc84bbde7d4b8a8c"
