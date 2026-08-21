"""Breathing-grid policy math.

Simplified port of the exchange-free policy layer from mantis-mainacct's
``cascade_rider/breathing_grid_policy.py`` (see that module for the full
production implementation with frozen reserves and ownership handoff).

Pure calculations only: no exchange connector, no mutable lifecycle state,
no I/O.  Callers (the strategy executor) remain responsible for orders,
positions, and persistence.

Core ideas carried over:
- grid rung depths derived from realized volatility
- fee-aware take-profit (net TP adjusted for maker entry + exit fees)
- "sequential add" discipline: a rung may only be re-added below/above if
  price *continued* adversely past it by a continuation buffer
- hard validation of every policy input (fail fast, ValueError)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

_EPS = 1e-9


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def realized_volatility(closes: Sequence[float]) -> float:
    """Std-dev of per-close simple returns; returns 0.0 for degenerate input."""
    if len(closes) < 3:
        return 0.0
    returns = []
    for prev, nxt in zip(closes, closes[1:]):
        if not (_finite(prev) and _finite(nxt) and prev > 0):
            continue
        returns.append((nxt - prev) / prev)
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(max(variance, 0.0))


def adverse_fraction(side: str, anchor: float, price: float) -> float:
    """Adverse movement of ``price`` relative to ``anchor`` for a position on ``side``."""
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    if not all(_finite(v) and v > 0 for v in (anchor, price)):
        raise ValueError("anchor and price must be finite and positive")
    if side == "short":
        return (price - anchor) / anchor
    return (anchor - price) / anchor


def adverse_continuation(side: str, touch: float, price: float) -> float:
    """Additional adverse movement after a rung was touched at ``touch``."""
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    if not all(_finite(v) and v > 0 for v in (touch, price)):
        raise ValueError("touch and price must be finite and positive")
    if side == "short":
        return (price - touch) / touch
    return (touch - price) / touch


def fee_aware_tp_fraction(
    net_tp_fraction: float,
    maker_entry_fee_fraction: float,
    exit_fee_fraction: float,
) -> float:
    """Gross TP fraction required to net ``net_tp_fraction`` after round-trip fees."""
    if min(net_tp_fraction, maker_entry_fee_fraction, exit_fee_fraction) < 0:
        raise ValueError("fee/tp fractions cannot be negative")
    return net_tp_fraction + maker_entry_fee_fraction + exit_fee_fraction


@dataclass(frozen=True)
class BreathingPolicy:
    """Six-level volatility-derived breathing grid parameters."""

    level_count: int = 6
    span_vol_multiplier: float = 6.0
    min_terminal_depth: float = 0.06
    max_terminal_depth: float = 0.15
    first_level_fraction: float = 0.20      # fraction of per-level qty on level 1... scaled up outward
    qty_growth: float = 1.0                 # geometric qty growth per level (1.0 = flat)
    net_tp_fraction: float = 0.003          # desired NET take profit per cycle
    maker_entry_fee_fraction: float = 0.0002
    exit_fee_fraction: float = 0.00055
    add_continuation_fraction: float = 0.002  # extra adverse move required past a touched rung before re-add
    rebound_fraction: float = 0.02            # exit rebound target off worst touch
    max_leverage: float = 5.0
    max_position_usd: Optional[float] = None  # absolute cap; None => budget * leverage bound only

    def __post_init__(self) -> None:
        numerics = (
            self.span_vol_multiplier,
            self.min_terminal_depth,
            self.max_terminal_depth,
            self.first_level_fraction,
            self.qty_growth,
            self.net_tp_fraction,
            self.maker_entry_fee_fraction,
            self.exit_fee_fraction,
            self.add_continuation_fraction,
            self.rebound_fraction,
            self.max_leverage,
        )
        if any(not _finite(v) or v < 0 for v in numerics):
            raise ValueError("breathing policy values must be finite and non-negative")
        if self.level_count < 2:
            raise ValueError("breathing policy requires at least two levels")
        if not 0 < self.min_terminal_depth <= self.max_terminal_depth < 1:
            raise ValueError("invalid terminal-depth bounds: need 0 < min <= max < 1")
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")

    @property
    def gross_tp_fraction(self) -> float:
        return fee_aware_tp_fraction(
            self.net_tp_fraction, self.maker_entry_fee_fraction, self.exit_fee_fraction
        )


def rung_depths(mid_price: float, volatility: float, policy: BreathingPolicy) -> List[float]:
    """Return ``level_count`` entry prices for one side, ordered nearest-first.

    Depths are spread linearly between the first-rung depth and the terminal
    depth; the whole span scales with volatility via ``span_vol_multiplier``
    and is clamped to [min_terminal_depth, max_terminal_depth].
    """
    if not _finite(mid_price) or mid_price <= 0:
        raise ValueError("mid_price must be finite and positive")
    vol_span = volatility * policy.span_vol_multiplier
    span = min(max(vol_span, policy.min_terminal_depth), policy.max_terminal_depth)
    n = policy.level_count
    depths = [span * policy.first_level_fraction + span * (1 - policy.first_level_fraction) * i / (n - 1) for i in range(n)]
    return [mid_price * (1 - d) for d in depths]  # long-side prices; short side mirrors around mid


def mirror_prices(prices: Sequence[float], mid_price: float) -> List[float]:
    """Mirror long-side rung prices to the short side around ``mid_price``."""
    mirrored = []
    for p in prices:
        if not _finite(p) or p <= 0 or p >= mid_price:
            raise ValueError("rung prices must be finite, positive, and below mid")
        mirrored.append(mid_price + (mid_price - p))
    return mirrored


def per_level_qty(budget_usd: float, mid_price: float, policy: BreathingPolicy) -> List[float]:
    """Qty per rung given a total budget split across levels (geometric growth)."""
    if not _finite(budget_usd) or budget_usd <= 0:
        raise ValueError("budget_usd must be finite and positive")
    g = policy.qty_growth
    weights = [g**i for i in range(policy.level_count)]
    total = sum(weights)
    qtys = [budget_usd * w / total / mid_price for w in weights]
    return qtys


def sequential_add_allowed(
    side: str,
    touched_rung_price: float,
    current_price: float,
    policy: BreathingPolicy,
) -> bool:
    """True iff price continued adversely past ``touched_rung_price`` by the buffer.

    Implements the mantis 'decide before touching the exchange' rule: never
    re-add at a rung the market merely wicked through.
    """
    cont = adverse_continuation(side, touched_rung_price, current_price)
    return cont >= policy.add_continuation_fraction


def take_profit_price(side: str, avg_entry: float, policy: BreathingPolicy) -> float:
    """Exit price achieving the net TP after fees, for the given side."""
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    tp = avg_entry * (1 + policy.gross_tp_fraction) if side == "long" else avg_entry * (
        1 - policy.gross_tp_fraction
    )
    return tp
