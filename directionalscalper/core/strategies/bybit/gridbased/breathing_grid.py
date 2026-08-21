"""Breathing Grid Futures — volume-farming market-making grid for Bybit USDT perps.

Simplified executor over :mod:`breathing_policy` (ported from mantis-mainacct's
cascade_rider breathing-grid policy).  This is NOT a port of mantis's full
production owner (no frozen reserves, no ownership ledger, no hedge handoff) —
it is a deliberately small, single-symbol, capped grid strategy:

- post-only-style maker entry ladders on both sides while flat
- while in a campaign, entries only on the held side, and a rung may only be
  (re-)added if price *continued* adversely past it (sequential-add rule)
- fee-aware take-profit reduce-only exit on the held side
- the ladder "breathes": unfilled quotes are re-priced to fresh
  volatility-derived depths every cycle
- hard caps: max leverage, max position notional, session drawdown lockout

Config comes from ``config.breathing_grid`` (dict).  All keys are optional;
sane defaults below.  This strategy is UNTESTED ON LIVE FUNDS — start with
testnet or dust-sized budget_usd.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

from directionalscalper.core.strategies.bybit.bybit_strategy import BybitStrategy
from directionalscalper.core.strategies.bybit.gridbased.breathing_policy import (
    BreathingPolicy,
    mirror_prices,
    per_level_qty,
    realized_volatility,
    rung_depths,
    sequential_add_allowed,
    take_profit_price,
)

DEFAULT_CONFIG = {
    "budget_usd": 200.0,
    "level_count": 6,
    "span_vol_multiplier": 6.0,
    "min_terminal_depth": 0.06,
    "max_terminal_depth": 0.15,
    "first_level_fraction": 0.20,
    "qty_growth": 1.0,
    "net_tp_fraction": 0.003,
    "maker_entry_fee_fraction": 0.0002,
    "exit_fee_fraction": 0.00055,
    "add_continuation_fraction": 0.002,
    "rebound_fraction": 0.02,
    "max_leverage": 5.0,
    "max_position_usd": None,          # absolute notional cap; None -> budget * levels bound
    "candle_timeframe": "1m",
    "vol_lookback": 60,
    "reprice_interval_secs": 20,
    "requote_tolerance_fraction": 0.25,  # re-quote when ladder drifted > this fraction of span
    "session_max_drawdown_pct": 0.02,    # flatten + halt at -2% equity from session start
}


class BreathingGridFutures(BybitStrategy):
    """Volume-farming breathing-grid strategy; see module docstring."""

    def __init__(self, exchange, manager, config, symbols_allowed=None, rotator_symbols_standardized=None, mfirsi_signal=None):
        super().__init__(exchange, config, manager, symbols_allowed)

    def _breathing_cfg(self) -> Dict:
        raw = dict(DEFAULT_CONFIG)
        user = getattr(self.config, "breathing_grid", None) or {}
        unknown = set(user) - set(raw)
        if unknown:
            raise ValueError(f"breathing_grid has unknown keys: {sorted(unknown)}")
        raw.update(user)
        return raw

    def _build_policy(self, cfg: Dict) -> BreathingPolicy:
        return BreathingPolicy(
            level_count=int(cfg["level_count"]),
            span_vol_multiplier=float(cfg["span_vol_multiplier"]),
            min_terminal_depth=float(cfg["min_terminal_depth"]),
            max_terminal_depth=float(cfg["max_terminal_depth"]),
            first_level_fraction=float(cfg["first_level_fraction"]),
            qty_growth=float(cfg["qty_growth"]),
            net_tp_fraction=float(cfg["net_tp_fraction"]),
            maker_entry_fee_fraction=float(cfg["maker_entry_fee_fraction"]),
            exit_fee_fraction=float(cfg["exit_fee_fraction"]),
            add_continuation_fraction=float(cfg["add_continuation_fraction"]),
            rebound_fraction=float(cfg["rebound_fraction"]),
            max_leverage=float(cfg["max_leverage"]),
            max_position_usd=cfg.get("max_position_usd"),
        )

    # ---------- helpers ----------

    def _round_price(self, symbol: str, price: float) -> float:
        try:
            precision = int(self.exchange.get_price_precision(symbol))
            return round(price, precision)
        except Exception:
            return round(price, 4)

    def _round_qty(self, symbol: str, qty: float) -> float:
        try:
            return float(self.exchange.exchange.amount_to_precision(symbol, qty))
        except Exception:
            return round(qty, 3)

    def _positions(self, symbol: str) -> Tuple[float, float, float, float]:
        """Return (long_qty, long_entry, short_qty, short_entry)."""
        long_qty = short_qty = 0.0
        long_entry = short_entry = 0.0
        try:
            data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)
            for pos in data or []:
                info = pos.get("info", {})
                pos_symbol = standardize_pos_symbol(info.get("symbol", ""))
                if pos_symbol != symbol.split("/")[0].split(":")[0].upper():
                    continue
                size = float(info.get("size", 0) or 0)
                avg = float(info.get("avgPrice", 0) or 0)
                side = (info.get("side") or "").lower()
                if side == "long":
                    long_qty, long_entry = size, avg
                elif side == "short":
                    short_qty, short_entry = size, avg
        except Exception as e:
            logging.warning(f"[breathing] position fetch failed: {e}")
        return long_qty, long_entry, short_qty, short_entry

    def _equity_usd(self) -> Optional[float]:
        try:
            bal = self.retry_api_call(self.exchange.get_futures_balance_bybit)
            return float(bal)
        except Exception as e:
            logging.warning(f"[breathing] equity fetch failed: {e}")
            return None

    def _volatility(self, cfg: Dict) -> float:
        try:
            ohlcv = self.retry_api_call(
                self.exchange.fetch_ohlcv,
                self.symbol,
                timeframe=cfg["candle_timeframe"],
                limit=int(cfg["vol_lookback"]),
            )
            closes = [float(c[4]) for c in ohlcv or []]
            return realized_volatility(closes)
        except Exception as e:
            logging.warning(f"[breathing] candle fetch failed ({e}); using zero vol floor")
            return 0.0

    # ---------- order reconciliation ----------

    @staticmethod
    def _desired_entries(
        side: str,
        prices: List[float],
        qtys: List[float],
        worst_touch: Optional[float],
        current_price: float,
        policy: BreathingPolicy,
    ) -> List[Tuple[str, float, float]]:
        """Entry orders we WANT live right now, respecting the sequential-add rule."""
        desired: List[Tuple[str, float, float]] = []
        for price, qty in zip(prices, qtys):
            adverse_to_rung = (
                (current_price <= price) if side == "long" else (current_price >= price)
            )
            if adverse_to_rung and worst_touch is not None:
                # rung already reached once: require continuation before re-add
                if not sequential_add_allowed(side, price, current_price, policy):
                    continue
            elif adverse_to_rung and worst_touch is None:
                continue  # market already beyond rung with no tracked touch: skip stale rung
            desired.append((side, price, qty))
        return desired

    # ---------- main loop ----------

    def run_single_symbol(self, symbol, rotator_symbols_standardized=None, mfirsi_signal=None, action=None):
        self.symbol = symbol
        cfg = self._breathing_cfg()
        policy = self._build_policy(cfg)

        logging.info(f"[breathing] starting {symbol} with policy: {policy}")

        # leverage caps — never exceed policy ceiling
        try:
            exchange_max = self.exchange.get_current_max_leverage_bybit(symbol)
            lev = min(float(policy.max_leverage), float(exchange_max or policy.max_leverage))
            self.exchange.set_leverage_bybit(int(lev), symbol)
            self.exchange.set_symbol_to_cross_margin(symbol, int(lev))
            logging.info(f"[breathing] leverage set to {lev}x")
        except Exception as e:
            logging.warning(f"[breathing] leverage setup failed: {e}")

        session_start_equity = self._equity_usd()
        halted = False
        prev_long_qty = prev_short_qty = 0.0
        worst_touch: Dict[str, Optional[float]] = {"long": None, "short": None}
        live_orders: Dict[str, dict] = {}   # order_id -> {side, price, qty, reduceOnly}

        while True:
            try:
                if halted:
                    time.sleep(60)
                    continue

                mid = self.retry_api_call(self.exchange.get_current_price, symbol)
                if not mid:
                    time.sleep(cfg["reprice_interval_secs"])
                    continue

                vol = self._volatility(cfg)
                long_qty, long_entry, short_qty, short_entry = self._positions(symbol)

                # ---- drawdown breaker ----
                equity = self._equity_usd()
                if session_start_equity and equity:
                    dd = 1 - (equity / session_start_equity)
                    if dd >= float(cfg["session_max_drawdown_pct"]):
                        logging.error(
                            f"[breathing] SESSION DRAWDOWN {dd:.2%} >= "
                            f"{cfg['session_max_drawdown_pct']:.2%}; flattening and halting."
                        )
                        self.exchange.cancel_all_orders_for_symbol_bybit(symbol)
                        if long_qty:
                            self.close_position(symbol, "long")
                        if short_qty:
                            self.close_position(symbol, "short")
                        halted = True
                        continue

                # ---- touch tracking (for sequential adds / rebound exits) ----
                if long_qty < prev_long_qty and long_qty == 0:
                    worst_touch["long"] = None
                if long_qty > prev_long_qty:
                    worst_touch["long"] = min(
                        [t for t in (worst_touch["long"], long_entry) if t], default=long_entry
                    ) or long_entry
                if short_qty < prev_short_qty and short_qty == 0:
                    worst_touch["short"] = None
                if short_qty > prev_short_qty:
                    worst_touch["short"] = max(
                        [t for t in (worst_touch["short"], short_entry) if t], default=short_entry
                    ) or short_entry
                prev_long_qty, prev_short_qty = long_qty, short_qty

                # ---- compute fresh ladders ("the breath") ----
                long_rungs = rung_depths(mid, vol, policy)
                short_rungs = mirror_prices(long_rungs, mid)
                qtys = [
                    self._round_qty(symbol, q)
                    for q in per_level_qty(float(cfg["budget_usd"]), mid, policy)
                ]
                span = abs(long_rungs[0] - long_rungs[-1])
                tol = float(cfg["requote_tolerance_fraction"]) * span

                desired_orders: List[Tuple[str, float, float, bool]] = []

                if long_qty == 0 and short_qty == 0:
                    # flat: quote both sides' ladders
                    for p, q in zip(long_rungs, qtys):
                        desired_orders.append(("buy", self._round_price(symbol, p), q, False))
                    for p, q in zip(short_rungs, qtys):
                        desired_orders.append(("sell", self._round_price(symbol, p), q, False))
                else:
                    side = "long" if long_qty else "short"
                    pos_qty = long_qty or short_qty
                    avg_entry = long_entry or short_entry

                    # take-profit exit (fee-aware net TP)
                    tp_price = self._round_price(symbol, take_profit_price(side, avg_entry, policy))
                    desired_orders.append((
                        "sell" if side == "long" else "buy",
                        tp_price,
                        self._round_qty(symbol, pos_qty),
                        True,
                    ))

                    # adds: next untouched rung beyond worst touch, gated by continuation rule
                    rungs = long_rungs if side == "long" else short_rungs
                    remaining_budget = float(cfg["budget_usd"]) - pos_qty * avg_entry
                    cap_usd = policy.max_position_usd or float(cfg["budget_usd"]) * 2
                    for i, (p, q) in enumerate(zip(rungs, qtys)):
                        if remaining_budget - q * p <= 0:
                            break
                        if pos_qty * avg_entry + q * p > cap_usd:
                            break
                        touched = (
                            worst_touch[side] is not None
                            and ((side == "long" and p >= worst_touch[side]) or (side == "short" and p <= worst_touch[side]))
                        )
                        if touched and not sequential_add_allowed(side, p, mid, policy):
                            continue
                        if (side == "long" and p >= mid) or (side == "short" and p <= mid):
                            continue  # never place an entry on the wrong side of mid
                        desired_orders.append((
                            "buy" if side == "long" else "sell",
                            self._round_price(symbol, p),
                            q,
                            False,
                        ))

                # ---- reconcile live orders against desired state ----
                def matches(live: dict, want: Tuple) -> bool:
                    o_side, o_price, o_qty, o_reduce = want
                    return (
                        live["side"] == o_side
                        and abs(live["price"] - o_price) <= tol
                        and abs(live["qty"] - o_qty) / max(o_qty, 1e-9) < 0.01
                        and live["reduceOnly"] == o_reduce
                    )

                keep_ids = set()
                for oid, live in list(live_orders.items()):
                    matched = any(matches(live, w) for w in desired_orders)
                    still_open = oid in self._open_order_ids(symbol) if hasattr(self, "_open_order_ids") else True
                    if matched and still_open:
                        keep_ids.add(oid)
                    else:
                        try:
                            self.exchange.cancel_order_bybit(oid, symbol)
                        except Exception:
                            pass
                        live_orders.pop(oid, None)

                have = {(live_orders[o]["side"], round(live_orders[o]["price"], 8)) for o in keep_ids}
                for want in desired_orders:
                    key = (want[0], round(want[1], 8))
                    if key in have:
                        continue
                    try:
                        idx = 1 if want[0] == "buy" else 2
                        order = self.limit_order_bybit(symbol, want[0], want[2], want[1], idx, reduceOnly=want[3])
                        oid = (order or {}).get("id")
                        if oid:
                            live_orders[oid] = {
                                "side": want[0], "price": want[1], "qty": want[2], "reduceOnly": want[3]
                            }
                    except Exception as e:
                        logging.warning(f"[breathing] order placement failed ({want}): {e}")

                time.sleep(float(cfg["reprice_interval_secs"]))

            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(f"[breathing] cycle error: {e}", exc_info=True)
                time.sleep(float(cfg["reprice_interval_secs"]))


def standardize_pos_symbol(raw: str) -> str:
    """Bybit returns e.g. 'BTCUSDT'; strip any suffix defensively."""
    return str(raw or "").split(":")[0].split("-")[0].upper()
