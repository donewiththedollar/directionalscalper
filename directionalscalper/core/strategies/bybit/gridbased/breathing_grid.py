"""Breathing Grid — volume-farming market-making grid, modular across venues.

Ported from the policy layer of mantis-mainacct's cascade_rider breathing
grid, following mantis's structure: a venue-agnostic core executor
(:class:`BreathingGridCore`) plus thin per-exchange adapters that implement a
small set of ``_venue_*`` operations.  Adding a new exchange means writing one
small adapter class, not touching the strategy logic.

Behaviour (see breathing_policy.py for the math):
- post-only-style maker entry ladders on both sides while flat
- single-side campaigns with sequential-add gating (no doubling into wicks)
- fee-aware reduce-only take-profit exits
- ladder re-priced ("breathes") every cycle to fresh volatility-derived depths
- hard caps: leverage, notional, session drawdown breaker (flatten + halt)

Config: ``config.breathing_grid`` dict, all keys optional. UNTESTED ON LIVE
FUNDS — start on testnet or dust-sized budget_usd.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

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


class BreathingGridCore:
    """Venue-agnostic breathing-grid executor.

    Subclasses must implement the ``_venue_*`` hooks for their exchange and
    inherit an __init__ from their strategy base (BybitStrategy / BaseStrategy)
    which sets ``self.exchange``, ``self.config``, ``self.retry_api_call`` etc.
    """

    # ---------- venue hooks (implement in adapters) ----------

    def _venue_set_leverage(self, symbol: str, leverage: int) -> None:
        raise NotImplementedError

    def _venue_mid_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

    def _venue_closes(self, symbol: str, timeframe: str, limit: int) -> List[float]:
        raise NotImplementedError

    def _venue_positions(self, symbol: str) -> Tuple[float, float, float, float]:
        """Return (long_qty, long_entry, short_qty, short_entry)."""
        raise NotImplementedError

    def _venue_equity_usd(self) -> Optional[float]:
        raise NotImplementedError

    def _venue_place_limit(
        self, symbol: str, side: str, qty: float, price: float, reduce_only: bool
    ) -> Optional[str]:
        """Place a limit order; return the venue order id or None."""
        raise NotImplementedError

    def _venue_cancel_order(self, symbol: str, order_id: str) -> None:
        raise NotImplementedError

    def _venue_open_order_ids(self, symbol: str) -> set:
        raise NotImplementedError

    def _venue_flatten(self, symbol: str, side: str, qty: float) -> None:
        """Market-close the given side's position."""
        raise NotImplementedError

    # ---------- config helpers ----------

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

    # ---------- shared helpers ----------

    @staticmethod
    def _round_price(symbol: str, price: float) -> float:
        return round(price, 4)

    @staticmethod
    def _round_qty(symbol: str, qty: float) -> float:
        return round(qty, 3)

    def _volatility(self, cfg: Dict) -> float:
        try:
            closes = self.retry_api_call(
                self._venue_closes,
                self.symbol,
                cfg["candle_timeframe"],
                int(cfg["vol_lookback"]),
            )
            return realized_volatility(closes)
        except Exception as e:
            logging.warning(f"[breathing] candle fetch failed ({e}); using zero vol floor")
            return 0.0

    # ---------- desired-state construction ----------

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
                if not sequential_add_allowed(side, price, current_price, policy):
                    continue
            elif adverse_to_rung and worst_touch is None:
                continue  # market beyond rung with no tracked touch: stale rung, skip
            desired.append((side, price, qty))
        return desired

    # ---------- main loop ----------

    def run_single_symbol(self, symbol, rotator_symbols_standardized=None, mfirsi_signal=None, action=None):
        self.symbol = symbol
        cfg = self._breathing_cfg()
        policy = self._build_policy(cfg)

        logging.info(f"[breathing] starting {symbol} with policy: {policy}")

        try:
            self._venue_set_leverage(symbol, int(min(float(policy.max_leverage), 100)))
        except Exception as e:
            logging.warning(f"[breathing] leverage setup failed: {e}")

        session_start_equity = self._venue_equity_usd()
        halted = False
        prev_long_qty = prev_short_qty = 0.0
        worst_touch: Dict[str, Optional[float]] = {"long": None, "short": None}
        live_orders: Dict[str, dict] = {}   # order_id -> {side, price, qty, reduceOnly}

        while True:
            try:
                if halted:
                    time.sleep(60)
                    continue

                mid = self.retry_api_call(self._venue_mid_price, symbol)
                if not mid:
                    time.sleep(cfg["reprice_interval_secs"])
                    continue

                vol = self._volatility(cfg)
                long_qty, long_entry, short_qty, short_entry = self._venue_positions(symbol)

                # ---- drawdown breaker ----
                equity = self._venue_equity_usd()
                if session_start_equity and equity:
                    dd = 1 - (equity / session_start_equity)
                    if dd >= float(cfg["session_max_drawdown_pct"]):
                        logging.error(
                            f"[breathing] SESSION DRAWDOWN {dd:.2%} >= "
                            f"{cfg['session_max_drawdown_pct']:.2%}; flattening and halting."
                        )
                        for oid in list(live_orders):
                            try:
                                self._venue_cancel_order(symbol, oid)
                            except Exception:
                                pass
                        live_orders.clear()
                        if long_qty:
                            self._venue_flatten(symbol, "long", long_qty)
                        if short_qty:
                            self._venue_flatten(symbol, "short", short_qty)
                        halted = True
                        continue

                # ---- touch tracking ----
                if long_qty < prev_long_qty and long_qty == 0:
                    worst_touch["long"] = None
                if long_qty > prev_long_qty:
                    candidates = [t for t in (worst_touch["long"], long_entry) if t]
                    worst_touch["long"] = min(candidates) if candidates else None
                if short_qty < prev_short_qty and short_qty == 0:
                    worst_touch["short"] = None
                if short_qty > prev_short_qty:
                    candidates = [t for t in (worst_touch["short"], short_entry) if t]
                    worst_touch["short"] = max(candidates) if candidates else None
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

                desired_orders: List[Tuple[str, float, float, bool]] = []  # side, price, qty, reduceOnly

                if long_qty == 0 and short_qty == 0:
                    for p, q in zip(long_rungs, qtys):
                        desired_orders.append(("buy", self._round_price(symbol, p), q, False))
                    for p, q in zip(short_rungs, qtys):
                        desired_orders.append(("sell", self._round_price(symbol, p), q, False))
                else:
                    side = "long" if long_qty else "short"
                    pos_qty = long_qty or short_qty
                    avg_entry = long_entry or short_entry

                    tp_price = self._round_price(symbol, take_profit_price(side, avg_entry, policy))
                    desired_orders.append((
                        "sell" if side == "long" else "buy",
                        tp_price,
                        self._round_qty(symbol, pos_qty),
                        True,
                    ))

                    rungs = long_rungs if side == "long" else short_rungs
                    remaining_budget = float(cfg["budget_usd"]) - pos_qty * avg_entry
                    cap_usd = policy.max_position_usd or float(cfg["budget_usd"]) * 2
                    for p, q in zip(rungs, qtys):
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

                open_ids = self._venue_open_order_ids(symbol)
                keep_ids = set()
                for oid, live in list(live_orders.items()):
                    matched = any(matches(live, w) for w in desired_orders)
                    if matched and oid in open_ids:
                        keep_ids.add(oid)
                    else:
                        try:
                            self._venue_cancel_order(symbol, oid)
                        except Exception:
                            pass
                        live_orders.pop(oid, None)

                have = {(live_orders[o]["side"], round(live_orders[o]["price"], 8)) for o in keep_ids}
                for want in desired_orders:
                    key = (want[0], round(want[1], 8))
                    if key in have:
                        continue
                    try:
                        oid = self._venue_place_limit(symbol, want[0], want[2], want[1], want[3])
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


# ---------------------------------------------------------------------------
# Bybit adapter
# ---------------------------------------------------------------------------

from directionalscalper.core.strategies.bybit.bybit_strategy import BybitStrategy  # noqa: E402


class BreathingGridFutures(BybitStrategy, BreathingGridCore):
    """Breathing grid on Bybit USDT perpetuals."""

    def _venue_set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            exchange_max = self.exchange.get_current_max_leverage_bybit(symbol)
            lev = min(leverage, int(exchange_max or leverage))
            self.exchange.set_leverage_bybit(lev, symbol)
            self.exchange.set_symbol_to_cross_margin(symbol, lev)
            logging.info(f"[breathing/bybit] leverage set to {lev}x")
        except Exception as e:
            logging.warning(f"[breathing/bybit] leverage setup failed: {e}")

    def _venue_mid_price(self, symbol: str) -> Optional[float]:
        return self.exchange.get_current_price(symbol)

    def _venue_closes(self, symbol: str, timeframe: str, limit: int) -> List[float]:
        ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        return [float(c[4]) for c in ohlcv or []]

    def _venue_positions(self, symbol: str) -> Tuple[float, float, float, float]:
        long_qty = short_qty = 0.0
        long_entry = short_entry = 0.0
        data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)
        base = symbol.split("/")[0].split(":")[0].upper()
        for pos in data or []:
            info = pos.get("info", {})
            if str(info.get("symbol", "")).split(":")[0].upper() != base:
                continue
            size = float(info.get("size", 0) or 0)
            avg = float(info.get("avgPrice", 0) or 0)
            side = (info.get("side") or "").lower()
            if side == "long":
                long_qty, long_entry = size, avg
            elif side == "short":
                short_qty, short_entry = size, avg
        return long_qty, long_entry, short_qty, short_entry

    def _venue_equity_usd(self) -> Optional[float]:
        try:
            return float(self.retry_api_call(self.exchange.get_futures_balance_bybit))
        except Exception as e:
            logging.warning(f"[breathing/bybit] equity fetch failed: {e}")
            return None

    def _venue_place_limit(self, symbol, side, qty, price, reduce_only):
        idx = 1 if side == "buy" else 2
        order = self.limit_order_bybit(symbol, side, qty, price, idx, reduceOnly=reduce_only)
        return (order or {}).get("id")

    def _venue_cancel_order(self, symbol: str, order_id: str) -> None:
        self.exchange.cancel_order_bybit(order_id, symbol)

    def _venue_open_order_ids(self, symbol: str) -> set:
        try:
            orders = self.retry_api_call(self.exchange.get_all_open_orders_bybit)
            return {o.get("id") for o in orders or []}
        except Exception:
            return set()

    def _venue_flatten(self, symbol: str, side: str, qty: float) -> None:
        self.close_position(symbol, side)

    # precision helpers use the venue's market metadata
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


def standardize_pos_symbol(raw: str) -> str:
    """Bybit/BloFin position symbols: strip settle suffix defensively."""
    return str(raw or "").split(":")[0].split("-")[0].upper()
