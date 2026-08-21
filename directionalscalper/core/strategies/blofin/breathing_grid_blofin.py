"""BloFin adapter for the breathing-grid strategy (mantis-style venue split)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from directionalscalper.core.strategies.base_strategy import BaseStrategy
from directionalscalper.core.strategies.bybit.gridbased.breathing_grid import (
    BreathingGridCore,
)


class BreathingGridBloFin(BaseStrategy, BreathingGridCore):
    """Breathing grid on BloFin USDT perpetuals.

    Order flow is attributed to the broker code configured on the exchange
    object (``BlofinExchange.DEFAULT_BROKER_ID``, overridable via the
    ``BLOFIN_BROKER_ID`` env var) — ccxt attaches it as ``brokerId`` on every
    order, mirroring mantis's blofin connector.
    """

    def _venue_set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.exchange.set_leverage_blofin(leverage, symbol)
            logging.info(f"[breathing/blofin] leverage set to {leverage}x")
        except Exception as e:
            logging.warning(f"[breathing/blofin] leverage setup failed: {e}")

    def _venue_mid_price(self, symbol: str) -> Optional[float]:
        return self.exchange.get_current_price(symbol)

    def _venue_closes(self, symbol: str, timeframe: str, limit: int) -> List[float]:
        ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        return [float(c[4]) for c in ohlcv or []]

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        return symbol.split("/")[0].split(":")[0].upper()

    def _venue_positions(self, symbol: str) -> Tuple[float, float, float, float]:
        long_qty = short_qty = 0.0
        long_entry = short_entry = 0.0
        base = self._base_symbol(symbol)
        data = self.retry_api_call(self.exchange.get_all_open_positions_blofin)
        for pos in data or []:
            pos_base = self._base_symbol(pos.get("symbol", ""))
            if pos_base != base:
                continue
            size = float(pos.get("size", 0) or 0)
            avg = float(pos.get("avgPrice", 0) or 0)
            side = (pos.get("side") or "").lower()
            if side == "long":
                long_qty, long_entry = size, avg
            elif side == "short":
                short_qty, short_entry = size, avg
        return long_qty, long_entry, short_qty, short_entry

    def _venue_equity_usd(self) -> Optional[float]:
        try:
            bal = self.retry_api_call(self.exchange.get_balance_blofin, "USDT")
            return float(bal) if bal is not None else None
        except Exception as e:
            logging.warning(f"[breathing/blofin] equity fetch failed: {e}")
            return None

    def _venue_place_limit(self, symbol, side, qty, price, reduce_only):
        params = {"reduceOnly": True} if reduce_only else {}
        order = self.exchange.create_limit_order_blofin(
            symbol, side, qty, price, positionIdx=0, params=params
        )
        if isinstance(order, dict) and order.get("error"):
            raise RuntimeError(f"blofin order rejected: {order['error']}")
        return (order or {}).get("id")

    def _venue_cancel_order(self, symbol: str, order_id: str) -> None:
        self.exchange.cancel_order_blofin(order_id, symbol)

    def _venue_open_order_ids(self, symbol: str) -> set:
        try:
            orders = self.retry_api_call(self.exchange.get_all_open_orders_blofin)
            base = self._base_symbol(symbol)
            ids = set()
            for o in orders or []:
                o_symbol = o.get("symbol", "")
                # ccxt returns 'BTC/USDT:USDT'; match on base for safety
                if not o_symbol or self._base_symbol(o_symbol) == base or o_symbol == symbol:
                    ids.add(o.get("id"))
            return ids
        except Exception:
            return set()

    def _venue_flatten(self, symbol: str, side: str, qty: float) -> None:
        close_side = "sell" if side == "long" else "buy"
        order = self.exchange.create_market_order_blofin(symbol, close_side, qty, positionIdx=0)
        if isinstance(order, dict) and order.get("error"):
            raise RuntimeError(f"blofin flatten rejected: {order['error']}")

    # precision helpers via ccxt market metadata
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
