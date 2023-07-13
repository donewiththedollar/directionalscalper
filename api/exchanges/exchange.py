from __future__ import annotations

import logging
import time

from directionalscalper.api.exchanges.utils import Intervals

log = logging.getLogger(__name__)


class Exchange:
    def __init__(self):
        pass

    exchange: str | None = None
    futures_api_url: str | None = None
    weight: int = 0
    max_weight: int = 100

    def check_api_permissions(self, account: dict) -> None:
        pass

    def check_weight(self) -> None:
        if self.weight >= self.max_weight:
            log.info(
                f"Weight {self.weight} is greater than {self.max_weight}, sleeping for 60 seconds"
            )
            time.sleep(60)

    def update_weight(self, weight: int) -> None:
        self.weight = weight

    def get_futures_symbols(self) -> dict:
        return {}

    def get_futures_price(self, symbol: str) -> float:
        return float(-1.0)

    def get_futures_prices(self) -> dict:
        return {}

    def get_futures_volumes(self) -> dict:
        return {}

    def get_futures_kline(
        self,
        symbol: str,
        interval: Intervals = Intervals.ONE_DAY,
        limit: int = 500,
    ) -> list:
        return []

    def get_funding_rate(self, symbol) -> float:
        return 0.0

    def get_open_interest(
        self, symbol: str, interval: Intervals = Intervals.ONE_DAY, limit: int = 200
    ) -> list:
        return []

    def get_symbol_info(self, symbol: str, info: str):
        symbols_info = self.get_futures_symbols()

        if symbol in symbols_info:
            if info in symbols_info[symbol]:
                return symbols_info[symbol][info]
            raise ValueError(f"{info} not found for {symbol}")
        raise ValueError(f"Symbol {symbol} not found in the symbols list.")
