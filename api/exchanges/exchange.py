from __future__ import annotations

import logging
import time
from decimal import Decimal

from exchanges.utils import Intervals

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

    def get_futures_price(self, symbol: str) -> Decimal:
        return Decimal(-1.0)

    def get_futures_prices(self) -> list:
        return []

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
