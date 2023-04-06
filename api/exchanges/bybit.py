from __future__ import annotations

import logging
from decimal import Decimal

from exchanges.exchange import Exchange
from exchanges.utils import Intervals, get_api_data

log = logging.getLogger(__name__)


class Bybit(Exchange):
    def __init__(self):
        super().__init__()
        log.info("Bybit initialised")

    exchange = "bybit"
    futures_api_url = "https://api.bybit.com"
    max_weight = 1200

    def get_futures_symbols(self) -> dict:
        self.check_weight()
        symbols_list = {}
        raw_json = get_api_data(
            url=self.futures_api_url,
            endpoint="/v5/market/instruments-info?category=linear",
        )
        if "result" in raw_json:
            if "list" in raw_json["result"]:
                for symbol in raw_json["result"]["list"]:
                    if symbol["status"] == "Trading" and symbol["symbol"].endswith(
                        "USDT"
                    ):
                        symbols_list[symbol["symbol"]] = {
                            "launch": int(symbol["launchTime"]),
                            "price_scale": Decimal(symbol["priceScale"]),
                            "max_leverage": Decimal(
                                symbol["leverageFilter"]["maxLeverage"]
                            ),
                            "tick_size": Decimal(symbol["priceFilter"]["tickSize"]),
                            "min_order_qty": Decimal(
                                symbol["lotSizeFilter"]["minOrderQty"]
                            ),
                            "qty_step": Decimal(symbol["lotSizeFilter"]["qtyStep"]),
                        }
        return symbols_list

    def get_futures_price(self, symbol: str) -> Decimal:
        self.check_weight()
        raw_json = get_api_data(
            url=self.futures_api_url,
            endpoint=f"/v5/market/tickers?category=linear&symbol={symbol}",
        )
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                if len(raw_json["result"]["list"]) > 0:
                    if "lastPrice" in [*raw_json["result"]["list"][0]]:
                        return Decimal(raw_json["result"]["list"][0]["lastPrice"])
        return Decimal(-1.0)

    def get_futures_prices(self) -> list:
        self.check_weight()
        raw_json = get_api_data(
            url=self.futures_api_url,
            endpoint="/v5/market/tickers?category=linear",
        )
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                return [
                    {"symbol": pair["symbol"], "price": Decimal(pair["lastPrice"])}
                    for pair in raw_json["result"]["list"]
                ]
        return []

    def get_futures_kline(
        self,
        symbol: str,
        interval: Intervals = Intervals.ONE_DAY,
        limit: int = 200,
    ) -> list:
        self.check_weight()
        custom_intervals = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": "D",
            "1w": "W",
        }

        raw_json = get_api_data(
            url=self.futures_api_url,
            endpoint=f"/v5/market/kline?category=linear&interval={custom_intervals[interval]}&limit={limit}&symbol={symbol}",
        )

        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                if len(raw_json["result"]["list"]) > 0:
                    return [
                        {
                            "timestamp": int(candle[0]),
                            "open": Decimal(candle[1]),
                            "high": Decimal(candle[2]),
                            "low": Decimal(candle[3]),
                            "close": Decimal(candle[4]),
                            "volume": Decimal(candle[5]),
                        }
                        for candle in raw_json["result"]["list"]
                    ]
        return []

    def get_funding_rate(self, symbol: str) -> float:
        self.check_weight()
        funding = 0.0
        raw_json = get_api_data(
            url=self.futures_api_url,
            endpoint=f"/v5/market/funding/history?category=linear&symbol={symbol}",
        )
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                if len(raw_json["result"]["list"]) > 0:
                    funding = float(raw_json["result"]["list"][0]["fundingRate"])
        return funding

    def get_open_interest(
        self, symbol: str, interval: Intervals = Intervals.ONE_DAY, limit: int = 200
    ) -> list:
        self.check_weight()
        oi = []
        custom_intervals = {
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        raw_json = get_api_data(
            url=self.futures_api_url,
            endpoint=f"/v5/market/funding/history?category=linear&interval={custom_intervals[interval]}&limit={limit}&symbol={symbol}",
        )
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                for item in raw_json["result"]["list"]:
                    oi.append(Decimal(item["openInterest"]))
        return oi
