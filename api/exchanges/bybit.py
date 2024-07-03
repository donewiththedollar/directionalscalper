from __future__ import annotations

import logging
import time
from datetime import datetime

from api.exchanges.exchange import Exchange
from api.exchanges.utils import Intervals
from directionalscalper.core.utils import send_public_request

log = logging.getLogger(__name__)


class Bybit(Exchange):
    def __init__(self):
        super().__init__()  # Call the parent class's initializer
        log.info("Bybit initialised")
        self.funding_rates_cache = {}  # Each instance has its own funding rates cache

    exchange = "bybit"
    futures_api_url = "https://api.bybit.com"
    max_weight = 1200

    def get_futures_symbols(self) -> dict:
        self.check_weight()
        symbols_list = {}
        params = {"category": "linear"}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/v5/market/instruments-info",
            payload=params,
        )
        if "result" in raw_json:
            if "list" in raw_json["result"]:
                for symbol in raw_json["result"]["list"]:
                    if symbol["status"] == "Trading":
                        symbols_list[symbol["symbol"]] = {
                            "launch": int(symbol["launchTime"]),
                            "price_scale": float(symbol["priceScale"]),
                            "max_leverage": float(
                                symbol["leverageFilter"]["maxLeverage"]
                            ),
                            "tick_size": float(symbol["priceFilter"]["tickSize"]),
                            "min_order_qty": float(
                                symbol["lotSizeFilter"]["minOrderQty"]
                            ),
                            "qty_step": float(symbol["lotSizeFilter"]["qtyStep"]),
                        }
        return symbols_list

    def get_futures_price(self, symbol: str) -> float:
        self.check_weight()
        params = {"category": "linear", "symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url, url_path="/v5/market/tickers", payload=params
        )
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                if len(raw_json["result"]["list"]) > 0:
                    if "lastPrice" in [*raw_json["result"]["list"][0]]:
                        return float(raw_json["result"]["list"][0]["lastPrice"])
        return float(-1.0)

    def get_futures_prices(self) -> dict:
        self.check_weight()
        params = {"category": "linear"}
        header, raw_json = send_public_request(
            url=self.futures_api_url, url_path="/v5/market/tickers", payload=params
        )
        prices = {}
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                for pair in raw_json["result"]["list"]:
                    prices[pair["symbol"]] = float(pair["lastPrice"])
        return prices

    def get_futures_volumes(self) -> dict:
        self.check_weight()
        params = {"category": "linear"}
        header, raw_json = send_public_request(
            url=self.futures_api_url, url_path="/v5/market/tickers", payload=params
        )
        volumes = {}
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                for pair in raw_json["result"]["list"]:
                    volumes[pair["symbol"]] = float(pair["volume24h"])
        return volumes

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

        params = {
            "category": "linear",
            "symbol": symbol,
            "limit": limit + 1,
            "interval": custom_intervals[interval],
        }
        header, raw_json = send_public_request(
            url=self.futures_api_url, url_path="/v5/market/kline", payload=params
        )

        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                if len(raw_json["result"]["list"]) > 1:  # Ensuring there's more than one candlestick
                    converted_data = [
                        {
                            "timestamp": int(candle[0]),
                            "open": float(candle[1]),
                            "high": float(candle[2]),
                            "low": float(candle[3]),
                            "close": float(candle[4]),
                            "volume": float(candle[5]),
                        }
                        for candle in raw_json["result"]["list"][1:]  # Skipping the first candlestick
                    ]
                    reversed_data = converted_data[::-1]
                    return reversed_data
        return []

    def get_funding_rate(self, symbol: str) -> float:
        # Get current timestamp
        current_time = datetime.now()

        # Check if the symbol is in the cache
        cached_data = self.funding_rates_cache.get(symbol, None)
        if cached_data:
            cached_time, cached_rate = cached_data
            # If cached data is less than 3 hours old, return the cached rate
            if (current_time - cached_time).total_seconds() < (3 * 3600):  # 3 hours in seconds
                return cached_rate

        # Fetch new rate if not in cache or if older than 3 hours
        self.check_weight()
        funding = 0.0
        params = {"category": "linear", "symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/v5/market/tickers",
            payload=params,
        )
        if "result" in raw_json and "list" in raw_json["result"] and raw_json["result"]["list"]:
            funding = float(raw_json["result"]["list"][0]["fundingRate"])

        # Cache the newly fetched rate with the current timestamp
        self.funding_rates_cache[symbol] = (current_time, funding)

        return funding


    # def get_funding_rate(self, symbol: str) -> float:
    #     # Get current timestamp
    #     current_time = datetime.now()

    #     # Check if the symbol is in the shared cache
    #     with self.lock:  # Ensure that access to the shared cache is thread-safe
    #         cached_data = self.funding_cache.get(symbol, None)
    #         if cached_data:
    #             cached_time, cached_rate = cached_data
    #             # If cached data is less than 3 hours old, return the cached rate
    #             if (current_time - cached_time).total_seconds() < (3 * 3600):  # 3 hours in seconds
    #                 return cached_rate

    #     # Fetch new rate if not in cache or if older than 3 hours
    #     self.check_weight()
    #     funding = 0.0
    #     params = {"category": "linear", "symbol": symbol}
    #     header, raw_json = send_public_request(
    #         url=self.futures_api_url,
    #         url_path="/v5/market/tickers",
    #         payload=params,
    #     )
    #     if "result" in raw_json and "list" in raw_json["result"] and raw_json["result"]["list"]:
    #         funding = float(raw_json["result"]["list"][0]["fundingRate"])

    #     # Cache the newly fetched rate with the current timestamp in the shared cache
    #     with self.lock:  # Ensure that access to the shared cache is thread-safe
    #         self.funding_cache[symbol] = (current_time, funding)

    #     return funding

    # def get_funding_rate(self, symbol: str) -> float:
    #     # Get current timestamp
    #     current_time = datetime.now()

    #     # Check if the symbol is in the shared cache
    #     cached_data = funding_cache.get(symbol, None)
    #     if cached_data:
    #         cached_time, cached_rate = cached_data
    #         # If cached data is less than 3 hours old, return the cached rate
    #         if (current_time - cached_time).total_seconds() < (3 * 3600):  # 3 hours in seconds
    #             return cached_rate

    #     # Fetch new rate if not in cache or if older than 3 hours
    #     self.check_weight()
    #     funding = 0.0
    #     params = {"category": "linear", "symbol": symbol}
    #     header, raw_json = send_public_request(
    #         url=self.futures_api_url,
    #         url_path="/v5/market/tickers",
    #         payload=params,
    #     )
    #     if "result" in raw_json and "list" in raw_json["result"] and raw_json["result"]["list"]:
    #         funding = float(raw_json["result"]["list"][0]["fundingRate"])

    #     # Cache the newly fetched rate with the current timestamp in the shared cache
    #     funding_cache[symbol] = (current_time, funding)

    #     return funding


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

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": custom_intervals[interval],
            "limit": limit,
        }
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/v5/market/open-interest",
            payload=params,
        )
        if "result" in [*raw_json]:
            if "list" in [*raw_json["result"]]:
                for item in raw_json["result"]["list"]:
                    oi.append(float(item["openInterest"]))
        return oi
