from __future__ import annotations

import logging

from directionalscalper.api.exchanges.exchange import Exchange
from directionalscalper.api.exchanges.utils import Intervals
from directionalscalper.core.utils import send_public_request, send_signed_request

log = logging.getLogger(__name__)


class Binance(Exchange):
    def __init__(self):
        super().__init__()  # Call the parent class's initializer
        log.info("Binance initialised")
        self.funding_cache = {}  # Each instance has its own funding cache
        
    exchange = "binance"
    futures_api_url = "https://fapi.binance.com"
    max_weight = 1000

    def get_futures_symbols(self) -> dict:
        self.check_weight()
        symbols_list = {}
        params: dict = {}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/exchangeInfo",
            payload=params,
        )
        leverages = self.get_max_leverages()
        if "symbols" in raw_json:
            for symbol in raw_json["symbols"]:
                if symbol["status"] == "TRADING":
                    tick_size, min_quantity, qty_step = (
                        float(0),
                        float(0),
                        float(0),
                    )
                    leverage = 0
                    if leverages:
                        # Check if the symbol's name exists in leverages
                        if symbol["symbol"] in leverages:
                            leverage = leverages[symbol["symbol"]]

                    for filter in symbol["filters"]:
                        if filter["filterType"] == "PRICE_FILTER":
                            tick_size = float(filter["tickSize"])
                        elif filter["filterType"] == "LOT_SIZE":
                            min_quantity = float(filter["minQty"])
                            qty_step = float(filter["stepSize"])
                    symbols_list[symbol["symbol"]] = {
                        "launch": int(symbol["deliveryDate"]),
                        "price_scale": float(symbol["pricePrecision"]),
                        "max_leverage": leverage,
                        "tick_size": tick_size,
                        "min_order_qty": min_quantity,
                        "qty_step": qty_step,
                    }
        return symbols_list



    def get_max_leverages(self) -> dict:  # requires authentication
        params: dict = {}
        header, raw_json = send_signed_request(
            base_url=self.futures_api_url,
            http_method="GET",
            url_path="/fapi/v1/exchangeInfo",
            payload=params,
        )
        
        # Log the entire raw_json only once
        #log.debug(f"raw_json: {raw_json}")
        
        leverages = {}
        for symbol_data in raw_json['symbols']:  # iterating over trading symbols specifically
            symbol = symbol_data['symbol']  # extracting symbol name
            leverages[symbol] = float(0)

            if "brackets" in symbol_data:
                if len(symbol_data["brackets"]) > 0:
                    leverages[symbol] = float(
                        symbol_data["brackets"][0]["initialLeverage"]
                    )
        
        return leverages


    def get_futures_volumes(self) -> dict:
        self.check_weight()
        params: dict = {}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/ticker/24hr",
            payload=params,
        )
        volumes = {}
        if len(raw_json) > 0:
            for pair in raw_json:
                volumes[pair["symbol"]] = float(pair["volume"])
        return volumes


    def get_funding_rate(self, symbol: str) -> float:
        # Get current timestamp
        current_time = datetime.now()

        # Check if the symbol is in the shared cache
        with self.lock:  # Ensure that access to the shared cache is thread-safe
            cached_data = self.funding_cache.get(symbol, None)
            if cached_data:
                cached_time, cached_rate = cached_data
                # If cached data is less than the cache duration, return the cached rate
                if (current_time - cached_time).total_seconds() < self.FUNDING_CACHE_DURATION.total_seconds():
                    return cached_rate

        # Fetch new rate if not in cache or if older than the cache duration
        self.check_weight()
        params = {"symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/fundingRate",
            payload=params,
        )
        funding_rate = 0.0
        if len(raw_json) > 0:
            funding_rate = float(raw_json[0]["fundingRate"])

        # Cache the newly fetched rate with the current timestamp in the shared cache
        with self.lock:  # Ensure that access to the shared cache is thread-safe
            self.funding_cache[symbol] = (current_time, funding_rate)

        return funding_rate

    # def get_funding_rate(self, symbol: str) -> float:
    #     self.check_weight()
    #     params = {"symbol": symbol}
    #     header, raw_json = send_public_request(
    #         url=self.futures_api_url,
    #         url_path="/fapi/v1/fundingRate",
    #         payload=params,
    #     )
    #     if len(raw_json) > 0:
    #         return float(raw_json[0]["fundingRate"])
    #     return 0.0

    def get_open_interest(self, symbol: str) -> float:
        self.check_weight()
        params = {"symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/openInterest",
            payload=params,
        )
        if "openInterest" in raw_json:
            return float(raw_json["openInterest"])
        return 0.0

    def get_futures_price(self, symbol: str) -> float:
        self.check_weight()
        params = {"symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/ticker/price",
            payload=params,
        )

        if "price" in [*raw_json]:
            return float(raw_json["price"])
        return float(-1.0)

    def get_futures_prices(self) -> dict:
        self.check_weight()
        params: dict = {}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/ticker/price",
            payload=params,
        )
        prices = {}
        if len(raw_json) > 0:
            for pair in raw_json:
                prices[pair["symbol"]] = float(pair["price"])
        return prices

    def get_futures_volumes(self) -> dict:
        self.check_weight()
        params: dict = {}

        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/ticker/24hr",
            payload=params,
        )
        volumes = {}
        if len(raw_json) > 0:
            for pair in raw_json:
                volumes[pair["symbol"]] = float(pair["volume"])
        return volumes

    def get_futures_kline(
        self,
        symbol: str,
        interval: Intervals = Intervals.ONE_DAY,
        limit: int = 200,
    ) -> list:
        self.check_weight()

        params = {"symbol": symbol, "limit": limit, "interval": interval}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/klines",
            payload=params,
        )

        if len(raw_json) > 0:
            return [
                {
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                }
                for candle in raw_json
            ]
        return []

    def get_funding_rate(self, symbol: str) -> float:
        self.check_weight()
        params = {"symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/fundingRate",
            payload=params,
        )
        if len(raw_json) > 0:
            return float(raw_json[0]["fundingRate"])
        return 0.0

    def get_open_interest(
        self, symbol: str, interval: Intervals = Intervals.ONE_DAY, limit: int = 200
    ) -> list:
        self.check_weight()
        oi: list = []
        params = {"symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/openInterest",
            payload=params,
        )
        if len(raw_json) > 0:
            return [float(raw_json["openInterest"])]
        return oi
