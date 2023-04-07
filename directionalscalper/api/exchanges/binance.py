from __future__ import annotations

import logging
from decimal import Decimal

from directionalscalper.api.exchanges.exchange import Exchange
from directionalscalper.api.exchanges.utils import Intervals
from directionalscalper.core.utils import send_public_request, send_signed_request

log = logging.getLogger(__name__)


class Binance(Exchange):
    def __init__(self):
        super().__init__()
        log.info("Binance initialised")

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
                if symbol["status"] == "TRADING" and symbol["symbol"].endswith("USDT"):
                    tick_size, min_quantity, qty_step = (
                        Decimal(0),
                        Decimal(0),
                        Decimal(0),
                    )
                    leverage = 0
                    if leverages:
                        if symbol in leverages:
                            leverage = leverages[symbol]

                    for filter in symbol["filters"]:
                        if filter["filterType"] == "PRICE_FILTER":
                            tick_size = Decimal(filter["tickSize"])
                        elif filter["filterType"] == "LOT_SIZE":
                            min_quantity = Decimal(filter["minQty"])
                            qty_step = Decimal(filter["stepSize"])
                    symbols_list[symbol["symbol"]] = {
                        "launch": int(symbol["deliveryDate"]),
                        "price_scale": Decimal(symbol["pricePrecision"]),
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
        leverages = {}
        for symbol in raw_json:
            leverages[symbol] = Decimal(0)
            if "brackets" in raw_json[symbol]:
                if len(raw_json[symbol]["brackets"]) > 0:
                    leverages[symbol] = Decimal(
                        raw_json[symbol]["brackets"][0]["initialLeverage"]
                    )
        return leverages

    def get_futures_price(self, symbol: str) -> Decimal:
        self.check_weight()
        params = {"symbol": symbol}
        header, raw_json = send_public_request(
            url=self.futures_api_url,
            url_path="/fapi/v1/ticker/price",
            payload=params,
        )

        if "price" in [*raw_json]:
            return Decimal(raw_json["price"])
        return Decimal(-1.0)

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
                prices[pair["symbol"]] = Decimal(pair["price"])
        return prices

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
                    "open": Decimal(candle[1]),
                    "high": Decimal(candle[2]),
                    "low": Decimal(candle[3]),
                    "close": Decimal(candle[4]),
                    "volume": Decimal(candle[5]),
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
            return [Decimal(raw_json["openInterest"])]
        return oi
