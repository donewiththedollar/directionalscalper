from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import requests  # type: ignore

log = logging.getLogger(__name__)


class InvalidAPI(Exception):
    def __init__(self, message="Invalid Manager setup"):
        self.message = message
        super().__init__(self.message)


class Manager:
    def __init__(
        self,
        api: str = "remote",
        cache_life_ms: int = 10,
        path: Path | None = None,
        url: str = "",
    ):
        log.info("Starting API Manager")
        self.api = api
        self.cache_life_ms = cache_life_ms
        self.path = path
        self.url = url
        self.last_checked = 0.0
        self.data = {}

        if self.api == "remote":
            if len(self.url) < 6:
                self.url = "http://api.tradesimple.xyz/data/quantdata.json"
            self.data = self.get_remote_data()

        elif self.api == "local":
            log.error("local API manager not implemented yet")
            raise InvalidAPI(message="local is not implemented yet")

        else:
            log.error("API must be 'local' or 'remote'")
            raise InvalidAPI(message="API must be 'local' or 'remote'")

        self.last_checked = datetime.now().timestamp()

    def get_data(self):
        if self.api == "remote":
            return self.get_remote_data()
        if self.api == "local":
            pass

    def get_remote_data(self):
        if not self.check_timestamp():
            return self.data
        try:
            response = requests.get(self.url)
            response.raise_for_status()  # Raise an exception if an HTTP error occurs
            self.data = response.json()
        except (requests.exceptions.HTTPError, json.JSONDecodeError) as e:
            log.warning(f"{e}")
        return self.data

    def check_timestamp(self):
        return datetime.now().timestamp() - self.last_checked > self.cache_life_ms

    def get_asset_data(self, symbol: str, data):
        try:
            for asset in data:
                if asset["Assets"] == symbol:
                    return asset
        except Exception as e:
            log.warning(f"{e}")
        return None

    def get_asset_value(self, symbol: str, data, value: str):
        try:
            asset_data = self.get_asset_data(symbol, data)
            if asset_data is not None:
                if value == "Price" and "Price" in asset_data:
                    return asset_data["Price"]
                if value == "1mVol" and "1m 1x Volume (USDT)" in asset_data:
                    return asset_data["1m 1x Volume (USDT)"]
                if value == "5mVol" and "5m 1x Volume (USDT)" in asset_data:
                    return asset_data["5m 1x Volume (USDT)"]
                if value == "1mSpread" and "1m Spread" in asset_data:
                    return asset_data["1m Spread"]
                if value == "5mSpread" and "5m Spread" in asset_data:
                    return asset_data["5m Spread"]
                if value == "15mSpread" and "15m Spread" in asset_data:
                    return asset_data["15m Spread"]
                if value == "30mSpread" and "30m Spread" in asset_data:
                    return asset_data["30m Spread"]
                if value == "Trend" and "Trend" in asset_data:
                    return asset_data["Trend"]
                if value == "Funding" and "Funding" in asset_data:
                    return asset_data["Funding"]
        except Exception as e:
            log.warning(f"{e}")
        return None
