from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

import requests  # type: ignore

from directionalscalper.core.utils import send_public_request


log = logging.getLogger(__name__)


class InvalidAPI(Exception):
    def __init__(self, message="Invalid Manager setup"):
        self.message = message
        super().__init__(self.message)

class Manager:
    def __init__(
        self,
        exchange,
        api: str = "remote",
        cache_life_seconds: int = 10,
        path: Path | None = None,
        url: str = "",
    ):
        self.exchange = exchange
        log.info("Starting API Manager")
        self.api = api
        self.cache_life_seconds = cache_life_seconds
        self.path = path
        self.url = url
        self.last_checked = 0.0
        self.data = {}

        if self.api == "remote":
            log.info("API manager mode: remote")
            if len(self.url) < 6:
                self.url = "http://api.tradesimple.xyz/data/quantdatav2.json"
            log.info(f"Remote API URL: {self.url}")
            self.data = self.get_remote_data()

        elif self.api == "local":
            if len(str(self.path)) < 6:
                self.path = Path("data", "quantdatav2.json")
            log.info(f"Local API directory: {self.path}")
            self.data = self.get_local_data()

        else:
            log.error("API must be 'local' or 'remote'")
            raise InvalidAPI(message="API must be 'local' or 'remote'")

        self.update_last_checked()

    def update_last_checked(self):
        self.last_checked = datetime.now().timestamp()

    def get_data(self):
        if self.api == "remote":
            return self.get_remote_data()
        if self.api == "local":
            return self.get_local_data()

    def get_local_data(self):
        if not self.check_timestamp():
            return self.data
        if not self.path.is_file():
            raise InvalidAPI(message=f"{self.path} is not a file")
        f = open(self.path)
        try:
            self.data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ERROR: Invalid JSON: {exc.msg}, line {exc.lineno}, column {exc.colno}"
            )
        self.update_last_checked()
        return self.data

    # def get_remote_data(self):
    #     if not self.check_timestamp():
    #         return self.data
    #     header, raw_json = send_public_request(url=self.url)
    #     self.data = raw_json
    #     self.update_last_checked()
    #     return self.data

    def get_remote_data(self):
            if not self.check_timestamp():
                return self.data
            while True:  # Keep trying until a successful request is made
                try:
                    header, raw_json = send_public_request(url=self.url)
                    self.data = raw_json
                    break  # if the request was successful, break the loop
                except requests.exceptions.RequestException as e:
                    log.error(f"Request failed: {e}, retrying...")
                except json.decoder.JSONDecodeError as e:
                    log.error(f"Failed to parse JSON: {e}, retrying...")
                except Exception as e:
                    log.error(f"Unexpected error occurred: {e}, retrying...")
                finally:
                    self.update_last_checked()
            return self.data

    def check_timestamp(self):
        return datetime.now().timestamp() - self.last_checked > self.cache_life_seconds

    def get_asset_data(self, symbol: str, data):
        try:
            for asset in data:
                if asset["Asset"] == symbol:
                    return asset
        except Exception as e:
            log.warning(f"{e}")
        return None

    def get_1m_moving_averages(self, symbol, num_bars=20):
        return self.exchange.get_moving_averages(symbol, "1m", num_bars)
    
    def get_5m_moving_averages(self, symbol, num_bars=20):
        return self.exchange.get_moving_averages(symbol, "5m", num_bars)

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
                if value == "1hSpread" and "1h Spread" in asset_data:
                    return asset_data["1h Spread"]
                if value == "4hSpread" and "4h Spread" in asset_data:
                    return asset_data["4h Spread"]
                if value == "Trend" and "Trend" in asset_data:
                    return asset_data["Trend"]
                if value == "Funding" and "Funding" in asset_data:
                    return asset_data["Funding"]
                if value == "MFI" and "MFI" in asset_data:
                    return asset_data["MFI"]
                if value == "ERI Bull Power" in asset_data:
                    return asset_data["ERI Bull Power"]
                if value == "ERI Bear Power" in asset_data:
                    return asset_data["ERI Bear Power"]
                if value == "ERI Trend" in asset_data:
                    return asset_data["ERI Trend"]
        except Exception as e:
            log.warning(f"{e}")
        return None