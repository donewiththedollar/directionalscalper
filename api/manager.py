from __future__ import annotations
from threading import Thread, Lock

import time
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

import requests  # type: ignore

from directionalscalper.core.utils import send_public_request


log = logging.getLogger(__name__)

from time import sleep

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

    def get_auto_rotate_symbols(self, min_qty_threshold: float = None, whitelist: list = None, blacklist: list = None, max_symbols: int = 12, max_retries: int = 10, delay_between_retries: int = 30):
        symbols = []
        url = "http://api.tradesimple.xyz/data/rotatorsymbols.json"

        for retry in range(max_retries):
            try:
                log.debug(f"Sending request to {url} (Attempt: {retry + 1})")
                header, raw_json = send_public_request(url=url)
                
                if isinstance(raw_json, list):
                    log.debug(f"Received {len(raw_json)} assets from API")
                    
                    for asset in raw_json:
                        symbol = asset.get("Asset", "")
                        min_qty = asset.get("Min qty", 0)
                        log.debug(f"Processing symbol {symbol} with min_qty {min_qty}")

                        # Only consider the whitelist if it's not empty or None
                        if whitelist and symbol not in whitelist and len(whitelist) > 0:
                            log.debug(f"Skipping {symbol} as it's not in whitelist")
                            continue

                        # Consider the blacklist regardless of whether it's empty or not
                        if blacklist and symbol in blacklist:
                            log.debug(f"Skipping {symbol} as it's in blacklist")
                            continue

                        if min_qty_threshold is None or min_qty <= min_qty_threshold:
                            symbols.append(symbol)

                        # Break the loop if we've reached the maximum number of allowed symbols
                        if len(symbols) >= max_symbols:
                            break

                    log.debug(f"Returning {len(symbols)} symbols")
                    return symbols

                else:
                    log.error("Unexpected data format. Expected a list of assets.")
                    if retry < max_retries - 1:
                        sleep(delay_between_retries)
                    else:
                        return []

            except requests.exceptions.RequestException as e:
                log.error(f"Request failed: {e}")
            except json.decoder.JSONDecodeError as e:
                log.error(f"Failed to parse JSON: {e}")
            except Exception as e:
                log.error(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                sleep(delay_between_retries)
        
        # Return empty list if all retries fail
        return []

    def get_symbols(self):
        url = "http://api.tradesimple.xyz/data/rotatorsymbols.json"
        try:
            header, raw_json = send_public_request(url=url)
            if isinstance(raw_json, list):
                return raw_json
            else:
                log.error("Unexpected data format. Expected a list of symbols.")
                return []
        except requests.exceptions.RequestException as e:
            log.error(f"Request failed: {e}")
            return []
        except json.decoder.JSONDecodeError as e:
            log.error(f"Failed to parse JSON: {e}")
            return []
        except Exception as e:
            log.error(f"Unexpected error occurred: {e}")
            return []

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
                if value == "1hVol" and "1m 1h Volume (USDT)" in asset_data:
                    return asset_data["1h 1x Volume (USDT)"]
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

    def get_api_data(self, symbol):
        data = self.get_data()
        api_data = {
            '1mVol': self.get_asset_value(symbol, data, "1mVol"),
            '1hVol': self.get_asset_value(symbol, data, "1hVol"),
            '1mSpread': self.get_asset_value(symbol, data, "1mSpread"),
            '5mSpread': self.get_asset_value(symbol, data, "5mSpread"),
            '30mSpread': self.get_asset_value(symbol, data, "30mSpread"),
            '1hSpread': self.get_asset_value(symbol, data, "1hSpread"),
            '4hSpread': self.get_asset_value(symbol, data, "4hSpread"),
            'Trend': self.get_asset_value(symbol, data, "Trend"),
            'MFI': self.get_asset_value(symbol, data, "MFI"),
            'ERI Trend': self.get_asset_value(symbol, data, "ERI Trend"),
            'Symbols': self.get_symbols()
        }
        return api_data