from __future__ import annotations
from threading import Thread, Lock

import fnmatch
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

import requests  # type: ignore

from directionalscalper.core.utils import send_public_request
from directionalscalper.core.strategies.logger import Logger

logging = Logger(logger_name="Manager", filename="Manager.log", stream=True) 

#log = logging.getLogger(__name__)

from time import sleep

class InvalidAPI(Exception):
    def __init__(self, message="Invalid Manager setup"):
        self.message = message
        super().__init__(self.message)

class Manager:
    def __init__(
        self,
        exchange,
        exchange_name: str = 'bybit',  # Defaulting to 'bybit'
        data_source_exchange: str = 'bybit',
        api: str = "remote",
        cache_life_seconds: int = 60,
        asset_value_cache_life_seconds: int = 60,
        path: Path | None = None,
        url: str = "",
    ):
        self.exchange = exchange
        self.exchange_name = exchange_name
        self.data_source_exchange = data_source_exchange
        logging.info("Starting API Manager")
        
        self.api = api
        self.cache_life_seconds = cache_life_seconds
        self.asset_value_cache_life_seconds = asset_value_cache_life_seconds
        
        self.path = path
        self.url = url

        # Initialize the time when data was last checked
        self.last_checked = 0.0
        
        # Initialize the main data cache and its expiry
        self.data = {}
        self.data_cache_expiry = datetime.now() - timedelta(seconds=self.cache_life_seconds)
        
        # Initialize the asset value cache and its expiry
        self.asset_value_cache = {}
        self.asset_value_cache_expiry = datetime.now() - timedelta(seconds=self.asset_value_cache_life_seconds)

        # Attributes for caching
        self.rotator_symbols_cache = None
        self.rotator_symbols_cache_expiry = datetime.now() - timedelta(seconds=1)  # Initialize to an old timestamp to force first fetch

        # Initialize the API data cache and its expiry
        self.api_data_cache = None
        self.api_data_cache_expiry = datetime.now() - timedelta(seconds=self.cache_life_seconds)

        # Attributes for 'everything' data cache
        self.everything_cache = None
        self.everything_cache_expiry = datetime.now() - timedelta(seconds=1)  # Initialize to an old timestamp to force first fetch

        # # Attributes for caching API data
        # self.api_data_cache = None
        # self.api_data_cache_expiry = datetime.now() - timedelta(seconds=1)

        if self.api == "remote":
            logging.info("API manager mode: remote")
            if len(self.url) < 6:
                # Adjusting the default URL based on the exchange_name
                self.url = f"https://api.quantumvoid.org/volumedata/quantdatav2_{self.exchange_name.replace('_', '')}.json"
            logging.info(f"Remote API URL: {self.url}")
            self.data = self.get_remote_data()

        elif self.api == "local":
            # You might also want to consider adjusting the local path based on the exchange_name in the future.
            if len(str(self.path)) < 6:
                self.path = Path("volumedata", f"quantdatav2_{self.exchange_name.replace('_', '')}.json")
            logging.info(f"Local API directory: {self.path}")
            self.data = self.get_local_data()

        else:
            logging.error("API must be 'local' or 'remote'")
            raise InvalidAPI(message="API must be 'local' or 'remote'")

        self.update_last_checked()

    def is_everything_cache_expired(self):
        return datetime.now() > self.everything_cache_expiry

    def get_everything(self, min_qty_threshold: float = None, blacklist: list = None, whitelist: list = None, max_usd_value: float = None, max_retries: int = 5):
        if self.everything_cache and not self.is_everything_cache_expired():
            return self.everything_cache

        symbols = []
        url = f"https://api.quantumvoid.org/volumedata/everything_{self.exchange_name.replace('_', '')}.json"

        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(58, delay)  # cap the delay to 58 seconds

            try:
                logging.info(f"Sending request to {url} (Attempt: {retry + 1})")
                header, raw_json = send_public_request(url=url)

                if isinstance(raw_json, list):
                    logging.info(f"Received {len(raw_json)} assets from API")

                    for asset in raw_json:
                        symbol = asset.get("Asset", "")
                        min_qty = asset.get("Min qty", 0)
                        usd_price = asset.get("Price", float('inf'))

                        if blacklist and any(fnmatch.fnmatch(symbol, pattern) for pattern in blacklist):
                            logging.debug(f"Skipping {symbol} as it's in blacklist")
                            continue

                        if whitelist:
                            logging.debug(f"Whitelist provided: {whitelist}")
                            if symbol not in whitelist:
                                logging.debug(f"Skipping {symbol} as it's not in whitelist")
                                continue

                        # Check against the max_usd_value, if provided
                        if max_usd_value is not None and usd_price > max_usd_value:
                            logging.debug(f"Skipping {symbol} as its USD price {usd_price} is greater than the max allowed {max_usd_value}")
                            continue

                        logging.debug(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if min_qty_threshold is None or min_qty <= min_qty_threshold:
                            symbols.append(symbol)

                    logging.info(f"Returning {len(symbols)} symbols")

                    # If successfully fetched, update the cache and its expiry time
                    if symbols:
                        self.everything_cache = symbols
                        self.everything_cache_expiry = datetime.now() + timedelta(seconds=self.cache_life_seconds)

                    return symbols

                else:
                    logging.warning("Unexpected data format. Expected a list of assets.")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Request failed: {e}")
            except json.decoder.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON: {e}. Response: {raw_json}")
            except Exception as e:
                logging.warning(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                time.sleep(delay)

        # Return cached symbols if all retries fail
        logging.warning(f"Couldn't fetch every symbols after {max_retries} attempts. Using cached symbols.")
        return self.everything_cache or []


    # def get_everything(self, min_qty_threshold: float = None, blacklist: list = None, whitelist: list = None, max_usd_value: float = None, max_retries: int = 5):
    #     if self.everything_cache and not self.is_everything_cache_expired(self.everything_cache_expiry):
    #         return self.everything_cache

    #     symbols = []
    #     url = f"https://api.quantumvoid.org/volumedata/everything_{self.exchange_name.replace('_', '')}.json"

    #     for retry in range(max_retries):
    #         delay = 2**retry  # exponential backoff
    #         delay = min(58, delay)  # cap the delay to 58 seconds

    #         try:
    #             logging.info(f"Sending request to {url} (Attempt: {retry + 1})")
    #             header, raw_json = send_public_request(url=url)

    #             if isinstance(raw_json, list):
    #                 logging.info(f"Received {len(raw_json)} assets from API")

    #                 for asset in raw_json:
    #                     symbol = asset.get("Asset", "")
    #                     min_qty = asset.get("Min qty", 0)
    #                     usd_price = asset.get("Price", float('inf'))

    #                     logging.info(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

    #                     if blacklist and any(fnmatch.fnmatch(symbol, pattern) for pattern in blacklist):
    #                         logging.info(f"Skipping {symbol} as it's in blacklist")
    #                         continue

    #                     if whitelist:
    #                         logging.info(f"Whitelist provided: {whitelist}")
    #                         if symbol not in whitelist:
    #                             logging.info(f"Skipping {symbol} as it's not in whitelist")
    #                             continue

    #                     # Check against the max_usd_value, if provided
    #                     if max_usd_value is not None and usd_price > max_usd_value:
    #                         logging.info(f"Skipping {symbol} as its USD price {usd_price} is greater than the max allowed {max_usd_value}")
    #                         continue

    #                     if min_qty_threshold is None or min_qty <= min_qty_threshold:
    #                         symbols.append(symbol)

    #                 logging.info(f"Returning {len(symbols)} symbols")

    #                 # If successfully fetched, update the cache and its expiry time
    #                 if symbols:
    #                     self.everything_cache = symbols
    #                     self.everything_cache_expiry = datetime.now() + timedelta(seconds=self.cache_life_seconds)

    #                 return symbols

    #             else:
    #                 logging.warning("Unexpected data format. Expected a list of assets.")

    #         except requests.exceptions.RequestException as e:
    #             logging.warning(f"Request failed: {e}")
    #         except json.decoder.JSONDecodeError as e:
    #             logging.warning(f"Failed to parse JSON: {e}. Response: {raw_json}")
    #         except Exception as e:
    #             logging.warning(f"Unexpected error occurred: {e}")

    #         # Wait before the next retry
    #         if retry < max_retries - 1:
    #             time.sleep(delay)

    #     # Return cached symbols if all retries fail
    #     logging.warning(f"Couldn't fetch everything symbols after {max_retries} attempts. Using cached symbols.")
    #     return self.everything_cache or []


        
    def update_last_checked(self):
        self.last_checked = datetime.now().timestamp()

    def fetch_data_from_url(self, url, max_retries: int = 5):
        current_time = datetime.now()
        if current_time <= self.data_cache_expiry:
            return self.data
        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(60, delay)  # cap the delay to 60 seconds
            try:
                header, raw_json = send_public_request(url=url)
                # Update the data cache and its expiry time
                self.data = raw_json
                self.data_cache_expiry = current_time + timedelta(seconds=self.cache_life_seconds)
                return raw_json
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {e}")
            except json.decoder.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON: {e}")
            except Exception as e:
                logging.error(f"Unexpected error occurred: {e}")
            
            # Wait before the next retry
            if retry < max_retries - 1:
                sleep(delay)

        # Return cached data if all retries fail
        return self.data
    
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

    def is_cache_expired(self):
        """Checks if the cache has expired based on cache_life_seconds."""
        return datetime.now() > self.rotator_symbols_cache_expiry

    def get_all_possible_symbols(self, max_retries: int = 5):
        url = f"https://api.quantumvoid.org/volumedata/quantdatav2_{self.exchange_name.replace('_', '')}.json"
        
        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(58, delay)  # cap the delay to 30 seconds

            try:
                logging.info(f"Sending request to {url} (Attempt: {retry + 1})")
                header, raw_json = send_public_request(url=url)
                
                if isinstance(raw_json, list):
                    logging.info(f"Received {len(raw_json)} symbols from API")
                    
                    symbols = [asset.get("Asset", "") for asset in raw_json if "Asset" in asset]
                    logging.info(f"Returning {len(symbols)} symbols")
                    return symbols
                else:
                    logging.warning("Unexpected data format. Expected a list of symbols.")
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request failed: {e}")
            except json.decoder.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON: {e}. Response: {raw_json}")
            except Exception as e:
                logging.warning(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                sleep(delay)
        
        # If all retries fail, return an empty list
        logging.warning(f"Couldn't fetch symbols after {max_retries} attempts.")
        return []

    def get_atrp_sorted_rotator_symbols(self, min_qty_threshold: float = None, blacklist: list = None, whitelist: list = None, max_usd_value: float = None, max_retries: int = 5):
        url = f"https://api.quantumvoid.org/volumedata/rotatorsymbols_{self.data_source_exchange.replace('_', '')}_atrp.json"
        
        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(58, delay)  # cap the delay to 30 seconds

            try:
                logging.info(f"Sending request to {url} (Attempt: {retry + 1})")
                header, raw_json = send_public_request(url=url)
                
                if isinstance(raw_json, list):
                    logging.info(f"Received {len(raw_json)} ATRP sorted rotator symbols from API")
                    
                    filtered_symbols = []
                    for asset in raw_json:
                        symbol = asset.get("Asset", "")
                        min_qty = asset.get("Min qty", 0)
                        usd_price = asset.get("Price", float('inf'))

                        logging.info(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if blacklist and any(fnmatch.fnmatch(symbol, pattern) for pattern in blacklist):
                            logging.debug(f"Skipping {symbol} as it's in blacklist")
                            continue

                        if whitelist and symbol not in whitelist:
                            logging.debug(f"Skipping {symbol} as it's not in whitelist")
                            continue

                        # Check against the max_usd_value, if provided
                        if max_usd_value is not None and usd_price > max_usd_value:
                            logging.debug(f"Skipping {symbol} as its USD price {usd_price} is greater than the max allowed {max_usd_value}")
                            continue

                        logging.debug(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if min_qty_threshold is None or min_qty <= min_qty_threshold:
                            filtered_symbols.append(asset)

                    logging.info(f"Returning {len(filtered_symbols)} ATRP sorted rotator symbols")
                    return filtered_symbols
                else:
                    logging.warning("Unexpected data format. Expected a list of ATRP sorted rotator symbols.")
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request failed: {e}")
            except json.decoder.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON: {e}. Response: {raw_json}")
            except Exception as e:
                logging.warning(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                sleep(delay)
        
        # If all retries fail, return an empty list
        logging.warning(f"Couldn't fetch ATRP sorted rotator symbols after {max_retries} attempts.")
        return []

    def get_bullish_rotator_symbols(self, min_qty_threshold: float = None, blacklist: list = None, whitelist: list = None, max_usd_value: float = None, max_retries: int = 5):
        url = f"https://api.quantumvoid.org/volumedata/rotatorsymbols_{self.data_source_exchange.replace('_', '')}_bullish.json"
        return self._get_rotator_symbols(url, min_qty_threshold, blacklist, whitelist, max_usd_value, max_retries)

    def get_bearish_rotator_symbols(self, min_qty_threshold: float = None, blacklist: list = None, whitelist: list = None, max_usd_value: float = None, max_retries: int = 5):
        url = f"https://api.quantumvoid.org/volumedata/rotatorsymbols_{self.data_source_exchange.replace('_', '')}_bearish.json"
        return self._get_rotator_symbols(url, min_qty_threshold, blacklist, whitelist, max_usd_value, max_retries)

    def _get_rotator_symbols(self, url, min_qty_threshold, blacklist, whitelist, max_usd_value, max_retries):
        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(58, delay)  # cap the delay to 30 seconds

            try:
                logging.info(f"Sending request to {url} (Attempt: {retry + 1})")
                header, raw_json = send_public_request(url=url)

                if isinstance(raw_json, list):
                    logging.info(f"Received {len(raw_json)} assets from API")
                    symbols = []
                    for asset in raw_json:
                        symbol = asset.get("Asset", "")
                        min_qty = asset.get("Min qty", 0)
                        usd_price = asset.get("Price", float('inf'))

                        logging.info(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if blacklist and any(fnmatch.fnmatch(symbol, pattern) for pattern in blacklist):
                            logging.debug(f"Skipping {symbol} as it's in blacklist")
                            continue

                        if whitelist and symbol not in whitelist:
                            logging.debug(f"Skipping {symbol} as it's not in whitelist")
                            continue

                        # Check against the max_usd_value, if provided
                        if max_usd_value is not None and usd_price > max_usd_value:
                            logging.debug(f"Skipping {symbol} as its USD price {usd_price} is greater than the max allowed {max_usd_value}")
                            continue

                        logging.debug(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if min_qty_threshold is None or min_qty <= min_qty_threshold:
                            symbols.append(symbol)

                    logging.info(f"Returning {len(symbols)} symbols")
                    return symbols
                else:
                    logging.warning("Unexpected data format. Expected a list of assets.")

            except requests.exceptions.RequestException as e:
                logging.warning(f"Request failed: {e}")
            except json.decoder.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON: {e}. Response: {raw_json}")
            except Exception as e:
                logging.warning(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                sleep(delay)

        # If all retries fail, return an empty list
        logging.warning(f"Couldn't fetch rotator symbols after {max_retries} attempts.")
        return []

    def get_auto_rotate_symbols(self, min_qty_threshold: float = None, blacklist: list = None, whitelist: list = None, max_usd_value: float = None, max_retries: int = 5):
        if self.rotator_symbols_cache and not self.is_cache_expired():
            return self.rotator_symbols_cache

        symbols = []
        url = f"https://api.quantumvoid.org/volumedata/rotatorsymbols_{self.data_source_exchange.replace('_', '')}.json"
        
        for retry in range(max_retries):
            delay = 2**retry  # exponential backoff
            delay = min(58, delay)  # cap the delay to 30 seconds

            try:
                logging.info(f"Sending request to {url} (Attempt: {retry + 1})")
                header, raw_json = send_public_request(url=url)
                
                if isinstance(raw_json, list):
                    logging.info(f"Received {len(raw_json)} assets from API")
                    
                    for asset in raw_json:
                        symbol = asset.get("Asset", "")
                        min_qty = asset.get("Min qty", 0)
                        usd_price = asset.get("Price", float('inf')) 
                        
                        logging.info(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if blacklist and any(fnmatch.fnmatch(symbol, pattern) for pattern in blacklist):
                            logging.debug(f"Skipping {symbol} as it's in blacklist")
                            continue

                        if whitelist and symbol not in whitelist:
                            logging.debug(f"Skipping {symbol} as it's not in whitelist")
                            continue

                        # Check against the max_usd_value, if provided
                        if max_usd_value is not None and usd_price > max_usd_value:
                            logging.debug(f"Skipping {symbol} as its USD price {usd_price} is greater than the max allowed {max_usd_value}")
                            continue

                        logging.debug(f"Processing symbol {symbol} with min_qty {min_qty} and USD price {usd_price}")

                        if min_qty_threshold is None or min_qty <= min_qty_threshold:
                            symbols.append(symbol)

                    logging.info(f"Returning {len(symbols)} symbols")
                    
                    # If successfully fetched, update the cache and its expiry time
                    if symbols:
                        self.rotator_symbols_cache = symbols
                        self.rotator_symbols_cache_expiry = datetime.now() + timedelta(seconds=self.cache_life_seconds)

                    return symbols

                else:
                    logging.warning("Unexpected data format. Expected a list of assets.")
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request failed: {e}")
            except json.decoder.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON: {e}. Response: {raw_json}")
            except Exception as e:
                logging.warning(f"Unexpected error occurred: {e}")

            # Wait before the next retry
            if retry < max_retries - 1:
                time.sleep(delay)
        
        # Return cached symbols if all retries fail
        logging.warning(f"Couldn't fetch rotator symbols after {max_retries} attempts. Using cached symbols.")
        return self.rotator_symbols_cache or []

    def get_symbols(self):
        url = f"https://api.quantumvoid.org/volumedata/quantdatav2_{self.exchange_name.replace('_', '')}.json"
        try:
            header, raw_json = send_public_request(url=url)
            if isinstance(raw_json, list):
                return raw_json
            else:
                logging.info("Unexpected data format. Expected a list of symbols.")
                return []
        except requests.exceptions.RequestException as e:
            logging.info(f"Request failed: {e}")
            return []
        except json.decoder.JSONDecodeError as e:
            logging.info(f"Failed to parse JSON: {e}")
            return []
        except Exception as e:
            logging.info(f"Unexpected error occurred: {e}")
            return []

    def get_remote_data(self):
            if not self.check_timestamp():
                return self.data
            while True:  # Keep trying until a successful request is made
                try:
                    header, raw_json = send_public_request(url=self.url)
                    self.data = raw_json
                    break  # if the request was successful, break the loop
                except requests.exceptions.RequestException as e:
                    logging.info(f"Request failed: {e}, retrying...")
                except json.decoder.JSONDecodeError as e:
                    logging.info(f"Failed to parse JSON: {e}, retrying...")
                except Exception as e:
                    logging.info(f"Unexpected error occurred: {e}, retrying...")
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
            logging.info(f"{e}")
        return None

    def get_1m_moving_averages(self, symbol, num_bars=20):
        return self.exchange.get_moving_averages(symbol, "1m", num_bars)
    
    def get_5m_moving_averages(self, symbol, num_bars=20):
        return self.exchange.get_moving_averages(symbol, "5m", num_bars)

    def get_asset_value(self, symbol: str, data, value: str):
        try:
            if value == "Funding":
                for asset_data in data:
                    if asset_data.get("Asset") == symbol:
                        return asset_data.get("Funding", 0)
            else:
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
                    if value == "MFI" and "MFI" in asset_data:
                        return asset_data["MFI"]
                    if value == "ERI Bull Power" in asset_data:
                        return asset_data["ERI Bull Power"]
                    if value == "ERI Bear Power" in asset_data:
                        return asset_data["ERI Bear Power"]
                    if value == "ERI Trend" in asset_data:
                        return asset_data["ERI Trend"]
                    if value == "HMA Trend" in asset_data:
                        return asset_data["HMA Trend"]
                    if value == "Top Signal 5m" in asset_data:
                        return asset_data["Top Signal 5m"]
                    if value == "Bottom Signal 5m" in asset_data:
                        return asset_data["Bottom signal 5m"]
                    if value == "Top Signal 1m" in asset_data:
                        return asset_data["Top Signal 1m"]
                    if value == "Bottom Signal 1m" in asset_data:
                        return asset_data["Bottom signal 1m"]
                    if value == "MA Trend" in asset_data:
                        return asset_data["MA Trend"]
                    if value == "EMA Trend" in asset_data:
                        return asset_data["EMA Trend"]
        except Exception as e:
            logging.info(f"{e}")
        return None

    def is_api_data_cache_expired(self):
        return datetime.now() > self.api_data_cache_expiry

    def get_api_data(self, symbol):
        api_data_url = f"https://api.quantumvoid.org/volumedata/quantdatav2_{self.data_source_exchange.replace('_', '')}.json"
        data = self.fetch_data_from_url(api_data_url)
        symbols = [asset.get("Asset", "") for asset in data if "Asset" in asset]

        # Fetch funding rate data from the new URL
        funding_data_url = f"https://api.quantumvoid.org/volumedata/funding_{self.data_source_exchange.replace('_', '')}.json"
        funding_data = self.fetch_data_from_url(funding_data_url)

        #logging.info(f"Funding data: {funding_data}")

        api_data = {
            '1mVol': self.get_asset_value(symbol, data, "1mVol"),
            '5mVol': self.get_asset_value(symbol, data, "5mVol"),
            '1hVol': self.get_asset_value(symbol, data, "1hVol"),
            '1mSpread': self.get_asset_value(symbol, data, "1mSpread"),
            '5mSpread': self.get_asset_value(symbol, data, "5mSpread"),
            '30mSpread': self.get_asset_value(symbol, data, "30mSpread"),
            '1hSpread': self.get_asset_value(symbol, data, "1hSpread"),
            '4hSpread': self.get_asset_value(symbol, data, "4hSpread"),
            'MA Trend': self.get_asset_value(symbol, data, "MA Trend"),
            'HMA Trend': self.get_asset_value(symbol, data, "HMA Trend"),
            'MFI': self.get_asset_value(symbol, data, "MFI"),
            'ERI Trend': self.get_asset_value(symbol, data, "ERI Trend"),
            'Funding': self.get_asset_value(symbol, funding_data, "Funding"),  # Use funding_data instead of data
            'Symbols': symbols,
            'Top Signal 5m': self.get_asset_value(symbol, data, "Top Signal 5m"),
            'Bottom Signal 5m': self.get_asset_value(symbol, data, "Bottom Signal 5m"),
            'Top Signal 1m': self.get_asset_value(symbol, data, "Top Signal 1m"),
            'Bottom Signal 1m': self.get_asset_value(symbol, data, "Bottom Signal 1m"),
            'EMA Trend': self.get_asset_value(symbol, data, "EMA Trend")
        }
        return api_data

    def extract_metrics(self, api_data, symbol):
        try:
            one_minute_volume = api_data.get('1mVol', 0)
            five_minute_volume = api_data.get('5mVol', 0)
            one_minute_distance = api_data.get('1mSpread', 0)
            five_minute_distance = api_data.get('5mSpread', 0)
            ma_trend = api_data.get('Trend', 'neutral')
            mfirsi_signal = api_data.get('MFI', 'neutral')
            funding_rate = api_data.get('Funding', 0)
            hma_trend = api_data.get('HMA Trend', 'neutral')
            eri_trend = api_data.get('ERI Trend', 'undefined')
            ema_trend = api_data.get('EMA Trend', 'undefined')
            
            fivemin_top_signal = str(api_data.get('Top Signal 5m', 'false')).lower() == 'true'
            fivemin_bottom_signal = str(api_data.get('Bottom Signal 5m', 'false')).lower() == 'true'

            onemin_top_signal = str(api_data.get('Top Signal 1m', 'false')).lower() == 'true'
            onemin_bottom_signal = str(api_data.get('Bottom Signal 1m', 'false')).lower() == 'true'

            return {
                "1mVol": one_minute_volume,
                "5mVol": five_minute_volume,
                "1mSpread": one_minute_distance,
                "5mSpread": five_minute_distance,
                "MFI": mfirsi_signal,
                "Funding": funding_rate,
                "HMA Trend": hma_trend,
                "ERI Trend": eri_trend,
                "Top Signal 5m": fivemin_top_signal,
                "Bottom Signal 5m": fivemin_bottom_signal,
                "Top Signal 1m": onemin_top_signal,
                "Bottom Signal 1m": onemin_bottom_signal,
                "EMA Trend": ema_trend,
                "MA Trend": ma_trend
            }
        except Exception as e:
            logging.warning(f"Error processing API data for symbol {symbol}: {e}")
            return {
                "1mVol": 0,
                "5mVol": 0,
                "5mSpread": 0,
                "MA Trend": 'neutral',
                "MFI": 'neutral',
                "Funding": 0,
                "HMA Trend": 'neutral',
                "ERI Trend": 'undefined',
                "Top Signal 5m": False,
                "Bottom Signal 5m": False,
                "Top Signal 1m": False,
                "Bottom Signal 1m": False,
                "EMA Trend": 'undefined'
            }