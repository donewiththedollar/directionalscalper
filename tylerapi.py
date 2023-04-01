import json
import logging
import datetime
import requests  # type: ignore

log = logging.getLogger(__name__)

_cached_data = None
_cached_time = None
CACHE_TIME_SECONDS = 10

def grab_api_data():
    """
    Grab cached api data if it exists and is not expired for 10 seconds.
    Otherwise, grab fresh data.
    API data is updated once a minute, so we don't need to grab fresh data every time.
    """
    global _cached_data
    global _cached_time
    if not has_cache() or is_cache_expired():
        _cached_data = grab_fresh_api_data()
        _cached_time = datetime.datetime.now()
    return _cached_data

def is_cache_expired():
    global _cached_time
    if _cached_time is None:
        return True
    return (datetime.datetime.now() - _cached_time).total_seconds() > CACHE_TIME_SECONDS

def has_cache():
    return _cached_data is not None and _cached_time is not None

def grab_fresh_api_data():
    try:
        response = requests.get("http://api.tradesimple.xyz/data/quantdata.json")
        response.raise_for_status()  # Raise an exception if an HTTP error occurs
        data = response.json()
    except (requests.exceptions.HTTPError, json.JSONDecodeError) as e:
        log.warning(f"{e}")
    return data

def get_asset_data(symbol: str, data):
    try:
        for asset in data:
            if asset["Assets"] == symbol:
                return asset
        return None
    except Exception as e:
        log.warning(f"{e}")


def get_asset_value(symbol: str, data, value: str):
    try:
        asset_data = get_asset_data(symbol, data)
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
        return None
    except Exception as e:
        log.warning(f"{e}")
