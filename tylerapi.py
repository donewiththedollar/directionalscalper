import json
import logging

import requests  # type: ignore

log = logging.getLogger(__name__)
# def grab_api_data():
#     # print("grab api data")
#     try:
#         tyler_api_unparsed = requests.get("http://api.tradesimple.xyz/data/quantdata.json")
#         api_data = tyler_api_unparsed.json()
#         return api_data
#     except (json.decoder.JSONDecodeError, requests.exceptions.RequestException):
#         #print("Error retrieving API data. Returning None...")
#         return None


def grab_api_data():
    data = None
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
