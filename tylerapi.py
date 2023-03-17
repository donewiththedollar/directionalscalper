import json

import requests  # type: ignore

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
    try:
        response = requests.get("http://api.tradesimple.xyz/data/quantdata.json")
        response.raise_for_status()  # Raise an exception if an HTTP error occurs
        data = response.json()
    except (requests.exceptions.HTTPError, json.JSONDecodeError):
        data = None
    return data

def get_asset_data(symbol, data):
    try:
        for asset in data:
            if asset["Assets"] == symbol:
                return asset
        return None
    except:
        pass


def get_asset_price(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["Price"]
        return None
    except:
        pass
        


# print(get_asset_price('BTCUSDT', api_data))


def get_asset_total_volume_1m(symbol, data):
    try:

        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["1m 1x Volume (USDT)"]
        return None
    except:
        pass


# print(get_asset_total_volume_1m('BTCUSDT', api_data))


def get_asset_volume_1m_1x(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["1m 1x Volume (USDT)"]
        return None
    except:
        pass


# print(get_asset_volume_1m_1x('BTCUSDT', api_data))


def get_asset_total_volume_5m(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["5m 1x Volume (USDT)"]
        return None
    except:
        pass


# print(get_asset_total_volume_5m('BTCUSDT', api_data))


def get_asset_volume_5m_1x(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["5m 1x Volume (USDT)"]
        return None
    except:
        pass


# print(get_asset_volume_5m_1x('BTCUSDT', api_data))


def get_asset_1m_spread(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["1m Spread"]
        return None
    except:
        pass


# print(get_asset_1m_spread('BTCUSDT', api_data))


def get_asset_5m_spread(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["5m Spread"]
        return None
    except:
        pass


# print(get_asset_5m_spread('BTCUSDT', api_data))


def get_asset_15m_spread(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["15m Spread"]
        return None
    except:
        pass

# print(get_asset_15m_spread('BTCUSDT', api_data))


def get_asset_30m_spread(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["30m Spread"]
        return None
    except:
        pass


# print(get_asset_15m_spread('BTCUSDT', api_data))


def get_asset_trend(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["Trend"]
        return None
    except:
        pass


# print(get_asset_trend('APTUSDT', api_data))


def get_asset_trend_pct(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["trend%"]
        return None
    except:
        pass


def get_asset_funding_rate(symbol, data):
    try:
        asset_data = get_asset_data(symbol, data)

        if asset_data:
            return asset_data["Funding"]
        return None
    except:
        pass
