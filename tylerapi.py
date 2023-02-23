import json

import requests  # type: ignore


def grab_api_data():
    try:
        tyler_api_unparsed = requests.get("http://13.127.240.18/data/quantdata.json")
        tyler_api_unparsed.raise_for_status()

        if not tyler_api_unparsed.text:
            raise Exception("The API response is empty")

        api_data = tyler_api_unparsed.json()

        return api_data
    except requests.exceptions.RequestException as e:
        raise Exception("Error in retrieving data from api: {}".format(str(e))) from e
    except json.decoder.JSONDecodeError as e:
        raise Exception("Error in parsing the JSON data: {}".format(str(e))) from e


# tyler_api_unparsed = requests.get("http://13.127.240.18/data/quantdata.json")

# api_data = tyler_api_unparsed.json()


def get_asset_data(symbol, data):
    for asset in data:
        if asset["Assets"] == symbol:
            return asset
    return None


def get_asset_price(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["Price"]
    return None


# print(get_asset_price('BTCUSDT', api_data))


def get_asset_total_volume_1m(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["1m 1x Volume (USDT)"]
    return None


# print(get_asset_total_volume_1m('BTCUSDT', api_data))


def get_asset_volume_1m_1x(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["1m 1x Volume (USDT)"]
    return None


# print(get_asset_volume_1m_1x('BTCUSDT', api_data))


def get_asset_total_volume_5m(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["5m 1x Volume (USDT)"]
    return None


# print(get_asset_total_volume_5m('BTCUSDT', api_data))


def get_asset_volume_5m_1x(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["5m 1x Volume (USDT)"]
    return None


# print(get_asset_volume_5m_1x('BTCUSDT', api_data))


def get_asset_1m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["1m Spread"]
    return None


# print(get_asset_1m_spread('BTCUSDT', api_data))


def get_asset_5m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["5m Spread"]
    return None


# print(get_asset_5m_spread('BTCUSDT', api_data))


def get_asset_15m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["15m Spread"]
    return None


# print(get_asset_15m_spread('BTCUSDT', api_data))


def get_asset_30m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["30m Spread"]
    return None


# print(get_asset_15m_spread('BTCUSDT', api_data))


def get_asset_trend(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["Trend"]
    return None


# print(get_asset_trend('APTUSDT', api_data))


def get_asset_trend_pct(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["trend%"]
    return None


def get_asset_funding_rate(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data["Funding"]
    return None
