import requests
import json
from typing import TypedDict, Dict

# response = requests.get("http://13.127.240.18/data/quantdataj.json")

# data = response.json()

# assets = data["Assets"]

# print(assets)


# class TylerData(TypedDict):
#     symbol: str
#     volume: float
#     distance: float

# def tyler_api_data() -> Dict[str, TylerData]:
#     data = requests.get("http://13.127.240.18/data/quantdataj.json").json()
    
#     parsed = {}
    
#     for x in data:
#         td: TylerData = {"symbol": x[0], }


def grab_api_data():
    tyler_api_unparsed = requests.get("http://13.127.240.18/data/quantdata.json")

    api_data = tyler_api_unparsed.json()

    return api_data
    
# tyler_api_unparsed = requests.get("http://13.127.240.18/data/quantdata.json")

# api_data = tyler_api_unparsed.json()

def get_asset_data(symbol, data):
    for asset in data:
        if asset['Assets'] == symbol:
            return asset
    return None

def get_asset_price(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['Price']
    return None

# print(get_asset_price('BTCUSDT', api_data))

def get_asset_total_volume_1m(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['1m Volume (USDT)']
    return None

# print(get_asset_total_volume_1m('BTCUSDT', api_data))

def get_asset_volume_1m_1x(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['1m 1x Volume (USDT)']
    return None

# print(get_asset_volume_1m_1x('BTCUSDT', api_data))

def get_asset_total_volume_5m(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['5m Volume (USDT)']
    return None

# print(get_asset_total_volume_5m('BTCUSDT', api_data))

def get_asset_volume_5m_1x(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['5m 1x Volume (USDT)']
    return None

# print(get_asset_volume_5m_1x('BTCUSDT', api_data))

def get_asset_1m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['1m Spread']
    return None

# print(get_asset_1m_spread('BTCUSDT', api_data))

def get_asset_5m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['5m Spread']
    return None

# print(get_asset_5m_spread('BTCUSDT', api_data))

def get_asset_15m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['15m Spread']
    return None

# print(get_asset_15m_spread('BTCUSDT', api_data))

def get_asset_30m_spread(symbol, data):
    asset_data = get_asset_data(symbol, data)
    
    if asset_data:
        return asset_data['30m Spread']
    return None

# print(get_asset_15m_spread('BTCUSDT', api_data))

def get_asset_trend(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data['Trend']
    return None

#print(get_asset_trend('APTUSDT', api_data))

def get_asset_trend_pct(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data['trend%']
    return None

def get_asset_funding_rate(symbol, data):
    asset_data = get_asset_data(symbol, data)

    if asset_data:
        return asset_data['Funding']
    return None