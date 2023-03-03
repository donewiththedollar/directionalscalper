import pandas as pd
import ccxt
import time
import pybit
from pybit import usdt_perpetual
import ta
import config
from config import *
import json

exchange = ccxt.bybit(
    {"enableRateLimit": True, "apiKey": config.api_key, "secret": config.api_secret}
)
# exchange = ccxt.bybit()
client = usdt_perpetual.HTTP(endpoint=endpoint,api_key=api_key,api_secret=api_secret)

symbols_list = ['XEMUSDT', 'NKNUSDT', 'CFXUSDT', 'COCOSUSDT', 'HIGHUSDT', 'BLURUSDT', 'BUSDUSDT', 'HOOKUSDT', 'GFTUSDT', 'FETUSDT', 'COREUSDT', 'AGIXUSDT', 'ZECUSDT', 'IOTXUSDT', 'OMGUSDT', 'KSMUSDT', 'TRXUSDT', 'VETUSDT', 'ICPUSDT', 'CROUSDT', 'BTCUSD','BTC/USD:BTC','SHIB1000USDT','OCEANUSDT','GRTUSDT','CHZUSDT','SCUSDT','BLZUSDT','IMXUSDT','RSRUSDT','RNDRUSDT','LDOUSDT','ACHUSDT','ICXUSDT','GMTUSDT','GALAUSDT','1000BONKUSDT','BTCUSDT', 'FXSUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'BCHUSDT', 'MATICUSDT', 'DOTUSDT', 'ADAUSDT', 'LINKUSDT', 'FTMUSDT', 'DOGEUSDT', 'ATOMUSDT', 'AVAXUSDT', 'EOSUSDT', 'LTCUSDT', 'NEARUSDT', 'AXSUSDT', 'SANDUSDT', 'SOLUSDT', 'OPUSDT', 'APTUSDT', 'APEUSDT', 'ETCUSDT', 'GALUSDT', 'MANAUSDT', 'DYDXUSDT', 'SUSHIUSDT', 'XTZUSDT', 'HBARUSDT', 'LUNA2USDT', 'BITUSDT']

prices_list = []
candle_high_close_1m = []
candle_high_close_5m = []
candle_high_close_15m = []
candle_high_close_30m = []

onexvolumes_1m = []
onexvolumes_5m = []
onexvolumes_30m = []

avg_true_range_1m = []

ma_order = []
ma_order_pct = []
ma_high_5m = []
ma_low_5m = []

funding_rate_results = []

def bybit_5min_candle_info(selected_symbol):
    global fivemin_df
    bars_5min = exchange.fetch_ohlcv(symbol=selected_symbol, timeframe='5m', limit=2) # Fetch last five, five minute candles
    fivemin_df = pd.DataFrame(bars_5min[:-1], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    fivemin_df['time'] = pd.to_datetime(fivemin_df['time'] / 1000, unit='s')


    return fivemin_df

def bybit_1min_candle_info(selected_symbol):
    global onemin_df
    bars_1min = exchange.fetch_ohlcv(symbol=selected_symbol, timeframe='1m', limit=2) # Fetch last five, five minute candles
    onemin_df = pd.DataFrame(bars_1min[:-1], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    onemin_df['time'] = pd.to_datetime(onemin_df['time'] / 1000, unit='s')
    
    # Add 6 EMA columns
    onemin_df['ema_6'] = onemin_df['close'].ewm(span=6).mean()
    onemin_df['ema_6_high'] = onemin_df['high'].ewm(span=6).mean()
    onemin_df['ema_6_low'] = onemin_df['low'].ewm(span=6).mean()

    # Calculate spread as (6 EMA high - 6 EMA low) / 6 EMA low * 100
    onemin_df['spread'] = (onemin_df['ema_6_high'] - onemin_df['ema_6_low']) / onemin_df['ema_6_low'] * 100

    return onemin_df

def bybit_5min_spread_calc(data):
    data['high-low'] = abs(data['high'] - abs(data['low']))

    spread_5min = data[['high-low']].max(axis=1)

    return spread_5min

def bybit_5min_volume_calc(data):
    
    volume_calc = data[['volume']].max(axis=1)

    return volume_calc

def bybit_30min_candle_info(selected_symbol):
    global thirtymin_df
    bars_30min = exchange.fetch_ohlcv(symbol=selected_symbol, timeframe='30m', limit=2) # Fetch last two 30-minute candles
    thirtymin_df = pd.DataFrame(bars_30min[:-1], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    thirtymin_df['time'] = pd.to_datetime(thirtymin_df['time'] / 1000, unit='s')

    return thirtymin_df

def get_decimals():

    symbol_decimals  = client.query_symbol()
    for decimal in symbol_decimals['result']:
        if decimal['name'] == symbol:
            global decimals
            global leverage
            global tick_size
            global min_trading_qty
            global qty_step
            decimals = decimal['price_scale']
            leverage = decimal['leverage_filter']['max_leverage']
            tick_size = decimal['price_filter']['tick_size']
            min_trading_qty = decimal['lot_size_filter']['min_trading_qty']
            qty_step = decimal['lot_size_filter']['qty_step']
            
def get_ema_6_1m_low_bybit(symbol):

    bars = exchange.fetchOHLCV(symbol=symbol, timeframe='1m', limit=18)
    df = pd.DataFrame(bars,columns=['Time','Open','High','Low','Close','Vol'])
    df['EMA 6-1 Low'] = ta.trend.EMAIndicator(df['Low'], window=6).ema_indicator()
    global ema_6_1_low_bybit
    ema_6_1_low_bybit = round((df['EMA 6-1 Low'][17]).astype(float),decimals)
    
    return ema_6_1_low_bybit

def get_ema_6_1_high_bybit(symbol):

    bars = exchange.fetchOHLCV(symbol=symbol, timeframe='1m', limit=18)
    df = pd.DataFrame(bars,columns=['Time','Open','High','Low','Close','Vol'])
    df['EMA 6-1 High'] = ta.trend.EMAIndicator(df['High'], window=6).ema_indicator()
    global ema_6_1_high_bybit
    ema_6_1_high_bybit = round((df['EMA 6-1 High'][17]).astype(float),decimals)
    
    return ema_6_1_high_bybit

def get_5m_data(symbol):
    timeframe = "5m"
    num_bars = 20
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=num_bars)
    df = pd.DataFrame(bars, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    df["Time"] = pd.to_datetime(df["Time"], unit="ms")
    df["MA_3_High"] = df.High.rolling(3).mean()
    df["MA_3_Low"] = df.Low.rolling(3).mean()
    df["MA_6_High"] = df.High.rolling(6).mean()
    df["MA_6_Low"] = df.Low.rolling(6).mean()
    get_5m_data_3_high = df["MA_3_High"].iat[-1]
    get_5m_data_3_low = df["MA_3_Low"].iat[-1]
    get_5m_data_6_high = df["MA_6_High"].iat[-1]
    get_5m_data_6_low = df["MA_6_Low"].iat[-1]
    return get_5m_data_3_high, get_5m_data_3_low, get_5m_data_6_high, get_5m_data_6_low
    
def get_spread_30m(asset):
    try:
        half_hour_ago = int((time.time() - (1800)) * 1000)
        # print(half_hour_ago)
        data = exchange.fetch_ohlcv(symbol=asset, timeframe='1m', since=half_hour_ago)
        # OHLCV
        lowest_low = 999999
        highest_high = 0
        for d in data:
            # print(d)
            if d[2] > highest_high:
                highest_high = d[2]
            if d[3] < lowest_low:
                lowest_low = d[3]

        # print(highest_high)
        # print(lowest_low)
        # print(highest_high - lowest_low)
        # print(round((highest_high - lowest_low) / highest_high * 100, 4))
        return round((highest_high - lowest_low) / highest_high * 100, 4)
    except Exception as e:
        print('Exception: {}'.format(e))
        
def get_spread_15m(asset):
    try:
        fifteen_minutes_ago = int((time.time() - 900) * 1000)
        # print(half_hour_ago)
        data = exchange.fetch_ohlcv(symbol=asset, timeframe='1m', since=fifteen_minutes_ago)
        # OHLCV
        lowest_low = 999999
        highest_high = 0
        for d in data:
            # print(d)
            if d[2] > highest_high:
                highest_high = d[2]
            if d[3] < lowest_low:
                lowest_low = d[3]

        # print(highest_high)
        # print(lowest_low)
        # print(highest_high - lowest_low)
        # print(round((highest_high - lowest_low) / highest_high * 100, 4))
        return round((highest_high - lowest_low) / highest_high * 100, 4)
    except Exception as e:
        print('Exception: {}'.format(e))
        
def get_spread_5m(asset):
    try:
        five_minutes_ago = int((time.time() - 300) * 1000)
        # print(half_hour_ago)
        data = exchange.fetch_ohlcv(symbol=asset, timeframe='1m', since=five_minutes_ago)
        # OHLCV
        lowest_low = 999999
        highest_high = 0
        for d in data:
            # print(d)
            if d[2] > highest_high:
                highest_high = d[2]
            if d[3] < lowest_low:
                lowest_low = d[3]

        # print(highest_high)
        # print(lowest_low)
        # print(highest_high - lowest_low)
        # print(round((highest_high - lowest_low) / highest_high * 100, 4))
        return round((highest_high - lowest_low) / highest_high * 100, 4)
    except Exception as e:
        print('Exception: {}'.format(e))
        
def get_spread_1m(asset):
    try:
        one_minute_ago = int((time.time() - 60) * 1000)
        data = exchange.fetch_ohlcv(symbol=asset, timeframe='1m', since=one_minute_ago)
        lowest_low = 999999
        highest_high = 0
        for d in data:
            # print(d)
            if d[2] > highest_high:
                highest_high = d[2]
            if d[3] < lowest_low:
                lowest_low = d[3]
        return round((highest_high - lowest_low) / highest_high * 100, 4)
    except Exception as e:
        print('Exception: {}'.format(e))
        
# def atr(data, period):
#     data['tr'] = tr(data)
#     atr = data['tr'].rolling(period).mean()

#     return atr

def tr(data):
    data['previous_close'] = data['close'].shift(1)
    data['high-low'] = abs(data['high'] - data['low'])
    data['high-pc'] = abs(data['high'] - data['previous_close'])
    data['low-pc'] = abs(data['low'] - data['previous_close'])

    tr = data[['high-low', 'high-pc', 'low-pc']].max(axis=1)

    return tr

def get_average_true_range_1m(asset, period):
    try:
        one_minute_ago = int((time.time() - 60) * 1000)
        
        data = exchange.fetch_ohlcv(symbol=asset, timeframe='1m', since=one_minute_ago)
        
        data['tr'] = tr(data)
        atr = data['tr'].rolling(period).mean()
        
        return atr
    except Exception as e:
        print('Exception: {}'.format(e))
        
def fetch_sma(asset):
    try:
        bars = exchange.fetchOHLCV(symbol=asset, timeframe='1m', limit=30)
        df = pd.DataFrame(bars,columns=['Time','Open','High','Low','Close','Vol'])
        sma = ta.trend.SMAIndicator(df['Close'], window=14).sma_indicator()
        
        current_sma = sma[29]
        global last_close_price
        last_close_price = bars[29][4]
        # if current_sma < last_close_price:
        #     print('LONG')
        # else:
        #     print('SHORT')
        
        return round((last_close_price - current_sma) / last_close_price * 100, 4)

    except Exception as e:
        print('Exception: {}'.format(e))
          
# fetch_sma('BTCUSDT')
        
def get_funding_rate(symbol):
    try:
        global returned_funding
        funding_rate = exchange.fetch_funding_rate(symbol)
        funding_rate = funding_rate['fundingRate']

        returned_funding = funding_rate

        return returned_funding
    except:
        print("Wtf?")
        pass
        
for symbol in symbols_list:
    ticker = exchange.fetch_ticker(symbol)
    price = ticker['last']
    prices_list.append(price)
    
    # Grab different timeframe candle data, organized
    candles_30m = exchange.fetch_ohlcv(symbol, timeframe='30m', limit=5)
    candles_15m = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=5)
    candles_5m = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=5)
    candles_1m = exchange.fetch_ohlcv(symbol, timeframe='1m', limit=5)

    # Define latest candles for each timeframe
    latest_candle_1m = candles_1m[-1]
    latest_candle_5m = candles_5m[-1]
    latest_candle_15m = candles_15m[-1]
    latest_candle_30m = candles_30m[-1]
    
    # Define candle spreads per timeframe
    one_minute_spread = get_spread_1m(symbol)
    candle_high_close_1m.append(one_minute_spread)
    
    five_minute_spread = get_spread_5m(symbol)
    candle_high_close_5m.append(five_minute_spread)
    
    fifteen_minute_spread = get_spread_15m(symbol)
    candle_high_close_15m.append(fifteen_minute_spread)
    
    thirty_minute_spread = get_spread_30m(symbol)
    candle_high_close_30m.append(thirty_minute_spread)
    
    # Define 1x 5m candle volume
    onexcandlevol = latest_candle_5m[5]
    volume_1x_5m = price * onexcandlevol
    onex_volume = round(volume_1x_5m)
    onexvolumes_5m.append(onex_volume)
    
    #Define 1x 1m candle volume
    onex1mcandlevol = latest_candle_1m[5]
    volume_1x = price * onex1mcandlevol
    onex_1m_volume = round(volume_1x)
    onexvolumes_1m.append(onex_1m_volume)    
    
    #Define 1x 30m candle volume
    onex30mcandlevol = latest_candle_30m[5]
    volume_1x_30m = price * onex30mcandlevol
    onex_30m_volume = round(volume_1x_30m)
    onexvolumes_30m.append(onex_30m_volume)
    
    #Define MA data
    ma_6_5m_high = get_5m_data(symbol)[2]
    ma_6_5m_low = get_5m_data(symbol)[3]
    
    ma_5m_high_resolution = ma_6_5m_high
    ma_5m_low_resolution = ma_6_5m_low
    ma_high_5m.append(ma_5m_high_resolution)
    ma_low_5m.append(ma_5m_low_resolution)
    
    ma_order_resolution = fetch_sma(symbol)
    ma_order_long_short = ma_order_resolution
    ma_order_pct.append(ma_order_resolution)
    
    trends = []

    for value in ma_order_pct:
        try:
            if value > 0:
                trends.append('short')
            else:
                trends.append('long')
        except:
            trends.append('mixed')
            pass
            
    # Define funding rates
    returned_funding = get_funding_rate(symbol) * 100
    funding_rate_results.append(returned_funding)
    

analysis_data = {'Assets': ['XEMUSDT', 'NKNUSDT', 'CFXUSDT', 'COCOSUSDT', 'HIGHUSDT', 'BLURUSDT', 'BUSDUSDT', 'HOOKUSDT', 'GFTUSDT', 'FETUSDT', 'COREUSDT', 'AGIXUSDT', 'ZECUSDT', 'IOTXUSDT', 'OMGUSDT', 'KSMUSDT', 'TRXUSDT', 'VETUSDT', 'ICPUSDT', 'CROUSDT', 'BTCUSD','BTC/USD:BTC','SHIB1000USDT','OCEANUSDT','GRTUSDT','CHZUSDT','SCUSDT','BLZUSDT','IMXUSDT','RSRUSDT','RNDRUSDT','LDOUSDT','ACHUSDT','ICXUSDT','GMTUSDT','GALAUSDT','1000BONKUSDT','BTCUSDT', 'FXSUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'BCHUSDT', 'MATICUSDT', 'DOTUSDT', 'ADAUSDT', 'LINKUSDT', 'FTMUSDT', 'DOGEUSDT', 'ATOMUSDT', 'AVAXUSDT', 'EOSUSDT', 'LTCUSDT', 'NEARUSDT', 'AXSUSDT', 'SANDUSDT', 'SOLUSDT', 'OPUSDT', 'APTUSDT', 'APEUSDT', 'ETCUSDT', 'GALUSDT', 'MANAUSDT', 'DYDXUSDT', 'SUSHIUSDT', 'XTZUSDT', 'HBARUSDT', 'LUNA2USDT', 'BITUSDT'],
'Price': prices_list,
'1m 1x Volume (USDT)': onexvolumes_1m,
'5m 1x Volume (USDT)': onexvolumes_5m,
'30m 1x Volume (USDT)': onexvolumes_30m,
'1m Spread': candle_high_close_1m,
'5m Spread': candle_high_close_5m,
'15m Spread': candle_high_close_15m,
'30m Spread': candle_high_close_30m
,'trend%': ma_order_pct,
'Trend': trends,
'5m MA6 high': ma_high_5m,
'5m MA6 low': ma_low_5m,
'Funding': funding_rate_results}

df = pd.DataFrame(analysis_data)

df.sort_values(by=['1m 1x Volume (USDT)', '5m Spread'], inplace=True, ascending= [False, False])

df.to_csv('data/quantdata.csv')

df.to_json('data/quantdata.json', orient='records')


df_what_to_trade = df[df["1m 1x Volume (USDT)"] > 15000]

# df_what_to_trade_filtered_by_spread = df_what_to_trade[df['5m Spread'] > round(config_min_distance)]

#df_what_to_trade = df[df["1m 1x Volume (USDT)"] > 15000 & df["5m Spread"] > 0.15]

#df_what_to_trade.sort_values(by=['1m 1x Volume (USDT)', '5m Spread'], inplace=True, ascending= [False, False])

# print("What to trade:")
# print(df_what_to_trade)

# df_what_to_trade_filtered_by_spread.to_csv('data/whattotrade.csv')
# df_what_to_trade_filtered_by_spread.to_json('data/whattotrade.json', orient='records')

# df_what_to_trade.to_csv('data/whattotrade.csv')
# df_what_to_trade.to_json('data/whattotrade.json', orient='records')

# df_what_negative_funding = df[df["Funding"] < 0]

# df_what_negative_funding.to_csv('data/negativefunding.csv')
# df_what_negative_funding.to_json('data/negativefunding.json', orient='records')

# df_what_positive_funding = df[df["Funding"] > 0]

# df_what_positive_funding.to_csv('data/positivefunding.csv')
# df_what_positive_funding.to_json('data/positivefunding.json', orient='records')