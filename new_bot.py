import ccxt
import pandas as pd
import config
from config import *
import time
import requests
from typing import TypedDict, Dict
import argparse
from telegram_status import notifier
from colorama import init, Fore, Back, Style
from rich.live import Live
from rich.table import Table
import tylerapi
from tylerapi import *

# Booleans and stuff
version = "1.1.0 alpha"
long_mode = False
short_mode = False
hedge_mode = False
violent_mode = False
high_vol_stack_mode = False

global symbol

print(Fore.LIGHTCYAN_EX +'CCXT Scalper ',version,' connecting to exchange'+ Style.RESET_ALL)


min_volume = config.config_min_volume
min_distance = config.config_min_distance
high_volume = config.config_high_volume
high_distance = config.config_high_distance
violent_volume = config.config_violent_volume
violent_distance = config.config_violent_distance
botname = config.config_botname

# class VolumeData(TypedDict):
#     symbol: str 
#     volume: float 
#     distance: float

exchange = ccxt.bybit(
    {"enableRateLimit": True, "apiKey": config.api_key, "secret": config.api_secret}
)

parser = argparse.ArgumentParser(description='Scalper supports 3 modes')

parser.add_argument('--mode', type=str, help='Mode to use', 
choices=['long_mode', 
'short_mode', 
'hedge_mode'],
required=True)

parser.add_argument('--symbol', type=str, help='Specify symbol',
required=True)

parser.add_argument('--iqty', type=str, help="Initial entry quantity",
required=True)

parser.add_argument('--tg', type=str, help="TG Notifications",
choices=['on', 'off'],
required=True)

args = parser.parse_args()

if args.symbol:
    symbol= (args.symbol)
else:
    symbol = input('Instrument undefined. \nInput instrument:')

# Functions

# Get min vol data

def get_min_vol_dist_data(symbol) -> bool:
    spread5m = tylerapi.get_asset_5m_spread(symbol, tylerapi.api_data)
    volume5m = tylerapi.get_asset_total_volume_5m(symbol, tylerapi.api_data)

    return volume5m > min_volume and spread5m > min_distance

# get_1m_data() [0]3 high, [1]3 low, [2]6 high, [3]6 low, [4]10 vol
def get_1m_data():
    timeframe = "1m"
    num_bars = 20
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=num_bars)
    df = pd.DataFrame(bars, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    df["Time"] = pd.to_datetime(df["Time"], unit="ms")
    df["MA_3_High"] = df.High.rolling(3).mean()
    df["MA_3_Low"] = df.Low.rolling(3).mean()
    df["MA_6_High"] = df.High.rolling(6).mean()
    df["MA_6_Low"] = df.Low.rolling(6).mean()
    get_1m_data_3_high = df["MA_3_High"].iat[-1]
    get_1m_data_3_low = df["MA_3_Low"].iat[-1]
    get_1m_data_6_high = df["MA_6_High"].iat[-1]
    get_1m_data_6_low = df["MA_6_Low"].iat[-1]
    return get_1m_data_3_high, get_1m_data_3_low, get_1m_data_6_high, get_1m_data_6_low


# get_5m_data() [0]3 high, [1]3 low, [2]6 high, [3]6 low
def get_5m_data():
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

def get_balance():
    global dex_balance, dex_upnl, dex_wallet, dex_equity
    dex = exchange.fetch_balance()["info"]["result"]
    dex_balance = dex["USDT"]["available_balance"]
    dex_pnl = dex["USDT"]["realised_pnl"]
    dex_upnl = dex["USDT"]["unrealised_pnl"]
    dex_wallet = round(float(dex["USDT"]["wallet_balance"]), 2)
    dex_equity = round(float(dex["USDT"]["equity"]), 2)

# get_orderbook() [0]bid, [1]ask
def get_orderbook():
    ob = exchange.fetch_order_book(symbol)
    bid = ob["bids"][0][0]
    ask = ob["asks"][0][0]
    return bid, ask

# get_market_data() [0]precision, [1]leverage, [2]min_trade_qty
def get_market_data():
    global leverage
    exchange.load_markets()
    precision = exchange.market(symbol)["info"]["price_scale"]
    leverage = exchange.market(symbol)["info"]["leverage_filter"]["max_leverage"]
    min_trade_qty = exchange.market(symbol)["info"]["lot_size_filter"][
        "min_trading_qty"
    ]
    return precision, leverage, min_trade_qty

def get_short_positions():
    global short_pos_qty, short_pos_price, short_symbol_realised, short_symbol_cum_realised, short_pos_unpl, short_pos_unpl_pct, short_liq_price
    pos_dict = exchange.fetch_positions([symbol])
    pos_dict = pos_dict[1]
    short_pos_qty = float(pos_dict["contracts"])
    short_symbol_realised = round(float(pos_dict["info"]["realised_pnl"] or 0), 2)
    short_symbol_cum_realised = round(float(pos_dict["info"]["cum_realised_pnl"] or 0), 2)
    short_pos_unpl = round(float(pos_dict["info"]["unrealised_pnl"] or 0), 2)
    short_pos_unpl_pct = round(float(pos_dict["percentage"] or 0), 2)
    short_pos_price = pos_dict["entryPrice"] or 0
    short_liq_price = pos_dict["liquidationPrice"] or 0
    
def get_long_positions():
    global long_pos_qty, long_pos_price, long_symbol_realised, long_symbol_cum_realised, long_pos_unpl, long_pos_unpl_pct, long_liq_price
    pos_dict = exchange.fetch_positions([symbol]) #TODO: We can fetch it just once to save some API time
    pos_dict = pos_dict[0]
    long_pos_qty = float(pos_dict["contracts"])
    long_symbol_realised = round(float(pos_dict["info"]["realised_pnl"]), 2)
    long_symbol_cum_realised = round(float(pos_dict["info"]["cum_realised_pnl"]), 2)
    long_pos_unpl = float(pos_dict["info"]["unrealised_pnl"] or 0)
    long_pos_unpl_pct = round(float(pos_dict["percentage"] or 0), 2)
    long_pos_price = pos_dict["entryPrice"] or 0
    long_liq_price = pos_dict["liquidationPrice"] or 0

# get_open_orders() [0]order_id, [1]order_price, [2]order_qty
def get_open_orders():
    order = exchange.fetch_open_orders(symbol)
    order_status = order[0]["info"]["order_status"]
    order_side = order[0]["info"]["side"]
    reduce_only = order[0]["info"]["reduce_only"]
    if (
        order_status == "New"
        and order_status != "Filled"
        and order_side == "Buy"
        and reduce_only == True
    ):
        order_id = order[0]["info"]["order_id"]
        order_price = order[0]["info"]["price"]
        order_qty = order[0]["info"]["qty"]
    else:
        pass
    return order_id, order_price, order_qty

def cancel_entry():
    order = exchange.fetch_open_orders(symbol)
    order_id = order[0]["info"]["order_id"]
    order_status = order[0]["info"]["order_status"]
    order_side = order[0]["info"]["side"]
    reduce_only = order[0]["info"]["reduce_only"]
    if (
        order_status != "Filled"
        and order_side == "Buy"
        and order_status != "Cancelled"
        and reduce_only == False
    ):
        exchange.cancel_order(symbol=symbol, id=order_id)
    elif (
        order_status != "Filled"
        and order_side == "Sell"
        and order_status != "Cancelled"
        and reduce_only == False
    ):
        exchange.cancel_order(symbol=symbol, id=order_id)


def cancel_close():
    order = exchange.fetch_open_orders(symbol)
    order_id = order[0]["info"]["order_id"]
    order_status = order[0]["info"]["order_status"]
    order_side = order[0]["info"]["side"]
    reduce_only = order[0]["info"]["reduce_only"]
    if (
        order_status != "Filled"
        and order_side == "Buy"
        and order_status != "Cancelled"
        and reduce_only == True
    ):
        exchange.cancel_order(symbol=symbol, id=order_id)
    elif (
        order_status != "Filled"
        and order_side == "Sell"
        and order_status != "Cancelled"
        and reduce_only == True
    ):
        exchange.cancel_order(symbol=symbol, id=order_id)


def trade_condition():
    trade_condition = get_orderbook()[0] > get_1m_data()[0]
    return trade_condition

def long_trade_condition():
    long_trade_condition = get_orderbook()[0] < get_1m_data()[0]
    return long_trade_condition

def add_short_trade_condition():
    add_short_trade_condition = short_pos_price < get_1m_data()[3]
    return add_short_trade_condition

def add_long_trade_condition():
    add_long_trade_condition = long_pos_price > get_1m_data()[3]
    return add_long_trade_condition


def tg_notification(msg):
    if args.tg == 'on':
        try:
            notifier.notify_message(msg)
            print(Fore.GREEN +'Telegram message sent'+ Style.RESET_ALL)
        except:
            pass
    else:
        try:
            print(Fore.RED +'Telegram disabled'+ Style.RESET_ALL)
        except:
            pass 

def leverage_verification(symbol):
    try:
        exchange.set_position_mode(
            hedged='BothSide',
            symbol=symbol
        )
        print(Fore.LIGHTYELLOW_EX +'Position mode changed to BothSide'+ Style.RESET_ALL)
    except:
        print(Fore.YELLOW +'Position mode unchanged'+ Style.RESET_ALL)
        pass
    # Set margin mode
    try:
        exchange.set_margin_mode(
            marginMode='cross',
            symbol=symbol
        )
        print(Fore.LIGHTYELLOW_EX +'Margin mode set to cross'+ Style.RESET_ALL)
    except:
        print(Fore.YELLOW +'Margin mode unchanged'+ Style.RESET_ALL)
    # Set leverage
    try:
        exchange.set_leverage(
            leverage=get_market_data()[1],
            symbol=symbol
        )
        print(Fore.YELLOW +'Leverage set'+ Style.RESET_ALL)
    except:
        print(Fore.YELLOW +"Leverage not modified, current leverage is", get_market_data()[1])
        pass

get_balance()

max_trade_qty = round(
    (float(dex_equity) / float(get_orderbook()[1]))
    / (100 / float(get_market_data()[1])),
    int(float(get_market_data()[2])),
)

current_leverage = get_market_data()[1]

print(f"Min Trade Qty: {get_market_data()[2]}")
print(Fore.LIGHTYELLOW_EX+'1x:',max_trade_qty,' ')
print(Fore.LIGHTCYAN_EX+'0.01x ',round(max_trade_qty/100, int(float(get_market_data()[2]))),'')
print(
    f"0.005x : {round(max_trade_qty/200, int(float(get_market_data()[2])))}"
)
print(
    f"0.001x : {round(max_trade_qty/500, int(float(get_market_data()[2])))}"
)

# Fix for the first run when variable is not yet assigned
short_symbol_cum_realised=0
short_symbol_realised=0
short_pos_unpl=0
short_pos_unpl_pct=0

# high_vol_stack_trade_qty = (
#     trade_qty * 2
#     )

def long_mode_func(symbol):
    long_mode == True
    print(Fore.LIGHTCYAN_EX +"Long mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    #print(tylerapi.get_asset_total_volume_5m(symbol, tylerapi.api_data))

def short_mode_func(symbol):
    short_mode == True
    print(Fore.LIGHTCYAN_EX +"Short mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)

def hedge_mode_func(symbol):
    hedge_mode == True
    print(Fore.LIGHTCYAN_EX +"Hedge mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)

# Put trading logic here, in a loop with the table most likely as well in here
# The idea is to use things like and long_mode == True as a filter to create an algorithm with many modes


# Argument declaration
if args.mode == 'long_mode':
    if args.symbol:
        long_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')
elif args.mode == 'short_mode':
    if args.symbol:
        short_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')
elif args.mode == 'hedge_mode':
    if args.symbol:
        hedge_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')

if args.tg == 'on':
    if args.tg:
        print(Fore.LIGHTCYAN_EX +'TG Enabled'+ Style.RESET_ALL)
    else:
        print(Fore.LIGHTCYAN_EX +'TG Disabled'+ Style.RESET_ALL)