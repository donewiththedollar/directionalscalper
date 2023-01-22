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
version = "Directional Scalper v1.0.1"
long_mode = False
short_mode = False
hedge_mode = False
persistent_mode = False
longbias_mode = False
violent_mode = False
high_vol_stack_mode = False

#global symbol

print(Fore.LIGHTCYAN_EX +'',version,'connecting to exchange'+ Style.RESET_ALL)

min_volume = config.config_min_volume
min_distance = config.config_min_distance
high_volume = config.config_high_volume
high_distance = config.config_high_distance
violent_volume = config.config_violent_volume
violent_distance = config.config_violent_distance
botname = config.config_botname

exchange = ccxt.bybit(
    {"enableRateLimit": True, "apiKey": config.api_key, "secret": config.api_secret}
)

parser = argparse.ArgumentParser(description='Scalper supports 5 modes')

parser.add_argument('--mode', type=str, help='Mode to use', 
choices=['long', 
'short', 
'hedge',
'persistent',
'longbias'],
required=True)

parser.add_argument('--symbol', type=str, help='Specify symbol',
required=True)

parser.add_argument('--iqty', type=str, help="Initial entry quantity",
required=True)

parser.add_argument('--tg', type=str, help="TG Notifications",
choices=['on', 'off'],
required=True)

args = parser.parse_args()

if args.mode == 'long':
    long_mode = True
else:
    pass

if args.mode == 'short':
    short_mode = True
else:
    pass

if args.mode == 'hedge':
    hedge_mode = True
else:
    pass

if args.mode == 'persistent':
    persistent_mode = True
else:
    pass

if args.mode == 'longbias':
    longbias_mode = True
else:
    pass

if args.symbol:
    symbol= (args.symbol)
else:
    symbol = input('Instrument undefined. \nInput instrument:')

if args.iqty:
    trade_qty= (args.iqty)
else:
    trade_qty= input('Lot size:')

# Functions

# Get min vol data

# def get_min_vol_dist_data(symbol) -> bool:
#     spread5m = tylerapi.get_asset_5m_spread(symbol, tylerapi.api_data)
#     volume5m = tylerapi.get_asset_total_volume_5m(symbol, tylerapi.api_data)

#     return volume5m > min_volume and spread5m > min_distance

def get_min_vol_dist_data(symbol) -> bool:
    tylerapi.grab_api_data()
    spread5m = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())
    volume1m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())

    return volume1m > min_volume and spread5m > min_distance

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


def short_trade_condition():
    short_trade_condition = get_orderbook()[0] > get_1m_data()[0]
    return short_trade_condition

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

long_symbol_cum_realised=0
long_symbol_realised=0
long_pos_unpl=0
long_pos_unpl_pct=0


# Define Tyler API Func for ease of use later on
# Should turn these into functions and reduce calls

vol_condition_true = get_min_vol_dist_data(symbol) == True
tyler_total_volume_1m = tylerapi.get_asset_total_volume_1m(symbol,tylerapi.grab_api_data())
tyler_total_volume_5m = tylerapi.get_asset_total_volume_5m(symbol,tylerapi.grab_api_data())
#tyler_1x_volume_1m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
tyler_1x_volume_5m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
#tyler_5m_spread = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())
tyler_1m_spread = tylerapi.get_asset_1m_spread(symbol, tylerapi.grab_api_data())
#tyler_trend = tylerapi.get_asset_trend(symbol, tylerapi.grab_api_data())

def find_trend():
    tylerapi.grab_api_data()
    tyler_trend = tylerapi.get_asset_trend(symbol, tylerapi.grab_api_data())

    return tyler_trend

def find_5m_spread():
    tylerapi.grab_api_data()
    tyler_spread = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())

    return tyler_spread

def find_1m_1x_volume():
    tylerapi.grab_api_data()
    tyler_1x_volume_1m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())

    return tyler_1x_volume_1m

def find_mode():
    mode = args.mode

    return mode

# Generate table
def generate_table_vol() -> Table:
    table = Table(width=50)
    table.add_column("Condition", justify="center")
    table.add_column("Config", justify="center")
    table.add_column("Current", justify="center")
    table.add_column("Status")
    table.add_row(f"Trading:", str(get_min_vol_dist_data(symbol) == True), str(), "[green]:heavy_check_mark:" if get_min_vol_dist_data(symbol) else "off")
    table.add_row(f"Min Vol.", str(min_volume), str(tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())).split('.')[0], "[red]TOO LOW" if tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data()) < min_volume else "[green]VOL. OK")
    table.add_row()
    table.add_row(f"Min Dist.", str(min_distance), str(find_5m_spread()), "[red]TOO SMALL" if find_5m_spread() < min_distance else "[green]DIST. OK")
    table.add_row(f"Mode", str(find_mode()))
    # table.add_row(f"Long mode:", str(long_mode), str(), "[green]:heavy_check_mark:" if long_mode == True else "off")
    # table.add_row(f"Short mode:", str(short_mode), str(), "[green]:heavy_check_mark:" if short_mode == True else "off")
    # table.add_row(f"Hedge mode:", str(hedge_mode), str(), "[green]:heavy_check_mark:" if hedge_mode == True else "off")
#    table.add_row(f"Telegram:", str(tgnotif))
    return table

get_short_positions()
get_long_positions()

def generate_table_info() -> Table:
    table = Table(show_header=False, width=50)
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_row(f"Symbol", str(symbol))
    table.add_row(f"Balance", str(dex_wallet))
    table.add_row(f"Equity", str(dex_equity))
    table.add_row(f"Realised cum.", f"[red]{str(short_symbol_cum_realised)}" if short_symbol_cum_realised < 0 else f"[green]{str(short_symbol_cum_realised)}")
    table.add_row(f"Realised recent", f"[red]{str(short_symbol_realised)}" if short_symbol_realised < 0 else f"[green]{str(short_symbol_realised)}")
    table.add_row(f"Unrealised USDT", f"[red]{str(short_pos_unpl)}" if short_pos_unpl < 0 else f"[green]{str(short_pos_unpl + short_pos_unpl_pct)}")
    table.add_row(f"Unrealised %", f"[red]{str(short_pos_unpl_pct)}" if short_pos_unpl_pct < 0 else f"[green]{str(short_pos_unpl_pct)}")
    table.add_row(f"Entry size", str(trade_qty))
    table.add_row(f"Long size", str(long_pos_qty))
    table.add_row(f"Short size", str(short_pos_qty))
    table.add_row(f"Long pos price: ", str(long_pos_price))
    table.add_row(f"Long liq price", str(long_liq_price))
    table.add_row(f"Short pos price: ", str(short_pos_price))
    table.add_row(f"Short liq price", str(short_liq_price))
    table.add_row(f"Max", str(max_trade_qty))
    table.add_row(f"0.001x", str(round(max_trade_qty/500, int(float(get_market_data()[2])))))
    #table.add_row(f"Trend:", str(tyler_trend))
    table.add_row(f"Trend:", str(find_trend()))

    return table

def generate_main_table() -> Table:
    table = Table(show_header=False, box=None, title=version)
    table.add_row(generate_table_info()),
    table.add_row(generate_table_vol())
    return table

# Long entry logic if long enabled
def initial_long_entry(current_bid):
    if (
        #long_mode == True
        long_trade_condition() == True
        and find_1m_1x_volume() > min_volume
        and find_5m_spread() > min_distance
        and long_pos_qty == 0
        and long_pos_qty < max_trade_qty
        and find_trend() == 'short'
    ):
        try:
            exchange.create_limit_buy_order(
                symbol,
                trade_qty,
                current_bid
            )
            time.sleep(0.01)
        except:
            pass
    else:
        pass

# Short entry logic if short enabled
def initial_short_entry(current_ask):

    if (
        #short_mode == True
        short_trade_condition() == True
        and find_1m_1x_volume() > min_volume
        and find_5m_spread() > min_distance
        and short_pos_qty == 0
        and short_pos_qty < max_trade_qty
        and find_trend() == 'long'
    ):
        try:
            exchange.create_limit_sell_order(
                symbol,
                trade_qty,
                current_ask
            )
            time.sleep(0.01)
        except:
            pass
    else:
        pass


def trade_func(symbol):
    with Live(generate_main_table(), refresh_per_second=2) as live:
        while True:
            try:
                tylerapi.grab_api_data()
                time.sleep(0.01)
                get_1m_data()
                time.sleep(0.01)
                get_5m_data()
                time.sleep(0.01)
                get_balance()
                time.sleep(0.01)
                get_orderbook()
                time.sleep(0.01)
                long_trade_condition()
                time.sleep(0.01)
                short_trade_condition()
                time.sleep(0.01)
                get_short_positions()
                time.sleep(0.01)
                get_long_positions()
                time.sleep(0.01)

            except:
                pass

            try: 
                get_min_vol_dist_data(symbol)
                tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
                time.sleep(30)
            except:
                pass

            live.update(generate_main_table())
            current_bid = get_orderbook()[0]
            current_ask = get_orderbook()[1]
            long_open_pos_qty = long_pos_qty
            short_open_pos_qty = short_pos_qty
            reduce_only = {"reduce_only": True}

            short_profit_price = round(
                short_pos_price - (get_5m_data()[2] - get_5m_data()[3]),
                int(get_market_data()[0]),
            )

            long_profit_price = round(
                long_pos_price + (get_5m_data()[2] - get_5m_data()[3]),
                int(get_market_data()[0]),
            )

            add_trade_qty = trade_qty


            #Longbias mode
            if longbias_mode == True:
                try:
                    if find_trend() == 'long':
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                            and add_short_trade_condition() == True
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol,
                                    trade_qty,
                                    current_bid
                                )
                                time.sleep(0.01)
                            except:
                                pass
                except:
                    pass

            # Long entry logic if long enabled
            if (
                long_mode == True
                and long_trade_condition() == True
                and tyler_total_volume_5m > min_volume
                and find_5m_spread() > min_distance
                and long_pos_qty == 0
                and long_pos_qty < max_trade_qty
                and find_trend() == 'short'
            ):
                try:
                    exchange.create_limit_buy_order(
                        symbol,
                        trade_qty,
                        current_bid
                    )
                    time.sleep(0.01)
                except:
                    pass
            else:
                pass

            # Add to long if long enabled
            if (
                long_pos_qty != 0
                and short_pos_qty < max_trade_qty
                and long_mode == True
                and find_1m_1x_volume() > min_volume
                and add_long_trade_condition() == True
                and find_trend() == 'short'
            ):
                try:
                    cancel_entry()
                    time.sleep(0.05)
                except:
                    pass
                try:
                    exchange.create_limit_sell_order(symbol, add_trade_qty, current_bid)
                except:
                    pass


            # Short entry logic if short enabled
            if (
                short_mode == True
                and short_trade_condition() == True
                and tyler_total_volume_5m > min_volume
                and find_5m_spread() > min_distance
                and short_pos_qty == 0
                and short_pos_qty < max_trade_qty
                and find_trend() == 'long'
            ):
                try:
                    exchange.create_limit_sell_order(
                        symbol,
                        trade_qty,
                        current_ask
                    )
                    time.sleep(0.01)
                except:
                    pass
            else:
                pass

            # Add to short if short enabled
            if (
                short_pos_qty != 0
                #and short_pos_qty < max_trade_qty
                and short_mode == True
                and find_1m_1x_volume() > min_volume
                and add_short_trade_condition() == True
                and find_trend() == 'long'
            ):
                try:
                    cancel_entry()
                    time.sleep(0.05)
                except:
                    pass
                try:
                    exchange.create_limit_sell_order(symbol, add_trade_qty, current_ask)
                except:
                    pass

            #LONG: Take profit logic
            if long_pos_qty > 0:
                try:
                    get_open_orders()
                    time.sleep(0.05)
                except:
                    pass

                if long_profit_price != 0 or long_pos_price != 0:

                    try:
                        cancel_close()
                        time.sleep(0.05)
                    except:
                        pass
                    try:
                        exchange.create_limit_sell_order(
                            symbol, long_open_pos_qty, long_profit_price, reduce_only
                        )
                        time.sleep(0.05)
                    except:
                        pass

            #SHORT: Take profit logic
            if short_pos_qty > 0:
                try:
                    get_open_orders()
                    time.sleep(0.05)
                except:
                    pass

                if short_profit_price != 0 or short_pos_price != 0:

                    try:
                        cancel_close()
                        time.sleep(0.05)
                    except:
                        pass
                    try:
                        exchange.create_limit_buy_order(
                            symbol, short_open_pos_qty, short_profit_price, reduce_only
                        )
                        time.sleep(0.05)
                    except:
                        pass

            #HEDGE: Full mode
            if hedge_mode == True:
                try:
                    if find_trend() == 'long':
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                            and add_short_trade_condition() == True
                        ):
                            try:
                                exchange.create_limit_sell_order(
                                    symbol,
                                    trade_qty,
                                    current_ask
                                )
                                time.sleep(0.01)
                            except:
                                pass
                    elif find_trend() == 'short':
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and long_pos_qty < max_trade_qty
                            and add_long_trade_condition() == True
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol,
                                    trade_qty,
                                    current_bid
                                )
                                time.sleep(0.01)
                            except:
                                pass
                    if get_orderbook()[1] < get_1m_data()[0] or get_orderbook()[1] < get_5m_data()[0]:
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except:
                            pass
                except:
                    pass

            

            #PERSISTENT HEDGE: Full mode
            if persistent_mode == True:
                try:
                    if find_trend() == 'long':
                        if (
                            short_trade_condition() == True
                            and find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                        ):
                            try:
                                exchange.create_limit_sell_order(
                                    symbol,
                                    trade_qty,
                                    current_ask
                                )
                                time.sleep(0.01)
                            except:
                                pass
                    elif find_trend() == 'short':
                        if (
                            long_trade_condition() == True
                            and find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and long_pos_qty < max_trade_qty
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol,
                                    trade_qty,
                                    current_bid
                                )
                                time.sleep(0.01)
                            except:
                                pass
                    if get_orderbook()[1] < get_1m_data()[0] or get_orderbook()[1] < get_5m_data()[0]:
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except:
                            pass
                except:
                    pass


            if get_orderbook()[1] < get_1m_data()[0] or get_orderbook()[1] < get_5m_data()[0]:
                try:
                    cancel_entry()
                    time.sleep(0.05)
                except:
                    pass


# Mode functions
def long_mode_func(symbol):
    long_mode == True
    print(Fore.LIGHTCYAN_EX +"Long mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)
    #print(tylerapi.get_asset_total_volume_5m(symbol, tylerapi.api_data))

def short_mode_func(symbol):
    short_mode == True
    print(Fore.LIGHTCYAN_EX +"Short mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)

def hedge_mode_func(symbol):
    hedge_mode == True
    print(Fore.LIGHTCYAN_EX +"Hedge mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)

def persistent_mode_func(symbol):
    persistent_mode == True
    print(Fore.LIGHTCYAN_EX +"Persistent hedge mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)

def longbias_mode_func(symbol):
    longbias_mode == True
    print(Fore.LIGHTCYAN_EX +"Longbias mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)

# TO DO: 

# Add a terminal like console / hotkeys for entries

# Argument declaration
if args.mode == 'long':
    if args.symbol:
        long_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')
elif args.mode == 'short':
    if args.symbol:
        short_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')
elif args.mode == 'hedge':
    if args.symbol:
        hedge_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')
elif args.mode == 'persistent':
    if args.symbol:
        persistent_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')
elif args.mode == 'longbias':
    if args.symbol:
        longbias_mode_func(args.symbol)
    else:
        symbol = input('Instrument undefined. \nInput instrument:')

if args.tg == 'on':
    if args.tg:
        print(Fore.LIGHTCYAN_EX +'TG Enabled'+ Style.RESET_ALL)
    else:
        print(Fore.LIGHTCYAN_EX +'TG Disabled'+ Style.RESET_ALL)
