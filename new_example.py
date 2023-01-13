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
version = "1.0.5 alpha"
long_mode = False
short_mode = False
violent_mode = False
high_vol_stack_mode = False

#print(f"CCXT Scalper {version} connecting to exchange")
print(Fore.LIGHTCYAN_EX +'CCXT Scalper ',version,' connecting to exchange'+ Style.RESET_ALL)

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


print(tylerapi.get_asset_total_volume_5m('DOGEUSDT', tylerapi.api_data))

# Command line arguments
parser = argparse.ArgumentParser(description="CCXT Scalper",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--symbol", help="symbol")
parser.add_argument("--iqty", help="Value 0-1. Percentual initial quantity converted to min_lot_size")
parser.add_argument("--tgnotif", help="Telegram notifications enabled or disabled")
parser.add_argument("--longmode", help="Type --longmode true to enable long mode")
args = parser.parse_args()
config = vars(args)

if args.symbol:
    symbol = (args.symbol)#+'USDT')
else:
    symbol = input('Instrument undefined. \nInput instrument:')
if args.tgnotif:
    tgnotif = True
    print(Fore.GREEN +'Telegram enabled'+ Style.RESET_ALL)
    #notifier.notify_message(f"CCXT Scalper notifications enabled for {botname}")
else:
    tgnotif = False
    print(Fore.RED +'Telegram disabled'+ Style.RESET_ALL)

if args.longmode:
    long_mode = True
    print(Fore.GREEN +'Long mode: ON'+ Style.RESET_ALL)
else:
    long_mode = False
    print(Fore.RED +'Long mode: OFF'+ Style.RESET_ALL)

class VolumeData(TypedDict):
    symbol: str
    volume: float
    distance: float

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

def get_api_data() -> Dict[str, VolumeData]:
    data = requests.get("https://aadresearch.xyz/api/api_data.php").json()

    parsed = {}
    for x in data:
        vd: VolumeData = {"symbol": x[0], "volume": x[1], "distance": x[2]}
        parsed[vd["symbol"]] = vd
    return parsed

# def tyler_get_api_data() -> Dict[str, VolumeData]:
#     data = requests.get("http://13.127.240.18/data/quantdata.json")

#     parsed = {}
#     for x in data:
#         vd: TylerVolumeData = {"symbol": x[0], "volume": x[1], "distance": x[2]}
#         parsed[vd["symbol"]] = vd
#     return parsed    


def get_current_volume_data(symbol: str, volume_data: Dict[str, VolumeData]) -> bool:
    symbol_data = volume_data[symbol]
    return symbol_data["volume"]

def get_current_distance_data(symbol: str, volume_data: Dict[str, VolumeData]) -> bool:
    symbol_data = volume_data[symbol]
    return symbol_data ["distance"]

# volume > 25000 and distance > 0.25
def get_volume_data(symbol: str, volume_data: Dict[str, VolumeData]) -> bool:
    symbol_data = volume_data[symbol]
    return symbol_data["volume"] > min_volume and symbol_data["distance"] > min_distance
    #return symbol_data["volume"] > 15000 and symbol_data["distance"] > 0.15
    

def get_stack_volume_data(symbol: str, volume_data: Dict[str, VolumeData]) -> bool:
    symbol_data = volume_data[symbol]
    return symbol_data["volume"] > high_volume and symbol_data["distance"] > high_distance

def get_violent_volume_data(symbol: str, volume_data: Dict[str, VolumeData]) -> bool:
    symbol_data = volume_data[symbol]
    return symbol_data["volume"] > violent_volume and symbol_data["distance"] > violent_distance

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


def volume_condition():
    try:
        get_api_data()
        get_volume_data(symbol, get_api_data())
        return get_volume_data(symbol, get_api_data())
    except:
        return False

def high_vol_dist_condition():
    try:
        get_api_data()
        get_stack_volume_data(symbol, get_api_data())
        return get_stack_volume_data(symbol, get_api_data())
    except:
        return False

def violent_vol_dist_condition():
    try:
        get_api_data()
        get_violent_volume_data(symbol, get_api_data())
        return get_violent_volume_data(symbol, get_api_data())
    except:
        return False

def display_current_volume_data():
    try:
        get_api_data()
        get_current_volume_data(symbol, get_api_data())
        return get_current_volume_data(symbol, get_api_data())
    except:
        return False

def tyler_display_current_vol_data():
    try:
        tylerapi.get_asset_total_volume_5m(symbol, tylerapi.api_data)
        return tylerapi.get_asset_total_volume_5m(symbol, tylerapi.api_data())
    except:
        return False

def display_current_distance_data():
    try:
        get_api_data()
        get_current_distance_data(symbol, get_api_data())
        return get_current_distance_data(symbol, get_api_data())
    except:
        return False

def tg_notification(msg):
    if tgnotif == True:
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

# Ensure position mode is set to hedge
try:
    exchange.set_position_mode(
        hedged='BothSide',
        symbol=symbol)
    print(Fore.LIGHTYELLOW_EX +'Position mode changed to BothSide'+ Style.RESET_ALL)
except:
    print(Fore.YELLOW +'Position mode unchanged'+ Style.RESET_ALL)

# Attempt to set leverage
try:
    exchange.set_leverage(
        # buyLeverage = get_market_data()[1],
        # sellLeverage=get_market_data()[1],
        leverage = get_market_data()[1],
        symbol=symbol,
    )
    print(Fore.YELLOW +'Leverage set'+ Style.RESET_ALL)
except:
    print(Fore.YELLOW +"Leverage not modified, current leverage is", get_market_data()[1])

#Set margin mode
try:
    exchange.set_margin_mode(
        marginMode='cross',
        symbol=symbol
    )
    print(Fore.LIGHTYELLOW_EX +'Margin mode set to cross'+ Style.RESET_ALL)
except:
    print(Fore.YELLOW +'Margin mode unchanged'+ Style.RESET_ALL)

get_balance()

max_trade_qty = round(
    (float(dex_equity) / float(get_orderbook()[1]))
    / (100 / float(get_market_data()[1])),
    int(float(get_market_data()[2])),
)

current_leverage = get_market_data()[1]

#print(f"Current symbol: {symbol}")
#print(Fore.LIGHTWHITE_EX+'Current symbol:',symbol,' ')#+ Style.RESET_ALL)
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

if args.iqty:
    trade_qty = args.iqty
else:
    trade_qty = input("Input lot size:")

high_vol_stack_trade_qty = (
    trade_qty * 2
    )

vol_condition_true = volume_condition() == True


tyler_total_volume = tylerapi.get_asset_total_volume_5m(symbol,tylerapi.api_data)
tyler_5m_spread = tylerapi.get_asset_5m_spread(symbol, tylerapi.api_data)

def generate_table_vol() -> Table:
    table = Table(width=50)
    table.add_column("Condition", justify="center")
    table.add_column("Config", justify="center")
    table.add_column("Current", justify="center")
    table.add_column("Status")
    table.add_row(f"Min Vol.", str(min_volume), str(tyler_total_volume).split('.')[0], "[red]TOO LOW" if tyler_total_volume < min_volume else "[green]VOL. OK")
#    table.add_row(f"Min Vol.", str(min_volume), str(display_current_volume_data()).split('.')[0], "[red]TOO LOW" if display_current_volume_data() < min_volume else "[green]VOL. OK")
    table.add_row(f"High Vol.", str(high_volume), str(), "[green]:heavy_check_mark:" if tyler_total_volume > high_volume else "off")
    table.add_row(f"Violent Vol.", str(violent_volume), str(), "[green]:heavy_check_mark:" if tyler_total_volume > violent_volume else "off")
    table.add_row()
    table.add_row(f"Min Dist.", str(min_distance), str(tyler_5m_spread), "[red]TOO SMALL" if tyler_5m_spread < min_distance else "[green]DIST. OK")
    table.add_row(f"High Dist.", str(high_distance), str(), "[green]:heavy_check_mark:" if tyler_5m_spread > high_distance else "off")
    table.add_row(f"Violent Dist.", str(violent_distance), str(), "[green]:heavy_check_mark:" if display_current_distance_data() > violent_distance else "off")
    table.add_row(f"Trading:", str(volume_condition() == True), str(), "[green]:heavy_check_mark:" if volume_condition() else "off")
    table.add_row(f"Long mode:", str(long_mode), str(), "[green]:heavy_check_mark:" if long_mode == True else "off")
    table.add_row(f"Telegram:", str(tgnotif))
    return table

get_short_positions()
get_long_positions()

def generate_table_pos() -> Table:
    table = Table(show_header=False, width=50)
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_row(f"Symbol", str(symbol))
    table.add_row(f"Balance", str(dex_wallet))
    table.add_row(f"Equity", str(dex_equity))
    table.add_row(f"Leverage", str(get_market_data()[1]))
    table.add_row(f"Realised cum.", f"[red]{str(short_symbol_cum_realised)}" if short_symbol_cum_realised < 0 else f"[green]{str(short_symbol_cum_realised)}")
    table.add_row(f"Realised recent", f"[red]{str(short_symbol_realised)}" if short_symbol_realised < 0 else f"[green]{str(short_symbol_realised)}")
    table.add_row(f"Unrealised USDT", f"[red]{str(short_pos_unpl)}" if short_pos_unpl < 0 else f"[green]{str(short_pos_unpl + short_pos_unpl_pct)}")
    table.add_row(f"Unrealised %", f"[red]{str(short_pos_unpl_pct)}" if short_pos_unpl_pct < 0 else f"[green]{str(short_pos_unpl_pct)}")
    table.add_row(f"Long size", str(long_pos_qty))
    table.add_row(f"Short size", str(short_pos_qty))
    table.add_row(f"Max", str(max_trade_qty))
    table.add_row(f"0.001x", str(round(max_trade_qty/500, int(float(get_market_data()[2])))))
    table.add_row(f"Entry size", str(trade_qty))
    table.add_row(f"Long liq price", str(long_liq_price))
    table.add_row(f"Short liq price", str(short_liq_price))
    return table
#        print(Fore.LIGHTYELLOW_EX + 'Size / Max:',short_pos_qty,' | ',max_trade_qty,'')
def generate_main_table() -> Table:
    table = Table(show_header=False, box=None, title=version)
    table.add_row(generate_table_pos()),
    table.add_row(generate_table_vol())
    return table

with Live(generate_main_table(), refresh_per_second=2) as live:
    while True:
        try:
            get_1m_data()
            time.sleep(0.01)
            get_5m_data()
            time.sleep(0.01)
            get_balance()
            time.sleep(0.01)
            get_orderbook()
            time.sleep(0.01)
            trade_condition()
            time.sleep(0.01)
            long_trade_condition()
            time.sleep(0.01)
            get_short_positions()
            time.sleep(0.01)
            get_long_positions()
            time.sleep(0.01)
        except:
            pass

        live.update(generate_main_table())
        current_bid = get_orderbook()[0]
        current_ask = get_orderbook()[1]
        long_open_pos_qty = long_pos_qty
        open_pos_qty = short_pos_qty
        reduce_only = {"reduce_only": True}

        #Profit price - get position price - 5m MA6 high - 5m MA6 Low
        profit_price = round(
            short_pos_price - (get_5m_data()[2] - get_5m_data()[3]),
            int(get_market_data()[0]),
        )

        long_profit_price = round(
            long_pos_price + (get_5m_data()[2] - get_5m_data()[3]),
            int(get_market_data()[0]),
        )

        violent_trade_qty = (
            short_pos_qty
            * (get_1m_data()[3] - short_pos_price)
            / (get_orderbook()[1] - get_1m_data()[3])
        )

        long_violent_trade_qty = (
            long_pos_qty
            * (get_1m_data()[3] - long_pos_price)
            / (get_orderbook()[1] - get_1m_data()[3])
        )

        add_trade_qty = trade_qty

        half_max_size = max_trade_qty / 2

        #LONG: Initial entry logic
        if (
            long_mode == True
            #and volume_condition() == True
            and tyler_total_volume > min_volume
            and tyler_5m_spread > min_distance
            and long_pos_qty == 0
            and long_pos_qty < max_trade_qty
        ):
            try:
                exchange.create_limit_buy_order(symbol, trade_qty, current_bid)
                tg_notification("Long entry")
                time.sleep(0.01)
            except:
                pass
        else:
            pass

        #SHORT: Initial entry logic
        if (
            trade_condition() == True
            #and volume_condition() == True
            and tyler_total_volume > min_volume
            and tyler_5m_spread > min_distance
            and short_pos_qty == 0
            and short_pos_qty < max_trade_qty
        ):
            try:
                exchange.create_limit_sell_order(symbol, trade_qty, current_ask)
                time.sleep(0.01)
            except:
                pass
        else:
            pass

        if get_orderbook()[1] < get_1m_data()[0] or get_orderbook()[1] < get_5m_data()[0]:
            try:
                cancel_entry()
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

            if profit_price != 0 or short_pos_price != 0:

                try:
                    cancel_close()
                    time.sleep(0.05)
                except:
                    pass
                try:
                    exchange.create_limit_buy_order(
                        symbol, open_pos_qty, profit_price, reduce_only
                    )
                    time.sleep(0.05)
                except:
                    pass

        #LONG: Take profit logic
        if (
            long_mode == True
            and long_pos_qty > 0
        ):
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

        #LONG: Trade logic
        if (
            long_mode == True
            and long_pos_qty != 0
            #and volume_condition() == True
            and tyler_total_volume > min_volume
            and tyler_5m_spread > min_distance
            and add_long_trade_condition() == True
        ):
            try:
                cancel_entry()
                time.sleep(0.05)
            except:
                pass
            try:
                exchange.create_limit_buy_order(symbol, add_trade_qty, current_bid)
            except:
                pass

        # SHORT: Trade logic
        if (
            short_pos_qty != 0
            #and short_pos_qty < max_trade_qty
            #and volume_condition() == True
            and tyler_total_volume > min_volume
            and tyler_5m_spread > min_distance
            and add_short_trade_condition() == True
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

        # # SHORT: Stack violent orders if pos > 0 and pos > max_trade_qty and vol condition true, also added trade_condition() check to prevent violent bottom shorts
        # if (
        #     short_pos_qty != 0
        #     #and short_pos_qty > half_max_size
        #     and short_pos_qty < max_trade_qty
        #     #and violent_vol_dist_condition() == True
        #     and tyler_total_volume > min_volume
        #     and tyler_5m_spread > min_distance
        #     and add_short_trade_condition() == True
        # ):
        #     try:
        #         cancel_entry()
        #         time.sleep(0.05)
        #     except:
        #         pass
        #     try:
        #         exchange.create_limit_sell_order(symbol, violent_trade_qty, current_ask)
        #         violent_mode = True
        #     except:
        #         pass

        # LONG Stack violent orders if pos > 0 and pos > max_trade_qty and vol condition true, also added trade_condition() check to prevent violent bottom shorts
        # if (
        #     long_pos_qty != 0
        #     #and short_pos_qty > half_max_size
        #     and long_pos_qty < max_trade_qty
        #     and violent_vol_dist_condition() == True
        #     and add_long_trade_condition() == True
        # ):
        #     try:
        #         cancel_entry()
        #         time.sleep(0.05)
        #     except:
        #         pass
        #     try:
        #         exchange.create_limit_buy_order(symbol, long_violent_trade_qty, current_bid)
        #         violent_mode = True
        #     except:
        #         pass

        if violent_mode == True:
            try:
                tg_notification(f"Violent mode: ON for {botname} for {symbol}")
            except:
                pass
        # else:
        #     try:
        #         # print(Fore.LIGHTGREEN_EX +'Violent mode: OFF'+ Style.RESET_ALL)
        #     except:
        #         pass

        # # SHORT: Stack 2x your input lot size when vol is high
        # if (
        #     short_pos_qty != 0
        #     and high_vol_dist_condition() == True
        # ):
        #     try:
        #         cancel_entry()
        #         time.sleep(0.05)
        #     except:
        #         pass
        #     try:
        #         exchange.create_limit_sell_order(symbol, high_vol_stack_trade_qty, current_ask)
        #         high_vol_stack_mode = True
        #     except:
        #         pass

        # # LONG: Stack 2x your input lot size when vol is high
        # if (
        #     long_pos_qty != 0
        #     and high_vol_dist_condition() == True
        # ):
        #     try:
        #         cancel_entry()
        #         time.sleep(0.05)
        #     except:
        #         pass
        #     try:
        #         exchange.create_limit_buy_order(symbol, high_vol_stack_trade_qty, current_bid)
        #         high_vol_stack_mode = True
        #     except:
        #         pass
