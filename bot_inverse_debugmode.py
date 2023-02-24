import argparse
import decimal
import logging
import math
import os
import time
from pathlib import Path

import ccxt
import pandas as pd
import telebot
from colorama import Fore, Style
from pybit import inverse_perpetual
from rich.live import Live
from rich.table import Table

import tylerapi
from config import load_config

config_file = Path(Path().resolve(), "config.json")
config = load_config(path=config_file)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO"),
)

log = logging.getLogger(__name__)

endpoint = "https://api.bybit.com"
unauth = inverse_perpetual.HTTP(endpoint=endpoint)
invpcl = inverse_perpetual.HTTP(
    endpoint=endpoint,
    api_key=config.exchange_api_key,
    api_secret=config.exchange_api_secret,
)

telegram_output = False

bot = telebot.TeleBot("6079948538:AAFuDS2GfSrSNlplbWAb8mGyFcpyUhXcWMo", parse_mode=None)


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)


def sendmessage(message):
    bot.send_message(1281751562, message)


# Booleans
version = "Directional Scalper v1.0.4"
inverse_mode_version = "Inverse Perps Directional Scalper v1.0.4"
long_mode = False
short_mode = False
hedge_mode = False
persistent_mode = False
longbias_mode = False
inverse_mode = False
leverage_verified = False
inverse_trading_status = 0

limit_sell_order_id = 0

dex_balance, dex_pnl, dex_upnl, dex_wallet, dex_equity = 0, 0, 0, 0, 0
(
    long_pos_qty,
    short_pos_qty,
    long_pos_price,
    long_liq_price,
    short_pos_price,
    short_liq_price,
) = (0, 0, 0, 0, 0, 0)

(
    sell_position_prce,
    dex_btc_upnl,
    dex_btc_equity,
    dex_btc_balance,
    inv_perp_cum_realised_pnl,
    sell_position_size,
) = (0, 0, 0, 0, 0, 0)

print(Fore.LIGHTCYAN_EX + "", version, "connecting to exchange" + Style.RESET_ALL)

min_volume = config.config_min_volume
min_distance = config.config_min_distance
botname = config.config_botname

exchange = ccxt.bybit(
    {"enableRateLimit": True, "apiKey": config.api_key, "secret": config.api_secret}
)

parser = argparse.ArgumentParser(description="Scalper supports 6 modes")

parser.add_argument(
    "--mode",
    type=str,
    help="Mode to use",
    choices=["long", "short", "hedge", "persistent", "longbias", "inverse"],
    required=True,
)

parser.add_argument("--symbol", type=str, help="Specify symbol", required=True)

parser.add_argument("--iqty", type=str, help="Initial entry quantity", required=True)

parser.add_argument(
    "--tg", type=str, help="TG Notifications", choices=["on", "off"], required=True
)

args = parser.parse_args()

if args.mode == "long":
    long_mode = True
elif args.mode == "short":
    short_mode = True
elif args.mode == "hedge":
    hedge_mode = True
elif args.mode == "persistent":
    persistent_mode = True
elif args.mode == "longbias":
    longbias_mode = True

if args.mode == "inverse":
    inverse_mode = True
    # symbol = 'BTC/USD:BTC'

if args.symbol:
    symbol = args.symbol
else:
    symbol = input("Instrument undefined. \nInput instrument:")

if args.iqty:
    trade_qty = args.iqty
else:
    trade_qty = input("Lot size:")

if args.tg == "on" or "true":
    telegram_output = True
    if inverse_mode:
        sendmessage("Inverse scalper started")
    elif hedge_mode:
        sendmessage("Hedge mode enabled")
else:
    print("Telegram disabled")


# Functions


def get_min_vol_dist_data(symbol) -> bool:
    try:
        tylerapi.grab_api_data()
        spread5m = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())
        volume1m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())

        return volume1m > min_volume and spread5m > min_distance
    except Exception as e:
        log.warning(f"{e}")
        return False


def find_decimals(value):
    return abs(decimal.Decimal(str(value)).as_tuple().exponent)


def get_inverse_symbols():
    try:
        get_symbols = unauth.query_symbol()
        for asset in get_symbols["result"]:
            if asset["name"] == symbol:
                global price_scale, tick_size, min_price, min_trading_qty, qty_step
                price_scale = asset["price_scale"]
                tick_size = float(asset["price_filter"]["tick_size"])
                min_price = asset["price_filter"]["min_price"]
                min_trading_qty = asset["lot_size_filter"]["min_trading_qty"]
                qty_step = asset["lot_size_filter"]["qty_step"]
    except Exception as e:
        log.warning(f"{e}")


# Get inverse balance info, uPNL, cum pnl
def get_inverse_balance():
    get_inverse_balance = invpcl.get_wallet_balance(coin="BTC")
    global inv_perp_equity, inv_perp_available_balance, inv_perp_used_margin, inv_perp_order_margin, inv_perp_order_margin, inv_perp_position_margin, inv_perp_occ_closing_fee, inv_perp_occ_funding_fee, inv_perp_wallet_balance, inv_perp_realised_pnl, inv_perp_unrealised_pnl, inv_perp_cum_realised_pnl
    inv_perp_equity = get_inverse_balance["result"]["BTC"]["equity"]
    inv_perp_available_balance = get_inverse_balance["result"]["BTC"][
        "available_balance"
    ]
    inv_perp_used_margin = get_inverse_balance["result"]["BTC"]["used_margin"]
    inv_perp_order_margin = get_inverse_balance["result"]["BTC"]["order_margin"]
    inv_perp_position_margin = get_inverse_balance["result"]["BTC"]["position_margin"]
    inv_perp_occ_closing_fee = get_inverse_balance["result"]["BTC"]["occ_closing_fee"]
    inv_perp_occ_funding_fee = get_inverse_balance["result"]["BTC"]["occ_funding_fee"]
    inv_perp_wallet_balance = get_inverse_balance["result"]["BTC"]["wallet_balance"]
    inv_perp_realised_pnl = get_inverse_balance["result"]["BTC"]["realised_pnl"]
    inv_perp_unrealised_pnl = get_inverse_balance["result"]["BTC"]["unrealised_pnl"]
    inv_perp_cum_realised_pnl = get_inverse_balance["result"]["BTC"]["cum_realised_pnl"]


def get_inverse_sell_position():
    try:
        position = invpcl.my_position(symbol=symbol)
        # print(position)
        if position["result"]["side"] == "None":
            global sell_position_size, sell_position_prce
            sell_position_size = 0
            sell_position_prce = 0
        if position["result"]["side"] == "Sell":
            sell_position_size = float(position["result"]["size"])
            sell_position_prce = float(position["result"]["entry_price"])
    except Exception as e:
        log.warning(f"{e}")


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


def inverse_get_balance():
    global dex_btc_balance, dex_btc_upnl, dex_btc_wallet, dex_btc_equity
    try:
        dex = exchange.fetch_balance()["info"]["result"]
        dex_btc_balance = dex["BTC"]["available_balance"]
        # dex_btc_pnl = dex["BTC"]["realised_pnl"]
        dex_btc_upnl = dex["BTC"]["unrealised_pnl"]
        dex_btc_wallet = round(float(dex["BTC"]["wallet_balance"]), 8)
        dex_btc_equity = round(float(dex["BTC"]["equity"]), 8)
    except KeyError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print("An unknown error occured in get_balance()")
        log.warning(f"{e}")


def get_balance():
    global dex_balance, dex_upnl, dex_wallet, dex_equity
    try:
        dex = exchange.fetch_balance()["info"]["result"]
        dex_balance = dex["USDT"]["available_balance"]
        # dex_pnl = dex["USDT"]["realised_pnl"]
        dex_upnl = dex["USDT"]["unrealised_pnl"]
        dex_wallet = round(float(dex["USDT"]["wallet_balance"]), 2)
        dex_equity = round(float(dex["USDT"]["equity"]), 2)
    except KeyError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print("An unknown error occured in get_balance()")
        log.warning(f"{e}")


# get_orderbook() [0]bid, [1]ask
def get_orderbook():
    ob = exchange.fetch_order_book(symbol)
    bid = ob["bids"][0][0]
    ask = ob["asks"][0][0]
    return bid, ask


get_orderbook()


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
    if not inverse_mode:
        global short_pos_qty, short_pos_price, short_symbol_realised, short_symbol_cum_realised, short_pos_unpl, short_pos_unpl_pct, short_liq_price
        pos_dict = exchange.fetch_positions([symbol])
        pos_dict = pos_dict[1]
        short_pos_qty = float(pos_dict["contracts"])
        short_symbol_realised = round(float(pos_dict["info"]["realised_pnl"] or 0), 2)
        short_symbol_cum_realised = round(
            float(pos_dict["info"]["cum_realised_pnl"] or 0), 2
        )
        short_pos_unpl = round(float(pos_dict["info"]["unrealised_pnl"] or 0), 2)
        short_pos_unpl_pct = round(float(pos_dict["percentage"] or 0), 2)
        short_pos_price = pos_dict["entryPrice"] or 0
        short_liq_price = pos_dict["liquidationPrice"] or 0


def get_long_positions():
    if not inverse_mode:
        global long_pos_qty, long_pos_price, long_symbol_realised, long_symbol_cum_realised, long_pos_unpl, long_pos_unpl_pct, long_liq_price
        pos_dict = exchange.fetch_positions(
            [symbol]
        )  # TODO: We can fetch it just once to save some API time
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
        and reduce_only
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
        and not reduce_only
    ):
        exchange.cancel_order(symbol=symbol, id=order_id)
    elif (
        order_status != "Filled"
        and order_side == "Sell"
        and order_status != "Cancelled"
        and not reduce_only
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
        and reduce_only
    ):
        exchange.cancel_order(symbol=symbol, id=order_id)
    elif (
        order_status != "Filled"
        and order_side == "Sell"
        and order_status != "Cancelled"
        and reduce_only
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


def inverse_short_trade_condition():
    inverse_short_trade_condition = get_orderbook()[0] > get_1m_data()[0]
    return inverse_short_trade_condition


def add_inverse_short_trade_condition():
    add_inverse_short_trade_condition = sell_position_prce < get_1m_data()[3]
    return add_inverse_short_trade_condition


def leverage_verification(symbol):
    if not inverse_mode:
        try:
            exchange.set_position_mode(hedged="BothSide", symbol=symbol)
            print(
                Fore.LIGHTYELLOW_EX
                + "Position mode changed to BothSide"
                + Style.RESET_ALL
            )
        except Exception as e:
            print(Fore.YELLOW + "Position mode unchanged" + Style.RESET_ALL)
            log.warning(f"{e}")
        # Set margin mode
        if not inverse_mode:
            try:
                exchange.set_margin_mode(marginMode="cross", symbol=symbol)
                print(
                    Fore.LIGHTYELLOW_EX + "Margin mode set to cross" + Style.RESET_ALL
                )
            except Exception as e:
                print(Fore.YELLOW + "Margin mode unchanged" + Style.RESET_ALL)
                log.warning(f"{e}")
        # Set leverage
        try:
            exchange.set_leverage(leverage=get_market_data()[1], symbol=symbol)
            print(Fore.YELLOW + "Leverage set" + Style.RESET_ALL)
        except Exception as e:
            print(
                Fore.YELLOW + "Leverage not modified, current leverage is",
                get_market_data()[1],
            )
            log.warning(f"{e}")


if not leverage_verified and not inverse_mode:
    try:
        leverage_verification(symbol)
    except KeyError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print("An unknown error occured in leverage verification")
        log.warning(f"{e}")


# get_inverse_balance()
if not inverse_mode:
    get_balance()
    max_trade_qty = round(
        (float(dex_equity) / float(get_orderbook()[1]))
        / (100 / float(get_market_data()[1])),
        int(float(get_market_data()[2])),
    )

    current_leverage = get_market_data()[1]

    print(f"Min Trade Qty: {get_market_data()[2]}")
    print(Fore.LIGHTYELLOW_EX + "1x:", max_trade_qty, " ")
    print(
        Fore.LIGHTCYAN_EX + "0.01x ",
        round(max_trade_qty / 100, int(float(get_market_data()[2]))),
        "",
    )
    print(f"0.005x : {round(max_trade_qty/200, int(float(get_market_data()[2])))}")
    print(f"0.001x : {round(max_trade_qty/500, int(float(get_market_data()[2])))}")
elif inverse_mode:
    inverse_get_balance()
    get_inverse_sell_position()  # Pybit
    get_inverse_balance()  # Pybit
    tp_price = float((100 - config.min_fee) * sell_position_prce / 100)
    tp_price = math.ceil(tp_price * 2) / 2

# Fix for the first run when variable is not yet assigned
short_symbol_cum_realised = 0
short_symbol_realised = 0
short_pos_unpl = 0
short_pos_unpl_pct = 0

long_symbol_cum_realised = 0
long_symbol_realised = 0
long_pos_unpl = 0
long_pos_unpl_pct = 0


# Define Tyler API Func for ease of use later on
# Should turn these into functions and reduce calls

# vol_condition_true = get_min_vol_dist_data(symbol)
# tyler_total_volume_1m = tylerapi.get_asset_total_volume_1m(symbol,tylerapi.grab_api_data())
tyler_total_volume_5m = tylerapi.get_asset_total_volume_5m(
    symbol, tylerapi.grab_api_data()
)
# #tyler_1x_volume_1m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
# tyler_1x_volume_5m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
# #tyler_5m_spread = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())
# tyler_1m_spread = tylerapi.get_asset_1m_spread(symbol, tylerapi.grab_api_data())
# #tyler_trend = tylerapi.get_asset_trend(symbol, tylerapi.grab_api_data())


def find_trend():
    tylerapi.grab_api_data()
    tyler_trend = tylerapi.get_asset_trend(symbol, tylerapi.grab_api_data())

    return tyler_trend


def find_1m_spread():
    tylerapi.grab_api_data()
    tyler_1m_spread = tylerapi.get_asset_1m_spread(symbol, tylerapi.get_api_data())

    return tyler_1m_spread


def find_5m_spread():
    tylerapi.grab_api_data()
    tyler_spread = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())

    return tyler_spread


def find_1m_1x_volume():
    tylerapi.grab_api_data()
    tyler_1x_volume_1m = tylerapi.get_asset_volume_1m_1x(
        symbol, tylerapi.grab_api_data()
    )

    return tyler_1x_volume_1m


def find_mode():
    mode = args.mode

    return mode


# Load all necessary data for the road ahead
if inverse_mode:
    try:
        inverse_get_balance()
        get_inverse_symbols()
        get_inverse_sell_position()
        dex_btc_upnl_pct = round((float(dex_btc_upnl) / float(dex_btc_equity)) * 100, 2)
    except Exception as e:
        log.warning(f"{e}")
elif not inverse_mode:
    try:
        get_short_positions()
        get_long_positions()
    except Exception as e:
        log.warning(f"{e}")


# Long entry logic if long enabled
def initial_long_entry(current_bid):
    if (
        long_trade_condition()
        and find_1m_1x_volume() > min_volume
        and find_5m_spread() > min_distance
        and long_pos_qty == 0
        and long_pos_qty < max_trade_qty
        and find_trend() == "short"
    ):
        try:
            exchange.create_limit_buy_order(symbol, trade_qty, current_bid)
            time.sleep(0.01)
        except Exception as e:
            log.warning(f"{e}")


# Short entry logic if short enabled
def initial_short_entry(current_ask):
    if (
        # short_mode
        short_trade_condition()
        and find_1m_1x_volume() > min_volume
        and find_5m_spread() > min_distance
        and short_pos_qty == 0
        and short_pos_qty < max_trade_qty
        and find_trend() == "long"
    ):
        try:
            exchange.create_limit_sell_order(symbol, trade_qty, current_ask)
            time.sleep(0.01)
        except Exception as e:
            log.warning(f"{e}")


def calc_tp_price():
    try:
        get_inverse_sell_position()
        tp_price = float((100 - config.min_fee) * sell_position_prce / 100)
        tp_price = math.ceil(tp_price * 2) / 2

        # print("TP price:", tp_price)

        return tp_price
    except Exception as e:
        print("Calc TP exception")
        log.warning(f"{e}")


def inverse_cancel_orders():
    try:
        if limit_sell_order_id != 0:
            try:
                invpcl.cancel_active_order(symbol=symbol, order_id=limit_sell_order_id)
            except Exception as e:
                log.warning(f"{e}")
    except Exception as e:
        log.warning(f"{e}")


def inverse_limit_short_with_cancel_order(current_ask):
    open_orders = exchange.fetch_open_orders(symbol)
    if len(open_orders) > 0:
        print("Debug: Cancelling open orders...")
        for order in open_orders:
            exchange.cancel_order(order["id"], symbol)
    try:
        get_orderbook()
        invpcl.place_active_order(
            side="Sell",
            symbol=symbol,
            order_type="Limit",
            qty=trade_qty,
            price=current_ask,
            reduce_only=False,
            time_in_force="GoodTillCancel",
            close_on_trigger=False,
            post_only=True,
        )
        print("Debug: Limit short placed")
        sendmessage("Limit short placed")
    except Exception as e:
        log.warning(f"{e}")


def inverse_limit_short(current_ask):
    try:
        get_orderbook()
        invpcl.place_active_order(
            side="Sell",
            symbol=symbol,
            order_type="Limit",
            qty=trade_qty,
            price=current_ask,
            reduce_only=False,
            time_in_force="GoodTillCancel",
            close_on_trigger=False,
            post_only=True,
        )
    except Exception as e:
        log.warning(f"{e}")


def inverse_initial_short_entry(current_ask):
    if (
        short_trade_condition()
        and find_5m_spread() > config.min_distance
        and find_1m_1x_volume() > config.min_volume
        and sell_position_size == 0
        and find_trend() == "long"
    ):
        try:
            inverse_cancel_orders()

        except Exception as e:
            log.warning(f"{e}")


order_ids: list = []


def place_new_limit_short(
    price,
):  # Place new order selecting entry price (current_ask for short)
    global order_ids
    try:
        # get_orderbook()
        # current_bid = get_orderbook()[0]
        # current_ask = get_orderbook()[1]
        for order_id in order_ids:
            invpcl.cancel_active_order(
                symbol=symbol,
                order_id=order_id,
            )
        order = invpcl.place_active_order(
            side="Sell",
            symbol=symbol,
            order_type="Limit",
            # qty = csize,
            qty=trade_qty,
            price=price,
            reduce_only=False,
            time_in_force="GootTillCancel",
            close_on_trigger=False,
            post_only=False,
        )
        order_id = order["result"]["order_id"]
        order_ids.append(order_id)
        time.sleep(0.05)
    except Exception as e:
        print("Error in placing limit short")
        log.warning(f"{e}")


def place_new_market_short():
    try:
        invpcl.place_active_order(
            side="Sell",
            symbol=symbol,
            order_type="Market",
            qty=trade_qty,
            time_in_force="GoodTillCancel",
            reduce_only=True,
            close_on_trigger=True,
        )
        print("Initial market short placed")
    except Exception as e:
        print("Market short exception")
        log.warning(f"{e}")


global lot_size_market_tp
# global tp_price


get_inverse_sell_position()
inverse_get_balance()


def generate_inverse_table_vol() -> Table:
    inverse_table = Table(width=50)
    inverse_table.add_column("Condition", justify="center")
    inverse_table.add_column("Config", justify="center")
    inverse_table.add_column("Current", justify="center")
    inverse_table.add_column("Status")
    inverse_table.add_row(
        "Trading:",
        str(get_min_vol_dist_data(symbol)),
        str(),
        "[green]:heavy_check_mark:" if get_min_vol_dist_data(symbol) else "off",
    )
    inverse_table.add_row(
        "Min Vol.",
        str(min_volume),
        str(tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())).split(
            "."
        )[0],
        "[red]TOO LOW"
        if tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
        < min_volume
        else "[green]VOL. OK",
    )
    inverse_table.add_row()
    inverse_table.add_row(
        "Min Dist.",
        str(min_distance),
        str(find_5m_spread()),
        "[red]TOO SMALL" if find_5m_spread() < min_distance else "[green]DIST. OK",
    )
    inverse_table.add_row(f"Mode {find_mode()}")

    return inverse_table


#  global dex_btc_balance, dex_btc_upnl, dex_btc_wallet, dex_btc_equity
#    global inv_perp_equity, inv_perp_available_balance, inv_perp_used_margin, inv_perp_order_margin,
# inv_perp_order_margin, inv_perp_position_margin, inv_perp_occ_closing_fee, inv_perp_occ_funding_fee,
#  inv_perp_wallet_balance, inv_perp_realised_pnl, inv_perp_unrealised_pnl, inv_perp_cum_realised_pnl
# global sell_position_size, sell_position_prce
# global dex_btc_balance, dex_btc_upnl, dex_btc_wallet, dex_btc_equity


def generate_inverse_table_info() -> Table:
    inverse_table = Table(show_header=False, width=50)
    inverse_table.add_column(justify="center")
    inverse_table.add_column(justify="center")
    inverse_table.add_row(f"Symbol {symbol}")
    inverse_table.add_row(f"Balance {dex_btc_balance}")
    inverse_table.add_row(f"Equity {dex_btc_equity}")
    # inverse_table.add_row(f"Realised cum.", f"[red]{str(inv_perp_cum_realised_pnl)}" if inv_perp_cum_realised_pnl < 0 else f"[green]{str(short_symbol_cum_realised)}")
    inverse_table.add_row(
        "Realised cum.",
        f"[red]{format(inv_perp_cum_realised_pnl, '.8f')}"
        if inv_perp_cum_realised_pnl < 0
        else f"[green]{format(inv_perp_cum_realised_pnl, '.8f')}",
    )
    # inverse_table.add_row(f"Unrealized PNL.", f"[red]{dex_btc_upnl}" if dex_btc_upnl < 0 else f"[green]{dex_btc_upnl}")
    # inverse_table.add_row(f"Realised recent", f"[red]{str(inv_perp_realised_pnl)}" if inv_perp_realised_pnl < 0 else f"[green]{str(inv_perp_realised_pnl)}")
    # inverse_table.add_row(f"Unrealised BTC", f"[red]{str(inv_perp_unrealised_pnl)}" if inv_perp_unrealised_pnl < 0 else f"[green]{str(short_pos_unpl + short_pos_unpl_pct)}")
    # inverse_table.add_row(f"Unrealised BTC", f"[red]{str(dex_btc_upnl)}" if dex_btc_upnl < 0 else f"[green]{str(dex_btc_upnl + dex_btc_upnl_pct)}")
    inverse_table.add_row(
        "Unrealised %",
        f"[red]{dex_btc_upnl_pct}"
        if dex_btc_upnl_pct < 0
        else f"[green]{dex_btc_upnl_pct}",
    )
    inverse_table.add_row("Entry size", str(trade_qty))
    inverse_table.add_row("Short pos size", str(sell_position_size))
    inverse_table.add_row("Trend:", str(find_trend()))
    inverse_table.add_row("Entry price", str(sell_position_prce))
    inverse_table.add_row("Take profit", str(calc_tp_price()))
    # inverse_table.add_row(f"Bid:", str)
    return inverse_table


# Generate table
def generate_table_vol() -> Table:
    table = Table(width=50)
    table.add_column("Condition", justify="center")
    table.add_column("Config", justify="center")
    table.add_column("Current", justify="center")
    table.add_column("Status")
    table.add_row(
        "Trading:",
        str(get_min_vol_dist_data(symbol)),
        str(),
        "[green]:heavy_check_mark:" if get_min_vol_dist_data(symbol) else "off",
    )
    table.add_row(
        "Min Vol.",
        str(min_volume),
        str(tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())).split(
            "."
        )[0],
        "[red]TOO LOW"
        if tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
        < min_volume
        else "[green]VOL. OK",
    )
    table.add_row()
    table.add_row(
        "Min Dist.",
        str(min_distance),
        str(find_5m_spread()),
        "[red]TOO SMALL" if find_5m_spread() < min_distance else "[green]DIST. OK",
    )
    table.add_row("Mode", str(find_mode()))
    # table.add_row(f"Long mode:", str(long_mode), str(), "[green]:heavy_check_mark:" if long_mode else "off")
    # table.add_row(f"Short mode:", str(short_mode), str(), "[green]:heavy_check_mark:" if short_mode else "off")
    # table.add_row(f"Hedge mode:", str(hedge_mode), str(), "[green]:heavy_check_mark:" if hedge_mode else "off")
    #    table.add_row(f"Telegram:", str(tgnotif))
    return table


def generate_table_info() -> Table:
    table = Table(show_header=False, width=50)
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_row("Symbol", str(symbol))
    table.add_row("Balance", str(dex_wallet))
    table.add_row("Equity", str(dex_equity))
    table.add_row(
        "Realised cum.",
        f"[red]{str(short_symbol_cum_realised)}"
        if short_symbol_cum_realised < 0
        else f"[green]{str(short_symbol_cum_realised)}",
    )
    table.add_row(
        "Realised recent",
        f"[red]{str(short_symbol_realised)}"
        if short_symbol_realised < 0
        else f"[green]{str(short_symbol_realised)}",
    )
    table.add_row(
        "Unrealised USDT",
        f"[red]{str(short_pos_unpl)}"
        if short_pos_unpl < 0
        else f"[green]{str(short_pos_unpl + short_pos_unpl_pct)}",
    )
    table.add_row(
        "Unrealised %",
        f"[red]{str(short_pos_unpl_pct)}"
        if short_pos_unpl_pct < 0
        else f"[green]{str(short_pos_unpl_pct)}",
    )
    table.add_row("Entry size", str(trade_qty))
    table.add_row("Long size", str(long_pos_qty))
    table.add_row("Short size", str(short_pos_qty))
    table.add_row("Long pos price: ", str(long_pos_price))
    table.add_row("Long liq price", str(long_liq_price))
    table.add_row("Short pos price: ", str(short_pos_price))
    table.add_row("Short liq price", str(short_liq_price))
    table.add_row("Max", str(max_trade_qty))
    table.add_row(
        "0.001x", str(round(max_trade_qty / 500, int(float(get_market_data()[2]))))
    )
    # table.add_row(f"Trend:", str(tyler_trend))
    table.add_row("Trend:", str(find_trend()))

    return table


def generate_main_table() -> Table:
    if not inverse_mode:
        table = Table(show_header=False, box=None, title=version)
        table.add_row(generate_table_info()),
        table.add_row(generate_table_vol())
        return table
    elif inverse_mode:
        inverse_table = Table(show_header=False, box=None, title=inverse_mode_version)
        inverse_table.add_row(generate_inverse_table_info()),
        inverse_table.add_row(generate_inverse_table_vol())
        return inverse_table


def trade_func(symbol):  # noqa
    with Live(generate_main_table(), refresh_per_second=2) as live:
        while True:
            try:
                if inverse_mode:
                    get_inverse_balance()
                    time.sleep(0.01)
                    get_inverse_sell_position()
                    time.sleep(0.01)
                    get_inverse_symbols()
                    time.sleep(0.01)
                    inverse_get_balance()
                    time.sleep(0.01)
                    get_orderbook()
                    time.sleep(0.01)
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
                if not inverse_mode:
                    get_short_positions()
                    time.sleep(0.01)
                    get_long_positions()
                    time.sleep(0.01)
            except Exception as e:
                log.warning(f"{e}")

            try:
                get_min_vol_dist_data(symbol)
                tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
                time.sleep(30)
            except Exception as e:
                log.warning(f"{e}")

            if not inverse_mode:
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
            elif inverse_mode:
                live.update(generate_main_table())
                find_decimals(min_trading_qty)
                decimal_for_tp_size = find_decimals(min_trading_qty)
                get_orderbook()
                current_bid = get_orderbook()[0]
                current_ask = get_orderbook()[1]
                # print("Current bid", current_bid)

                reduce_only = {"reduce_only": True}

                get_inverse_sell_position()
                # inverse_short_profit_price = round(
                #     sell_position_prce - (get_5m_data()[2] - get_5m_data()[3]),
                #     int(get_market_data()[0]),
                # )
                # print("Short profit price:", inverse_short_profit_price)

                # calculated_tp_price = calc_tp_price()

                # print("Recalculated TP price {calculated_tp_price}")

                if sell_position_size / config.divider < min_trading_qty:
                    lot_size_market_tp = sell_position_size
                    print(f"Market TP size (1): {lot_size_market_tp}")

                if (
                    sell_position_size / config.divider
                    < min_trading_qty * config.divider
                ):
                    lot_size_market_tp = sell_position_size
                    print(f"Market TP size (2): {lot_size_market_tp}")

                else:
                    lot_size_market_tp = round(
                        (sell_position_size / config.divider), decimal_for_tp_size
                    )
                    print("Market TP size (3):", lot_size_market_tp)

                # tp_price = float((100 - config.min_fee) * sell_position_prce / 100)
                # tp_price = math.ceil(tp_price * 2) / 2

                # print("TP price:", tp_price)
                # get_5m_data() [0]3 high, [1]3 low, [2]6 high, [3]6 low
                # ma6high = get_5m_data()[2]
                # ma6low = get_5m_data()[3]
                # avr_price = (ma6high + ma6low) / 2
                # print("MA6 High:", ma6high)
                # print("MA6 Low:", ma6low)
                # print("Avg price:", avr_price)

            # Longbias mode
            if longbias_mode:
                try:
                    if find_trend() == "long":
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                            and add_short_trade_condition()
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, trade_qty, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            # Long entry logic if long enabled
            if (
                long_mode
                and long_trade_condition()
                and tyler_total_volume_5m > min_volume
                and find_5m_spread() > min_distance
                and long_pos_qty == 0
                and long_pos_qty < max_trade_qty
                and find_trend() == "short"
            ):
                try:
                    exchange.create_limit_buy_order(symbol, trade_qty, current_bid)
                    time.sleep(0.01)
                except Exception as e:
                    log.warning(f"{e}")

            # Add to long if long enabled
            if (
                not inverse_mode
                and long_pos_qty != 0
                and short_pos_qty < max_trade_qty
                and long_mode
                and find_1m_1x_volume() > min_volume
                and add_long_trade_condition()
                and find_trend() == "short"
            ):
                try:
                    cancel_entry()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")
                try:
                    exchange.create_limit_sell_order(symbol, add_trade_qty, current_bid)
                except Exception as e:
                    log.warning(f"{e}")

            # Short entry logic if short enabled
            if (
                short_mode
                and short_trade_condition()
                and tyler_total_volume_5m > min_volume
                and find_5m_spread() > min_distance
                and short_pos_qty == 0
                and short_pos_qty < max_trade_qty
                and find_trend() == "long"
            ):
                try:
                    exchange.create_limit_sell_order(symbol, trade_qty, current_ask)
                    time.sleep(0.01)
                except Exception as e:
                    log.warning(f"{e}")

            # Add to short if short enabled
            if (
                not inverse_mode
                and short_pos_qty != 0
                # and short_pos_qty < max_trade_qty
                and short_mode
                and find_1m_1x_volume() > min_volume
                and add_short_trade_condition()
                and find_trend() == "long"
            ):
                try:
                    cancel_entry()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")
                try:
                    exchange.create_limit_sell_order(symbol, add_trade_qty, current_ask)
                except Exception as e:
                    log.warning(f"{e}")

            # LONG: Take profit logic
            if not inverse_mode:
                if long_pos_qty > 0:
                    try:
                        get_open_orders()
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")

                    if long_profit_price != 0 or long_pos_price != 0:
                        try:
                            cancel_close()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                        try:
                            exchange.create_limit_sell_order(
                                symbol,
                                long_open_pos_qty,
                                long_profit_price,
                                reduce_only,
                            )
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")

                # SHORT: Take profit logic
                if short_pos_qty > 0:
                    try:
                        get_open_orders()
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")

                    if short_profit_price != 0 or short_pos_price != 0:
                        try:
                            cancel_close()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                        try:
                            exchange.create_limit_buy_order(
                                symbol,
                                short_open_pos_qty,
                                short_profit_price,
                                reduce_only,
                            )
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")

            # Inverse perps BTCUSD only
            if inverse_mode:
                try:
                    get_inverse_sell_position()
                except Exception as e:
                    log.warning(f"{e}")
                # limit_sell_order_id = 0
                # First entry
                if sell_position_size == 0 and sell_position_prce == 0:
                    try:
                        inverse_limit_short_with_cancel_order(current_ask)
                    except Exception as e:
                        log.warning(f"{e}")

                # Additional entry
                if (
                    sell_position_size > 0
                    and inverse_short_trade_condition()
                    and find_trend() == "long"
                ):
                    try:
                        # inverse_limit_short(current_ask)
                        inverse_limit_short_with_cancel_order(current_ask)
                    except Exception as e:
                        log.warning(f"{e}")
                else:
                    print("Not time for additional entry, condition not met yet")

                # Take profit for inverse
                if sell_position_size > 0:
                    try:
                        # get_orderbook()
                        # current_bid = get_orderbook()[0]
                        # current_ask = get_orderbook()[1]
                        if float(current_bid) < float(calc_tp_price()):
                            try:
                                get_inverse_sell_position()
                                # Take profit logic first
                                invpcl.place_active_order(
                                    side="Buy",
                                    symbol=symbol,
                                    order_type="Market",
                                    qty=lot_size_market_tp,
                                    time_in_force="GoodTilCancel",
                                    reduce_only=True,
                                    close_on_trigger=True,
                                )
                                print(f"Placed order at: {calc_tp_price()}")
                                sendmessage("Market take profit placed")
                            except Exception as e:
                                print("Error in placing TP")
                                log.warning(f"{e}")
                        else:
                            print("You have position but not time for TP")
                            print(f"Current bid {current_bid}")
                    except Exception as e:
                        log.warning(f"{e}")

                try:
                    if (
                        get_orderbook()[1] < get_1m_data()[0]
                        or get_orderbook()[1] < get_5m_data()[0]
                    ):
                        try:
                            cancel_entry()
                            print("Canceled entry")
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            # HEDGE: Full mode
            # We find if trend is long or short
            # places initial short entry
            # Checks if volume is ok and spread is ok,
            # Checks position quantity and comapres to max trade qty
            if hedge_mode:
                try:
                    if find_trend() == "long":
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                            and add_short_trade_condition()
                        ):
                            try:
                                exchange.create_limit_sell_order(
                                    symbol, trade_qty, current_ask
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    elif find_trend() == "short":
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and long_pos_qty < max_trade_qty
                            and add_long_trade_condition()
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, trade_qty, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    if (
                        get_orderbook()[1] < get_1m_data()[0]
                        or get_orderbook()[1] < get_5m_data()[0]
                    ):
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            # if inverse_mode:
            #     try:
            #         if sell_position_size == 0 and sell_position_prce == 0:
            #             # Cancel sell limit first if it exists
            #             if limit_sell_order_id !=0:
            #                 try:
            #                     cancel_limit_sell_entry = invpcl.cancel_active_order(
            #                         symbol = symbol,
            #                         order_id = limit_sell_order_id
            #                     )
            #                 except Exception as e:
            #                     log.warning(f"{e}")
            #             try:
            #                 limit_sell = invpcl.place_active_order(
            #                     side = 'Sell',
            #                     symbol = symbol,
            #                     order_type = 'Limit',
            #                     qty = csize,
            #                     price = ask_price,
            #                     reduce_only = False, time_in_force = 'GoodTillCancel', close_on_trigger = False, post_only = True
            #                 )
            #             except Exception as e:
            #                 log.warning(f"{e}")
            #     except Exception as e:
            #         log.warning(f"{e}")

            # if inverse_mode:
            #     try:
            #         print("Debug: ", inv_perp_equity)
            #         print("Debug min price", min_price)
            #     except Exception as e:
            #         log.warning(f"{e}")

            # PERSISTENT HEDGE: Full mode
            if persistent_mode:
                try:
                    if find_trend() == "long":
                        if (
                            short_trade_condition()
                            and find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                        ):
                            try:
                                exchange.create_limit_sell_order(
                                    symbol, trade_qty, current_ask
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    elif find_trend() == "short":
                        if (
                            long_trade_condition()
                            and find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and long_pos_qty < max_trade_qty
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, trade_qty, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    if (
                        get_orderbook()[1] < get_1m_data()[0]
                        or get_orderbook()[1] < get_5m_data()[0]
                    ):
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            if (
                get_orderbook()[1] < get_1m_data()[0]
                or get_orderbook()[1] < get_5m_data()[0]
            ):
                try:
                    cancel_entry()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")


# Mode functions
def long_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Long mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)
    # print(tylerapi.get_asset_total_volume_5m(symbol, tylerapi.api_data))


def short_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Short mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)


def hedge_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Hedge mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)


def persistent_mode_func(symbol):
    print(
        Fore.LIGHTCYAN_EX + "Persistent hedge mode enabled for",
        symbol + Style.RESET_ALL,
    )
    leverage_verification(symbol)
    trade_func(symbol)


def longbias_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Longbias mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)


def inverse_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Inverse mode enabled for", symbol + Style.RESET_ALL)
    # leverage_verification(symbol)
    trade_func(symbol)


# TO DO:

# Add a terminal like console / hotkeys for entries

# Argument declaration
if args.mode == "long":
    if args.symbol:
        long_mode_func(args.symbol)
    else:
        symbol = input("Instrument undefined. \nInput instrument:")
elif args.mode == "short":
    if args.symbol:
        short_mode_func(args.symbol)
    else:
        symbol = input("Instrument undefined. \nInput instrument:")
elif args.mode == "hedge":
    if args.symbol:
        hedge_mode_func(args.symbol)
    else:
        symbol = input("Instrument undefined. \nInput instrument:")
elif args.mode == "persistent":
    if args.symbol:
        persistent_mode_func(args.symbol)
    else:
        symbol = input("Instrument undefined. \nInput instrument:")
elif args.mode == "longbias":
    if args.symbol:
        longbias_mode_func(args.symbol)
    else:
        symbol = input("Instrument undefined. \nInput instrument:")
elif args.mode == "inverse":
    if args.symbol:
        inverse_mode_func(args.symbol)
    else:
        symbol = input("Inverse mode only works with BTCUSD. \nInput BTCUSD here:")

if args.tg == "on":
    if args.tg:
        print(Fore.LIGHTCYAN_EX + "TG Enabled" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTCYAN_EX + "TG Disabled" + Style.RESET_ALL)
