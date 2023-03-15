import argparse
import logging
import os
import time
from pathlib import Path

import ccxt
import pandas as pd
import telebot
from colorama import Fore, Style
from rich.live import Live
from rich.table import Table

import tylerapi
from config import load_config

from util import tables

# 1. Create config.json from config.json.example
# 2. Enter exchange_api_key and exchange_api_secret
# 3. Check/fill all other options. For telegram see below

# 1. Get token from botfather after creating new bot, send a message to your new bot
# 2. Go to https://api.telegram.org/bot<bot_token>/getUpdates
# 3. Replacing <bot_token> with your token from the botfather after creating new bot
# 4. Look for chat id and copy the chat id into config.json

log = logging.getLogger(__name__)

def sendmessage(message):
    bot.send_message(config.telegram_chat_id, message)


# Bools
version = "Directional Scalper v1.0.7"
long_mode = False
short_mode = False
hedge_mode = False
persistent_mode = False
longbias_mode = False
violent_mode = False
high_vol_stack_mode = False
leverage_verified = False
tg_notifications = False

print(Fore.LIGHTCYAN_EX + "", version, "connecting to exchange" + Style.RESET_ALL)

dex_balance, dex_pnl, dex_upnl, dex_wallet, dex_equity = 0, 0, 0, 0, 0
(
    long_pos_qty,
    short_pos_qty,
    long_pos_price,
    long_liq_price,
    short_pos_price,
    short_liq_price,
) = (0, 0, 0, 0, 0, 0)

parser = argparse.ArgumentParser(description="Scalper supports 5 modes")

parser.add_argument(
    "--mode",
    type=str,
    help="Mode to use",
    choices=["long", "short", "hedge", "persistent", "longbias"],
    required=True,
)

parser.add_argument("--symbol", type=str, help="Specify symbol", required=True)

parser.add_argument("--iqty", type=str, help="Initial entry quantity", required=True)

parser.add_argument(
    "--tg", type=str, help="TG Notifications", choices=["on", "off"], required=True
)

parser.add_argument(
    "--config", type=str, help="Config file. Example: my_config.json", required=False
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

if args.symbol:
    symbol = args.symbol
else:
    symbol = input("Instrument undefined. \nInput instrument:")

if args.iqty:
    trade_qty = args.iqty
else:
    trade_qty = input("Lot size:")

if args.tg == "on":
    tg_notifications = True

if tg_notifications:
    bot = telebot.TeleBot(config.telegram_api_token, parse_mode=None)

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message):
        bot.reply_to(message, "Howdy, how are you doing?")

    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        bot.reply_to(message, message.text)


config_file = "config.json"
if args.config:
    config_file = args.config

# Load config
print("Loading config: " + config_file)
config_file = Path(Path().resolve(), config_file)
config = load_config(path=config_file)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO"),
)


min_volume = config.min_volume
min_distance = config.min_distance
botname = config.bot_name

exchange = ccxt.bybit(
    {
        "enableRateLimit": True,
        "apiKey": config.exchange_api_key,
        "secret": config.exchange_api_secret,
    }
)

# Functions


# Get min vol & spread data from API
def get_min_vol_dist_data(symbol) -> bool:
    try:
        tylerapi.grab_api_data()
        spread5m = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())
        volume1m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())

        return volume1m > min_volume and spread5m > min_distance
    except Exception as e:
        log.warning(f"{e}")
        return False


# get_1m_data() [0]3 high, [1]3 low, [2]6 high, [3]6 low, [4]10 vol
def get_1m_data():
    try: 
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
    except Exception as e:
        log.warning(f"{e}")



# get_5m_data() [0]3 high, [1]3 low, [2]6 high, [3]6 low
def get_5m_data():
    try:
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
    except Exception as e:
        log.warning(f"{e}")


def get_balance():
    global dex_balance, dex_pnl, dex_upnl, dex_wallet, dex_equity
    try:
        dex = exchange.fetch_balance()["info"]["result"]
        dex_balance = dex["USDT"]["available_balance"]
        dex_pnl = dex["USDT"]["realised_pnl"]
        dex_upnl = dex["USDT"]["unrealised_pnl"]
        dex_wallet = round(float(dex["USDT"]["wallet_balance"]), 2)
        dex_equity = round(float(dex["USDT"]["equity"]), 2)
    except KeyError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unknown error occured in get_balance(): {e}")
        log.warning(f"{e}")


# get_orderbook() [0]bid, [1]ask
def get_orderbook():
    try:      
        ob = exchange.fetch_order_book(symbol)
        bid = ob["bids"][0][0]
        ask = ob["asks"][0][0]
        return bid, ask
    except:
        pass


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
    try:
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
    except Exception as e:
        log.warning(f"{e}")

def get_long_positions():
    try:
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
    except Exception as e:
        log.warning(f"{e}")


# get_open_orders() [0]order_id, [1]order_price, [2]order_qty
def get_open_orders():
    try:
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
            return 0, 0, 0
        return order_id, order_price, order_qty
    except Exception as e:
        log.warning(f"{e}")


def cancel_entry():
    try:
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
    except Exception as e:
        log.warning(f"{e}")

def cancel_close():
    try:
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
    except Exception as e:
        log.warning(f"{e}")


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


def leverage_verification(symbol):
    try:
        exchange.set_position_mode(hedged="BothSide", symbol=symbol)
        print(
            Fore.LIGHTYELLOW_EX + "Position mode changed to BothSide" + Style.RESET_ALL
        )
    except Exception as e:
        print(Fore.YELLOW + "Position mode unchanged" + Style.RESET_ALL)
        #log.warning(f"{e}")
    # Set margin mode
    try:
        exchange.set_margin_mode(marginMode="cross", symbol=symbol)
        print(Fore.LIGHTYELLOW_EX + "Margin mode set to cross" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.YELLOW + "Margin mode unchanged" + Style.RESET_ALL)
        #log.warning(f"{e}")
    # Set leverage
    try:
        exchange.set_leverage(leverage=get_market_data()[1], symbol=symbol)
        print(Fore.YELLOW + "Leverage set" + Style.RESET_ALL)
    except Exception as e:
        print(
            Fore.YELLOW + "Leverage not modified, current leverage is",
            get_market_data()[1],
        )
        #log.warning(f"{e}")


if not leverage_verified:
    try:
        leverage_verification(symbol)
    except KeyError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unknown error occured in leverage verification: {e}")
        log.warning(f"{e}")
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
print(f"0.005x : {round(max_trade_qty / 200, int(float(get_market_data()[2])))}")
print(f"0.001x : {round(max_trade_qty / 500, int(float(get_market_data()[2])))}")

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

vol_condition_true = get_min_vol_dist_data(symbol)
tyler_total_volume_1m = tylerapi.get_asset_total_volume_1m(
    symbol, tylerapi.grab_api_data()
)
tyler_total_volume_5m = tylerapi.get_asset_total_volume_5m(
    symbol, tylerapi.grab_api_data()
)
# tyler_1x_volume_1m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
tyler_1x_volume_5m = tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
# tyler_5m_spread = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())
tyler_1m_spread = tylerapi.get_asset_1m_spread(symbol, tylerapi.grab_api_data())


# tyler_trend = tylerapi.get_asset_trend(symbol, tylerapi.grab_api_data())


def find_trend():
    tylerapi.grab_api_data()
    tyler_trend = tylerapi.get_asset_trend(symbol, tylerapi.grab_api_data())

    return tyler_trend


def find_1m_spread():
    tylerapi.grab_api_data()
    tyler_1m_spread = tylerapi.get_asset_1m_spread(symbol, tylerapi.grab_api_data())

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



get_short_positions()
get_long_positions()


# Long entry logic if long enabled
def initial_long_entry(current_bid):
    if (
        # long_mode
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
    else:
        pass


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
    else:
        pass

def generate_main_table():
    min_vol_dist_data = get_min_vol_dist_data(symbol)
    mode = find_mode()
    trend = find_trend()
    market_data = get_market_data()
    return tables.generate_main_table(version, short_pos_unpl, long_pos_unpl, short_pos_unpl_pct, long_pos_unpl_pct, symbol, dex_wallet, 
                        dex_equity, short_symbol_cum_realised, long_symbol_realised, short_symbol_realised,
                        trade_qty, long_pos_qty, short_pos_qty, long_pos_price, long_liq_price, short_pos_price, 
                        short_liq_price, max_trade_qty, market_data, trend, min_vol_dist_data,
                        min_volume, min_distance, mode)

def trade_func(symbol):  # noqa
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

            except Exception as e:
                log.warning(f"{e}")

            try:
                get_min_vol_dist_data(symbol)
                tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
                time.sleep(30)
            except Exception as e:
                log.warning(f"{e}")

            live.update(generate_main_table())
            try:
                current_bid = get_orderbook()[0]
                current_ask = get_orderbook()[1]
            except Exception as e:
                log.warning(f"{e}")
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

            # Longbias mode
            if longbias_mode:
                try:
                    if find_trend() == "long":
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and long_pos_qty < max_trade_qty
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
                and find_trend() == "long"
            ):
                try:
                    exchange.create_limit_buy_order(symbol, trade_qty, current_bid)
                    time.sleep(0.01)
                except Exception as e:
                    log.warning(f"{e}")
            else:
                pass

            # Add to long if long enabled
            if (
                long_pos_qty != 0
                and long_pos_qty < max_trade_qty
                and long_mode
                and find_1m_1x_volume() > min_volume
                and add_long_trade_condition()
                and find_trend() == "long"
                and current_bid < long_pos_price
            ):
                try:
                    cancel_entry()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")
                try:
                    exchange.create_limit_buy_order(symbol, add_trade_qty, current_bid)
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
                and find_trend() == "short"
            ):
                try:
                    exchange.create_limit_sell_order(symbol, trade_qty, current_ask)
                    time.sleep(0.01)
                except Exception as e:
                    log.warning(f"{e}")
            else:
                pass

            # Add to short if short enabled
            if (
                short_pos_qty != 0
                and short_pos_qty < max_trade_qty
                and short_mode
                and find_1m_1x_volume() > min_volume
                and add_short_trade_condition()
                and find_trend() == "short"
                and current_ask > short_pos_price
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
                            symbol, long_open_pos_qty, long_profit_price, reduce_only
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
                            symbol, short_open_pos_qty, short_profit_price, reduce_only
                        )
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")

            # HEDGE: Full mode
            if hedge_mode:
                try:
                    if find_trend() == "short":
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                            and add_short_trade_condition()
                            and current_ask > short_pos_price
                        ):
                            try:
                                exchange.create_limit_sell_order(
                                    symbol, trade_qty, current_ask
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    elif find_trend() == "long":
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and long_pos_qty < max_trade_qty
                            and add_long_trade_condition()
                            and current_bid < long_pos_price
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

if args.tg == "on":
    if args.tg:
        print(Fore.LIGHTCYAN_EX + "TG Enabled" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTCYAN_EX + "TG Disabled" + Style.RESET_ALL)
