import argparse
import decimal
import math
import logging
import logging.handlers as handlers
import time
from pathlib import Path

import ccxt
from pybit import inverse_perpetual
import pandas as pd
import telebot
from colorama import Fore, Style
from rich.live import Live
from rich.table import Table

from api.manager import Manager
from config import load_config
from util import tables
from util.functions import print_lot_sizes

# 1. Create config.json from config.json.example
# 2. Enter exchange_api_key and exchange_api_secret
# 3. Check/fill all other options. For telegram see below

# 1. Get token from botfather after creating new bot, send a message to your new bot
# 2. Go to https://api.telegram.org/bot<bot_token>/getUpdates
# 3. Replacing <bot_token> with your token from the botfather after creating new bot
# 4. Look for chat id and copy the chat id into config.json


log = logging.getLogger()
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - %(message)s"
)
logHandler = handlers.RotatingFileHandler("ds.log", maxBytes=5000000, backupCount=5)
logHandler.setFormatter(formatter)
log.setLevel(logging.INFO)
log.addHandler(logHandler)

manager = Manager()

def sendmessage(message):
    bot.send_message(config.telegram_chat_id, message)

# Booleans
version = "Directional Scalper v1.2.0"
long_mode = False
short_mode = False
hedge_mode = False
aggressive_mode = False
btclinear_long_mode = False
btclinear_short_mode = False
deleveraging_mode = False
violent_mode = False
high_vol_stack_mode = False
inverse_mode = False
leverage_verified = False
tg_notifications = False
inverse_trading_status = 0

api_data = manager.get_data()

print(Fore.LIGHTCYAN_EX + "", version, "connecting to exchange" + Style.RESET_ALL)

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

max_trade_qty = 0
dex_btc_upnl_pct = 0.0

parser = argparse.ArgumentParser(description="Scalper supports 6 modes")

parser.add_argument(
    "--mode",
    type=str,
    help="Mode to use",
    choices=[
        "long",
        "short",
        "hedge",
        "aggressive",
        "violent",
        "inverse",
    ],
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

parser.add_argument(
    "--deleverage",
    type=str,
    help="Deleveraging enabled",
    choices=["on", "off"],
    required=False,
)

parser.add_argument(
    "--avoidfees",
    type=str,
    help="Avoid all fees",
    choices=["on", "off"],
    required=False,
)

args = parser.parse_args()

if args.mode == "long":
    long_mode = True
elif args.mode == "short":
    short_mode = True
elif args.mode == "hedge":
    hedge_mode = True
elif args.mode == "aggressive":
    aggressive_mode = True
elif args.mode == "inverse":
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

if args.tg == "on":
    tg_notifications = True
else:
    tg_notifications = False


config_file = "config.json"
if args.config:
    config_file = args.config
    
# Load config
print(f"Loading config: {config_file}")
config_file_path = Path(Path().resolve(), config_file)
config = load_config(path=config_file_path)

if tg_notifications:
    bot = telebot.TeleBot(config.telegram_api_token, parse_mode=None)

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message):
        bot.reply_to(message, "Howdy, how are you doing?")

    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        bot.reply_to(message, message.text)

min_volume = config.min_volume
min_distance = config.min_distance
botname = config.bot_name
linear_taker_fee = config.linear_taker_fee
wallet_exposure = config.wallet_exposure
violent_multiplier = config.violent_multiplier

profit_percentages = [0.3, 0.5, 0.2]
profit_increment_percentage = config.profit_multiplier_pct

inverse_direction = config.inverse_direction

exchange = ccxt.bybit(
    {
        "enableRateLimit": True,
        "apiKey": config.exchange_api_key,
        "secret": config.exchange_api_secret,
    }
)

endpoint = "https://api.bybit.com"
unauth = inverse_perpetual.HTTP(endpoint=endpoint)
invpcl = inverse_perpetual.HTTP(
    endpoint=endpoint,
    api_key=config.exchange_api_key,
    api_secret=config.exchange_api_secret,
)

# Get min vol & spread data from API
def get_min_vol_dist_data(symbol) -> bool:
    try:
        api_data = manager.get_data()
        spread5m = manager.get_asset_value(
            symbol=symbol, data=api_data, value="5mSpread"
        )
        volume1m = manager.get_asset_value(symbol=symbol, data=api_data, value="1mVol")

        return volume1m > min_volume and spread5m > min_distance
    except Exception as e:
        log.warning(f"{e}")
        return False

def find_decimals(value):
    return abs(decimal.Decimal(str(value)).as_tuple().exponent)

# Get inverse symbol info
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

# Inverse short position data
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

def get_inverse_buy_position():
    try:
        position = invpcl.my_position(symbol=symbol)

        if position["result"]["side"] == "None":
            global buy_position_size, buy_position_prce
            buy_position_size = 0
            buy_position_prce = 0
        if position["result"]["side"] == "Buy":
            buy_position_size = float(position["result"]["size"])
            buy_position_prce = float(position["result"]["entry_price"])
    except Exception as e:
        log.warning(f"{e}")


def get_m_data(timeframe: str = "1m", num_bars: int = 20):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=num_bars)
        df = pd.DataFrame(
            bars, columns=["Time", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Time"] = pd.to_datetime(df["Time"], unit="ms")
        df["MA_3_High"] = df.High.rolling(3).mean()
        df["MA_3_Low"] = df.Low.rolling(3).mean()
        df["MA_6_High"] = df.High.rolling(6).mean()
        df["MA_6_Low"] = df.Low.rolling(6).mean()
        get_data_3_high = df["MA_3_High"].iat[-1]
        get_data_3_low = df["MA_3_Low"].iat[-1]
        get_data_6_high = df["MA_6_High"].iat[-1]
        get_data_6_low = df["MA_6_Low"].iat[-1]
        return (
            get_data_3_high,
            get_data_3_low,
            get_data_6_high,
            get_data_6_low,
        )
    except Exception as e:
        log.warning(f"{e}")

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
    global dex_balance, dex_pnl, dex_upnl, dex_wallet, dex_equity
    try:
        dex = exchange.fetch_balance()["info"]["result"]
        dex_balance = dex["USDT"]["available_balance"]
        dex_pnl = dex["USDT"]["realised_pnl"]
        dex_upnl = dex["USDT"]["unrealised_pnl"]
        #print(f"dex_upnl: {dex_upnl}, type: {type(dex_upnl)}")  # Add this line to check the type and value of dex_upnl
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
    except Exception as e:
        log.warning(f"{e}")


# get_market_data() [0]precision, [1]leverage, [2]min_trade_qty
def get_market_data():
    try:
        global leverage
        exchange.load_markets()
        precision = exchange.market(symbol)["info"]["price_scale"]
        leverage = exchange.market(symbol)["info"]["leverage_filter"]["max_leverage"]
        min_trade_qty = exchange.market(symbol)["info"]["lot_size_filter"][
            "min_trading_qty"
        ]
        return precision, leverage, min_trade_qty
    except Exception as e:
        log.warning(f"{e}")


def get_short_positions(pos_dict):
    try:
        global short_pos_qty, short_pos_price, short_symbol_realised, short_symbol_cum_realised, short_pos_unpl, short_pos_unpl_pct, short_liq_price, short_pos_price_at_entry
    
        pos_dict = pos_dict[1]
        short_pos_qty = float(pos_dict["contracts"])
        short_symbol_realised = round(float(pos_dict["info"]["realised_pnl"] or 0), 4)
        short_symbol_cum_realised = round(
            float(pos_dict["info"]["cum_realised_pnl"] or 0), 4
        )
        short_pos_unpl = round(float(pos_dict["info"]["unrealised_pnl"] or 0), 4)
        short_pos_unpl_pct = round(float(pos_dict["percentage"] or 0), 4)
        short_pos_price = pos_dict["entryPrice"] or 0
        short_liq_price = pos_dict["liquidationPrice"] or 0
        short_pos_price_at_entry = short_pos_price
    except Exception as e:
        log.warning(f"{e}")


def get_long_positions(pos_dict):
    try:
        global long_pos_qty, long_pos_price, long_symbol_realised, long_symbol_cum_realised, long_pos_unpl, long_pos_unpl_pct, long_liq_price, long_pos_price_at_entry
        pos_dict = pos_dict[0]
        long_pos_qty = float(pos_dict["contracts"])
        long_symbol_realised = round(float(pos_dict["info"]["realised_pnl"]), 4)
        long_symbol_cum_realised = round(float(pos_dict["info"]["cum_realised_pnl"]), 4)
        long_pos_unpl = float(pos_dict["info"]["unrealised_pnl"] or 0)
        long_pos_unpl_pct = round(float(pos_dict["percentage"] or 0), 4)
        long_pos_price = pos_dict["entryPrice"] or 0
        long_liq_price = pos_dict["liquidationPrice"] or 0
        long_pos_price_at_entry = long_pos_price
    except Exception as e:
        log.warning(f"{e}")


# get_open_orders() [0]order_id, [1]order_price, [2]order_qty
def get_open_orders():
    try:
        order = exchange.fetch_open_orders(symbol)
        if len(order) > 0:
            if "info" in order[0]:
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
                    return order_id, order_price, order_qty
    except Exception as e:
        log.warning(f"{e}")
    return 0, 0, 0

def cancel_entry():
    try:
        order = exchange.fetch_open_orders(symbol)
        if len(order) > 0:
            if "info" in order[0]:
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

# Trade conditions
def short_trade_condition():
    short_trade_condition = get_orderbook()[0] > get_m_data(timeframe="1m")[0]
    return short_trade_condition

def long_trade_condition():
    long_trade_condition = get_orderbook()[0] < get_m_data(timeframe="1m")[0]
    return long_trade_condition

def add_short_trade_condition():
    add_short_trade_condition = short_pos_price < get_m_data(timeframe="1m")[3]
    return add_short_trade_condition

def add_long_trade_condition():
    add_long_trade_condition = long_pos_price > get_m_data(timeframe="1m")[3]
    return add_long_trade_condition

def inverse_short_trade_condition():
    inverse_short_trade_condition = get_orderbook()[0] > get_m_data(timeframe="1m")[0]
    return inverse_short_trade_condition

def add_inverse_short_trade_condition():
    add_inverse_short_trade_condition = sell_position_prce < get_m_data(timeframe="1m")[3]
    return add_inverse_short_trade_condition

def inverse_long_trade_condition():
    inverse_long_trade_condition = get_orderbook()[1] > get_m_data(timeframe="1m")[0]
    return inverse_long_trade_condition

def add_inverse_long_trade_condition():
    add_inverse_long_trade_condition = buy_position_prce < get_m_data(timeframe="5m")[3]
    return add_inverse_long_trade_condition

# Leverage verification
def leverage_verification(symbol):
    if not inverse_mode:
        try:
            exchange.set_position_mode(hedged="BothSide", symbol=symbol)
            print(
                Fore.LIGHTYELLOW_EX + "Position mode changed to BothSide" + Style.RESET_ALL
            )
        except Exception as e:
            print(Fore.YELLOW + "Position mode unchanged" + Style.RESET_ALL)
            log.debug(f"{e}")
        # Set margin mode
        try:
            exchange.set_margin_mode(marginMode="cross", symbol=symbol)
            print(Fore.LIGHTYELLOW_EX + "Margin mode set to cross" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.YELLOW + "Margin mode unchanged" + Style.RESET_ALL)
            log.debug(f"{e}")
        # Set leverage
        try:
            exchange.set_leverage(leverage=get_market_data()[1], symbol=symbol)
            print(Fore.YELLOW + "Leverage set" + Style.RESET_ALL)
        except Exception as e:
            print(
                Fore.YELLOW + "Leverage not modified, current leverage is",
                get_market_data()[1],
            )
            log.debug(f"{e}")


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
        (float(dex_equity) * wallet_exposure / float(get_orderbook()[1]))
        / (100 / float(get_market_data()[1])),
        int(float(get_market_data()[2])),
    )

    violent_max_trade_qty = max_trade_qty * violent_multiplier

    current_leverage = get_market_data()[1]

    print_lot_sizes(max_trade_qty, get_market_data())
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


#api_data = manager.get_data()
vol_condition_true = get_min_vol_dist_data(symbol)
tyler_total_volume_1m = manager.get_asset_value(
    symbol=symbol, data=api_data, value="1mVol"
)
tyler_total_volume_5m = manager.get_asset_value(
    symbol=symbol, data=api_data, value="5mVol"
)

tyler_1x_volume_5m = manager.get_asset_value(
    symbol=symbol, data=api_data, value="5mVol"
)

tyler_1m_spread = manager.get_asset_value(
    symbol=symbol, data=api_data, value="1mSpread"
)


def find_trend():
    try:
        api_data = manager.get_data()
        tyler_trend = manager.get_asset_value(
            symbol=symbol, data=api_data, value="Trend"
        )

        return tyler_trend
    except Exception as e:
        log.warning(f"{e}")


def find_1m_spread():
    try:
        api_data = manager.get_data()
        tyler_1m_spread = manager.get_asset_value(
            symbol=symbol, data=api_data, value="1mSpread"
        )

        return tyler_1m_spread
    except Exception as e:
        log.warning(f"{e}")


def find_5m_spread():
    try:
        api_data = manager.get_data()
        tyler_spread = manager.get_asset_value(
            symbol=symbol, data=api_data, value="5mSpread"
        )

        return tyler_spread
    except Exception as e:
        log.warning(f"{e}")


def find_1m_1x_volume():
    try:
        api_data = manager.get_data()
        tyler_1x_volume_1m = manager.get_asset_value(
            symbol=symbol, data=api_data, value="1mVol"
        )
        return tyler_1x_volume_1m
    except Exception as e:
        log.warning(f"{e}")


def find_mode():
    mode = args.mode

    return mode


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

def inverse_limit_long_with_cancel_order(current_bid):
    open_orders = exchange.fetch_open_orders(symbol)
    if len(open_orders) > 0:
        print("Debug: Cancelling open orders...")
        for order in open_orders:
            exchange.cancel_order(order["id"], symbol)
    try:
        get_orderbook()
        invpcl.place_active_order(
            side="Buy",
            symbol=symbol,
            order_type="Limit",
            qty=trade_qty,
            price=current_bid,
            reduce_only=False,
            time_in_force="GoodTillCancel",
            close_on_trigger=False,
            post_only=True,
        )
        print("Debug: Limit long placed")
        sendmessage("Limit long placed")
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


# Generate table
def generate_main_table() -> Table:
    if inverse_mode:
        return generate_inverse_table()
    else:
        return generate_table()


def generate_inverse_table(mode='inverse'):
    min_vol_dist_data = get_min_vol_dist_data(symbol)
    trend = find_trend()

    inverse_table = Table(show_header=False, box=None, title=version)
    if inverse_mode:
        tp_price = calc_tp_price()
        inverse_table.add_row(
            tables.generate_inverse_table_info(
                symbol,
                dex_btc_balance,
                dex_btc_equity,
                inv_perp_cum_realised_pnl,
                dex_btc_upnl_pct,
                trade_qty,
                sell_position_size,
                trend,
                sell_position_prce,
                tp_price,
                False,
            )
        )
    inverse_table.add_row(
        tables.generate_table_vol(
            min_vol_dist_data, min_volume, min_distance, symbol, True, mode
        )
    )
    return inverse_table

def generate_table():
    min_vol_dist_data = get_min_vol_dist_data(symbol)
    mode = find_mode()
    trend = find_trend()
    market_data = get_market_data()
    table_data = {
        "version": version,
        "short_pos_unpl": short_pos_unpl,
        "long_pos_unpl": long_pos_unpl,
        "short_pos_unpl_pct": short_pos_unpl_pct,
        "long_pos_unpl_pct": long_pos_unpl_pct,
        "symbol": symbol,
        "dex_wallet": dex_wallet,
        "dex_equity": dex_equity,
        "short_symbol_cum_realised": short_symbol_cum_realised,
        "long_symbol_realised": long_symbol_realised,
        "short_symbol_realised": short_symbol_realised,
        "trade_qty": trade_qty,
        "long_pos_qty": long_pos_qty,
        "short_pos_qty": short_pos_qty,
        "long_pos_price": long_pos_price,
        "long_liq_price": long_liq_price,
        "short_pos_price": short_pos_price,
        "short_liq_price": short_liq_price,
        "max_trade_qty": max_trade_qty,
        "market_data": market_data,
        "trend": trend,
        "min_vol_dist_data": min_vol_dist_data,
        "min_volume": min_volume,
        "min_distance": min_distance,
        "mode": mode,
    }
    return tables.generate_main_table(data=table_data)

# Long entry logic if long enabled
def initial_long_entry(current_bid):
    if (
        # long_mode
        long_trade_condition()
        and find_1m_1x_volume() > min_volume
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
            global long_pos_price_at_entry
            long_pos_price_at_entry = long_pos_price


# Short entry logic if short enabled
def initial_short_entry(current_ask):
    if (
        # short_mode
        short_trade_condition()
        and find_1m_1x_volume() > min_volume
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
            global short_pos_price_at_entry
            short_pos_price_at_entry = short_pos_price


def get_current_price(exchange, symbol):
    ticker = exchange.fetch_ticker(symbol)
    current_price = (ticker["bid"] + ticker["ask"]) / 2
    return current_price

def generate_main_table():
    try:
        min_vol_dist_data = get_min_vol_dist_data(symbol)
        mode = find_mode()
        trend = find_trend()
        market_data = get_market_data()
        table_data = {
            "version": version,
            "short_pos_unpl": short_pos_unpl,
            "long_pos_unpl": long_pos_unpl,
            "short_pos_unpl_pct": short_pos_unpl_pct,
            "long_pos_unpl_pct": long_pos_unpl_pct,
            "symbol": symbol,
            "dex_wallet": dex_wallet,
            "dex_equity": dex_equity,
            "short_symbol_cum_realised": short_symbol_cum_realised,
            "long_symbol_cum_realised": long_symbol_cum_realised,
            "long_symbol_realised": long_symbol_realised,
            "short_symbol_realised": short_symbol_realised,
            "trade_qty": trade_qty,
            "long_pos_qty": long_pos_qty,
            "short_pos_qty": short_pos_qty,
            "long_pos_price": long_pos_price,
            "long_liq_price": long_liq_price,
            "short_pos_price": short_pos_price,
            "short_liq_price": short_liq_price,
            "max_trade_qty": max_trade_qty,
            "market_data": market_data,
            "trend": trend,
            "min_vol_dist_data": min_vol_dist_data,
            "min_volume": min_volume,
            "min_distance": min_distance,
            "mode": mode,
        }
        return tables.generate_main_table(manager=manager, data=table_data)
    except Exception as e:
        log.warning(f"{e}")

def inverse_trade_func(symbol):
    with Live(generate_inverse_table(), refresh_per_second=2) as live:
        while True:
            try:
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
                manager.get_data()
                time.sleep(0.01)
                get_m_data(timeframe="1m")
                time.sleep(0.01)
                get_m_data(timeframe="5m")
                time.sleep(0.01)
            except Exception as e:
                log.warning(f"{e}")

            try:
                get_min_vol_dist_data(symbol)
                manager.get_asset_value(
                    symbol=symbol, data=manager.get_data(), value="1mVol"
                )
                time.sleep(30)
            except Exception as e:
                log.warning(f"{e}")

                five_min_data = get_m_data(timeframe="5m")
                one_min_data = get_m_data(timeframe="1m")
                live.update(generate_main_table())
                find_decimals(min_trading_qty)
                decimal_for_tp_size = find_decimals(min_trading_qty)
                get_orderbook()
                current_bid = get_orderbook()[0]
                current_ask = get_orderbook()[1]
                # print("Current bid", current_bid)

                reduce_only = {"reduce_only": True}

                if inverse_direction == "short":
                    get_inverse_sell_position()

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
                elif inverse_direction == "long":
                    get_inverse_buy_position()

                    if buy_position_size / config.divider < min_trading_qty:
                        lot_size_market_tp = buy_position_size
                        print(f"Long Market TP size (1): {lot_size_market_tp}")

                    if (
                        buy_position_size / config.divider
                        < min_trading_qty * config.divider
                    ):
                        lot_size_market_tp = buy_position_size
                        print(f"Long Market TP size (2): {lot_size_market_tp}")

                    else:
                        lot_size_market_tp = round(
                            (buy_position_size / config.divider), decimal_for_tp_size
                        )
                        print(f"Long Market TP size (3): {lot_size_market_tp}")

            # Inverse perps BTCUSD long
            if (inverse_mode and inverse_direction == "long"):  
                try:
                    get_inverse_buy_position()
                except Exception as e:
                    log.warning(f"{e}")
                # limit_sell_order_id = 0
                # First entry
                if buy_position_size == 0 and buy_position_prce == 0:
                    try:
                        inverse_limit_long_with_cancel_order(current_bid)
                    except Exception as e:
                        log.warning(f"{e}")

                # Additional entry
                if (
                    buy_position_size > 0
                    and inverse_long_trade_condition()
                    and find_trend() == "long"
                    and current_bid < buy_position_prce
                ):
                    try:
                        # inverse_limit_short(current_ask)
                        inverse_limit_long_with_cancel_order(current_bid)
                    except Exception as e:
                        log.warning(f"{e}")
                else:
                    print("Not time for additional entry, condition not met yet")

                # Short Take profit for inverse
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
                                sendmessage("Short market take profit placed")
                            except Exception as e:
                                print("Error in placing TP")
                                log.warning(f"{e}")
                        else:
                            print("You have short position but not time for TP")
                            print(f"Current bid {current_bid}")
                    except Exception as e:
                        log.warning(f"{e}")

                try:
                    if (
                        get_orderbook()[1] < get_m_data(timeframe="1m")[0]
                        or get_orderbook()[1] < get_m_data(timeframe="5m")[0]
                    ):
                        try:
                            cancel_entry()
                            print("Canceled entry")
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")
            # Inverse perps BTCUSD short
            if (inverse_mode and inverse_direction == "short"):    
                try:
                    get_inverse_sell_position()
                except Exception as e:
                    log.warning(f"{e}")
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

                # Take profit for inverse short
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
                        get_orderbook()[1] < get_m_data(timeframe="1m")[0]
                        or get_orderbook()[1] < get_m_data(timeframe="5m")[0]
                    ):
                        try:
                            cancel_entry()
                            print("Canceled entry")
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")


def trade_func(symbol):  # noqa
    with Live(generate_main_table(), refresh_per_second=2) as live:
        while True:
            try:
                if not inverse_mode:
                    manager.get_data()
                    time.sleep(0.01)
                    get_m_data(timeframe="1m")
                    time.sleep(0.01)
                    get_m_data(timeframe="5m")
                    time.sleep(0.01)
                    get_balance()
                    time.sleep(0.01)
                    get_orderbook()
                    time.sleep(0.01)
                    long_trade_condition()
                    time.sleep(0.01)
                    short_trade_condition()
                    time.sleep(0.01)
                    pos_dict = exchange.fetch_positions([symbol])
                    get_short_positions(pos_dict)
                    get_long_positions(pos_dict)
                    time.sleep(0.01)
            except Exception as e:
                log.warning(f"{e}")

            try:
                get_min_vol_dist_data(symbol)
                manager.get_asset_value(
                    symbol=symbol, data=manager.get_data(), value="1mVol"
                )
                time.sleep(30)
            except Exception as e:
                log.warning(f"{e}")

            if not inverse_mode:
                live.update(generate_main_table())
                try:
                    current_bid = get_orderbook()[0]
                    current_ask = get_orderbook()[1]
                except Exception as e:
                    log.warning(f"{e}")
                long_open_pos_qty = long_pos_qty
                short_open_pos_qty = short_pos_qty
                reduce_only = {"reduce_only": True}

                five_min_data = get_m_data(timeframe="5m")
                market_data = get_market_data()

                if five_min_data is not None and market_data is not None:
                    short_profit_price = round(
                        short_pos_price - (five_min_data[2] - five_min_data[3]),
                        int(market_data[0]),
                    )

                if five_min_data is not None and market_data is not None:
                    long_profit_price = round(
                        long_pos_price + (five_min_data[2] - five_min_data[3]),
                        int(market_data[0]),
                    )

                if violent_mode:
                    denominator = get_orderbook()[1] - get_m_data(timeframe="1m")[3]
                    if denominator == 0:
                        short_violent_trade_qty, long_violent_trade_qty = 0, 0
                    else:
                        short_violent_trade_qty = (
                            short_open_pos_qty
                            * (get_m_data(timeframe="1m")[3] - short_pos_price)
                            / denominator
                        )

                        long_violent_trade_qty = (
                            long_open_pos_qty
                            * (get_m_data(timeframe="1m")[3] - long_pos_price)
                            / denominator
                        )

                if config.avoid_fees:
                    taker_fee_rate = config.linear_taker_fee
                else:
                    if deleveraging_mode:
                        taker_fee_rate = config.linear_taker_fee

                add_trade_qty = trade_qty

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
                and long_pos_qty < max_trade_qty
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

            if config.avoid_fees:
                taker_fee_rate = config.linear_taker_fee
            else:
                if deleveraging_mode:
                    taker_fee_rate = config.linear_taker_fee

            add_trade_qty = trade_qty

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

            # Long incremental TP
            if (
                (deleveraging_mode or config.avoid_fees)
                and long_pos_qty > 0
                and (
                    hedge_mode
                    or long_mode
                    or aggressive_mode
                    or btclinear_long_mode
                    or violent_mode
                )
            ):
                try:
                    get_open_orders()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")

                if long_pos_price != 0:
                    try:
                        cancel_close()
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")

                    first_profit_target = long_profit_price
                    profit_targets = [first_profit_target]

                    # Calculate subsequent profit targets
                    for _ in range(len(profit_percentages) - 1):
                        next_target = profit_targets[-1] * (1 + profit_increment_percentage)
                        profit_targets.append(next_target)

                    remaining_position = long_open_pos_qty

                    for idx, profit_percentage in enumerate(profit_percentages):
                        if idx == len(profit_percentages) - 1:
                            partial_qty = remaining_position
                        else:
                            partial_qty = long_open_pos_qty * profit_percentage
                            remaining_position -= partial_qty

                        target_price = profit_targets[idx]

                        try:
                            exchange.create_limit_sell_order(
                                symbol, partial_qty, target_price, reduce_only
                            )
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")


            # Long: Normal take profit logic
            if (
                (not deleveraging_mode or not config.avoid_fees)
                and long_pos_qty > 0
                and (
                    hedge_mode
                    or long_mode
                    or aggressive_mode
                    or btclinear_long_mode
                    or violent_mode
                )
            ):
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

            # Short incremental TP
            if (
                (deleveraging_mode or config.avoid_fees)
                and short_pos_qty > 0
                and (
                    hedge_mode
                    or short_mode
                    or aggressive_mode
                    or btclinear_short_mode
                    or violent_mode
                )
            ):
                try:
                    get_open_orders()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")

                if short_pos_price != 0:
                    try:
                        cancel_close()
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")

                    first_profit_target = short_profit_price
                    profit_targets = [first_profit_target]

                    # Calculate subsequent profit targets
                    for _ in range(len(profit_percentages) - 1):
                        next_target = profit_targets[-1] * (1 - profit_increment_percentage)
                        profit_targets.append(next_target)

                    remaining_position = short_open_pos_qty

                    for idx, profit_percentage in enumerate(profit_percentages):
                        if idx == len(profit_percentages) - 1:
                            partial_qty = remaining_position
                        else:
                            partial_qty = short_open_pos_qty * profit_percentage
                            remaining_position -= partial_qty

                        target_price = profit_targets[idx]

                        try:
                            exchange.create_limit_buy_order(
                                symbol, partial_qty, target_price, reduce_only
                            )
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")

            # SHORT: Take profit logic
            if (
                (not deleveraging_mode and not config.avoid_fees)
                and short_pos_qty > 0
                and (
                    hedge_mode
                    or short_mode
                    or aggressive_mode
                    or btclinear_short_mode
                    or violent_mode
                )
            ):
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

            # Violent: Full mode
            if violent_mode:
                try:
                    if find_trend() == "short":
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and (add_short_trade_condition() or (current_ask > short_pos_price) or float(dex_upnl) < 0.0)
                        ):
                            trade_size = (
                                short_violent_trade_qty
                                if short_pos_qty < violent_max_trade_qty
                                else trade_qty
                            )
                            try:
                                exchange.create_limit_sell_order(
                                    symbol, trade_size, current_ask
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")

                    elif find_trend() == "long":
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and (add_long_trade_condition() or (current_bid < long_pos_price) or float(dex_upnl) < 0.0)
                        ):
                            trade_size = (
                                long_violent_trade_qty
                                if long_pos_qty < violent_max_trade_qty
                                else trade_qty
                            )
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, trade_size, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    if (
                        get_orderbook()[1] < get_m_data(timeframe="1m")[0]
                        or get_orderbook()[1] < get_m_data(timeframe="5m")[0]
                    ):
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
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
                            #and current_ask > short_pos_price
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
                            #and current_bid < long_pos_price
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, trade_qty, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    if (
                        get_orderbook()[1] < get_m_data(timeframe="1m")[0]
                        or get_orderbook()[1] < get_m_data(timeframe="5m")[0]
                    ):
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            if aggressive_mode:
                try:
                    if find_trend() == "short":
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                            and (add_short_trade_condition() or (current_ask > short_pos_price) or float(dex_upnl) < 0.0)
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
                            and (add_long_trade_condition() or (current_bid < long_pos_price) or float(dex_upnl) < 0.0)
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, trade_qty, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    if (
                        get_orderbook()[1] < get_m_data(timeframe="1m")[0]
                        or get_orderbook()[1] < get_m_data(timeframe="5m")[0]
                    ):
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            orderbook_data = get_orderbook()
            data_1m = get_m_data(timeframe="1m")
            data_5m = get_m_data(timeframe="5m")

            if (
                orderbook_data is not None
                and data_1m is not None
                and data_5m is not None
            ):
                if orderbook_data[1] < data_1m[0] or orderbook_data[1] < data_5m[0]:
                    try:
                        cancel_entry()
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")
            else:
                log.warning("One or more functions returned None")


# Mode functions
def long_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Long mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)

def short_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Short mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)

def hedge_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Hedge mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    trade_func(symbol)

def aggressive_mode_func(symbol):
    print(
        Fore.LIGHTCYAN_EX + "Aggressive hedge mode enabled for",
        symbol + Style.RESET_ALL,
    )
    leverage_verification(symbol)
    trade_func(symbol)

def violent_mode_func(symbol):
    print(
        Fore.LIGHTCYAN_EX
        + "Violent mode enabled use at your own risk use LOW lot size",
        symbol + Style.RESET_ALL,
    )
    leverage_verification(symbol)
    trade_func(symbol)

def inverse_mode_func(symbol):
    print(
        Fore.LIGHTCYAN_EX + "Inverse mode enabled for",
          symbol + Style.RESET_ALL)
    inverse_trade_func(symbol)


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
elif args.mode == "aggressive":
    if args.symbol:
        aggressive_mode_func(args.symbol)
    else:
        symbol = input("Instrument undefined. \nInput instrument:")
elif args.mode == "violent":
    if args.symbol:
        violent_mode_func(args.symbol)
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