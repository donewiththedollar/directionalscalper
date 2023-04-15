import argparse
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd
from colorama import Fore, Style
from rich.live import Live

sys.path.append(".")
from directionalscalper.api.manager import Manager
from directionalscalper.core import tables
from directionalscalper.core.config import load_config
from directionalscalper.core.functions import print_lot_sizes
from directionalscalper.core.logger import Logger
from directionalscalper.messengers.manager import MessageManager
from directionalscalper.core.functions import DataAnalyzer, CandlestickData, BalanceData, OrderBookData, MarketData

# 1. Create config.json from config.example.json
# 2. Enter exchange_api_key and exchange_api_secret
# 3. Check/fill all other options. For telegram see below

# 1. Get token from botfather after creating new bot, send a message to your new bot
# 2. Go to https://api.telegram.org/bot<bot_token>/getUpdates
# 3. Replacing <bot_token> with your token from the botfather after creating new bot
# 4. Look for chat id and copy the chat id into config.json


# Bools
version = "Directional Scalper v1.2.0"
long_mode = False
short_mode = False
hedge_mode = False
aggressive_mode = False
deleveraging_mode = False
violent_mode = False
blackjack_mode = False
leverage_verified = False


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

parser = argparse.ArgumentParser(description="Scalper supports 6 modes")

parser.add_argument(
    "--mode",
    type=str,
    help="Mode to use",
    choices=["long", "short", "hedge", "aggressive", "violent", "blackjack"],
    required=True,
)

parser.add_argument("--symbol", type=str, help="Specify symbol", required=True)

parser.add_argument("--iqty", type=str, help="Initial entry quantity", required=True)

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

args = parser.parse_args()

if args.mode == "long":
    long_mode = True
elif args.mode == "short":
    short_mode = True
elif args.mode == "hedge":
    hedge_mode = True
elif args.mode == "aggressive":
    aggressive_mode = True
elif args.mode == "violent":
    violent_mode = True
elif args.mode == "blackjack":
    blackjack_mode = True

if args.symbol:
    symbol = args.symbol
else:
    symbol = input("Instrument undefined. \nInput instrument:")

# if args.iqty:
#     trade_qty = args.iqty
# else:
#     trade_qty = input("Lot size:")

if args.iqty:
    trade_qty = float(args.iqty)
else:
    trade_qty = float(input("Lot size:"))


config_file = "config.json"
if args.config:
    config_file = args.config

print(f"Loading config: {config_file}")
config_file_path = Path(Path().resolve(), "config", config_file)
config = load_config(path=config_file_path)

log = Logger(filename="ds.log", level=config.logger.level)

manager = Manager(
    api=config.api.mode,
    path=Path("data", config.api.filename),
    url=f"{config.api.url}{config.api.filename}",
)

messengers = MessageManager(config=config.messengers)
messengers.send_message_to_all_messengers(
    message=f"Initialising {version}. Mode: {args.mode} Symbol: {symbol} iqty:{trade_qty}"
)


deleverage = config.bot.deleverage_mode
min_volume = config.bot.min_volume
min_distance = config.bot.min_distance
botname = config.bot.bot_name
wallet_exposure = config.bot.wallet_exposure
violent_multiplier = config.bot.violent_multiplier
risk_factor = config.bot.blackjack_risk_factor
scalein_mode = config.bot.scalein_mode
scalein_mode_dca = config.bot.scalein_mode_dca

profit_percentages = [0.3, 0.5, 0.2]
profit_increment_percentage = config.bot.profit_multiplier_pct

if scalein_mode:
    messengers.send_message_to_all_messengers(
        message=f"[ScaleIn Mode] enabled"
    )

# CCXT connect to bybit
exchange = ccxt.bybit(
    {
        "enableRateLimit": True,
        "apiKey": config.exchange.api_key,
        "secret": config.exchange.api_secret,
    }
)

analyzer = DataAnalyzer(manager, min_volume, min_distance)
candlestick_data = CandlestickData(exchange, symbol)
balance_data = BalanceData(exchange)
market_data = MarketData(exchange, symbol)
orderbook_data_instance = OrderBookData(exchange, symbol)


# Call the get_balance method
balance = balance_data.get_balance()

# Access the balance data
dex_balance = balance["dex_balance"]
dex_pnl = balance["dex_pnl"]
dex_upnl = balance["dex_upnl"]
dex_wallet = balance["dex_wallet"]
dex_equity = balance["dex_equity"]

current_bid, current_ask = orderbook_data_instance.get_orderbook()

#exchange.get_symbol_info()

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

def cancel_close(side):
    try:
        orders = exchange.fetch_open_orders(symbol)
        for order in orders:
            if "info" in order:
                order_id = order["info"]["order_id"]
                order_status = order["info"]["order_status"]
                order_side = order["info"]["side"]
                reduce_only = order["info"]["reduce_only"]

                if (
                    order_status != "Filled"
                    and order_status != "Cancelled"
                    and reduce_only
                    and order_side == side
                ):
                    exchange.cancel_order(symbol=symbol, id=order_id)
    except Exception as e:
        log.warning(f"{e}")


def set_auto_stop_loss(position, entry_size):
    stop_loss_buffer = 0.01  # Adjust this value based on your desired risk level

    try:
        if position == "short":
            # Calculate stop loss price for short position
            stop_loss_price = short_pos_price * (1 + stop_loss_buffer)

            # Place stop loss order with reduce-only flag
            exchange.create_order(
                symbol,
                "stop_loss_limit",
                "sell",
                entry_size,
                stop_loss_price,
                params={"reduce_only": True, "stop_px": stop_loss_price},
            )

        elif position == "long":
            # Calculate stop loss price for long position
            stop_loss_price = long_pos_price * (1 - stop_loss_buffer)

            # Place stop loss order with reduce-only flag
            exchange.create_order(
                symbol,
                "stop_loss_limit",
                "buy",
                entry_size,
                stop_loss_price,
                params={"reduce_only": True, "stop_px": stop_loss_price},
            )

    except Exception as e:
        log.warning(f"{e}")


# def short_trade_condition():
#     short_trade_condition = get_orderbook()[0] > candlestick_data.get_m_data(timeframe="1m")[0]
#     return short_trade_condition

def short_trade_condition():
    ma_3_high, _, _, _ = candlestick_data.get_m_data(timeframe="1m")
    short_condition = current_ask > ma_3_high
    return short_condition

def long_trade_condition():
    current_bid = orderbook_data_instance.get_bid()
    ma_3_low, _, _, _ = candlestick_data.get_m_data(timeframe="1m")
    return current_bid < ma_3_low

def add_short_trade_condition():
    _, _, _, ma_6_low = candlestick_data.get_m_data(timeframe="1m")
    return short_pos_price < ma_6_low

def add_long_trade_condition():
    _, _, _, ma_6_low = candlestick_data.get_m_data(timeframe="1m")
    return long_pos_price > ma_6_low


def leverage_verification(symbol):
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
        _, leverage, _ = market_data.get_market_data()
        exchange.set_leverage(leverage=leverage, symbol=symbol)
        print(Fore.YELLOW + "Leverage set" + Style.RESET_ALL)
    except Exception as e:
        print(
            Fore.YELLOW + "Leverage not modified, current leverage is",
            leverage,
        )
        log.debug(f"{e}")


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

balance_data.get_balance()

# max_trade_qty = round(
#     (float(dex_equity) / float(get_orderbook()[1]))
#     / (100 / float(get_market_data()[1])),
#     int(float(get_market_data()[2])),
# )

# Implement basic wallet exposure

# max_trade_qty = round(
#     (float(dex_equity) * wallet_exposure / float(get_orderbook()[1]))
#     / (100 / float(get_market_data()[1])),
#     int(float(get_market_data()[2])),
# )

ask_price = orderbook_data_instance.get_ask()
_, liquidation_price, min_order_qty = market_data.get_market_data()

max_trade_qty = round(
    (float(dex_equity) * wallet_exposure / float(ask_price))
    / (100 / float(liquidation_price)),
    int(float(min_order_qty)),
)

# Initialize variables
initial_long_max_trade_qty = max_trade_qty
initial_short_max_trade_qty = max_trade_qty
initial_max_trade_qty = max_trade_qty
time_interval = 30 * 60  # 30 minutes
last_size_increase_time = time.time()


violent_max_trade_qty = max_trade_qty * violent_multiplier

_, current_leverage, _ = market_data.get_market_data()

# _, leverage, min_trade_qty = market_data.get_market_data()
# print_lot_sizes(max_trade_qty, leverage, min_trade_qty)

_, leverage, min_trade_qty = market_data.get_market_data()
print_lot_sizes(max_trade_qty, leverage, min_trade_qty)

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

api_data = manager.get_data()
vol_condition_true = analyzer.get_min_vol_dist_data(symbol)
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

min_order_qty = manager.get_asset_value(
    symbol=symbol, data=api_data, value="Min qty")


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


try:
    pos_dict = exchange.fetch_positions([symbol])
    get_short_positions(pos_dict)
    get_long_positions(pos_dict)
except Exception as e:
    log.warning(f"{e}")


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


def initial_long_entry_linear_btc(current_bid):
    if (
        # long_mode
        long_trade_condition()
        and find_1m_1x_volume() > min_volume
        and find_5m_spread() > min_distance
        and long_pos_qty == 0
        and find_trend() == "long"
    ):
        try:
            exchange.create_limit_buy_order(symbol, trade_qty, current_bid)
            time.sleep(0.01)
        except Exception as e:
            log.warning(f"{e}")


# Short entry logic if short enabled
def initial_short_entry_linear_btc(current_ask):
    if (
        # short_mode
        short_trade_condition()
        and find_1m_1x_volume() > min_volume
        and find_5m_spread() > min_distance
        and short_pos_qty == 0
        and find_trend() == "short"
    ):
        try:
            exchange.create_limit_sell_order(symbol, trade_qty, current_ask)
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


def check_take_profit_executed(order_side):
    global long_position_closed, short_position_closed
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        for order in open_orders:
            if order["side"] == order_side and order["info"]["reduce_only"]:
                return False
        if order_side == "Sell":
            long_position_closed = True
        elif order_side == "Buy":
            short_position_closed = True
        return True
    except Exception as e:
        log.warning(f"{e}")
        return False


def get_current_price(exchange, symbol):
    ticker = exchange.fetch_ticker(symbol)
    current_price = (ticker["bid"] + ticker["ask"]) / 2
    return current_price


def generate_main_table():
    try:
        min_vol_dist_data = analyzer.get_min_vol_dist_data(symbol)
        mode = find_mode()
        trend = find_trend()
        _, leverage, min_trade_qty = market_data.get_market_data()
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


def trade_func(symbol, last_size_increase_time, max_trade_qty, trade_qty, initial_trade_qty, initial_long_max_trade_qty, initial_short_max_trade_qty, initial_long_trade_qty, initial_short_trade_qty):  # noqa
    position_closed = False
    long_position_closed = False
    short_position_closed = False
    long_max_trade_qty = initial_long_max_trade_qty
    short_max_trade_qty = initial_short_max_trade_qty
    with Live(generate_main_table(), refresh_per_second=2) as live:
        while True:
            try:
                manager.get_data()
                time.sleep(0.01)
                candlestick_data.get_m_data(timeframe="1m")
                time.sleep(0.01)
                candlestick_data.get_m_data(timeframe="5m")
                time.sleep(0.01)
                balance_data.get_balance()
                time.sleep(0.01)
                bid, ask = orderbook_data_instance.get_orderbook()
                #get_orderbook()
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
                analyzer.get_min_vol_dist_data(symbol)
                manager.get_asset_value(
                    symbol=symbol, data=manager.get_data(), value="1mVol"
                )
                time.sleep(30)
            except Exception as e:
                log.warning(f"{e}")

            live.update(generate_main_table())

            try:
                # current_bid = get_orderbook()[0]
                # current_ask = get_orderbook()[1]
                current_bid, current_ask = orderbook_data_instance.get_orderbook()
            except Exception as e:
                log.warning(f"{e}")

            long_open_pos_qty = long_pos_qty
            short_open_pos_qty = short_pos_qty

            reduce_only = {"reduce_only": True}

            five_min_data = candlestick_data.get_m_data(timeframe="5m")
            _, leverage, min_trade_qty = market_data.get_market_data()

            if five_min_data is not None and market_data is not None:
                _, _, ma_6_high, ma_6_low = candlestick_data.get_m_data(timeframe="5m")
                precision, _, _ = market_data.get_market_data()

                short_profit_price = round(
                    short_pos_price - (ma_6_high - ma_6_low),
                    int(precision),
                )

            if five_min_data is not None and market_data is not None:
                _, _, ma_6_high, ma_6_low = candlestick_data.get_m_data(timeframe="5m")
                precision, _, _ = market_data.get_market_data()

                long_profit_price = round(
                    long_pos_price + (ma_6_high - ma_6_low),
                    int(precision),
                )


            # short_profit_price = round(
            #     short_pos_price - (get_m_data(timeframe="5m")[2] - get_m_data(timeframe="5m")[3]),
            #     int(get_market_data()[0]),
            # )

            # long_profit_price = round(
            #     long_pos_price + (get_m_data(timeframe="5m")[2] - get_m_data(timeframe="5m")[3]),
            #     int(get_market_data()[0]),
            # )

            # Check elapsed time
            elapsed_time = time.time() - last_size_increase_time

            if long_position_closed:
                long_max_trade_qty = initial_long_max_trade_qty
                long_trade_qty = initial_long_trade_qty
                long_position_closed = False

            if short_position_closed:
                short_max_trade_qty = initial_short_max_trade_qty
                short_trade_qty = initial_short_trade_qty
                short_position_closed = False

            if scalein_mode and elapsed_time >= time_interval:
                # Long scale-in logic
                if long_max_trade_qty < 5 * initial_long_max_trade_qty:
                    long_max_trade_qty += min_order_qty
                    long_max_trade_qty = min(long_max_trade_qty, 5 * initial_long_max_trade_qty)
                    messengers.send_message_to_all_messengers(
                        message=f"[ScaleIn Mode] Long max trade qty: {long_max_trade_qty}"
                    )
                else:
                    long_max_trade_qty = initial_long_max_trade_qty

                # Short scale-in logic
                if short_max_trade_qty < 5 * initial_short_max_trade_qty:
                    short_max_trade_qty += min_order_qty
                    short_max_trade_qty = min(short_max_trade_qty, 5 * initial_short_max_trade_qty)
                    messengers.send_message_to_all_messengers(
                        message=f"[ScaleIn Mode] Short max trade qty: {short_max_trade_qty}"
                    )
                else:
                    short_max_trade_qty = initial_short_max_trade_qty

                last_size_increase_time = time.time()

            if scalein_mode_dca and elapsed_time >= time_interval:
                # Long DCA logic
                long_max_possible_size = 5 * long_max_trade_qty
                messengers.send_message_to_all_messengers(
                    message=f"[ScaleIn Mode] Maximum possible long trade quantity: {long_max_possible_size}"
                )
                if long_trade_qty < long_max_possible_size:
                    long_trade_qty += min_order_qty
                    long_trade_qty = min(long_trade_qty, long_max_possible_size)
                    messengers.send_message_to_all_messengers(
                        message=f"[ScaleIn Mode] Long trade qty: {long_trade_qty}"
                    )
                else:
                    long_trade_qty = initial_long_trade_qty

                # Short DCA logic
                short_max_possible_size = 5 * short_max_trade_qty
                messengers.send_message_to_all_messengers(
                    message=f"[ScaleIn Mode] Maximum possible short trade quantity: {short_max_possible_size}"
                )
                if short_trade_qty < short_max_possible_size:
                    short_trade_qty += min_order_qty
                    short_trade_qty = min(short_trade_qty, short_max_possible_size)
                    messengers.send_message_to_all_messengers(
                        message=f"[ScaleIn Mode] Short trade qty: {short_trade_qty}"
                    )
                else:
                    short_trade_qty = initial_short_trade_qty

                last_size_increase_time = time.time()


            if violent_mode:
                current_ask = orderbook_data_instance.get_ask()
                _, _, _, ma_6_low = candlestick_data.get_m_data(timeframe="1m")
                denominator = current_ask - ma_6_low
                if denominator == 0:
                    short_violent_trade_qty, long_violent_trade_qty = 0, 0
                else:
                    short_violent_trade_qty = (
                        short_open_pos_qty
                        * (ma_6_low - short_pos_price)
                        / denominator
                    )

                    long_violent_trade_qty = (
                        long_open_pos_qty
                        * (ma_6_low - long_pos_price)
                        / denominator
                    )


            if blackjack_mode:
                current_ask = orderbook_data.get_ask()
                _, ma_3_low, _, _ = candlestick_data.get_m_data(timeframe="1m")
                denominator = current_ask - ma_3_low
                # risk_factor = 0.01  # Adjust this value according to your risk tolerance

                if denominator == 0:
                    short_blackjack_trade_qty, long_blackjack_trade_qty = 0, 0
                else:
                    short_blackjack_trade_qty = (
                        short_open_pos_qty
                        * (ma_3_low - short_pos_price)
                        * risk_factor
                        / denominator
                    )

                    long_blackjack_trade_qty = (
                        long_open_pos_qty
                        * (ma_3_low - long_pos_price)
                        * risk_factor
                        / denominator
                    )

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
                (deleveraging_mode)
                and long_pos_qty > 0
                and (hedge_mode or long_mode or aggressive_mode or violent_mode)
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
                        next_target = profit_targets[-1] * (
                            1 + profit_increment_percentage
                        )
                        profit_targets.append(next_target)

                    remaining_position = long_open_pos_qty

                    for idx, profit_percentage in enumerate(profit_percentages):
                        if idx == len(profit_percentages) - 1:
                            partial_qty = remaining_position
                        else:
                            partial_qty = long_open_pos_qty * profit_percentage
                            remaining_position -= partial_qty

                        target_price = profit_targets[idx]
                        min_trade_qty = market_data.get_min_trade_qty()
                        if partial_qty < float(min_trade_qty):
                            partial_qty = float(min_trade_qty)


                        try:
                            exchange.create_limit_sell_order(
                                symbol, partial_qty, target_price, reduce_only
                            )
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")

            # Long: Normal take profit logic
            if (
                (not deleveraging_mode)
                and long_pos_qty > 0
                and (
                    hedge_mode
                    or long_mode
                    or aggressive_mode
                    or violent_mode
                    or blackjack_mode
                )
            ):
                try:
                    get_open_orders()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")

                if long_profit_price != 0 or long_pos_price != 0:
                    try:
                        cancel_close("Sell")
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")
                    try:
                        exchange.create_limit_sell_order(
                            symbol, long_open_pos_qty, long_profit_price, reduce_only
                        )
                        time.sleep(0.05)
                        check_take_profit_executed("Sell")  # Update long_position_closed using the function
                    except Exception as e:
                        log.warning(f"{e}")

            # Short incremental TP
            if (
                (deleveraging_mode)
                and short_pos_qty > 0
                and (hedge_mode or short_mode or aggressive_mode or violent_mode)
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
                        next_target = profit_targets[-1] * (
                            1 - profit_increment_percentage
                        )
                        profit_targets.append(next_target)

                    remaining_position = short_open_pos_qty

                    for idx, profit_percentage in enumerate(profit_percentages):
                        if idx == len(profit_percentages) - 1:
                            partial_qty = remaining_position
                        else:
                            partial_qty = short_open_pos_qty * profit_percentage
                            remaining_position -= partial_qty

                        target_price = profit_targets[idx]
                        min_trade_qty = market_data.get_min_trade_qty()
                        if partial_qty < float(min_trade_qty):
                            partial_qty = float(min_trade_qty)

                        try:
                            exchange.create_limit_buy_order(
                                symbol, partial_qty, target_price, reduce_only
                            )
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")

            # SHORT: Take profit logic
            if (
                (not deleveraging_mode)
                and short_pos_qty > 0
                and (
                    hedge_mode
                    or short_mode
                    or aggressive_mode
                    or violent_mode
                    or blackjack_mode
                )
            ):
                try:
                    get_open_orders()
                    time.sleep(0.05)
                except Exception as e:
                    log.warning(f"{e}")

                if short_profit_price != 0 or short_pos_price != 0:
                    try:
                        #cancel_close()
                        cancel_close("Buy")
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")
                    try:
                        exchange.create_limit_buy_order(
                            symbol, short_open_pos_qty, short_profit_price, reduce_only
                        )
                        #position_closed = check_take_profit_executed("Buy")
                        check_take_profit_executed("Buy")
                        time.sleep(0.05)
                    except Exception as e:
                        log.warning(f"{e}")

            # Violent: Full mode
            if violent_mode:
                try:
                    current_ask = orderbook_data_instance.get_ask()
                    current_bid = orderbook_data_instance.get_bid()

                    if find_trend() == "short":
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and add_short_trade_condition()
                            and current_ask > short_pos_price
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
                            and add_long_trade_condition()
                            and current_bid < long_pos_price
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
                        current_ask < candlestick_data.get_m_data(timeframe="1m")[0]
                        or current_ask < candlestick_data.get_m_data(timeframe="5m")[0]
                    ):
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            # BLACKJACK: Full mode
            if blackjack_mode:
                try:
                    current_ask = orderbook_data_instance.get_ask()
                    current_bid = orderbook_data_instance.get_bid()
                    if find_trend() == "short":
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            # and short_pos_qty < max_trade_qty
                            and add_short_trade_condition()
                            and current_ask > short_pos_price
                        ):
                            try:
                                exchange.create_limit_sell_order(
                                    symbol, short_blackjack_trade_qty, current_ask
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")

                            # Set automatic stop loss in profit
                            set_auto_stop_loss("short", short_blackjack_trade_qty)

                    elif find_trend() == "long":
                        initial_long_entry(current_bid)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            # and long_pos_qty < max_trade_qty
                            and add_long_trade_condition()
                            and current_bid < long_pos_price
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, long_blackjack_trade_qty, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")

                            # Set automatic stop loss in profit
                            set_auto_stop_loss("long", long_blackjack_trade_qty)

                    if (
                        current_ask < candlestick_data.get_m_data(timeframe="1m")[0]
                        or current_ask < candlestick_data.get_m_data(timeframe="5m")[0]
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
                    current_ask = orderbook_data_instance.get_ask()
                    current_bid = orderbook_data_instance.get_bid()
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
                        current_ask < candlestick_data.get_m_data(timeframe="1m")[0]
                        or current_ask < candlestick_data.get_m_data(timeframe="5m")[0]
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
                    current_ask = orderbook_data_instance.get_ask()
                    current_bid = orderbook_data_instance.get_bid()
                    if find_trend() == "short":
                        initial_short_entry(current_ask)
                        if (
                            find_1m_1x_volume() > min_volume
                            and find_5m_spread() > min_distance
                            and short_pos_qty < max_trade_qty
                            and (
                                add_short_trade_condition()
                                or (current_ask > short_pos_price)
                                or float(dex_upnl) < 0.0
                            )
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
                            and (
                                add_long_trade_condition()
                                or (current_bid < long_pos_price)
                                or float(dex_upnl) < 0.0
                            )
                        ):
                            try:
                                exchange.create_limit_buy_order(
                                    symbol, trade_qty, current_bid
                                )
                                time.sleep(0.01)
                            except Exception as e:
                                log.warning(f"{e}")
                    if (
                        current_ask < candlestick_data.get_m_data(timeframe="1m")[0]
                        or current_ask < candlestick_data.get_m_data(timeframe="5m")[0]
                    ):
                        try:
                            cancel_entry()
                            time.sleep(0.05)
                        except Exception as e:
                            log.warning(f"{e}")
                except Exception as e:
                    log.warning(f"{e}")

            orderbook_data = orderbook_data_instance.get_orderbook()
            data_1m = candlestick_data.get_m_data(timeframe="1m")
            data_5m = candlestick_data.get_m_data(timeframe="5m")

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

            # if (
            #     get_orderbook()[1] < get_m_data(timeframe="1m")[0]
            #     or get_orderbook()[1] < get_m_data(timeframe="5m")[0]
            # ):
            #     try:
            #         cancel_entry()
            #         time.sleep(0.05)
            #     except Exception as e:
            #         log.warning(f"{e}")


# Mode functions
def long_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Long mode enabled for", symbol + Style.RESET_ALL)
    leverage_verification(symbol)
    initial_trade_qty = trade_qty
    initial_long_max_trade_qty = max_trade_qty
    initial_short_max_trade_qty = max_trade_qty
    initial_long_trade_qty = trade_qty
    initial_short_trade_qty = trade_qty
    trade_func(symbol, last_size_increase_time, max_trade_qty, trade_qty, initial_trade_qty, initial_long_max_trade_qty, initial_short_max_trade_qty, initial_long_trade_qty, initial_short_trade_qty)


def short_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Short mode enabled for", symbol + Style.RESET_ALL)
    initial_trade_qty = trade_qty
    initial_long_max_trade_qty = max_trade_qty
    initial_short_max_trade_qty = max_trade_qty
    initial_long_trade_qty = trade_qty
    initial_short_trade_qty = trade_qty
    leverage_verification(symbol)
    trade_func(symbol, last_size_increase_time, max_trade_qty, trade_qty, initial_trade_qty, initial_long_max_trade_qty, initial_short_max_trade_qty, initial_long_trade_qty, initial_short_trade_qty)


def hedge_mode_func(symbol):
    print(Fore.LIGHTCYAN_EX + "Hedge mode enabled for", symbol + Style.RESET_ALL)
    initial_trade_qty = trade_qty
    initial_long_max_trade_qty = max_trade_qty
    initial_short_max_trade_qty = max_trade_qty
    initial_long_trade_qty = trade_qty
    initial_short_trade_qty = trade_qty
    leverage_verification(symbol)
    trade_func(symbol, last_size_increase_time, max_trade_qty, trade_qty, initial_trade_qty, initial_long_max_trade_qty, initial_short_max_trade_qty, initial_long_trade_qty, initial_short_trade_qty)


def aggressive_mode_func(symbol):
    print(
        Fore.LIGHTCYAN_EX + "Aggressive hedge mode enabled for",
        symbol + Style.RESET_ALL,
    )
    initial_trade_qty = trade_qty
    initial_long_max_trade_qty = max_trade_qty
    initial_short_max_trade_qty = max_trade_qty
    initial_long_trade_qty = trade_qty
    initial_short_trade_qty = trade_qty
    leverage_verification(symbol)
    trade_func(symbol, last_size_increase_time, max_trade_qty, trade_qty, initial_trade_qty, initial_long_max_trade_qty, initial_short_max_trade_qty, initial_long_trade_qty, initial_short_trade_qty)


def violent_mode_func(symbol):
    print(
        Fore.LIGHTCYAN_EX
        + "Violent mode enabled use at your own risk use LOW lot size",
        symbol + Style.RESET_ALL,
    )
    initial_trade_qty = trade_qty
    initial_long_max_trade_qty = max_trade_qty
    initial_short_max_trade_qty = max_trade_qty
    initial_long_trade_qty = trade_qty
    initial_short_trade_qty = trade_qty
    leverage_verification(symbol)
    trade_func(symbol, last_size_increase_time, max_trade_qty, trade_qty, initial_trade_qty, initial_long_max_trade_qty, initial_short_max_trade_qty, initial_long_trade_qty, initial_short_trade_qty)


def blackjack_mode_func(symbol):
    print(
        Fore.LIGHTCYAN_EX + "Blackjack mode enabled. Have fun!",
        symbol + Style.RESET_ALL,
    )
    initial_trade_qty = trade_qty
    initial_long_max_trade_qty = max_trade_qty
    initial_short_max_trade_qty = max_trade_qty
    initial_long_trade_qty = trade_qty
    initial_short_trade_qty = trade_qty
    leverage_verification(symbol)
    trade_func(symbol, last_size_increase_time, max_trade_qty, trade_qty, initial_trade_qty, initial_long_max_trade_qty, initial_short_max_trade_qty, initial_long_trade_qty, initial_short_trade_qty)


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
elif args.mode == "blackjack":
    if args.symbol:
        blackjack_mode_func(args.symbol)
    else:
        symbol = input("Instrument undefined. \nInput instrument:")

if args.tg == "on":
    if args.tg:
        print(Fore.LIGHTCYAN_EX + "TG Enabled" + Style.RESET_ALL)
    else:
        print(Fore.LIGHTCYAN_EX + "TG Disabled" + Style.RESET_ALL)
