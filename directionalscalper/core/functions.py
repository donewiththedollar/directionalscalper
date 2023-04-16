from colorama import Fore
from directionalscalper.core.logger import Logger
from directionalscalper.messengers.manager import MessageManager
import pandas as pd

log = Logger(filename="ds.log", level="info")

class DataAnalyzer:
    def __init__(self, manager, min_volume, min_distance):
        self.manager = manager
        self.min_volume = min_volume
        self.min_distance = min_distance

    def get_min_vol_dist_data(self, symbol) -> bool:
        try:
            api_data = self.manager.get_data()
            spread5m = self.manager.get_asset_value(
                symbol=symbol, data=api_data, value="5mSpread"
            )
            volume1m = self.manager.get_asset_value(symbol=symbol, data=api_data, value="1mVol")

            return volume1m > self.min_volume and spread5m > self.min_distance
        except Exception as e:
            log.warning(f"{e}")
            return False

class CandlestickData:
    def __init__(self, exchange, symbol):
        self.exchange = exchange
        self.symbol = symbol

    def get_m_data(self, timeframe: str = "1m", num_bars: int = 20):
        try:
            bars = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=num_bars)
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

class BalanceData:
    def __init__(self, exchange):
        self.exchange = exchange

    def get_balance(self):
        try:
            dex = self.exchange.fetch_balance()["info"]["result"]
            dex_balance = dex["USDT"]["available_balance"]
            dex_pnl = dex["USDT"]["realised_pnl"]
            dex_upnl = dex["USDT"]["unrealised_pnl"]
            dex_wallet = round(float(dex["USDT"]["wallet_balance"]), 2)
            dex_equity = round(float(dex["USDT"]["equity"]), 2)

            return {
                "dex_balance": dex_balance,
                "dex_pnl": dex_pnl,
                "dex_upnl": dex_upnl,
                "dex_wallet": dex_wallet,
                "dex_equity": dex_equity,
            }
        except KeyError as e:
            print(f"Error: {e}")
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An unknown error occured in get_balance(): {e}")
            log.warning(f"{e}")

class OrderBookData:
    def __init__(self, exchange, symbol):
        self.exchange = exchange
        self.symbol = symbol

    def get_orderbook(self):
        try:
            ob = self.exchange.fetch_order_book(self.symbol)
            bid = ob["bids"][0][0]
            ask = ob["asks"][0][0]
            return bid, ask
        except Exception as e:
            log.warning(f"{e}")

    def get_bid(self):
        bid, _ = self.get_orderbook()
        return bid

    def get_ask(self):
        _, ask = self.get_orderbook()
        return ask


class MarketData:
    def __init__(self, exchange, symbol):
        self.exchange = exchange
        self.symbol = symbol

    def get_market_data(self):
        try:
            self.exchange.load_markets()
            precision = self.exchange.market(self.symbol)["info"]["price_scale"]
            leverage = self.exchange.market(self.symbol)["info"]["leverage_filter"]["max_leverage"]
            min_trade_qty = self.exchange.market(self.symbol)["info"]["lot_size_filter"]["min_trading_qty"]
            return precision, leverage, min_trade_qty
        except Exception as e:
            log.warning(f"{e}")


def send_pnl_message(messengers, short_pos_unpl, long_pos_unpl, short_pos_unpl_pct, long_pos_unpl_pct):
    # Prepare the PNL message
    pnl_message = f"PNL Information:\n" \
                  f"Short Position Unpl: {short_pos_unpl}\n" \
                  f"Long Position Unpl: {long_pos_unpl}\n" \
                  f"Short Position Unpl %: {short_pos_unpl_pct}\n" \
                  f"Long Position Unpl %: {long_pos_unpl_pct}"

    # Send the PNL message using the messengers instance
    messengers.send_message_to_all_messengers(message=pnl_message)


def print_lot_sizes(max_trade_qty, leverage, min_trade_qty):
    print(f"Min Trade Qty: {min_trade_qty}")
    print_lot_size(1, Fore.LIGHTRED_EX, max_trade_qty, leverage, min_trade_qty)
    print_lot_size(0.01, Fore.LIGHTCYAN_EX, max_trade_qty, leverage, min_trade_qty)
    print_lot_size(0.005, Fore.LIGHTCYAN_EX, max_trade_qty, leverage, min_trade_qty)
    print_lot_size(0.002, Fore.LIGHTGREEN_EX, max_trade_qty, leverage, min_trade_qty)
    print_lot_size(0.001, Fore.LIGHTGREEN_EX, max_trade_qty, leverage, min_trade_qty)

def calc_lot_size(lot_size, max_trade_qty, min_trade_qty):
    trade_qty_x = max_trade_qty / (1.0 / lot_size)
    decimals_count = count_decimal_places(min_trade_qty)
    trade_qty_x_round = round(trade_qty_x, decimals_count)
    return trade_qty_x, trade_qty_x_round

def print_lot_size(lot_size, color, max_trade_qty, leverage, min_trade_qty):
    not_enough_equity = Fore.RED + "({:.5g}) Not enough equity"
    trade_qty_x, trade_qty_x_round = calc_lot_size(lot_size, max_trade_qty, min_trade_qty)
    if trade_qty_x_round == 0:
        trading_not_possible = not_enough_equity.format(trade_qty_x)
        color = Fore.RED
    else:
        trading_not_possible = ""
    print(
        color
        + "{:.4g}x : {:.4g} {}".format(
            lot_size, trade_qty_x_round, trading_not_possible
        )
    )


def count_decimal_places(number):
    """
    Counts the number of digits after the decimal point in a floating-point number.

    :param number: The number to count decimal places for.
    :type number: float
    :return: The number of digits after the decimal point.
    :rtype: int
    """
    decimal_places = 0
    if "." in str(number):
        decimal_places = len(str(number).split(".")[1])
    return decimal_places
