from colorama import Fore
from typing import Optional, Tuple, List, Dict, Union
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
import pandas as pd
import time
import math
import numpy
import random
import ta as ta
import os
import uuid
import logging
import json
import threading
import ccxt
import pytz
import sqlite3
from .logger import Logger
from datetime import datetime, timedelta
from threading import Thread, Lock

from ..bot_metrics import BotDatabase


logging = Logger(logger_name="Strategy", filename="Strategy.log", stream=True)

class Strategy:
    initialized_symbols = set()
    initialized_symbols_lock = threading.Lock()

    class Bybit:
        def __init__(self, parent):
            self.parent = parent

    def __init__(self, exchange, config, manager, symbols_allowed=None):
    # def __init__(self, exchange, config, manager):
        self.exchange = exchange
        self.config = config
        self.manager = manager
        self.symbol = config.symbol
        self.symbols_allowed = symbols_allowed
        self.order_timestamps = {}
        self.entry_order_ids = {}
        self.long_dynamic_amount = {}
        self.short_dynamic_amount = {}
        self.printed_trade_quantities = False
        self.last_mfirsi_signal = None
        self.TAKER_FEE_RATE = Decimal("0.00055")  # 0.055%
        self.taker_fee_rate = 0.055 / 100
        self.max_long_trade_qty = None
        self.max_short_trade_qty = None
        self.initial_max_long_trade_qty = None
        self.initial_max_short_trade_qty = None
        self.long_leverage_increased = False
        self.short_leverage_increased = False
        self.open_symbols_count = 0
        self.last_stale_order_check_time = time.time()
        self.should_spoof = True
        self.max_long_trade_qty_per_symbol = {}
        self.max_short_trade_qty_per_symbol = {}
        self.initial_max_long_trade_qty_per_symbol = {}
        self.initial_max_short_trade_qty_per_symbol = {}
        self.long_pos_leverage_per_symbol = {}
        self.short_pos_leverage_per_symbol = {}
        self.last_cancel_time = 0
        self.spoofing_active = False
        self.spoofing_wall_size = 5
        self.spoofing_interval = 1  # Time interval between spoofing actions
        self.spoofing_duration = 5  # Spoofing duration in seconds
        self.whitelist = self.config.whitelist
        self.blacklist = self.config.blacklist
        self.max_usd_value = self.config.max_usd_value
        self.LEVERAGE_STEP = 0.002  # The step at which to increase leverage
        self.MAX_LEVERAGE = 0.3 #0.3  # The maximum allowable leverage
        self.QTY_INCREMENT = 0.02 # How much your position size increases
        self.MAX_PCT_EQUITY = 0.1
        self.ORDER_BOOK_DEPTH = 10
        self.lock = threading.Lock()  # Create a lock
        self.last_order_time = {}  # 
        self.symbol_locks = {}
        self.order_ids = {}

        self.bybit = self.Bybit(self)

    def initialize_symbol(self, symbol, total_equity, best_ask_price):
        with self.initialized_symbols_lock:
            if symbol not in self.initialized_symbols:
                self.initialize_trade_quantities(symbol, total_equity, best_ask_price, self.max_leverage)
                logging.info(f"Initialized quantities for {symbol}. Initial long qty: {self.initial_max_long_trade_qty_per_symbol.get(symbol, 'N/A')}, Initial short qty: {self.initial_max_short_trade_qty_per_symbol.get(symbol, 'N/A')}")
                self.initialized_symbols.add(symbol)
                return True
            else:
                logging.info(f"{symbol} is already initialized.")
                return False
            
    class OrderBookAnalyzer:
        def __init__(self, exchange, symbol, depth=10):
            self.exchange = exchange
            self.symbol = symbol
            self.depth = depth

        def get_order_book(self):
            return self.exchange.get_orderbook(self.symbol)

        def buying_pressure(self):
            order_book = self.get_order_book()
            top_bids = order_book['bids'][:self.depth]
            total_bids = sum([bid[1] for bid in top_bids])
            
            top_asks = order_book['asks'][:self.depth]
            total_asks = sum([ask[1] for ask in top_asks])
            
            return total_bids > total_asks

        def selling_pressure(self):
            order_book = self.get_order_book()
            top_bids = order_book['bids'][:self.depth]
            total_bids = sum([bid[1] for bid in top_bids])
            
            top_asks = order_book['asks'][:self.depth]
            total_asks = sum([ask[1] for ask in top_asks])
            
            return total_asks > total_bids

        def order_book_imbalance(self):
            if self.buying_pressure():
                return "buy_wall"
            elif self.selling_pressure():
                return "sell_wall"
            else:
                return "neutral"

    def detect_order_book_walls(self, symbol, threshold=5.0):
        order_book = self.exchange.get_orderbook(symbol)
        bids = order_book['bids']
        asks = order_book['asks']

        avg_bid_size = sum([bid[1] for bid in bids[:10]]) / 10
        bid_walls = [(price, size) for price, size in bids if size > avg_bid_size * threshold]

        avg_ask_size = sum([ask[1] for ask in asks[:10]]) / 10
        ask_walls = [(price, size) for price, size in asks if size > avg_ask_size * threshold]

        if bid_walls:
            logging.info(f"Detected buy walls at {bid_walls} for {symbol}")
        if ask_walls:
            logging.info(f"Detected sell walls at {ask_walls} for {symbol}")

        return bid_walls, ask_walls

    TAKER_FEE_RATE = 0.00055

    def calculate_trading_fee(self, qty, executed_price, fee_rate=TAKER_FEE_RATE):
        order_value = qty / executed_price
        trading_fee = order_value * fee_rate
        return trading_fee

    def calculate_orderbook_strength(self, symbol):
        analyzer = self.OrderBookAnalyzer(self.exchange, symbol, depth=self.ORDER_BOOK_DEPTH)
        
        order_book = analyzer.get_order_book()
        
        top_bids = order_book['bids'][:self.ORDER_BOOK_DEPTH]
        total_bid_quantity = sum([bid[1] for bid in top_bids])
        
        top_asks = order_book['asks'][:self.ORDER_BOOK_DEPTH]
        total_ask_quantity = sum([ask[1] for ask in top_asks])
        
        if (total_bid_quantity + total_ask_quantity) == 0:
            return 0.5  # Neutral strength
        
        strength = total_bid_quantity / (total_bid_quantity + total_ask_quantity)
        
        return strength

    def identify_walls(self, order_book, type="buy"):
        # Threshold for what constitutes a wall (this can be adjusted)
        WALL_THRESHOLD = 5.0  # for example, 5 times the average size of top orders
        
        if type == "buy":
            orders = order_book['bids']
        else:
            orders = order_book['asks']

        avg_size = sum([order[1] for order in orders[:10]]) / 10  # average size of top 10 orders
        
        walls = []
        for price, size in orders:
            if size > avg_size * WALL_THRESHOLD:
                walls.append(price)
        
        return walls
    
    def get_order_book_imbalance(self, symbol):
        analyzer = self.OrderBookAnalyzer(self.exchange, symbol, self.ORDER_BOOK_DEPTH)
        return analyzer.order_book_imbalance()
        
    def print_order_book_imbalance(self, symbol):
        imbalance = self.get_order_book_imbalance(symbol)
        print(f"Order Book Imbalance for {symbol}: {imbalance}")

    def log_order_book_walls(self, symbol, interval_in_seconds):
        """
        Log the presence of buy/sell walls every 'interval_in_seconds'.
        """
        # Initialize counters for buy and sell wall occurrences
        buy_wall_count = 0
        sell_wall_count = 0

        start_time = time.time()

        while True:
            # Fetch the current order book for the symbol
            order_book = self.exchange.get_orderbook(symbol)
            
            # Identify buy and sell walls
            buy_walls = self.identify_walls(order_book, type="buy")
            sell_walls = self.identify_walls(order_book, type="sell")

            if buy_walls:
                buy_wall_count += 1
            if sell_walls:
                sell_wall_count += 1

            elapsed_time = time.time() - start_time

            # Log the counts every 'interval_in_seconds'
            if elapsed_time >= interval_in_seconds:
                logging.info(f"Buy Walls detected in the last {interval_in_seconds/60} minutes: {buy_wall_count}")
                logging.info(f"Sell Walls detected in the last {interval_in_seconds/60} minutes: {sell_wall_count}")

                # Reset the counters and start time
                buy_wall_count = 0
                sell_wall_count = 0
                start_time = time.time()

            time.sleep(60)  # Check every minute

    def start_wall_logging(self, symbol):
        """
        Start logging buy/sell walls at different intervals.
        """
        intervals = [300, 600, 1800, 3600]  # 5 minutes, 10 minutes, 30 minutes, 1 hour in seconds

        # Start a new thread for each interval
        for interval in intervals:
            t = threading.Thread(target=self.log_order_book_walls, args=(symbol, interval))
            t.start()

    def compute_average_daily_gain_percentage(self, initial_equity, current_equity, days_passed):
        """Compute average daily gain percentage."""
        if days_passed == 0:  # To prevent division by zero
            return 0
        gain = (current_equity - initial_equity) / initial_equity * 100
        avg_daily_gain = gain / days_passed
        return avg_daily_gain

    def calculate_adg(self, days=30):
        """
        Calculate the Average Daily Gain over a specified number of days.
        """
        try:
            # Fetch closed trade history or daily balance history
            history = self.exchange.fetch_closed_trades_history(days)

            # Calculate daily gains
            daily_gains = []
            for day in range(1, days + 1):
                # Assuming history data has 'date' and 'profit_loss' fields
                day_data = [trade for trade in history if trade['date'] == day]
                daily_gain = sum(trade['profit_loss'] for trade in day_data)
                daily_gains.append(daily_gain)

            # Calculate ADG
            adg = sum(daily_gains) / days
            return adg

        except Exception as e:
            logging.error(f"Error in calculate_adg: {e}")
            return None

    def fetch_closed_trades_history(self, days):
        """
        Fetch the closed trades history for the specified number of days.
        This method should be implemented in the Exchange class.
        """
        # This is a placeholder. You need to implement this method based on your exchange's API.
        pass

    def get_symbols_allowed(self, account_name):
        for exchange in self.config["exchanges"]:
            if exchange["account_name"] == account_name:
                return exchange.get("symbols_allowed", None)
        return None

    def get_funding_rate(self, symbol):
        api_data = self.manager.get_api_data(symbol)
        return api_data.get('Funding', None)

    def is_funding_rate_acceptable(self, symbol: str) -> bool:
        """
        Check if the funding rate for a symbol is within the acceptable bounds defined by the MaxAbsFundingRate.

        :param symbol: The symbol for which the check is being made.
        :return: True if the funding rate is within acceptable bounds, False otherwise.
        """
        MaxAbsFundingRate = self.config.MaxAbsFundingRate

        logging.info(f"Max Abs Funding Rate: {self.config.MaxAbsFundingRate}")

        api_data = self.manager.get_api_data(symbol)
        funding_rate = api_data['Funding']

        logging.info(f"Funding rate for {symbol} : {funding_rate}")

        # Check if funding rate is None
        if funding_rate is None:
            logging.warning(f"Funding rate for {symbol} is None.")
            return False

        # Check for longs and shorts combined
        return -MaxAbsFundingRate <= funding_rate <= MaxAbsFundingRate

    def convert_to_binance_symbol(symbol: str) -> str:
        """Convert Bybit's symbol name to Binance's format."""
        if symbol.startswith("SHIB1000"):
            return "1000SHIBUSDT"
        # Add more conversions as needed
        # if symbol.startswith("ANOTHEREXAMPLE"):
        #     return "BINANCEFORMAT"
        return symbol
    
    def can_proceed_with_trade(self, symbol: str) -> dict:
        """
        Check if we can proceed with a long or short trade based on the funding rate.

        Parameters:
            symbol (str): The trading symbol to check.
            
        Returns:
            dict: A dictionary containing boolean values for 'can_long' and 'can_short'.
        """
        # Initialize the result dictionary
        result = {
            'can_long': False,
            'can_short': False
        }

        # Retrieve the maximum absolute funding rate from config
        max_abs_funding_rate = self.config.MaxAbsFundingRate

        # Get the current funding rate for the symbol
        funding_rate = self.get_funding_rate(symbol)
        
        # If funding_rate is None, we can't make a decision
        if funding_rate is None:
            return result
        
        # Check conditions for long and short trades
        if funding_rate <= max_abs_funding_rate:
            result['can_long'] = True

        if funding_rate >= -max_abs_funding_rate:
            result['can_short'] = True

        return result

    def initialize_trade_quantities(self, symbol, total_equity, best_ask_price, max_leverage):
        # Check if the symbol has been initialized before
        if symbol in self.initialized_symbols:
            return

        # Calculate the max trade quantity if it hasn't been initialized for long or short trades
        if symbol not in self.max_long_trade_qty_per_symbol or symbol not in self.max_short_trade_qty_per_symbol:
            try:
                max_trade_qty = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
            except Exception as e:
                logging.error(f"Error calculating max trade quantity for {symbol}: {e}")
                return  # Exit the function if there's an error

            self.max_long_trade_qty_per_symbol.setdefault(symbol, max_trade_qty)
            self.max_short_trade_qty_per_symbol.setdefault(symbol, max_trade_qty)

            logging.info(f"For symbol {symbol} Calculated max_long_trade_qty: {max_trade_qty}, max_short_trade_qty: {max_trade_qty}")

        # Initialize the initial max trade quantities if not set
        if symbol not in self.initial_max_long_trade_qty_per_symbol:
            self.initial_max_long_trade_qty_per_symbol[symbol] = self.max_long_trade_qty_per_symbol[symbol]
            logging.info(f"Initial max long trade qty set for {symbol} to {self.initial_max_long_trade_qty_per_symbol[symbol]}")

        if symbol not in self.initial_max_short_trade_qty_per_symbol:
            self.initial_max_short_trade_qty_per_symbol[symbol] = self.max_short_trade_qty_per_symbol[symbol]
            logging.info(f"Initial max short trade qty set for {symbol} to {self.initial_max_short_trade_qty_per_symbol[symbol]}")

        # Add the symbol to the initialized symbols set
        self.initialized_symbols.add(symbol)

    def calculate_dynamic_amount_obstrength(self, symbol, total_equity, best_ask_price, max_leverage):
        self.initialize_trade_quantities(symbol, total_equity, best_ask_price, max_leverage)

        market_data = self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)
        min_qty = float(market_data["min_qty"])
        logging.info(f"Min qty for {symbol} : {min_qty}")

        # Starting with 0.1% of total equity for both long and short orders
        long_dynamic_amount = 0.001 * total_equity
        short_dynamic_amount = 0.001 * total_equity

        # Calculate the order book strength
        strength = self.calculate_orderbook_strength(symbol)
        logging.info(f"OB strength: {strength}")

        # Reduce the aggressive multiplier from 10 to 5
        aggressive_steps = max(0, (strength - 0.5) * 5)  # This ensures values are always non-negative
        long_dynamic_amount += aggressive_steps * min_qty
        short_dynamic_amount += aggressive_steps * min_qty

        logging.info(f"Long dynamic amount for {symbol} {long_dynamic_amount}")
        logging.info(f"Short dynamic amount for {symbol} {short_dynamic_amount}")

        # Reduce the maximum allowed dynamic amount to be more conservative
        AGGRESSIVE_MAX_PCT_EQUITY = 0.05  # 5% of the total equity
        max_allowed_dynamic_amount = AGGRESSIVE_MAX_PCT_EQUITY * total_equity
        logging.info(f"Max allowed dynamic amount for {symbol} : {max_allowed_dynamic_amount}")

        # Determine precision level directly
        precision_level = len(str(min_qty).split('.')[-1]) if '.' in str(min_qty) else 0
        logging.info(f"min_qty: {min_qty}, precision_level: {precision_level}")

        # Round the dynamic amounts based on precision level
        long_dynamic_amount = round(long_dynamic_amount, precision_level)
        short_dynamic_amount = round(short_dynamic_amount, precision_level)
        logging.info(f"Rounded long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        # Apply the cap to the dynamic amounts
        long_dynamic_amount = min(long_dynamic_amount, max_allowed_dynamic_amount)
        short_dynamic_amount = min(short_dynamic_amount, max_allowed_dynamic_amount)

        logging.info(f"Forced min qty long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)
        self.check_amount_validity_once_bybit(short_dynamic_amount, symbol)

        # Using min_qty if dynamic amount is too small
        if long_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for long_dynamic_amount")
            long_dynamic_amount = min_qty
        if short_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for short_dynamic_amount")
            short_dynamic_amount = min_qty

        logging.info(f"Symbol: {symbol} Final long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        return long_dynamic_amount, short_dynamic_amount, min_qty

    def update_dynamic_amounts(self, symbol, total_equity, best_ask_price):
        if symbol not in self.long_dynamic_amount or symbol not in self.short_dynamic_amount:
            long_dynamic_amount, short_dynamic_amount, _ = self.calculate_dynamic_amount_v2(symbol, total_equity, best_ask_price, self.max_leverage)
            self.long_dynamic_amount[symbol] = long_dynamic_amount
            self.short_dynamic_amount[symbol] = short_dynamic_amount

        if symbol in self.max_long_trade_qty_per_symbol:
            self.long_dynamic_amount[symbol] = min(
                self.long_dynamic_amount[symbol], 
                self.max_long_trade_qty_per_symbol[symbol]
            )
        if symbol in self.max_short_trade_qty_per_symbol:
            self.short_dynamic_amount[symbol] = min(
                self.short_dynamic_amount[symbol], 
                self.max_short_trade_qty_per_symbol[symbol]
            )

        logging.info(f"Updated dynamic amounts for {symbol}. New long_dynamic_amount: {self.long_dynamic_amount[symbol]}, New short_dynamic_amount: {self.short_dynamic_amount[symbol]}")

    def calculate_dynamic_amount_v2(self, symbol, total_equity, best_ask_price, max_leverage):

        # self.initialize_trade_quantities(symbol, total_equity, best_ask_price, max_leverage)

        market_data = self.get_market_data_with_retry(symbol, max_retries = 100, retry_delay = 5)

        min_qty = float(market_data["min_qty"])

        logging.info(f"Min qty for {symbol} : {min_qty}")

        long_dynamic_amount = 0.0001 * total_equity
        short_dynamic_amount = 0.0001 * total_equity

        # long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty_per_symbol[symbol]
        # short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty_per_symbol[symbol]

        logging.info(f"Initial long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        # Cap the dynamic amount if it exceeds the maximum allowed
        max_allowed_dynamic_amount = (self.MAX_PCT_EQUITY / 100) * total_equity
        logging.info(f"Max allowed dynamic amount for {symbol} : {max_allowed_dynamic_amount}")
        
        # Determine precision level directly
        precision_level = len(str(min_qty).split('.')[-1]) if '.' in str(min_qty) else 0
        logging.info(f"min_qty: {min_qty}, precision_level: {precision_level}")

        # Round the dynamic amounts based on precision level
        long_dynamic_amount = round(long_dynamic_amount, precision_level)
        short_dynamic_amount = round(short_dynamic_amount, precision_level)

        logging.info(f"Rounded long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        long_dynamic_amount = min(long_dynamic_amount, max_allowed_dynamic_amount)
        short_dynamic_amount = min(short_dynamic_amount, max_allowed_dynamic_amount)

        logging.info(f"Forced min qty long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)
        self.check_amount_validity_once_bybit(short_dynamic_amount, symbol)

        # Using min_qty if dynamic amount is too small
        if long_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for long_dynamic_amount")
            long_dynamic_amount = min_qty
        if short_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for short_dynamic_amount")
            short_dynamic_amount = min_qty

        logging.info(f"Symbol: {symbol} Final long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        return long_dynamic_amount, short_dynamic_amount, min_qty

    def calculate_dynamic_amount(self, symbol, total_equity, best_ask_price, max_leverage):

        self.initialize_trade_quantities(symbol, total_equity, best_ask_price, max_leverage)

        market_data = self.get_market_data_with_retry(symbol, max_retries = 100, retry_delay = 5)

        long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty_per_symbol[symbol]
        short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty_per_symbol[symbol]

        logging.info(f"Initial long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        # Cap the dynamic amount if it exceeds the maximum allowed
        max_allowed_dynamic_amount = (self.MAX_PCT_EQUITY / 100) * total_equity
        logging.info(f"Max allowed dynamic amount for {symbol} : {max_allowed_dynamic_amount}")

        min_qty = float(market_data["min_qty"])

        logging.info(f"Min qty for {symbol} : {min_qty}")
        
        # Determine precision level directly
        precision_level = len(str(min_qty).split('.')[-1]) if '.' in str(min_qty) else 0
        logging.info(f"min_qty: {min_qty}, precision_level: {precision_level}")

        # Round the dynamic amounts based on precision level
        long_dynamic_amount = round(long_dynamic_amount, precision_level)
        short_dynamic_amount = round(short_dynamic_amount, precision_level)

        logging.info(f"Rounded long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        long_dynamic_amount = min(long_dynamic_amount, max_allowed_dynamic_amount)
        short_dynamic_amount = min(short_dynamic_amount, max_allowed_dynamic_amount)

        logging.info(f"Forced min qty long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)
        self.check_amount_validity_once_bybit(short_dynamic_amount, symbol)

        # Using min_qty if dynamic amount is too small
        if long_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for long_dynamic_amount")
            long_dynamic_amount = min_qty
        if short_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for short_dynamic_amount")
            short_dynamic_amount = min_qty

        logging.info(f"Symbol: {symbol} Final long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        return long_dynamic_amount, short_dynamic_amount, min_qty

    def get_all_moving_averages(self, symbol, max_retries=3, delay=5):
        for _ in range(max_retries):
            m_moving_averages = self.manager.get_1m_moving_averages(symbol)
            m5_moving_averages = self.manager.get_5m_moving_averages(symbol)

            ma_6_high = m_moving_averages["MA_6_H"]
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

            # Check if the data is correct
            if all(isinstance(value, (float, int, numpy.number)) for value in [ma_6_high, ma_6_low, ma_3_low, ma_3_high, ma_1m_3_high, ma_5m_3_high]):
                return {
                    "ma_6_high": ma_6_high,
                    "ma_6_low": ma_6_low,
                    "ma_3_low": ma_3_low,
                    "ma_3_high": ma_3_high,
                    "ma_1m_3_high": ma_1m_3_high,
                    "ma_5m_3_high": ma_5m_3_high,
                }

            # If the data is not correct, wait for a short delay
            time.sleep(delay)

        raise ValueError("Failed to fetch valid moving averages after multiple attempts.")

    def get_current_price(self, symbol):
        return self.exchange.get_current_price(symbol)

    def market_open_order(self, symbol: str, side: str, amount: float, position_idx: int):
        """
        Opens a new position with a market order.
        """
        try:
            params = {'position_idx': position_idx}  # include the position_idx for hedge mode
            order = self.exchange.create_contract_v3_order(symbol, 'Market', side, amount, params=params)
            logging.info(f"Market order to {side} {amount} of {symbol} placed successfully.")
        except Exception as e:
            logging.error(f"Failed to place market order: {e}")

    def market_close_order(self, symbol: str, side: str, amount: float, position_idx: int):
        """
        Closes an existing position with a market order.
        """
        try:
            params = {'position_idx': position_idx}  # include the position_idx for hedge mode
            # The side should be 'sell' for long positions and 'buy' for short positions to close them.
            order = self.exchange.create_contract_v3_order(symbol, 'Market', side, amount, params=params)
            logging.info(f"Market order to close {side} position of {amount} {symbol} placed successfully.")
        except Exception as e:
            logging.error(f"Failed to place market close order: {e}")

    def can_place_order(self, symbol, interval=60):
        with self.lock:
            current_time = time.time()
            logging.info(f"Attempting to check if an order can be placed for {symbol} at {current_time}")
            
            if symbol in self.last_order_time:
                time_difference = current_time - self.last_order_time[symbol]
                logging.info(f"Time since last order for {symbol}: {time_difference} seconds")
                
                if time_difference <= interval:
                    logging.warning(f"Rate limit exceeded for {symbol}. Denying order placement.")
                    return False
                
            self.last_order_time[symbol] = current_time
            logging.info(f"Order allowed for {symbol} at {current_time}")
            return True

    ## v5
    def process_position_data(self, open_position_data):
        position_details = {}

        for position in open_position_data:
            info = position.get('info', {})
            symbol = info.get('symbol', '').split(':')[0]  # Splitting to get the base symbol

            # Ensure 'size', 'side', and 'avgPrice' keys exist in the info dictionary
            if 'size' in info and 'side' in info and 'avgPrice' in info:
                size = float(info['size'])
                side = info['side'].lower()
                avg_price = float(info['avgPrice'])

                # Initialize the nested dictionary if the symbol is not already in position_details
                if symbol not in position_details:
                    position_details[symbol] = {'long': {'qty': 0, 'avg_price': None}, 'short': {'qty': 0, 'avg_price': None}}

                # Update the quantities and average prices based on the side of the position
                if side == 'buy':
                    position_details[symbol]['long']['qty'] += size
                    position_details[symbol]['long']['avg_price'] = avg_price
                elif side == 'sell':
                    position_details[symbol]['short']['qty'] += size
                    position_details[symbol]['short']['avg_price'] = avg_price

        return position_details
    
        
    def get_position_update_time(self, symbol):
        try:
            # Fetch position information
            position = self.exchange.fetch_position(symbol)

            # Extract the updated time in milliseconds
            updated_time_ms = position.get('info', {}).get('updatedTime')

            if updated_time_ms:
                # Convert from milliseconds to a datetime object
                updated_time = datetime.datetime.fromtimestamp(updated_time_ms / 1000.0)
                return updated_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return "Updated time not available"
        except Exception as e:
            return f"Error fetching position update time: {e}"

    def fetch_recent_trades_for_symbol(self, symbol, since=None, limit=100):
        """
        Fetch recent trades for a specific symbol.
        :param str symbol: The symbol to fetch trades for.
        :param int since: Timestamp in milliseconds to fetch trades since.
        :param int limit: The number of trades to fetch.
        :returns: A list of recent trades.
        """
        try:
            recent_trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)
            logging.info(f"Recent trades fetched for {symbol}: {recent_trades}")
            return recent_trades
        except Exception as e:
            logging.error(f"Error fetching recent trades for {symbol}: {e}")
            return []
        
    # def fetch_recent_trades_for_symbol(self, symbol, since=None, limit=100):
    #     """
    #     Fetch recent trades for a given symbol.

    #     :param str symbol: The trading pair symbol.
    #     :param int since: Timestamp in milliseconds for fetching trades since this time.
    #     :param int limit: The maximum number of trades to fetch.
    #     :return: List of recent trades.
    #     """
    #     return self.exchange.fetch_recent_trades(symbol, since, limit)
    
    def place_postonly_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        logging.info(f"Attempting to place post-only order for {symbol}. Side: {side}, Amount: {amount}, Price: {price}, PositionIdx: {positionIdx}, ReduceOnly: {reduceOnly}")
        if not self.can_place_order(symbol):
            logging.warning(f"Order placement rate limit exceeded for {symbol}. Skipping...")
            return None
        return self.postonly_limit_order_bybit(symbol, side, amount, price, positionIdx, reduceOnly)

    def postonly_limit_entry_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        """Places a post-only limit entry order and stores its ID."""
        order = self.postonly_limit_order_bybit(symbol, side, amount, price, positionIdx, reduceOnly)
        
        # If the order was successfully placed, store its ID as an entry order ID for the symbol
        if order and 'id' in order:
            if symbol not in self.entry_order_ids:
                self.entry_order_ids[symbol] = []
            self.entry_order_ids[symbol].append(order['id'])
            logging.info(f"Stored order ID {order['id']} for symbol {symbol}. Current order IDs for {symbol}: {self.entry_order_ids[symbol]}")
        else:
            logging.warning(f"Failed to store order ID for symbol {symbol} due to missing 'id' or unsuccessful order placement.")

        return order

    def limit_order_bybit_unified(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit_unified(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

    def is_entry_order(self, symbol, order_id):
        """Checks if the given order ID is an entry order for the symbol."""
        is_entry = order_id in self.entry_order_ids.get(symbol, [])
        logging.info(f"Checking if order ID {order_id} for symbol {symbol} is an entry order: {is_entry}")
        return is_entry

    def remove_entry_order(self, symbol, order_id):
        """Removes the given order ID from the entry orders list for the symbol."""
        if symbol in self.entry_order_ids:
            self.entry_order_ids[symbol] = [oid for oid in self.entry_order_ids[symbol] if oid != order_id]
            logging.info(f"Removed order ID {order_id} from entry orders for symbol {symbol}. Current order IDs for {symbol}: {self.entry_order_ids[symbol]}")
        else:
            logging.warning(f"Symbol {symbol} not found in entry_order_ids. Cannot remove order ID {order_id}.")

    def postonly_limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        """Directly places the order with the exchange."""
        params = {"reduceOnly": reduceOnly, "postOnly": True}
        order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)

        # Log and store the order ID if the order was placed successfully
        if order and 'id' in order:
            logging.info(f"Successfully placed post-only limit order for {symbol}. Order ID: {order['id']}. Side: {side}, Amount: {amount}, Price: {price}, PositionIdx: {positionIdx}, ReduceOnly: {reduceOnly}")
            if symbol not in self.order_ids:
                self.order_ids[symbol] = []
            self.order_ids[symbol].append(order['id'])
        else:
            logging.warning(f"Failed to place post-only limit order for {symbol}. Side: {side}, Amount: {amount}, Price: {price}, PositionIdx: {positionIdx}, ReduceOnly: {reduceOnly}")

        return order

    def postonly_limit_order_bybit_nolimit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly, "postOnly": True}
        logging.info(f"Placing {side} limit order for {symbol} at {price} with qty {amount} and params {params}...")
        try:
            order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
            logging.info(f"Nolimit postonly order result for {symbol}: {order}")
            if order is None:
                logging.warning(f"Order result is None for {side} limit order on {symbol}")
            return order
        except Exception as e:
            logging.error(f"Error placing order: {str(e)}")
            logging.exception("Stack trace for error in placing order:")  # This will log the full stack trace

    def postonly_limit_order_bybit_s(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly, "postOnly": True}
        logging.info(f"Placing {side} limit order for {symbol} at {price} with qty {amount} and params {params}...")
        try:
            order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
            logging.info(f"Order result: {order}")
            if order is None:
                logging.warning(f"Order result is None for {side} limit order on {symbol}")
            return order
        except Exception as e:
            logging.error(f"Error placing order: {str(e)}")
            logging.exception("Stack trace for error in placing order:")  # This will log the full stack trace

    def limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

    def entry_order_exists(self, open_orders: list, side: str) -> bool:
        for order in open_orders:
            if order["side"].lower() == side and order["reduce_only"] == False:
                logging.info(f"An entry order for side {side} already exists.")
                return True
        logging.info(f"No entry order found for side {side}.")
        return False
    
    def get_open_take_profit_order_quantity(self, orders, side):
        for order in orders:
            if order['side'].lower() == side.lower() and order['reduce_only']:
                return order['qty'], order['id']
        return None, None

    def get_open_take_profit_order_quantities(self, orders, side):
        take_profit_orders = []
        for order in orders:
            logging.info(f"Raw order data: {order}")
            order_side = order.get('side')
            reduce_only = order.get('reduce_only', False)
            
            if order_side and isinstance(order_side, str) and order_side.lower() == side.lower() and reduce_only:
                qty = order.get('qty', 0)
                order_id = order.get('id')
                take_profit_orders.append((qty, order_id))
        return take_profit_orders

    def get_open_additional_entry_orders(self, symbol, orders, side):
        additional_entry_orders = []
        for order in orders:
            logging.info(f"Raw order data additional entries: {order}")
            order_side = order.get('side')
            order_id = order.get('id')
            
            if order_id and self.is_entry_order(symbol, order_id) and order_side and isinstance(order_side, str) and order_side.lower() == side.lower():
                qty = order.get('qty', 0)
                additional_entry_orders.append((qty, order_id))
        return additional_entry_orders

    def cancel_take_profit_orders(self, symbol, side):
        self.exchange.cancel_close_bybit(symbol, side)

    def limit_order_binance(self, symbol, side, amount, price, reduceOnly=False):
        try:
            params = {"reduceOnly": reduceOnly}
            order = self.exchange.create_limit_order_binance(symbol, side, amount, price, params=params)
            return order
        except Exception as e:
            print(f"An error occurred in limit_order(): {e}")

    def get_open_take_profit_order_quantities_binance(self, open_orders, order_side):
        return [(order['amount'], order['id']) for order in open_orders
                if order['type'] == 'TAKE_PROFIT_MARKET' and
                order['side'].lower() == order_side.lower() and
                order.get('reduce_only', False)]        
                
    def get_open_take_profit_limit_order_quantities_binance(self, open_orders, order_side):
        return [(order['amount'], order['id']) for order in open_orders
                if order['type'] == 'LIMIT' and
                order['side'].lower() == order_side.lower() and
                order.get('reduce_only', False)]

    def cancel_take_profit_orders_binance(self, symbol, side):
        self.exchange.cancel_close_bybit(symbol, side)


    def calculate_short_conditions(self, short_pos_price, ma_6_low, short_take_profit, short_pos_qty):
        if short_pos_price is not None:
            should_add_to_short = short_pos_price < ma_6_low
            short_tp_distance_percent = ((short_take_profit - short_pos_price) / short_pos_price) * 100
            short_expected_profit_usdt = short_tp_distance_percent / 100 * short_pos_price * short_pos_qty
            logging.info(f"Short TP price: {short_take_profit}, TP distance in percent: {-short_tp_distance_percent:.2f}%, Expected profit: {-short_expected_profit_usdt:.2f} USDT")
            return should_add_to_short, short_tp_distance_percent, short_expected_profit_usdt
        return None, None, None

    def calculate_long_conditions(self, long_pos_price, ma_6_low, long_take_profit, long_pos_qty):
        if long_pos_price is not None:
            should_add_to_long = long_pos_price > ma_6_low
            long_tp_distance_percent = ((long_take_profit - long_pos_price) / long_pos_price) * 100
            long_expected_profit_usdt = long_tp_distance_percent / 100 * long_pos_price * long_pos_qty
            logging.info(f"Long TP price: {long_take_profit}, TP distance in percent: {long_tp_distance_percent:.2f}%, Expected profit: {long_expected_profit_usdt:.2f} USDT")
            return should_add_to_long, long_tp_distance_percent, long_expected_profit_usdt
        return None, None, None
    
    def short_trade_condition(self, current_ask, ma_3_high):
        if current_ask is None or ma_3_high is None:
            return False
        return current_ask > ma_3_high

    def long_trade_condition(self, current_bid, ma_3_low):
        if current_bid is None or ma_3_low is None:
            return False
        return current_bid < ma_3_low

    def add_short_trade_condition(self, short_pos_price, ma_6_high):
        if short_pos_price is None or ma_6_high is None:
            return False
        return short_pos_price > ma_6_high

    def add_long_trade_condition(self, long_pos_price, ma_6_low):
        if long_pos_price is None or ma_6_low is None:
            return False
        return long_pos_price < ma_6_low

    def get_market_data_with_retry(self, symbol, max_retries=5, retry_delay=5):
        for i in range(max_retries):
            try:
                return self.exchange.get_market_data_bybit(symbol)
            except Exception as e:
                if i < max_retries - 1:
                    print(f"Error occurred while fetching market data: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e

    def get_market_data_with_retry_binance(self, symbol, max_retries=5, retry_delay=5):
        for i in range(max_retries):
            try:
                return self.exchange.get_market_data_binance(symbol)
            except Exception as e:
                if i < max_retries - 1:
                    print(f"Error occurred while fetching market data: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e

    def get_balance_with_retry(self, quote_currency, max_retries=5, retry_delay=5):
        for i in range(max_retries):
            try:
                return self.exchange.get_balance_bybit(quote_currency)
            except Exception as e:
                if i < max_retries - 1:
                    print(f"Error occurred while fetching balance: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise e

    def calc_max_trade_qty(self, symbol, total_equity, best_ask_price, max_leverage, max_retries=5, retry_delay=5):
        wallet_exposure = self.config.wallet_exposure
        for i in range(max_retries):
            try:
                market_data = self.get_market_data_with_retry(symbol, max_retries = 5, retry_delay = 5)
                max_trade_qty = round(
                    (float(total_equity) * wallet_exposure / float(best_ask_price))
                    / (100 / max_leverage),
                    int(float(market_data["min_qty"])),
                )

                logging.info(f"Max trade qty for {symbol} calculated: {max_trade_qty} ")
                
                return max_trade_qty
            except TypeError as e:
                if total_equity is None:
                    print(f"Error: total_equity is None. Retrying in {retry_delay} seconds...")
                if best_ask_price is None:
                    print(f"Error: best_ask_price is None. Retrying in {retry_delay} seconds...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

        raise Exception("Failed to calculate maximum trade quantity after maximum retries.")

    def calc_max_trade_qty_multiv2(self, symbol, total_equity, best_ask_price, max_leverage, long_pos_qty_open_symbol, short_pos_qty_open_symbol, max_retries=5, retry_delay=5):
        wallet_exposure = self.config.wallet_exposure
        for i in range(max_retries):
            try:
                market_data = self.exchange.get_market_data_bybit(symbol)
                base_max_trade_qty = round(
                    (float(total_equity) * wallet_exposure / float(best_ask_price))
                    / (100 / max_leverage),
                    int(float(market_data["min_qty"])),
                )
                
                # Apply your logic to differentiate between long and short here
                max_long_trade_qty = base_max_trade_qty  # Modify based on long_pos_qty_open_symbol
                max_short_trade_qty = base_max_trade_qty  # Modify based on short_pos_qty_open_symbol
                
                return max_long_trade_qty, max_short_trade_qty
                    
            except TypeError as e:
                if total_equity is None:
                    print(f"Error: total_equity is None. Retrying in {retry_delay} seconds...")
                if best_ask_price is None:
                    print(f"Error: best_ask_price is None. Retrying in {retry_delay} seconds...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...")
                
            time.sleep(retry_delay)
            
        raise Exception("Failed to calculate maximum trade quantity after maximum retries.")

    def calc_max_trade_qty_multi(self, total_equity, best_ask_price, max_leverage, max_retries=5, retry_delay=5):
        wallet_exposure = self.config.wallet_exposure
        for i in range(max_retries):
            try:
                market_data = self.exchange.get_market_data_bybit(self.symbol)
                max_trade_qty = round(
                    (float(total_equity) * wallet_exposure / float(best_ask_price))
                    / (100 / max_leverage),
                    int(float(market_data["min_qty"])),
                )
                
                # Assuming the logic for max_long_trade_qty and max_short_trade_qty is the same
                max_long_trade_qty = max_trade_qty
                max_short_trade_qty = max_trade_qty
                
                return max_long_trade_qty, max_short_trade_qty
                
            except TypeError as e:
                if total_equity is None:
                    print(f"Error: total_equity is None. Retrying in {retry_delay} seconds...")
                if best_ask_price is None:
                    print(f"Error: best_ask_price is None. Retrying in {retry_delay} seconds...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...")
                
            time.sleep(retry_delay)
            
        raise Exception("Failed to calculate maximum trade quantity after maximum retries.")

    def calc_max_trade_qty_binance(self, total_equity, best_ask_price, max_leverage, step_size, max_retries=5, retry_delay=5):
        wallet_exposure = self.config.wallet_exposure
        precision = int(-math.log10(float(step_size)))
        for i in range(max_retries):
            try:
                max_trade_qty = (
                    float(total_equity) * wallet_exposure / float(best_ask_price)
                ) / (100 / max_leverage)
                max_trade_qty = math.floor(max_trade_qty * 10**precision) / 10**precision

                return max_trade_qty
            except TypeError as e:
                if total_equity is None:
                    print(f"Error: total_equity is None. Retrying in {retry_delay} seconds...")
                if best_ask_price is None:
                    print(f"Error: best_ask_price is None. Retrying in {retry_delay} seconds...")
            except Exception as e:
                print(f"An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

        raise Exception("Failed to calculate maximum trade quantity after maximum retries.")

    def check_amount_validity_bybit(self, amount, symbol):
        market_data = self.exchange.get_market_data_bybit(symbol)
        min_qty_bybit = market_data["min_qty"]
        if float(amount) < min_qty_bybit:
            logging.info(f"The amount you entered ({amount}) is less than the minimum required by Bybit for {symbol}: {min_qty_bybit}.")
            return False
        else:
            logging.info(f"The amount you entered ({amount}) is valid for {symbol}")
            return True

    def check_amount_validity_once_bybit(self, amount, symbol):
        if not self.check_amount_validity_bybit:
            market_data = self.exchange.get_market_data_bybit(symbol)
            min_qty_bybit = market_data["min_qty"]
            if float(amount) < min_qty_bybit:
                logging.info(f"The amount you entered ({amount}) is less than the minimum required by Bybit for {symbol}: {min_qty_bybit}.")
                return False
            else:
                logging.info(f"The amount you entered ({amount}) is valid for {symbol}")
                return True

    def check_amount_validity_once_binance(self, amount, symbol):
        if not self.checked_amount_validity_binance:
            market_data = self.exchange.get_market_data_binance(symbol)
            min_qty = float(market_data["min_qty"])
            step_size = float(market_data['step_size'])
            
            if step_size == 0.0:
                print(f"Step size is zero for {symbol}. Cannot calculate precision.")
                return False

            precision = int(-math.log10(step_size))
            
            # Ensure the amount is a multiple of step_size
            amount = round(amount, precision)
            
            if amount < min_qty:
                print(f"The amount you entered ({amount}) is less than the minimum required by Binance for {symbol}: {min_qty}.")
                return False
            else:
                print(f"The amount you entered ({amount}) is valid for {symbol}")
                return True

    def monitor_and_close_positions(self, symbol, threshold=0.02):
        """
        Monitors liquidation risk and closes positions if the current price is within the threshold
        of the liquidation price.
        
        Parameters:
            symbol (str): The trading symbol (e.g., "BTCUSD").
            threshold (float): The percentage threshold for closing positions (default is 2% or 0.02).
        """
        
        # Fetch the current positions
        position_data = self.exchange.get_positions_bybit(symbol)
        short_liq_price = float(position_data["short"]["liq_price"])
        long_liq_price = float(position_data["long"]["liq_price"])
        
        # Fetch the current market price
        current_price = float(self.exchange.get_current_price(symbol))
        
        # Calculate the thresholds
        short_close_threshold = short_liq_price * (1 + threshold)
        long_close_threshold = long_liq_price * (1 - threshold)
        
        # Check if the current price is within the threshold for the short position and close if necessary
        if current_price >= short_close_threshold:
            logging.warning(f"Closing short position for {symbol} as the current price {current_price} is close to the liquidation price {short_liq_price}.")
            self.market_close_order_bybit(symbol, "sell")  # Assuming this is your function to close a market order
        
        # Check if the current price is within the threshold for the long position and close if necessary
        if current_price <= long_close_threshold:
            logging.warning(f"Closing long position for {symbol} as the current price {current_price} is close to the liquidation price {long_liq_price}.")
            self.market_close_order_bybit(symbol, "buy")  # Assuming this is your function to close a market order

        # If neither condition is met, log that positions are safe
        else:
            logging.info(f"Positions for {symbol} are currently safe from liquidation.")

    def print_trade_quantities_once_bybit(self, symbol):
        if not self.printed_trade_quantities:
            if symbol not in self.max_long_trade_qty_per_symbol:
                logging.warning(f"Symbol {symbol} not initialized in max_long_trade_qty_per_symbol. Unable to print trade quantities.")
                return

            wallet_exposure = self.config.wallet_exposure
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            self.exchange.print_trade_quantities_bybit(
                self.max_long_trade_qty_per_symbol[symbol], 
                [0.001, 0.01, 0.1, 1, 2.5, 5], 
                wallet_exposure, 
                best_ask_price
            )
            self.printed_trade_quantities = True

    # def print_trade_quantities_once_bybit(self, symbol, max_trade_qty):
    #     if not self.printed_trade_quantities:
    #         wallet_exposure = self.config.wallet_exposure
    #         best_ask_price = self.exchange.get_orderbook(self.symbol)['asks'][0][0]
    #         #self.exchange.print_trade_quantities_bybit(max_trade_qty, [0.001, 0.01, 0.1, 1, 2.5, 5], wallet_exposure, best_ask_price)
    #         self.exchange.print_trade_quantities_bybit(self.max_long_trade_qty_per_symbol[symbol], [0.001, 0.01, 0.1, 1, 2.5, 5], wallet_exposure, best_ask_price)
    #         self.printed_trade_quantities = True

    def print_trade_quantities_once_huobi(self, max_trade_qty, symbol):
        if not self.printed_trade_quantities:
            wallet_exposure = self.config.wallet_exposure
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            self.exchange.print_trade_quantities_bybit(max_trade_qty, [0.001, 0.01, 0.1, 1, 2.5, 5], wallet_exposure, best_ask_price)
            self.printed_trade_quantities = True


    def get_1m_moving_averages(self, symbol):
        return self.manager.get_1m_moving_averages(symbol)

    def get_5m_moving_averages(self, symbol):
        return self.manager.get_5m_moving_averages(symbol)

    def get_positions_bybit(self):
        position_data = self.exchange.get_positions_bybit(self.symbol)
        return position_data

    def calculate_next_update_time(self):
        """Returns the time for the next TP update, which is 6 minutes from the current time."""
        now = datetime.now()
        next_update_time = now + timedelta(minutes=1)
        return next_update_time.replace(microsecond=0)

    def calculate_short_take_profit_bybit(self, short_pos_price, symbol):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating short_target_price. short_pos_price={short_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            try:
                short_target_price = short_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing short_target_price. short_target_price={short_target_price}, price_precision={price_precision}")
                return None

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None

    def calculate_long_take_profit_bybit(self, long_pos_price, symbol):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating long_target_price. long_pos_price={long_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            try:
                long_target_price = long_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing long_target_price. long_target_price={long_target_price}, price_precision={price_precision}")
                return None

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def calculate_long_take_profit_spread_bybit_fees(self, long_pos_price, quantity, symbol, increase_percentage=0):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating long_target_price. long_pos_price={long_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            if increase_percentage is None:
                increase_percentage = 0

            # Add the specified percentage to the take profit target price
            long_target_price = long_target_price * (1 + Decimal(increase_percentage)/100)

            # Adjust for taker fee
            order_value = Decimal(quantity) * Decimal(long_pos_price)
            fee_amount = order_value * self.TAKER_FEE_RATE
            long_target_price += fee_amount / Decimal(quantity)  # Convert the fee back to price terms

            try:
                long_target_price = long_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing long_target_price. long_target_price={long_target_price}, price_precision={price_precision}")
                return None

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def calculate_short_take_profit_spread_bybit_fees(self, short_pos_price, quantity, symbol, increase_percentage=0):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating short_target_price. short_pos_price={short_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            if increase_percentage is None:
                increase_percentage = 0

            # Apply increase percentage to the calculated short target price
            short_target_price = short_target_price * (Decimal('1') - Decimal(increase_percentage) / Decimal('100'))

            # Adjust for taker fee
            order_value = Decimal(quantity) * Decimal(short_pos_price)
            fee_amount = order_value * self.TAKER_FEE_RATE
            short_target_price -= fee_amount / Decimal(quantity)  # Convert the fee back to price terms

            try:
                short_target_price = short_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing short_target_price. short_target_price={short_target_price}, price_precision={price_precision}")
                return None

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None


    def calculate_long_take_profit_spread_bybit(self, long_pos_price, symbol, increase_percentage=0):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        logging.info(f"Five min data for {symbol}: {five_min_data}")
        logging.info(f"Price precision for {symbol}: {price_precision}")

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating long_target_price. long_pos_price={long_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            if increase_percentage is None:
                increase_percentage = 0

            # Add the specified percentage to the take profit target price
            long_target_price = long_target_price * (1 + Decimal(increase_percentage)/100)

            try:
                long_target_price = long_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing long_target_price. long_target_price={long_target_price}, price_precision={price_precision}")
                return None

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def calculate_short_take_profit_spread_bybit(self, short_pos_price, symbol, increase_percentage=0):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        logging.info(f"Five min data for {symbol}: {five_min_data}")
        logging.info(f"Price precision for {symbol}: {price_precision}")

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating short_target_price. short_pos_price={short_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            if increase_percentage is None:
                increase_percentage = 0

            # Apply increase percentage to the calculated short target price
            short_target_price = short_target_price * (Decimal('1') - Decimal(increase_percentage) / Decimal('100'))

            try:
                short_target_price = short_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing short_target_price. short_target_price={short_target_price}, price_precision={price_precision}")
                return None

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None

    def calculate_take_profits_based_on_spread(self, short_pos_price, long_pos_price, symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit):
        """
        Calculate long and short take profits based on the spread.
        :param short_pos_price: The short position price.
        :param long_pos_price: The long position price.
        :param symbol: The symbol for which the take profits are being calculated.
        :param five_minute_distance: The five-minute distance.
        :param previous_five_minute_distance: The previous five-minute distance.
        :param short_take_profit: Existing short take profit.
        :param long_take_profit: Existing long take profit.
        :return: Calculated short_take_profit, long_take_profit.
        """
        # Log the inputs
        logging.info(f"Inputs to calculate_take_profits_based_on_spread: short_pos_price={short_pos_price}, long_pos_price={long_pos_price}, symbol={symbol}, five_minute_distance={five_minute_distance}, previous_five_minute_distance={previous_five_minute_distance}, short_take_profit={short_take_profit}, long_take_profit={long_take_profit}")

        if five_minute_distance != previous_five_minute_distance or short_take_profit is None or long_take_profit is None:
            short_take_profit = self.calculate_short_take_profit_spread_bybit(short_pos_price, symbol, five_minute_distance)
            long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, five_minute_distance)
            
            # Log the calculated values
            logging.info(f"Newly calculated short_take_profit: {short_take_profit}")
            logging.info(f"Newly calculated long_take_profit: {long_take_profit}")
        
        return short_take_profit, long_take_profit

    def calculate_short_take_profit_binance(self, short_pos_price, symbol):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        print(f"five_min_data: {five_min_data}")

        market_data = self.get_market_data_with_retry_binance(symbol, max_retries = 5, retry_delay = 5)
        print(f"market_data: {market_data}")

        step_size = market_data['step_size']
        price_precision = int(-math.log10(float(step_size))) if float(step_size) < 1 else 8
        print(f"price_precision: {price_precision}")


        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
                print(f"short_target_price: {short_target_price}")
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating short_target_price. short_pos_price={short_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            try:
                short_target_price = short_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )

                print(f"quantized short_target_price: {short_target_price}")
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing short_target_price. short_target_price={short_target_price}, price_precision={price_precision}")
                return None

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None

    def calculate_long_take_profit_binance(self, long_pos_price, symbol):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        print(f"five_min_data: {five_min_data}")

        market_data = self.get_market_data_with_retry_binance(symbol, max_retries = 5, retry_delay = 5)
        print(f"market_data: {market_data}")

        step_size = market_data['step_size']
        price_precision = int(-math.log10(float(step_size))) if float(step_size) < 1 else 8
        print(f"price_precision: {price_precision}")
        
        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
                print(f"long_target_price: {long_target_price}")
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating long_target_price. long_pos_price={long_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            try:
                long_target_price = long_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
                print(f"quantized long_target_price: {long_target_price}")
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing long_target_price. long_target_price={long_target_price}, price_precision={price_precision}")
                return None

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None
        
    def check_short_long_conditions(self, best_bid_price, ma_3_high):
        should_short = best_bid_price > ma_3_high
        should_long = best_bid_price < ma_3_high
        return should_short, should_long

    def get_5m_averages(self):
        ma_values = self.manager.get_5m_moving_averages(self.symbol)
        if ma_values is not None:
            high_value = ma_values["MA_3_H"]
            low_value = ma_values["MA_3_L"]
            range_5m = high_value - low_value
            return high_value, low_value
        else:
            return None, None

    def print_lot_sizes(self, max_trade_qty, market_data):
        print(f"Min Trade Qty: {market_data['min_qty']}")
        self.print_lot_size(1, Fore.LIGHTRED_EX, max_trade_qty, market_data)
        self.print_lot_size(0.01, Fore.LIGHTCYAN_EX, max_trade_qty, market_data)
        self.print_lot_size(0.005, Fore.LIGHTCYAN_EX, max_trade_qty, market_data)
        self.print_lot_size(0.002, Fore.LIGHTGREEN_EX, max_trade_qty, market_data)
        self.print_lot_size(0.001, Fore.LIGHTGREEN_EX, max_trade_qty, market_data)

    def calc_lot_size(self, lot_size, max_trade_qty, market_data):
        trade_qty_x = max_trade_qty / (1.0 / lot_size)
        decimals_count = self.count_decimal_places(market_data['min_qty'])
        trade_qty_x_round = round(trade_qty_x, decimals_count)
        return trade_qty_x, trade_qty_x_round

    def print_lot_size(self, lot_size, color, max_trade_qty, market_data):
        not_enough_equity = Fore.RED + "({:.5g}) Not enough equity"
        trade_qty_x, trade_qty_x_round = self.calc_lot_size(lot_size, max_trade_qty, market_data)
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

    def count_decimal_places(self, number):
        decimal_str = str(number)
        if '.' in decimal_str:
            return len(decimal_str.split('.')[1])
        else:
            return 0

    def calculate_trade_quantity(self, symbol, leverage):
        dex_equity = self.exchange.get_balance_bybit('USDT')
        trade_qty = (float(dex_equity) * self.current_wallet_exposure) / leverage
        return trade_qty

    def adjust_position_wallet_exposure(self, symbol):
        if self.current_wallet_exposure > self.wallet_exposure_limit:
            desired_wallet_exposure = self.wallet_exposure_limit
            # Calculate the necessary position size to achieve the desired wallet exposure
            max_trade_qty = self.calculate_trade_quantity(symbol, 1)
            current_trade_qty = self.calculate_trade_quantity(symbol, 1 / self.current_wallet_exposure)
            reduction_qty = current_trade_qty - max_trade_qty
            # Reduce the position to the desired wallet exposure level
            self.exchange.reduce_position_bybit(symbol, reduction_qty)

    def truncate(self, number: float, precision: int) -> float:
        return float(Decimal(number).quantize(Decimal('0.' + '0'*precision), rounding=ROUND_DOWN))

    def format_symbol(self, symbol):
        """
        Format the given symbol string to include a '/' between the base and quote currencies.
        The function handles base currencies of 3 to 4 characters and quote currencies of 3 to 4 characters.
        """
        quote_currencies = ["USDT", "USD", "BTC", "ETH"]
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base + '/' + quote
        return None

#### HUOBI ####

    def calculate_short_take_profit_huobi(self, short_pos_price, symbol):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
            short_target_price = short_target_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                #rounding=ROUND_HALF_UP
                rounding=ROUND_DOWN
            )

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None

    def calculate_long_take_profit_huobi(self, long_pos_price, symbol):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
            long_target_price = long_target_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_UP
            )

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def get_open_take_profit_order_quantities_huobi(self, orders, side):
        take_profit_orders = []
        for order in orders:
            order_info = {
                "id": order['id'],
                "price": order['price'],
                "qty": order['qty'],
                "order_status": order['order_status'],
                "side": order['side']
            }
            if (
                order_info['side'].lower() == side.lower()
                and order_info['order_status'] == '3'  # Adjust the condition based on your order status values
                and order_info['id'] not in (self.long_entry_order_ids if side.lower() == 'sell' else self.short_entry_order_ids)
            ):
                take_profit_orders.append((order_info['qty'], order_info['id']))
        return take_profit_orders


    def get_open_take_profit_order_quantity_huobi(self, symbol, orders, side):
        current_price = self.get_current_price(symbol)  # You'd need to implement this function
        long_quantity = None
        long_order_id = None
        short_quantity = None
        short_order_id = None

        for order in orders:
            order_price = float(order['price'])
            if order['side'] == 'sell':
                if side == "close_long" and order_price > current_price:
                    if 'reduce_only' in order and order['reduce_only']:
                        continue
                    long_quantity = order['qty']
                    long_order_id = order['id']
                elif side == "close_short" and order_price < current_price:
                    if 'reduce_only' in order and order['reduce_only']:
                        continue
                    short_quantity = order['qty']
                    short_order_id = order['id']
            else:
                if side == "close_short" and order_price > current_price:
                    if 'reduce_only' in order and not order['reduce_only']:
                        continue
                    short_quantity = order['qty']
                    short_order_id = order['id']
                elif side == "close_long" and order_price < current_price:
                    if 'reduce_only' in order and not order['reduce_only']:
                        continue
                    long_quantity = order['qty']
                    long_order_id = order['id']

        if side == "close_long":
            return long_quantity, long_order_id
        elif side == "close_short":
            return short_quantity, short_order_id

        return None, None

    def calculate_actual_quantity_huobi(self, position_qty, parsed_symbol_swap):
        contract_size_per_unit = self.exchange.get_contract_size_huobi(parsed_symbol_swap)
        return position_qty * contract_size_per_unit

    def parse_symbol_swap_huobi(self, symbol):
        if "huobi" in self.exchange.name.lower():
            base_currency = symbol[:-4]
            quote_currency = symbol[-4:] 
            return f"{base_currency}/{quote_currency}:{quote_currency}"
        return symbol

    def cancel_take_profit_orders_huobi(self, symbol, side):
        self.exchange.cancel_close_huobi(symbol, side)

    def verify_account_type_huobi(self):
        if not self.account_type_verified:
            try:
                current_account_type = self.exchange.check_account_type_huobi()
                print(f"Current account type at start: {current_account_type}")
                if current_account_type['data']['account_type'] != '1':
                    self.exchange.switch_account_type_huobi(1)
                    time.sleep(0.05)
                    print(f"Changed account type")
                else:
                    print(f"Account type is already 1")

                self.account_type_verified = True  # set to True after account type is verified or changed
            except Exception as e:
                print(f"Error in switching account type {e}")
                
    # MFIRSI with retry
    def initialize_MFIRSI(self, symbol):
        max_retries = 5
        retry_delay = 2  # delay in seconds
        for attempt in range(max_retries):
            try:
                df = self.exchange.fetch_ohlcv(symbol, timeframe='5m')

                #print(df.head())
                df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                df['ma'] = ta.trend.sma_indicator(df['close'], window=14)
                df['open_less_close'] = (df['open'] < df['close']).astype(int)

                df['buy_condition'] = ((df['mfi'] < 20) & (df['rsi'] < 35) & (df['open_less_close'] == 1)).astype(int)
                df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 65) & (df['open_less_close'] == 0)).astype(int)

                return df
            except Exception as e:
                if attempt < max_retries - 1:  # If not the last attempt
                    print(f"Error occurred while fetching OHLCV data: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:  # If the last attempt
                    print(f"Error occurred while fetching OHLCV data: {e}. No more retries left.")
                    raise  # Re-raise the last exception

    def should_long_MFI(self, symbol):
        df = self.initialize_MFIRSI(symbol)
        condition = df.iloc[-1]['buy_condition'] == 1
        if condition:
            self.last_mfirsi_signal = 'long'
        return condition

    def should_short_MFI(self, symbol):
        df = self.initialize_MFIRSI(symbol)
        condition = df.iloc[-1]['sell_condition'] == 1
        if condition:
            self.last_mfirsi_signal = 'short'
        return condition

    def parse_contract_code(self, symbol):
        parsed_symbol = symbol.split(':')[0]  # Remove ':USDT'
        parsed_symbol = parsed_symbol.replace('/', '-')  # Replace '/' with '-'
        return parsed_symbol

    def extract_symbols_from_positions_bybit(self, positions: List[dict]) -> List[str]:
        """
        Extract symbols from the positions data.
        
        :param positions: List of position dictionaries.
        :return: List of extracted symbols.
        """
        # Ensure only valid symbols are considered
        symbols = [pos.get('symbol').split(':')[0] for pos in positions if isinstance(pos, dict) and pos.get('symbol')]
        return symbols

    def retry_api_call(self, function, *args, max_retries=5, base_delay=5, max_delay=60, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                return function(*args, **kwargs)
            except Exception as e:  # Catch all exceptions
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Error occurred: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        raise Exception(f"Failed to execute the API function after {max_retries} retries.")

    def can_trade_new_symbol(self, open_symbols: list, symbols_allowed: int, current_symbol: str) -> bool:
        """
        Checks if the bot can trade a given symbol.
        """
        self.open_symbols_count = len(open_symbols)  # Update the attribute with the current count

        logging.info(f"Open symbols count: {self.open_symbols_count}")

        if symbols_allowed is None:
            symbols_allowed = 10  # Use a default value if symbols_allowed is not specified

        # If the current symbol is already being traded, allow it
        if current_symbol in open_symbols:
            return True

        # If we haven't reached the symbol limit, allow a new symbol to be traded
        if self.open_symbols_count < symbols_allowed:
            return True

        # If none of the above conditions are met, don't allow the new trade
        return False

    def update_shared_data(self, symbol_data: dict, open_position_data: dict, open_symbols_count: int):
        # Update and serialize symbol data
        with open("symbol_data.json", "w") as f:
            json.dump(symbol_data, f)

        # Update and serialize open position data
        with open("open_positions_data.json", "w") as f:
            json.dump(open_position_data, f)
        
        # Update and serialize count of open symbols
        with open("open_symbols_count.json", "w") as f:
            json.dump({"count": open_symbols_count}, f)

    def manage_liquidation_risk(self, long_pos_price, short_pos_price, long_liq_price, short_liq_price, symbol, amount):
        # Create some thresholds for when to act
        long_threshold = self.config.long_liq_pct
        short_threshold = self.config.short_liq_pct

        # Let's assume you have methods to get the best bid and ask prices
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]

        # Check if the long position is close to being liquidated
        if long_pos_price is not None and long_liq_price is not None:
            long_diff = abs(long_pos_price - long_liq_price) / long_pos_price
            if long_diff < long_threshold:
                # Place a post-only limit order to offset the risk
                self.postonly_limit_order_bybit(symbol, "buy", amount, best_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Placed a post-only limit order to offset long position risk on {symbol} at {best_bid_price}")

        # Check if the short position is close to being liquidated
        if short_pos_price is not None and short_liq_price is not None:
            short_diff = abs(short_pos_price - short_liq_price) / short_pos_price
            if short_diff < short_threshold:
                # Place a post-only limit order to offset the risk
                self.postonly_limit_order_bybit(symbol, "sell", amount, best_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Placed a post-only limit order to offset short position risk on {symbol} at {best_ask_price}")

    def calculate_spoofing_amount(self, symbol, total_equity, best_ask_price, max_leverage):
        if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
            max_trade_qty = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
            self.max_long_trade_qty = max_trade_qty
            self.max_short_trade_qty = max_trade_qty

        # For demonstration, I'm using a much larger base.
        long_spoofing_amount = 0.1 * self.initial_max_long_trade_qty
        short_spoofing_amount = 0.1 * self.initial_max_short_trade_qty

        market_data = self.get_market_data_with_retry(symbol, max_retries = 5, retry_delay = 5)
        min_qty = float(market_data["min_qty"])

        # Respect the min_qty requirement.
        long_spoofing_amount = max(long_spoofing_amount, min_qty)
        short_spoofing_amount = max(short_spoofing_amount, min_qty)

        return long_spoofing_amount, short_spoofing_amount

    def get_active_order_count(self, symbol):
        try:
            active_orders = self.exchange.fetch_open_orders(symbol)
            return len(active_orders)
        except Exception as e:
            logging.warning(f"Could not fetch active orders for {symbol}: {e}")
            return 0

    def pm(self, symbol, short_dynamic_amount, long_dynamic_amount):
        if self.spoofing_active:
            # Fetch orderbook and positions
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = Decimal(orderbook['bids'][0][0])
            best_ask_price = Decimal(orderbook['asks'][0][0])

            position_data = self.exchange.get_positions_bybit(symbol)
            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            if short_pos_qty is None and long_pos_qty is None:
                logging.warning(f"Could not fetch position quantities for {symbol}. Skipping manipulation.")
                return

            # Determine which position is larger
            larger_position = "long" if long_pos_qty > short_pos_qty else "short"

            # Adjust spoofing_wall_size based on the larger position
            base_spoofing_wall_size = self.spoofing_wall_size
            adjusted_spoofing_wall_size = base_spoofing_wall_size + 10

            spoofing_orders = []
            safety_margin = Decimal('0.05')
            base_gap = Decimal('0.005')
            momentum_ignition_multiplier = 2  # To really push the price 

            for i in range(adjusted_spoofing_wall_size):
                gap = base_gap + Decimal(i) * Decimal('0.002')

                if larger_position == "long":
                    # Calculate long spoof price based on best ask price (top of the order book)
                    spoof_price_long = best_ask_price + gap + safety_margin
                    spoof_price_long = spoof_price_long.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_long = self.limit_order_bybit(symbol, "sell", long_dynamic_amount * momentum_ignition_multiplier, spoof_price_long, positionIdx=2, reduceOnly=False)
                    spoofing_orders.append(spoof_order_long)

                if larger_position == "short":
                    # Calculate short spoof price based on best bid price (top of the order book)
                    spoof_price_short = best_bid_price - gap - safety_margin
                    spoof_price_short = spoof_price_short.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_short = self.limit_order_bybit(symbol, "buy", short_dynamic_amount * momentum_ignition_multiplier, spoof_price_short, positionIdx=1, reduceOnly=False)
                    spoofing_orders.append(spoof_order_short)

            # Ignite momentum by placing a real order in the desired direction (with a slightly larger amount)
            if larger_position == "long":
                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount * 1.5, best_bid_price, positionIdx=2, reduceOnly=False)
            else:
                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount * 1.5, best_ask_price, positionIdx=1, reduceOnly=False)

            # Sleep for the spoofing duration and then cancel all placed orders
            time.sleep(self.spoofing_duration)

            # Cancel orders and handle errors
            for order in spoofing_orders:
                if 'id' in order:
                    logging.info(f"Manipulation order: {order}")
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                else:
                    logging.warning(f"Could not place manipulation order: {order.get('error', 'Unknown error')}")

            # Deactivate spoofing for the next cycle
            self.spoofing_active = False

    def apm(self, symbol, short_dynamic_amount, long_dynamic_amount):
        if self.spoofing_active:
            # Fetch orderbook and positions
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = Decimal(orderbook['bids'][0][0])
            best_ask_price = Decimal(orderbook['asks'][0][0])

            position_data = self.exchange.get_positions_bybit(symbol)
            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            if short_pos_qty is None and long_pos_qty is None:
                logging.warning(f"Could not fetch position quantities for {symbol}. Skipping action.")
                return

            larger_position = "long" if long_pos_qty > short_pos_qty else "short"

            # Adjust order_wall_size
            base_order_wall_size = self.spoofing_wall_size
            adjusted_order_wall_size = base_order_wall_size + 5

            aggressive_orders = []
            #safety_margin = Decimal('0.05')
            #base_gap = Decimal('0.005')
            safety_margin = Decimal('0.10')
            base_gap = Decimal('0.15')

            for i in range(adjusted_order_wall_size):
                gap = base_gap + Decimal(i) * Decimal('0.002')  # Increasing gap for each subsequent order

                if larger_position == "long":
                    # Progressively push the price upwards
                    aggressive_price_long = best_ask_price + gap + safety_margin
                    aggressive_price_long = aggressive_price_long.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    aggressive_order_long = self.limit_order_bybit(symbol, "sell", long_dynamic_amount * 2, aggressive_price_long, positionIdx=2, reduceOnly=False)
                    aggressive_orders.append(aggressive_order_long)

                if larger_position == "short":
                    # Progressively push the price downwards
                    aggressive_price_short = best_bid_price - gap - safety_margin
                    aggressive_price_short = aggressive_price_short.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    aggressive_order_short = self.limit_order_bybit(symbol, "buy", short_dynamic_amount * 2, aggressive_price_short, positionIdx=1, reduceOnly=False)
                    aggressive_orders.append(aggressive_order_short)

            # Short delay to give illusion of real activity
            time.sleep(0.5)

            # Rapid cancellation to further the illusion
            for order in aggressive_orders:
                if 'id' in order:
                    logging.info(f"Aggressive order: {order}")
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                else:
                    logging.warning(f"Could not place aggressive order: {order.get('error', 'Unknown error')}")

            # Deactivate for the next cycle
            self.spoofing_active = False

    def mp(self, symbol, short_dynamic_amount, long_dynamic_amount, direction):
        if self.spoofing_active:
            with threading.Lock():
                if symbol not in self.symbol_locks:
                    self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                # Fetch orderbook
                orderbook = self.exchange.get_orderbook(symbol)
                best_bid_price = Decimal(orderbook['bids'][0][0])
                best_ask_price = Decimal(orderbook['asks'][0][0])

                # Initialize variables
                spoofing_orders = []

                # Dynamic safety_margin and base_gap based on asset's price
                safety_margin = best_ask_price * Decimal('0.0040')  # 0.10% of current price
                base_gap = best_ask_price * Decimal('0.0040')  # 0.10% of current price

                for i in range(self.spoofing_wall_size):
                    gap = base_gap + Decimal(i) * Decimal('0.002')  # Increasing gap for each subsequent order

                    size_variation = random.uniform(0.5, 1.5)
                    spoof_multiplier = 10  # Amplified order sizes
                    aggressiveness = Decimal('0.002')  # Adjusted based on desired level of aggressiveness

                    if direction == 'up':
                        spoof_price = best_ask_price + gap + safety_margin - aggressiveness
                        spoof_order = self.limit_order_bybit(symbol, "sell", long_dynamic_amount * spoof_multiplier * size_variation, spoof_price, positionIdx=2, reduceOnly=False)
                        spoofing_orders.append(spoof_order)
                    elif direction == 'down':
                        spoof_price = best_bid_price - gap - safety_margin + aggressiveness
                        spoof_order = self.limit_order_bybit(symbol, "buy", short_dynamic_amount * spoof_multiplier * size_variation, spoof_price, positionIdx=1, reduceOnly=False)
                        spoofing_orders.append(spoof_order)

                # Sleep for the spoofing duration and then cancel all placed orders
                sleep_duration = random.uniform(self.spoofing_duration - 5, self.spoofing_duration + 5)
                time.sleep(sleep_duration)

                # Cancel orders and handle errors
                for order in spoofing_orders:
                    if 'id' in order:
                        self.exchange.cancel_order_by_id(order['id'], symbol)
                    else:
                        logging.warning(f"Could not place spoofing order for {symbol}: {order.get('error', 'Unknown error')}")

                # Deactivate spoofing for the next cycle
                self.spoofing_active = False

    def helper(self, symbol, short_dynamic_amount, long_dynamic_amount):
        if self.spoofing_active:
            # Fetch orderbook and positions
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = Decimal(orderbook['bids'][0][0])
            best_ask_price = Decimal(orderbook['asks'][0][0])

            position_data = self.exchange.get_positions_bybit(symbol)
            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            if short_pos_qty is None and long_pos_qty is None:
                logging.warning(f"Could not fetch position quantities for {symbol}. Skipping spoofing.")
                return

            # Determine which position is larger
            larger_position = "long" if long_pos_qty > short_pos_qty else "short"

            # Adjust spoofing_wall_size based on the larger position
            base_spoofing_wall_size = self.spoofing_wall_size
            adjusted_spoofing_wall_size = base_spoofing_wall_size + 5

            # Initialize variables
            spoofing_orders = []

            # Dynamic safety_margin and base_gap based on asset's price
            safety_margin = best_ask_price * Decimal('0.0040')  # 0.0030 # 0.10% of current price
            base_gap = best_ask_price * Decimal('0.0040') #0.0030  # 0.10% of current price

            for i in range(adjusted_spoofing_wall_size):
                gap = base_gap + Decimal(i) * Decimal('0.002')  # Increasing gap for each subsequent order

                if larger_position == "long":
                    # Calculate long spoof price based on best ask price (top of the order book)
                    spoof_price_long = best_ask_price + gap + safety_margin
                    spoof_price_long = spoof_price_long.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_long = self.limit_order_bybit(symbol, "sell", long_dynamic_amount * 1.5, spoof_price_long, positionIdx=2, reduceOnly=False)
                    spoofing_orders.append(spoof_order_long)

                if larger_position == "short":
                    # Calculate short spoof price based on best bid price (top of the order book)
                    spoof_price_short = best_bid_price - gap - safety_margin
                    spoof_price_short = spoof_price_short.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_short = self.limit_order_bybit(symbol, "buy", short_dynamic_amount * 1.5, spoof_price_short, positionIdx=1, reduceOnly=False)
                    spoofing_orders.append(spoof_order_short)

            # Sleep for the spoofing duration and then cancel all placed orders
            time.sleep(self.spoofing_duration)

            # Cancel orders and handle errors
            for order in spoofing_orders:
                if 'id' in order:
                    logging.info(f"Spoofing order for {symbol}: {order}")
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                else:
                    logging.warning(f"Could not place spoofing order for {symbol}: {order.get('error', 'Unknown error')}")

            # Deactivate spoofing for the next cycle
            self.spoofing_active = False

    def spoofing_action(self, symbol, short_dynamic_amount, long_dynamic_amount):
        if self.spoofing_active:
            # Fetch orderbook and positions
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = Decimal(orderbook['bids'][0][0])
            best_ask_price = Decimal(orderbook['asks'][0][0])

            position_data = self.exchange.get_positions_bybit(symbol)
            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            if short_pos_qty is None and long_pos_qty is None:
                logging.warning(f"Could not fetch position quantities for {symbol}. Skipping spoofing.")
                return

            # Determine which position is larger
            larger_position = "long" if long_pos_qty > short_pos_qty else "short"

            # Adjust spoofing_wall_size based on the larger position
            base_spoofing_wall_size = self.spoofing_wall_size
            adjusted_spoofing_wall_size = base_spoofing_wall_size + 5
            # adjusted_spoofing_wall_size = base_spoofing_wall_size + math.ceil(abs(long_pos_qty - short_pos_qty) / SOME_CONSTANT)

            # Initialize variables
            spoofing_orders = []
            #safety_margin = Decimal('0.05')  # 1% safety margin
            safety_margin = Decimal('0.10')
            #base_gap = Decimal('0.005')  # Base gap for spoofing orders
            #base_gap = Decimal ('0.01')
            #base_gap = Decimal('0.05')
            base_gap = Decimal('0.10')

            for i in range(adjusted_spoofing_wall_size):
                gap = base_gap + Decimal(i) * Decimal('0.002')  # Increasing gap for each subsequent order

                if larger_position == "long":
                    # Calculate long spoof price based on best ask price (top of the order book)
                    spoof_price_long = best_ask_price + gap + safety_margin
                    spoof_price_long = spoof_price_long.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_long = self.limit_order_bybit(symbol, "sell", long_dynamic_amount * 1.5, spoof_price_long, positionIdx=2, reduceOnly=False)
                    spoofing_orders.append(spoof_order_long)

                if larger_position == "short":
                    # Calculate short spoof price based on best bid price (top of the order book)
                    spoof_price_short = best_bid_price - gap - safety_margin
                    spoof_price_short = spoof_price_short.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_short = self.limit_order_bybit(symbol, "buy", short_dynamic_amount * 1.5, spoof_price_short, positionIdx=1, reduceOnly=False)
                    spoofing_orders.append(spoof_order_short)

            # Sleep for the spoofing duration and then cancel all placed orders
            time.sleep(self.spoofing_duration)

            # Cancel orders and handle errors
            for order in spoofing_orders:
                if 'id' in order:
                    logging.info(f"Spoofing order for {symbol}: {order}")
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                else:
                    logging.warning(f"Could not place spoofing order for {symbol}: {order.get('error', 'Unknown error')}")

            # Deactivate spoofing for the next cycle
            self.spoofing_active = False

    def aggressive_action(self, symbol, short_dynamic_amount, long_dynamic_amount):
        if self.spoofing_active:
            # Fetch orderbook and positions
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = Decimal(orderbook['bids'][0][0])
            best_ask_price = Decimal(orderbook['asks'][0][0])

            position_data = self.exchange.get_positions_bybit(symbol)
            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            if short_pos_qty is None and long_pos_qty is None:
                logging.warning(f"Could not fetch position quantities for {symbol}. Skipping spoofing.")
                return

            # Initialize variables
            spoofing_orders = []
            safety_margin = Decimal('0.02')  # 2% safety margin
            base_gap = Decimal('0.002')  # Base gap for spoofing orders

            for i in range(self.spoofing_wall_size):
                gap = base_gap + Decimal(i) * Decimal('0.001')  # Increasing gap for each subsequent order

                if long_pos_qty > 0:  # If there's a long position, place sell spoof orders
                    spoof_price_long = best_ask_price + gap + safety_margin
                    spoof_price_long = spoof_price_long.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_long = self.limit_order_bybit(symbol, "sell", long_dynamic_amount, spoof_price_long, positionIdx=2, reduceOnly=False)
                    spoofing_orders.append(spoof_order_long)

                if short_pos_qty > 0:  # If there's a short position, place buy spoof orders
                    spoof_price_short = best_bid_price - gap - safety_margin
                    spoof_price_short = spoof_price_short.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    spoof_order_short = self.limit_order_bybit(symbol, "buy", short_dynamic_amount, spoof_price_short, positionIdx=1, reduceOnly=False)
                    spoofing_orders.append(spoof_order_short)

            # Sleep for the spoofing duration and then cancel all placed orders
            time.sleep(self.spoofing_duration)

            # Cancel orders and handle errors
            for order in spoofing_orders:
                if 'id' in order:
                    logging.info(f"Spoofing order: {order}")
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                else:
                    logging.warning(f"Could not place spoofing order: {order.get('error', 'Unknown error')}")

            # Deactivate spoofing for the next cycle
            self.spoofing_active = False

    def spoof_order(self, symbol, side, max_amount, max_price, price_step=3.0, num_orders=5):
        total_amount = 0.0  # Initialize the total amount to zero

        # Calculate the minimum order size based on your requirements
        min_order_size = max_amount / num_orders

        # Get the current order book
        orderbook = self.exchange.get_orderbook(symbol)
        
        # Ensure there are enough levels in the order book
        if len(orderbook['bids']) >= num_orders:
            for i in range(num_orders):
                # Calculate the remaining amount to distribute among the orders
                remaining_amount = max_amount - total_amount

                # Ensure that the order size is not larger than the remaining amount or the minimum size
                order_size = min(min_order_size, remaining_amount)

                # Calculate the price level based on the current top of the book and move according to price_step
                top_price = orderbook['bids'][0][0] if side.lower() == "buy" else orderbook['asks'][0][0]
                price = top_price - (i + 1) * price_step if side.lower() == "buy" else top_price + (i + 1) * price_step

                # Ensure the price doesn't go out of the specified range
                if side.lower() == "buy":
                    price = max(price, max_price)
                else:
                    price = min(price, max_price)

                # Place the order
                order = self.limit_order(symbol, side, order_size, price, reduce_only=False)

                if order is not None:
                    total_amount += order_size  # Update the total amount with the order size
                    logging.info(f"Placed spoof order - Amount: {order_size:.4f} {symbol}, Price: {price:.2f}")

                # Break the loop if the total amount reaches the maximum
                if total_amount >= max_amount:
                    break

            logging.info(f"Total spoof orders placed: {total_amount:.4f} {symbol}")
        else:
            logging.info("Not enough levels in the order book to place spoof orders.")

    def spoof_scalping_strategy(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        # Define the max_spoof_amount
        max_spoof_amount = 100  # This should be adjusted according to your desired spoof order size.

        # Check if the volume and distance conditions are met
        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                # For long trends, we will spoof a buy order and then scalp a sell
                if trend.lower() == "long":
                    # Place a spoof buy order
                    best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
                    self.spoof_order(symbol, 'buy', max_spoof_amount, best_bid_price)

                    # Wait a moment for the market to potentially react
                    time.sleep(5)

                    # Check for an opportunity to scalp a short based on the boosted price
                    best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                    if should_short:
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    # Now, cancel the spoofed order
                    self.cancel_all_orders(symbol)

                # For short trends, we will spoof a sell order and then scalp a buy
                elif trend.lower() == "short":
                    # Place a spoof sell order
                    best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                    self.spoof_order(symbol, 'sell', max_spoof_amount, best_ask_price)

                    # Wait a moment for the market to potentially react
                    time.sleep(5)

                    # Check for an opportunity to scalp a long based on the suppressed price
                    best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
                    if should_long:
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    # Now, cancel the spoofed order
                    self.cancel_all_orders(symbol)

                # Handling long positions
                if (trend.lower() == "long" and mfi.lower() == "long") and should_long and long_pos_qty == 0:
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                elif (trend.lower() == "long" and mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price:
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Handling short positions
                if (trend.lower() == "short" and mfi.lower() == "short") and should_short and short_pos_qty == 0:
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info("Placed initial short entry")

                elif (trend.lower() == "short" and mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price:
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def calculate_qfl_levels(self, symbol: str, timeframe='5m', lookback_period=12):
        # Fetch historical candle data
        candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback_period)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamps to readable dates (optional)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Find the lowest lows and highest highs of the lookback period for QFL bases and ceilings
        qfl_base = df['low'].min()  # Support level
        qfl_ceiling = df['high'].max()  # Resistance level

        return qfl_base, qfl_ceiling

    def calculate_qfl_base(self, symbol: str, timeframe='5m', lookback_period=12):
        # Fetch historical candle data
        candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback_period)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamps to readable dates (optional)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Find the lowest lows of the lookback period
        qfl_base = df['low'].min()
        return qfl_base

    # Bybit regular auto hedge logic
    # Bybit entry logic
    def bybit_hedge_entry_maker(self, symbol: str, trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        if trend is not None and isinstance(trend, str):
            if one_minute_volume is not None and five_minute_distance is not None:
                if one_minute_volume > min_vol and five_minute_distance > min_dist:

                    if trend.lower() == "long" and should_long and long_pos_qty == 0:
                        logging.info(f"Placing initial long entry")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        logging.info(f"Placed initial long entry")
                    else:
                        if trend.lower() == "long" and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price:
                            logging.info(f"Placed additional long entry")
                            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if trend.lower() == "short" and should_short and short_pos_qty == 0:
                        logging.info(f"Placing initial short entry")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        logging.info("Placed initial short entry")
                    else:
                        if trend.lower() == "short" and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price:
                            logging.info(f"Placed additional short entry")
                            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_turbocharged_entry_maker_walls(self, symbol, trend, mfi, one_minute_volume, five_minute_distance, min_vol, min_dist, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):
        if one_minute_volume is None or five_minute_distance is None or one_minute_volume <= min_vol or five_minute_distance <= min_dist:
            logging.warning(f"Either 'one_minute_volume' or 'five_minute_distance' does not meet the criteria for symbol {symbol}. Skipping current execution...")
            return
        
        self.order_book_analyzer = self.OrderBookAnalyzer(self.exchange, symbol)
        order_book = self.order_book_analyzer.get_order_book()

        best_ask_price = order_book['asks'][0][0]
        best_bid_price = order_book['bids'][0][0]

        market_data = self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)
        min_qty = float(market_data["min_qty"])

        largest_bid = max(order_book['bids'], key=lambda x: x[1])
        largest_ask = min(order_book['asks'], key=lambda x: x[1])

        spread = best_ask_price - best_bid_price

        # Adjusting the multiplier based on the size of the wall
        bid_wall_size_multiplier = 0.05 + (0.02 if largest_bid[1] > 10 * min_qty else 0)
        ask_wall_size_multiplier = 0.05 + (0.02 if largest_ask[1] > 10 * min_qty else 0)

        front_run_bid_price = round(largest_bid[0] + (spread * bid_wall_size_multiplier), 4)
        front_run_ask_price = round(largest_ask[0] - (spread * ask_wall_size_multiplier), 4)

        # Check for long position and ensure take_profit_long is not None
        if long_pos_qty > 0 and take_profit_long:
            distance_to_tp_long = take_profit_long - best_bid_price
            dynamic_long_amount = distance_to_tp_long * 5
            if trend.lower() == "long" and mfi.lower() == "long" and best_bid_price < long_pos_price:
                self.postonly_limit_order_bybit(symbol, "buy", dynamic_long_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Additional Long Entry Placed at {front_run_bid_price} with {dynamic_long_amount} amount!")

        # Check for short position and ensure take_profit_short is not None
        if short_pos_qty > 0 and take_profit_short:
            distance_to_tp_short = best_ask_price - take_profit_short
            dynamic_short_amount = distance_to_tp_short * 5
            if trend.lower() == "short" and mfi.lower() == "short" and best_ask_price > short_pos_price:
                self.postonly_limit_order_bybit(symbol, "sell", dynamic_short_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Additional Short Entry Placed at {front_run_ask_price} with {dynamic_short_amount} amount!")

        # Entries for when there's no position yet
        if long_pos_qty == 0:
            if trend.lower() == "long" or mfi.lower() == "long":
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        if short_pos_qty == 0:
            if trend.lower() == "short" or mfi.lower() == "short":
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")

    def bybit_turbocharged_additional_entry_maker(self, open_orders, symbol, trend, mfi, one_minute_volume: float, five_minute_distance: float, min_vol, min_dist, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_add_to_long, should_add_to_short):
        if one_minute_volume is None or five_minute_distance is None or one_minute_volume <= min_vol or five_minute_distance <= min_dist:
            logging.warning(f"Either 'one_minute_volume' or 'five_minute_distance' does not meet the criteria for symbol {symbol}. Skipping current execution...")
            return
        
        self.order_book_analyzer = self.OrderBookAnalyzer(self.exchange, symbol)
        order_book = self.order_book_analyzer.get_order_book()

        best_ask_price = order_book['asks'][0][0]
        best_bid_price = order_book['bids'][0][0]

        market_data = self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)
        min_qty = float(market_data["min_qty"])

        largest_bid = max(order_book['bids'], key=lambda x: x[1])
        largest_ask = min(order_book['asks'], key=lambda x: x[1])

        spread = best_ask_price - best_bid_price
        front_run_bid_price = round(largest_bid[0] + (spread * 0.05), 4)
        front_run_ask_price = round(largest_ask[0] - (spread * 0.05), 4)

        if take_profit_long is not None:
            distance_to_tp_long = take_profit_long - best_bid_price
            long_dynamic_amount += distance_to_tp_long * 1
            long_dynamic_amount = max(long_dynamic_amount, min_qty)

        if take_profit_short is not None:
            distance_to_tp_short = best_ask_price - take_profit_short
            short_dynamic_amount += distance_to_tp_short * 1
            short_dynamic_amount = max(short_dynamic_amount, min_qty)

        if long_pos_qty > 0 and take_profit_long:
            if trend.lower() == "long" and mfi.lower() == "long" and (long_pos_price is not None and best_bid_price < long_pos_price) and should_add_to_long and not self.entry_order_exists(open_orders, "buy"):
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Additional Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        if short_pos_qty > 0 and take_profit_short:
            if trend.lower() == "short" and mfi.lower() == "short" and (short_pos_price is not None and best_ask_price > short_pos_price) and should_add_to_short and not self.entry_order_exists(open_orders, "sell"):
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Additional Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")

    def bybit_turbocharged_entry_maker(self, open_orders, symbol, trend, mfi, one_minute_volume: float, five_minute_distance: float, min_vol, min_dist, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_long, should_add_to_long, should_short, should_add_to_short):
        
        if not (one_minute_volume and five_minute_distance) or one_minute_volume <= min_vol or five_minute_distance <= min_dist:
            logging.warning(f"Either 'one_minute_volume' or 'five_minute_distance' does not meet the criteria for symbol {symbol}. Skipping current execution...")
            return

        order_book = self.OrderBookAnalyzer(self.exchange, symbol).get_order_book()
        best_ask_price, best_bid_price = order_book['asks'][0][0], order_book['bids'][0][0]

        spread = best_ask_price - best_bid_price
        front_run_bid_price = round(max(order_book['bids'], key=lambda x: x[1])[0] + spread * 0.05, 4)
        front_run_ask_price = round(min(order_book['asks'], key=lambda x: x[1])[0] - spread * 0.05, 4)

        min_qty = float(self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)["min_qty"])

        long_dynamic_amount += max((take_profit_long - best_bid_price) if take_profit_long else 0, min_qty)
        short_dynamic_amount += max((best_ask_price - take_profit_short) if take_profit_short else 0, min_qty)

        if not trend or not mfi:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        if trend.lower() == "long" and mfi.lower() == "long":
            if long_pos_qty == 0 and should_long and not self.entry_order_exists(open_orders, "buy"):
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Long Entry Placed at {front_run_bid_price} for {symbol} with {long_dynamic_amount} amount!")
            elif should_add_to_long and long_pos_qty > 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Additional Long Entry Placed at {front_run_bid_price} for {symbol} with {long_dynamic_amount} amount!")

        elif trend.lower() == "short" and mfi.lower() == "short":
            if short_pos_qty == 0 and should_short and not self.entry_order_exists(open_orders, "sell"):
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Short Entry Placed at {front_run_ask_price} for {symbol} with {short_dynamic_amount} amount!")
            elif should_add_to_short and short_pos_qty > 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Additional Short Entry Placed at {front_run_ask_price} for {symbol} with {short_dynamic_amount} amount!")
    
    def bybit_turbocharged_new_entry_maker(self, open_orders, symbol, trend, mfi, one_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount):
        if one_minute_volume is None or five_minute_distance is None or one_minute_volume <= min_vol or five_minute_distance <= min_dist:
            logging.warning(f"Either 'one_minute_volume' or 'five_minute_distance' does not meet the criteria for symbol {symbol}. Skipping current execution...")
            return

        order_book = self.OrderBookAnalyzer(self.exchange, symbol).get_order_book()
        best_ask_price, best_bid_price = order_book['asks'][0][0], order_book['bids'][0][0]
        min_qty = float(self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)["min_qty"])

        largest_bid, largest_ask = max(order_book['bids'], key=lambda x: x[1]), min(order_book['asks'], key=lambda x: x[1])
        spread = best_ask_price - best_bid_price
        front_run_bid_price = round(largest_bid[0] + spread * 0.05, 4)  # front-run by 5% of the spread
        front_run_ask_price = round(largest_ask[0] - spread * 0.05, 4)  # front-run by 5% of the spread

        position_data = self.exchange.get_positions_bybit(symbol)
        long_pos_qty, short_pos_qty = position_data["long"]["qty"], position_data["short"]["qty"]

        # Ensure the calculated amounts are not below the minimum order quantity
        long_dynamic_amount = max(long_dynamic_amount, min_qty)
        short_dynamic_amount = max(short_dynamic_amount, min_qty)

        # Entries for when there's no position yet
        if long_pos_qty == 0 and trend.lower() == "long" and mfi.lower() == "long" and not self.entry_order_exists(open_orders, "buy"):
            if long_dynamic_amount <= self.max_long_trade_qty_per_symbol[symbol]:  # Ensure we don't exceed max qty for the symbol
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Long Entry Placed at {front_run_bid_price} for {symbol} with {long_dynamic_amount} amount!")

        if short_pos_qty == 0 and trend.lower() == "short" and mfi.lower() == "short" and not self.entry_order_exists(open_orders, "sell"):
            if short_dynamic_amount <= self.max_short_trade_qty_per_symbol[symbol]:  # Ensure we don't exceed max qty for the symbol
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Short Entry Placed at {front_run_ask_price} for {symbol} with {short_dynamic_amount} amount!")

    def bybit_hedge_entry_maker_v3(self, open_orders: list, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        if one_minute_volume > min_vol and five_minute_distance > min_dist:
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

            # Initial long entry
            if trend.lower() == "long" and mfi.lower() == "long" and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
                logging.info(f"Placing initial long entry for {symbol}")
                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            # Additional long entry
            elif trend.lower() == "long" and mfi.lower() == "long" and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                logging.info(f"Placing additional long entry for {symbol}")
                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            # Initial short entry
            if trend.lower() == "short" and mfi.lower() == "short" and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
                logging.info(f"Placing initial short entry for {symbol}")
                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            # Additional short entry
            elif trend.lower() == "short" and mfi.lower() == "short" and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                logging.info(f"Placing additional short entry for {symbol}")
                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

# long_dynamic_amount = self.m_order_amount(symbol, "long", long_dynamic_amount)
# short_dynamic_amount = self.m_order_amount(symbol, "short", short_dynamic_amount)

    def bybit_hedge_entry_maker_obstrength_random(self, open_orders: list, symbol: str, trend: str, mfi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return


        self.m_order_amount(symbol, "long", long_dynamic_amount)
        self.m_order_amount(symbol, "short", short_dynamic_amount)

        logging.info(f"M order: Long {long_dynamic_amount}")
        logging.info(f"M order: Short {short_dynamic_amount}")

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        # Initial long entry
        if trend.lower() == "long" and mfi.lower() == "long" and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing initial long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        # Additional long entry
        elif trend.lower() == "long" and mfi.lower() == "long" and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing additional long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        # Initial short entry
        if trend.lower() == "short" and mfi.lower() == "short" and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing initial short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

        # Additional short entry
        elif trend.lower() == "short" and mfi.lower() == "short" and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing additional short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_additional_entry_obstrength_random(self, open_orders: list, symbol: str, trend: str, mfi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        self.m_order_amount(symbol, "long", long_dynamic_amount)
        self.m_order_amount(symbol, "short", short_dynamic_amount)

        # Additional long entry
        if trend.lower() == "long" and mfi.lower() == "long" and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing additional long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        # Additional short entry
        if trend.lower() == "short" and mfi.lower() == "short" and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing additional short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_initial_entry_obstrength_random(self, open_orders: list, symbol: str, trend: str, mfi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        self.m_order_amount(symbol, "long", long_dynamic_amount)
        self.m_order_amount(symbol, "short", short_dynamic_amount)

        # Initial long entry
        if trend.lower() == "long" and mfi.lower() == "long" and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing initial long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        # Initial short entry
        if trend.lower() == "short" and mfi.lower() == "short" and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing initial short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_entry_maker_obstrength(self, open_orders: list, symbol: str, trend: str, mfi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        # Initial long entry
        if trend.lower() == "long" and mfi.lower() == "long" and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing initial long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
            time.sleep(1.5)

        # Additional long entry
        elif trend.lower() == "long" and mfi.lower() == "long" and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing additional long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
            time.sleep(1.5)

        # Initial short entry
        if trend.lower() == "short" and mfi.lower() == "short" and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing initial short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
            time.sleep(1.5)

        # Additional short entry
        elif trend.lower() == "short" and mfi.lower() == "short" and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing additional short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
            time.sleep(1.5)

    def bybit_hedge_initial_entry_obstrength(self, open_orders: list, symbol: str, trend: str, mfi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        # Initial long entry
        if trend.lower() == "long" and mfi.lower() == "long" and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing initial long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        # Initial short entry
        if trend.lower() == "short" and mfi.lower() == "short" and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing initial short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_additional_entry_obstrength(self, open_orders: list, symbol: str, trend: str, mfi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        # Additional long entry
        if trend.lower() == "long" and mfi.lower() == "long" and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing additional long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
            time.sleep(1.5)

        # Additional short entry
        if trend.lower() == "short" and mfi.lower() == "short" and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing additional short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
            time.sleep(1.5)

    def bybit_hedge_entry_maker_hma_walls(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, buy_wall: bool, sell_wall: bool):

        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Check for long entry conditions
                if (trend.lower() == "long" or hma_trend.lower() == "long") and (mfi.lower() == "long" or (buy_wall and eri == "bullish")) and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                # Check for additional long entry conditions
                elif (trend.lower() == "long" or hma_trend.lower() == "long") and (mfi.lower() == "long" or (buy_wall and eri == "bullish")) and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Check for short entry conditions
                if (trend.lower() == "short" or hma_trend.lower() == "short") and (mfi.lower() == "short" or (sell_wall and eri == "bearish")) and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry")

                # Check for additional short entry conditions
                elif (trend.lower() == "short" or hma_trend.lower() == "short") and (mfi.lower() == "short" or (sell_wall and eri == "bearish")) and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_entry_maker_hma(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        logging.info(f"Hedge entry maker hma function hit")

        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        logging.info(f"Trend is {trend}")
        logging.info(f"MFI is {mfi}")
        logging.info(f"HMA is {hma_trend}")

        logging.info(f"One min vol for {symbol} is {one_minute_volume}")
        logging.info(f"Five min dist for {symbol} is {five_minute_distance}")

        logging.info(f"Should long for {symbol} : {should_long}")
        logging.info(f"Should short for {symbol} : {should_short}")
        logging.info(f"Should add to long for {symbol} : {should_add_to_long}")
        logging.info(f"Should add to short for {symbol} : {should_add_to_short}")

        logging.info(f"Min dist: {min_dist}")
        logging.info(f"Min vol: {min_vol}")

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                logging.info(f"Made it into the entry maker function for {symbol}")

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Check for long entry conditions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                # Check for additional long entry conditions
                elif ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Check for short entry conditions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry")

                # Check for additional short entry conditions
                elif ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def m_order_amount(self, symbol, side, amount):
        order_book = self.exchange.get_orderbook(symbol)
        top_asks = order_book['asks'][:10]
        top_bids = order_book['bids'][:10]
        placed_orders = []  # Initialize the list to keep track of placed orders

        QS_BUFFER_PERCENTAGE = 0.05  # Use as a percentage
        L_BUFFER_PERCENTAGE = 0.05  # Use as a percentage

        # Place QS orders
        if random.randint(1, 10) > 8:
            for _ in range(5):
                try:
                    price_adjustment = top_bids[0][0] * QS_BUFFER_PERCENTAGE if side == "long" else top_asks[0][0] * QS_BUFFER_PERCENTAGE
                    order_price = top_bids[0][0] * (1 - price_adjustment) if side == "long" else top_asks[0][0] * (1 + price_adjustment)
                    order = self.limit_order_bybit(symbol, "buy" if side == "long" else "sell", amount, order_price, positionIdx=1 if side == "long" else 2, reduceOnly=False)
                    if order is not None:
                        placed_orders.append(order)
                except Exception as e:
                    logging.error(f"Error placing order: {e}")

        # Place L orders
        if random.randint(1, 10) > 7:
            for _ in range(3):
                try:
                    price_adjustment = top_bids[0][0] * L_BUFFER_PERCENTAGE if side == "long" else top_asks[0][0] * L_BUFFER_PERCENTAGE
                    order_price = top_bids[0][0] * (1 - price_adjustment) if side == "long" else top_asks[0][0] * (1 + price_adjustment)
                    order = self.limit_order_bybit(symbol, "buy" if side == "long" else "sell", amount * 1.5, order_price, positionIdx=1 if side == "long" else 2, reduceOnly=False)
                    if order is not None:
                        placed_orders.append(order)
                except Exception as e:
                    logging.error(f"Error placing order: {e}")

        # Cancel orders
        for order in placed_orders:
            if order and 'id' in order:
                self.exchange.cancel_order_by_id(order['id'], symbol)

        return amount

    def play_the_spread_entry_and_tp(self, symbol, open_orders, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):
        analyzer = self.OrderBookAnalyzer(self.exchange, symbol, depth=self.ORDER_BOOK_DEPTH)
        
        imbalance = self.get_order_book_imbalance(symbol)

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        long_dynamic_amount = self.m_order_amount(symbol, "long", long_dynamic_amount)
        short_dynamic_amount = self.m_order_amount(symbol, "short", short_dynamic_amount)

        # Entry Logic
        if imbalance == "buy_wall" and not self.entry_order_exists(open_orders, "buy"):
            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
        elif imbalance == "sell_wall" and not self.entry_order_exists(open_orders, "sell"):
            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

        # Take Profit Logic
        order_book = analyzer.get_order_book()
        top_asks = order_book['asks'][:5]
        top_bids = order_book['bids'][:5]

        # Calculate the average of top asks and bids
        avg_top_asks = sum([ask[0] for ask in top_asks]) / 5
        avg_top_bids = sum([bid[0] for bid in top_bids]) / 5

        # Identify potential resistance (sell walls) and support (buy walls)
        sell_walls = self.identify_walls(order_book, "sell")
        buy_walls = self.identify_walls(order_book, "buy")

        # Calculate the current profit for long and short positions
        long_profit = (avg_top_asks - long_pos_price) * long_pos_qty if long_pos_qty > 0 else 0
        short_profit = (short_pos_price - avg_top_bids) * short_pos_qty if short_pos_qty > 0 else 0

        logging.info(f"Current profit for {symbol} for long: {long_profit}")
        logging.info(f"Current profit for {symbol} for short: {short_profit}")

        # Dynamic TP setting
        PROFIT_THRESHOLD = 0.002  # for instance, 0.2%

        # Calculate the trading fee for long and short positions
        if long_pos_price is not None:
            long_trading_fee = self.calculate_trading_fee(long_pos_qty, long_pos_price)
            logging.info(f"Long trading fee for {symbol} : {long_trading_fee}")
        else:
            long_trading_fee = 0

        if short_pos_price is not None:
            short_trading_fee = self.calculate_trading_fee(short_pos_qty, short_pos_price)
            logging.info(f"Short trading fee for {symbol} : {short_trading_fee}")
        else:
            short_trading_fee = 0
            
        # For long positions
        if long_pos_qty > 0:
            if sell_walls and sell_walls[0] > long_pos_price:  # Check if the detected sell wall is above the long position price
                logging.info(f"Sell wall found for {symbol}")
                # Adjust TP upwards from the sell wall by the calculated fee amount
                long_take_profit = sell_walls[0] * (1 - long_trading_fee)
            elif long_profit > PROFIT_THRESHOLD * long_pos_price and (best_bid_price + 0.0001) > long_pos_price:  # Ensure TP is above the long position price
                long_take_profit = best_bid_price + 0.0001
            else:
                # Adjust TP upwards from the avg top asks by the calculated fee amount
                long_take_profit = max(avg_top_asks * (1 - long_trading_fee), long_pos_price + 0.0001)  # Ensure TP is above the long position price

            self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

        # For short positions
        if short_pos_qty > 0:
            if buy_walls and buy_walls[0] < short_pos_price:  # Check if the detected buy wall is below the short position price
                logging.info(f"Buy wall found for {symbol}")
                # Adjust TP downwards from the buy wall by the calculated fee amount
                short_take_profit = buy_walls[0] * (1 + short_trading_fee)
            elif short_profit > PROFIT_THRESHOLD * short_pos_price and (best_ask_price - 0.0001) < short_pos_price:  # Ensure TP is below the short position price
                short_take_profit = best_ask_price - 0.0001
            else:
                # Adjust TP downwards from the avg top bids by the calculated fee amount
                short_take_profit = min(avg_top_bids * (1 + short_trading_fee), short_pos_price - 0.0001)  # Ensure TP is below the short position price

            self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)

    def initiate_spread_entry(self, symbol, open_orders, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty):
        analyzer = self.OrderBookAnalyzer(self.exchange, symbol, depth=self.ORDER_BOOK_DEPTH)
        
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        long_dynamic_amount = self.m_order_amount(symbol, "long", long_dynamic_amount)
        short_dynamic_amount = self.m_order_amount(symbol, "short", short_dynamic_amount)
        
        imbalance = self.get_order_book_imbalance(symbol)

        # Entry Logic
        if imbalance == "buy_wall" and not self.entry_order_exists(open_orders, "buy") and long_pos_qty <= 0:
            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
        elif imbalance == "sell_wall" and not self.entry_order_exists(open_orders, "sell") and short_pos_qty <= 0:
            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def set_spread_take_profits(self, symbol, open_orders, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):
        analyzer = self.OrderBookAnalyzer(self.exchange, symbol, depth=self.ORDER_BOOK_DEPTH)

        order_book = analyzer.get_order_book()
        top_asks = order_book['asks'][:5]
        top_bids = order_book['bids'][:5]

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        
        # Calculate average of top asks and bids
        avg_top_asks = sum([ask[0] for ask in top_asks]) / 5
        avg_top_bids = sum([bid[0] for bid in top_bids]) / 5

        # Identify potential resistance (sell walls) and support (buy walls)
        sell_walls = self.identify_walls(order_book, "sell")
        buy_walls = self.identify_walls(order_book, "buy")

        # Calculate the current profit for long and short positions
        long_profit = (avg_top_asks - long_pos_price) * long_pos_qty if long_pos_qty > 0 else 0
        short_profit = (short_pos_price - avg_top_bids) * short_pos_qty if short_pos_qty > 0 else 0

        logging.info(f"Current profit for {symbol} for long: {long_profit}")
        logging.info(f"Current profit for {symbol} for short: {short_profit}")

        # Dynamic TP setting
        PROFIT_THRESHOLD = 0.002  # for instance, 0.2%

        # Calculate the trading fee for long and short positions
        if long_pos_price is not None:
            long_trading_fee = self.calculate_trading_fee(long_pos_qty, long_pos_price)
            logging.info(f"Long trading fee for {symbol} : {long_trading_fee}")
        else:
            long_trading_fee = 0

        if short_pos_price is not None:
            short_trading_fee = self.calculate_trading_fee(short_pos_qty, short_pos_price)
            logging.info(f"Short trading fee for {symbol} : {short_trading_fee}")
        else:
            short_trading_fee = 0

        # For long positions
        if long_pos_qty > 0:
            if sell_walls and sell_walls[0] > long_pos_price:  # Check if the detected sell wall is above the long position price
                logging.info(f"Sell wall found for {symbol}")
                # Adjust TP upwards from the sell wall by the calculated fee amount
                long_take_profit = sell_walls[0] * (1 - long_trading_fee)
            elif long_profit > PROFIT_THRESHOLD * long_pos_price and (best_bid_price + 0.0001) > long_pos_price:  # Ensure TP is above the long position price
                long_take_profit = best_bid_price + 0.0001
            else:
                # Adjust TP upwards from the avg top asks by the calculated fee amount
                long_take_profit = max(avg_top_asks * (1 - long_trading_fee), long_pos_price + 0.0001)  # Ensure TP is above the long position price

            self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

        # For short positions
        if short_pos_qty > 0:
            if buy_walls and buy_walls[0] < short_pos_price:  # Check if the detected buy wall is below the short position price
                logging.info(f"Buy wall found for {symbol}")
                # Adjust TP downwards from the buy wall by the calculated fee amount
                short_take_profit = buy_walls[0] * (1 + short_trading_fee)
            elif short_profit > PROFIT_THRESHOLD * short_pos_price and (best_ask_price - 0.0001) < short_pos_price:  # Ensure TP is below the short position price
                short_take_profit = best_ask_price - 0.0001
            else:
                # Adjust TP downwards from the avg top bids by the calculated fee amount
                short_take_profit = min(avg_top_bids * (1 + short_trading_fee), short_pos_price - 0.0001)  # Ensure TP is below the short position price

            self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)

    def bybit_entry_mm_5m_with_qfl_mfi_and_auto_hedge(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, hedge_ratio: float, price_difference_threshold: float):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Entry function with QFL, MFI, and auto-hedging initialized for {symbol}")

            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)
            
            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
            
            min_order_size = 1

            # Auto-hedging logic for long position
            if long_pos_qty > 0:
                price_diff_percentage_long = abs(current_price - long_pos_price) / long_pos_price
                current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0
                if price_diff_percentage_long >= price_difference_threshold and current_hedge_ratio_long < hedge_ratio:
                    additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                    if additional_hedge_needed_long > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)

            # Auto-hedging logic for short position
            if short_pos_qty > 0:
                price_diff_percentage_short = abs(current_price - short_pos_price) / short_pos_price
                current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0
                if price_diff_percentage_short >= price_difference_threshold and current_hedge_ratio_short < hedge_ratio:
                    additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                    if additional_hedge_needed_short > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                if should_long and trend.lower() == "long" and mfi.lower() == "long" and current_price >= qfl_base:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                if should_short and trend.lower() == "short" and mfi.lower() == "short" and current_price <= qfl_ceiling:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_entry_mm_5m_with_qfl_and_mfi(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Entry function with QFL and MFI filter initialized for {symbol}")

            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)
            
            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            logging.info(f"QFL Base for {symbol}: {qfl_base}")
            logging.info(f"QFL Ceiling for {symbol}: {qfl_ceiling}")
            current_price = self.exchange.get_current_price(symbol)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                if should_long and trend.lower() == "long" and mfi.lower() == "long" and current_price >= qfl_base:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                if should_short and trend.lower() == "short" and mfi.lower() == "short" and current_price <= qfl_ceiling:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)
                
            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_initial_entry_with_qfl_and_mfi_eri(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Initial entry function with QFL, MFI, and ERI filter initialized for {symbol}")

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Long entry condition with ERI trend consideration
                if should_long and trend.lower() == "long" and mfi.lower() == "long" and eri_trend.lower() == "bullish" and current_price >= qfl_base:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Short entry condition with ERI trend consideration
                if should_short and trend.lower() == "short" and mfi.lower() == "short" and eri_trend.lower() == "bearish" and current_price <= qfl_ceiling:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_entry_mm_5m_with_qfl_mfi_and_auto_hedge_with_eri(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, hedge_ratio: float, price_difference_threshold: float):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Entry function with QFL, MFI, and auto-hedging initialized for {symbol}")

            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)
            
            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
            
            min_order_size = 1

            # Auto-hedging logic for long position
            if long_pos_qty > 0:
                price_diff_percentage_long = abs(current_price - long_pos_price) / long_pos_price
                current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0
                if price_diff_percentage_long >= price_difference_threshold and current_hedge_ratio_long < hedge_ratio:
                    additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                    if additional_hedge_needed_long > min_order_size:
                        self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)

            # Auto-hedging logic for short position
            if short_pos_qty > 0:
                price_diff_percentage_short = abs(current_price - short_pos_price) / short_pos_price
                current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0
                if price_diff_percentage_short >= price_difference_threshold and current_hedge_ratio_short < hedge_ratio:
                    additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                    if additional_hedge_needed_short > min_order_size:
                        self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)

            if five_minute_volume > min_vol: #and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Long Entry
                if (should_long or should_add_to_long) and current_price >= qfl_base:
                    trend_aligned_long = (eri_trend == "bullish" or trend.lower() == "long")
                    mfi_signal_long = mfi.lower() == "long"

                    if trend_aligned_long and mfi_signal_long:
                        if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing initial long entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing additional long entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                # Short Entry
                if (should_short or should_add_to_short) and current_price <= qfl_ceiling:
                    trend_aligned_short = (eri_trend == "bearish" or trend.lower() == "short")
                    mfi_signal_short = mfi.lower() == "short"

                    if trend_aligned_short and mfi_signal_short:
                        if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing initial short entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing additional short entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_initial_entry_with_qfl_mfi_and_eri(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Initial entry function with QFL, MFI, and ERI trend initialized for {symbol}")

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Long Entry Logic
                if should_long and long_pos_qty == 0:
                    # Check if trend aligned for long (either eri trend or trend is bullish/long)
                    trend_aligned_long = eri_trend.lower() == "bullish" or trend.lower() == "long"
                    # Check if MFI signal is for long
                    mfi_signal_long = mfi.lower() == "long"
                    # Execute long entry if aligned with QFL base
                    if trend_aligned_long and mfi_signal_long and current_price >= qfl_base:
                        # Check if there is no existing buy order
                        if not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing initial long entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Short Entry Logic
                if should_short and short_pos_qty == 0:
                    # Check if trend aligned for short (either eri trend or trend is bearish/short)
                    trend_aligned_short = eri_trend.lower() == "bearish" or trend.lower() == "short"
                    # Check if MFI signal is for short
                    mfi_signal_short = mfi.lower() == "short"
                    # Execute short entry if aligned with QFL ceiling
                    if trend_aligned_short and mfi_signal_short and current_price <= qfl_ceiling:
                        # Check if there is no existing sell order
                        if not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing initial short entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_additional_entry_with_qfl_mfi_and_eri(self, open_orders: list, symbol: str, trend: str, mfi: str, eri_trend: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_add_to_long: bool, should_add_to_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Additional entry function with QFL, MFI, and ERI trend initialized for {symbol}")

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                bid_walls, ask_walls = self.detect_order_book_walls(symbol)
                largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
                largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)
                
                # Additional Long Entry Logic
                if should_add_to_long and long_pos_qty > 0:
                    trend_aligned_long = eri_trend.lower() == "bullish" or trend.lower() == "long"
                    mfi_signal_long = mfi.lower() == "long"
                    if trend_aligned_long and mfi_signal_long and current_price >= qfl_base:
                        if not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing additional long entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                # Additional Short Entry Logic
                if should_add_to_short and short_pos_qty > 0:
                    trend_aligned_short = eri_trend.lower() == "bearish" or trend.lower() == "short"
                    mfi_signal_short = mfi.lower() == "short"
                    if trend_aligned_short and mfi_signal_short and current_price <= qfl_ceiling:
                        if not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing additional short entry for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping additional entry.")

            time.sleep(5)

    def bybit_entry_mm_5m_with_qfl_mfi_and_auto_hedge_eri(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, hedge_ratio: float, price_difference_threshold: float):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Entry function with QFL, MFI, ERI, and auto-hedging initialized for {symbol}")

            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)
            
            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
            
            min_order_size = 1

            # Auto-hedging logic for long position
            if long_pos_qty > 0:
                price_diff_percentage_long = abs(current_price - long_pos_price) / long_pos_price
                current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0
                if price_diff_percentage_long >= price_difference_threshold and current_hedge_ratio_long < hedge_ratio:
                    additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                    if additional_hedge_needed_long > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)

            # Auto-hedging logic for short position
            if short_pos_qty > 0:
                price_diff_percentage_short = abs(current_price - short_pos_price) / short_pos_price
                current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0
                if price_diff_percentage_short >= price_difference_threshold and current_hedge_ratio_short < hedge_ratio:
                    additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                    if additional_hedge_needed_short > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                if should_long and trend.lower() == "long" and mfi.lower() == "long" and eri.lower() != "short" and current_price >= qfl_base:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                if should_short and trend.lower() == "short" and mfi.lower() == "short" and eri.lower() != "long" and current_price <= qfl_ceiling:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_entry_mm_5m_with_wall_detection(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"5m Hedge entry function initialized for {symbol}")

            if trend is None or mfi is None or hma_trend is None:
                logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
                return

            logging.info(f"Trend is {trend}")
            logging.info(f"MFI is {mfi}")
            logging.info(f"HMA is {hma_trend}")

            logging.info(f"Five min vol for {symbol} is {five_minute_volume}")
            logging.info(f"Five min dist for {symbol} is {five_minute_distance}")

            logging.info(f"Should long for {symbol}: {should_long}")
            logging.info(f"Should short for {symbol}: {should_short}")
            logging.info(f"Should add to long for {symbol}: {should_add_to_long}")
            logging.info(f"Should add to short for {symbol}: {should_add_to_short}")

            logging.info(f"Min dist: {min_dist}")
            logging.info(f"Min vol: {min_vol}")

            if five_minute_volume is None or five_minute_distance is None:
                logging.warning("Five minute volume or distance is None. Skipping current execution...")
                return

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                logging.info(f"Made it into the entry maker function for {symbol}")

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Detect order book walls
                bid_walls, ask_walls = self.detect_order_book_walls(symbol)

                # Select the largest walls (by size)
                largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
                largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

                if largest_bid_wall:
                    logging.info(f"Detected largest buy wall at {largest_bid_wall} for {symbol}")
                if largest_ask_wall:
                    logging.info(f"Detected largest sell wall at {largest_ask_wall} for {symbol}")

                # Trading logic for long positions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol.get(symbol, 0) and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Additional trading logic for short positions based on order book walls
                if short_pos_qty < self.max_short_trade_qty_per_symbol.get(symbol, 0) and largest_ask_wall and trend.lower() == "long" and mfi.lower() == "long" and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short trade due to detected sell wall and trend {trend} for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", long_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Trading logic for short positions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol.get(symbol, 0) and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Additional trading logic for long positions based on order book walls
                if long_pos_qty < self.max_long_trade_qty_per_symbol.get(symbol, 0) and largest_bid_wall and trend.lower() == "short" and mfi.lower() == "short" and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long trade due to detected bid wall and trend {trend} for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", short_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                time.sleep(5)

    def bybit_entry_mm_5m_initial_only(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"5m Hedge initial entry function initialized for {symbol}")

            if trend is None or mfi is None or hma_trend is None:
                logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
                return

            logging.info(f"Trend is {trend}")
            logging.info(f"MFI is {mfi}")
            logging.info(f"HMA is {hma_trend}")

            logging.info(f"Five min vol for {symbol} is {five_minute_volume}")
            logging.info(f"Five min dist for {symbol} is {five_minute_distance}")

            logging.info(f"Should long for {symbol}: {should_long}")
            logging.info(f"Should short for {symbol}: {should_short}")

            logging.info(f"Min dist: {min_dist}")
            logging.info(f"Min vol: {min_vol}")

            if five_minute_volume is None or five_minute_distance is None:
                logging.warning("Five minute volume or distance is None. Skipping current execution...")
                return

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                logging.info(f"Made it into the initial entry maker function for {symbol}")

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Detect order book walls
                bid_walls, ask_walls = self.detect_order_book_walls(symbol)
                if bid_walls:
                    logging.info(f"Detected buy walls at {bid_walls} for {symbol}")
                if ask_walls:
                    logging.info(f"Detected sell walls at {ask_walls} for {symbol}")

                # Initial trading logic for long positions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Additional trading logic for long positions based on order book walls
                if ask_walls and trend.lower() == "long":
                    logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Initial trading logic for short positions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Additional trading logic for short positions based on order book walls
                if bid_walls and trend.lower() == "short":
                    logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                time.sleep(5)

    def bybit_entry_mm_1m_with_wall_detection(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"5m Hedge entry function initialized for {symbol}")

            if trend is None or mfi is None or hma_trend is None:
                logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
                return

            logging.info(f"Trend is {trend}")
            logging.info(f"MFI is {mfi}")
            logging.info(f"HMA is {hma_trend}")

            logging.info(f"Five min vol for {symbol} is {one_minute_volume}")
            logging.info(f"Five min dist for {symbol} is {five_minute_distance}")

            logging.info(f"Should long for {symbol}: {should_long}")
            logging.info(f"Should short for {symbol}: {should_short}")
            logging.info(f"Should add to long for {symbol}: {should_add_to_long}")
            logging.info(f"Should add to short for {symbol}: {should_add_to_short}")

            logging.info(f"Min dist: {min_dist}")
            logging.info(f"Min vol: {min_vol}")

            if one_minute_volume is None or five_minute_distance is None:
                logging.warning("Five minute volume or distance is None. Skipping current execution...")
                return

            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                logging.info(f"Made it into the entry maker function for {symbol}")

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Detect order book walls
                bid_walls, ask_walls = self.detect_order_book_walls(symbol)

                # Select the largest walls (by size)
                largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
                largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

                if largest_bid_wall:
                    logging.info(f"Detected largest buy wall at {largest_bid_wall} for {symbol}")
                if largest_ask_wall:
                    logging.info(f"Detected largest sell wall at {largest_ask_wall} for {symbol}")

                # Trading logic for long positions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol.get(symbol, 0) and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Additional trading logic for short positions based on order book walls
                if short_pos_qty < self.max_short_trade_qty_per_symbol.get(symbol, 0) and largest_ask_wall and trend.lower() == "long" and mfi.lower() == "long" and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short trade due to detected sell wall and trend {trend} for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", long_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Trading logic for short positions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol.get(symbol, 0) and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Additional trading logic for long positions based on order book walls
                if long_pos_qty < self.max_long_trade_qty_per_symbol.get(symbol, 0) and largest_bid_wall and trend.lower() == "short" and mfi.lower() == "short" and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long trade due to detected bid wall and trend {trend} for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", short_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                time.sleep(5)

    def bybit_entry_mm_1m_initial_only(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"5m Hedge initial entry function initialized for {symbol}")

            if trend is None or mfi is None or hma_trend is None:
                logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
                return

            logging.info(f"Trend is {trend}")
            logging.info(f"MFI is {mfi}")
            logging.info(f"HMA is {hma_trend}")

            logging.info(f"Five min vol for {symbol} is {one_minute_volume}")
            logging.info(f"Five min dist for {symbol} is {five_minute_distance}")

            logging.info(f"Should long for {symbol}: {should_long}")
            logging.info(f"Should short for {symbol}: {should_short}")

            logging.info(f"Min dist: {min_dist}")
            logging.info(f"Min vol: {min_vol}")

            if one_minute_volume is None or five_minute_distance is None:
                logging.warning("Five minute volume or distance is None. Skipping current execution...")
                return

            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                logging.info(f"Made it into the initial entry maker function for {symbol}")

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Detect order book walls
                bid_walls, ask_walls = self.detect_order_book_walls(symbol)
                if bid_walls:
                    logging.info(f"Detected buy walls at {bid_walls} for {symbol}")
                if ask_walls:
                    logging.info(f"Detected sell walls at {ask_walls} for {symbol}")

                # Initial trading logic for long positions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Additional trading logic for long positions based on order book walls
                if ask_walls and trend.lower() == "long":
                    logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Initial trading logic for short positions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Additional trading logic for short positions based on order book walls
                if bid_walls and trend.lower() == "short":
                    logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                time.sleep(5)

    def bybit_initial_entry_mm_5m(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        if five_minute_volume is not None and five_minute_distance is not None:
            if five_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                max_long_trade_qty_for_symbol = self.max_long_trade_qty_per_symbol.get(symbol, 0)  # Get value for symbol or default to 0
                max_short_trade_qty_for_symbol = self.max_short_trade_qty_per_symbol.get(symbol, 0)  # Get value for symbol or default to 0

                # Check for long entry conditions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and long_pos_qty < max_long_trade_qty_for_symbol and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry for {symbol}")

                # Check for short entry conditions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and short_pos_qty < max_short_trade_qty_for_symbol and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry for {symbol}")

    def bybit_qs_entry_exit_eri(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, hedge_ratio: float, price_difference_threshold: float):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Entry function with QFL, MFI, and auto-hedging initialized for {symbol}")

            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)
            
            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
            
            min_order_size = 1

            # Auto-hedging logic for long position
            if long_pos_qty > 0:
                price_diff_percentage_long = abs(current_price - long_pos_price) / long_pos_price
                current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0
                if price_diff_percentage_long >= price_difference_threshold and current_hedge_ratio_long < hedge_ratio:
                    additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                    if additional_hedge_needed_long > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)

            # Auto-hedging logic for short position
            if short_pos_qty > 0:
                price_diff_percentage_short = abs(current_price - short_pos_price) / short_pos_price
                current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0
                if price_diff_percentage_short >= price_difference_threshold and current_hedge_ratio_short < hedge_ratio:
                    additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                    if additional_hedge_needed_short > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Long entry condition with ERI trend consideration
                if should_long and trend.lower() == "long" and mfi.lower() == "long" and eri_trend.lower() == "bullish" and current_price >= qfl_base:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        self.m_order_amount(symbol, "long", long_dynamic_amount)
                        time.sleep(5)
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                # Short entry condition with ERI trend consideration
                if should_short and trend.lower() == "short" and mfi.lower() == "short" and eri_trend.lower() == "bearish" and current_price <= qfl_ceiling:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        self.m_order_amount(symbol, "short", short_dynamic_amount)
                        time.sleep(5)
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)


    def bybit_qs_entry_exit(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, hedge_ratio: float, price_difference_threshold: float):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Entry function with QFL, MFI, and auto-hedging initialized for {symbol}")

            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)
            
            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
            
            min_order_size = 1

            # Auto-hedging logic for long position
            if long_pos_qty > 0:
                price_diff_percentage_long = abs(current_price - long_pos_price) / long_pos_price
                current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0
                if price_diff_percentage_long >= price_difference_threshold and current_hedge_ratio_long < hedge_ratio:
                    additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                    if additional_hedge_needed_long > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)

            # Auto-hedging logic for short position
            if short_pos_qty > 0:
                price_diff_percentage_short = abs(current_price - short_pos_price) / short_pos_price
                current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0
                if price_diff_percentage_short >= price_difference_threshold and current_hedge_ratio_short < hedge_ratio:
                    additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                    if additional_hedge_needed_short > min_order_size:  # Check if additional hedge is needed
                        self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)

            if five_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                if should_long and trend.lower() == "long" and mfi.lower() == "long" and current_price >= qfl_base:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        self.m_order_amount(symbol, "long", long_dynamic_amount)
                        time.sleep(5)
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                    if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                if should_short and trend.lower() == "short" and mfi.lower() == "short" and current_price <= qfl_ceiling:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        self.m_order_amount(symbol, "short", short_dynamic_amount)
                        time.sleep(5)
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                    if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_additional_entries_mm_5m(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

        logging.info(f"Additional entry function hit for {symbol}")

        # Checking for required conditions
        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        if five_minute_volume is None or five_minute_distance is None:
            logging.warning(f"Either 'five_minute_volume' or 'five_minute_distance' is None for symbol {symbol}. Skipping current execution...")
            return

        if five_minute_volume <= min_vol or five_minute_distance <= min_dist:
            logging.info(f"Volume or distance below the threshold for {symbol}. Skipping current execution...")
            return

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        # Cancel existing additional long entries
        existing_additional_longs = self.get_open_additional_entry_orders(symbol, open_orders, "buy")
        for _, existing_long_id in existing_additional_longs:
            self.exchange.cancel_order_by_id(existing_long_id, symbol)
            logging.info(f"Additional long entry {existing_long_id} canceled")
            time.sleep(0.05)

        # Check for additional long entry conditions
        if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol.get(symbol, 0) and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing additional long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
            time.sleep(1.5)

        # Cancel existing additional short entries
        existing_additional_shorts = self.get_open_additional_entry_orders(symbol, open_orders, "sell")
        for _, existing_short_id in existing_additional_shorts:
            self.exchange.cancel_order_by_id(existing_short_id, symbol)
            logging.info(f"Additional short entry {existing_short_id} canceled")
            time.sleep(0.05)

        # Check for additional short entry conditions
        if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol.get(symbol, 0) and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing additional short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
            time.sleep(1.5)

    def bybit_additional_entry_mm_5m(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, 
                                            five_minute_volume: float, five_minute_distance: float, min_vol: float, 
                                            min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, 
                                            long_pos_qty: float, short_pos_qty: float, long_pos_price: float, 
                                            short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

        if None in [trend, mfi, hma_trend]:
            return

        if not (five_minute_volume and five_minute_distance):
            return

        if five_minute_volume <= min_vol or five_minute_distance <= min_dist:
            return

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

        max_long_trade_qty_for_symbol = self.max_long_trade_qty_per_symbol.get(symbol, 0)
        max_short_trade_qty_for_symbol = self.max_short_trade_qty_per_symbol.get(symbol, 0)  # Get value for symbol or default to 0


        # Check for additional long entry conditions
        if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_add_to_long:
            if long_pos_qty < max_long_trade_qty_for_symbol and best_bid_price < long_pos_price:
                if not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placing additional short for {symbol}")
                    time.sleep(5)
        # Check for additional short entry conditions
        if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short:
            if short_pos_qty < max_short_trade_qty_for_symbol and best_ask_price > short_pos_price:
                if not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placing additional long for {symbol}")
                    time.sleep(5)

    def bybit_hedge_initial_entry_maker_hma(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Check for long entry conditions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                # Check for short entry conditions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry")

    def bybit_hedge_additional_entry_maker_hma(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        logging.info(f"Checking volume condition in manage positions")

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                logging.info(f"Made it past volume check in manage positions")

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Additional Long Entry Condition Checks
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_add_to_long:
                    if symbol in self.max_long_trade_qty_per_symbol and long_pos_qty >= self.max_long_trade_qty_per_symbol[symbol]:
                        logging.warning(f"Reached or exceeded max long trade qty for symbol: {symbol}. Current qty: {long_pos_qty}, Max allowed qty: {self.max_long_trade_qty_per_symbol[symbol]}. Skipping additional long entry.")
                    elif best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Additional Short Entry Condition Checks
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short:
                    if symbol in self.max_short_trade_qty_per_symbol and short_pos_qty >= self.max_short_trade_qty_per_symbol[symbol]:
                        logging.warning(f"Reached or exceeded max short trade qty for symbol: {symbol}. Current qty: {short_pos_qty}, Max allowed qty: {self.max_short_trade_qty_per_symbol[symbol]}. Skipping additional short entry.")
                    elif best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_entry_maker_v3_ratio(self, open_orders: list, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):
            
        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                position_data = self.exchange.get_positions_bybit(symbol)

                short_liq_price = position_data["short"]["liq_price"]
                long_liq_price = position_data["long"]["liq_price"]

                long_liq_pct = self.config.long_liq_pct
                short_liq_pct = self.config.short_liq_pct

                short_liq_buffer = short_liq_price * short_liq_pct
                long_liq_buffer = long_liq_price * long_liq_pct

                HEDGE_RATIO = 1  # 1:1 hedge for this example
                COOLDOWN_PERIOD = 10  # 10 seconds for this example
                last_hedge_time = 0

                if time.time() - last_hedge_time > COOLDOWN_PERIOD:
                    logging.info("Hedge cooldown period complete")
                    if short_pos_qty > 0:
                        logging.info(f"Auto hedge best ask price {best_ask_price} > short liq price {short_liq_price}")
                        if best_ask_price > short_liq_price - short_liq_buffer:
                            if short_liq_buffer == 0:
                                logging.warning(f"Short liquidation buffer is zero. Skipping hedge for {symbol}.")
                            else:
                                hedge_amount = short_pos_qty * HEDGE_RATIO * ((short_liq_price - best_ask_price) / short_liq_buffer)
                                logging.info(f"Close to short liquidation and high volume. Hedging by buying {hedge_amount}")
                                self.postonly_limit_order_bybit(symbol, "buy", hedge_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                last_hedge_time = time.time()
                        else:
                            logging.info("Auto hedge enabled but liq buffer not met")

                    if long_pos_qty > 0:
                        logging.info("Auto hedge found long position")
                        if best_bid_price < long_liq_price + long_liq_buffer:
                            logging.info(f"Auto hedge best bid price {best_bid_price} > long_liq_price {long_liq_price}")
                            if long_liq_buffer == 0:
                                logging.warning(f"Long liquidation buffer is zero. Skipping hedge for {symbol}.")
                            else:
                                hedge_amount = long_pos_qty * HEDGE_RATIO * ((best_bid_price - long_liq_price) / long_liq_buffer)
                                logging.info(f"Close to long liquidation and high volume. Hedging by selling {hedge_amount}")
                                self.postonly_limit_order_bybit(symbol, "sell", hedge_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                last_hedge_time = time.time()
                        else:
                            logging.info("Auto hedge enabled but liq buffer not met")

                if trend.lower() != mfi.lower():  # This checks if there's a trend reversal
                    logging.info(f"Trend and MFI mismatch detected. Consider reducing positions or hedging.")

                if (trend.lower() == "long" and mfi.lower() == "long"):
                    
                    # Check for initial long entry conditions
                    if should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        logging.info(f"Placed initial long entry")

                    # Check for additional long entry conditions
                    elif should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                if (trend.lower() == "short" and mfi.lower() == "short"):
                    
                    # Check for initial short entry conditions
                    if should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        logging.info(f"Placed initial short entry")

                    # Check for additional short entry conditions
                    elif should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_additional_entry_maker_v3_ratio(self, open_orders: list, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                position_data = self.exchange.get_positions_bybit(symbol)

                short_liq_price = position_data["short"]["liq_price"]
                long_liq_price = position_data["long"]["liq_price"]

                long_liq_pct = self.config.long_liq_pct
                short_liq_pct = self.config.short_liq_pct

                short_liq_buffer = short_liq_price * short_liq_pct
                long_liq_buffer = long_liq_price * long_liq_pct

                HEDGE_RATIO = 1  # 1:1 hedge for this example
                COOLDOWN_PERIOD = 10  # 10 seconds for this example
                last_hedge_time = 0

                if time.time() - last_hedge_time > COOLDOWN_PERIOD:
                    logging.info("Hedge cooldown period complete")
                    if short_pos_qty > 0:
                        logging.info(f"Auto hedge best ask price {best_ask_price} > short liq price {short_liq_price}")
                        if best_ask_price > short_liq_price - short_liq_buffer:
                            if short_liq_buffer == 0:
                                logging.warning(f"Short liquidation buffer is zero. Skipping hedge for {symbol}.")
                            else:
                                hedge_amount = short_pos_qty * HEDGE_RATIO * ((short_liq_price - best_ask_price) / short_liq_buffer)
                                logging.info(f"Close to short liquidation and high volume. Hedging by buying {hedge_amount}")
                                self.postonly_limit_order_bybit(symbol, "buy", hedge_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                last_hedge_time = time.time()
                        else:
                            logging.info("Auto hedge enabled but liq buffer not met")

                    if long_pos_qty > 0:
                        logging.info("Auto hedge found long position")
                        if best_bid_price < long_liq_price + long_liq_buffer:
                            logging.info(f"Auto hedge best bid price {best_bid_price} > long_liq_price {long_liq_price}")
                            if long_liq_buffer == 0:
                                logging.warning(f"Long liquidation buffer is zero. Skipping hedge for {symbol}.")
                            else:
                                hedge_amount = long_pos_qty * HEDGE_RATIO * ((best_bid_price - long_liq_price) / long_liq_buffer)
                                logging.info(f"Close to long liquidation and high volume. Hedging by selling {hedge_amount}")
                                self.postonly_limit_order_bybit(symbol, "sell", hedge_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                last_hedge_time = time.time()
                        else:
                            logging.info("Auto hedge enabled but liq buffer not met")

                # Place additional long entry
                if (trend.lower() == "long" and mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Place additional short entry
                elif (trend.lower() == "short" and mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    # Revised consistent maker strategy using MA Trend OR MFI as well while maintaining same original MA logic
    def bybit_hedge_entry_maker_v2(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                if (trend.lower() == "long" or mfi.lower() == "long") and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                elif (trend.lower() == "long" or mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                if (trend.lower() == "short" or mfi.lower() == "short") and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info("Placed initial short entry")

                elif (trend.lower() == "short" or mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    # Revised for ERI
    def bybit_hedge_entry_maker_eritrend(self, symbol: str, trend: str, eri: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                if (trend.lower() == "long" or eri.lower() == "short") and should_long and long_pos_qty == 0:
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")
                else:
                    if (trend.lower() == "long" or eri.lower() == "short") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price:
                        logging.info(f"Placing additional long entry")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                if (trend.lower() == "short" or eri.lower() == "long") and should_short and short_pos_qty == 0:
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info("Placed initial short entry")
                else:
                    if (trend.lower() == "short" or eri.lower() == "long") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price:
                        logging.info(f"Placing additional short entry")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def update_take_profit_if_profitable_for_one_minute(self, symbol):
        try:
            # Fetch position data and open position data
            position_data = self.exchange.get_positions_bybit(symbol)
            open_position_data = self.exchange.get_all_open_positions_bybit()
            current_time = datetime.utcnow()
            
            # Fetch current order book prices
            best_ask_price = float(self.exchange.get_orderbook(symbol)['asks'][0][0])
            best_bid_price = float(self.exchange.get_orderbook(symbol)['bids'][0][0])

            # Initialize next_tp_update using your calculate_next_update_time function
            next_tp_update = self.calculate_next_update_time()
            
            # Loop through all open positions to find the one for the given symbol
            for position in open_position_data:
                if position['symbol'].split(':')[0] == symbol:
                    timestamp = datetime.utcfromtimestamp(position['timestamp'] / 1000.0)  # Convert to seconds from milliseconds
                    time_in_position = current_time - timestamp
                    
                    # Check if the position has been open for more than a minute
                    if time_in_position > timedelta(minutes=1):
                        side = position['side']
                        pos_qty = position['contracts']
                        entry_price = position['entryPrice']
                        
                        # Check if the position is profitable
                        is_profitable = (best_ask_price > entry_price) if side == 'long' else (best_bid_price < entry_price)
                        
                        if is_profitable:
                            # Calculate take profit price based on the current price
                            take_profit_price = best_ask_price if side == 'long' else best_bid_price
                            
                            # Fetch open orders for the symbol
                            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                            
                            # Update the take profit
                            positionIdx = 1 if side == 'long' else 2
                            order_side = 'Buy' if side == 'short' else 'Sell'
                            
                            next_tp_update = self.update_take_profit_spread_bybit(
                                symbol, pos_qty, take_profit_price, positionIdx, order_side, 
                                open_orders, next_tp_update
                            )
                            logging.info(f"Updated take profit for {side} position on {symbol} to {take_profit_price}")

        except Exception as e:
            logging.error(f"Error in updating take profit: {e}")

    # Aggressive TP spread update
    def update_aggressive_take_profit_bybit(self, symbol, pos_qty, current_price, positionIdx, order_side, open_orders, next_tp_update, entry_time):
        existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
        total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
        logging.info(f"Existing {order_side} TPs: {existing_tps}")

        now = datetime.now()
        time_since_entry = now - entry_time

        # Aggressively set the take-profit price close to the current market price
        aggressive_take_profit_price = current_price * 1.01 if order_side == 'buy' else current_price * 0.99

        if now >= next_tp_update or not math.isclose(total_existing_tp_qty, pos_qty) or time_since_entry > timedelta(minutes=5):  # 5-minute check
            try:
                for qty, existing_tp_id in existing_tps:
                    self.exchange.cancel_order_by_id(existing_tp_id, symbol)
                    logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                    time.sleep(0.05)
                
                # Create multiple take-profit orders at different levels
                for i in range(1, 4):  # Creating 3 take-profit levels
                    partial_qty = pos_qty // 3
                    partial_tp_price = aggressive_take_profit_price * (1 + 0.005 * i) if order_side == 'buy' else aggressive_take_profit_price * (1 - 0.005 * i)
                    self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, partial_qty, partial_tp_price, positionIdx=positionIdx, reduce_only=True)
                    logging.info(f"{order_side.capitalize()} take profit set at {partial_tp_price} with qty {partial_qty}")

                next_tp_update = self.calculate_next_update_time()  # Calculate the next update time after placing the order
            except Exception as e:
                logging.info(f"Error in updating {order_side} TP: {e}")
                
        return next_tp_update

    # Bybit update take profit based on spread
    def update_take_profit_spread_bybit(self, symbol, pos_qty, short_take_profit, long_take_profit, short_pos_price, long_pos_price, positionIdx, order_side, next_tp_update, five_minute_distance, previous_five_minute_distance, max_retries=10):
        # Fetch the current open TP orders for the symbol
        long_tp_orders, short_tp_orders = self.exchange.bybit.get_open_tp_orders(symbol)

        # # Calculate the TP values based on the spread
        # short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)
        # #short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(None, None, symbol, five_minute_distance, previous_five_minute_distance, None, None)

        logging.info(f"From update_take_profit_spread : Calculated short TP for {symbol}: {short_take_profit}")
        logging.info(f"From update_take_profit_spread : Calculated long TP for {symbol}: {long_take_profit}")

        # Determine the take profit price based on the order side
        take_profit_price = long_take_profit if order_side == "sell" else short_take_profit
        logging.info(f"Determined TP price for {symbol} {order_side}: {take_profit_price}") 

        # Determine the relevant TP orders and quantities based on the order side
        relevant_tp_orders = long_tp_orders if order_side == "sell" else short_tp_orders

        # Check if there's an existing TP order with a mismatched quantity
        mismatched_qty_orders = [order for order in relevant_tp_orders if order['qty'] != pos_qty]

        # If mismatched TP orders exist, cancel them
        if mismatched_qty_orders:
            for order in mismatched_qty_orders:
                try:
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                    logging.info(f"{order_side.capitalize()} take profit {order['id']} canceled due to mismatched quantity.")
                    time.sleep(0.05)
                except Exception as e:
                    logging.error(f"Error in cancelling {order_side} TP order {order['id']}. Error: {e}")

        # If there's no TP order or there was a mismatched TP order, create a new TP order
        if not relevant_tp_orders or mismatched_qty_orders:
            now = datetime.now()
            if now >= next_tp_update:
                try:
                    logging.info(f"Next TP updating for {symbol} : {next_tp_update}")

                    retries = 0
                    success = False
                    while retries < max_retries and not success:
                        try:
                            self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, take_profit_price, positionIdx=positionIdx, reduce_only=True)
                            logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price}")
                            success = True
                        except Exception as e:
                            logging.error(f"Failed to set {order_side} TP for {symbol}. Retry {retries + 1}/{max_retries}. Error: {e}")
                            retries += 1
                            time.sleep(1)  # Wait for a moment before retrying

                    next_tp_update = self.calculate_next_update_time()  # Calculate the next update time after placing the order
                except Exception as e:
                    logging.error(f"Error in updating {order_side} TP: {e}")

        else:
            logging.info(f"Take profit already exists for {symbol} {order_side} with correct quantity. Skipping update.")

        return next_tp_update

    def bybit_hedge_placetp_maker(self, symbol, pos_qty, take_profit_price, positionIdx, order_side, open_orders):
        logging.info(f"TP maker function Trying to place TP for {symbol}")
        existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
        logging.info(f"Existing TP from TP maker functions: {existing_tps}")
        total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
        logging.info(f"TP maker function Existing {order_side} TPs: {existing_tps}")
        if not math.isclose(total_existing_tp_qty, pos_qty):
            try:
                for qty, existing_tp_id in existing_tps:
                    if not math.isclose(qty, pos_qty):
                        self.exchange.cancel_order_by_id(existing_tp_id, symbol)
                        logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                        time.sleep(0.05)
            except Exception as e:
                logging.info(f"Error in cancelling {order_side} TP orders {e}")

        if len(existing_tps) < 1:
            try:
                # Use postonly_limit_order_bybit function to place take profit order
                self.postonly_limit_order_bybit_nolimit(symbol, order_side, pos_qty, take_profit_price, positionIdx, reduceOnly=True)
                logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price}")
                time.sleep(0.05)
            except Exception as e:
                logging.info(f"Error in placing {order_side} TP: {e}")

    def long_entry_maker(self, symbol: str, trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, long_pos_qty: float, long_pos_price: float, should_long: bool, should_add_to_long: bool):
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        
        if trend is not None and isinstance(trend, str) and trend.lower() == "long":
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                if should_long and long_pos_qty == 0:
                    logging.info(f"Placing initial long entry for {symbol}")
                    #postonly_limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                elif should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price:
                    logging.info(f"Placing additional long entry for {symbol}")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

    def short_entry_maker(self, symbol: str, trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, short_dynamic_amount: float, short_pos_qty: float, short_pos_price: float, should_short: bool, should_add_to_short: bool):
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        
        if trend is not None and isinstance(trend, str) and trend.lower() == "short":
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                if should_short and short_pos_qty == 0:
                    logging.info(f"Placing initial short entry for {symbol}")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                elif should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price:
                    logging.info(f"Placing additional short entry for {symbol}")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    # Bybit cancel all entries
    def cancel_entries_bybit(self, symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high):
        # Cancel entries
        current_time = time.time()
        if current_time - self.last_cancel_time >= 60: #60 # Execute this block every 1 minute
            try:
                if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                    self.exchange.cancel_all_entries_bybit(symbol)
                    logging.info(f"Canceled entry orders for {symbol}")
                    time.sleep(0.05)
            except Exception as e:
                logging.info(f"An error occurred while canceling entry orders: {e}")

            self.last_cancel_time = current_time

    def clear_stale_positions(self, rotator_symbols, max_time_without_volume=3600): # default time is 1 hour
        open_positions = self.exchange.get_open_positions()
        
        for position in open_positions:
            symbol = position['symbol']
            
            # Check if the symbol is not in the rotator list
            if symbol not in rotator_symbols:
                
                # Check how long the position has been open
                position_open_time = position.get('timestamp', None)  # assuming your position has a 'timestamp' field
                current_time = time.time()
                time_elapsed = current_time - position_open_time
                
                # Fetch volume for the coin
                volume = self.exchange.get_24hr_volume(symbol)
                
                # Check if the volume is low and position has been open for too long
                if volume < self.MIN_VOLUME_THRESHOLD and time_elapsed > max_time_without_volume:
                    
                    # Place take profit order at the current price
                    current_price = self.exchange.get_current_price(symbol)
                    amount = position['amount']  # assuming your position has an 'amount' field
                    
                    # Determine if it's a buy or sell based on position type
                    order_type = "sell" if position['side'] == 'long' else "buy"
                    self.bybit_hedge_placetp_maker(symbol, amount, current_price, positionIdx=1, order_side="sell", open_orders=open_orders)
                    #self.exchange.place_order(symbol, order_type, amount, current_price, take_profit=True)
                    
                    logging.info(f"Placed take profit order for stale position: {symbol} at price: {current_price}")


    def cancel_stale_orders_bybit(self, symbol):
        current_time = time.time()
        if current_time - self.last_stale_order_check_time < 3720:  # 3720 seconds = 1 hour 12 minutes
            return  # Skip the rest of the function if it's been less than 1 hour 12 minutes

        # Directly cancel orders for the given symbol
        self.exchange.cancel_all_open_orders_bybit(symbol)
        logging.info(f"Stale orders for {symbol} canceled")

        self.last_stale_order_check_time = current_time  # Update the last check time

    def cancel_all_orders_for_symbol_bybit(self, symbol):
        try:
            self.exchange.cancel_all_open_orders_bybit(symbol)
            logging.info(f"All orders for {symbol} canceled")
        except Exception as e:
            logging.error(f"An error occurred while canceling all orders for {symbol}: {e}")

    def get_all_open_orders_bybit(self):
        """
        Fetch all open orders for all symbols from the Bybit API.
        
        :return: A list of open orders for all symbols.
        """
        try:
            # Call fetch_open_orders with no symbol to get orders for all symbols
            all_open_orders = self.exchange.fetch_open_orders()
            return all_open_orders
        except Exception as e:
            print(f"An error occurred while fetching all open orders: {e}")
            return []

    def cancel_old_entries_bybit(self, symbol):        
        # Cancel entries
        try:
            self.exchange.cancel_all_entries_bybit(symbol)
            logging.info(f"Canceled entry orders for {symbol}")
            time.sleep(0.05)
        except Exception as e:
            logging.info(f"An error occurred while canceling entry orders: {e}")


# Bybit cancel all entries
    def cancel_entries_binance(self, symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high):
        # Cancel entries
        current_time = time.time()
        if current_time - self.last_cancel_time >= 60:  # Execute this block every 1 minute
            try:
                if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                    self.exchange.cancel_all_entries_binance(symbol)
                    logging.info(f"Canceled entry orders for {symbol}")
                    time.sleep(0.05)
            except Exception as e:
                logging.info(f"An error occurred while canceling entry orders: {e}")

            self.last_cancel_time = current_time

# Bybit MFI ERI Trend entry logic

    def bybit_hedge_entry_maker_mfirsitrenderi(self, symbol, data, min_vol, min_dist, one_minute_volume, five_minute_distance, 
                                           eri_trend, open_orders, long_pos_qty, should_add_to_long, 
                                           max_long_trade_qty, best_bid_price, long_pos_price, long_dynamic_amount,
                                           short_pos_qty, should_add_to_short, max_short_trade_qty, 
                                           best_ask_price, short_pos_price, short_dynamic_amount):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                mfi = self.manager.get_asset_value(symbol, data, "MFI")
                trend = self.manager.get_asset_value(symbol, data, "Trend")

                if mfi is not None and isinstance(mfi, str):
                    if mfi.lower() == "neutral":
                        mfi = trend

                    # Place long orders when MFI is long and ERI trend is bearish
                    if (mfi.lower() == "long" and eri_trend.lower() == "bearish") or (mfi.lower() == "long" and trend.lower() == "long"):
                        existing_order = next((o for o in open_orders if o['side'] == 'Buy' and o['position_idx'] == 1), None)
                        if long_pos_qty == 0 or (should_add_to_long and long_pos_qty < max_long_trade_qty and best_bid_price < long_pos_price):
                            if existing_order is None or existing_order['price'] != best_bid_price:
                                if existing_order is not None:
                                    self.exchange.cancel_order_by_id(existing_order['id'], symbol)
                                logging.info(f"Placing long entry")
                                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                logging.info(f"Placed long entry")

                    # Place short orders when MFI is short and ERI trend is bullish
                    if (mfi.lower() == "short" and eri_trend.lower() == "bullish") or (mfi.lower() == "short" and trend.lower() == "short"):
                        existing_order = next((o for o in open_orders if o['side'] == 'Sell' and o['position_idx'] == 2), None)
                        if short_pos_qty == 0 or (should_add_to_short and short_pos_qty < max_short_trade_qty and best_ask_price > short_pos_price):
                            if existing_order is None or existing_order['price'] != best_ask_price:
                                if existing_order is not None:
                                    self.exchange.cancel_order_by_id(existing_order['id'], symbol)
                                logging.info(f"Placing short entry")
                                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                logging.info(f"Placed short entry")

    def bybit_hedge_entry_maker_mfirsitrend(self, symbol, data, min_vol, min_dist, one_minute_volume, five_minute_distance, 
                                            open_orders, long_pos_qty, should_add_to_long, 
                                           max_long_trade_qty, best_bid_price, long_pos_price, long_dynamic_amount,
                                           short_pos_qty, should_long: bool, should_short: bool, should_add_to_short, max_short_trade_qty, 
                                           best_ask_price, short_pos_price, short_dynamic_amount):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                mfi = self.manager.get_asset_value(symbol, data, "MFI")
                trend = self.manager.get_asset_value(symbol, data, "Trend")

                if mfi is not None and isinstance(mfi, str):
                    if mfi.lower() == "neutral":
                        mfi = trend

                    # Place long orders when MFI is long and ERI trend is bearish
                    if (mfi.lower() == "long" and trend.lower() == "long"):
                        existing_order = next((o for o in open_orders if o['side'] == 'Buy' and o['position_idx'] == 1), None)
                        if (should_long and long_pos_qty == 0) or (should_add_to_long and long_pos_qty < max_long_trade_qty and best_bid_price < long_pos_price):
                            if existing_order is None or existing_order['price'] != best_bid_price:
                                if existing_order is not None:
                                    self.exchange.cancel_order_by_id(existing_order['id'], symbol)
                                logging.info(f"Placing long entry")
                                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                logging.info(f"Placed long entry")

                    # Place short orders when MFI is short and ERI trend is bullish
                    if (mfi.lower() == "short" and trend.lower() == "short"):
                        existing_order = next((o for o in open_orders if o['side'] == 'Sell' and o['position_idx'] == 2), None)
                        if (should_short and short_pos_qty == 0) or (should_add_to_short and short_pos_qty < max_short_trade_qty and best_ask_price > short_pos_price):
                            if existing_order is None or existing_order['price'] != best_ask_price:
                                if existing_order is not None:
                                    self.exchange.cancel_order_by_id(existing_order['id'], symbol)
                                logging.info(f"Placing short entry")
                                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                logging.info(f"Placed short entry")

# Bybit MFIRSI only entry logic

    def bybit_hedge_entry_maker_mfirsi(self, symbol, data, min_vol, min_dist, one_minute_volume, five_minute_distance, 
                                       long_pos_qty, max_long_trade_qty, best_bid_price, long_pos_price, long_dynamic_amount,
                                       short_pos_qty, max_short_trade_qty, best_ask_price, short_pos_price, short_dynamic_amount):
        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                mfi = self.manager.get_asset_value(symbol, data, "MFI")

                max_long_trade_qty_for_symbol = self.max_long_trade_qty_per_symbol.get(symbol, 0)  # Get value for symbol or default to 0
                max_short_trade_qty_for_symbol = self.max_short_trade_qty_per_symbol.get(symbol, 0)  # Get value for symbol or default to 0


                if mfi is not None and isinstance(mfi, str):
                    if mfi.lower() == "long" and long_pos_qty == 0:
                        logging.info(f"Placing initial long entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1)
                        logging.info(f"Placed initial long entry with post-only order")
                    elif mfi.lower() == "long" and long_pos_qty < max_long_trade_qty_for_symbol and best_bid_price < long_pos_price:
                        logging.info(f"Placing additional long entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1)
                    elif mfi.lower() == "short" and short_pos_qty == 0:
                        logging.info(f"Placing initial short entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2)
                        logging.info(f"Placed initial short entry with post-only order")
                    elif mfi.lower() == "short" and short_pos_qty < max_short_trade_qty_for_symbol and best_ask_price > short_pos_price:
                        logging.info(f"Placing additional short entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2)

    def adjust_leverage_and_qty(self, symbol, current_qty, current_leverage, max_leverage, increase=True):
        logging.info(f"Symbol: {symbol}")
        logging.info(f"Max leverage: {max_leverage}")
        logging.info(f"Current leverage: {current_leverage}")
        logging.info(f"Current qty: {current_qty}")

        if increase:
            new_leverage = min(current_leverage + self.LEVERAGE_STEP, max_leverage, self.MAX_LEVERAGE)
            new_qty = current_qty * (1 + self.QTY_INCREMENT)
            logging.info(f"Increasing position. New qty: {new_qty}, New leverage: {new_leverage}")
        else:
            new_leverage = max(1.0, current_leverage - self.LEVERAGE_STEP)
            new_qty = max(self.MINIMUM_TRADE_QTY, current_qty * (1 - self.QTY_DECREMENT))
            logging.info(f"Decreasing position. New qty: {new_qty}, New leverage: {new_leverage}")

        return new_qty, new_leverage

    def set_position_leverage_long_bybit(self, symbol, long_pos_qty, total_equity, best_ask_price, max_leverage):
        # Ensure a lock exists for this symbol
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            if symbol not in self.initial_max_long_trade_qty_per_symbol:
                logging.warning(f"Symbol {symbol} not initialized in initial_max_long_trade_qty_per_symbol. Initializing now...")
                self.initial_max_long_trade_qty_per_symbol[symbol] = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)

            if symbol not in self.long_pos_leverage_per_symbol:
                logging.warning(f"Symbol {symbol} not initialized in long_pos_leverage_per_symbol. Initializing now...")
                self.long_pos_leverage_per_symbol[symbol] = 0.001  # starting leverage
                logging.info(f"Long leverage set to {self.long_pos_leverage_per_symbol[symbol]} for {symbol}")

            if long_pos_qty >= self.initial_max_long_trade_qty_per_symbol[symbol] and self.long_pos_leverage_per_symbol[symbol] < self.MAX_LEVERAGE:
                self.max_long_trade_qty_per_symbol[symbol], self.long_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                    symbol,
                    long_pos_qty,
                    self.long_pos_leverage_per_symbol[symbol], 
                    max_leverage, 
                    increase=True
                )
                logging.info(f"Long leverage for {symbol} temporarily increased to {self.long_pos_leverage_per_symbol[symbol]}x")

            elif long_pos_qty < (self.max_long_trade_qty_per_symbol.get(symbol, 0) / 2) and self.long_pos_leverage_per_symbol.get(symbol, 0) > 1.0:
                self.max_long_trade_qty_per_symbol[symbol], self.long_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                    symbol,
                    long_pos_qty,
                    self.long_pos_leverage_per_symbol[symbol], 
                    max_leverage, 
                    increase=False
                )
                logging.info(f"Long leverage for {symbol} returned to normal {self.long_pos_leverage_per_symbol[symbol]}x")

    def set_position_leverage_short_bybit(self, symbol, short_pos_qty, total_equity, best_ask_price, max_leverage):
        # Ensure a lock exists for this symbol
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            if symbol not in self.initial_max_short_trade_qty_per_symbol:
                logging.warning(f"Symbol {symbol} not initialized in initial_max_short_trade_qty_per_symbol. Initializing now...")
                self.initial_max_short_trade_qty_per_symbol[symbol] = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)

            if symbol not in self.short_pos_leverage_per_symbol:
                logging.warning(f"Symbol {symbol} not initialized in short_pos_leverage_per_symbol. Initializing now...")
                self.short_pos_leverage_per_symbol[symbol] = 0.001  # starting leverage

            if short_pos_qty >= self.initial_max_short_trade_qty_per_symbol[symbol] and self.short_pos_leverage_per_symbol[symbol] < self.MAX_LEVERAGE:
                self.max_short_trade_qty_per_symbol[symbol], self.short_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                    symbol,
                    short_pos_qty,
                    self.short_pos_leverage_per_symbol[symbol], 
                    max_leverage, 
                    increase=True
                )
                logging.info(f"Short leverage for {symbol} temporarily increased to {self.short_pos_leverage_per_symbol[symbol]}x")

            elif short_pos_qty < (self.max_short_trade_qty_per_symbol.get(symbol, 0) / 2) and self.short_pos_leverage_per_symbol.get(symbol, 0) > 1.0:
                self.max_short_trade_qty_per_symbol[symbol], self.short_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                    symbol,
                    short_pos_qty,
                    self.short_pos_leverage_per_symbol[symbol], 
                    max_leverage, 
                    increase=False
                )
                logging.info(f"Short leverage for {symbol} returned to normal {self.short_pos_leverage_per_symbol[symbol]}x")

    def calculate_atr(self, symbol, period=14):
        # Fetch historical OHLC data for the symbol and timeframe
        ohlc_data = self.exchange.get_ohlc_data(symbol, timeframe="1H", limit=period+1)  # Assuming hourly data for example, adjust as needed
        
        tr_values = []
        
        for i in range(1, len(ohlc_data)):
            high = ohlc_data[i]['high']
            low = ohlc_data[i]['low']
            prev_close = ohlc_data[i-1]['close']
            
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)
        
        # Calculate the average TR over the specified period
        atr = sum(tr_values) / period
        
        return atr

# Bybit position leverage management

    def bybit_reset_position_leverage_long(self, symbol, long_pos_qty, total_equity, best_ask_price, max_leverage):
        # Leverage increase logic for long positions
        if long_pos_qty >= self.initial_max_long_trade_qty and self.long_pos_leverage <= 1.0:
            self.max_long_trade_qty = 2 * self.initial_max_long_trade_qty  # double the maximum long trade quantity
            self.long_leverage_increased = True
            self.long_pos_leverage = 2.0
            logging.info(f"Long leverage for temporarily increased to {self.long_pos_leverage}x")
        elif long_pos_qty >= 2 * self.initial_max_long_trade_qty and self.long_pos_leverage <= 2.0:
            self.max_long_trade_qty = 3 * self.initial_max_long_trade_qty  # triple the maximum long trade quantity
            self.long_pos_leverage = 3.0
            logging.info(f"Long leverage temporarily increased to {self.long_pos_leverage}x")
        elif long_pos_qty < (self.max_long_trade_qty / 2) and self.long_pos_leverage > 1.0:
            max_trade_qty = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
            if isinstance(max_trade_qty, float):
                self.max_long_trade_qty = max_trade_qty
            else:
                logging.error(f"Expected max_trade_qty to be float, got {type(max_trade_qty)}")
            self.long_leverage_increased = False
            self.long_pos_leverage = 1.0
            logging.info(f"Long leverage returned to normal {self.long_pos_leverage}x")

    def bybit_reset_position_leverage_short(self, symbol, short_pos_qty, total_equity, best_ask_price, max_leverage):
        # Leverage increase logic for short positions
        if short_pos_qty >= self.initial_max_short_trade_qty and self.short_pos_leverage <= 1.0:
            self.max_short_trade_qty = 2 * self.initial_max_short_trade_qty  # double the maximum short trade quantity
            self.short_leverage_increased = True
            self.short_pos_leverage = 2.0
            logging.info(f"Short leverage temporarily increased to {self.short_pos_leverage}x")
        elif short_pos_qty >= 2 * self.initial_max_short_trade_qty and self.short_pos_leverage <= 2.0:
            self.max_short_trade_qty = 3 * self.initial_max_short_trade_qty  # triple the maximum short trade quantity
            self.short_pos_leverage = 3.0
            logging.info(f"Short leverage temporarily increased to {self.short_pos_leverage}x")
        elif short_pos_qty < (self.max_short_trade_qty / 2) and self.short_pos_leverage > 1.0:
            max_trade_qty = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
            if isinstance(max_trade_qty, float):
                self.max_short_trade_qty = max_trade_qty
            else:
                logging.error(f"Expected max_trade_qty to be float, got {type(max_trade_qty)}")
            self.short_leverage_increased = False
            self.short_pos_leverage = 1.0
            logging.info(f"Short leverage returned to normal {self.short_pos_leverage}x")

    def binance_auto_hedge_entry(self, trend, one_minute_volume, five_minute_distance, min_vol, min_dist,
                                should_long, long_pos_qty, long_dynamic_amount, best_bid_price, long_pos_price,
                                should_add_to_long, max_long_trade_qty, 
                                should_short, short_pos_qty, short_dynamic_amount, best_ask_price, short_pos_price,
                                should_add_to_short, max_short_trade_qty, symbol):

        if trend is not None and isinstance(trend, str):
            if one_minute_volume is not None and five_minute_distance is not None:
                if one_minute_volume > min_vol and five_minute_distance > min_dist:

                    if trend.lower() == "long" and should_long and long_pos_qty == 0:
                        print(f"Placing initial long entry")
                        self.exchange.binance_create_limit_order(symbol, "buy", long_dynamic_amount, best_bid_price)
                        print(f"Placed initial long entry")
                    elif trend.lower() == "long" and should_add_to_long and long_pos_qty < max_long_trade_qty and best_bid_price < long_pos_price:
                        print(f"Placing additional long entry")
                        self.exchange.binance_create_limit_order(symbol, "buy", long_dynamic_amount, best_bid_price)

                    if trend.lower() == "short" and should_short and short_pos_qty == 0:
                        print(f"Placing initial short entry")
                        self.exchange.binance_create_limit_order(symbol, "sell", short_dynamic_amount, best_ask_price)
                        print("Placed initial short entry")
                    elif trend.lower() == "short" and should_add_to_short and short_pos_qty < max_short_trade_qty and best_ask_price > short_pos_price:
                        print(f"Placing additional short entry")
                        self.exchange.binance_create_limit_order(symbol, "sell", short_dynamic_amount, best_ask_price)

    def binance_auto_hedge_entry_maker(self, trend, one_minute_volume, five_minute_distance, min_vol, min_dist,
                                should_long, long_pos_qty, long_dynamic_amount, best_bid_price, long_pos_price,
                                should_add_to_long, max_long_trade_qty, 
                                should_short, short_pos_qty, short_dynamic_amount, best_ask_price, short_pos_price,
                                should_add_to_short, max_short_trade_qty, symbol):

        if trend is not None and isinstance(trend, str):
            if one_minute_volume is not None and five_minute_distance is not None:
                if one_minute_volume > min_vol and five_minute_distance > min_dist:

                    if trend.lower() == "long" and should_long and long_pos_qty == 0:
                        print(f"Placing initial long entry")
                        self.exchange.binance_create_limit_order_with_time_in_force(symbol, "buy", long_dynamic_amount, best_bid_price, "GTC")
                        print(f"Placed initial long entry")
                    elif trend.lower() == "long" and should_add_to_long and long_pos_qty < max_long_trade_qty and best_bid_price < long_pos_price:
                        print(f"Placing additional long entry")
                        self.exchange.binance_create_limit_order_with_time_in_force(symbol, "buy", long_dynamic_amount, best_bid_price, "GTC")

                    if trend.lower() == "short" and should_short and short_pos_qty == 0:
                        print(f"Placing initial short entry")
                        self.exchange.binance_create_limit_order_with_time_in_force(symbol, "sell", short_dynamic_amount, best_ask_price, "GTC")
                        print("Placed initial short entry")
                    elif trend.lower() == "short" and should_add_to_short and short_pos_qty < max_short_trade_qty and best_ask_price > short_pos_price:
                        print(f"Placing additional short entry")
                        self.exchange.binance_create_limit_order_with_time_in_force(symbol, "sell", short_dynamic_amount, best_ask_price, "GTC")

    def binance_hedge_placetp_maker(self, symbol, pos_qty, take_profit_price, position_side, open_orders):
        order_side = 'SELL' if position_side == 'LONG' else 'BUY'
        existing_tps = self.get_open_take_profit_order_quantities_binance(open_orders, order_side)

        print(f"Existing TP IDs: {[order_id for _, order_id in existing_tps]}")
        print(f"Existing {order_side} TPs: {existing_tps}")

        # Cancel all TP orders if there is more than one existing TP order for the side
        if len(existing_tps) > 1:
            logging.info(f"More than one existing TP order found. Cancelling all {order_side} TP orders.")
            for qty, existing_tp_id in existing_tps:
                try:
                    self.exchange.cancel_order_by_id_binance(existing_tp_id, symbol)
                    logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                    time.sleep(0.05)
                except Exception as e:
                    raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e

        # If there is exactly one TP order for the side, and its quantity doesn't match the position quantity, cancel it
        elif len(existing_tps) == 1 and not math.isclose(existing_tps[0][0], pos_qty):
            logging.info(f"Existing TP qty {existing_tps[0][0]} and position qty {pos_qty} not close. Cancelling the TP order.")
            try:
                self.exchange.cancel_order_by_id_binance(existing_tps[0][1], symbol)
                logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                time.sleep(0.05)
            except Exception as e:
                raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e

        # Re-check the status of TP orders for the side
        existing_tps = self.get_open_take_profit_order_quantities_binance(self.exchange.get_open_orders(symbol), order_side)

        # Create a new TP order if no TP orders exist for the side or if all existing TP orders have been cancelled
        if not existing_tps:
            logging.info(f"No existing TP orders. Attempting to create new TP order.")
            try:
                new_order_id = f"tp_{position_side[:1]}_{uuid.uuid4().hex[:10]}"
                self.exchange.create_normal_take_profit_order_binance(symbol, order_side, pos_qty, take_profit_price, take_profit_price)#, {'newClientOrderId': new_order_id, 'reduceOnly': True})
                logging.info(f"{position_side} take profit set at {take_profit_price}")
                time.sleep(0.05)
            except Exception as e:
                raise Exception(f"Error in placing {position_side} TP: {e}") from e
        else:
            logging.info(f"Existing TP orders found. Not creating new TP order.")

#    def create_normal_take_profit_order_binance(self, symbol, side, quantity, price, stopPrice):

    # def binance_hedge_placetp_maker(self, symbol, pos_qty, take_profit_price, position_side, open_orders):
    #     order_side = 'sell' if position_side == 'LONG' else 'buy'
    #     existing_tps = self.get_open_take_profit_limit_order_quantities_binance(open_orders, order_side)

    #     print(f"Existing TP IDs: {[order_id for _, order_id in existing_tps]}")
    #     print(f"Existing {order_side} TPs: {existing_tps}")

    #     # Cancel all TP orders if there is more than one existing TP order for the side
    #     if len(existing_tps) > 1:
    #         logging.info(f"More than one existing TP order found. Cancelling all {order_side} TP orders.")
    #         for qty, existing_tp_id in existing_tps:
    #             try:
    #                 self.exchange.cancel_order_by_id_binance(existing_tp_id, symbol)
    #                 logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
    #                 time.sleep(0.05)
    #             except Exception as e:
    #                 raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e
    #     # If there is exactly one TP order for the side, and its quantity doesn't match the position quantity, cancel it
    #     elif len(existing_tps) == 1 and not math.isclose(existing_tps[0][0], pos_qty):
    #         logging.info(f"Existing TP qty {existing_tps[0][0]} and position qty {pos_qty} not close. Cancelling the TP order.")
    #         try:
    #             self.exchange.cancel_order_by_id_binance(existing_tps[0][1], symbol)
    #             logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
    #             time.sleep(0.05)
    #         except Exception as e:
    #             raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e

    #     # Re-check the status of TP orders for the side
    #     existing_tps = self.get_open_take_profit_limit_order_quantities_binance(self.exchange.get_open_orders(symbol), order_side)
    #     # Create a new TP order if no TP orders exist for the side or if all existing TP orders have been cancelled
    #     if not existing_tps:
    #         logging.info(f"No existing TP orders. Attempting to create new TP order.")
    #         try:
    #             self.exchange.binance_create_reduce_only_limit_order(symbol, order_side, pos_qty, take_profit_price)
    #             logging.info(f"{position_side} take profit set at {take_profit_price}")
    #             time.sleep(0.05)
    #         except Exception as e:
    #             raise Exception(f"Error in placing {position_side} TP: {e}") from e
    #     else:
    #         logging.info(f"Existing TP orders found. Not creating new TP order.")


    #MARKET ORDER THOUGH
    def binance_hedge_placetp_market(self, symbol, pos_qty, take_profit_price, position_side, open_orders):
        order_side = 'sell' if position_side == 'LONG' else 'buy'
        existing_tps = self.get_open_take_profit_order_quantities_binance(open_orders, order_side)

        print(f"Existing TP IDs: {[order_id for _, order_id in existing_tps]}")
        print(f"Existing {order_side} TPs: {existing_tps}")

        # Cancel all TP orders if there is more than one existing TP order for the side
        if len(existing_tps) > 1:
            logging.info(f"More than one existing TP order found. Cancelling all {order_side} TP orders.")
            for qty, existing_tp_id in existing_tps:
                try:
                    self.exchange.cancel_order_by_id_binance(existing_tp_id, symbol)
                    logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                    time.sleep(0.05)
                except Exception as e:
                    raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e
        # If there is exactly one TP order for the side, and its quantity doesn't match the position quantity, cancel it
        elif len(existing_tps) == 1 and not math.isclose(existing_tps[0][0], pos_qty):
            logging.info(f"Existing TP qty {existing_tps[0][0]} and position qty {pos_qty} not close. Cancelling the TP order.")
            try:
                existing_tp_id = existing_tps[0][1]
                self.exchange.cancel_order_by_id_binance(existing_tp_id, symbol)
                logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                time.sleep(0.05)
            except Exception as e:
                raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e

        # elif len(existing_tps) == 1 and not math.isclose(existing_tps[0][0], pos_qty):
        #     logging.info(f"Existing TP qty {existing_tps[0][0]} and position qty {pos_qty} not close. Cancelling the TP order.")
        #     try:
        #         self.exchange.cancel_order_by_id_binance(existing_tps[0][1], symbol)
        #         logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
        #         time.sleep(0.05)
        #     except Exception as e:
        #         raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e

        # Re-check the status of TP orders for the side
        existing_tps = self.get_open_take_profit_order_quantities_binance(self.exchange.get_open_orders(symbol), order_side)
        # Create a new TP order if no TP orders exist for the side or if all existing TP orders have been cancelled
        if not existing_tps:
            logging.info(f"No existing TP orders. Attempting to create new TP order.")
            try:
                new_order_id = f"tp_{position_side[:1]}_{uuid.uuid4().hex[:10]}"
                self.exchange.binance_create_take_profit_order(symbol, order_side, position_side, pos_qty, take_profit_price, {'stopPrice': take_profit_price, 'newClientOrderId': new_order_id})
                logging.info(f"{position_side} take profit set at {take_profit_price}")
                time.sleep(0.05)
            except Exception as e:
                raise Exception(f"Error in placing {position_side} TP: {e}") from e
        else:
            logging.info(f"Existing TP orders found. Not creating new TP order.")

    # def binance_hedge_placetp_maker(self, symbol, pos_qty, take_profit_price, position_side, open_orders):
    #     order_side = 'sell' if position_side == 'LONG' else 'buy'
    #     existing_tps = self.get_open_take_profit_order_quantities_binance(open_orders, order_side)

    #     print(f"Existing TP IDs: {[order_id for _, order_id in existing_tps]}")
    #     print(f"Existing {order_side} TPs: {existing_tps}")

    #     # Cancel all TP orders if there is more than one existing TP order for the side
    #     if len(existing_tps) > 1:
    #         logging.info(f"More than one existing TP order found. Cancelling all {order_side} TP orders.")
    #         for qty, existing_tp_id in existing_tps:
    #             try:
    #                 self.exchange.cancel_order_by_id_binance(existing_tp_id, symbol)
    #                 logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
    #                 time.sleep(0.05)
    #             except Exception as e:
    #                 raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e
    #     # If there is exactly one TP order for the side, and its quantity doesn't match the position quantity, cancel it
    #     elif len(existing_tps) == 1 and not math.isclose(existing_tps[0][0], pos_qty):
    #         logging.info(f"Existing TP qty {existing_tps[0][0]} and position qty {pos_qty} not close. Cancelling the TP order.")
    #         try:
    #             self.exchange.cancel_order_by_id_binance(existing_tps[0][1], symbol)
    #             logging.info(f"{order_side.capitalize()} take profit {existing_tps[0][1]} canceled")
    #             time.sleep(0.05)
    #         except Exception as e:
    #             raise Exception(f"Error in cancelling {order_side} TP orders: {e}") from e

    #     # Create a new TP order if no TP orders exist for the side or if all existing TP orders have been cancelled
    #     if not self.get_open_take_profit_order_quantities_binance(self.exchange.get_open_orders(symbol), order_side):
    #         logging.info(f"No existing TP orders. Attempting to create new TP order.")
    #         try:
    #             new_order_id = f"tp_{position_side[:1]}_{uuid.uuid4().hex[:10]}"
    #             self.exchange.binance_create_take_profit_order(symbol, order_side, position_side, pos_qty, take_profit_price, {'stopPrice': take_profit_price, 'newClientOrderId': new_order_id})
    #             logging.info(f"{position_side} take profit set at {take_profit_price}")
    #             time.sleep(0.05)
    #         except Exception as e:
    #             raise Exception(f"Error in placing {position_side} TP: {e}") from e
    #     else:
    #         logging.info(f"Existing TP orders found. Not creating new TP order.")