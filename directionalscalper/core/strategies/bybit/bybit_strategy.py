from colorama import Fore
from typing import Optional, Tuple, List, Dict, Union
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_DOWN
import inspect
import pandas as pd
import time
import math
import numpy as np
import random
import ta as ta
import uuid
import os
import uuid
import logging
import json
import threading
import traceback
import ccxt
import pytz
import sqlite3
import keyboard
from collections import defaultdict
from ..logger import Logger
from datetime import datetime, timedelta
from threading import Thread, Lock

from ...bot_metrics import BotDatabase

from directionalscalper.core.config_initializer import ConfigInitializer
from directionalscalper.core.strategies.base_strategy import BaseStrategy

from rate_limit import RateLimit

logging = Logger(logger_name="BybitBaseStrategy", filename="BybitBaseStrategy.log", stream=True)

grid_lock = threading.Lock()

class BybitStrategy(BaseStrategy):
    def __init__(self, exchange, config, manager, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.exchange = exchange
        self.general_rate_limiter = RateLimit(50, 1)
        self.order_rate_limiter = RateLimit(5, 1) 
        self.previous_long_pos_qty = {}
        self.previous_short_pos_qty = {}
        self.symbol_max_leverage = {}
        self.grid_levels = {}
        self.linear_grid_orders = {}
        self.last_price = {}
        self.last_cancel_time = {}
        self.last_signal_time = {}
        self.last_mfirsi_signal = {}
        self.cancel_all_orders_interval = 240
        self.cancel_interval = 120
        self.order_refresh_interval = 120  # seconds
        self.last_order_refresh_time = 0
        self.last_grid_cancel_time = {}
        self.entered_grid_levels = {}
        self.filled_order_levels = {}
        self.filled_levels = {}
        self.max_qty_reached_symbol_long = set()  # Tracking symbols that exceed max long position qty
        self.max_qty_reached_symbol_short = set()  # Tracking symbols that exceed max short position qty
        self.active_grids = set()
        self.active_long_grids = set()  # Tracking active long grids
        self.active_short_grids = set() 
        self.position_inactive_threshold = 150
        self.no_entry_signal_threshold = 150
        self.order_inactive_threshold = 150
        self.last_activity_time = {}
        self.last_open_position_timestamp = defaultdict(lambda: {"buy": None, "sell": None})
        self.last_cleared_time = {}  # Dictionary to store last cleared time for symbols
        self.clear_interval = timedelta(minutes=30)  # Time interval threshold for clearing grids
        self.last_empty_grid_time = {}
        self.last_reissue_price_long = {}
        self.last_reissue_price_short = {}
        self.placed_levels = {}
        self.last_processed_signal = {}
        self.last_processed_time_long = {}  # Dictionary to store the last processed time for long positions
        self.last_processed_time_short = {}
        self.invalid_long_condition_time = {}
        self.invalid_short_condition_time = {}
        self.next_long_tp_update = datetime.now() - timedelta(seconds=10)
        self.next_short_tp_update = datetime.now() - timedelta(seconds=10)
        self.grid_cleared_status = defaultdict(lambda: {"long": False, "short": False})
        ConfigInitializer.initialize_config_attributes(self, config)

        try:
            # Hotkey-related attributes
            self.hotkey_flags = {
                "enter_long": False,
                "take_profit_long": False,
                "enter_short": False,
                "take_profit_short": False
            }
            self.hotkeys = config.hotkeys  # Accessing hotkeys directly from config
            self.hotkey_listener_enabled = self.hotkeys.hotkeys_enabled
            if self.hotkey_listener_enabled:
                self.start_hotkey_listener()
        except Exception as e:
            logging.info(f"Exception caught in hotkeys {e}")

    def start_hotkey_listener(self):
        hotkey_thread = threading.Thread(target=self.listen_hotkeys, daemon=True)
        hotkey_thread.start()

    def listen_hotkeys(self):
        while True:
            if keyboard.is_pressed(self.hotkeys.enter_long):
                self.hotkey_flags['enter_long'] = True
            if keyboard.is_pressed(self.hotkeys.take_profit_long):
                self.hotkey_flags['take_profit_long'] = True
            if keyboard.is_pressed(self.hotkeys.enter_short):
                self.hotkey_flags['enter_short'] = True
            if keyboard.is_pressed(self.hotkeys.take_profit_short):
                self.hotkey_flags['take_profit_short'] = True
            time.sleep(0.1)  # Add a small delay to avoid high CPU usage
            
    def hotkey_trading_strategy(self, open_orders: list, symbol: str, total_equity: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, long_dynamic_amount: float, short_dynamic_amount: float, upnl_profit_pct: float, tp_order_counts: dict):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                if self.hotkey_flags["enter_long"] and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry order for {symbol} due to hotkey")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    self.hotkey_flags["enter_long"] = False

                if self.hotkey_flags["take_profit_long"] and long_pos_qty > 0:
                    logging.info(f"Taking profit on long position for {symbol} due to hotkey")
                    self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, upnl_profit_pct, open_orders)
                    self.hotkey_flags["take_profit_long"] = False

                if self.hotkey_flags["enter_short"] and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry order for {symbol} due to hotkey")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    self.hotkey_flags["enter_short"] = False

                if self.hotkey_flags["take_profit_short"] and short_pos_qty > 0:
                    logging.info(f"Taking profit on short position for {symbol} due to hotkey")
                    self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, upnl_profit_pct, open_orders)
                    self.hotkey_flags["take_profit_short"] = False

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in hotkey trading strategy: {e}")

    TAKER_FEE_RATE = 0.00055

    def generate_l_signals(self, symbol):
        with self.general_rate_limiter:
            return self.exchange.generate_l_signals(symbol)
        
    def get_market_data_with_retry(self, symbol, max_retries=5, retry_delay=5):
        for i in range(max_retries):
            try:
                with self.general_rate_limiter:
                    return self.exchange.get_market_data_bybit(symbol)
            except Exception as e:
                if i < max_retries - 1:
                    logging.info(f"Error occurred while fetching market data: {e}. Retrying in {retry_delay} seconds...")
                    logging.info(f"Call Stack: {traceback.format_exc()}")
                    time.sleep(retry_delay)
                else:
                    #logging.info("Excessive call stack:\n" + "".join(traceback.format_stack()))
                    logging.info(f"get_market_data_with_retry failure from bybit_strategy.py")
                    raise e
   
    # def get_market_data_with_retry(self, symbol, max_retries=5, initial_retry_delay=5):
    #     retry_delay = initial_retry_delay

    #     for i in range(max_retries):
    #         try:
    #             with self.general_rate_limiter:
    #                 return self.exchange.get_market_data_bybit(symbol)
    #         except Exception as e:
    #             if i < max_retries - 1:
    #                 logging.info(f"Error occurred while fetching market data: {e}. Retrying in {retry_delay} seconds...")
    #                 logging.info(f"Call Stack: {traceback.format_exc()}")
    #                 time.sleep(retry_delay)
    #                 retry_delay *= 2  # Exponential backoff
    #             else:
    #                 raise e
    #                 logging.info(f"Exception in get market data {e}")
                
    # def get_market_data_with_retry(self, symbol, max_retries=5, retry_delay=5):
    #     for i in range(max_retries):
    #         try:
    #             return self.exchange.get_market_data_bybit(symbol)
    #         except Exception as e:
    #             if i < max_retries - 1:
    #                 print(f"Error occurred while fetching market data: {e}. Retrying in {retry_delay} seconds...")
    #                 time.sleep(retry_delay)
    #             else:
    #                 raise e
                
    def update_dynamic_amounts(self, symbol, total_equity, best_ask_price, best_bid_price):
        if symbol not in self.long_dynamic_amount or symbol not in self.short_dynamic_amount:
            long_dynamic_amount, short_dynamic_amount, _ = self.calculate_dynamic_amounts(symbol, total_equity, best_ask_price, best_bid_price)
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
    
    def get_open_symbols(self):
        open_position_data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)
        position_symbols = set()
        for position in open_position_data:
            info = position.get('info', {})
            position_symbol = info.get('symbol', '').split(':')[0]
            if 'size' in info and 'side' in info:
                position_symbols.add(position_symbol.replace("/", ""))
        return position_symbols

    def place_reduce_only_limit_order(exchange, symbol, side, quantity, stop_loss_price, positionIdx=0):
        """
        Place a reduce-only limit order for stop loss.

        Parameters:
        - exchange: ccxt exchange instance.
        - symbol: Trading pair symbol (e.g., 'BTC/USDT').
        - side: 'sell' for long position stop loss, 'buy' for short position stop loss.
        - quantity: Amount of the position to reduce.
        - stop_loss_price: Price at which the stop loss should trigger.
        - positionIdx: Position index for Bybit (1 for long, 2 for short).
        """

        try:
            # Fetch market info to ensure precision for quantity and price
            market = exchange.markets[symbol]
            precision_amount = market['precision']['amount']
            precision_price = market['precision']['price']

            # Round quantity and stop_loss_price to appropriate precision
            quantity = round(quantity, precision_amount)
            stop_loss_price = round(stop_loss_price, precision_price)

            order = exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=quantity,
                price=stop_loss_price,
                params={'reduceOnly': True, 'positionIdx': positionIdx}  # Include positionIdx for Bybit
            )
            logging.info(f"Reduce-only limit order placed: {order}")
            return order

        except Exception as e:
            logging.error(f"Error placing reduce-only limit order for {symbol}: {e}")
            raise


    def execute_grid_auto_reduce(self, position_type, symbol, pos_qty, dynamic_amount, market_price, total_equity, long_pos_price, short_pos_price, min_qty):
        """
        Executes auto-reduction of positions by placing tagged limit orders.
        """
        amount_precision, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
        price_precision_level = -int(math.log10(price_precision))
        qty_precision_level = -int(math.log10(amount_precision))
        
        market_price = Decimal(str(market_price))
        max_levels, price_interval = self.calculate_dynamic_auto_reduce_levels(symbol, pos_qty, market_price, total_equity, long_pos_price, short_pos_price)
        
        for i in range(1, max_levels + 1):
            step_price = self.calculate_step_price(position_type, market_price, price_interval, i)
            if not self.is_price_valid(position_type, step_price, market_price):
                continue
            
            step_price = round(step_price, price_precision_level)
            adjusted_dynamic_amount = max(dynamic_amount, min_qty)
            adjusted_dynamic_amount = round(adjusted_dynamic_amount, qty_precision_level)
            
            tag = f"auto_reduce_{position_type}_{symbol}_{step_price}_{i}"
            self.place_tagged_limit_order(symbol, 'sell' if position_type == 'long' else 'buy', adjusted_dynamic_amount, step_price, True, tag)

    def calculate_step_price(self, position_type, market_price, price_interval, level):
        """
        Calculates the step price for auto-reduce orders.
        """
        return market_price + (price_interval * level) if position_type == 'long' else market_price - (price_interval * level)

    def is_price_valid(self, position_type, step_price, market_price):
        """
        Checks if the calculated step price is valid for the given position type.
        """
        return step_price > market_price if position_type == 'long' else step_price < market_price

    def place_tagged_limit_order(self, symbol, side, amount, price, reduce_only, tag):
        """
        Places a tagged limit order with the option for it to be a reduce-only order.
        """
        try:
            order = self.exchange.create_tagged_limit_order_bybit(symbol, side, amount, price, reduceOnly=reduce_only, orderLinkId=tag)
            logging.info(f"Placed {side} order at {price} with tag {tag} and amount {amount}")
            return order.get('id', None) if order else None
        except Exception as e:
            logging.info(f"Error placing {side} order at {price} with tag {tag}: {e}")
            return None

    def check_symbol_inactivity(self, symbol, inactive_time_threshold):
        current_time = time.time()
        logging.info(f"Checking inactivity for {symbol} at {current_time}")

        # Check if the symbol has open positions
        open_position_data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)
        symbol_formatted = f"{symbol.split('USDT')[0]}/USDT:USDT"
        long_pos_qty = sum(pos['size'] for pos in open_position_data if pos['symbol'] == symbol_formatted and pos['side'] == 'Buy')
        short_pos_qty = sum(pos['size'] for pos in open_position_data if pos['symbol'] == symbol_formatted and pos['side'] == 'Sell')

        has_open_long_position = long_pos_qty > 0
        has_open_short_position = short_pos_qty > 0

        logging.info(f"Open positions for {symbol} - Long: {'found' if has_open_long_position else 'none'}, Short: {'found' if has_open_short_position else 'none'}")

        # Check if the symbol has open orders
        open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
        has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
        has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

        logging.info(f"Open orders for {symbol} - Long: {'found' if has_open_long_order else 'none'}, Short: {'found' if has_open_short_order else 'none'}")

        # Determine inactivity and handle accordingly
        if not has_open_long_position and not has_open_long_order:
            if self.handle_inactivity(symbol, 'long', current_time, inactive_time_threshold):
                return True
        
        if not has_open_short_position and not has_open_short_order:
            if self.handle_inactivity(symbol, 'short', current_time, inactive_time_threshold):
                return True

        return False

    def handle_inactivity(self, symbol, side, current_time, inactive_time_threshold, previous_qty):
        if symbol in self.last_activity_time:
            last_activity_time = self.last_activity_time[symbol]
            inactive_time = current_time - last_activity_time
            logging.info(f"{symbol} ({side}) last active {inactive_time} seconds ago")
            if inactive_time >= inactive_time_threshold and previous_qty > 0:
                logging.info(f"{symbol} ({side}) has been inactive for {inactive_time} seconds, exceeding threshold of {inactive_time_threshold} seconds")
                if side == 'long':
                    self.cancel_grid_orders(symbol, 'buy')
                    self.active_long_grids.discard(symbol)
                    self.running_long = False
                elif side == 'short':
                    self.cancel_grid_orders(symbol, 'sell')
                    self.active_short_grids.discard(symbol)
                    self.running_short = False
                return True
        else:
            self.last_activity_time[symbol] = current_time
            logging.info(f"Recording initial activity time for {symbol} ({side})")
        return False

    def check_position_inactivity(self, symbol, inactive_pos_time_threshold, long_pos_qty, short_pos_qty, previous_long_pos_qty, previous_short_pos_qty):
        current_time = time.time()
        logging.info(f"Checking position inactivity for {symbol} at {current_time}")

        has_open_long_position = long_pos_qty > 0
        has_open_short_position = short_pos_qty > 0

        logging.info(f"Open positions status for {symbol} - Long: {'found' if has_open_long_position else 'none'}, Short: {'found' if has_open_short_position else 'none'}")

        # Determine inactivity and handle accordingly
        if not has_open_long_position:
            if self.handle_inactivity(symbol, 'long', current_time, inactive_pos_time_threshold, previous_long_pos_qty):
                return True
        
        if not has_open_short_position:
            if self.handle_inactivity(symbol, 'short', current_time, inactive_pos_time_threshold, previous_short_pos_qty):
                return True

        return False
    
    def should_terminate_open_orders(self, symbol, long_pos_qty, short_pos_qty, open_positions_data, open_orders, current_time):
        try:
            # Assuming input is normalized
            normalized_symbol = symbol.replace('/', '')

            # Check for the presence of the symbol in open positions
            has_position = any(normalized_symbol in pos['symbol'].replace('/', '') for pos in open_positions_data)

            logging.info(f"{symbol} has position: {has_position}")

            # Filter active orders
            active_orders = [order for order in open_orders if not order.get('reduceOnly', False)]

            #logging.info(f"Active orders for {symbol}: {active_orders}")

            # Identify active long and short orders
            long_orders = [order for order in active_orders if order['side'] == 'buy']
            short_orders = [order for order in active_orders if order['side'] == 'sell']

            # logging.info(f"Long orders for {symbol} {long_orders}")
            # logging.info(f"Short orders for {symbol} {short_orders}")
            
            # Determine if orders should be terminated
            should_terminate_long = long_orders and not has_position and long_pos_qty == 0
            should_terminate_short = short_orders and not has_position and short_pos_qty == 0

            logging.info(f"Should terminate long for {symbol} {should_terminate_long}")
            logging.info(f"Should terminate short for {symbol} {should_terminate_short}")

            terminate_long = False
            terminate_short = False

            # Manage termination based on stored last active times
            if should_terminate_long:
                if symbol not in self.exchange.last_active_long_order_time:
                    self.exchange.last_active_long_order_time[symbol] = current_time
                elif current_time - self.exchange.last_active_long_order_time[symbol] > self.order_inactive_threshold:
                    terminate_long = True
                    logging.info(f"Terminate long for {symbol} : {terminate_long}")
            else:
                if symbol in self.exchange.last_active_long_order_time:
                    del self.exchange.last_active_long_order_time[symbol]

            if should_terminate_short:
                if symbol not in self.exchange.last_active_short_order_time:
                    self.exchange.last_active_short_order_time[symbol] = current_time
                elif current_time - self.exchange.last_active_short_order_time[symbol] > self.order_inactive_threshold:
                    terminate_short = True
                    logging.info(f"Terminate short for {symbol} : {terminate_short}")
            else:
                if symbol in self.exchange.last_active_short_order_time:
                    del self.exchange.last_active_short_order_time[symbol]

            return terminate_long, terminate_short

        except Exception as e:
            logging.error(f"Exception caught in should terminate open orders: {e}")
            return False, False
        
    # Threading locks
    def should_terminate_full(self, symbol, current_time, previous_long_pos_qty, long_pos_qty, previous_short_pos_qty, short_pos_qty):
        open_symbols = self.get_open_symbols()  # Fetch open symbols

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            if symbol not in open_symbols:
                if not hasattr(self, 'position_closed_time'):
                    self.position_closed_time = current_time
                elif current_time - self.position_closed_time > self.position_inactive_threshold:
                    logging.info(f"Position for {symbol} has been inactive for more than {self.position_inactive_threshold} seconds.")
                    return True

            if not hasattr(self, 'last_entry_signal_time'):
                self.last_entry_signal_time = current_time
            elif current_time - self.last_entry_signal_time > self.no_entry_signal_threshold:
                logging.info(f"No entry signal for {symbol} in the last {self.no_entry_signal_threshold} seconds.")
                return True
            else:
                if hasattr(self, 'position_closed_time'):
                    del self.position_closed_time

            if previous_long_pos_qty > 0 and long_pos_qty == 0:
                logging.info(f"Long position closed for {symbol}. Canceling long grid orders.")
                self.cancel_grid_orders(symbol, "buy")
                return True

            if previous_short_pos_qty > 0 and short_pos_qty == 0:
                logging.info(f"Short position closed for {symbol}. Canceling short grid orders.")
                self.cancel_grid_orders(symbol, "sell")
                return True

        return False
        
    def should_terminate(self, symbol, current_time):
        open_symbols = self.get_open_symbols()  # Fetch open symbols
        if symbol not in open_symbols:
            if not hasattr(self, 'position_closed_time'):
                self.position_closed_time = current_time
            elif current_time - self.position_closed_time > self.position_inactive_threshold:
                logging.info(f"Position for {symbol} has been inactive for more than {self.position_inactive_threshold} seconds.")
                return True
        else:
            if hasattr(self, 'position_closed_time'):
                del self.position_closed_time
        return False

    def cleanup_before_termination(self, symbol):
        # Cancel all orders for the symbol and perform any other cleanup needed
        self.exchange.cancel_all_orders_for_symbol_bybit(symbol)

    def calculate_next_update_time(self):
        """Returns the time for the next TP update, which is 3 seconds from the current time."""
        now = datetime.now()
        next_update_time = now + timedelta(seconds=1)
        return next_update_time.replace(microsecond=0)

    # Bybit cancel all entries
    def cancel_entries_bybit(self, symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high):
        # Cancel entries
        current_time = time.time()
        if current_time - self.last_entries_cancel_time >= 60: #60 # Execute this block every 1 minute
            try:
                if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                    self.exchange.cancel_all_entries_bybit(symbol)
                    logging.info(f"Canceled entry orders for {symbol}")
                    time.sleep(0.05)
            except Exception as e:
                logging.info(f"An error occurred while canceling entry orders: {e}")

            self.last_entries_cancel_time = current_time

    def clear_stale_positions(self, open_orders, rotator_symbols, max_time_without_volume=3600): # default time is 1 hour
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
            logging.info(f"An error occurred while canceling all orders for {symbol}: {e}")

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

    def update_quickscalp_tp_dynamic(self, symbol, pos_qty, upnl_profit_pct, max_upnl_profit_pct, short_pos_price, long_pos_price, positionIdx, order_side, last_tp_update, tp_order_counts, open_orders):
        # Fetch the current open TP orders and TP order counts for the symbol
        long_tp_orders, short_tp_orders = self.retry_api_call(self.exchange.get_open_tp_orders, open_orders)
        long_tp_count = tp_order_counts['long_tp_count']
        short_tp_count = tp_order_counts['short_tp_count']

        # Determine the minimum notional value for dynamic scaling
        min_notional_value = self.min_notional(symbol)
        current_price = self.exchange.get_current_price(symbol)

        # Calculate the position's market value
        position_market_value = pos_qty * current_price

        # Calculate the dynamic TP range based on how many minimum notional units fit in the position's market value
        num_units = position_market_value / min_notional_value

        # Modify scaling factor calculation using logarithmic scaling for a smoother increase
        scaling_factor = math.log10(num_units + 1)  # Logarithmic scaling to smooth out the scaling progression

        # Calculate scaled TP percentage within the defined range
        scaled_tp_pct = upnl_profit_pct + (max_upnl_profit_pct - upnl_profit_pct) * min(scaling_factor, 1)  # Cap scaling at 100% to avoid excessive TP targets

        # Calculate the new TP values using the quickscalp method
        new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, upnl_profit_pct, scaled_tp_pct)
        new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, upnl_profit_pct, scaled_tp_pct)

        # Determine the relevant TP orders based on the order side
        relevant_tp_orders = long_tp_orders if order_side == "sell" else short_tp_orders

        # Check if there's an existing TP order with a mismatched quantity
        mismatched_qty_orders = [order for order in relevant_tp_orders if order['qty'] != pos_qty and order['id'] not in self.auto_reduce_order_ids.get(symbol, [])]

        # Cancel mismatched TP orders if any
        for order in mismatched_qty_orders:
            try:
                self.exchange.cancel_order_by_id(order['id'], symbol)
                logging.info(f"Cancelled TP order {order['id']} for update.")
            except Exception as e:
                logging.info(f"Error in cancelling {order_side} TP order {order['id']}. Error: {e}")

        # Using datetime.now() for checking if update is needed
        now = datetime.now()
        if now >= last_tp_update or mismatched_qty_orders:
            # Check if a TP order already exists
            tp_order_exists = (order_side == "sell" and long_tp_count > 0) or (order_side == "buy" and short_tp_count > 0)

            # Set new TP order with updated prices only if no TP order exists
            if not tp_order_exists:
                new_tp_price_min = new_long_tp_min if order_side == "sell" else new_short_tp_min
                logging.info(f"New tp price min: {new_tp_price_min} with order side as {order_side}")
                new_tp_price_max = new_long_tp_max if order_side == "sell" else new_short_tp_max
                current_price = self.exchange.get_current_price(symbol)
                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                # Ensure TP setting checks are correct for direction
                if (order_side == "sell" and current_price >= new_tp_price_min) or (order_side == "buy" and current_price <= new_tp_price_max):
                    # Check if current price has already surpassed the max TP price
                    if (order_side == "sell" and current_price > new_tp_price_max) or (order_side == "buy" and current_price < new_tp_price_min):
                        try:
                            tp_price = best_ask_price if order_side == "sell" else best_bid_price
                            self.exchange.create_normal_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, tp_price, positionIdx=positionIdx, reduce_only=True)
                            logging.info(f"New {order_side.capitalize()} TP set at current/best price {tp_price} as current price has surpassed the max TP")
                        except Exception as e:
                            logging.info(f"Failed to set new {order_side} TP for {symbol} at current/best price. Error: {e}")
                    else:
                        try:
                            self.exchange.create_normal_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, new_tp_price_min, positionIdx=positionIdx, reduce_only=True)
                            logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price_min} using a normal limit order")
                        except Exception as e:
                            logging.info(f"Failed to set new {order_side} TP for {symbol} using a normal limit order. Error: {e}")
                else:
                    try:
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, new_tp_price_max, positionIdx=positionIdx, reduce_only=True)
                        logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price_max} using a post-only order")
                    except Exception as e:
                        logging.info(f"Failed to set new {order_side} TP for {symbol} using a post-only order. Error: {e}")
            else:
                logging.info(f"Skipping TP update as a TP order already exists for {symbol}")

            # Calculate and return the next update time
            return self.calculate_next_update_time()
        else:
            logging.info(f"No immediate update needed for TP orders for {symbol}. Last update at: {last_tp_update}")
            return last_tp_update
        
    def update_quickscalp_tp(self, symbol, pos_qty, upnl_profit_pct, short_pos_price, long_pos_price, positionIdx, order_side, last_tp_update, tp_order_counts, open_orders, max_retries=10):
        # Fetch the current open TP orders and TP order counts for the symbol
        long_tp_orders, short_tp_orders = self.exchange.get_open_tp_orders(open_orders)
        long_tp_count = tp_order_counts['long_tp_count']
        short_tp_count = tp_order_counts['short_tp_count']

        # Calculate the new TP values using quickscalp method
        new_short_tp = self.calculate_quickscalp_short_take_profit(short_pos_price, symbol, upnl_profit_pct)
        new_long_tp = self.calculate_quickscalp_long_take_profit(long_pos_price, symbol, upnl_profit_pct)

        # Determine the relevant TP orders based on the order side
        relevant_tp_orders = long_tp_orders if order_side == "sell" else short_tp_orders

        # Check if there's an existing TP order with a mismatched quantity
        mismatched_qty_orders = [order for order in relevant_tp_orders if order['qty'] != pos_qty and order['id'] not in self.auto_reduce_order_ids.get(symbol, [])]

        # Cancel mismatched TP orders if any
        for order in mismatched_qty_orders:
            try:
                self.exchange.cancel_order_by_id(order['id'], symbol)
                logging.info(f"Cancelled TP order {order['id']} for update.")
            except Exception as e:
                logging.info(f"Error in cancelling {order_side} TP order {order['id']}. Error: {e}")

        now = datetime.now()
        if now >= last_tp_update or mismatched_qty_orders:
            # Check if a TP order already exists
            tp_order_exists = (order_side == "sell" and long_tp_count > 0) or (order_side == "buy" and short_tp_count > 0)

            # Set new TP order with updated prices only if no TP order exists
            if not tp_order_exists:
                new_tp_price = new_long_tp if order_side == "sell" else new_short_tp
                current_price = self.exchange.get_current_price(symbol)

                if (order_side == "sell" and current_price >= new_tp_price) or (order_side == "buy" and current_price <= new_tp_price):
                    # If the current price has surpassed the new TP price, use a normal limit order
                    try:
                        self.exchange.create_normal_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, new_tp_price, positionIdx=positionIdx, reduce_only=True)
                        logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price} using a normal limit order")
                    except Exception as e:
                        logging.info(f"Failed to set new {order_side} TP for {symbol} using a normal limit order. Error: {e}")
                else:
                    # If the current price hasn't surpassed the new TP price, use a post-only order
                    try:
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, new_tp_price, positionIdx=positionIdx, reduce_only=True)
                        logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price} using a post-only order")
                    except Exception as e:
                        logging.info(f"Failed to set new {order_side} TP for {symbol} using a post-only order. Error: {e}")
            else:
                logging.info(f"Skipping TP update as a TP order already exists for {symbol}")

            # Calculate and return the next update time
            return self.calculate_next_update_time()
        else:
            logging.info(f"No immediate update needed for TP orders for {symbol}. Last update at: {last_tp_update}")
            return last_tp_update

    def calculate_quickscalp_long_take_profit_dynamic_distance(self, long_pos_price, symbol, min_upnl_profit_pct, max_upnl_profit_pct):
        if long_pos_price is None or long_pos_price <= 0:
            return None, None

        price_precision = int(self.exchange.get_price_precision(symbol))
        logging.info(f"Price precision for {symbol}: {price_precision}")

        # Calculate the minimum and maximum target profit prices
        min_target_profit_price = Decimal(long_pos_price) * (1 + Decimal(min_upnl_profit_pct))
        max_target_profit_price = Decimal(long_pos_price) * (1 + Decimal(max_upnl_profit_pct))

        # Quantize the target profit prices
        try:
            min_target_profit_price = min_target_profit_price.quantize(
                Decimal('1e-{}'.format(price_precision)), rounding=ROUND_HALF_UP
            )
            max_target_profit_price = max_target_profit_price.quantize(
                Decimal('1e-{}'.format(price_precision)), rounding=ROUND_HALF_UP
            )
        except InvalidOperation as e:
            logging.info(f"Error when quantizing target_profit_prices. {e}")
            return None, None

        # Return the minimum and maximum target profit prices as a tuple
        return float(min_target_profit_price), float(max_target_profit_price)

    def calculate_quickscalp_short_take_profit_dynamic_distance(self, short_pos_price, symbol, min_upnl_profit_pct, max_upnl_profit_pct):
        if short_pos_price is None or short_pos_price <= 0:
            return None, None

        price_precision = int(self.exchange.get_price_precision(symbol))
        logging.info(f"Price precision for {symbol}: {price_precision}")

        # Calculate the minimum and maximum target profit prices
        min_target_profit_price = Decimal(short_pos_price) * (1 - Decimal(min_upnl_profit_pct))
        max_target_profit_price = Decimal(short_pos_price) * (1 - Decimal(max_upnl_profit_pct))

        # Quantize the target profit prices
        try:
            min_target_profit_price = min_target_profit_price.quantize(
                Decimal('1e-{}'.format(price_precision)), rounding=ROUND_HALF_UP
            )
            max_target_profit_price = max_target_profit_price.quantize(
                Decimal('1e-{}'.format(price_precision)), rounding=ROUND_HALF_UP
            )
        except InvalidOperation as e:
            logging.info(f"Error when quantizing target_profit_prices. {e}")
            return None, None

        # Return the minimum and maximum target profit prices as a tuple
        return float(min_target_profit_price), float(max_target_profit_price)
    
    def calculate_quickscalp_long_take_profit(self, long_pos_price, symbol, upnl_profit_pct):
        if long_pos_price is None:
            return None

        price_precision = int(self.exchange.get_price_precision(symbol))
        logging.info(f"Price precision for {symbol}: {price_precision}")

        # Calculate the target profit price
        target_profit_price = Decimal(long_pos_price) * (1 + Decimal(upnl_profit_pct))
        
        # Quantize the target profit price
        try:
            target_profit_price = target_profit_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_UP
            )
        except InvalidOperation as e:
            logging.info(f"Error when quantizing target_profit_price. {e}")
            return None

        return float(target_profit_price)

    def calculate_quickscalp_short_take_profit(self, short_pos_price, symbol, upnl_profit_pct):
        if short_pos_price is None:
            return None

        price_precision = int(self.exchange.get_price_precision(symbol))
        logging.info(f"Price precision for {symbol}: {price_precision}")

        # Calculate the target profit price
        target_profit_price = Decimal(short_pos_price) * (1 - Decimal(upnl_profit_pct))
        
        # Quantize the target profit price
        try:
            target_profit_price = target_profit_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_UP
            )
        except InvalidOperation as e:
            logging.info(f"Error when quantizing target_profit_price. {e}")
            return None

        return float(target_profit_price)
    
# price_precision, qty_precision = self.exchange.get_symbol_precision_bybit(symbol)
    def calculate_dynamic_long_take_profit(self, best_bid_price, long_pos_price, symbol, upnl_profit_pct, max_deviation_pct=0.0040):
        if long_pos_price is None:
            logging.info("Long position price is None for symbol: " + symbol)
            return None

        _, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
        logging.info(f"Price precision for {symbol}: {price_precision}")

        original_tp = long_pos_price * (1 + upnl_profit_pct)
        logging.info(f"Original long TP for {symbol}: {original_tp}")

        bid_walls, ask_walls = self.detect_significant_order_book_walls_atr(symbol)
        if not ask_walls:
            logging.info(f"No significant ask walls found for {symbol}")

        adjusted_tp = original_tp
        for price, size in ask_walls:
            if price > original_tp:
                extended_tp = price - float(price_precision)
                if extended_tp > 0:
                    adjusted_tp = max(adjusted_tp, extended_tp)
                    logging.info(f"Adjusted long TP for {symbol} based on ask wall: {adjusted_tp}")
                break

        # Check if the adjusted TP is within the allowed deviation from the original TP
        if adjusted_tp > original_tp * (1 + max_deviation_pct):
            logging.info(f"Adjusted long TP for {symbol} exceeds the allowed deviation. Reverting to original TP: {original_tp}")
            adjusted_tp = original_tp

        # Adjust TP to best bid price if surpassed
        if best_bid_price >= adjusted_tp:
            adjusted_tp = best_bid_price
            logging.info(f"TP surpassed, adjusted to best bid price for {symbol}: {adjusted_tp}")

        rounded_tp = round(adjusted_tp, len(str(price_precision).split('.')[-1]))
        logging.info(f"Final rounded long TP for {symbol}: {rounded_tp}")
        return rounded_tp

    def calculate_dynamic_short_take_profit(self, best_ask_price, short_pos_price, symbol, upnl_profit_pct, max_deviation_pct=0.05):
        if short_pos_price is None:
            logging.info("Short position price is None for symbol: " + symbol)
            return None

        _, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
        logging.info(f"Price precision for {symbol}: {price_precision}")

        original_tp = short_pos_price * (1 - upnl_profit_pct)
        logging.info(f"Original short TP for {symbol}: {original_tp}")

        bid_walls, ask_walls = self.detect_significant_order_book_walls_atr(symbol)
        if not bid_walls:
            logging.info(f"No significant bid walls found for {symbol}")

        adjusted_tp = original_tp
        for price, size in bid_walls:
            if price < original_tp:
                extended_tp = price + float(price_precision)
                if extended_tp > 0:
                    adjusted_tp = min(adjusted_tp, extended_tp)
                    logging.info(f"Adjusted short TP for {symbol} based on bid wall: {adjusted_tp}")
                break

        # Check if the adjusted TP is within the allowed deviation from the original TP
        if adjusted_tp < original_tp * (1 - max_deviation_pct):
            logging.info(f"Adjusted short TP for {symbol} exceeds the allowed deviation. Reverting to original TP: {original_tp}")
            adjusted_tp = original_tp

        # Adjust TP to best ask price if surpassed
        if best_ask_price <= adjusted_tp:
            adjusted_tp = best_ask_price
            logging.info(f"TP surpassed, adjusted to best ask price for {symbol}: {adjusted_tp}")

        rounded_tp = round(adjusted_tp, len(str(price_precision).split('.')[-1]))
        logging.info(f"Final rounded short TP for {symbol}: {rounded_tp}")
        return rounded_tp
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
        
    def detect_significant_order_book_walls_atr(self, symbol, timeframe='1h', base_threshold_factor=5.0, atr_proximity_percentage=10.0):
        order_book = self.exchange.get_orderbook(symbol)
        bids, asks = order_book['bids'], order_book['asks']

        # Ensure there are enough orders to perform analysis
        if len(bids) < 20 or len(asks) < 20:
            logging.info("Not enough data in the order book to detect walls.")
            return [], []

        # Calculate ATR for market volatility
        historical_data = self.fetch_historical_data(symbol, timeframe)
        atr = self.calculate_atr(historical_data)
        
        # Calculate dynamic threshold based on ATR
        dynamic_threshold = base_threshold_factor * atr

        # Calculate average order size for a larger sample of orders
        sample_size = 20  # Increased sample size
        avg_bid_size = sum([bid[1] for bid in bids[:sample_size]]) / sample_size
        avg_ask_size = sum([ask[1] for ask in asks[:sample_size]]) / sample_size

        # Current market price
        current_price = self.exchange.get_current_price(symbol)

        # Calculate proximity threshold as a percentage of the current price
        proximity_threshold = (atr_proximity_percentage / 100) * current_price

        # Function to check wall significance
        def is_wall_significant(price, size, threshold, avg_size):
            return size > max(threshold, avg_size * base_threshold_factor) and abs(price - current_price) <= proximity_threshold

        # Detect significant bid and ask walls
        significant_bid_walls = [(price, size) for price, size in bids if is_wall_significant(price, size, dynamic_threshold, avg_bid_size)]
        significant_ask_walls = [(price, size) for price, size in asks if is_wall_significant(price, size, dynamic_threshold, avg_ask_size)]

        logging.info(f"Significant bid walls: {significant_bid_walls} for {symbol}")
        logging.info(f"Significant ask walls: {significant_ask_walls} for {symbol}")

        return significant_bid_walls, significant_ask_walls

    def is_price_approaching_wall(self, current_price, wall_price, wall_type):
        # Define a relative proximity threshold, e.g., 0.5%
        proximity_percentage = 0.005  # 0.5%

        # Calculate the proximity threshold in price units
        proximity_threshold = wall_price * proximity_percentage

        # Check if current price is within the threshold of the wall price
        if wall_type == 'bid' and current_price >= wall_price - proximity_threshold:
            # Price is approaching a bid wall
            return True
        elif wall_type == 'ask' and current_price <= wall_price + proximity_threshold:
            # Price is approaching an ask wall
            return True

        return False

    def calculate_trading_fee(self, qty, executed_price, fee_rate=TAKER_FEE_RATE):
        order_value = qty / executed_price
        trading_fee = order_value * fee_rate
        return trading_fee

    def calculate_orderbook_strength(self, symbol, depth=10):
        order_book = self.exchange.get_orderbook(symbol)
        
        top_bids = order_book['bids'][:depth]
        total_bid_quantity = sum([bid[1] for bid in top_bids])
        
        top_asks = order_book['asks'][:depth]
        total_ask_quantity = sum([ask[1] for ask in top_asks])
        
        if (total_bid_quantity + total_ask_quantity) == 0:
            return 0.5  # Neutral strength
        
        strength = total_bid_quantity / (total_bid_quantity + total_ask_quantity)
        
        return strength

    def initialize_symbol(self, symbol, total_equity, best_ask_price, max_leverage):
        with self.initialized_symbols_lock:
            if symbol not in self.initialized_symbols:
                self.initialize_trade_quantities(symbol, total_equity, best_ask_price, max_leverage)
                logging.info(f"Initialized quantities for {symbol}. Initial long qty: {self.initial_max_long_trade_qty_per_symbol.get(symbol, 'N/A')}, Initial short qty: {self.initial_max_short_trade_qty_per_symbol.get(symbol, 'N/A')}")
                self.initialized_symbols.add(symbol)
                return True
            else:
                logging.info(f"{symbol} is already initialized.")
                return False

    def adjust_risk_parameters_qstrend(self, exchange_max_leverage):
        """
        Adjust risk parameters based on user preferences.
        
        :param exchange_max_leverage: The maximum leverage allowed by the exchange.
        """
        # Ensure the wallet exposure limit is within a practical range (1% to 100%)
        # self.wallet_exposure_limit = max(0.01, min(self.wallet_exposure_limit, 1.0))
        
        # Ensure the wallet exposure limit is within a practical range (0.1% to 100%)
        self.wallet_exposure_limit = min(self.wallet_exposure_limit, 1.0)

        # Adjust user-defined leverage for long and short positions to not exceed exchange maximum
        self.user_defined_leverage_long = max(1, min(self.user_defined_leverage_long, exchange_max_leverage))
        self.user_defined_leverage_short = max(1, min(self.user_defined_leverage_short, exchange_max_leverage))
        
        logging.info(f"Wallet exposure limit set to {self.wallet_exposure_limit*100}%")
        logging.info(f"User-defined leverage for long positions set to {self.user_defined_leverage_long}x")
        logging.info(f"User-defined leverage for short positions set to {self.user_defined_leverage_short}x")

    def adjust_risk_parameters(self, exchange_max_leverage):
        """
        Adjust risk parameters based on user preferences.
        
        :param exchange_max_leverage: The maximum leverage allowed by the exchange.
        """
        # Ensure the wallet exposure limit is within a practical range (0.1% to 100%)
        self.wallet_exposure_limit = min(self.wallet_exposure_limit, 1.0)

        # Check if user-defined leverage is zero and adjust to use exchange's maximum if true
        if self.user_defined_leverage_long == 0:
            self.user_defined_leverage_long = exchange_max_leverage
        else:
            # Otherwise, ensure it's between 1 and the exchange maximum
            self.user_defined_leverage_long = max(1, min(self.user_defined_leverage_long, exchange_max_leverage))

        if self.user_defined_leverage_short == 0:
            self.user_defined_leverage_short = exchange_max_leverage
        else:
            # Otherwise, ensure it's between 1 and the exchange maximum
            self.user_defined_leverage_short = max(1, min(self.user_defined_leverage_short, exchange_max_leverage))
        
        logging.info(f"Wallet exposure limit set to {self.wallet_exposure_limit*100}%")
        logging.info(f"User-defined leverage for long positions set to {self.user_defined_leverage_long}x")
        logging.info(f"User-defined leverage for short positions set to {self.user_defined_leverage_short}x")

    # Handle the calculation of trade quantities per symbol
    def handle_trade_quantities(self, symbol, total_equity, best_ask_price):
        if symbol not in self.initialized_symbols:
            max_trade_qty = self.calculate_max_trade_qty(symbol, total_equity, best_ask_price)
            self.max_trade_qty_per_symbol[symbol] = max_trade_qty
            self.initialized_symbols.add(symbol)
            logging.info(f"Symbol {symbol} initialization: Max Trade Qty: {max_trade_qty}, Total Equity: {total_equity}, Best Ask Price: {best_ask_price}")

        dynamic_amount = self.calculate_dynamic_amount(symbol, total_equity, best_ask_price)
        self.dynamic_amount_per_symbol[symbol] = dynamic_amount
        logging.info(f"Dynamic Amount Updated: Symbol: {symbol}, Dynamic Amount: {dynamic_amount}")

    # Calculate maximum trade quantity for a symbol
    def calculate_max_trade_qty(self, symbol, total_equity, best_ask_price, max_leverage):
        leveraged_equity = total_equity * max_leverage
        max_trade_qty = (self.dynamic_amount_multiplier * leveraged_equity) / best_ask_price
        logging.info(f"Calculating Max Trade Qty: Symbol: {symbol}, Leveraged Equity: {leveraged_equity}, Max Trade Qty: {max_trade_qty}")
        return max_trade_qty

    
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

    def can_proceed_with_trade_funding(self, symbol: str) -> dict:
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

    def bybit_hedge_placetp_maker(self, symbol, pos_qty, take_profit_price, positionIdx, order_side, open_orders):
        logging.info(f"TP maker function Trying to place TP for {symbol}")
        existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
        logging.info(f"Existing TP from TP maker functions: {existing_tps}")
        total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
        logging.info(f"TP maker function Existing {order_side} TPs: {existing_tps}")

        if not math.isclose(total_existing_tp_qty, pos_qty):
            try:
                for qty, existing_tp_id in existing_tps:
                    if not math.isclose(qty, pos_qty) and existing_tp_id not in self.auto_reduce_order_ids.get(symbol, []):
                        self.exchange.cancel_order_by_id(existing_tp_id, symbol)
                        logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                        time.sleep(0.05)
            except Exception as e:
                logging.info(f"Error in cancelling {order_side} TP orders {e}")

        if len(existing_tps) < 1:
            try:
                # Use postonly_limit_order_bybit function to place take profit order
                tp_order = self.postonly_limit_order_bybit_nolimit(symbol, order_side, pos_qty, take_profit_price, positionIdx, reduceOnly=True)
                if tp_order and 'id' in tp_order:
                    logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price} with ID {tp_order['id']}")
                else:
                    logging.warning(f"Failed to place {order_side} take profit for {symbol}")
                time.sleep(0.05)
            except Exception as e:
                logging.info(f"Error in placing {order_side} TP: {e}")

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
      
    def place_hedge_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        """Places a hedge order and updates the hedging status."""
        order = self.place_postonly_order_bybit(symbol, side, amount, price, positionIdx, reduceOnly)
        if order and 'id' in order:
            self.update_hedged_status(symbol, True)
            logging.info(f"Hedge order placed for {symbol}: {order['id']}")
        else:
            logging.warning(f"Failed to place hedge order for {symbol}")
        return order

    def place_postonly_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        current_thread_id = threading.get_ident()  # Get the current thread ID
        logging.info(f"[Thread ID: {current_thread_id}] Attempting to place post-only order for {symbol}. Side: {side}, Amount: {amount}, Price: {price}, PositionIdx: {positionIdx}, ReduceOnly: {reduceOnly}")

        if not self.can_place_order(symbol):
            logging.warning(f"[Thread ID: {current_thread_id}] Order placement rate limit exceeded for {symbol}. Skipping...")
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

    def create_normal_stop_loss_order_bybit(self, symbol, order_type, side, amount, price=None, positionIdx=None, reduce_only=True):
        """
        Create a stop-loss order using a limit order on Bybit.

        Parameters:
        - symbol: Trading pair symbol (e.g., 'BTC/USDT').
        - order_type: The type of order, should be 'limit'.
        - side: 'buy' for closing a short position, 'sell' for closing a long position.
        - amount: The amount/quantity for the stop-loss order.
        - price: The stop-loss price.
        - positionIdx: Position index for Bybit (1 for long, 2 for short).
        - reduce_only: Flag to make the order reduce-only, default is True.
        """
        logging.info(f"Calling create_normal_stop_loss_order_bybit with symbol={symbol}, order_type={order_type}, side={side}, amount={amount}, price={price}")

        if positionIdx is None:
            raise ValueError("positionIdx must be specified (1 for long, 2 for short)")

        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")

            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")

            params = {"reduceOnly": reduce_only}
            return self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

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

    def limit_order_bybit_reduce_nolimit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        logging.info(f"Placing {side} limit order for {symbol} at {price} with qty {amount} and params {params}...")
        try:
            order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
            logging.info(f"Order result: {order}")
            if order is None:
                logging.warning(f"Order result is None for {side} limit order on {symbol}")
            return order
        except Exception as e:
            logging.info(f"Error placing order: {str(e)}")
            logging.exception("Stack trace for error in placing order:")
            return None
            
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
            logging.info(f"Error placing order: {str(e)}")
            logging.exception("Stack trace for error in placing order:")  # This will log the full stack trace

    def limit_order_bybit_nolimit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly, "postOnly": False}
        logging.info(f"Placing {side} limit order for {symbol} at {price} with qty {amount} and params {params}...")
        try:
            order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
            logging.info(f"Nolimit postonly order result for {symbol}: {order}")
            if order is None:
                logging.warning(f"Order result is None for {side} limit order on {symbol}")
            return order
        except Exception as e:
            logging.info(f"Error placing order: {str(e)}")
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
            logging.info(f"Error placing order: {str(e)}")
            logging.exception("Stack trace for error in placing order:")  # This will log the full stack trace

    def limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order


    def limit_order_bybit_unified(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit_unified(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

    def limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

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
            logging.info(f"Error in updating take profit: {e}")

    def update_dynamic_quickscalp_tp(self, symbol, best_ask_price, best_bid_price, pos_qty, upnl_profit_pct, short_pos_price, long_pos_price, positionIdx, order_side, last_tp_update, tp_order_counts, max_retries=10):
        # Fetch the current open TP orders and TP order counts for the symbol
        long_tp_orders, short_tp_orders = self.exchange.get_open_tp_orders(symbol)

        long_tp_count = tp_order_counts['long_tp_count']
        short_tp_count = tp_order_counts['short_tp_count']

        # Calculate the new TP values using quickscalp method w/ dynamic
        new_short_tp = self.calculate_dynamic_short_take_profit(
            best_ask_price,
            short_pos_price,
            symbol,
            upnl_profit_pct
        )

        new_long_tp = self.calculate_dynamic_long_take_profit(
            best_bid_price,
            long_pos_price,
            symbol,
            upnl_profit_pct
        )

        # Determine the relevant TP orders based on the order side
        relevant_tp_orders = long_tp_orders if order_side == "sell" else short_tp_orders

        # Check if there's an existing TP order with a mismatched quantity
        mismatched_qty_orders = [order for order in relevant_tp_orders if order['qty'] != pos_qty]

        # Cancel mismatched TP orders if any
        for order in mismatched_qty_orders:
            try:
                self.exchange.cancel_order_by_id(order['id'], symbol)
                logging.info(f"Cancelled TP order {order['id']} for update.")
                time.sleep(0.05)
            except Exception as e:
                logging.info(f"Error in cancelling {order_side} TP order {order['id']}. Error: {e}")

        now = datetime.now()
        if now >= last_tp_update or mismatched_qty_orders:
            # Check if a TP order already exists
            tp_order_exists = (order_side == "sell" and long_tp_count > 0) or (order_side == "buy" and short_tp_count > 0)

            # Set new TP order with updated prices only if no TP order exists
            if not tp_order_exists:
                new_tp_price = new_long_tp if order_side == "sell" else new_short_tp
                try:
                    self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, new_tp_price, positionIdx=positionIdx, reduce_only=True)
                    logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price}")
                except Exception as e:
                    logging.info(f"Failed to set new {order_side} TP for {symbol}. Error: {e}")
            else:
                logging.info(f"Skipping TP update as a TP order already exists for {symbol}")

            # Calculate and return the next update time
            return self.calculate_next_update_time()
        else:
            logging.info(f"No immediate update needed for TP orders for {symbol}. Last update at: {last_tp_update}")
            return last_tp_update


    def place_long_tp_order(self, symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders):
        try:
            if long_pos_qty > 0:  # Check if long position quantity is greater than 0
                tp_order_counts = self.exchange.get_open_tp_order_count(symbol)
                logging.info(f"Long TP order counts for {symbol}: {tp_order_counts}")
                if tp_order_counts['long_tp_count'] == 0:
                    if long_pos_price is not None and best_ask_price is not None and long_pos_price >= long_take_profit:
                        long_take_profit = best_ask_price
                        logging.info(f"Adjusted long TP to current bid price for {symbol}: {long_take_profit}")
                    if long_take_profit is not None:
                        logging.info(f"Placing long TP order for {symbol} at {long_take_profit} with {long_pos_qty}")
                        self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)
        except Exception as e:
            logging.info(f"Exception caught in placing long TP order for {symbol}: {e}")

    def place_short_tp_order(self, symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders):
        try:
            if short_pos_qty > 0:  # Check if short position quantity is greater than 0
                tp_order_counts = self.exchange.get_open_tp_order_count(symbol)
                logging.info(f"Short TP order counts for {symbol}: {tp_order_counts}")
                if tp_order_counts['short_tp_count'] == 0:
                    if short_pos_price is not None and best_bid_price is not None and short_pos_price <= short_take_profit:
                        short_take_profit = best_bid_price
                        logging.info(f"Adjusted short TP to current ask price for {symbol}: {short_take_profit}")
                    if short_take_profit is not None:
                        logging.info(f"Placing short TP order for {symbol} at {short_take_profit} with {short_pos_qty}")
                        self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)
        except Exception as e:
            logging.info(f"Exception caught in placing short TP order for {symbol}: {e}")
            
    def entry_order_exists(self, open_orders, side):
        for order in open_orders:
            # Assuming order details include 'info' which contains the API's raw response
            if order.get("info", {}).get("orderLinkId", "").startswith("helperOrder"):
                continue  # Skip helper orders based on their unique identifier
            
            if order["side"].lower() == side and not order.get("reduceOnly", False):
                logging.info(f"An entry order for side {side} already exists.")
                return True
        
        logging.info(f"No entry order found for side {side}, excluding helper orders.")
        return False

    def calculate_dynamic_amounts_notional_nowelimit(self, symbol, total_equity, best_ask_price, best_bid_price, max_retries=20):
        market_data = self.get_market_data_with_retry(symbol, max_retries=max_retries, retry_delay=5)

        # Log the market data for debugging
        logging.info(f"Market data for {symbol}: {market_data}")

        try:
            min_qty = float(market_data.get('min_qty', 1.0))  # Use min_qty directly
            qty_precision = float(market_data.get('precision', 1))  # Use precision if available

            # Determine the minimum notional value based on the symbol
            if symbol in ["BTCUSDT", "BTC-PERP"]:
                min_notional_value = 100  # Minimum value in USDT
            elif symbol in ["ETHUSDT", "ETH-PERP"] or "BTC" in symbol or "ETH" in symbol:
                min_notional_value = 20  # Minimum value in USDT
            else:
                min_notional_value = 5  # Minimum value in USDT for other contracts

            # Ensure values are non-zero and valid
            if total_equity <= 0 or best_ask_price <= 0 or best_bid_price <= 0:
                logging.warning(f"Invalid values detected: total_equity={total_equity}, best_ask_price={best_ask_price}, best_bid_price={best_bid_price}")
                return 0, 0  # Return zero values for long and short entry sizes

            # Calculate dynamic entry sizes without amplifying by wallet exposure limit
            long_entry_size = max(total_equity / best_ask_price, min_qty)
            short_entry_size = max(total_equity / best_bid_price, min_qty)

            # Ensure the adjusted entry sizes meet the minimum notional value requirement
            long_notional = long_entry_size * best_ask_price
            short_notional = short_entry_size * best_bid_price

            if long_notional < min_notional_value:
                long_entry_size = min_notional_value / best_ask_price

            if short_notional < min_notional_value:
                short_entry_size = min_notional_value / best_bid_price

            # Adjust for precision
            long_entry_size = round(long_entry_size / qty_precision) * qty_precision
            short_entry_size = round(short_entry_size / qty_precision) * qty_precision

            # Ensure quantities respect the minimum quantity requirement
            long_entry_size = max(min_qty, round(long_entry_size))
            short_entry_size = max(min_qty, round(short_entry_size))

            logging.info(f"Calculated long entry size for {symbol}: {long_entry_size} units")
            logging.info(f"Calculated short entry size for {symbol}: {short_entry_size} units")

            return long_entry_size, short_entry_size
        except (TypeError, ValueError) as e:
            logging.error(f"Error occurred: {e}")
            return 0, 0

    def calculate_dynamic_amounts_notional(self, symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit_long, wallet_exposure_limit_short, max_retries=20):
        market_data = self.get_market_data_with_retry(symbol, max_retries=max_retries, retry_delay=5)

        # Log the market data for debugging
        logging.info(f"Market data for {symbol}: {market_data}")

        try:
            min_qty = float(market_data.get('min_qty', 1.0))  # Use min_qty directly
            qty_precision = float(market_data.get('precision', 1))  # Use precision if available

            # Determine the minimum notional value based on the symbol
            if symbol in ["BTCUSDT", "BTC-PERP"]:
                min_notional_value = 100  # Minimum value in USDT
            elif symbol in ["ETHUSDT", "ETH-PERP"] or "BTC" in symbol or "ETH" in symbol:
                min_notional_value = 20  # Minimum value in USDT
            else:
                min_notional_value = 5  # Minimum value in USDT for other contracts

            # Ensure values are non-zero and valid
            if total_equity <= 0 or best_ask_price <= 0 or best_bid_price <= 0:
                logging.warning(f"Invalid values detected: total_equity={total_equity}, best_ask_price={best_ask_price}, best_bid_price={best_bid_price}")
                return 0, 0  # Return zero values for long and short entry sizes

            # Calculate dynamic entry sizes based on risk parameters
            max_equity_for_long_trade = total_equity * wallet_exposure_limit_long
            long_entry_size = max(max_equity_for_long_trade / best_ask_price, min_qty)

            max_equity_for_short_trade = total_equity * wallet_exposure_limit_short
            short_entry_size = max(max_equity_for_short_trade / best_bid_price, min_qty)

            # Ensure the adjusted entry sizes meet the minimum notional value requirement
            long_notional = long_entry_size * best_ask_price
            short_notional = short_entry_size * best_bid_price

            if long_notional < min_notional_value:
                long_entry_size = min_notional_value / best_ask_price

            if short_notional < min_notional_value:
                short_entry_size = min_notional_value / best_bid_price

            # Adjust for precision
            long_entry_size = round(long_entry_size / qty_precision) * qty_precision
            short_entry_size = round(short_entry_size / qty_precision) * qty_precision

            # Ensure quantities respect the minimum quantity requirement
            long_entry_size = max(min_qty, round(long_entry_size))
            short_entry_size = max(min_qty, round(short_entry_size))

            logging.info(f"Calculated long entry size for {symbol}: {long_entry_size} units")
            logging.info(f"Calculated short entry size for {symbol}: {short_entry_size} units")

            return long_entry_size, short_entry_size
        except (TypeError, ValueError) as e:
            logging.error(f"Error occurred: {e}")
            return 0, 0

    def calculate_dynamic_amounts(self, symbol, total_equity, best_ask_price, best_bid_price):
        """
        Calculate the dynamic entry sizes for both long and short positions based on wallet exposure limit and user-defined leverage,
        ensuring compliance with the exchange's minimum trade quantity in USD value.
        
        :param symbol: Trading symbol.
        :param total_equity: Total equity in the wallet.
        :param best_ask_price: Current best ask price of the symbol for buying (long entry).
        :param best_bid_price: Current best bid price of the symbol for selling (short entry).
        :return: A tuple containing entry sizes for long and short trades.
        """
        # Fetch market data to get the minimum trade quantity for the symbol
        market_data = self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)
        min_qty = float(market_data["min_qty"])
        # Simplify by using best_ask_price for min_qty in USD value calculation
        min_qty_usd_value = min_qty * best_ask_price

        # Calculate dynamic entry sizes based on risk parameters
        max_equity_for_long_trade = total_equity * self.wallet_exposure_limit
        max_long_position_value = max_equity_for_long_trade * self.user_defined_leverage_long

        logging.info(f"Max long pos value for {symbol} : {max_long_position_value}")

        long_entry_size = max(max_long_position_value / best_ask_price, min_qty_usd_value / best_ask_price)

        max_equity_for_short_trade = total_equity * self.wallet_exposure_limit
        max_short_position_value = max_equity_for_short_trade * self.user_defined_leverage_short

        logging.info(f"Max short pos value for {symbol} : {max_short_position_value}")
        
        short_entry_size = max(max_short_position_value / best_bid_price, min_qty_usd_value / best_bid_price)

        # Adjusting entry sizes based on the symbol's minimum quantity precision
        qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
        if qty_precision is None:
            long_entry_size_adjusted = round(long_entry_size)
            short_entry_size_adjusted = round(short_entry_size)
        else:
            long_entry_size_adjusted = round(long_entry_size, -int(math.log10(qty_precision)))
            short_entry_size_adjusted = round(short_entry_size, -int(math.log10(qty_precision)))

        logging.info(f"Calculated long entry size for {symbol}: {long_entry_size_adjusted} units")
        logging.info(f"Calculated short entry size for {symbol}: {short_entry_size_adjusted} units")

        return long_entry_size_adjusted, short_entry_size_adjusted

    def calculate_dynamic_amounts_notional_bybit_2(self, symbol, total_equity, best_ask_price, best_bid_price):
        """
        Calculate the dynamic entry sizes for both long and short positions based on wallet exposure limit and user-defined leverage,
        ensuring compliance with the exchange's minimum notional value requirements.
        
        :param symbol: Trading symbol.
        :param total_equity: Total equity in the wallet.
        :param best_ask_price: Current best ask price of the symbol for buying (long entry).
        :param best_bid_price: Current best bid price of the symbol for selling (short entry).
        :return: A tuple containing entry sizes for long and short trades.
        """
        # Fetch the exchange's minimum quantity for the symbol
        min_qty = self.exchange.get_min_qty_bybit(symbol)
        
        # Set the minimum notional value based on the contract type
        if symbol in ["BTCUSDT", "BTC-PERP"]:
            min_notional_value = 100  # $100 for BTCUSDT and BTC-PERP
        elif symbol in ["ETHUSDT", "ETH-PERP"]:
            min_notional_value = 20  # $20 for ETHUSDT and ETH-PERP
        else:
            min_notional_value = 5  # $5 for other USDT and USDC perpetual contracts
        
        # Check if the exchange's minimum quantity meets the minimum notional value requirement
        if min_qty * best_ask_price < min_notional_value:
            min_qty_long = min_notional_value / best_ask_price
        else:
            min_qty_long = min_qty
        
        if min_qty * best_bid_price < min_notional_value:
            min_qty_short = min_notional_value / best_bid_price
        else:
            min_qty_short = min_qty
        
        # Calculate dynamic entry sizes based on risk parameters
        max_equity_for_long_trade = total_equity * self.wallet_exposure_limit
        max_long_position_value = max_equity_for_long_trade * self.user_defined_leverage_long

        logging.info(f"Max long pos value for {symbol} : {max_long_position_value}")

        long_entry_size = max(max_long_position_value / best_ask_price, min_qty_long)

        max_equity_for_short_trade = total_equity * self.wallet_exposure_limit
        max_short_position_value = max_equity_for_short_trade * self.user_defined_leverage_short

        logging.info(f"Max short pos value for {symbol} : {max_short_position_value}")
        
        short_entry_size = max(max_short_position_value / best_bid_price, min_qty_short)

        # Adjusting entry sizes based on the symbol's minimum quantity precision
        qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
        long_entry_size_adjusted = round(long_entry_size, -int(math.log10(qty_precision)))
        short_entry_size_adjusted = round(short_entry_size, -int(math.log10(qty_precision)))

        logging.info(f"Calculated long entry size for {symbol}: {long_entry_size_adjusted} units")
        logging.info(f"Calculated short entry size for {symbol}: {short_entry_size_adjusted} units")

        return long_entry_size_adjusted, short_entry_size_adjusted

    def calculate_dynamic_amounts_notional_bybit(self, symbol, total_equity, best_ask_price, best_bid_price):
        """
        Calculate the dynamic entry sizes for both long and short positions based on wallet exposure limit and user-defined leverage,
        ensuring compliance with the exchange's minimum notional value requirements.
        
        :param symbol: Trading symbol.
        :param total_equity: Total equity in the wallet.
        :param best_ask_price: Current best ask price of the symbol for buying (long entry).
        :param best_bid_price: Current best bid price of the symbol for selling (short entry).
        :return: A tuple containing entry sizes for long and short trades.
        """
        # Set the minimum notional value based on the contract type
        if symbol in ["BTCUSDT", "BTC-PERP"]:
            min_notional_value = 100  # $100 for BTCUSDT and BTC-PERP
        elif symbol in ["ETHUSDT", "ETH-PERP"]:
            min_notional_value = 20  # $20 for ETHUSDT and ETH-PERP
        else:
            min_notional_value = 5  # $5 for other USDT and USDC perpetual contracts
        
        # Calculate dynamic entry sizes based on risk parameters
        max_equity_for_long_trade = total_equity * self.wallet_exposure_limit
        max_long_position_value = max_equity_for_long_trade * self.user_defined_leverage_long

        logging.info(f"Max long pos value for {symbol} : {max_long_position_value}")

        long_entry_size = max(max_long_position_value / best_ask_price, min_notional_value / best_ask_price)

        max_equity_for_short_trade = total_equity * self.wallet_exposure_limit
        max_short_position_value = max_equity_for_short_trade * self.user_defined_leverage_short

        logging.info(f"Max short pos value for {symbol} : {max_short_position_value}")
        
        short_entry_size = max(max_short_position_value / best_bid_price, min_notional_value / best_bid_price)

        # Adjusting entry sizes based on the symbol's minimum quantity precision
        qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
        long_entry_size_adjusted = round(long_entry_size, -int(math.log10(qty_precision)))
        short_entry_size_adjusted = round(short_entry_size, -int(math.log10(qty_precision)))

        logging.info(f"Calculated long entry size for {symbol}: {long_entry_size_adjusted} units")
        logging.info(f"Calculated short entry size for {symbol}: {short_entry_size_adjusted} units")

        return long_entry_size_adjusted, short_entry_size_adjusted

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

    def bybit_1m_mfi_quickscalp_trend_noeri_maxposbal(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, entry_during_autoreduce: bool, volume_check: bool, long_take_profit: float, short_take_profit: float, upnl_profit_pct: float, tp_order_counts: dict, total_equity: float, max_pos_balance_pct: float):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                mfi_signal_long = mfirsi.lower() == "long"
                mfi_signal_short = mfirsi.lower() == "short"

                logging.info(f"MFI signal for {symbol}: Long={mfi_signal_long}, Short={mfi_signal_short}")
                logging.info(f"Volume check for {symbol}: Enabled={volume_check}, One-minute volume={one_minute_volume}, Min volume={min_vol}")

                # Calculate position balances
                long_position_balance = long_pos_price * long_pos_qty if long_pos_price else 0
                short_position_balance = short_pos_price * short_pos_qty if short_pos_price else 0
                long_position_balance_pct = (long_position_balance / total_equity) if total_equity else 0
                short_position_balance_pct = (short_position_balance / total_equity) if total_equity else 0

                # Check if volume check is enabled or not
                if not volume_check or (one_minute_volume > min_vol):
                    if not self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is not active")
                        if long_pos_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing initial long entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            time.sleep(1)
                            if long_pos_qty > 0:
                                logging.info(f"Initial long entry order filled for {symbol}, placing take-profit order")
                                self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                # Update TP for long position
                                self.next_long_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=long_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=1,
                                    order_side="sell",
                                    last_tp_update=self.next_long_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial long entry order not filled for {symbol}")
                        elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                            if long_position_balance_pct < max_pos_balance_pct and (entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False)):
                                logging.info(f"Placing additional long entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                time.sleep(1)
                                if long_pos_qty > 0:
                                    logging.info(f"Additional long entry order filled for {symbol}, placing take-profit order")
                                    self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                    # Update TP for long position
                                    self.next_long_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=long_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=1,
                                        order_side="sell",
                                        last_tp_update=self.next_long_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional long entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional long entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for long entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for long position on {symbol} is active, skipping entry")

                    if not self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is not active")
                        if short_pos_qty == 0 and mfi_signal_short and not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing initial short entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                            time.sleep(1)
                            if short_pos_qty > 0:
                                logging.info(f"Initial short entry order filled for {symbol}, placing take-profit order")
                                self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                # Update TP for short position
                                self.next_short_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=short_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=2,
                                    order_side="buy",
                                    last_tp_update=self.next_short_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial short entry order not filled for {symbol}")
                        elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                            if short_position_balance_pct < max_pos_balance_pct and (entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False)):
                                logging.info(f"Placing additional short entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                time.sleep(1)
                                if short_pos_qty > 0:
                                    logging.info(f"Additional short entry order filled for {symbol}, placing take-profit order")
                                    self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                    # Update TP for short position
                                    self.next_short_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=short_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=2,
                                        order_side="buy",
                                        last_tp_update=self.next_short_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional short entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional short entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for short entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for short position on {symbol} is active, skipping entry")
                else:
                    logging.info(f"Volume check failed for {symbol}, skipping entry")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in quickscalp trend: {e}")


    def bybit_1m_mfi_quickscalp_trend_noeri(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, entry_during_autoreduce: bool, volume_check: bool, long_take_profit: float, short_take_profit: float, upnl_profit_pct: float, tp_order_counts: dict):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                mfi_signal_long = mfirsi.lower() == "long"
                mfi_signal_short = mfirsi.lower() == "short"

                logging.info(f"MFI signal for {symbol}: Long={mfi_signal_long}, Short={mfi_signal_short}")
                logging.info(f"Volume check for {symbol}: Enabled={volume_check}, One-minute volume={one_minute_volume}, Min volume={min_vol}")

                # Check if volume check is enabled or not
                if not volume_check or (one_minute_volume > min_vol):
                    if not self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is not active")
                        if long_pos_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing initial long entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            time.sleep(1)
                            if long_pos_qty > 0:
                                logging.info(f"Initial long entry order filled for {symbol}, placing take-profit order")
                                self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                # Update TP for long position
                                self.next_long_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=long_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=1,
                                    order_side="sell",
                                    last_tp_update=self.next_long_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial long entry order not filled for {symbol}")
                        elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                            if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                                logging.info(f"Placing additional long entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                time.sleep(1)
                                if long_pos_qty > 0:
                                    logging.info(f"Additional long entry order filled for {symbol}, placing take-profit order")
                                    self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                    # Update TP for long position
                                    self.next_long_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=long_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=1,
                                        order_side="sell",
                                        last_tp_update=self.next_long_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional long entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional long entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for long entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for long position on {symbol} is active, skipping entry")

                    if not self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is not active")
                        if short_pos_qty == 0 and mfi_signal_short and not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing initial short entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                            time.sleep(1)
                            if short_pos_qty > 0:
                                logging.info(f"Initial short entry order filled for {symbol}, placing take-profit order")
                                self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                # Update TP for short position
                                self.next_short_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=short_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=2,
                                    order_side="buy",
                                    last_tp_update=self.next_short_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial short entry order not filled for {symbol}")
                        elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                            if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                                logging.info(f"Placing additional short entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                time.sleep(1)
                                if short_pos_qty > 0:
                                    logging.info(f"Additional short entry order filled for {symbol}, placing take-profit order")
                                    self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                    # Update TP for short position
                                    self.next_short_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=short_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=2,
                                        order_side="buy",
                                        last_tp_update=self.next_short_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional short entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional short entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for short entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for short position on {symbol} is active, skipping entry")
                else:
                    logging.info(f"Volume check failed for {symbol}, skipping entry")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in quickscalp trend: {e}")
            
    def bybit_1m_mfi_quickscalp_ema_trend(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, ema_trend: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, entry_during_autoreduce: bool, volume_check: bool, long_take_profit: float, short_take_profit: float, upnl_profit_pct: float, tp_order_counts: dict):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                mfi_signal_long = mfirsi.lower() == "long"
                mfi_signal_short = mfirsi.lower() == "short"

                logging.info(f"MFI signal for {symbol}: Long={mfi_signal_long}, Short={mfi_signal_short}")
                logging.info(f"EMA trend for {symbol}: {ema_trend}")
                logging.info(f"Volume check for {symbol}: Enabled={volume_check}, One-minute volume={one_minute_volume}, Min volume={min_vol}")

                # Check if volume check is enabled or not
                if not volume_check or (one_minute_volume > min_vol):
                    if not self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is not active")
                        if long_pos_qty == 0 and mfi_signal_long and ema_trend == "long" and not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing initial long entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            time.sleep(1)
                            if long_pos_qty > 0:
                                logging.info(f"Initial long entry order filled for {symbol}, placing take-profit order")
                                self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                # Update TP for long position
                                self.next_long_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=long_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=1,
                                    order_side="sell",
                                    last_tp_update=self.next_long_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial long entry order not filled for {symbol}")
                        elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                            if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                                logging.info(f"Placing additional long entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                time.sleep(1)
                                if long_pos_qty > 0:
                                    logging.info(f"Additional long entry order filled for {symbol}, placing take-profit order")
                                    self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                    # Update TP for long position
                                    self.next_long_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=long_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=1,
                                        order_side="sell",
                                        last_tp_update=self.next_long_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional long entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional long entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for long entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for long position on {symbol} is active, skipping entry")

                    if not self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is not active")
                        if short_pos_qty == 0 and mfi_signal_short and ema_trend == "short" and not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing initial short entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                            time.sleep(1)
                            if short_pos_qty > 0:
                                logging.info(f"Initial short entry order filled for {symbol}, placing take-profit order")
                                self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                # Update TP for short position
                                self.next_short_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=short_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=2,
                                    order_side="buy",
                                    last_tp_update=self.next_short_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial short entry order not filled for {symbol}")
                        elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                            if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                                logging.info(f"Placing additional short entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                time.sleep(1)
                                if short_pos_qty > 0:
                                    logging.info(f"Additional short entry order filled for {symbol}, placing take-profit order")
                                    self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                    # Update TP for short position
                                    self.next_short_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=short_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=2,
                                        order_side="buy",
                                        last_tp_update=self.next_short_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional short entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional short entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for short entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for short position on {symbol} is active, skipping entry")
                else:
                    logging.info(f"Volume check failed for {symbol}, skipping entry")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in quickscalp trend: {e}")
            
    def bybit_1m_mfi_quickscalp_trend_eri(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, eri_trend: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, entry_during_autoreduce: bool, volume_check: bool, long_take_profit: float, short_take_profit: float, upnl_profit_pct: float, tp_order_counts: dict):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                mfi_signal_long = mfirsi.lower() == "long"
                mfi_signal_short = mfirsi.lower() == "short"

                logging.info(f"MFI signal for {symbol}: Long={mfi_signal_long}, Short={mfi_signal_short}")
                logging.info(f"ERI trend for {symbol}: {eri_trend}")
                logging.info(f"Volume check for {symbol}: Enabled={volume_check}, One-minute volume={one_minute_volume}, Min volume={min_vol}")

                # Check if volume check is enabled or not
                if not volume_check or (one_minute_volume > min_vol):
                    if not self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is not active")
                        if long_pos_qty == 0 and mfi_signal_long and eri_trend == "bullish" and not self.entry_order_exists(open_orders, "buy"):
                            logging.info(f"Placing initial long entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            time.sleep(1)
                            if long_pos_qty > 0:
                                logging.info(f"Initial long entry order filled for {symbol}, placing take-profit order")
                                self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                # Update TP for long position
                                self.next_long_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=long_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=1,
                                    order_side="sell",
                                    last_tp_update=self.next_long_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial long entry order not filled for {symbol}")
                        elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                            if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                                logging.info(f"Placing additional long entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                time.sleep(1)
                                if long_pos_qty > 0:
                                    logging.info(f"Additional long entry order filled for {symbol}, placing take-profit order")
                                    self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                                    # Update TP for long position
                                    self.next_long_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=long_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=1,
                                        order_side="sell",
                                        last_tp_update=self.next_long_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional long entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional long entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for long entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for long position on {symbol} is active, skipping entry")

                    if not self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is not active")
                        if short_pos_qty == 0 and mfi_signal_short and eri_trend == "bearish" and not self.entry_order_exists(open_orders, "sell"):
                            logging.info(f"Placing initial short entry order for {symbol}")
                            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                            time.sleep(1)
                            if short_pos_qty > 0:
                                logging.info(f"Initial short entry order filled for {symbol}, placing take-profit order")
                                self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                # Update TP for short position
                                self.next_short_tp_update = self.update_quickscalp_tp(
                                    symbol=symbol,
                                    pos_qty=short_pos_qty,
                                    upnl_profit_pct=upnl_profit_pct,
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=2,
                                    order_side="buy",
                                    last_tp_update=self.next_short_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                            else:
                                logging.info(f"Initial short entry order not filled for {symbol}")
                        elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                            if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                                logging.info(f"Placing additional short entry order for {symbol}")
                                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                                time.sleep(1)
                                if short_pos_qty > 0:
                                    logging.info(f"Additional short entry order filled for {symbol}, placing take-profit order")
                                    self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                                    # Update TP for short position
                                    self.next_short_tp_update = self.update_quickscalp_tp(
                                        symbol=symbol,
                                        pos_qty=short_pos_qty,
                                        upnl_profit_pct=upnl_profit_pct,
                                        short_pos_price=short_pos_price,
                                        long_pos_price=long_pos_price,
                                        positionIdx=2,
                                        order_side="buy",
                                        last_tp_update=self.next_short_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                                else:
                                    logging.info(f"Additional short entry order not filled for {symbol}")
                            else:
                                logging.info(f"Skipping additional short entry for {symbol} due to active auto-reduce.")
                        else:
                            logging.info(f"Conditions not met for short entry on {symbol}")
                    else:
                        logging.info(f"Auto-reduce for short position on {symbol} is active, skipping entry")
                else:
                    logging.info(f"Volume check failed for {symbol}, skipping entry")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in quickscalp trend: {e}")
            
    def bybit_1m_mfi_quickscalp_trend_long_only(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, long_pos_qty: float, long_pos_price: float, volume_check: bool, long_take_profit: float, upnl_profit_pct: float, tp_order_counts: dict):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                mfi_signal_long = mfirsi.lower() == "long"

                if not volume_check or (one_minute_volume > min_vol):
                    if long_pos_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(1)
                        if long_pos_qty > 0:
                            self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                            # Update TP for long position
                            self.next_long_tp_update = self.update_quickscalp_tp(
                                symbol=symbol,
                                pos_qty=long_pos_qty,
                                upnl_profit_pct=upnl_profit_pct,
                                long_pos_price=long_pos_price,
                                positionIdx=1,
                                order_side="sell",
                                last_tp_update=self.next_long_tp_update,
                                tp_order_counts=tp_order_counts
                            )
                    elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(1)
                        if long_pos_qty > 0:
                            self.place_long_tp_order(symbol, best_ask_price, long_pos_price, long_pos_qty, long_take_profit, open_orders)
                            # Update TP for long position
                            self.next_long_tp_update = self.update_quickscalp_tp(
                                symbol=symbol,
                                pos_qty=long_pos_qty,
                                upnl_profit_pct=upnl_profit_pct,
                                long_pos_price=long_pos_price,
                                positionIdx=1,
                                order_side="sell",
                                last_tp_update=self.next_long_tp_update,
                                tp_order_counts=tp_order_counts
                            )
                else:
                    logging.info(f"Volume check is disabled or conditions not met for {symbol}, proceeding without volume check.")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in quickscalp trend long only: {e}")

    def bybit_1m_mfi_quickscalp_trend_short_only(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, short_dynamic_amount: float, short_pos_qty: float, short_pos_price: float, volume_check: bool, short_take_profit: float, upnl_profit_pct: float, tp_order_counts: dict):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                mfi_signal_short = mfirsi.lower() == "short"

                if not volume_check or (one_minute_volume > min_vol):
                    if short_pos_qty == 0 and mfi_signal_short and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(1)
                        if short_pos_qty > 0:
                            self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                            # Update TP for short position
                            self.next_short_tp_update = self.update_quickscalp_tp(
                                symbol=symbol,
                                pos_qty=short_pos_qty,
                                upnl_profit_pct=upnl_profit_pct,
                                short_pos_price=short_pos_price,
                                positionIdx=2,
                                order_side="buy",
                                last_tp_update=self.next_short_tp_update,
                                tp_order_counts=tp_order_counts
                            )
                    elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(1)
                        if short_pos_qty > 0:
                            self.place_short_tp_order(symbol, best_bid_price, short_pos_price, short_pos_qty, short_take_profit, open_orders)
                            # Update TP for short position
                            self.next_short_tp_update = self.update_quickscalp_tp(
                                symbol=symbol,
                                pos_qty=short_pos_qty,
                                upnl_profit_pct=upnl_profit_pct,
                                short_pos_price=short_pos_price,
                                positionIdx=2,
                                order_side="buy",
                                last_tp_update=self.next_short_tp_update,
                                tp_order_counts=tp_order_counts
                            )
                else:
                    logging.info(f"Volume check is disabled or conditions not met for {symbol}, proceeding without volume check.")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in quickscalp trend short only: {e}")

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

    def bybit_1m_mfi_quickscalp_trend_long_only_spot(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, spot_position_qty: float, spot_position_price: float, volume_check: bool, upnl_profit_pct: float):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current price for {symbol}: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

                mfi_signal_long = mfirsi.lower() == "long"

                if not volume_check or (one_minute_volume > min_vol):
                    if spot_position_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing spot market buy order for {symbol} with amount {long_dynamic_amount}")
                        self.place_spot_market_order(symbol, "buy", long_dynamic_amount)
                        time.sleep(1)

                    if spot_position_qty > 0:
                        take_profit_price = spot_position_price * (1 + upnl_profit_pct)
                        logging.info(f"Placing spot limit sell order for {symbol} with amount {spot_position_qty} and take profit price {take_profit_price}")
                        self.place_spot_limit_order(symbol, "sell", spot_position_qty, take_profit_price)

                    elif spot_position_qty > 0 and mfi_signal_long and current_price < spot_position_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing spot market buy order to add to existing position for {symbol} with amount {long_dynamic_amount}")
                        self.place_spot_market_order(symbol, "buy", long_dynamic_amount)
                        time.sleep(1)

                else:
                    logging.info(f"Volume check is disabled or conditions not met for {symbol}, proceeding without volume check.")
                    time.sleep(5)

        except Exception as e:
            logging.info(f"Exception caught in bybit_1m_mfi_quickscalp_trend_long_only_spot: {e}")

    def calculate_dynamic_outer_price_distance_atr(self, atrp, min_outer_price_distance, max_outer_price_distance):
        """
        Calculate dynamic outer price distance using scaled ATRP.

        :param atrp: ATRP value as a percentage
        :param min_outer_price_distance: Minimum outer price distance as a percentage
        :param max_outer_price_distance: Maximum outer price distance as a percentage
        :return: Scaled dynamic outer price distance
        """
        # Scale ATRP to the range of min_outer_price_distance and max_outer_price_distance
        # Each 0.1 ATRP corresponds to 1% (0.01) of the outer price distance
        dynamic_distance = atrp / 10.0
        
        # Ensure dynamic distance falls within min and max bounds
        dynamic_distance = max(min(dynamic_distance, max_outer_price_distance), min_outer_price_distance)
        
        logging.info(f"Dynamic outer price distance calculated using scaled ATRP: {dynamic_distance}")
        
        return dynamic_distance
    
    def calculate_dynamic_outer_price_distance_normal(self, min_outer_price_distance: float, max_outer_price_distance: float) -> float:
        """
        Calculate a consistent outer price distance within specified bounds.

        :param min_outer_price_distance: Minimum outer price distance as a percentage
        :param max_outer_price_distance: Maximum outer price distance as a percentage
        :return: Dynamic outer price distance within the specified bounds
        """
        dynamic_distance = (min_outer_price_distance + max_outer_price_distance) / 2.0
        logging.info(f"Dynamic outer price distance calculated: {dynamic_distance}")
        return dynamic_distance


    def calculate_dynamic_outer_price_distance(self, order_book, current_price, max_outer_price_distance):
        # Calculate cumulative volume thresholds for asks and bids
        total_ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
        total_bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
        target_ask_volume = total_ask_volume * 0.1  # Target 10% of total ask volume
        target_bid_volume = total_bid_volume * 0.1  # Target 10% of total bid volume

        # Initialize maximum distance
        max_ask_distance = max_bid_distance = 0

        # Calculate maximum distance for asks
        cumulative_volume = 0
        for ask in order_book['asks']:
            cumulative_volume += float(ask[1])
            if cumulative_volume >= target_ask_volume:
                max_ask_distance = abs(float(ask[0]) - current_price) / current_price
                break

        # Calculate maximum distance for bids
        cumulative_volume = 0
        for bid in order_book['bids']:
            cumulative_volume += float(bid[1])
            if cumulative_volume >= target_bid_volume:
                max_bid_distance = abs(float(bid[0]) - current_price) / current_price
                break

        # Determine the dynamic distance based on the deepest part of the book covered by the target volume
        dynamic_distance = min(max(max_ask_distance, max_bid_distance), max_outer_price_distance)
        return dynamic_distance

    def calculate_dynamic_outer_price_distance_orderbook(self, order_book, current_price, max_outer_price_distance, min_outer_price_distance):
        total_ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
        total_bid_volume = sum(float(bid[1]) for bid in order_book['bids'])
        target_volume = 0.1  # Looking for the 10% volume mark

        ask_distance = next((abs(float(ask[0]) - current_price) / current_price
                            for ask, cum_vol in zip(order_book['asks'], np.cumsum([ask[1] for ask in order_book['asks']]))
                            if cum_vol >= total_ask_volume * target_volume), max_outer_price_distance)

        bid_distance = next((abs(current_price - float(bid[0])) / current_price
                            for bid, cum_vol in zip(order_book['bids'], np.cumsum([bid[1] for bid in order_book['bids']]))
                            if cum_vol >= total_bid_volume * target_volume), max_outer_price_distance)

        dynamic_distance = max(min(max(ask_distance, bid_distance), max_outer_price_distance), min_outer_price_distance)
        logging.info(f"Dynamic outer price distance calculated: {dynamic_distance}")
        return dynamic_distance

    def adjust_distance_based_on_order_book(order_book, target_volume_percent):
        """
        Adjust the outer price distance based on order book to cover a certain percentage of volume.
        """
        total_volume = sum([order['quantity'] for order in order_book['asks']] + [order['quantity'] for order in order_book['bids']])
        cumulative_volume = 0
        target_volume = total_volume * target_volume_percent

        for order in order_book['asks']:
            cumulative_volume += order['quantity']
            if cumulative_volume >= target_volume:
                max_ask_distance = abs(order['price'] - order_book['mid_price']) / order_book['mid_price']
                break

        cumulative_volume = 0
        for order in order_book['bids']:
            cumulative_volume += order['quantity']
            if cumulative_volume >= target_volume:
                max_bid_distance = abs(order['price'] - order_book['mid_price']) / order_book['mid_price']
                break

        return max(max_ask_distance, max_bid_distance)

    def adjust_distance_based_on_momentum(current_price, previous_prices, momentum_threshold=0.05):
        """
        Adjust the outer price distance based on price momentum.
        """
        average_previous_price = sum(previous_prices) / len(previous_prices)
        momentum = (current_price - average_previous_price) / average_previous_price

        if abs(momentum) > momentum_threshold:
            # Increase distance if momentum is strong
            return 0.1  # Example of an increased distance
        return 0.05  # Normal distance

    def calculate_orderbook_based_grid_levels(self, order_book, current_price, long_pos_price, short_pos_price, levels, max_outer_price_distance, min_buffer_percentage, max_buffer_percentage):
        try:
            asks = order_book['asks']
            bids = order_book['bids']

            # Initialize buffer percentages
            buffer_percentage_long = min_buffer_percentage
            buffer_percentage_short = min_buffer_percentage

            # Update buffer percentages if position prices are non-zero
            if long_pos_price:
                buffer_percentage_long += (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            if short_pos_price:
                buffer_percentage_short += (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            # Function to calculate volume-weighted price levels
            def volume_weighted_price(levels, side, buffer_percentage):
                weighted_prices = []
                cumulative_volume = 0
                total_volume = sum(float(level[1]) for level in side)
                volume_thresholds = np.linspace(0.1, 1, levels) * total_volume
                current_index = 0

                for threshold in volume_thresholds:
                    while cumulative_volume < threshold and current_index < len(side):
                        price, volume = float(side[current_index][0]), float(side[current_index][1])
                        cumulative_volume += volume
                        if current_index == 0 or abs(price - current_price) / current_price <= buffer_percentage:
                            weighted_prices.append(price)
                        current_index += 1

                return weighted_prices

            # Calculate weighted prices for asks and bids considering the dynamic buffer
            ask_prices = volume_weighted_price(levels, asks, buffer_percentage_short)
            bid_prices = volume_weighted_price(levels, bids[::-1], buffer_percentage_long)  # Reverse bids for ascending order

            # Determine grid levels within max distance and ensure they are within the max_outer_price_distance
            grid_levels = {
                'long': [p for p in bid_prices if p <= current_price and abs(p - current_price) / current_price <= max_outer_price_distance],
                'short': [p for p in ask_prices if p >= current_price and abs(p - current_price) / current_price <= max_outer_price_distance]
            }

            return grid_levels
        except Exception as e:
            logging.error(f"Error calculating orderbook based grid levels: {e}")
            return {'long': [], 'short': []}


    def get_best_prices(self, symbol, order_book, current_price):
        best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
        best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
        return best_ask_price, best_bid_price

    def calculate_buffers(self, symbol, current_price, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty, initial_entry_buffer_pct, min_buffer_percentage, max_buffer_percentage, order_book):
        # Calculate the average spread between best bid and ask prices
        best_ask_price = float(order_book['asks'][0][0])
        best_bid_price = float(order_book['bids'][0][0])
        average_spread = (best_ask_price - best_bid_price) / current_price

        logging.info(f"Average spread for {symbol}: {average_spread}")

        # Determine buffer percentages dynamically based on the order book
        if long_pos_qty == 0:
            buffer_percentage_long = initial_entry_buffer_pct
        else:
            buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)

        if short_pos_qty == 0:
            buffer_percentage_short = initial_entry_buffer_pct
        else:
            buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

        # Adjust buffer distances using the average spread
        buffer_distance_long = current_price * buffer_percentage_long * average_spread
        buffer_distance_short = current_price * buffer_percentage_short * average_spread

        logging.info(f"[{symbol}] Buffer percentage long: {buffer_percentage_long}, Buffer percentage short: {buffer_percentage_short}")
        logging.info(f"[{symbol}] Buffer distance long: {buffer_distance_long}, Buffer distance short: {buffer_distance_short}")

        return buffer_distance_long, buffer_distance_short

    def calculate_grid_levels_based_on_order_book(self, order_book, current_price, levels, strength, max_outer_price_distance, min_outer_price_distance):
        """
        Calculate grid levels based on order book data while ensuring levels stay within the specified outer price distance bounds.
        """
        dynamic_distance = self.calculate_dynamic_outer_price_distance_orderbook(order_book, current_price, max_outer_price_distance, min_outer_price_distance)

        # Calculate the price range for the grid levels
        outer_price_long = current_price * (1 - dynamic_distance)
        outer_price_short = current_price * (1 + dynamic_distance)
        price_range_long = current_price - outer_price_long
        price_range_short = outer_price_short - current_price

        logging.info(f"Outer price bounds: long={outer_price_long}, short={outer_price_short}")

        # Calculate factors for grid levels
        factors = np.linspace(0.0, 1.0, num=levels) ** strength

        # Calculate grid levels within the bounds
        grid_levels_long = [current_price - price_range_long * factor for factor in factors]
        grid_levels_short = [current_price + price_range_short * factor for factor in factors]

        logging.info(f"Initial grid levels long (price range based): {grid_levels_long}")
        logging.info(f"Initial grid levels short (price range based): {grid_levels_short}")

        # Adjust grid levels to fit within the actual order book volume levels
        total_ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
        total_bid_volume = sum(float(bid[1]) for bid in order_book['bids'])

        cumulative_ask_volumes = np.cumsum([float(ask[1]) for ask in order_book['asks']])
        cumulative_bid_volumes = np.cumsum([float(bid[1]) for bid in order_book['bids']])

        ask_price_levels = [float(ask[0]) for ask in order_book['asks']]
        bid_price_levels = [float(bid[0]) for bid in order_book['bids']]

        grid_levels_long = []
        grid_levels_short = []

        target_volumes_long = np.linspace(0, total_bid_volume, levels) ** strength
        target_volumes_short = np.linspace(0, total_ask_volume, levels) ** strength

        for target_volume in target_volumes_long:
            level_index = np.searchsorted(cumulative_bid_volumes, target_volume)
            if level_index < len(bid_price_levels):
                price = bid_price_levels[level_index]
                if price > outer_price_long:
                    grid_levels_long.append(price)

        for target_volume in target_volumes_short:
            level_index = np.searchsorted(cumulative_ask_volumes, target_volume)
            if level_index < len(ask_price_levels):
                price = ask_price_levels[level_index]
                if price < outer_price_short:
                    grid_levels_short.append(price)

        grid_levels_long = [price for price in grid_levels_long if price > outer_price_long]
        grid_levels_short = [price for price in grid_levels_short if price < outer_price_short]

        # Ensure we have the desired number of levels
        while len(grid_levels_long) < levels:
            last_level = grid_levels_long[-1] if grid_levels_long else outer_price_long
            new_level = last_level - (current_price - outer_price_long) / levels
            if new_level <= outer_price_long:
                break
            grid_levels_long.append(new_level)

        while len(grid_levels_short) < levels:
            last_level = grid_levels_short[-1] if grid_levels_short else outer_price_short
            new_level = last_level + (outer_price_short - current_price) / levels
            if new_level >= outer_price_short:
                break
            grid_levels_short.append(new_level)

        logging.info(f"Cumulative bid volumes: {cumulative_bid_volumes}")
        logging.info(f"Cumulative ask volumes: {cumulative_ask_volumes}")
        logging.info(f"Calculated grid levels long: {grid_levels_long}")
        logging.info(f"Calculated grid levels short: {grid_levels_short}")

        return grid_levels_long, grid_levels_short

    def get_atrp(self, symbol: str, timeframe: str = '1m', period: int = 14, limit: int = 1000) -> float:
        """
        Calculate the Average True Range Percentile (ATRP) for a given symbol and timeframe.
        
        :param symbol: The trading symbol
        :param timeframe: The timeframe for the OHLCV data (e.g., '1m', '5m', '1h')
        :param period: The period for ATR calculation
        :param limit: The number of data points to fetch
        :return: The ATRP value as a percentage of the current price
        """
        # Fetch OHLCV data
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Calculate the True Range (TR)
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        
        # Calculate the Average True Range (ATR)
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate ATRP as a percentage of the closing price
        df['ATRP'] = (df['atr'] / df['close']) * 100
        
        # Return the latest ATRP value
        atrp_value = df['ATRP'].iloc[-1]
        
        return atrp_value

    def calculate_grid_levels_based_on_order_book_atr(self, atrp, order_book, current_price, levels, strength, max_outer_price_distance, min_outer_price_distance):
        """
        Calculate grid levels based on order book data while ensuring levels stay within the specified outer price distance bounds.
        """
        dynamic_distance = self.calculate_dynamic_outer_price_distance_atr(atrp, min_outer_price_distance, max_outer_price_distance)

        logging.info(f"Dynamic distance: {dynamic_distance}")
        
        # Calculate the price range for the grid levels
        outer_price_long = current_price * (1 - dynamic_distance)
        outer_price_short = current_price * (1 + dynamic_distance)
        price_range_long = current_price - outer_price_long
        price_range_short = outer_price_short - current_price

        logging.info(f"Outer price bounds: long={outer_price_long}, short={outer_price_short}")

        # Calculate factors for grid levels
        factors = np.linspace(0.0, 1.0, num=levels) ** strength

        # Calculate grid levels within the bounds
        grid_levels_long = [current_price - price_range_long * factor for factor in factors]
        grid_levels_short = [current_price + price_range_short * factor for factor in factors]

        logging.info(f"Initial grid levels long (price range based): {grid_levels_long}")
        logging.info(f"Initial grid levels short (price range based): {grid_levels_short}")

        # Adjust grid levels to fit within the actual order book volume levels
        total_ask_volume = sum(float(ask[1]) for ask in order_book['asks'])
        total_bid_volume = sum(float(bid[1]) for bid in order_book['bids'])

        cumulative_ask_volumes = np.cumsum([float(ask[1]) for ask in order_book['asks']])
        cumulative_bid_volumes = np.cumsum([float(bid[1]) for bid in order_book['bids']])

        ask_price_levels = [float(ask[0]) for ask in order_book['asks']]
        bid_price_levels = [float(bid[0]) for bid in order_book['bids']]

        grid_levels_long = []
        grid_levels_short = []

        target_volumes_long = np.linspace(0, total_bid_volume, levels) ** strength
        target_volumes_short = np.linspace(0, total_ask_volume, levels) ** strength

        for target_volume in target_volumes_long:
            level_index = np.searchsorted(cumulative_bid_volumes, target_volume)
            if level_index < len(bid_price_levels):
                price = bid_price_levels[level_index]
                if price > outer_price_long:
                    grid_levels_long.append(price)

        for target_volume in target_volumes_short:
            level_index = np.searchsorted(cumulative_ask_volumes, target_volume)
            if level_index < len(ask_price_levels):
                price = ask_price_levels[level_index]
                if price < outer_price_short:
                    grid_levels_short.append(price)

        grid_levels_long = [price for price in grid_levels_long if price > outer_price_long]
        grid_levels_short = [price for price in grid_levels_short if price < outer_price_short]

        # Ensure we have the desired number of levels
        while len(grid_levels_long) < levels:
            last_level = grid_levels_long[-1] if grid_levels_long else outer_price_long
            new_level = last_level - (current_price - outer_price_long) / levels
            if new_level <= outer_price_long:
                break
            grid_levels_long.append(new_level)

        while len(grid_levels_short) < levels:
            last_level = grid_levels_short[-1] if grid_levels_short else outer_price_short
            new_level = last_level + (outer_price_short - current_price) / levels
            if new_level >= outer_price_short:
                break
            grid_levels_short.append(new_level)

        logging.info(f"Cumulative bid volumes: {cumulative_bid_volumes}")
        logging.info(f"Cumulative ask volumes: {cumulative_ask_volumes}")
        logging.info(f"Calculated grid levels long: {grid_levels_long}")
        logging.info(f"Calculated grid levels short: {grid_levels_short}")

        return grid_levels_long, grid_levels_short

    def calculate_grid_levels_orderbook_based_volumes(self, symbol: str, current_price: float, buffer_distance: float, levels: int, side: str, min_outer_price_distance: float, max_outer_price_distance: float, strength: float):
        """
        Calculate grid levels based on the order book, ensuring levels stay within specified bounds.

        :param symbol: The trading symbol
        :param current_price: The current price of the symbol
        :param buffer_distance: The buffer distance from the current price
        :param levels: The number of grid levels
        :param side: The side of the order ('buy' or 'sell')
        :param min_outer_price_distance: Minimum outer price distance as a percentage
        :param max_outer_price_distance: Maximum outer price distance as a percentage
        :param strength: Strength factor for grid level spacing
        :return: A list of grid levels
        """
        logging.info(f"[{symbol}] min_outer_price_distance: {min_outer_price_distance}, max_outer_price_distance: {max_outer_price_distance}")

        if min_outer_price_distance <= 0 or max_outer_price_distance <= 0:
            raise ValueError(f"min_outer_price_distance and max_outer_price_distance must be positive values.")
        if min_outer_price_distance >= max_outer_price_distance:
            raise ValueError(f"min_outer_price_distance must be less than max_outer_price_distance.")

        order_book = self.exchange.get_orderbook(symbol)
        logging.info(f"[{symbol}] Order book fetched for {side}: {order_book}")

        if side == 'buy':
            price_points = [order[0] for order in order_book['bids']]
            volumes = [float(order[1]) for order in order_book['bids']]
        elif side == 'sell':
            price_points = [order[0] for order in order_book['asks']]
            volumes = [float(order[1]) for order in order_book['asks']]

        if not price_points:
            raise ValueError(f"No {side} orders available in the order book for {symbol}")

        logging.info(f"[{symbol}] Price points for {side}: {price_points[:levels]}")
        logging.info(f"[{symbol}] Volumes for {side}: {volumes[:levels]}")

        outer_price_long = current_price * (1 - min_outer_price_distance)
        outer_price_short = current_price * (1 + max_outer_price_distance)
        price_range_long = current_price - outer_price_long
        price_range_short = outer_price_short - current_price

        logging.info(f"[{symbol}] Outer price bounds: long={outer_price_long}, short={outer_price_short}")
        logging.info(f"[{symbol}] Price ranges: long={price_range_long}, short={price_range_short}")

        cumulative_volumes = np.cumsum(volumes)
        total_volume = cumulative_volumes[-1]

        logging.info(f"[{symbol}] Total volume ({side}): {total_volume}")
        logging.info(f"[{symbol}] Cumulative volumes ({side}): {cumulative_volumes}")

        target_volumes = np.linspace(0, total_volume, levels) ** strength
        logging.info(f"[{symbol}] Target volumes ({side}): {target_volumes}")

        adjusted_grid_levels = []
        for target_volume in target_volumes:
            level_index = np.searchsorted(cumulative_volumes, target_volume)
            logging.info(f"[{symbol}] Target volume: {target_volume}, Level index: {level_index}")
            if level_index < len(price_points):
                price = price_points[level_index]
                logging.info(f"[{symbol}] Price at level index {level_index}: {price}")
                if side == 'buy' and outer_price_long <= price <= current_price:
                    adjusted_grid_levels.append(price)
                elif side == 'sell' and current_price <= price <= outer_price_short:
                    adjusted_grid_levels.append(price)

        logging.info(f"[{symbol}] Adjusted grid levels before bounds check ({side}): {adjusted_grid_levels}")

        adjusted_grid_levels = [price for price in adjusted_grid_levels if (price > outer_price_long if side == 'buy' else price < outer_price_short)]
        logging.info(f"[{symbol}] Adjusted grid levels after bounds check ({side}): {adjusted_grid_levels}")

        # Ensure we have the desired number of levels by filling in the gaps if needed
        while len(adjusted_grid_levels) < levels:
            if side == 'buy':
                last_level = adjusted_grid_levels[-1] if adjusted_grid_levels else outer_price_long
                new_level = last_level - (current_price - outer_price_long) / levels
                if new_level <= outer_price_long:
                    break
                adjusted_grid_levels.append(new_level)
            elif side == 'sell':
                last_level = adjusted_grid_levels[-1] if adjusted_grid_levels else outer_price_short
                new_level = last_level + (outer_price_short - current_price) / levels
                if new_level >= outer_price_short:
                    break
                adjusted_grid_levels.append(new_level)

        logging.info(f"[{symbol}] Final adjusted grid levels ({side}): {adjusted_grid_levels}")

        if adjusted_grid_levels:
            if side == 'buy':
                adjusted_grid_levels[0] = current_price - buffer_distance
            elif side == 'sell':
                adjusted_grid_levels[0] = current_price + buffer_distance

        logging.info(f"[{symbol}] Final grid levels with buffer ({side}): {adjusted_grid_levels}")

        return adjusted_grid_levels
    
    def calculate_grid_levels_orderbook_based(self, symbol: str, current_price: float, buffer_distance: float, levels: int, side: str, min_outer_price_distance: float, max_outer_price_distance: float, strength: float):
        """
        Calculate grid levels based on the order book, ensuring levels stay within specified bounds.

        :param symbol: The trading symbol
        :param current_price: The current price of the symbol
        :param buffer_distance: The buffer distance from the current price
        :param levels: The number of grid levels
        :param side: The side of the order ('buy' or 'sell')
        :param min_outer_price_distance: Minimum outer price distance as a percentage
        :param max_outer_price_distance: Maximum outer price distance as a percentage
        :param strength: Strength factor for grid level spacing
        :return: A list of grid levels
        """
        # Fetch the order book data
        order_book = self.exchange.get_orderbook(symbol)

        # Calculate dynamic outer price distance using ATR
        atrp_timeframe = "1m"
        atrp_period = 14
        atrp = self.get_atrp(symbol, timeframe=atrp_timeframe, period=atrp_period)
        dynamic_distance = self.calculate_dynamic_outer_price_distance_atr(atrp, min_outer_price_distance, max_outer_price_distance)

        if side == 'buy':
            price_points = [order[0] for order in order_book['bids'][:levels]]
        elif side == 'sell':
            price_points = [order[0] for order in order_book['asks'][:levels]]

        if not price_points:
            raise ValueError(f"No {side} orders available in the order book for {symbol}")

        # Calculate the price range for the grid levels
        outer_price_long = current_price * (1 - dynamic_distance)
        outer_price_short = current_price * (1 + dynamic_distance)
        price_range_long = current_price - outer_price_long
        price_range_short = outer_price_short - current_price

        logging.info(f"Outer price bounds: long={outer_price_long}, short={outer_price_short}")

        # Calculate factors for grid levels
        factors = np.linspace(0.0, 1.0, num=levels) ** strength

        # Calculate initial grid levels within the bounds, ensuring buffer is applied
        if side == 'buy':
            initial_grid_levels = [current_price - buffer_distance - price_range_long * factor for factor in factors]
        elif side == 'sell':
            initial_grid_levels = [current_price + buffer_distance + price_range_short * factor for factor in factors]

        logging.info(f"Initial grid levels ({side}): {initial_grid_levels}")

        # Adjust grid levels to fit within the actual order book volume levels
        if side == 'buy':
            total_volume = sum(float(bid[1]) for bid in order_book['bids'])
            cumulative_volumes = np.cumsum([float(bid[1]) for bid in order_book['bids']])
            price_levels = [float(bid[0]) for bid in order_book['bids']]
        elif side == 'sell':
            total_volume = sum(float(ask[1]) for ask in order_book['asks'])
            cumulative_volumes = np.cumsum([float(ask[1]) for ask in order_book['asks']])
            price_levels = [float(ask[0]) for ask in order_book['asks']]

        adjusted_grid_levels = []
        target_volumes = np.linspace(0, total_volume, levels) ** strength

        for target_volume in target_volumes:
            level_index = np.searchsorted(cumulative_volumes, target_volume)
            if level_index < len(price_levels):
                price = price_levels[level_index]
                if side == 'buy' and price > outer_price_long:
                    adjusted_grid_levels.append(price)
                elif side == 'sell' and price < outer_price_short:
                    adjusted_grid_levels.append(price)

        adjusted_grid_levels = [price for price in adjusted_grid_levels if (price > outer_price_long if side == 'buy' else price < outer_price_short)]

        # Ensure we have the desired number of levels
        while len(adjusted_grid_levels) < levels:
            if side == 'buy':
                last_level = adjusted_grid_levels[-1] if adjusted_grid_levels else outer_price_long
                new_level = last_level - (current_price - outer_price_long) / levels
                if new_level <= outer_price_long:
                    break
                adjusted_grid_levels.append(new_level)
            elif side == 'sell':
                last_level = adjusted_grid_levels[-1] if adjusted_grid_levels else outer_price_short
                new_level = last_level + (outer_price_short - current_price) / levels
                if new_level >= outer_price_short:
                    break
                adjusted_grid_levels.append(new_level)

        logging.info(f"Cumulative volumes: {cumulative_volumes}")
        logging.info(f"Adjusted grid levels ({side}): {adjusted_grid_levels}")

        # Ensure the first level always uses the buffer distance correctly
        if side == 'buy':
            adjusted_grid_levels[0] = current_price - buffer_distance
        elif side == 'sell':
            adjusted_grid_levels[0] = current_price + buffer_distance

        return adjusted_grid_levels

    def get_30m_candle_spread(self, symbol: str, limit: int = 1) -> float:
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='30m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        spread = df['high'].iloc[0] - df['low'].iloc[0]
        return spread

    # def get_4h_candle_spread(self, symbol: str) -> float:
    #     ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='4h', limit=1)
    #     df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    #     high_low_spread = df['high'].iloc[0] - df['low'].iloc[0]
    #     return high_low_spread
    
    def get_4h_candle_spread(self, symbol: str) -> float:
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='4h', limit=1)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Convert 'high' and 'low' columns to numeric
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        
        high_low_spread = df['high'].iloc[0] - df['low'].iloc[0]
        return high_low_spread

    def linear_grid_hardened_gridspan_orderbook_maxposqty_properdca(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
        max_qty_percent_long: float, max_qty_percent_short: float
    ):
        try:
            # Calculate dynamic outer price distance based on 4h candle spread
            spread = self.get_4h_candle_spread(symbol)
            logging.info(f"4h Candle spread for {symbol}: {spread}")

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Ensure dynamic outer price distance is not too tight
            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))

            logging.info(f"Dynamic outer price distance for {symbol}: {dynamic_outer_price_distance}")

            # Ensure the outer price distance can span all levels
            required_distance = outer_price_distance / levels
            if dynamic_outer_price_distance < required_distance:
                logging.info(f"Dynamic outer price distance {dynamic_outer_price_distance} is less than required distance {required_distance}. Adjusting it.")
                dynamic_outer_price_distance = required_distance

            logging.info(f"Dynamic outer price distance after spread: {dynamic_outer_price_distance}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate the grid levels ensuring they start from the buffer distance and span the dynamic outer price distance
            price_range_long = dynamic_outer_price_distance * current_price
            price_range_short = dynamic_outer_price_distance * current_price

            # Get price levels from the order book within the defined range
            ob_levels_long = [level[0] for level in order_book['bids'] if level[0] >= current_price - price_range_long]
            ob_levels_short = [level[0] for level in order_book['asks'] if level[0] <= current_price + price_range_short]

            # Fill in the remaining levels with linear spacing if there are not enough order book levels
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = ob_levels_long + [current_price - buffer_distance_long - price_range_long * factor for factor in factors[len(ob_levels_long):]]
            grid_levels_short = ob_levels_short + [current_price + buffer_distance_short + price_range_short * factor for factor in factors[len(ob_levels_short):]]

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Check for grid replacement conditions
            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            # Track the last time the grid was emptied
            if replace_empty_long_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            if replace_empty_short_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}

            current_time = time.time()

            # Check and log if the symbol is in max_qty_reached_symbol_long
            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")

            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                dynamic_outer_price_distance=dynamic_outer_price_distance
            )

            # Replace long grid if conditions are met
            if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', 0) > 240))) and not self.auto_reduce_active_long.get(symbol, False):
                if symbol not in self.max_qty_reached_symbol_long:
                    logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid timeout.")
                    self.clear_grid(symbol, 'buy')
                    buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                    buffer_distance_long = current_price * buffer_percentage_long
                    price_range_long = dynamic_outer_price_distance * current_price
                    grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                    self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                    self.active_grids.add(symbol)
                    self.last_empty_grid_time[symbol]['long'] = current_time
                    logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                else:
                    logging.info(f"{symbol} is in max qty reached symbol long cannot replace grid")

            # Replace short grid if conditions are met
            if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', 0) > 240))) and not self.auto_reduce_active_short.get(symbol, False):
                if symbol not in self.max_qty_reached_symbol_short:
                    logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid timeout.")
                    self.clear_grid(symbol, 'sell')
                    buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                    buffer_distance_short = current_price * buffer_percentage_short
                    price_range_short = dynamic_outer_price_distance * current_price
                    grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                    self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                    self.active_grids.add(symbol)
                    self.last_empty_grid_time[symbol]['short'] = current_time
                    logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                else:
                    logging.info(f"{symbol} is in max qty reached symbol short cannot replace grid")
                    
            # Additional logic for managing open symbols and checking trading permissions
            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def linear_grid_hardened_gridspan_orderbook_maxposqty_nosignal_properdca(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
        max_qty_percent_long: float, max_qty_percent_short: float
    ):
        try:
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)

            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            dynamic_outer_price_distance = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)

            outer_price_long = current_price * (1 - dynamic_outer_price_distance)
            outer_price_short = current_price * (1 + dynamic_outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting dynamic outer price distance.")
                adjustment_factor = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - adjustment_factor)
                outer_price_short = current_price * (1 + adjustment_factor)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                dynamic_outer_price_distance=dynamic_outer_price_distance
            )

            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                buffer_distance_long = current_price * buffer_percentage_long
                dynamic_outer_price_distance_long = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)
                outer_price_distance_long = current_price * dynamic_outer_price_distance_long
                grid_levels_long = [current_price - buffer_distance_long - (outer_price_distance_long - buffer_distance_long) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                buffer_distance_short = current_price * buffer_percentage_short
                dynamic_outer_price_distance_short = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)
                outer_price_distance_short = current_price * dynamic_outer_price_distance_short
                grid_levels_short = [current_price + buffer_distance_short + (outer_price_distance_short - buffer_distance_short) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < 2 * symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            else:
                logging.info(f"[{symbol}] Trading not allowed or grid conditions not met. Skipping grid placement.")
                time.sleep(5)

            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )
        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def linear_grid_hardened_gridspan_ob_volumelevels_nosignal(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                                                    short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                                                    strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
                                                    wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
                                                    user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
                                                    short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                                                    symbols_allowed: int, enforce_full_grid: bool, upnl_profit_pct: float,
                                                    max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                                                    max_qty_percent_long: float, max_qty_percent_short: float):
        try:
            # Calculate dynamic outer price distance based on 4h candle spread
            spread = self.get_4h_candle_spread(symbol)
            logging.info(f"4h Candle spread for {symbol}: {spread}")

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Ensure dynamic outer price distance is not too tight
            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))
            logging.info(f"Dynamic outer price distance for {symbol}: {dynamic_outer_price_distance}")

            # Ensure the outer price distance can span all levels
            required_distance = outer_price_distance / levels
            if dynamic_outer_price_distance < required_distance:
                logging.info(f"Dynamic outer price distance {dynamic_outer_price_distance} is less than required distance {required_distance}. Adjusting it.")
                dynamic_outer_price_distance = required_distance
            logging.info(f"Dynamic outer price distance after spread adjustment: {dynamic_outer_price_distance}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Analyze orderbook depth and identify significant price levels
            min_price = current_price - max_outer_price_distance * current_price
            max_price = current_price + max_outer_price_distance * current_price

            # Create a histogram of orderbook volume within the price range
            price_range = np.arange(min_price, max_price, (max_price - min_price) / 100)
            volume_histogram_long = np.zeros_like(price_range)
            volume_histogram_short = np.zeros_like(price_range)

            for order in order_book['bids']:
                price, volume = order[0], order[1]
                if min_price <= price <= current_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_long[index] += volume

            for order in order_book['asks']:
                price, volume = order[0], order[1]
                if current_price <= price <= max_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_short[index] += volume

            # Identify significant price levels based on volume histogram
            volume_threshold_long = np.mean(volume_histogram_long) * 1.5  # Adjust the threshold as needed
            significant_levels_long = price_range[volume_histogram_long >= volume_threshold_long]

            volume_threshold_short = np.mean(volume_histogram_short) * 1.5  # Adjust the threshold as needed
            significant_levels_short = price_range[volume_histogram_short >= volume_threshold_short]

            # Calculate grid levels based on dynamic_outer_price_distance
            grid_levels_long = [current_price - i * dynamic_outer_price_distance * current_price for i in range(1, levels + 1)]
            grid_levels_short = [current_price + i * dynamic_outer_price_distance * current_price for i in range(1, levels + 1)]

            # Function to find the nearest significant level
            def find_nearest_significant_level(level, significant_levels, tolerance, min_distance, max_distance, current_price):
                for sig_level in significant_levels:
                    if abs(level - sig_level) / level < tolerance:
                        if current_price - max_distance * current_price <= sig_level <= current_price - min_distance * current_price:
                            return sig_level
                return level

            # Adjust grid levels to align with significant levels
            tolerance = 0.01  # 1% tolerance, adjust as needed
            grid_levels_long = [
                find_nearest_significant_level(
                    level,
                    significant_levels_long,
                    tolerance,
                    buffer_distance_long / current_price,
                    min_outer_price_distance,
                    current_price
                ) for level in grid_levels_long
            ]
            grid_levels_short = [
                find_nearest_significant_level(
                    level,
                    significant_levels_short,
                    tolerance,
                    buffer_distance_short / current_price,
                    min_outer_price_distance,
                    current_price
                ) for level in grid_levels_short
            ]

            # Ensure the grid levels are within the buffer distances and respect min/max outer price distance
            grid_levels_long = [
                level for level in grid_levels_long
                if current_price - min_outer_price_distance * current_price <= level <= current_price - buffer_distance_long
            ]
            grid_levels_short = [
                level for level in grid_levels_short
                if current_price + buffer_distance_short <= level <= current_price + min_outer_price_distance * current_price
            ]

            # Ensure the desired number of grid levels is achieved
            if len(grid_levels_long) < levels:
                additional_levels_long = np.linspace(
                    current_price - min_outer_price_distance * current_price,
                    current_price - buffer_distance_long,
                    levels - len(grid_levels_long)
                )
                grid_levels_long = np.concatenate((grid_levels_long, additional_levels_long))

            if len(grid_levels_short) < levels:
                additional_levels_short = np.linspace(
                    current_price + buffer_distance_short,
                    current_price + min_outer_price_distance * current_price,
                    levels - len(grid_levels_short)
                )
                grid_levels_short = np.concatenate((grid_levels_short, additional_levels_short))

            # Sort the grid levels in ascending order
            grid_levels_long = sorted(grid_levels_long, reverse=True)
            grid_levels_short = sorted(grid_levels_short)

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Check for grid replacement conditions
            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            logging.info(f"Symbol {symbol} has open long order: {has_open_long_order}")
            logging.info(f"Symbol {symbol} has open short order: {has_open_short_order}")

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            current_time = datetime.now()

            # Track the last time the grid was emptied
            if symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {'long': datetime.min, 'short': datetime.min}

            if replace_empty_long_grid:
                self.last_empty_grid_time[symbol]['long'] = current_time

            if replace_empty_short_grid:
                self.last_empty_grid_time[symbol]['short'] = current_time

            # Check and log if the symbol is in max_qty_reached_symbol_long
            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")

            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            # Additional logic for managing open symbols and checking trading permissions
            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                logging.info(f"Allowed symbol: {symbol}")

                replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                    symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                    dynamic_outer_price_distance=dynamic_outer_price_distance
                )

                # Replace long grid if conditions are met
                if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', datetime.min) > timedelta(seconds=240)))) and not self.auto_reduce_active_long.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_long:
                        logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'buy')
                        buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                        buffer_distance_long = current_price * buffer_percentage_long
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['long'] = current_time
                        logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol long cannot replace grid")

                # Replace short grid if conditions are met
                if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', datetime.min) > timedelta(seconds=240)))) and not self.auto_reduce_active_short.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_short:
                        logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'sell')
                        buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                        buffer_distance_short = current_price * buffer_percentage_short
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['short'] = current_time
                        logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol short cannot replace grid")

                # Ensure orders are not issued twice for the same side
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and not has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and not has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")

            if symbol in open_symbols or trading_allowed:
                # Check if the grid levels have already been placed
                placed_long_levels = self.placed_levels.get(symbol, {}).get("buy", set())
                placed_short_levels = self.placed_levels.get(symbol, {}).get("sell", set())

                # Check if there are open long or short orders
                has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders):
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.clear_grid(symbol, 'buy')
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                            self.placed_levels.setdefault(symbol, {})["buy"] = set(grid_levels_long)  # Update placed levels
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.clear_grid(symbol, 'buy')
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                            self.placed_levels.setdefault(symbol, {})["buy"] = set(grid_levels_long)  # Update placed levels
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders):
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.clear_grid(symbol, 'sell')
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                            self.placed_levels.setdefault(symbol, {})["sell"] = set(grid_levels_short)  # Update placed levels
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.clear_grid(symbol, 'sell')
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                            self.placed_levels.setdefault(symbol, {})["sell"] = set(grid_levels_short)  # Update placed levels
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.clear_grid(symbol, 'buy')
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.clear_grid(symbol, 'sell')
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            time.sleep(5)

        except Exception as e:
            logging.info(f"Error in executing grid strategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())
            
    def linear_grid_hardened_gridspan_ob_volumes_nosignal(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                                                short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                                                strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
                                                wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
                                                user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
                                                short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                                                symbols_allowed: int, enforce_full_grid: bool, upnl_profit_pct: float,
                                                max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                                                max_qty_percent_long: float, max_qty_percent_short: float):
        try:
            # Calculate dynamic outer price distance based on 4h candle spread
            spread = self.get_4h_candle_spread(symbol)
            logging.info(f"4h Candle spread for {symbol}: {spread}")

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Ensure dynamic outer price distance is not too tight
            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))

            logging.info(f"Dynamic outer price distance for {symbol} : {dynamic_outer_price_distance}")

            # Ensure the outer price distance can span all levels
            required_distance = outer_price_distance / levels
            if dynamic_outer_price_distance < required_distance:
                logging.info(f"Dynamic outer price distance {dynamic_outer_price_distance} is less than required distance {required_distance}. Adjusting it.")
                dynamic_outer_price_distance = required_distance

            logging.info(f"Dynamic outer price distance after spread: {dynamic_outer_price_distance}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate the grid levels ensuring they start from the buffer distance and span the dynamic outer price distance
            price_range_long = dynamic_outer_price_distance * current_price
            price_range_short = dynamic_outer_price_distance * current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength

            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Check for grid replacement conditions
            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            # Track the last time the grid was emptied
            if replace_empty_long_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            if replace_empty_short_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}

            current_time = time.time()

            # Check and log if the symbol is in max_qty_reached_symbol_long
            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")

            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                dynamic_outer_price_distance=dynamic_outer_price_distance
            )

            # Replace long grid if conditions are met
            if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', 0) > 240))) and not self.auto_reduce_active_long.get(symbol, False):
                if symbol not in self.max_qty_reached_symbol_long:
                    logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid timeout.")
                    self.clear_grid(symbol, 'buy')
                    buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                    buffer_distance_long = current_price * buffer_percentage_long
                    price_range_long = dynamic_outer_price_distance * current_price
                    grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                    self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                    self.active_grids.add(symbol)
                    self.last_empty_grid_time[symbol]['long'] = current_time
                    logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                else:
                    logging.info(f"{symbol} is in max qty reached symbol long cannot replace grid")

            # Replace short grid if conditions are met
            if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', 0) > 240))) and not self.auto_reduce_active_short.get(symbol, False):
                if symbol not in self.max_qty_reached_symbol_short:
                    logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid timeout.")
                    self.clear_grid(symbol, 'sell')
                    buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                    buffer_distance_short = current_price * buffer_percentage_short
                    price_range_short = dynamic_outer_price_distance * current_price
                    grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                    self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                    self.active_grids.add(symbol)
                    self.last_empty_grid_time[symbol]['short'] = current_time
                    logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                else:
                    logging.info(f"{symbol} is in max qty reached symbol short cannot replace grid")
                    
            # Additional logic for managing open symbols and checking trading permissions
            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and (long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and (long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and (short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and (short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def lingrid_oblevels(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                                                                        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                                                                        strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
                                                                        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
                                                                        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
                                                                        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                                                                        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
                                                                        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                                                                        max_qty_percent_long: float, max_qty_percent_short: float):
        try:
            
            spread = self.get_4h_candle_spread(symbol)
            current_price = self.exchange.get_current_price(symbol)
            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))
            
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.placed_levels:
                self.placed_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.placed_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.placed_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            min_price = current_price - max_outer_price_distance * current_price
            max_price = current_price + max_outer_price_distance * current_price

            price_range = np.arange(min_price, max_price, (max_price - min_price) / 100)
            volume_histogram_long = np.zeros_like(price_range)
            volume_histogram_short = np.zeros_like(price_range)

            for order in order_book['bids']:
                price, volume = order[0], order[1]
                if min_price <= price <= current_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_long[index] += volume

            for order in order_book['asks']:
                price, volume = order[0], order[1]
                if current_price <= price <= max_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_short[index] += volume

            volume_threshold_long = np.mean(volume_histogram_long) * 1.5
            significant_levels_long = price_range[volume_histogram_long >= volume_threshold_long]

            volume_threshold_short = np.mean(volume_histogram_short) * 1.5
            significant_levels_short = price_range[volume_histogram_short >= volume_threshold_short]

            initial_entry_long = current_price - buffer_distance_long
            initial_entry_short = current_price + buffer_distance_short

            if long_pos_qty == 0:
                grid_levels_long = [initial_entry_long] + [
                    current_price - buffer_distance_long - i * ((max_outer_price_distance * current_price - buffer_distance_long) / (levels - 1)) 
                    for i in range(1, levels)
                ]
            else:
                grid_levels_long = [
                    current_price - buffer_distance_long - i * ((max_outer_price_distance * current_price - buffer_distance_long) / levels) 
                    for i in range(levels)
                ]

            if short_pos_qty == 0:
                grid_levels_short = [initial_entry_short] + [
                    current_price + buffer_distance_short + i * ((max_outer_price_distance * current_price - buffer_distance_short) / (levels - 1)) 
                    for i in range(1, levels)
                ]
            else:
                grid_levels_short = [
                    current_price + buffer_distance_short + i * ((max_outer_price_distance * current_price - buffer_distance_short) / levels) 
                    for i in range(levels)
                ]

            def find_nearest_significant_level(level, significant_levels, tolerance, min_distance, max_distance, current_price, previous_level=None):
                for sig_level in significant_levels:
                    if abs(level - sig_level) / level < tolerance:
                        if current_price - max_distance * current_price <= sig_level <= current_price - min_distance * current_price:
                            if previous_level is None or abs(sig_level - previous_level) > (max_distance * current_price / levels):
                                return sig_level
                return level

            tolerance = 0.01
            adjusted_grid_levels_long = []
            adjusted_grid_levels_short = []

            for i, level in enumerate(grid_levels_long):
                previous_level = adjusted_grid_levels_long[-1] if adjusted_grid_levels_long else None
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels_long,
                    tolerance,
                    min_outer_price_distance,
                    max_outer_price_distance,
                    current_price,
                    previous_level
                )
                if current_price - max_outer_price_distance * current_price <= adjusted_level <= current_price - buffer_distance_long:
                    adjusted_grid_levels_long.append(adjusted_level)

            for i, level in enumerate(grid_levels_short):
                previous_level = adjusted_grid_levels_short[-1] if adjusted_grid_levels_short else None
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels_short,
                    tolerance,
                    min_outer_price_distance,
                    max_outer_price_distance,
                    current_price,
                    previous_level
                )
                if current_price + buffer_distance_short <= adjusted_level <= current_price + max_outer_price_distance * current_price:
                    adjusted_grid_levels_short.append(adjusted_level)

            grid_levels_long = adjusted_grid_levels_long
            grid_levels_short = adjusted_grid_levels_short

            if len(grid_levels_long) < levels:
                remaining_levels = levels - len(grid_levels_long)
                additional_levels_long = np.linspace(
                    grid_levels_long[-1] if grid_levels_long else current_price - buffer_distance_long,
                    current_price - max_outer_price_distance * current_price,
                    remaining_levels + 1
                )[1:]
                grid_levels_long = np.concatenate((grid_levels_long, additional_levels_long))

            if len(grid_levels_short) < levels:
                remaining_levels = levels - len(grid_levels_short)
                additional_levels_short = np.linspace(
                    grid_levels_short[-1] if grid_levels_short else current_price + buffer_distance_short,
                    current_price + max_outer_price_distance * current_price,
                    remaining_levels + 1
                )[1:]
                grid_levels_short = np.concatenate((grid_levels_short, additional_levels_short))

            grid_levels_long = sorted(grid_levels_long, reverse=True)
            grid_levels_short = sorted(grid_levels_short)

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')

            if self.auto_reduce_active_long.get(symbol, False):
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)

            if self.auto_reduce_active_short.get(symbol, False):
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)

            replace_empty_long_grid = (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders))
            replace_empty_short_grid = (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders))

            current_time = datetime.now()

            if symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {'long': datetime.min, 'short': datetime.min}

            if replace_empty_long_grid:
                self.last_empty_grid_time[symbol]['long'] = current_time

            if replace_empty_short_grid:
                self.last_empty_grid_time[symbol]['short'] = current_time

            open_symbols = list(set(open_symbols))

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                    symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                    dynamic_outer_price_distance=dynamic_outer_price_distance
                )

                if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', datetime.min) > timedelta(seconds=240)))) and not self.auto_reduce_active_long.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_long:
                        self.clear_grid(symbol, 'buy')
                        buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                        buffer_distance_long = current_price * buffer_percentage_long
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.placed_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['long'] = current_time

                if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', datetime.min) > timedelta(seconds=240)))) and not self.auto_reduce_active_short.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_short:
                        self.clear_grid(symbol, 'sell')
                        buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                        buffer_distance_short = current_price * buffer_percentage_short
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.placed_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['short'] = current_time

                has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.placed_levels[symbol] and not has_open_long_order:
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.placed_levels[symbol]["buy"])
                                self.active_grids.add(symbol)

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.placed_levels[symbol] and not has_open_short_order:
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.placed_levels[symbol]["sell"])
                                self.active_grids.add(symbol)

                if symbol in open_symbols or trading_allowed:
                    if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                        last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                        if current_time - last_cleared > self.clear_interval:
                            self.clear_grid(symbol, 'buy')
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.last_cleared_time[symbol] = current_time

                    if not self.auto_reduce_active_long.get(symbol, False):
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)
                                self.placed_levels[symbol]["buy"].clear()

                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders):
                                self.clear_grid(symbol, 'buy')
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.placed_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                                self.placed_levels.setdefault(symbol, {})["buy"] = set(grid_levels_long)

                    if not self.auto_reduce_active_short.get(symbol, False):
                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)
                                self.placed_levels[symbol]["sell"].clear()

                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders):
                                self.clear_grid(symbol, 'sell')
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.placed_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                                self.placed_levels.setdefault(symbol, {})["sell"] = set(grid_levels_short)

                    if long_mode and mfi_signal_long and not long_pos_qty and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            self.clear_grid(symbol, 'buy')
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.placed_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                            self.placed_levels[symbol]["buy"] = set(grid_levels_long)

                    if short_mode and mfi_signal_short and not short_pos_qty and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            self.clear_grid(symbol, 'sell')
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.placed_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                            self.placed_levels[symbol]["sell"] = set(grid_levels_short)

                time.sleep(2)

        except Exception as e:
            logging.info(f"Error in executing grid strategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def get_position_qty(self, symbol, side):
        # Fetch open position data
        open_position_data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)
        position_details = {}

        # Process the fetched position data
        for position in open_position_data:
            info = position.get('info', {})
            position_symbol = info.get('symbol', '').split(':')[0]

            if 'size' in info and 'side' in info and 'avgPrice' in info and 'liqPrice' in info:
                size = float(info['size'])
                side_type = info['side'].lower()
                avg_price = float(info['avgPrice'])
                liq_price = info.get('liqPrice', None)

                if position_symbol not in position_details:
                    position_details[position_symbol] = {
                        'long': {'qty': 0, 'avg_price': 0, 'liq_price': None}, 
                        'short': {'qty': 0, 'avg_price': 0, 'liq_price': None}
                    }

                if side_type == 'buy':
                    position_details[position_symbol]['long']['qty'] += size
                    position_details[position_symbol]['long']['avg_price'] = avg_price
                    position_details[position_symbol]['long']['liq_price'] = liq_price
                elif side_type == 'sell':
                    position_details[position_symbol]['short']['qty'] += size
                    position_details[position_symbol]['short']['avg_price'] = avg_price
                    position_details[position_symbol]['short']['liq_price'] = liq_price

        if side == 'long':
            return position_details.get(symbol, {}).get('long', {}).get('qty', 0)
        elif side == 'short':
            return position_details.get(symbol, {}).get('short', {}).get('qty', 0)
        return 0

    def lineargrid_base(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                        strength: float, min_outer_price_distance_long: float, min_outer_price_distance_short: float,
                        max_outer_price_distance_long: float, max_outer_price_distance_short: float, reissue_threshold: float,
                        wallet_exposure_limit_long: float, wallet_exposure_limit_short: float, long_mode: bool,
                        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
                        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                        max_qty_percent_long: float, max_qty_percent_short: float, graceful_stop_long: bool, graceful_stop_short: bool,
                        additional_entries_from_signal: bool, open_position_data: list, drawdown_behavior: str, grid_behavior: str,
                        stop_loss_long: float, stop_loss_short: float, stop_loss_enabled: bool):
        try:
            long_pos_qty = long_pos_qty if long_pos_qty is not None else 0
            short_pos_qty = short_pos_qty if short_pos_qty is not None else 0

            # Ensure long_pos_qty and short_pos_qty are floats
            try:
                long_pos_qty = float(long_pos_qty)
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid value for long_pos_qty: {long_pos_qty}, Error: {e}")
                long_pos_qty = 0

            try:
                short_pos_qty = float(short_pos_qty)
            except (ValueError, TypeError) as e:
                logging.error(f"Invalid value for short_pos_qty: {short_pos_qty}, Error: {e}")
                short_pos_qty = 0

            spread, current_price = self.get_spread_and_price(symbol)

            # Calculate dynamic outer price distances separately for long and short
            # dynamic_outer_price_distance_long, dynamic_outer_price_distance_short = self.calculate_dynamic_outer_price_distance(
            #     spread, min_outer_price_distance_long, min_outer_price_distance_short, max_outer_price_distance_long, max_outer_price_distance_short
            # )

            dynamic_outer_price_distance_long, dynamic_outer_price_distance_short = self.calculate_dynamic_outer_price_distance(
                spread, 
                min_outer_price_distance_long, 
                min_outer_price_distance_short, 
                max_outer_price_distance_long, 
                max_outer_price_distance_short
            )


            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            self.initialize_filled_levels(symbol)
            long_grid_active, short_grid_active = self.check_grid_active(symbol, open_orders)

            logging.info(f"Long grid active: {long_grid_active}")
            logging.info(f"Short grid active: {short_grid_active}")

            self.check_and_manage_positions(
                long_pos_qty,
                short_pos_qty,
                symbol,
                total_equity,
                current_price,
                wallet_exposure_limit_long,
                wallet_exposure_limit_short,
                max_qty_percent_long,
                max_qty_percent_short
            )

            buffer_percentage_long, buffer_percentage_short = self.calculate_buffer_percentages(
                long_pos_qty, short_pos_qty, current_price, long_pos_price, short_pos_price,
                initial_entry_buffer_pct, min_buffer_percentage, max_buffer_percentage
            )
            buffer_distance_long, buffer_distance_short = self.calculate_buffer_distances(
                current_price, buffer_percentage_long, buffer_percentage_short
            )

            order_book, best_ask_price, best_bid_price = self.get_order_book_prices(symbol, current_price)

            (min_price_long, max_price_long, price_range_long, volume_histogram_long,
            min_price_short, max_price_short, price_range_short, volume_histogram_short) = self.calculate_price_range_and_volume_histograms(
                order_book, current_price, max_outer_price_distance_long, max_outer_price_distance_short
            )

            # Use the correct price range for each histogram
            volume_threshold_long, significant_levels_long = self.calculate_volume_thresholds_and_significant_levels(volume_histogram_long, price_range_long)
            volume_threshold_short, significant_levels_short = self.calculate_volume_thresholds_and_significant_levels(volume_histogram_short, price_range_short)

            # Call dbscan_classification if grid_behavior is set to "dbscanalgo"
            if grid_behavior == "dbscanalgo":
                initial_entry_long, initial_entry_short = self.calculate_initial_entries(current_price, buffer_distance_long, buffer_distance_short)
                zigzag_length = 5  # Example value; replace with actual config value
                epsilon_deviation = 2.5  # Example value; replace with actual config value
                aggregate_range = 5  # Example value; replace with actual config value
                
                # Extract high, low, and volume data from order book
                highs = [float(order[0]) for order in order_book['bids']]  # Using bids for highs
                lows = [float(order[0]) for order in order_book['asks']]   # Using asks for lows
                volumes = [float(order[1]) for order in order_book['bids'] + order_book['asks']]  # Combining volumes from bids and asks

                # Create OHLCV format expected by dbscan_classification
                ohlcv_data = [{'high': high, 'low': low, 'volume': volume} for high, low, volume in zip(highs, lows, volumes)]
                
                # dbscan_classification call
                support_resistance_levels = self.dbscan_classification(ohlcv_data, zigzag_length, epsilon_deviation, aggregate_range)
                initial_entry_long, initial_entry_short = self.calculate_initial_entries(current_price, buffer_distance_long, buffer_distance_short)

                # Extract levels from the dbscan classification results
                grid_levels_long = [level['level'] for level in support_resistance_levels if level['level'] >= current_price]
                grid_levels_short = [level['level'] for level in support_resistance_levels if level['level'] <= current_price]

            else:
                initial_entry_long, initial_entry_short = self.calculate_initial_entries(current_price, buffer_distance_long, buffer_distance_short)
                grid_levels_long, grid_levels_short = self.calculate_grid_levels(
                    long_pos_qty, short_pos_qty, levels, 
                    initial_entry_long, initial_entry_short, 
                    current_price, 
                    buffer_distance_long, buffer_distance_short, 
                    max_outer_price_distance_long, max_outer_price_distance_short
                )

            # adjusted_grid_levels_long = self.adjust_grid_levels(
            #     grid_levels_long,
            #     significant_levels_long,
            #     tolerance=0.01,
            #     min_outer_price_distance_long=min_outer_price_distance_long,
            #     min_outer_price_distance_short=min_outer_price_distance_short,
            #     max_outer_price_distance_long=max_outer_price_distance_long,
            #     max_outer_price_distance_short=max_outer_price_distance_short,
            #     current_price=current_price,
            #     levels=levels
            # )

            # adjusted_grid_levels_short = self.adjust_grid_levels(
            #     grid_levels_short,
            #     significant_levels_short,
            #     tolerance=0.01,
            #     min_outer_price_distance_long=min_outer_price_distance_long,
            #     min_outer_price_distance_short=min_outer_price_distance_short,
            #     max_outer_price_distance_long=max_outer_price_distance_long,
            #     max_outer_price_distance_short=max_outer_price_distance_short,
            #     current_price=current_price,
            #     levels=levels
            # )

            adjusted_grid_levels_long = self.adjust_grid_levels(
                grid_levels_long,
                significant_levels_long,
                tolerance=0.01,
                min_outer_price_distance_long=min_outer_price_distance_long,
                min_outer_price_distance_short=min_outer_price_distance_short,
                max_outer_price_distance_long=max_outer_price_distance_long,
                max_outer_price_distance_short=max_outer_price_distance_short,
                current_price=current_price,
                levels=levels
            )

            adjusted_grid_levels_short = self.adjust_grid_levels(
                grid_levels_short,
                significant_levels_short,
                tolerance=0.01,
                min_outer_price_distance_long=min_outer_price_distance_long,
                min_outer_price_distance_short=min_outer_price_distance_short,
                max_outer_price_distance_long=max_outer_price_distance_long,
                max_outer_price_distance_short=max_outer_price_distance_short,
                current_price=current_price,
                levels=levels
            )


            grid_levels_long, grid_levels_short = self.finalize_grid_levels(
                adjusted_grid_levels_long, adjusted_grid_levels_short, 
                levels, current_price, 
                buffer_distance_long, buffer_distance_short, 
                max_outer_price_distance_long, max_outer_price_distance_short, 
                initial_entry_long, initial_entry_short
            )

            qty_precision, min_qty = self.get_precision_and_min_qty(symbol)

            # Apply drawdown behavior based on configuration
            if drawdown_behavior == "full_distribution":
                logging.info(f"Activating full distribution drawdown behavior for {symbol}")

                # Calculate order amounts for aggressive drawdown with strength
                amounts_long = self.calculate_order_amounts_aggressive_drawdown(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='buy', strength=strength, long_pos_qty=long_pos_qty
                )
                amounts_short = self.calculate_order_amounts_aggressive_drawdown(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='sell', strength=strength, short_pos_qty=short_pos_qty
                )

            elif drawdown_behavior == "progressive_drawdown":
                logging.info(f"Activating progressive drawdown behavior for {symbol}")

                # Calculate order amounts for progressive drawdown using progressive distribution
                amounts_long = self.calculate_order_amounts_progressive_distribution(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='buy', strength=strength, long_pos_qty=long_pos_qty
                )
                amounts_short = self.calculate_order_amounts_progressive_distribution(
                    symbol, total_equity, best_ask_price, best_bid_price,
                    wallet_exposure_limit_long, wallet_exposure_limit_short,
                    levels, qty_precision, side='sell', strength=strength, short_pos_qty=short_pos_qty
                )

            else:
                logging.info(f"Applying standard grid behavior for {symbol}")

                # Calculate the total amounts without aggressive or progressive drawdown behaviors
                total_amount_long = self.calculate_total_amount_refactor(
                    symbol,
                    total_equity,
                    best_ask_price,
                    best_bid_price,
                    wallet_exposure_limit_long,
                    wallet_exposure_limit_short,
                    "buy",
                    levels,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    long_mode
                )

                total_amount_short = self.calculate_total_amount_refactor(
                    symbol,
                    total_equity,
                    best_ask_price,
                    best_bid_price,
                    wallet_exposure_limit_long,
                    wallet_exposure_limit_short,
                    "sell",
                    levels,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    short_mode
                )

                amounts_long = self.calculate_order_amounts_refactor(
                    symbol,
                    total_amount_long,
                    levels,
                    strength,
                    qty_precision,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    'buy'
                )

                amounts_short = self.calculate_order_amounts_refactor(
                    symbol,
                    total_amount_short,
                    levels,
                    strength,
                    qty_precision,
                    enforce_full_grid,
                    long_pos_qty,
                    short_pos_qty,
                    'sell'
                )

            self.handle_grid_trades(
                symbol,
                grid_levels_long,
                grid_levels_short,
                long_grid_active,
                short_grid_active,
                long_pos_qty,
                short_pos_qty,
                current_price,
                dynamic_outer_price_distance_long,
                dynamic_outer_price_distance_short,
                min_outer_price_distance_long,
                min_outer_price_distance_short,
                buffer_percentage_long,
                buffer_percentage_short,
                adjusted_grid_levels_long,
                adjusted_grid_levels_short,
                levels,
                amounts_long,
                amounts_short,
                best_bid_price,
                best_ask_price,
                mfirsi_signal,
                open_orders,
                initial_entry_buffer_pct,
                reissue_threshold,
                entry_during_autoreduce,
                min_qty,
                open_symbols,
                symbols_allowed,
                long_mode,
                short_mode,
                long_pos_price,
                short_pos_price,
                graceful_stop_long,
                graceful_stop_short,
                min_buffer_percentage,
                max_buffer_percentage,
                additional_entries_from_signal,
                open_position_data,
                upnl_profit_pct,
                max_upnl_profit_pct,
                tp_order_counts,
                stop_loss_long,
                stop_loss_short,
                stop_loss_enabled
            )

        except Exception as e:
            logging.error(f"Error in executing gridstrategy: {e}")
            logging.error("Traceback: %s", traceback.format_exc())



    def get_spread_and_price(self, symbol):
        spread = self.get_4h_candle_spread(symbol)
        logging.info(f"4h Candle spread for {symbol}: {spread}")

        current_price = self.exchange.get_current_price(symbol)
        logging.info(f"[{symbol}] Current price: {current_price}")

        return spread, current_price

    # def calculate_dynamic_outer_price_distance(self, spread, min_outer_price_distance, max_outer_price_distance_long, max_outer_price_distance_short):
    #     dynamic_outer_price_distance_long = max(min_outer_price_distance, min(max_outer_price_distance_long, spread))
    #     dynamic_outer_price_distance_short = max(min_outer_price_distance, min(max_outer_price_distance_short, spread))
        
    #     logging.info(f"Dynamic outer price distance (Long): {dynamic_outer_price_distance_long}")
    #     logging.info(f"Dynamic outer price distance (Short): {dynamic_outer_price_distance_short}")
        
    #     return dynamic_outer_price_distance_long, dynamic_outer_price_distance_short

    def calculate_dynamic_outer_price_distance(self, spread: float, 
                                            min_outer_price_distance_long: float, 
                                            min_outer_price_distance_short: float, 
                                            max_outer_price_distance_long: float, 
                                            max_outer_price_distance_short: float):
        dynamic_outer_price_distance_long = max(min_outer_price_distance_long, 
                                                min(max_outer_price_distance_long, spread))
        dynamic_outer_price_distance_short = max(min_outer_price_distance_short, 
                                                min(max_outer_price_distance_short, spread))
        
        logging.info(f"Dynamic outer price distance (Long): {dynamic_outer_price_distance_long}")
        logging.info(f"Dynamic outer price distance (Short): {dynamic_outer_price_distance_short}")
        
        return dynamic_outer_price_distance_long, dynamic_outer_price_distance_short


    def initialize_filled_levels(self, symbol):
        if symbol not in self.filled_levels:
            self.filled_levels[symbol] = {"buy": set(), "sell": set()}

    def check_grid_active(self, symbol, open_orders):
        logging.info(f"Checking grid active for {symbol}")
        #logging.info(f"Open orders test: {open_orders}")
        # Check for grid replacement conditions
        has_open_long_order = any(order['info']['symbol'] == symbol and order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
        has_open_short_order = any(order['info']['symbol'] == symbol and order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

        logging.info(f"{symbol} Has open long order check_grid_active: {has_open_long_order}")
        logging.info(f"{symbol} Has open short order check_grid_active: {has_open_short_order}")

        long_grid_active = symbol in self.active_long_grids and has_open_long_order
        short_grid_active = symbol in self.active_short_grids and has_open_short_order
        return long_grid_active, short_grid_active

    def calculate_buffer_percentages(self, long_pos_qty, short_pos_qty, current_price, long_pos_price, short_pos_price, initial_entry_buffer_pct, min_buffer_percentage, max_buffer_percentage):
        buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
        buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
        return buffer_percentage_long, buffer_percentage_short

    def calculate_buffer_distances(self, current_price, buffer_percentage_long, buffer_percentage_short):
        buffer_distance_long = current_price * buffer_percentage_long
        buffer_distance_short = current_price * buffer_percentage_short
        logging.info(f"Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")
        return buffer_distance_long, buffer_distance_short

    def get_order_book_prices(self, symbol, current_price):
        order_book = self.exchange.get_orderbook(symbol)
        best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
        best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
        return order_book, best_ask_price, best_bid_price

    def calculate_price_range_and_volume_histograms(self, order_book, current_price, max_outer_price_distance_long, max_outer_price_distance_short):
        try:
            # Ensure current_price is a float
            current_price = float(current_price)
            
            # Calculate min and max prices using respective max outer price distances
            min_price_long = current_price - max_outer_price_distance_long * current_price
            max_price_long = current_price + max_outer_price_distance_long * current_price

            min_price_short = current_price - max_outer_price_distance_short * current_price
            max_price_short = current_price + max_outer_price_distance_short * current_price

            # Create price ranges
            price_range_long = np.arange(min_price_long, max_price_long, (max_price_long - min_price_long) / 100)
            price_range_short = np.arange(min_price_short, max_price_short, (max_price_short - min_price_short) / 100)

            volume_histogram_long = np.zeros_like(price_range_long)
            volume_histogram_short = np.zeros_like(price_range_short)

            # Populate volume histogram for long positions
            for order in order_book['bids']:
                price, volume = float(order[0]), float(order[1])  # Convert price and volume to float
                if min_price_long <= price <= current_price:
                    index = int((price - min_price_long) / (max_price_long - min_price_long) * 100)
                    volume_histogram_long[index] += volume

            # Populate volume histogram for short positions
            for order in order_book['asks']:
                price, volume = float(order[0]), float(order[1])  # Convert price and volume to float
                if current_price <= price <= max_price_short:
                    index = int((price - min_price_short) / (max_price_short - min_price_short) * 100)
                    volume_histogram_short[index] += volume

            return min_price_long, max_price_long, price_range_long, volume_histogram_long, min_price_short, max_price_short, price_range_short, volume_histogram_short
        except Exception as e:
            logging.info(f"Exception in calculate_price_range_and_volume_histograms: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def calculate_volume_thresholds_and_significant_levels(self, volume_histogram, price_range):
        volume_threshold = np.mean(volume_histogram) * 1.5
        significant_levels = price_range[volume_histogram >= volume_threshold]
        return volume_threshold, significant_levels

    def calculate_initial_entries(self, current_price, buffer_distance_long, buffer_distance_short):
        initial_entry_long = current_price - buffer_distance_long
        initial_entry_short = current_price + buffer_distance_short
        return initial_entry_long, initial_entry_short
    
    def calculate_grid_levels(self, long_pos_qty, short_pos_qty, levels, initial_entry_long, initial_entry_short, 
                            current_price, buffer_distance_long, buffer_distance_short, 
                            max_outer_price_distance_long, max_outer_price_distance_short):
        if long_pos_qty == 0:
            grid_levels_long = [initial_entry_long] + [
                current_price - buffer_distance_long - i * ((max_outer_price_distance_long * current_price - buffer_distance_long) / (levels - 1)) 
                for i in range(1, levels)
            ]
        else:
            grid_levels_long = [
                current_price - buffer_distance_long - i * ((max_outer_price_distance_long * current_price - buffer_distance_long) / levels) 
                for i in range(levels)
            ]

        if short_pos_qty == 0:
            grid_levels_short = [initial_entry_short] + [
                current_price + buffer_distance_short + i * ((max_outer_price_distance_short * current_price - buffer_distance_short) / (levels - 1)) 
                for i in range(1, levels)
            ]
        else:
            grid_levels_short = [
                current_price + buffer_distance_short + i * ((max_outer_price_distance_short * current_price - buffer_distance_short) / levels) 
                for i in range(levels)
            ]
        
        return grid_levels_long, grid_levels_short

    # def adjust_grid_levels(self, grid_levels, significant_levels, tolerance, min_outer_price_distance, max_outer_price_distance_long, max_outer_price_distance_short, current_price, levels):
    #     def find_nearest_significant_level(level, significant_levels, tolerance, min_distance, max_distance, current_price, previous_level=None):
    #         for sig_level in significant_levels:
    #             if abs(level - sig_level) / level < tolerance:
    #                 if current_price - max_distance * current_price <= sig_level <= current_price - min_distance * current_price:
    #                     if previous_level is None or abs(sig_level - previous_level) > (max_distance * current_price / levels):
    #                         return sig_level
    #         return level

    #     adjusted_grid_levels = []

    #     for i, level in enumerate(grid_levels):
    #         previous_level = adjusted_grid_levels[-1] if adjusted_grid_levels else None
            
    #         # Choose the appropriate max distance based on the direction of the grid
    #         if level <= current_price:  # Long grid
    #             adjusted_level = find_nearest_significant_level(
    #                 level,
    #                 significant_levels,
    #                 tolerance,
    #                 min_outer_price_distance,
    #                 max_outer_price_distance_long,  # Use long max distance
    #                 current_price,
    #                 previous_level
    #             )
    #         else:  # Short grid
    #             adjusted_level = find_nearest_significant_level(
    #                 level,
    #                 significant_levels,
    #                 tolerance,
    #                 min_outer_price_distance,
    #                 max_outer_price_distance_short,  # Use short max distance
    #                 current_price,
    #                 previous_level
    #             )
            
    #         adjusted_grid_levels.append(adjusted_level)

    #     return adjusted_grid_levels

    def adjust_grid_levels(self, grid_levels, significant_levels, tolerance,
                        min_outer_price_distance_long, min_outer_price_distance_short,
                        max_outer_price_distance_long, max_outer_price_distance_short,
                        current_price, levels):
        def find_nearest_significant_level(level, significant_levels, tolerance, min_distance, max_distance, current_price, previous_level=None):
            for sig_level in significant_levels:
                # Check if significant level is within the tolerance of the current level
                if abs(level - sig_level) / level < tolerance:
                    # Determine whether we are dealing with a long side (below current price) or short side (above current price)
                    if level <= current_price:
                        # Long side levels (below current price)
                        # Allowed range: current_price*(1 - max_distance) <= sig_level <= current_price*(1 - min_distance)
                        lower_bound = current_price * (1 - max_distance)
                        upper_bound = current_price * (1 - min_distance)
                    else:
                        # Short side levels (above current price)
                        # Allowed range: current_price*(1 + min_distance) <= sig_level <= current_price*(1 + max_distance)
                        lower_bound = current_price * (1 + min_distance)
                        upper_bound = current_price * (1 + max_distance)

                    if lower_bound <= sig_level <= upper_bound:
                        # Check spacing from previous adjusted level
                        if previous_level is None or abs(sig_level - previous_level) > (max_distance * current_price / levels):
                            return sig_level
            return level

        adjusted_grid_levels = []

        for i, level in enumerate(grid_levels):
            previous_level = adjusted_grid_levels[-1] if adjusted_grid_levels else None

            # Choose the appropriate min/max distances based on whether this is a long or short level
            if level <= current_price:
                # Long grid
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels,
                    tolerance,
                    min_outer_price_distance_long,
                    max_outer_price_distance_long,
                    current_price,
                    previous_level
                )
            else:
                # Short grid
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels,
                    tolerance,
                    min_outer_price_distance_short,
                    max_outer_price_distance_short,
                    current_price,
                    previous_level
                )

            adjusted_grid_levels.append(adjusted_level)

        return adjusted_grid_levels

    def finalize_grid_levels(self, adjusted_grid_levels_long, adjusted_grid_levels_short, levels, current_price, buffer_distance_long, buffer_distance_short, max_outer_price_distance_long, max_outer_price_distance_short, initial_entry_long, initial_entry_short):
        if len(adjusted_grid_levels_long) < levels:
            remaining_levels = levels - len(adjusted_grid_levels_long)
            additional_levels_long = np.linspace(
                adjusted_grid_levels_long[-1] if adjusted_grid_levels_long else current_price - buffer_distance_long,
                current_price - max_outer_price_distance_long * current_price,
                remaining_levels + 1
            )[1:]
            adjusted_grid_levels_long = np.concatenate((adjusted_grid_levels_long, additional_levels_long))

        if len(adjusted_grid_levels_short) < levels:
            remaining_levels = levels - len(adjusted_grid_levels_short)
            additional_levels_short = np.linspace(
                adjusted_grid_levels_short[-1] if adjusted_grid_levels_short else current_price + buffer_distance_short,
                current_price + max_outer_price_distance_short * current_price,
                remaining_levels + 1
            )[1:]
            adjusted_grid_levels_short = np.concatenate((adjusted_grid_levels_short, additional_levels_short))

        adjusted_grid_levels_long = sorted(adjusted_grid_levels_long, reverse=True)
        adjusted_grid_levels_short = sorted(adjusted_grid_levels_short)

        logging.info(f"Initial long entry level: {initial_entry_long}")
        logging.info(f"Initial short entry level: {initial_entry_short}")
        logging.info(f"Long grid levels: {adjusted_grid_levels_long}")
        logging.info(f"Short grid levels: {adjusted_grid_levels_short}")

        return adjusted_grid_levels_long, adjusted_grid_levels_short


    def get_precision_and_min_qty(self, symbol):
        qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
        min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
        logging.info(f"Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")
        return qty_precision, min_qty

    def calculate_total_amount_refactor(self, symbol, total_equity, best_ask_price, best_bid_price, 
                                        wallet_exposure_limit_long, wallet_exposure_limit_short, 
                                        side, levels, enforce_full_grid, 
                                        long_pos_qty, short_pos_qty, mode):
        if not mode:
            return 0
        return self.calculate_total_amount_notional_ls_properdca(
            symbol=symbol, 
            total_equity=total_equity,
            best_ask_price=best_ask_price,
            best_bid_price=best_bid_price, 
            wallet_exposure_limit_long=wallet_exposure_limit_long,
            wallet_exposure_limit_short=wallet_exposure_limit_short, 
            side=side, 
            levels=levels,
            enforce_full_grid=enforce_full_grid, 
            long_pos_qty=long_pos_qty, 
            short_pos_qty=short_pos_qty
        )

    def calculate_order_amounts_refactor(self, symbol, total_amount, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side):
        return self.calculate_order_amounts_notional_properdca(
            symbol, total_amount, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side
        )

    def issue_reduce_only_order(self, exchange, symbol, side, position_qty, stop_loss_pct, current_price, positionIdx):
        """
        Issue a reduce-only limit order for a stop loss based on the current price and stop loss percentage.

        Parameters:
        - exchange: ccxt exchange instance.
        - symbol: Trading pair symbol (e.g., 'BTC/USDT').
        - side: 'sell' for a long position stop loss, 'buy' for a short position stop loss.
        - position_qty: Quantity of the position to reduce.
        - stop_loss_pct: Percentage of price action at which the stop loss should trigger.
        - current_price: The current market price of the asset.
        - positionIdx: Position index for Bybit (1 for long, 2 for short).
        """

        try:
            # Calculate the stop loss price based on the current price and the stop loss percentage
            if side == 'sell':
                stop_loss_price = current_price * (1 - stop_loss_pct / 100)
            else:  # side == 'buy'
                stop_loss_price = current_price * (1 + stop_loss_pct / 100)

            logging.info(f"[{symbol}] Attempting to place a reduce-only {side} order at {stop_loss_price} for {position_qty} {symbol} with positionIdx {positionIdx}")

            # Retry logic for placing the reduce-only limit order
            max_retries = 5
            retry_delay = 5  # seconds
            for attempt in range(max_retries):
                try:
                    # Place the reduce-only limit order
                    order = self.place_reduce_only_limit_order(exchange, symbol, side, position_qty, stop_loss_price, positionIdx)
                    logging.info(f"[{symbol}] Stop loss order issued on attempt {attempt + 1}: {order}")
                    
                    # Check if the order was successful and break out of the retry loop
                    if order:
                        return order
                except Exception as e:
                    logging.error(f"[{symbol}] Failed to place reduce-only stop loss order on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        logging.info(f"[{symbol}] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"[{symbol}] Exhausted retries for placing reduce-only stop loss order.")
                        raise e

        except Exception as e:
            logging.error(f"[{symbol}] Failed to issue reduce-only stop loss order: {e}")
            raise

    def trigger_stop_loss(self, symbol, pos_qty, side, stop_loss_price, best_price):
        try:
            logging.info(f"[{symbol}] {side.capitalize()} position hit stop-loss at {stop_loss_price}. Issuing reduce-only limit order.")
            
            retry_counter = 0
            max_retries = 50  # Maximum retries

            while pos_qty > 0.00001 and retry_counter < max_retries:
                best_price = self.get_best_bid_price(symbol) if side == 'long' else self.get_best_ask_price(symbol)

                self.create_normal_stop_loss_order_bybit(
                    symbol, order_type='limit', side='sell' if side == 'long' else 'buy', 
                    amount=pos_qty, price=best_price, positionIdx=1 if side == 'long' else 2, reduce_only=True
                )

                logging.info(f"[{symbol}] Issued stop-loss order to {side} {pos_qty} at {best_price}")
                time.sleep(5)

                pos_qty = self.get_position_qty(symbol, side)
                retry_counter += 1

            if pos_qty <= 0.00001:
                self.clear_grid(symbol, 'buy' if side == 'long' else 'sell')
                logging.info(f"[{symbol}] {side.capitalize()} position fully closed at stop-loss.")

        except Exception as e:
            logging.error(f"[{symbol}] Error in trigger_stop_loss for {side}: {e}")

    def has_active_grid(self, symbol, side, open_orders):
        grid_set = self.active_long_grids if side == 'long' else self.active_short_grids
        has_open_order = any(order['info']['symbol'] == symbol and 
                            order['info']['side'].lower() == ('buy' if side == 'long' else 'sell') and 
                            not order['info']['reduceOnly'] for order in open_orders)
        return symbol in grid_set or has_open_order

    # def handle_grid_trades(self, symbol, grid_levels_long, grid_levels_short, long_grid_active, short_grid_active,
    #                     long_pos_qty, short_pos_qty, current_price, dynamic_outer_price_distance_long, dynamic_outer_price_distance_short, min_outer_price_distance, 
    #                     buffer_percentage_long, buffer_percentage_short, adjusted_grid_levels_long, adjusted_grid_levels_short, levels, amounts_long, amounts_short, 
    #                     best_bid_price, best_ask_price, mfirsi_signal, open_orders, initial_entry_buffer_pct, 
    #                     reissue_threshold, entry_during_autoreduce, min_qty, open_symbols, symbols_allowed, long_mode, 
    #                     short_mode, long_pos_price, short_pos_price, graceful_stop_long, graceful_stop_short, 
    #                     min_buffer_percentage, max_buffer_percentage, additional_entries_from_signal, 
    #                     open_position_data, upnl_profit_pct, max_upnl_profit_pct, tp_order_counts, 
    #                     stop_loss_long, stop_loss_short, stop_loss_enabled=True):

    #     try:
    #         # Determine if there is an open long or short position
    #         has_open_long_position = long_pos_qty > 0
    #         has_open_short_position = short_pos_qty > 0

    #         # Store previous state of long and short positions for comparison
    #         if not hasattr(self, 'previous_position_state'):
    #             self.previous_position_state = {}

    #         # Initialize symbol if not present in previous_position_state
    #         if symbol not in self.previous_position_state:
    #             self.previous_position_state[symbol] = {
    #                 'long': False,
    #                 'short': False,
    #                 'long_initial_entry': None,  # Initialize long_initial_entry
    #                 'short_initial_entry': None  # Initialize short_initial_entry
    #             }

    #         # Check for changes in position state (open -> closed)
    #         long_position_closed = self.previous_position_state[symbol]['long'] and not has_open_long_position
    #         short_position_closed = self.previous_position_state[symbol]['short'] and not has_open_short_position

    #         # Reset initial entry prices if positions have closed
    #         if long_position_closed:
    #             logging.info(f"[{symbol}] Long position closed. Resetting initial entry price for long.")
    #             self.previous_position_state[symbol]['long_initial_entry'] = None
            
    #         if short_position_closed:
    #             logging.info(f"[{symbol}] Short position closed. Resetting initial entry price for short.")
    #             self.previous_position_state[symbol]['short_initial_entry'] = None

    #         # Update position state tracking
    #         self.previous_position_state[symbol]['long'] = has_open_long_position
    #         self.previous_position_state[symbol]['short'] = has_open_short_position

    #         # Set initial entry price when a position is opened
    #         if has_open_long_position and not self.previous_position_state[symbol]['long_initial_entry']:
    #             self.previous_position_state[symbol]['long_initial_entry'] = long_pos_price
    #             logging.info(f"[{symbol}] Long position opened. Recording initial entry price for stop-loss: {long_pos_price}")

    #         if has_open_short_position and not self.previous_position_state[symbol]['short_initial_entry']:
    #             self.previous_position_state[symbol]['short_initial_entry'] = short_pos_price
    #             logging.info(f"[{symbol}] Short position opened. Recording initial entry price for stop-loss: {short_pos_price}")

    #         # Handle stop-loss logic using the initial entry prices
    #         if stop_loss_enabled:
    #             stop_loss_price_long = self.previous_position_state[symbol]['long_initial_entry'] * (1 - stop_loss_long / 100) if has_open_long_position else None
    #             stop_loss_price_short = self.previous_position_state[symbol]['short_initial_entry'] * (1 + stop_loss_short / 100) if has_open_short_position else None

    #             logging.info(f"[{symbol}] Calculated stop-loss prices: Long - {stop_loss_price_long}, Short - {stop_loss_price_short}")

    #             # Long position stop-loss check
    #             if has_open_long_position:
    #                 logging.info(f"[{symbol}] Long Position Quantity: {long_pos_qty}, Entry Price: {long_pos_price}, Stop-Loss Price: {stop_loss_price_long}")
    #                 if current_price <= stop_loss_price_long:
    #                     logging.info(f"[{symbol}] Long position hit stop-loss at {stop_loss_price_long}. Triggering stop-loss.")
    #                     self.trigger_stop_loss(symbol, long_pos_qty, 'long', stop_loss_price_long, best_bid_price)

    #             # Short position stop-loss check
    #             if has_open_short_position:
    #                 logging.info(f"[{symbol}] Short Position Quantity: {short_pos_qty}, Entry Price: {short_pos_price}, Stop-Loss Price: {stop_loss_price_short}")
    #                 if current_price >= stop_loss_price_short:
    #                     logging.info(f"[{symbol}] Short position hit stop-loss at {stop_loss_price_short}. Triggering stop-loss.")
    #                     self.trigger_stop_loss(symbol, short_pos_qty, 'short', stop_loss_price_short, best_ask_price)
    #         else:
    #             logging.info(f"Stop-loss disabled")

    #         # Fetch open symbols for long and short positions
    #         open_symbols_long = self.get_open_symbols_long(open_position_data)
    #         open_symbols_short = self.get_open_symbols_short(open_position_data)

    #         logging.info(f"Open symbols long: {open_symbols_long}")
    #         logging.info(f"Open symbols short: {open_symbols_short}")

    #         # Number of open symbols for long and short positions
    #         length_of_open_symbols_long = len(open_symbols_long)
    #         length_of_open_symbols_short = len(open_symbols_short)

    #         logging.info(f"Length of open symbols long: {length_of_open_symbols_long}")
    #         logging.info(f"Length of open symbols short: {length_of_open_symbols_short}")

    #         # Combine open symbols from both long and short positions
    #         all_open_symbols = open_symbols_long + open_symbols_short

    #         # Count unique open symbols across both long and short positions
    #         unique_open_symbols = len(set(all_open_symbols))

    #         logging.info(f"Unique open symbols: {unique_open_symbols}")

    #         should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
    #             symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)

    #         if self.auto_reduce_active_long.get(symbol, False):
    #             logging.info(f"Auto-reduce for long position on {symbol} is active")
    #             self.clear_grid(symbol, 'buy')
    #             #self.active_long_grids.discard(symbol)
    #         else:
    #             logging.info(f"Auto-reduce for long position on {symbol} is not active")

    #         if self.auto_reduce_active_short.get(symbol, False):
    #             logging.info(f"Auto-reduce for short position on {symbol} is active")
    #             self.clear_grid(symbol, 'sell')
    #             #self.active_short_grids.discard(symbol)
    #         else:
    #             logging.info(f"Auto-reduce for short position on {symbol} is not active")

    #         # Initialize last_empty_grid_time for symbol if not present
    #         if symbol not in self.last_empty_grid_time:
    #             self.last_empty_grid_time[symbol] = {'long': 0, 'short': 0}

    #         # Check for grid replacement conditions
    #         # has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
    #         # has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

    #         logging.info(f"Symbol format: {symbol}")
    #         logging.info(f"Test open orders: {open_orders}")
    #         has_open_long_order = any(order['info']['symbol'] == symbol and order['info']['side'].lower() == 'buy' and not order['info']['reduceOnly'] for order in open_orders)
    #         has_open_short_order = any(order['info']['symbol'] == symbol and order['info']['side'].lower() == 'sell' and not order['info']['reduceOnly'] for order in open_orders)

    #         logging.info(f"{symbol} Has open long order: {has_open_long_order}")
    #         logging.info(f"{symbol} Has open short order: {has_open_short_order}")

    #         replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
    #         replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

    #         current_time = time.time()

    #         # Check and log if the symbol is in max_qty_reached_symbol_long or short
    #         if symbol in self.max_qty_reached_symbol_long:
    #             logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")
    #         if symbol in self.max_qty_reached_symbol_short:
    #             logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

    #         # Additional logic for managing open symbols and checking trading permissions
    #         open_symbols = list(set(all_open_symbols))
    #         logging.info(f"Open symbols: {open_symbols}")

    #         trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
    #         logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
    #         logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

    #         fresh_mfirsi_signal = self.generate_l_signals(symbol)
    #         logging.info(f"Fresh MFIRSI signal for {symbol}: {fresh_mfirsi_signal}")
    #         mfi_signal_long = fresh_mfirsi_signal == "long"
    #         mfi_signal_short = fresh_mfirsi_signal == "short"

    #         logging.info(f"MFIRSI SIGNAL FOR {symbol}: {mfirsi_signal}")

    #         def issue_grid_safely(symbol: str, side: str, grid_levels: list, amounts: list):
    #             with grid_lock:  # Lock to ensure no simultaneous grid issuance
    #                 try:
    #                     # Use local attributes like active_long_grids or active_short_grids
    #                     grid_set = self.active_long_grids if side == 'long' else self.active_short_grids
    #                     order_side = 'buy' if side == 'long' else 'sell'

    #                     # Use has_active_grid to check if a grid is already active for this symbol and side
    #                     if self.has_active_grid(symbol, side, open_orders):  # Calls the local function inside handle_grid_trades
    #                         logging.warning(f"[{symbol}] {side.capitalize()} grid already active or existing open order detected. Skipping grid issuance.")
    #                         return  # Exit if the grid is already active or an open order exists

    #                     assert isinstance(grid_levels, list), f"Expected grid_levels to be a list, but got {type(grid_levels)}"
                        
    #                     if isinstance(amounts, int):
    #                         amounts = [amounts] * len(grid_levels)
    #                     assert isinstance(amounts, list), f"Expected amounts to be a list, but got {type(amounts)}"

    #                     if self.grid_cleared_status.get(symbol, {}).get(side, False):
    #                         logging.info(f"[{symbol}] Issuing new {side} grid orders.")
    #                         self.grid_cleared_status[symbol][side] = False  # Reset grid cleared status
    #                         self.issue_grid_orders(symbol, order_side, grid_levels, amounts, side == 'long', self.filled_levels[symbol][order_side])
    #                         grid_set.add(symbol)  # Add symbol to the active grid set
    #                         logging.info(f"[{symbol}] Successfully issued {side} grid orders.")
    #                     else:
    #                         logging.warning(f"[{symbol}] Attempted to issue {side} grid orders, but grid clearance not confirmed. Skipping grid creation.")
    #                 except Exception as e:
    #                     logging.error(f"Exception in issue_grid_safely for {symbol} - {side}: {e}")

    #         replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
    #             symbol, 
    #             long_pos_price, 
    #             short_pos_price, 
    #             long_pos_qty, 
    #             short_pos_qty,
    #             dynamic_outer_price_distance_long=dynamic_outer_price_distance_long,
    #             dynamic_outer_price_distance_short=dynamic_outer_price_distance_short
    #         )

    #         # Determine if there are open long and short positions based on provided quantities
    #         has_open_long_position = long_pos_qty > 0
    #         has_open_short_position = short_pos_qty > 0

    #         logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

    #         logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

    #         if unique_open_symbols <= symbols_allowed or symbol in open_symbols:
    #             fresh_signal = self.generate_l_signals(symbol)

    #             try:
    #                 # Handling for Long Positions
    #                 if (fresh_signal.lower() == "long" and long_mode and not has_open_long_position 
    #                     and not graceful_stop_long and not self.has_active_grid(symbol, 'long', open_orders) and symbol not in self.max_qty_reached_symbol_long):
                        
    #                     logging.info(f"[{symbol}] Creating new long position based on MFIRSI long signal")

    #                     # Use the new comprehensive check to ensure no double grids
    #                     if not self.has_active_grid(symbol, 'long', open_orders):
    #                         self.clear_grid(symbol, 'buy')

    #                         # Make a copy of grid_levels_long to safely modify
    #                         modified_grid_levels_long = grid_levels_long.copy()

    #                         # Set the first grid level to the best bid price for initial entry
    #                         best_bid_price = self.get_best_bid_price(symbol)  # Fetch the latest best bid price
    #                         logging.info(f"[{symbol}] Setting first level of modified grid to best_bid_price: {best_bid_price}")
    #                         modified_grid_levels_long[0] = best_bid_price

    #                         # Issue the grid only once initially
    #                         issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

    #                         retry_counter = 0
    #                         max_retries = 100  # Set a maximum number of retries

    #                         # Retry loop for issuing the grid
    #                         while long_pos_qty < 0.00001 and retry_counter < max_retries:
    #                             time.sleep(3)  # Wait for some time to allow order to be filled
    #                             try:
    #                                 long_pos_qty = self.get_position_qty(symbol, 'long')  # Re-fetch the long position quantity
    #                             except Exception as e:
    #                                 logging.error(f"[{symbol}] Error fetching long position quantity: {e}")
    #                                 break

    #                             retry_counter += 1
    #                             logging.info(f"[{symbol}] Long position quantity after waiting: {long_pos_qty}, retry attempt: {retry_counter}")

    #                             # Retry placing the grid every 10 retries (every 30 seconds)
    #                             if retry_counter % 10 == 0 and long_pos_qty < 0.00001:
    #                                 logging.info(f"[{symbol}] Retrying long grid orders due to MFIRSI signal long after {retry_counter} retries.")
                                    
    #                                 # Fetch the latest best bid price before retrying
    #                                 best_bid_price = self.get_best_bid_price(symbol)
    #                                 logging.info(f"[{symbol}] Updated best bid price on retry: {best_bid_price}")
                                    
    #                                 # Clear and re-issue the grid with updated best bid price
    #                                 self.clear_grid(symbol, 'buy')
    #                                 modified_grid_levels_long[0] = best_bid_price
    #                                 issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

    #                         logging.info(f"[{symbol}] Long position filled or max retries reached, exiting loop.")
    #                         self.last_signal_time[symbol] = current_time
    #                         self.last_mfirsi_signal[symbol] = "neutral"  # Reset to neutral after processing

    #                 # Handling for Short Positions
    #                 elif (fresh_signal.lower() == "short" and short_mode and not has_open_short_position 
    #                     and not graceful_stop_short and not self.has_active_grid(symbol, 'short', open_orders) and symbol not in self.max_qty_reached_symbol_short):
                        
    #                     logging.info(f"[{symbol}] Creating new short position based on MFIRSI short signal")

    #                     # Use the new comprehensive check to ensure no double grids
    #                     if not self.has_active_grid(symbol, 'short', open_orders):
    #                         self.clear_grid(symbol, 'sell')

    #                         # Make a copy of grid_levels_short to safely modify
    #                         modified_grid_levels_short = grid_levels_short.copy()

    #                         # Set the first grid level to the best ask price for initial entry
    #                         best_ask_price = self.get_best_ask_price(symbol)  # Fetch the latest best ask price
    #                         logging.info(f"[{symbol}] Setting first level of modified grid to best_ask_price: {best_ask_price}")
    #                         modified_grid_levels_short[0] = best_ask_price

    #                         # Issue the grid only once initially
    #                         issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

    #                         retry_counter = 0
    #                         max_retries = 50  # Set a maximum number of retries

    #                         # Retry loop for issuing the grid
    #                         while short_pos_qty < 0.00001 and retry_counter < max_retries:
    #                             time.sleep(5)  # Wait for some time to allow order to be filled
    #                             try:
    #                                 short_pos_qty = self.get_position_qty(symbol, 'short')  # Re-fetch the short position quantity
    #                             except Exception as e:
    #                                 logging.error(f"[{symbol}] Error fetching short position quantity: {e}")
    #                                 break

    #                             retry_counter += 1
    #                             logging.info(f"[{symbol}] Short position quantity after waiting: {short_pos_qty}, retry attempt: {retry_counter}")

    #                             # Retry placing the grid every 10 retries (every 50 seconds)
    #                             if retry_counter % 10 == 0 and short_pos_qty < 0.00001:
    #                                 logging.info(f"[{symbol}] Retrying short grid orders due to MFIRSI signal short after {retry_counter} retries.")
                                    
    #                                 # Fetch the latest best ask price before retrying
    #                                 best_ask_price = self.get_best_ask_price(symbol)
    #                                 logging.info(f"[{symbol}] Updated best ask price on retry: {best_ask_price}")
                                    
    #                                 # Clear and re-issue the grid with updated best ask price
    #                                 self.clear_grid(symbol, 'sell')
    #                                 modified_grid_levels_short[0] = best_ask_price
    #                                 issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

    #                         logging.info(f"[{symbol}] Short position filled or max retries reached, exiting loop.")
    #                         self.last_signal_time[symbol] = current_time
    #                         self.last_mfirsi_signal[symbol] = "neutral"  # Reset to neutral after processing

    #             except Exception as e:
    #                 logging.info(f"Exception caught in placing orders: {e}")
    #                 logging.info("Traceback: %s", traceback.format_exc())


    #         if additional_entries_from_signal:
    #             if symbol in open_symbols:
    #                 logging.info(f"Allowed symbol: {symbol}")

    #                 fresh_signal = self.generate_l_signals(symbol)

    #                 logging.info(f"Fresh signal for {symbol}: {fresh_signal}")

    #                 if not hasattr(self, 'last_mfirsi_signal'):
    #                     self.last_mfirsi_signal = {}
    #                 if not hasattr(self, 'last_signal_time'):
    #                     self.last_signal_time = {}

    #                 if self.last_mfirsi_signal is None:
    #                     self.last_mfirsi_signal = {}
    #                 if self.last_signal_time is None:
    #                     self.last_signal_time = {}

    #                 if symbol not in self.last_mfirsi_signal:
    #                     self.last_mfirsi_signal[symbol] = "neutral"
    #                 if symbol not in self.last_signal_time:
    #                     self.last_signal_time[symbol] = 0

    #                 current_time = time.time()
    #                 last_signal_time = self.last_signal_time.get(symbol, 0)
    #                 time_since_last_signal = current_time - last_signal_time

    #                 if time_since_last_signal < 180:  # 3 minutes
    #                     logging.info(f"[{symbol}] Waiting for signal cooldown. Time since last signal: {time_since_last_signal:.2f} seconds")
    #                     return

    #                 if fresh_signal.lower() != self.last_mfirsi_signal[symbol]:
    #                     logging.info(f"[{symbol}] MFIRSI signal changed to {fresh_signal}")
    #                     self.last_mfirsi_signal[symbol] = fresh_signal.lower()
    #                 else:
    #                     logging.info(f"[{symbol}] MFIRSI signal unchanged: {fresh_signal}")

    #                 try:
    #                     # Handling for additional Long entries
    #                     if fresh_signal.lower() == "long" and long_mode and not self.auto_reduce_active_long.get(symbol, False):
    #                         if long_pos_qty > 0.00001 and symbol not in self.max_qty_reached_symbol_long:  # Check if a long position already exists
    #                             if current_price <= long_pos_price:  # Enter additional entry only if current price <= long_pos_price
    #                                 logging.info(f"[{symbol}] Adding to existing long position based on MFIRSI long signal")

    #                                 # Ensure there are no existing active grids
    #                                 if not self.has_active_grid(symbol, 'long', open_orders):
    #                                     self.clear_grid(symbol, 'buy')

    #                                     modified_grid_levels_long = grid_levels_long.copy()
    #                                     modified_grid_levels_long[0] = best_bid_price
    #                                     issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

    #                                     retry_counter = 0
    #                                     max_retries = 50

    #                                     while long_pos_qty < 0.00001 and retry_counter < max_retries:
    #                                         time.sleep(5)
    #                                         try:
    #                                             long_pos_qty = self.get_position_qty(symbol, 'long')
    #                                         except Exception as e:
    #                                             logging.error(f"[{symbol}] Error fetching long position quantity: {e}")
    #                                             break

    #                                         retry_counter += 1
    #                                         logging.info(f"[{symbol}] Long position quantity after retry: {long_pos_qty}, retry attempt: {retry_counter}")

    #                                         if long_pos_qty < 0.00001 and retry_counter < max_retries:
    #                                             logging.info(f"[{symbol}] Retrying long grid orders.")
    #                                             self.clear_grid(symbol, 'buy')
    #                                             modified_grid_levels_long[0] = best_bid_price
    #                                             issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)
    #                                             time.sleep(4)
    #                                         else:
    #                                             logging.info(f"[{symbol}] Long position filled or max retries reached, exiting loop.")
    #                                             break

    #                                     self.last_signal_time[symbol] = current_time
    #                                     self.last_mfirsi_signal[symbol] = "neutral"
    #                                 else:
    #                                     logging.info(f"[{symbol}] Active long grid detected, skipping additional entry.")
    #                             else:
    #                                 logging.info(f"[{symbol}] Current price {current_price} is above long position price {long_pos_price}. Not adding to long position.")

    #                     # Handling for additional Short entries
    #                     elif fresh_signal.lower() == "short" and short_mode and not self.auto_reduce_active_short.get(symbol, False):
    #                         if short_pos_qty > 0.00001 and symbol not in self.max_qty_reached_symbol_short:  # Check if a short position already exists
    #                             if current_price >= short_pos_price:  # Enter additional entry only if current price >= short_pos_price
    #                                 logging.info(f"[{symbol}] Adding to existing short position based on MFIRSI short signal")

    #                                 # Ensure there are no existing active grids
    #                                 if not self.has_active_grid(symbol, 'short', open_orders):
    #                                     self.clear_grid(symbol, 'sell')

    #                                     modified_grid_levels_short = grid_levels_short.copy()
    #                                     modified_grid_levels_short[0] = best_ask_price
    #                                     issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

    #                                     retry_counter = 0
    #                                     max_retries = 50

    #                                     while short_pos_qty < 0.00001 and retry_counter < max_retries:
    #                                         time.sleep(5)
    #                                         try:
    #                                             short_pos_qty = self.get_position_qty(symbol, 'short')
    #                                         except Exception as e:
    #                                             logging.error(f"[{symbol}] Error fetching short position quantity: {e}")
    #                                             break

    #                                         retry_counter += 1
    #                                         logging.info(f"[{symbol}] Short position quantity after retry: {short_pos_qty}, retry attempt: {retry_counter}")

    #                                         if short_pos_qty < 0.00001 and retry_counter < max_retries:
    #                                             logging.info(f"[{symbol}] Retrying short grid orders.")
    #                                             self.clear_grid(symbol, 'sell')
    #                                             modified_grid_levels_short[0] = best_ask_price
    #                                             issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)
    #                                             time.sleep(4)
    #                                         else:
    #                                             logging.info(f"[{symbol}] Short position filled or max retries reached, exiting loop.")
    #                                             break

    #                                     self.last_signal_time[symbol] = current_time
    #                                     self.last_mfirsi_signal[symbol] = "neutral"
    #                                 else:
    #                                     logging.info(f"[{symbol}] Active short grid detected, skipping additional entry.")
    #                             else:
    #                                 logging.info(f"[{symbol}] Current price {current_price} is below short position price {short_pos_price}. Not adding to short position.")

    #                     elif fresh_signal.lower() == "neutral":
    #                         logging.info(f"[{symbol}] MFIRSI signal is neutral. No new grid orders.")

    #                     self.last_signal_time[symbol] = current_time
    #                     self.last_mfirsi_signal[symbol] = "neutral"  # Reset to neutral after processing

    #                 except Exception as e:
    #                     logging.info(f"Exception caught in placing entries {e}")
    #                     logging.info("Traceback: %s", traceback.format_exc())

    #         else:
    #             logging.info(f"Additional entries disabled from signal")

    #         time.sleep(5)

    #         logging.info(f"Symbol type for grid active check: {symbol}")
    #         long_grid_active, short_grid_active = self.check_grid_active(symbol, open_orders)

    #         logging.info(f"{symbol} Updated long grid active: {long_grid_active}")
    #         logging.info(f"{symbol} Updated short grid active: {short_grid_active}")

    #         # Check if the symbol is in active grids without open orders
    #         if not has_open_long_order and symbol in self.active_long_grids:
    #             self.active_long_grids.discard(symbol)
    #             logging.info(f"[{symbol}] No open long orders, removed from active long grids")

    #         if not has_open_short_order and symbol in self.active_short_grids:
    #             self.active_short_grids.discard(symbol)
    #             logging.info(f"[{symbol}] No open short orders, removed from active short grids")

    #         if symbol in open_symbols: 
    #             if (long_pos_qty > 0 and not long_grid_active and not self.has_active_grid(symbol, 'long', open_orders)) or \
    #             (short_pos_qty > 0 and not short_grid_active and not self.has_active_grid(symbol, 'short', open_orders)):
    #                 logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    
    #                 # Long Grid Logic
    #                 if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
    #                     if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
    #                         logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            
    #                         # Clear existing long grid
    #                         self.clear_grid(symbol, 'buy')

    #                         # Adjust the first grid level 0.5% below the best bid price
    #                         modified_grid_levels_long = grid_levels_long.copy()
    #                         best_bid_price = self.get_best_bid_price(symbol)
    #                         modified_grid_levels_long[0] = best_bid_price * 0.995  # 0.5% lower than the best bid price
    #                         logging.info(f"[{symbol}] Setting first level of modified long grid to best_bid_price - 0.5%: {modified_grid_levels_long[0]}")
                            
    #                         # Issue long grid safely
    #                         issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

    #                 # Short Grid Logic
    #                 if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
    #                     if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
    #                         logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            
    #                         # Clear existing short grid
    #                         self.clear_grid(symbol, 'sell')

    #                         # Adjust the first grid level 0.5% above the best ask price
    #                         modified_grid_levels_short = grid_levels_short.copy()
    #                         best_ask_price = self.get_best_ask_price(symbol)
    #                         modified_grid_levels_short[0] = best_ask_price * 1.005  # 0.5% higher than the best ask price
    #                         logging.info(f"[{symbol}] Setting first level of modified short grid to best_ask_price + 0.5%: {modified_grid_levels_short[0]}")
                            
    #                         # Issue short grid safely
    #                         issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

    #             current_time = time.time()

    #             # Grid clearing logic if no positions are open
    #             if not long_pos_qty and not short_pos_qty and symbol in self.active_long_grids | self.active_short_grids:
    #                 last_cleared = self.last_cleared_time.get(symbol, datetime.min)
    #                 if current_time - last_cleared > self.clear_interval:
    #                     logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
    #                     self.clear_grid(symbol, 'buy')
    #                     self.clear_grid(symbol, 'sell')
    #                     self.last_cleared_time[symbol] = current_time
    #                 else:
    #                     logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

    #         else:
    #             logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

    #         # Update TP for long position
    #         if long_pos_qty > 0:
    #             new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
    #                 long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
    #             )
    #             if new_long_tp_min is not None and new_long_tp_max is not None:
    #                 self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
    #                     symbol=symbol,
    #                     pos_qty=long_pos_qty,
    #                     upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
    #                     max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
    #                     short_pos_price=None,  # Not relevant for long TP settings
    #                     long_pos_price=long_pos_price,
    #                     positionIdx=1,
    #                     order_side="sell",
    #                     last_tp_update=self.next_long_tp_update,
    #                     tp_order_counts=tp_order_counts,
    #                     open_orders=open_orders
    #                 )

    #         if short_pos_qty > 0:
    #             new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
    #                 short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
    #             )
    #             if new_short_tp_min is not None and new_short_tp_max is not None:
    #                 self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
    #                     symbol=symbol,
    #                     pos_qty=short_pos_qty,
    #                     upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
    #                     max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
    #                     short_pos_price=short_pos_price,
    #                     long_pos_price=None,  # Not relevant for short TP settings
    #                     positionIdx=2,
    #                     order_side="buy",
    #                     last_tp_update=self.next_short_tp_update,
    #                     tp_order_counts=tp_order_counts,
    #                     open_orders=open_orders
    #                 )

    #         # Clear long grid if conditions are met
    #         if has_open_long_order and (long_pos_price is None or long_pos_price <= 0) and not mfi_signal_long:
    #             if symbol not in self.max_qty_reached_symbol_long:
    #                 current_time = time.time()

    #                 # Record the time when the invalid condition is first encountered
    #                 if symbol not in self.invalid_long_condition_time:
    #                     self.invalid_long_condition_time[symbol] = current_time
    #                     logging.info(f"Invalid long condition first met for {symbol}. Waiting before clearing grid...")

    #                 # If 1 minute has passed since the condition was first encountered
    #                 elif current_time - self.invalid_long_condition_time[symbol] >= 60:
    #                     self.clear_grid(symbol, "buy")
    #                     logging.info(f"Cleared long grid for {symbol} after 1 minute of invalid long_pos_price: {long_pos_price} and no long signal.")
    #                     # Reset the tracking time after clearing
    #                     del self.invalid_long_condition_time[symbol]
    #             else:
    #                 logging.info(f"{symbol} is in max qty reached symbol long, cannot replace grid")
    #         else:
    #             # Reset the tracking if conditions are no longer met
    #             if symbol in self.invalid_long_condition_time:
    #                 del self.invalid_long_condition_time[symbol]

    #         # Clear short grid if conditions are met
    #         if has_open_short_order and (short_pos_price is None or short_pos_price <= 0) and not mfi_signal_short:
    #             if symbol not in self.max_qty_reached_symbol_short:
    #                 current_time = time.time()

    #                 # Record the time when the invalid condition is first encountered
    #                 if symbol not in self.invalid_short_condition_time:
    #                     self.invalid_short_condition_time[symbol] = current_time
    #                     logging.info(f"Invalid short condition first met for {symbol}. Waiting before clearing grid...")

    #                 # If 1 minute has passed since the condition was first encountered
    #                 elif current_time - self.invalid_short_condition_time[symbol] >= 60:
    #                     self.clear_grid(symbol, "sell")
    #                     logging.info(f"Cleared short grid for {symbol} after 1 minute of invalid short_pos_price: {short_pos_price} and no short signal.")
    #                     # Reset the tracking time after clearing
    #                     del self.invalid_short_condition_time[symbol]
    #             else:
    #                 logging.info(f"{symbol} is in max qty reached symbol short, cannot replace grid")
    #         else:
    #             # Reset the tracking if conditions are no longer met
    #             if symbol in self.invalid_short_condition_time:
    #                 del self.invalid_short_condition_time[symbol]
                            
                            
    #     except Exception as e:
    #         logging.info(f"Error in executing gridstrategy: {e}")
    #         logging.info("Traceback: %s", traceback.format_exc())

    def handle_grid_trades(self, symbol, 
                        grid_levels_long, grid_levels_short, 
                        long_grid_active, short_grid_active,
                        long_pos_qty, short_pos_qty, current_price, 
                        dynamic_outer_price_distance_long, dynamic_outer_price_distance_short, 
                        min_outer_price_distance_long, min_outer_price_distance_short,
                        buffer_percentage_long, buffer_percentage_short, 
                        adjusted_grid_levels_long, adjusted_grid_levels_short, 
                        levels, amounts_long, amounts_short, 
                        best_bid_price, best_ask_price, 
                        mfirsi_signal, open_orders, initial_entry_buffer_pct, 
                        reissue_threshold, entry_during_autoreduce, min_qty, open_symbols, 
                        symbols_allowed, long_mode, short_mode, 
                        long_pos_price, short_pos_price, 
                        graceful_stop_long, graceful_stop_short, 
                        min_buffer_percentage, max_buffer_percentage, 
                        additional_entries_from_signal, open_position_data, 
                        upnl_profit_pct, max_upnl_profit_pct, tp_order_counts, 
                        stop_loss_long, stop_loss_short, stop_loss_enabled=True):

        try:
            # Determine if there is an open long or short position
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            # Store previous state of long and short positions for comparison
            if not hasattr(self, 'previous_position_state'):
                self.previous_position_state = {}

            # Initialize symbol if not present in previous_position_state
            if symbol not in self.previous_position_state:
                self.previous_position_state[symbol] = {
                    'long': False,
                    'short': False,
                    'long_initial_entry': None,  # Initialize long_initial_entry
                    'short_initial_entry': None  # Initialize short_initial_entry
                }

            # Check for changes in position state (open -> closed)
            long_position_closed = self.previous_position_state[symbol]['long'] and not has_open_long_position
            short_position_closed = self.previous_position_state[symbol]['short'] and not has_open_short_position

            # Reset initial entry prices if positions have closed
            if long_position_closed:
                logging.info(f"[{symbol}] Long position closed. Resetting initial entry price for long.")
                self.previous_position_state[symbol]['long_initial_entry'] = None
            
            if short_position_closed:
                logging.info(f"[{symbol}] Short position closed. Resetting initial entry price for short.")
                self.previous_position_state[symbol]['short_initial_entry'] = None

            # Update position state tracking
            self.previous_position_state[symbol]['long'] = has_open_long_position
            self.previous_position_state[symbol]['short'] = has_open_short_position

            # Set initial entry price when a position is opened
            if has_open_long_position and not self.previous_position_state[symbol]['long_initial_entry']:
                self.previous_position_state[symbol]['long_initial_entry'] = long_pos_price
                logging.info(f"[{symbol}] Long position opened. Recording initial entry price for stop-loss: {long_pos_price}")

            if has_open_short_position and not self.previous_position_state[symbol]['short_initial_entry']:
                self.previous_position_state[symbol]['short_initial_entry'] = short_pos_price
                logging.info(f"[{symbol}] Short position opened. Recording initial entry price for stop-loss: {short_pos_price}")

            # Handle stop-loss logic using the initial entry prices
            if stop_loss_enabled:
                stop_loss_price_long = self.previous_position_state[symbol]['long_initial_entry'] * (1 - stop_loss_long / 100) if has_open_long_position else None
                stop_loss_price_short = self.previous_position_state[symbol]['short_initial_entry'] * (1 + stop_loss_short / 100) if has_open_short_position else None

                logging.info(f"[{symbol}] Calculated stop-loss prices: Long - {stop_loss_price_long}, Short - {stop_loss_price_short}")

                # Long position stop-loss check
                if has_open_long_position:
                    logging.info(f"[{symbol}] Long Position Quantity: {long_pos_qty}, Entry Price: {long_pos_price}, Stop-Loss Price: {stop_loss_price_long}")
                    if current_price <= stop_loss_price_long:
                        logging.info(f"[{symbol}] Long position hit stop-loss at {stop_loss_price_long}. Triggering stop-loss.")
                        self.trigger_stop_loss(symbol, long_pos_qty, 'long', stop_loss_price_long, best_bid_price)

                # Short position stop-loss check
                if has_open_short_position:
                    logging.info(f"[{symbol}] Short Position Quantity: {short_pos_qty}, Entry Price: {short_pos_price}, Stop-Loss Price: {stop_loss_price_short}")
                    if current_price >= stop_loss_price_short:
                        logging.info(f"[{symbol}] Short position hit stop-loss at {stop_loss_price_short}. Triggering stop-loss.")
                        self.trigger_stop_loss(symbol, short_pos_qty, 'short', stop_loss_price_short, best_ask_price)
            else:
                logging.info(f"Stop-loss disabled")

            # Fetch open symbols for long and short positions
            open_symbols_long = self.get_open_symbols_long(open_position_data)
            open_symbols_short = self.get_open_symbols_short(open_position_data)

            logging.info(f"Open symbols long: {open_symbols_long}")
            logging.info(f"Open symbols short: {open_symbols_short}")

            # Number of open symbols for long and short positions
            length_of_open_symbols_long = len(open_symbols_long)
            length_of_open_symbols_short = len(open_symbols_short)

            logging.info(f"Length of open symbols long: {length_of_open_symbols_long}")
            logging.info(f"Length of open symbols short: {length_of_open_symbols_short}")

            # Combine open symbols from both long and short positions
            all_open_symbols = open_symbols_long + open_symbols_short

            # Count unique open symbols across both long and short positions
            unique_open_symbols = len(set(all_open_symbols))

            logging.info(f"Unique open symbols: {unique_open_symbols}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Initialize last_empty_grid_time for symbol if not present
            if symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {'long': 0, 'short': 0}

            logging.info(f"Symbol format: {symbol}")
            logging.info(f"Test open orders: {open_orders}")
            has_open_long_order = any(order['info']['symbol'] == symbol and order['info']['side'].lower() == 'buy' and not order['info']['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['info']['symbol'] == symbol and order['info']['side'].lower() == 'sell' and not order['info']['reduceOnly'] for order in open_orders)

            logging.info(f"{symbol} Has open long order: {has_open_long_order}")
            logging.info(f"{symbol} Has open short order: {has_open_short_order}")

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            current_time = time.time()

            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")
            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            open_symbols = list(set(all_open_symbols))
            logging.info(f"Open symbols: {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            fresh_mfirsi_signal = self.generate_l_signals(symbol)
            logging.info(f"Fresh MFIRSI signal for {symbol}: {fresh_mfirsi_signal}")
            mfi_signal_long = fresh_mfirsi_signal == "long"
            mfi_signal_short = fresh_mfirsi_signal == "short"

            logging.info(f"MFIRSI SIGNAL FOR {symbol}: {mfirsi_signal}")

            def issue_grid_safely(symbol: str, side: str, grid_levels: list, amounts: list):
                with grid_lock:  # Lock to ensure no simultaneous grid issuance
                    try:
                        # Use local attributes like active_long_grids or active_short_grids
                        grid_set = self.active_long_grids if side == 'long' else self.active_short_grids
                        order_side = 'buy' if side == 'long' else 'sell'

                        # Use has_active_grid to check if a grid is already active for this symbol and side
                        if self.has_active_grid(symbol, side, open_orders):
                            logging.warning(f"[{symbol}] {side.capitalize()} grid already active or existing open order detected. Skipping grid issuance.")
                            return  # Exit if the grid is already active or an open order exists

                        assert isinstance(grid_levels, list), f"Expected grid_levels to be a list, but got {type(grid_levels)}"
                        
                        if isinstance(amounts, int):
                            amounts = [amounts] * len(grid_levels)
                        assert isinstance(amounts, list), f"Expected amounts to be a list, but got {type(amounts)}"

                        if self.grid_cleared_status.get(symbol, {}).get(side, False):
                            logging.info(f"[{symbol}] Issuing new {side} grid orders.")
                            self.grid_cleared_status[symbol][side] = False  # Reset grid cleared status
                            self.issue_grid_orders(symbol, order_side, grid_levels, amounts, side == 'long', self.filled_levels[symbol][order_side])
                            grid_set.add(symbol)  # Add symbol to the active grid set
                            logging.info(f"[{symbol}] Successfully issued {side} grid orders.")
                        else:
                            logging.warning(f"[{symbol}] Attempted to issue {side} grid orders, but grid clearance not confirmed. Skipping grid creation.")
                    except Exception as e:
                        logging.error(f"Exception in issue_grid_safely for {symbol} - {side}: {e}")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                symbol, 
                long_pos_price, 
                short_pos_price, 
                long_pos_qty, 
                short_pos_qty,
                dynamic_outer_price_distance_long=dynamic_outer_price_distance_long,
                dynamic_outer_price_distance_short=dynamic_outer_price_distance_short
            )

            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if unique_open_symbols <= symbols_allowed or symbol in open_symbols:
                fresh_signal = self.generate_l_signals(symbol)

                try:
                    # Handling for Long Positions
                    if (fresh_signal.lower() == "long" and long_mode and not has_open_long_position 
                        and not graceful_stop_long and not self.has_active_grid(symbol, 'long', open_orders) and symbol not in self.max_qty_reached_symbol_long):
                        
                        logging.info(f"[{symbol}] Creating new long position based on MFIRSI long signal")

                        if not self.has_active_grid(symbol, 'long', open_orders):
                            self.clear_grid(symbol, 'buy')

                            modified_grid_levels_long = grid_levels_long.copy()

                            best_bid_price = self.get_best_bid_price(symbol)
                            logging.info(f"[{symbol}] Setting first level of modified grid to best_bid_price: {best_bid_price}")
                            modified_grid_levels_long[0] = best_bid_price

                            issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

                            retry_counter = 0
                            max_retries = 100

                            while long_pos_qty < 0.00001 and retry_counter < max_retries:
                                time.sleep(3)
                                try:
                                    long_pos_qty = self.get_position_qty(symbol, 'long')
                                except Exception as e:
                                    logging.error(f"[{symbol}] Error fetching long position quantity: {e}")
                                    break

                                retry_counter += 1
                                logging.info(f"[{symbol}] Long position quantity after waiting: {long_pos_qty}, retry attempt: {retry_counter}")

                                if retry_counter % 10 == 0 and long_pos_qty < 0.00001:
                                    logging.info(f"[{symbol}] Retrying long grid orders due to MFIRSI signal long after {retry_counter} retries.")
                                    best_bid_price = self.get_best_bid_price(symbol)
                                    logging.info(f"[{symbol}] Updated best bid price on retry: {best_bid_price}")

                                    self.clear_grid(symbol, 'buy')
                                    modified_grid_levels_long[0] = best_bid_price
                                    issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

                            logging.info(f"[{symbol}] Long position filled or max retries reached, exiting loop.")
                            self.last_signal_time[symbol] = current_time
                            self.last_mfirsi_signal[symbol] = "neutral"

                    # Handling for Short Positions
                    elif (fresh_signal.lower() == "short" and short_mode and not has_open_short_position 
                        and not graceful_stop_short and not self.has_active_grid(symbol, 'short', open_orders) and symbol not in self.max_qty_reached_symbol_short):
                        
                        logging.info(f"[{symbol}] Creating new short position based on MFIRSI short signal")

                        if not self.has_active_grid(symbol, 'short', open_orders):
                            self.clear_grid(symbol, 'sell')

                            modified_grid_levels_short = grid_levels_short.copy()

                            best_ask_price = self.get_best_ask_price(symbol)
                            logging.info(f"[{symbol}] Setting first level of modified grid to best_ask_price: {best_ask_price}")
                            modified_grid_levels_short[0] = best_ask_price

                            issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

                            retry_counter = 0
                            max_retries = 50

                            while short_pos_qty < 0.00001 and retry_counter < max_retries:
                                time.sleep(5)
                                try:
                                    short_pos_qty = self.get_position_qty(symbol, 'short')
                                except Exception as e:
                                    logging.error(f"[{symbol}] Error fetching short position quantity: {e}")
                                    break

                                retry_counter += 1
                                logging.info(f"[{symbol}] Short position quantity after waiting: {short_pos_qty}, retry attempt: {retry_counter}")

                                if retry_counter % 10 == 0 and short_pos_qty < 0.00001:
                                    logging.info(f"[{symbol}] Retrying short grid orders due to MFIRSI signal short after {retry_counter} retries.")
                                    
                                    best_ask_price = self.get_best_ask_price(symbol)
                                    logging.info(f"[{symbol}] Updated best ask price on retry: {best_ask_price}")

                                    self.clear_grid(symbol, 'sell')
                                    modified_grid_levels_short[0] = best_ask_price
                                    issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

                            logging.info(f"[{symbol}] Short position filled or max retries reached, exiting loop.")
                            self.last_signal_time[symbol] = current_time
                            self.last_mfirsi_signal[symbol] = "neutral"

                except Exception as e:
                    logging.info(f"Exception caught in placing orders: {e}")
                    logging.info("Traceback: %s", traceback.format_exc())


            if additional_entries_from_signal:
                if symbol in open_symbols:
                    logging.info(f"Allowed symbol: {symbol}")
                    fresh_signal = self.generate_l_signals(symbol)
                    logging.info(f"Fresh signal for {symbol}: {fresh_signal}")

                    if not hasattr(self, 'last_mfirsi_signal'):
                        self.last_mfirsi_signal = {}
                    if not hasattr(self, 'last_signal_time'):
                        self.last_signal_time = {}

                    if self.last_mfirsi_signal is None:
                        self.last_mfirsi_signal = {}
                    if self.last_signal_time is None:
                        self.last_signal_time = {}

                    if symbol not in self.last_mfirsi_signal:
                        self.last_mfirsi_signal[symbol] = "neutral"
                    if symbol not in self.last_signal_time:
                        self.last_signal_time[symbol] = 0

                    current_time = time.time()
                    last_signal_time = self.last_signal_time.get(symbol, 0)
                    time_since_last_signal = current_time - last_signal_time

                    if time_since_last_signal < 180:  # 3 minutes
                        logging.info(f"[{symbol}] Waiting for signal cooldown. Time since last signal: {time_since_last_signal:.2f} seconds")
                        return

                    if fresh_signal.lower() != self.last_mfirsi_signal[symbol]:
                        logging.info(f"[{symbol}] MFIRSI signal changed to {fresh_signal}")
                        self.last_mfirsi_signal[symbol] = fresh_signal.lower()
                    else:
                        logging.info(f"[{symbol}] MFIRSI signal unchanged: {fresh_signal}")

                    try:
                        # Additional Long entries
                        if fresh_signal.lower() == "long" and long_mode and not self.auto_reduce_active_long.get(symbol, False):
                            if long_pos_qty > 0.00001 and symbol not in self.max_qty_reached_symbol_long:
                                if current_price <= long_pos_price:
                                    logging.info(f"[{symbol}] Adding to existing long position based on MFIRSI long signal")
                                    if not self.has_active_grid(symbol, 'long', open_orders):
                                        self.clear_grid(symbol, 'buy')
                                        modified_grid_levels_long = grid_levels_long.copy()
                                        modified_grid_levels_long[0] = best_bid_price
                                        issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

                                        retry_counter = 0
                                        max_retries = 50
                                        while long_pos_qty < 0.00001 and retry_counter < max_retries:
                                            time.sleep(5)
                                            try:
                                                long_pos_qty = self.get_position_qty(symbol, 'long')
                                            except Exception as e:
                                                logging.error(f"[{symbol}] Error fetching long position quantity: {e}")
                                                break

                                            retry_counter += 1
                                            logging.info(f"[{symbol}] Long position quantity after retry: {long_pos_qty}, retry attempt: {retry_counter}")

                                            if long_pos_qty < 0.00001 and retry_counter < max_retries:
                                                logging.info(f"[{symbol}] Retrying long grid orders.")
                                                self.clear_grid(symbol, 'buy')
                                                modified_grid_levels_long[0] = best_bid_price
                                                issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)
                                                time.sleep(4)
                                            else:
                                                logging.info(f"[{symbol}] Long position filled or max retries reached, exiting loop.")
                                                break

                                        self.last_signal_time[symbol] = current_time
                                        self.last_mfirsi_signal[symbol] = "neutral"
                                    else:
                                        logging.info(f"[{symbol}] Active long grid detected, skipping additional entry.")
                                else:
                                    logging.info(f"[{symbol}] Current price {current_price} is above long position price {long_pos_price}. Not adding to long position.")

                        # Additional Short entries
                        elif fresh_signal.lower() == "short" and short_mode and not self.auto_reduce_active_short.get(symbol, False):
                            if short_pos_qty > 0.00001 and symbol not in self.max_qty_reached_symbol_short:
                                if current_price >= short_pos_price:
                                    logging.info(f"[{symbol}] Adding to existing short position based on MFIRSI short signal")
                                    if not self.has_active_grid(symbol, 'short', open_orders):
                                        self.clear_grid(symbol, 'sell')
                                        modified_grid_levels_short = grid_levels_short.copy()
                                        modified_grid_levels_short[0] = best_ask_price
                                        issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

                                        retry_counter = 0
                                        max_retries = 50
                                        while short_pos_qty < 0.00001 and retry_counter < max_retries:
                                            time.sleep(5)
                                            try:
                                                short_pos_qty = self.get_position_qty(symbol, 'short')
                                            except Exception as e:
                                                logging.error(f"[{symbol}] Error fetching short position quantity: {e}")
                                                break

                                            retry_counter += 1
                                            logging.info(f"[{symbol}] Short position quantity after retry: {short_pos_qty}, retry attempt: {retry_counter}")

                                            if short_pos_qty < 0.00001 and retry_counter < max_retries:
                                                logging.info(f"[{symbol}] Retrying short grid orders.")
                                                self.clear_grid(symbol, 'sell')
                                                modified_grid_levels_short[0] = best_ask_price
                                                issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)
                                                time.sleep(4)
                                            else:
                                                logging.info(f"[{symbol}] Short position filled or max retries reached, exiting loop.")
                                                break

                                        self.last_signal_time[symbol] = current_time
                                        self.last_mfirsi_signal[symbol] = "neutral"
                                    else:
                                        logging.info(f"[{symbol}] Active short grid detected, skipping additional entry.")
                                else:
                                    logging.info(f"[{symbol}] Current price {current_price} is below short position price {short_pos_price}. Not adding to short position.")

                        elif fresh_signal.lower() == "neutral":
                            logging.info(f"[{symbol}] MFIRSI signal is neutral. No new grid orders.")

                        self.last_signal_time[symbol] = current_time
                        self.last_mfirsi_signal[symbol] = "neutral"

                    except Exception as e:
                        logging.info(f"Exception caught in placing entries {e}")
                        logging.info("Traceback: %s", traceback.format_exc())

            else:
                logging.info(f"Additional entries disabled from signal")

            time.sleep(5)

            logging.info(f"Symbol type for grid active check: {symbol}")
            long_grid_active, short_grid_active = self.check_grid_active(symbol, open_orders)

            logging.info(f"{symbol} Updated long grid active: {long_grid_active}")
            logging.info(f"{symbol} Updated short grid active: {short_grid_active}")

            # Check if the symbol is in active grids without open orders
            if not has_open_long_order and symbol in self.active_long_grids:
                self.active_long_grids.discard(symbol)
                logging.info(f"[{symbol}] No open long orders, removed from active long grids")

            if not has_open_short_order and symbol in self.active_short_grids:
                self.active_short_grids.discard(symbol)
                logging.info(f"[{symbol}] No open short orders, removed from active short grids")

            if symbol in open_symbols: 
                if (long_pos_qty > 0 and not long_grid_active and not self.has_active_grid(symbol, 'long', open_orders)) or \
                (short_pos_qty > 0 and not short_grid_active and not self.has_active_grid(symbol, 'short', open_orders)):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    
                    # Long Grid Logic
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')

                            modified_grid_levels_long = grid_levels_long.copy()
                            best_bid_price = self.get_best_bid_price(symbol)
                            modified_grid_levels_long[0] = best_bid_price * 0.995
                            logging.info(f"[{symbol}] Setting first level of modified long grid to best_bid_price - 0.5%: {modified_grid_levels_long[0]}")
                            
                            issue_grid_safely(symbol, 'long', modified_grid_levels_long, amounts_long)

                    # Short Grid Logic
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')

                            modified_grid_levels_short = grid_levels_short.copy()
                            best_ask_price = self.get_best_ask_price(symbol)
                            modified_grid_levels_short[0] = best_ask_price * 1.005
                            logging.info(f"[{symbol}] Setting first level of modified short grid to best_ask_price + 0.5%: {modified_grid_levels_short[0]}")
                            
                            issue_grid_safely(symbol, 'short', modified_grid_levels_short, amounts_short)

                current_time = time.time()

                # Grid clearing logic if no positions are open
                if not long_pos_qty and not short_pos_qty and symbol in self.active_long_grids | self.active_short_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,
                        max_upnl_profit_pct=max_upnl_profit_pct,
                        short_pos_price=None,
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            # Update TP for short position
            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,
                        max_upnl_profit_pct=max_upnl_profit_pct,
                        short_pos_price=short_pos_price,
                        long_pos_price=None,
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            # Clear long grid if conditions are met
            if has_open_long_order and (long_pos_price is None or long_pos_price <= 0) and not mfi_signal_long:
                if symbol not in self.max_qty_reached_symbol_long:
                    current_time = time.time()
                    if symbol not in self.invalid_long_condition_time:
                        self.invalid_long_condition_time[symbol] = current_time
                        logging.info(f"Invalid long condition first met for {symbol}. Waiting before clearing grid...")
                    elif current_time - self.invalid_long_condition_time[symbol] >= 60:
                        self.clear_grid(symbol, "buy")
                        logging.info(f"Cleared long grid for {symbol} after 1 minute of invalid long_pos_price: {long_pos_price} and no long signal.")
                        del self.invalid_long_condition_time[symbol]
                else:
                    logging.info(f"{symbol} is in max qty reached symbol long, cannot replace grid")
            else:
                if symbol in self.invalid_long_condition_time:
                    del self.invalid_long_condition_time[symbol]

            # Clear short grid if conditions are met
            if has_open_short_order and (short_pos_price is None or short_pos_price <= 0) and not mfi_signal_short:
                if symbol not in self.max_qty_reached_symbol_short:
                    current_time = time.time()
                    if symbol not in self.invalid_short_condition_time:
                        self.invalid_short_condition_time[symbol] = current_time
                        logging.info(f"Invalid short condition first met for {symbol}. Waiting before clearing grid...")
                    elif current_time - self.invalid_short_condition_time[symbol] >= 60:
                        self.clear_grid(symbol, "sell")
                        logging.info(f"Cleared short grid for {symbol} after 1 minute of invalid short_pos_price: {short_pos_price} and no short signal.")
                        del self.invalid_short_condition_time[symbol]
                else:
                    logging.info(f"{symbol} is in max qty reached symbol short, cannot replace grid")
            else:
                if symbol in self.invalid_short_condition_time:
                    del self.invalid_short_condition_time[symbol]

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())


    def lingrid_ob_lsignal_entryuponsignal(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                                                                        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                                                                        strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
                                                                        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
                                                                        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
                                                                        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                                                                        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
                                                                        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                                                                        max_qty_percent_long: float, max_qty_percent_short: float):
        try:
            spread = self.get_4h_candle_spread(symbol)
            logging.info(f"4h Candle spread for {symbol}: {spread}")

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))
            logging.info(f"Dynamic outer price distance for {symbol}: {dynamic_outer_price_distance}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 or mfirsi_signal.lower() == "long" else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 or mfirsi_signal.lower() == "short" else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            min_price = current_price - max_outer_price_distance * current_price
            max_price = current_price + max_outer_price_distance * current_price

            price_range = np.arange(min_price, max_price, (max_price - min_price) / 100)
            volume_histogram_long = np.zeros_like(price_range)
            volume_histogram_short = np.zeros_like(price_range)

            for order in order_book['bids']:
                price, volume = order[0], order[1]
                if min_price <= price <= current_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_long[index] += volume

            for order in order_book['asks']:
                price, volume = order[0], order[1]
                if current_price <= price <= max_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_short[index] += volume

            volume_threshold_long = np.mean(volume_histogram_long) * 1.5
            significant_levels_long = price_range[volume_histogram_long >= volume_threshold_long]

            volume_threshold_short = np.mean(volume_histogram_short) * 1.5
            significant_levels_short = price_range[volume_histogram_short >= volume_threshold_short]

            initial_entry_long = current_price - buffer_distance_long
            initial_entry_short = current_price + buffer_distance_short

            if long_pos_qty == 0 or mfirsi_signal.lower() == "long":
                grid_levels_long = [initial_entry_long] + [
                    current_price - buffer_distance_long - i * ((max_outer_price_distance * current_price - buffer_distance_long) / (levels - 1)) 
                    for i in range(1, levels)
                ]
            else:
                grid_levels_long = [
                    current_price - buffer_distance_long - i * ((max_outer_price_distance * current_price - buffer_distance_long) / levels) 
                    for i in range(levels)
                ]

            if short_pos_qty == 0 or mfirsi_signal.lower() == "short":
                grid_levels_short = [initial_entry_short] + [
                    current_price + buffer_distance_short + i * ((max_outer_price_distance * current_price - buffer_distance_short) / (levels - 1)) 
                    for i in range(1, levels)
                ]
            else:
                grid_levels_short = [
                    current_price + buffer_distance_short + i * ((max_outer_price_distance * current_price - buffer_distance_short) / levels) 
                    for i in range(levels)
                ]

            def find_nearest_significant_level(level, significant_levels, tolerance, min_distance, max_distance, current_price, previous_level=None):
                for sig_level in significant_levels:
                    if abs(level - sig_level) / level < tolerance:
                        if current_price - max_distance * current_price <= sig_level <= current_price - min_distance * current_price:
                            if previous_level is None or abs(sig_level - previous_level) > (max_distance * current_price / levels):
                                return sig_level
                return level

            tolerance = 0.01
            adjusted_grid_levels_long = []
            adjusted_grid_levels_short = []

            for i, level in enumerate(grid_levels_long):
                previous_level = adjusted_grid_levels_long[-1] if adjusted_grid_levels_long else None
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels_long,
                    tolerance,
                    min_outer_price_distance,
                    max_outer_price_distance,
                    current_price,
                    previous_level
                )
                if current_price - max_outer_price_distance * current_price <= adjusted_level <= current_price - buffer_distance_long:
                    adjusted_grid_levels_long.append(adjusted_level)

            for i, level in enumerate(grid_levels_short):
                previous_level = adjusted_grid_levels_short[-1] if adjusted_grid_levels_short else None
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels_short,
                    tolerance,
                    min_outer_price_distance,
                    max_outer_price_distance,
                    current_price,
                    previous_level
                )
                if current_price + buffer_distance_short <= adjusted_level <= current_price + max_outer_price_distance * current_price:
                    adjusted_grid_levels_short.append(adjusted_level)

            grid_levels_long = adjusted_grid_levels_long
            grid_levels_short = adjusted_grid_levels_short

            if len(grid_levels_long) < levels:
                remaining_levels = levels - len(grid_levels_long)
                additional_levels_long = np.linspace(
                    grid_levels_long[-1] if grid_levels_long else current_price - buffer_distance_long,
                    current_price - max_outer_price_distance * current_price,
                    remaining_levels + 1
                )[1:]
                grid_levels_long = np.concatenate((grid_levels_long, additional_levels_long))

            if len(grid_levels_short) < levels:
                remaining_levels = levels - len(grid_levels_short)
                additional_levels_short = np.linspace(
                    grid_levels_short[-1] if grid_levels_short else current_price + buffer_distance_short,
                    current_price + max_outer_price_distance * current_price,
                    remaining_levels + 1
                )[1:]
                grid_levels_short = np.concatenate((grid_levels_short, additional_levels_short))

            grid_levels_long = sorted(grid_levels_long, reverse=True)
            grid_levels_short = sorted(grid_levels_short)

            logging.info(f"[{symbol}] Initial long entry level: {initial_entry_long}")
            logging.info(f"[{symbol}] Initial short entry level: {initial_entry_short}")
            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")
    

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            # Update position quantities before processing
            long_pos_qty = self.get_position_qty(symbol, 'long')
            short_pos_qty = self.get_position_qty(symbol, 'short')

            logging.info(f"Updated Long pos qty: {long_pos_qty}")
            logging.info(f"Updated Short pos qty: {short_pos_qty}")

            # Skip if the signal is the same as the last processed signal
            if not hasattr(self, 'last_attempted_signal'):
                self.last_attempted_signal = None

            if self.last_attempted_signal == mfirsi_signal and (long_pos_qty > 0.00001 or short_pos_qty > 0.00001):
                logging.info(f"Skipping processing for {symbol} as the mfirsi_signal is the same as the last attempted signal: {mfirsi_signal}")
                return

            # Handle long and short grid replacement based on mfirsi_signal
            if mfirsi_signal.lower() == "long" and long_mode and not self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"[{symbol}] Replacing long grid orders due to MFIRSI signal long.")
                self.clear_grid(symbol, 'buy')
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                self.last_attempted_signal = mfirsi_signal  # Set the attempted signal
                while long_pos_qty < 0.00001:
                    time.sleep(5)  # Wait for some time to allow order to be filled
                    long_pos_qty = self.get_position_qty(symbol, 'long')  # Re-fetch the long position quantity
                    if long_pos_qty < 0.00001:
                        logging.info(f"[{symbol}] Retrying long grid orders due to MFIRSI signal long.")
                        self.clear_grid(symbol, 'buy')
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        break  # Exit loop once the order is filled

            elif mfirsi_signal.lower() == "short" and short_mode and not self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"[{symbol}] Replacing short grid orders due to MFIRSI signal short.")
                self.clear_grid(symbol, 'sell')
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                self.last_attempted_signal = mfirsi_signal  # Set the attempted signal
                while short_pos_qty < 0.00001:
                    time.sleep(5)  # Wait for some time to allow order to be filled
                    short_pos_qty = self.get_position_qty(symbol, 'short')  # Re-fetch the short position quantity
                    if short_pos_qty < 0.00001:
                        logging.info(f"[{symbol}] Retrying short grid orders due to MFIRSI signal short.")
                        self.clear_grid(symbol, 'sell')
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        break  # Exit loop once the order is filled
                    
            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Initialize last_empty_grid_time for symbol if not present
            if symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {'long': 0, 'short': 0}

            # Check for grid replacement conditions
            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            # Track the last time the grid was emptied
            if replace_empty_long_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            if replace_empty_short_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}

            current_time = time.time()

            # Check and log if the symbol is in max_qty_reached_symbol_long
            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")

            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            # Additional logic for managing open symbols and checking trading permissions
            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                logging.info(f"Allowed symbol: {symbol}")

                has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                logging.info(f"MFIRSI Signal for {symbol} : {mfirsi_signal}")

                replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                    symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                    dynamic_outer_price_distance=dynamic_outer_price_distance
                )

                # Replace long grid if conditions are met
                if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', 0) > 240))) and not self.auto_reduce_active_long.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_long:
                        logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'buy')
                        buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                        buffer_distance_long = current_price * buffer_percentage_long
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['long'] = current_time
                        logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol long cannot replace grid")

                # Replace short grid if conditions are met
                if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', 0) > 240))) and not self.auto_reduce_active_short.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_short:
                        logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'sell')
                        buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                        buffer_distance_short = current_price * buffer_percentage_short
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['short'] = current_time
                        logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol short cannot replace grid")

                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
        

            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            logging.info(f"Long pos qty: {long_pos_qty}")
            logging.info(f"Short pos qty: {short_pos_qty}")

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            time.sleep(5)

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())


    def linear_grid_hardened_gridspan_ob_volumelevels_dynamictp_lsignal(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                                                        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                                                        strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
                                                        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
                                                        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
                                                        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                                                        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
                                                        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                                                        max_qty_percent_long: float, max_qty_percent_short: float):
        try:
            spread = self.get_4h_candle_spread(symbol)
            logging.info(f"4h Candle spread for {symbol}: {spread}")

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))
            logging.info(f"Dynamic outer price distance for {symbol}: {dynamic_outer_price_distance}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            min_price = current_price - max_outer_price_distance * current_price
            max_price = current_price + max_outer_price_distance * current_price

            price_range = np.arange(min_price, max_price, (max_price - min_price) / 100)
            volume_histogram_long = np.zeros_like(price_range)
            volume_histogram_short = np.zeros_like(price_range)

            for order in order_book['bids']:
                price, volume = order[0], order[1]
                if min_price <= price <= current_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_long[index] += volume

            for order in order_book['asks']:
                price, volume = order[0], order[1]
                if current_price <= price <= max_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_short[index] += volume

            volume_threshold_long = np.mean(volume_histogram_long) * 1.5
            significant_levels_long = price_range[volume_histogram_long >= volume_threshold_long]

            volume_threshold_short = np.mean(volume_histogram_short) * 1.5
            significant_levels_short = price_range[volume_histogram_short >= volume_threshold_short]

            initial_entry_long = current_price - buffer_distance_long
            initial_entry_short = current_price + buffer_distance_short

            if long_pos_qty == 0:
                grid_levels_long = [initial_entry_long] + [
                    current_price - buffer_distance_long - i * ((max_outer_price_distance * current_price - buffer_distance_long) / (levels - 1)) 
                    for i in range(1, levels)
                ]
            else:
                grid_levels_long = [
                    current_price - buffer_distance_long - i * ((max_outer_price_distance * current_price - buffer_distance_long) / levels) 
                    for i in range(levels)
                ]

            if short_pos_qty == 0:
                grid_levels_short = [initial_entry_short] + [
                    current_price + buffer_distance_short + i * ((max_outer_price_distance * current_price - buffer_distance_short) / (levels - 1)) 
                    for i in range(1, levels)
                ]
            else:
                grid_levels_short = [
                    current_price + buffer_distance_short + i * ((max_outer_price_distance * current_price - buffer_distance_short) / levels) 
                    for i in range(levels)
                ]

            def find_nearest_significant_level(level, significant_levels, tolerance, min_distance, max_distance, current_price, previous_level=None):
                for sig_level in significant_levels:
                    if abs(level - sig_level) / level < tolerance:
                        if current_price - max_distance * current_price <= sig_level <= current_price - min_distance * current_price:
                            if previous_level is None or abs(sig_level - previous_level) > (max_distance * current_price / levels):
                                return sig_level
                return level

            tolerance = 0.01
            adjusted_grid_levels_long = []
            adjusted_grid_levels_short = []

            for i, level in enumerate(grid_levels_long):
                previous_level = adjusted_grid_levels_long[-1] if adjusted_grid_levels_long else None
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels_long,
                    tolerance,
                    min_outer_price_distance,
                    max_outer_price_distance,
                    current_price,
                    previous_level
                )
                if current_price - max_outer_price_distance * current_price <= adjusted_level <= current_price - buffer_distance_long:
                    adjusted_grid_levels_long.append(adjusted_level)

            for i, level in enumerate(grid_levels_short):
                previous_level = adjusted_grid_levels_short[-1] if adjusted_grid_levels_short else None
                adjusted_level = find_nearest_significant_level(
                    level,
                    significant_levels_short,
                    tolerance,
                    min_outer_price_distance,
                    max_outer_price_distance,
                    current_price,
                    previous_level
                )
                if current_price + buffer_distance_short <= adjusted_level <= current_price + max_outer_price_distance * current_price:
                    adjusted_grid_levels_short.append(adjusted_level)

            grid_levels_long = adjusted_grid_levels_long
            grid_levels_short = adjusted_grid_levels_short

            if len(grid_levels_long) < levels:
                remaining_levels = levels - len(grid_levels_long)
                additional_levels_long = np.linspace(
                    grid_levels_long[-1] if grid_levels_long else current_price - buffer_distance_long,
                    current_price - max_outer_price_distance * current_price,
                    remaining_levels + 1
                )[1:]
                grid_levels_long = np.concatenate((grid_levels_long, additional_levels_long))

            if len(grid_levels_short) < levels:
                remaining_levels = levels - len(grid_levels_short)
                additional_levels_short = np.linspace(
                    grid_levels_short[-1] if grid_levels_short else current_price + buffer_distance_short,
                    current_price + max_outer_price_distance * current_price,
                    remaining_levels + 1
                )[1:]
                grid_levels_short = np.concatenate((grid_levels_short, additional_levels_short))

            grid_levels_long = sorted(grid_levels_long, reverse=True)
            grid_levels_short = sorted(grid_levels_short)

            logging.info(f"[{symbol}] Initial long entry level: {initial_entry_long}")
            logging.info(f"[{symbol}] Initial short entry level: {initial_entry_short}")
            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")
            
            # The rest of the function would continue here...
            
            # # Sort the grid levels in ascending order
            # grid_levels_long = sorted(grid_levels_long, reverse=True)
            # grid_levels_short = sorted(grid_levels_short)

            # logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            # logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Initialize last_empty_grid_time for symbol if not present
            if symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {'long': 0, 'short': 0}

            # Check for grid replacement conditions
            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            # Track the last time the grid was emptied
            if replace_empty_long_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            if replace_empty_short_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}

            current_time = time.time()

            # Check and log if the symbol is in max_qty_reached_symbol_long
            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")

            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            # Additional logic for managing open symbols and checking trading permissions
            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"
            
            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                logging.info(f"Allowed symbol: {symbol}")

                replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                    symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                    dynamic_outer_price_distance=dynamic_outer_price_distance
                )

                # Replace long grid if conditions are met
                if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', 0) > 240))) and not self.auto_reduce_active_long.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_long:
                        logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'buy')
                        buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                        buffer_distance_long = current_price * buffer_percentage_long
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['long'] = current_time
                        logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol long cannot replace grid")

                # Replace short grid if conditions are met
                if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', 0) > 240))) and not self.auto_reduce_active_short.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_short:
                        logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'sell')
                        buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                        buffer_distance_short = current_price * buffer_percentage_short
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['short'] = current_time
                        logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol short cannot replace grid")

                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            time.sleep(5)

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def linear_grid_hardened_gridspan_ob_volumelevels_dynamictp(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                                                short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                                                strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
                                                wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
                                                user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
                                                short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                                                symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
                                                max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                                                max_qty_percent_long: float, max_qty_percent_short: float):
        try:
            # Calculate dynamic outer price distance based on 4h candle spread
            spread = self.get_4h_candle_spread(symbol)
            logging.info(f"4h Candle spread for {symbol}: {spread}")

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")
            
            # Ensure dynamic outer price distance is not too tight
            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))
            
            logging.info(f"Dynamic outer price distance for {symbol} : {dynamic_outer_price_distance}")
            
            # Ensure the outer price distance can span all levels
            required_distance = outer_price_distance / levels
            if dynamic_outer_price_distance < required_distance:
                logging.info(f"Dynamic outer price distance {dynamic_outer_price_distance} is less than required distance {required_distance}. Adjusting it.")
                dynamic_outer_price_distance = required_distance

            logging.info(f"Dynamic outer price distance after spread: {dynamic_outer_price_distance}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Analyze orderbook depth and identify significant price levels
            min_price = current_price - max_outer_price_distance * current_price
            max_price = current_price + max_outer_price_distance * current_price

            # Create a histogram of orderbook volume within the price range
            price_range = np.arange(min_price, max_price, (max_price - min_price) / 100)
            volume_histogram_long = np.zeros_like(price_range)
            volume_histogram_short = np.zeros_like(price_range)

            for order in order_book['bids']:
                price, volume = order[0], order[1]
                if min_price <= price <= current_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_long[index] += volume

            for order in order_book['asks']:
                price, volume = order[0], order[1]
                if current_price <= price <= max_price:
                    index = int((price - min_price) / (max_price - min_price) * 100)
                    volume_histogram_short[index] += volume

            # Identify significant price levels based on volume histogram
            volume_threshold_long = np.mean(volume_histogram_long) * 1.5  # Adjust the threshold as needed
            significant_levels_long = price_range[volume_histogram_long >= volume_threshold_long]

            volume_threshold_short = np.mean(volume_histogram_short) * 1.5  # Adjust the threshold as needed
            significant_levels_short = price_range[volume_histogram_short >= volume_threshold_short]

            # Calculate grid levels based on dynamic_outer_price_distance
            grid_levels_long = [current_price - i * dynamic_outer_price_distance * current_price for i in range(1, levels + 1)]
            grid_levels_short = [current_price + i * dynamic_outer_price_distance * current_price for i in range(1, levels + 1)]

            # Function to find the nearest significant level
            def find_nearest_significant_level(level, significant_levels, tolerance, min_distance, max_distance, current_price):
                for sig_level in significant_levels:
                    if abs(level - sig_level) / level < tolerance:
                        if current_price - max_distance * current_price <= sig_level <= current_price - min_distance * current_price:
                            return sig_level
                return level

            # Adjust grid levels to align with significant levels
            tolerance = 0.01  # 1% tolerance, adjust as needed
            grid_levels_long = [
                find_nearest_significant_level(
                    level, 
                    significant_levels_long, 
                    tolerance, 
                    buffer_distance_long / current_price, 
                    min_outer_price_distance, 
                    current_price
                ) for level in grid_levels_long
            ]
            grid_levels_short = [
                find_nearest_significant_level(
                    level, 
                    significant_levels_short, 
                    tolerance, 
                    buffer_distance_short / current_price, 
                    min_outer_price_distance, 
                    current_price
                ) for level in grid_levels_short
            ]

            # Ensure the grid levels are within the buffer distances and respect min/max outer price distance
            grid_levels_long = [
                level for level in grid_levels_long 
                if current_price - min_outer_price_distance * current_price <= level <= current_price - buffer_distance_long
            ]
            grid_levels_short = [
                level for level in grid_levels_short 
                if current_price + buffer_distance_short <= level <= current_price + min_outer_price_distance * current_price
            ]

            # Ensure the desired number of grid levels is achieved
            if len(grid_levels_long) < levels:
                additional_levels_long = np.linspace(
                    current_price - min_outer_price_distance * current_price, 
                    current_price - buffer_distance_long, 
                    levels - len(grid_levels_long)
                )
                grid_levels_long = np.concatenate((grid_levels_long, additional_levels_long))

            if len(grid_levels_short) < levels:
                additional_levels_short = np.linspace(
                    current_price + buffer_distance_short, 
                    current_price + min_outer_price_distance * current_price, 
                    levels - len(grid_levels_short)
                )
                grid_levels_short = np.concatenate((grid_levels_short, additional_levels_short))

            # Sort the grid levels in ascending order
            grid_levels_long = sorted(grid_levels_long, reverse=True)
            grid_levels_short = sorted(grid_levels_short)

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            # The rest of the function would continue here...
            
            # # Sort the grid levels in ascending order
            # grid_levels_long = sorted(grid_levels_long, reverse=True)
            # grid_levels_short = sorted(grid_levels_short)

            # logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            # logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Check for grid replacement conditions
            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            # Track the last time the grid was emptied
            if replace_empty_long_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            if replace_empty_short_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}

            current_time = time.time()

            # Check and log if the symbol is in max_qty_reached_symbol_long
            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")

            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            # Additional logic for managing open symbols and checking trading permissions
            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"
            
            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                logging.info(f"Allowed symbol: {symbol}")

                replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                    symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                    dynamic_outer_price_distance=dynamic_outer_price_distance
                )

                # Replace long grid if conditions are met
                if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', 0) > 240))) and not self.auto_reduce_active_long.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_long:
                        logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'buy')
                        buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                        buffer_distance_long = current_price * buffer_percentage_long
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['long'] = current_time
                        logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol long cannot replace grid")

                # Replace short grid if conditions are met
                if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', 0) > 240))) and not self.auto_reduce_active_short.get(symbol, False):
                    if symbol not in self.max_qty_reached_symbol_short:
                        logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid timeout.")
                        self.clear_grid(symbol, 'sell')
                        buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                        buffer_distance_short = current_price * buffer_percentage_short
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                        self.last_empty_grid_time[symbol]['short'] = current_time
                        logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                    else:
                        logging.info(f"{symbol} is in max qty reached symbol short cannot replace grid")

                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            time.sleep(5)

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())


    def linear_grid_hardened_gridspan_ob_volumes(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
                                                short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
                                                strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
                                                wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
                                                user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
                                                short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
                                                symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
                                                max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
                                                max_qty_percent_long: float, max_qty_percent_short: float):
        try:
            # Calculate dynamic outer price distance based on 4h candle spread
            spread = self.get_4h_candle_spread(symbol)
            logging.info(f"4h Candle spread for {symbol}: {spread}")

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")
            
            # Ensure dynamic outer price distance is not too tight
            dynamic_outer_price_distance = max(min_outer_price_distance, min(max_outer_price_distance, spread))
            
            logging.info(f"Dynamic outer price distance for {symbol} : {dynamic_outer_price_distance}")
            
            # Ensure the outer price distance can span all levels
            required_distance = outer_price_distance / levels
            if dynamic_outer_price_distance < required_distance:
                logging.info(f"Dynamic outer price distance {dynamic_outer_price_distance} is less than required distance {required_distance}. Adjusting it.")
                dynamic_outer_price_distance = required_distance

            logging.info(f"Dynamic outer price distance after spread: {dynamic_outer_price_distance}")

            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate the grid levels ensuring they start from the buffer distance and span the dynamic outer price distance
            price_range_long = dynamic_outer_price_distance * current_price
            price_range_short = dynamic_outer_price_distance * current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength

            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls_properdca(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short, long_pos_qty=long_pos_qty, short_pos_qty=short_pos_qty
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional_properdca(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='buy')
            amounts_short = self.calculate_order_amounts_notional_properdca(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid, long_pos_qty, short_pos_qty, side='sell')
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Check for grid replacement conditions
            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            # Track the last time the grid was emptied
            if replace_empty_long_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            if replace_empty_short_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}

            current_time = time.time()

            # Check and log if the symbol is in max_qty_reached_symbol_long
            if symbol in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_long")

            if symbol in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Symbol is in max_qty_reached_symbol_short")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                dynamic_outer_price_distance=dynamic_outer_price_distance
            )

            # Replace long grid if conditions are met
            if (replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', 0) > 240))) and not self.auto_reduce_active_long.get(symbol, False):
                if symbol not in self.max_qty_reached_symbol_long:
                    logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid timeout.")
                    self.clear_grid(symbol, 'buy')
                    buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                    buffer_distance_long = current_price * buffer_percentage_long
                    price_range_long = dynamic_outer_price_distance * current_price
                    grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                    self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                    self.active_grids.add(symbol)
                    self.last_empty_grid_time[symbol]['long'] = current_time
                    logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                else:
                    logging.info(f"{symbol} is in max qty reached symbol long cannot replace grid")

            # Replace short grid if conditions are met
            if (replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', 0) > 240))) and not self.auto_reduce_active_short.get(symbol, False):
                if symbol not in self.max_qty_reached_symbol_short:
                    logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid timeout.")
                    self.clear_grid(symbol, 'sell')
                    buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                    buffer_distance_short = current_price * buffer_percentage_short
                    price_range_short = dynamic_outer_price_distance * current_price
                    grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                    self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                    self.active_grids.add(symbol)
                    self.last_empty_grid_time[symbol]['short'] = current_time
                    logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                else:
                    logging.info(f"{symbol} is in max qty reached symbol short cannot replace grid")
                    
            # Additional logic for managing open symbols and checking trading permissions
            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"
            mfi_signal_neutral = mfirsi_signal.lower() == "neutral"

            if len(open_symbols) < symbols_allowed or symbol in open_symbols:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or symbol in open_symbols: #(symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities with {mfirsi_signal}")
                logging.info(f"[{symbol}] MFIRSI Long: {mfi_signal_long}")
                logging.info(f"[{symbol}] MFIRSI Short: {mfi_signal_short}")
                logging.info(f"[{symbol}] MFIRSI Neutral: {mfi_signal_neutral}")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())


    def linear_grid_hardened_gridspan_orderbook_levels_atrp_maxposqty(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
        max_qty_percent_long: float, max_qty_percent_short: float
    ):
        try:
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            # Fetch the order book to calculate the average spread
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
            average_spread = (best_ask_price - best_bid_price) / current_price

            # Calculate dynamic buffer percentage based on average spread
            buffer_percentage_long = min_buffer_percentage + (average_spread * (max_buffer_percentage - min_buffer_percentage))
            buffer_percentage_short = min_buffer_percentage + (average_spread * (max_buffer_percentage - min_buffer_percentage))

            # Ensure buffer percentage is within specified bounds
            buffer_percentage_long = min(max(buffer_percentage_long, min_buffer_percentage), max_buffer_percentage)
            buffer_percentage_short = min(max(buffer_percentage_short, min_buffer_percentage), max_buffer_percentage)

            # Use initial_entry_buffer_pct if the position quantity is zero
            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else buffer_percentage_long
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else buffer_percentage_short

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

            # Fetch ATRP for the specific timeframe
            atrp_timeframe = "1m"
            atrp_period = 14
            atrp = self.get_atrp(symbol, timeframe=atrp_timeframe, period=atrp_period)
            logging.info(f"[{symbol}] ATRP value for {atrp_timeframe} timeframe: {atrp}")

            # Calculate dynamic outer price distance using scaled ATRP
            outer_price_distance = self.calculate_dynamic_outer_price_distance_atr(atrp, min_outer_price_distance, max_outer_price_distance)

            # Calculate grid levels based on order book prices
            grid_levels_long = self.calculate_grid_levels_orderbook_based(
                symbol, current_price, buffer_distance_long, levels, 'buy', min_outer_price_distance, max_outer_price_distance, strength
            )
            grid_levels_short = self.calculate_grid_levels_orderbook_based(
                symbol, current_price, buffer_distance_short, levels, 'sell', min_outer_price_distance, max_outer_price_distance, strength
            )

            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                grid_levels_long = self.calculate_grid_levels_orderbook_based(
                    symbol, current_price, buffer_distance_long, levels, 'buy', min_outer_price_distance, max_outer_price_distance, strength
                )
                grid_levels_short = self.calculate_grid_levels_orderbook_based(
                    symbol, current_price, buffer_distance_short, levels, 'sell', min_outer_price_distance, max_outer_price_distance, strength
                )

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                dynamic_outer_price_distance=outer_price_distance
            )

            has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
            has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

            replace_empty_long_grid = (long_pos_qty > 0 and not has_open_long_order)
            replace_empty_short_grid = (short_pos_qty > 0 and not has_open_short_order)

            # Track the last time the grid was emptied
            if replace_empty_long_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            if replace_empty_short_grid and symbol not in self.last_empty_grid_time:
                self.last_empty_grid_time[symbol] = {}
            
            current_time = time.time()

            # Replace long grid if conditions are met and it has been more than 5 minutes since it was last emptied
            if replace_long_grid or (replace_empty_long_grid and (current_time - self.last_empty_grid_time[symbol].get('long', 0) > 240)):
                if not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                    logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer or empty grid for open position.")
                    self.clear_grid(symbol, 'buy')
                    buffer_percentage_long = min_buffer_percentage + (average_spread * (max_buffer_percentage - min_buffer_percentage))
                    buffer_percentage_long = min(max(buffer_percentage_long, min_buffer_percentage), max_buffer_percentage)
                    buffer_distance_long = current_price * buffer_percentage_long
                    outer_price_distance_long = current_price * outer_price_distance
                    grid_levels_long = self.calculate_grid_levels_orderbook_based(
                        symbol, current_price, buffer_distance_long, levels, 'buy', min_outer_price_distance, max_outer_price_distance, strength
                    )
                    self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                    self.active_grids.add(symbol)
                    logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")
                    self.last_empty_grid_time[symbol]['long'] = current_time

            # Replace short grid if conditions are met and it has been more than 5 minutes since it was last emptied
            if replace_short_grid or (replace_empty_short_grid and (current_time - self.last_empty_grid_time[symbol].get('short', 0) > 240)):
                if not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                    logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer or empty grid for open position.")
                    self.clear_grid(symbol, 'sell')
                    buffer_percentage_short = min_buffer_percentage + (average_spread * (max_buffer_percentage - min_buffer_percentage))
                    buffer_percentage_short = min(max(buffer_percentage_short, min_buffer_percentage), max_buffer_percentage)
                    buffer_distance_short = current_price * buffer_percentage_short
                    outer_price_distance_short = current_price * outer_price_distance
                    grid_levels_short = self.calculate_grid_levels_orderbook_based(
                        symbol, current_price, buffer_distance_short, levels, 'sell', min_outer_price_distance, max_outer_price_distance, strength
                    )
                    self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                    self.active_grids.add(symbol)
                    logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                    self.last_empty_grid_time[symbol]['short'] = current_time

            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                # Check if auto-reduce is not active for long position
                if not self.auto_reduce_active_long.get(symbol, False):
                    logging.info(f"Auto-reduce for long position on {symbol} is not active")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                    if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                # Check if auto-reduce is not active for short position
                if not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for short position on {symbol} is not active")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                else:
                    logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                    if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                        else:
                            logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        

            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())


    def linear_grid_hardened_gridspan_orderbook_levels_maxposqty(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, min_outer_price_distance: float, max_outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
        max_qty_percent_long: float, max_qty_percent_short: float
    ):
        try:

            # Initial checks and setup
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Calculate max position quantities
            max_qty_long, max_qty_short = self.calculate_max_positions(symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            # Check precision and minimum quantity for trading on Bybit
            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price, best_bid_price = self.get_best_prices(symbol, order_book, current_price)

            # buffer_distance_long, buffer_distance_short = self.calculate_buffers(
            #     symbol, current_price, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty, 
            #     initial_entry_buffer_pct, min_buffer_percentage, max_buffer_percentage
            # )

            # Calculate buffers with dynamic adjustment based on order book
            buffer_distance_long, buffer_distance_short = self.calculate_buffers(
                symbol, current_price, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty, 
                initial_entry_buffer_pct, min_buffer_percentage, max_buffer_percentage, order_book
            )

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            grid_levels_long, grid_levels_short = self.calculate_grid_levels_based_on_order_book(
                order_book, current_price, levels, strength, max_outer_price_distance, min_outer_price_distance)

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            # Calculate total amounts for long and short positions
            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            # Auto-reduce and grid replacement logic
            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                buffer_distance_long = current_price * buffer_percentage_long
                dynamic_outer_price_distance_long = self.calculate_dynamic_outer_price_distance_orderbook(order_book, current_price, max_outer_price_distance=max_outer_price_distance, min_outer_price_distance=min_outer_price_distance)
                outer_price_distance_long = current_price * dynamic_outer_price_distance_long
                grid_levels_long = [current_price - buffer_distance_long - (outer_price_distance_long - buffer_distance_long) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                buffer_distance_short = current_price * buffer_percentage_short
                dynamic_outer_price_distance_short = self.calculate_dynamic_outer_price_distance_orderbook(order_book, current_price, max_outer_price_distance=max_outer_price_distance, min_outer_price_distance=min_outer_price_distance)
                outer_price_distance_short = current_price * dynamic_outer_price_distance_short
                grid_levels_short = [current_price + buffer_distance_short + (outer_price_distance_short - buffer_distance_short) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_long:
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_short:
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["buy"].clear()

                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)

                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["sell"].clear()

                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )
        except Exception as e:
            logging.error(f"Error in linear_grid_hardened_gridspan_orderbook_maxposqty: {str(e)}")
            logging.info("Traceback: %s", traceback.format_exc())

    def linear_grid_hardened_gridspan_orderbook_maxposqty(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
        max_qty_percent_long: float, max_qty_percent_short: float
    ):
        try:
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)

            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            dynamic_outer_price_distance = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)

            outer_price_long = current_price * (1 - dynamic_outer_price_distance)
            outer_price_short = current_price * (1 + dynamic_outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting dynamic outer price distance.")
                adjustment_factor = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - adjustment_factor)
                outer_price_short = current_price * (1 + adjustment_factor)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_min_outerpricedist_v2(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                dynamic_outer_price_distance=dynamic_outer_price_distance
            )

            # replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer(
            #     symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty, min_buffer_percentage, max_buffer_percentage
            # )

            # replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_outerpricedist(
            #     symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
            #     dynamic_outer_price_distance=dynamic_outer_price_distance

            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                buffer_distance_long = current_price * buffer_percentage_long
                dynamic_outer_price_distance_long = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)
                outer_price_distance_long = current_price * dynamic_outer_price_distance_long
                grid_levels_long = [current_price - buffer_distance_long - (outer_price_distance_long - buffer_distance_long) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                buffer_distance_short = current_price * buffer_percentage_short
                dynamic_outer_price_distance_short = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)
                outer_price_distance_short = current_price * dynamic_outer_price_distance_short
                grid_levels_short = [current_price + buffer_distance_short + (outer_price_distance_short - buffer_distance_short) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            # max_qty_long = total_equity * (max_qty_percent_long / 100) / current_price
            # max_qty_short = total_equity * (max_qty_percent_short / 100) / current_price

            max_qty_long, max_qty_short = self.calculate_max_positions(symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_long:
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_short:
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["buy"].clear()

                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)

                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["sell"].clear()

                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            else:
                logging.info(f"[{symbol}] Trading not allowed or MFIRSI Signal not met. Skipping grid placement.")
                time.sleep(5)

            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )
        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def linear_grid_hardened_gridspan_orderbook(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool
    ):
        try:
            # Check reissue necessity for both long and short positions
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            # Initialize filled levels if not already present
            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            # Check if grids for buying or selling are active
            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            # Get current market price and log it
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Set buffer percentages based on whether positions are open
            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            # Calculate buffer distances from the current market price
            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            # Log the calculated buffer distances
            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            # Fetch order book data to get best ask and bid prices
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate dynamic outer price distance based on order book
            dynamic_outer_price_distance = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)

            # Calculate grid levels using the current price, dynamically calculated outer price distance, and buffer distances
            outer_price_long = current_price * (1 - dynamic_outer_price_distance)
            outer_price_short = current_price * (1 + dynamic_outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Check if long and short grid levels overlap and adjust if necessary
            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting dynamic outer price distance.")
                adjustment_factor = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - adjustment_factor)
                outer_price_short = current_price * (1 + adjustment_factor)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Log calculated grid levels
            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")


            # Check precision and minimum quantity for trading on Bybit
            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            # Calculate the total notional amounts for long and short sides based on exposure limits and leverage
            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            # Log the total amounts calculated for long and short trades
            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            # Calculate the individual order amounts for the long and short grids
            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            # Log auto-reduce status for long position
            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            # Log auto-reduce status for short position
            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Determine if grids need replacement based on dynamic buffer
            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            # Replace long grid if necessary
            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                buffer_distance_long = current_price * buffer_percentage_long
                dynamic_outer_price_distance_long = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)
                outer_price_distance_long = current_price * dynamic_outer_price_distance_long
                grid_levels_long = [current_price - buffer_distance_long - (outer_price_distance_long - buffer_distance_long) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            # Replace short grid if necessary
            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                buffer_distance_short = current_price * buffer_percentage_short
                dynamic_outer_price_distance_short = self.calculate_dynamic_outer_price_distance(order_book, current_price, max_outer_price_distance=outer_price_distance)
                outer_price_distance_short = current_price * dynamic_outer_price_distance_short
                grid_levels_short = [current_price + buffer_distance_short + (outer_price_distance_short - buffer_distance_short) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            open_symbols = list(set(open_symbols))  # Ensure symbols are unique
            logging.info(f"Open symbols {open_symbols}")

            # Check if trading a new symbol is allowed based on open symbols and allowed count
            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            # Set MFI signals for long and short
            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                # Reissue orders if necessary based on the reissue threshold and current orders
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                    #logging.info(f"Open orders for {symbol}: {open_orders}")

                    # Flags to check for existing buy or sell orders, excluding reduce-only orders
                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    # Handling reissue for long orders
                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')  # Clear existing long orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    # Handling reissue for short orders
                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')  # Clear existing short orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                # Check if there are open positions and no active grids
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    if short_pos_qty > 0 and not short_grid_active:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                current_time = datetime.now()
                
                # Logic to clear grids if there are no open positions and it's time to clear them
                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')  # Clear any lingering buy grid orders
                        self.clear_grid(symbol, 'sell')  # Clear any lingering sell grid orders
                        self.active_grids.discard(symbol)  # Remove the symbol from active grids
                        self.last_cleared_time[symbol] = current_time  # Update the last cleared time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["sell"].clear()

                        # Place new long and short grid orders only if there are no existing orders (excluding TP orders) and no active grids
                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having active grids
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["buy"].clear()

                            # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["sell"].clear()

                            # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")
                
            # Check if there is room for trading new symbols
            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
            if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                # Place grid orders for the new symbol if MFI signal is present and auto-reduce is not blocking
                if long_mode and mfi_signal_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")
                if short_mode and mfi_signal_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"[{symbol}] Trading not allowed or MFIRSI Signal not met. Skipping grid placement.")
                time.sleep(5)
                
            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )
        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def linear_grid_hardened_gridspan_maxposqty(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
        max_qty_percent_long: float, max_qty_percent_short: float
    ):
        try:
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            outer_price_long = current_price * (1 - outer_price_distance)
            outer_price_short = current_price * (1 + outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                buffer_distance_long = current_price * buffer_percentage_long
                outer_price_distance_long = current_price * outer_price_distance
                grid_levels_long = [current_price - buffer_distance_long - (outer_price_distance_long - buffer_distance_long) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                buffer_distance_short = current_price * buffer_percentage_short
                outer_price_distance_short = current_price * outer_price_distance
                grid_levels_short = [current_price + buffer_distance_short + (outer_price_distance_short - buffer_distance_short) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")


            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_long:
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_short:
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["buy"].clear()

                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)

                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["sell"].clear()

                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            else:
                logging.info(f"[{symbol}] Trading not allowed or MFIRSI Signal not met. Skipping grid placement.")
                time.sleep(5)

            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())
                            

    def linear_grid_hardened_gridspan(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool
    ):
        try:
            # Check reissue necessity for both long and short positions
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            # Initialize filled levels if not already present
            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            # Check if grids for buying or selling are active
            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            # Get current market price and log it
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Set buffer percentages based on whether positions are open
            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            # Calculate buffer distances from the current market price
            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            # Log the calculated buffer distances
            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            # Fetch order book data to get best ask and bid prices
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate grid levels using the current price, outer price distance, and buffer distances
            outer_price_long = current_price * (1 - outer_price_distance)
            outer_price_short = current_price * (1 + outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Check if long and short grid levels overlap and adjust if necessary
            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Log calculated grid levels
            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            # Check precision and minimum quantity for trading on Bybit
            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            # Calculate the total notional amounts for long and short sides based on exposure limits and leverage
            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            # Log the total amounts calculated for long and short trades
            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            # Calculate the individual order amounts for the long and short grids
            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            # Log auto-reduce status for long position
            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            # Log auto-reduce status for short position
            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Determine if grids need replacement based on dynamic buffer
            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            # Replace long grid if necessary
            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
                buffer_distance_long = current_price * buffer_percentage_long
                outer_price_distance_long = current_price * outer_price_distance  # Direct usage of outer price distance
                grid_levels_long = [current_price - buffer_distance_long - (outer_price_distance_long - buffer_distance_long) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            # Replace short grid if necessary
            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)
                buffer_distance_short = current_price * buffer_percentage_short
                outer_price_distance_short = current_price * outer_price_distance  # Direct usage of outer price distance
                grid_levels_short = [current_price + buffer_distance_short + (outer_price_distance_short - buffer_distance_short) * factor for factor in np.linspace(0.0, 1.0, num=levels)**strength]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")
                
            open_symbols = list(set(open_symbols))  # Ensure symbols are unique
            logging.info(f"Open symbols {open_symbols}")

            # Check if trading a new symbol is allowed based on open symbols and allowed count
            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            # Set MFI signals for long and short
            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            # self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                # Reissue orders if necessary based on the reissue threshold and current orders
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                    #logging.info(f"Open orders for {symbol}: {open_orders}")

                    # Flags to check for existing buy or sell orders, excluding reduce-only orders
                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    # Handling reissue for long orders
                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')  # Clear existing long orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    # Handling reissue for short orders
                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')  # Clear existing short orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                # Check if there are open positions and no active grids
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    if short_pos_qty > 0 and not short_grid_active:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                current_time = datetime.now()
                
                # Logic to clear grids if there are no open positions and it's time to clear them
                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')  # Clear any lingering buy grid orders
                        self.clear_grid(symbol, 'sell')  # Clear any lingering sell grid orders
                        self.active_grids.discard(symbol)  # Remove the symbol from active grids
                        self.last_cleared_time[symbol] = current_time  # Update the last cleared time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["sell"].clear()

                        # Place new long and short grid orders only if there are no existing orders (excluding TP orders) and no active grids
                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having active grids
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["buy"].clear()

                            # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["sell"].clear()

                            # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")
                
            # Check if there is room for trading new symbols
            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
            if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                # Place grid orders for the new symbol if MFI signal is present and auto-reduce is not blocking
                if long_mode and mfi_signal_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")
                if short_mode and mfi_signal_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"[{symbol}] Trading not allowed or MFIRSI Signal not met. Skipping grid placement.")
                time.sleep(5)
                
            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )
        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def adjust_outer_price_distance(self, current_price, buffer_distance, levels, strength):
        # Calculate the base outer distance without the buffer effect
        base_outer_distance = buffer_distance  # This could be a set value or calculated from other parameters

        # Calculate the spread needed beyond the buffer to place all levels effectively
        additional_spread = (levels - 1) * base_outer_distance  # Simplified assumption of linear spread

        # Factor in the non-linear distribution if needed
        max_factor = max(np.linspace(0.0, 1.0, num=levels)**strength)
        adjusted_outer_distance = base_outer_distance + (additional_spread * max_factor)

        return adjusted_outer_distance / current_price  # Convert absolute to relative distance

    def linear_grid_dynamictp_linspaced_maxposqty(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool,
        max_qty_percent_long: float, max_qty_percent_short: float
    ):
        try:
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            open_symbols = list(set(open_symbols))
            logging.info(f"Open symbols {open_symbols}")

            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            self.check_and_manage_positions(long_pos_qty, short_pos_qty, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            outer_price_long = current_price * (1 - outer_price_distance)
            outer_price_short = current_price * (1 + outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_dynamic(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)

                buffer_distance_long = current_price * buffer_percentage_long

                adjusted_outer_price_distance_long = self.adjust_outer_price_distance(
                    current_price, buffer_distance_long, self.levels, self.strength
                )

                outer_price_long = current_price * (1 - adjusted_outer_price_distance_long)
                price_range_long = current_price - outer_price_long
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in np.linspace(0.0, 1.0, num=self.levels)**self.strength]

                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)

                buffer_distance_short = current_price * buffer_percentage_short

                adjusted_outer_price_distance_short = self.adjust_outer_price_distance(
                    current_price, buffer_distance_short, self.levels, self.strength
                )

                outer_price_short = current_price * (1 + adjusted_outer_price_distance_short)
                price_range_short = outer_price_short - current_price
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in np.linspace(0.0, 1.0, num=self.levels)**self.strength]

                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            max_qty_long, max_qty_short = self.calculate_max_positions(symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short)

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False) and symbol not in self.max_qty_reached_symbol_long:
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False) and symbol not in self.max_qty_reached_symbol_short:
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')
                                self.active_grids.discard(symbol)
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
            else:
                logging.info(f"Open symbols is {open_symbols} and symbols allowed is {symbols_allowed}")

            if symbol in open_symbols or trading_allowed:
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active and symbol not in self.max_qty_reached_symbol_long:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)
                    if short_pos_qty > 0 and not short_grid_active and symbol not in self.max_qty_reached_symbol_short:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')
                            self.active_grids.discard(symbol)
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)

                current_time = datetime.now()

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')
                        self.clear_grid(symbol, 'sell')
                        self.active_grids.discard(symbol)
                        self.last_cleared_time[symbol] = current_time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_long:
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and symbol not in self.max_qty_reached_symbol_short:
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)
                            self.filled_levels[symbol]["sell"].clear()

                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["buy"].clear()

                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)

                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)
                                self.filled_levels[symbol]["sell"].clear()

                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_long:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0) and symbol not in self.max_qty_reached_symbol_short:
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
            else:
                logging.info(f"Symbol {symbol} not in open_symbols: {open_symbols} or trading not allowed")

            # Determine if there are open long and short positions based on provided quantities
            has_open_long_position = long_pos_qty > 0
            has_open_short_position = short_pos_qty > 0

            logging.info(f"{symbol} has long position: {has_open_long_position}, has short position: {has_open_short_position}")

            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")

            if (len(open_symbols) < symbols_allowed and symbol not in self.active_grids) or (symbol in open_symbols and (not has_open_long_position or not has_open_short_position)):
                logging.info(f"[{symbol}] Checking for new trading opportunities.")

                if long_mode and mfi_signal_long and not has_open_long_position and symbol not in self.max_qty_reached_symbol_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                if short_mode and mfi_signal_short and not has_open_short_position and symbol not in self.max_qty_reached_symbol_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")
                        
            else:
                logging.info(f"[{symbol}] Trading not allowed or MFIRSI Signal not met. Skipping grid placement.")
                time.sleep(5)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )
        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())

    def linear_grid_dynamictp_linspaced(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool
    ):
        try:
            # Check reissue necessity for both long and short positions
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            # Initialize filled levels if not already present
            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            open_symbols = list(set(open_symbols))  # Ensure symbols are unique
            logging.info(f"Open symbols {open_symbols}")

            # Check if grids for buying or selling are active
            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            # Get current market price and log it
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Set buffer percentages based on whether positions are open
            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            # Calculate buffer distances from the current market price
            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            # Log the calculated buffer distances
            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            # Fetch order book data to get best ask and bid prices
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate grid levels using the current price, outer price distance, and buffer distances
            outer_price_long = current_price * (1 - outer_price_distance)
            outer_price_short = current_price * (1 + outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Check if long and short grid levels overlap and adjust if necessary
            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Log calculated grid levels
            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            # Check precision and minimum quantity for trading on Bybit
            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            # Calculate the total notional amounts for long and short sides based on exposure limits and leverage
            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            # Log the total amounts calculated for long and short trades
            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            # Calculate the individual order amounts for the long and short grids
            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            # Log auto-reduce status for long position
            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            # Log auto-reduce status for short position
            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Determine if the grid needs to be replaced based on the dynamic buffer adjustments
            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer_dynamic(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            # Replace long grid if necessary
            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)

                # Calculate buffer distance from the current market price
                buffer_distance_long = current_price * buffer_percentage_long

                # Dynamically adjust outer price distance to ensure proper level spread
                adjusted_outer_price_distance_long = self.adjust_outer_price_distance(
                    current_price, buffer_distance_long, self.levels, self.strength
                )
                
                # Recalculate price range and grid levels using the adjusted distance
                outer_price_long = current_price * (1 - adjusted_outer_price_distance_long)
                price_range_long = current_price - outer_price_long
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in np.linspace(0.0, 1.0, num=self.levels)**self.strength]

                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            # Replace short grid if necessary
            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)

                # Calculate buffer distance from the current market price
                buffer_distance_short = current_price * buffer_percentage_short

                # Dynamically adjust outer price distance to ensure proper level spread
                adjusted_outer_price_distance_short = self.adjust_outer_price_distance(
                    current_price, buffer_distance_short, self.levels, self.strength
                )

                # Recalculate price range and grid levels using the adjusted distance
                outer_price_short = current_price * (1 + adjusted_outer_price_distance_short)
                price_range_short = outer_price_short - current_price
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in np.linspace(0.0, 1.0, num=self.levels)**self.strength]

                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                self.active_grids.add(symbol)
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")


            # # Replace long grid if necessary
            # if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False):
            #     logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
            #     self.clear_grid(symbol, 'buy')
            #     self.active_grids.discard(symbol)

            #     # Adjust outer price distance dynamically
            #     adjusted_outer_price_distance_long = self.adjust_outer_price_distance(
            #         current_price, long_pos_price, outer_price_distance, buffer_percentage_long
            #     )
                
            #     # Recalculate price range and grid levels using the adjusted distance
            #     outer_price_long = current_price * (1 - adjusted_outer_price_distance_long)
            #     price_range_long = current_price - outer_price_long
            #     grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in np.linspace(0.0, 1.0, num=self.levels)**self.strength]

            #     self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
            #     self.active_grids.add(symbol)
            #     logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            # # Replace short grid if necessary
            # if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False):
            #     logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
            #     self.clear_grid(symbol, 'sell')
            #     self.active_grids.discard(symbol)

            #     # Adjust outer price distance dynamically
            #     adjusted_outer_price_distance_short = self.adjust_outer_price_distance(
            #         current_price, short_pos_price, outer_price_distance, buffer_percentage_short
            #     )

            #     # Recalculate price range and grid levels using the adjusted distance
            #     outer_price_short = current_price * (1 + adjusted_outer_price_distance_short)
            #     price_range_short = outer_price_short - current_price
            #     grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in np.linspace(0.0, 1.0, num=self.levels)**self.strength]

            #     self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
            #     self.active_grids.add(symbol)
            #     logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")


            # # Replace long grid if necessary
            # if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False):
            #     logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
            #     self.clear_grid(symbol, 'buy')  # Cancel existing long grid orders
            #     self.active_grids.discard(symbol)
            #     long_distance_from_entry = abs(current_price - long_pos_price) / long_pos_price
            #     buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * long_distance_from_entry
            #     buffer_distance_long = current_price * buffer_percentage_long
            #     price_range_long = current_price - outer_price_long
            #     grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            #     self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])  # Place new long grid orders
            #     self.active_grids.add(symbol)  # Add this line
            #     logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            # # Replace short grid if necessary
            # if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False):
            #     logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
            #     self.clear_grid(symbol, 'sell')  # Cancel existing short grid orders
            #     self.active_grids.discard(symbol)
            #     short_distance_from_entry = abs(current_price - short_pos_price) / short_pos_price
            #     buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * short_distance_from_entry
            #     buffer_distance_short = current_price * buffer_percentage_short
            #     price_range_short = outer_price_short - current_price
            #     grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]
            #     self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])  # Place new short grid orders
            #     self.active_grids.add(symbol)  # Add this line
            #     logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            # Check if trading a new symbol is allowed based on open symbols and allowed count
            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            # Set MFI signals for long and short
            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                # Reissue orders if necessary based on the reissue threshold and current orders
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                    #logging.info(f"Open orders for {symbol}: {open_orders}")

                    # Flags to check for existing buy or sell orders, excluding reduce-only orders
                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    # Handling reissue for long orders
                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')  # Clear existing long orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    # Handling reissue for short orders
                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')  # Clear existing short orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")

            if symbol in open_symbols or trading_allowed:
                # Check if there are open positions and no active grids
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    if short_pos_qty > 0 and not short_grid_active:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    logging.info(f"[{symbol}] No open positions. Canceling leftover grid orders.")
                    self.clear_grid(symbol, 'buy')  # Clear any lingering buy grid orders
                    self.clear_grid(symbol, 'sell')  # Clear any lingering sell grid orders
                    self.active_grids.discard(symbol)  # Remove the symbol from active grids

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["sell"].clear()

                        # Place new long and short grid orders only if there are no existing orders (excluding TP orders) and no active grids
                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having active grids
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["buy"].clear()

                            # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["sell"].clear()

                            # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            # Check if there is room for trading new symbols
            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
            if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                # Place grid orders for the new symbol if MFI signal is present and auto-reduce is not blocking
                if long_mode and mfi_signal_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")
                if short_mode and mfi_signal_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )
        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())
        else:
            logging.info(f"[{symbol}] Trading not allowed. Skipping grid placement.")
            time.sleep(5)


    def linear_grid_handle_positions_mfirsi_persistent_notional_dynamic_buffer_qs_dynamictp(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool
    ):
        try:
            # Check reissue necessity for both long and short positions
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            open_symbols = list(set(open_symbols))  # Ensure symbols are unique
            logging.info(f"Open symbols {open_symbols}")

            # Initialize filled levels if not already present
            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            # Check if grids for buying or selling are active
            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            # Get current market price and log it
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Set buffer percentages based on whether positions are open
            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            # Calculate buffer distances from the current market price
            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            # Log the calculated buffer distances
            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            # Fetch order book data to get best ask and bid prices
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate grid levels using the current price, outer price distance, and buffer distances
            outer_price_long = current_price * (1 - outer_price_distance)
            outer_price_short = current_price * (1 + outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Check if long and short grid levels overlap and adjust if necessary
            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Log calculated grid levels
            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            # Check precision and minimum quantity for trading on Bybit
            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            # Calculate the total notional amounts for long and short sides based on exposure limits and leverage
            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            # Log the total amounts calculated for long and short trades
            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            # Calculate the individual order amounts for the long and short grids
            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            # Log auto-reduce status for long position
            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            # Log auto-reduce status for short position
            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Determine if the grid needs to be replaced based on the dynamic buffer adjustments
            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            # Replace long grid if necessary
            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')  # Cancel existing long grid orders
                self.active_grids.discard(symbol)
                long_distance_from_entry = abs(current_price - long_pos_price) / long_pos_price
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * long_distance_from_entry
                buffer_distance_long = long_pos_price * buffer_percentage_long  # Updated calculation
                price_range_long = current_price - outer_price_long
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])  # Place new long grid orders
                self.active_grids.add(symbol)  # Add this line
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            # Replace short grid if necessary
            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')  # Cancel existing short grid orders
                self.active_grids.discard(symbol)
                short_distance_from_entry = abs(current_price - short_pos_price) / short_pos_price
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * short_distance_from_entry
                buffer_distance_short = short_pos_price * buffer_percentage_short  # Updated calculation
                price_range_short = outer_price_short - current_price
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])  # Place new short grid orders
                self.active_grids.add(symbol)  # Add this line
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            # Check if trading a new symbol is allowed based on open symbols and allowed count
            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            # Set MFI signals for long and short
            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                # Reissue orders if necessary based on the reissue threshold and current orders
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                    #logging.info(f"Open orders for {symbol}: {open_orders}")

                    # Flags to check for existing buy or sell orders, excluding reduce-only orders
                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    # Handling reissue for long orders
                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')  # Clear existing long orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    # Handling reissue for short orders
                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')  # Clear existing short orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")

            if symbol in open_symbols or trading_allowed:
                # Check if there are open positions and no active grids
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    if short_pos_qty > 0 and not short_grid_active:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                current_time = datetime.now()
                
                # Logic to clear grids if there are no open positions and it's time to clear them
                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')  # Clear any lingering buy grid orders
                        self.clear_grid(symbol, 'sell')  # Clear any lingering sell grid orders
                        self.active_grids.discard(symbol)  # Remove the symbol from active grids
                        self.last_cleared_time[symbol] = current_time  # Update the last cleared time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["sell"].clear()

                        # Place new long and short grid orders only if there are no existing orders (excluding TP orders) and no active grids
                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having active grids
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["buy"].clear()

                            # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["sell"].clear()

                            # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            # Check if there is room for trading new symbols
            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
            if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                # Place grid orders for the new symbol if MFI signal is present and auto-reduce is not blocking
                if long_mode and mfi_signal_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")
                if short_mode and mfi_signal_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            current_latest_time = datetime.datetime.now()

            # Check for long positions
            if long_pos_qty > 0:
                if current_latest_time >= self.next_long_tp_update:
                    self.next_long_tp_update = self.update_quickscalp_tp(
                        symbol=symbol, 
                        pos_qty=long_pos_qty, 
                        upnl_profit_pct=upnl_profit_pct,  # Add the quickscalp percentage
                        short_pos_price=short_pos_price,
                        long_pos_price=long_pos_price,
                        positionIdx=1, 
                        order_side="sell", 
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts
                    )

            # Check for short positions
            if short_pos_qty > 0:
                if current_latest_time >= self.next_short_tp_update:
                    self.next_short_tp_update = self.update_quickscalp_tp(
                        symbol=symbol, 
                        pos_qty=short_pos_qty, 
                        upnl_profit_pct=upnl_profit_pct,  # Add the quickscalp percentage
                        short_pos_price=short_pos_price,
                        long_pos_price=long_pos_price,
                        positionIdx=2, 
                        order_side="buy", 
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts
                    )

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())
        else:
            logging.info(f"[{symbol}] Trading not allowed. Skipping grid placement.")
            time.sleep(5)

    def linear_grid_handle_positions_mfirsi_persistent_notional_dynamic_buffer_qs(
        self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float,
        short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int,
        strength: float, outer_price_distance: float, reissue_threshold: float,
        wallet_exposure_limit: float, wallet_exposure_limit_long: float, wallet_exposure_limit_short: float,
        user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool,
        short_mode: bool, initial_entry_buffer_pct: float, min_buffer_percentage: float, max_buffer_percentage: float,
        symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float,
        max_upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool
    ):
        try:
            # Check reissue necessity for both long and short positions
            should_reissue_long, should_reissue_short = self.should_reissue_orders_revised(
                symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            open_symbols = list(set(open_symbols))  # Ensure symbols are unique
            logging.info(f"Open symbols {open_symbols}")

            # Initialize filled levels if not already present
            if symbol not in self.filled_levels:
                self.filled_levels[symbol] = {"buy": set(), "sell": set()}

            # Check if grids for buying or selling are active
            long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
            short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

            # Get current market price and log it
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Set buffer percentages based on whether positions are open
            buffer_percentage_long = initial_entry_buffer_pct if long_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - long_pos_price) / long_pos_price)
            buffer_percentage_short = initial_entry_buffer_pct if short_pos_qty == 0 else min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * (abs(current_price - short_pos_price) / short_pos_price)

            # Calculate buffer distances from the current market price
            buffer_distance_long = current_price * buffer_percentage_long
            buffer_distance_short = current_price * buffer_percentage_short

            # Log the calculated buffer distances
            logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer_distance: {buffer_distance_short}")

            # Fetch order book data to get best ask and bid prices
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)

            # Calculate grid levels using the current price, outer price distance, and buffer distances
            outer_price_long = current_price * (1 - outer_price_distance)
            outer_price_short = current_price * (1 + outer_price_distance)
            price_range_long = current_price - outer_price_long
            price_range_short = outer_price_short - current_price
            factors = np.linspace(0.0, 1.0, num=levels) ** strength
            grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
            grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Check if long and short grid levels overlap and adjust if necessary
            if grid_levels_long[-1] >= grid_levels_short[0]:
                logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

            # Log calculated grid levels
            logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
            logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

            # Check precision and minimum quantity for trading on Bybit
            qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
            min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
            logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

            # Calculate the total notional amounts for long and short sides based on exposure limits and leverage
            total_amount_long = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="buy", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=user_defined_leverage_long,
                user_defined_leverage_short=None
            ) if long_mode else 0

            total_amount_short = self.calculate_total_amount_notional_ls(
                symbol=symbol, total_equity=total_equity, best_ask_price=best_ask_price,
                best_bid_price=best_bid_price, wallet_exposure_limit_long=wallet_exposure_limit_long,
                wallet_exposure_limit_short=wallet_exposure_limit_short, side="sell", levels=levels,
                enforce_full_grid=enforce_full_grid, user_defined_leverage_long=None,
                user_defined_leverage_short=user_defined_leverage_short
            ) if short_mode else 0

            # Log the total amounts calculated for long and short trades
            logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

            # Calculate the individual order amounts for the long and short grids
            amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
            amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
            logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
            logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

            # Log auto-reduce status for long position
            if self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"Auto-reduce for long position on {symbol} is active")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for long position on {symbol} is not active")

            # Log auto-reduce status for short position
            if self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"Auto-reduce for short position on {symbol} is active")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
            else:
                logging.info(f"Auto-reduce for short position on {symbol} is not active")

            # Determine if the grid needs to be replaced based on the dynamic buffer adjustments
            replace_long_grid, replace_short_grid = self.should_replace_grid_updated_buffer(
                symbol, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                min_buffer_percentage, max_buffer_percentage
            )

            # Replace long grid if necessary
            if replace_long_grid and not self.auto_reduce_active_long.get(symbol, False):
                logging.info(f"[{symbol}] Replacing long grid orders due to updated buffer.")
                self.clear_grid(symbol, 'buy')  # Cancel existing long grid orders
                self.active_grids.discard(symbol)
                long_distance_from_entry = abs(current_price - long_pos_price) / long_pos_price
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * long_distance_from_entry
                buffer_distance_long = long_pos_price * buffer_percentage_long  # Updated calculation
                price_range_long = current_price - outer_price_long
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])  # Place new long grid orders
                self.active_grids.add(symbol)  # Add this line
                logging.info(f"[{symbol}] Recalculated long grid levels with updated buffer: {grid_levels_long}")

            # Replace short grid if necessary
            if replace_short_grid and not self.auto_reduce_active_short.get(symbol, False):
                logging.info(f"[{symbol}] Replacing short grid orders due to updated buffer.")
                self.clear_grid(symbol, 'sell')  # Cancel existing short grid orders
                self.active_grids.discard(symbol)
                short_distance_from_entry = abs(current_price - short_pos_price) / short_pos_price
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * short_distance_from_entry
                buffer_distance_short = short_pos_price * buffer_percentage_short  # Updated calculation
                price_range_short = outer_price_short - current_price
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]
                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])  # Place new short grid orders
                self.active_grids.add(symbol)  # Add this line
                logging.info(f"[{symbol}] Recalculated short grid levels with updated buffer: {grid_levels_short}")

            # Check if trading a new symbol is allowed based on open symbols and allowed count
            trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
            logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

            # Set MFI signals for long and short
            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            if len(open_symbols) < symbols_allowed:
                logging.info(f"Allowed symbol: {symbol}")
                # Reissue orders if necessary based on the reissue threshold and current orders
                if self.should_reissue_orders_revised(symbol, reissue_threshold, long_pos_qty, short_pos_qty, initial_entry_buffer_pct):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                    #logging.info(f"Open orders for {symbol}: {open_orders}")

                    # Flags to check for existing buy or sell orders, excluding reduce-only orders
                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    # Handling reissue for long orders
                    if not long_pos_qty and long_mode and not self.auto_reduce_active_long.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                            if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                                logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'buy')  # Clear existing long orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    # Handling reissue for short orders
                    if not short_pos_qty and short_mode and not self.auto_reduce_active_short.get(symbol, False):
                        if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                            if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                                logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                                self.clear_grid(symbol, 'sell')  # Clear existing short orders
                                self.active_grids.discard(symbol)  # Add this line
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Add this line
                            elif symbol not in self.active_grids:
                                logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")

            if symbol in open_symbols or trading_allowed:
                # Check if there are open positions and no active grids
                if (long_pos_qty > 0 and not long_grid_active) or (short_pos_qty > 0 and not short_grid_active):
                    logging.info(f"[{symbol}] Open positions found without active grids. Issuing grid orders.")
                    if long_pos_qty > 0 and not long_grid_active:
                        if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing long grid orders for existing open position.")
                            self.clear_grid(symbol, 'buy')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    if short_pos_qty > 0 and not short_grid_active:
                        if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                            logging.info(f"[{symbol}] Placing short grid orders for existing open position.")
                            self.clear_grid(symbol, 'sell')  # Ensure no previous orders conflict
                            self.active_grids.discard(symbol)  # Add this line
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                current_time = datetime.now()
                
                # Logic to clear grids if there are no open positions and it's time to clear them
                if not long_pos_qty and not short_pos_qty and symbol in self.active_grids:
                    last_cleared = self.last_cleared_time.get(symbol, datetime.min)
                    if current_time - last_cleared > self.clear_interval:
                        logging.info(f"[{symbol}] No open positions and time interval passed. Canceling leftover grid orders.")
                        self.clear_grid(symbol, 'buy')  # Clear any lingering buy grid orders
                        self.clear_grid(symbol, 'sell')  # Clear any lingering sell grid orders
                        self.active_grids.discard(symbol)  # Remove the symbol from active grids
                        self.last_cleared_time[symbol] = current_time  # Update the last cleared time
                    else:
                        logging.info(f"[{symbol}] No open positions, but time interval not passed. Skipping grid clearing.")

                if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                    logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                    if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                        if (should_reissue_long or long_pos_qty > 0) and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "buy")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["buy"].clear()

                        if (should_reissue_short or short_pos_qty > 0) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders):
                            # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                            self.cancel_grid_orders(symbol, "sell")
                            self.active_grids.discard(symbol)  # Add this line
                            self.filled_levels[symbol]["sell"].clear()

                        # Place new long and short grid orders only if there are no existing orders (excluding TP orders) and no active grids
                        if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                            logging.info(f"[{symbol}] Placing new long and short grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having active grids
                    else:
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if should_reissue_long or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "buy")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["buy"].clear()

                            # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                logging.info(f"[{symbol}] Placing new long grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if should_reissue_short or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                self.cancel_grid_orders(symbol, "sell")
                                self.active_grids.discard(symbol)  # Add this line
                                self.filled_levels[symbol]["sell"].clear()

                            # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                            if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new short grid orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                else:
                    if self.auto_reduce_active_long.get(symbol, False):
                        logging.info(f"Auto-reduce for long position on {symbol} is active, entry during auto-reduce.")
                        if long_mode and (mfi_signal_long or long_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new long orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")

                    if self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for short position on {symbol} is active, entry during auto-reduce.")
                        if short_mode and (mfi_signal_short or short_pos_qty > 0):
                            if entry_during_autoreduce:
                                logging.info(f"[{symbol}] Placing new short orders despite active auto-reduce due to entry_during_autoreduce setting.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            # Check if there is room for trading new symbols
            logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
            if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                # Place grid orders for the new symbol if MFI signal is present and auto-reduce is not blocking
                if long_mode and mfi_signal_long:
                    if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new long orders (either no active long auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new long orders due to active long auto-reduce and entry_during_autoreduce set to False.")
                if short_mode and mfi_signal_short:
                    if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                        logging.info(f"[{symbol}] Placing new short orders (either no active short auto-reduce or entry during auto-reduce is allowed).")
                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        logging.info(f"[{symbol}] Skipping new short orders due to active short auto-reduce and entry_during_autoreduce set to False.")

            # Calculate take profit for short and long positions using quickscalp method
            short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
            long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

            # Update TP for long position
            if long_pos_qty > 0:
                new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                    long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_long_tp_min is not None and new_long_tp_max is not None:
                    self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=long_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=None,  # Not relevant for long TP settings
                        long_pos_price=long_pos_price,
                        positionIdx=1,
                        order_side="sell",
                        last_tp_update=self.next_long_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

            if short_pos_qty > 0:
                new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                    short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                )
                if new_short_tp_min is not None and new_short_tp_max is not None:
                    self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                        symbol=symbol,
                        pos_qty=short_pos_qty,
                        upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                        max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                        short_pos_price=short_pos_price,
                        long_pos_price=None,  # Not relevant for short TP settings
                        positionIdx=2,
                        order_side="buy",
                        last_tp_update=self.next_short_tp_update,
                        tp_order_counts=tp_order_counts,
                        open_orders=open_orders
                    )

        except Exception as e:
            logging.info(f"Error in executing gridstrategy: {e}")
            logging.info("Traceback: %s", traceback.format_exc())
        else:
            logging.info(f"[{symbol}] Trading not allowed. Skipping grid placement.")
            time.sleep(5)

    def linear_grid_handle_positions_mfirsi_persistent_notional_dynamic_buffer(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float, short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int, strength: float, outer_price_distance: float, reissue_threshold: float, wallet_exposure_limit: float, user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool, short_mode: bool, min_buffer_percentage: float, max_buffer_percentage: float, symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                should_reissue = self.should_reissue_orders(symbol, reissue_threshold)
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                #logging.info(f"Open orders: {open_orders}")

                if symbol not in self.filled_levels:
                    self.filled_levels[symbol] = {"buy": set(), "sell": set()}

                long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
                short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"[{symbol}] Current price: {current_price}")

                # Handle non-existing positions by applying default buffer percentages if positions are not active
                default_buffer = min_buffer_percentage  # Could set to a reasonable default like the min_buffer_percentage

                if long_pos_price and long_pos_price > 0:
                    long_distance_from_entry = abs(current_price - long_pos_price) / long_pos_price
                    buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * long_distance_from_entry
                else:
                    buffer_percentage_long = default_buffer  # Use default buffer if no valid long position

                if short_pos_price and short_pos_price > 0:
                    short_distance_from_entry = abs(current_price - short_pos_price) / short_pos_price
                    buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * short_distance_from_entry
                else:
                    buffer_percentage_short = default_buffer  # Use default buffer if no valid short position

                buffer_distance_long = current_price * buffer_percentage_long
                buffer_distance_short = current_price * buffer_percentage_short

                logging.info(f"[{symbol}] Long buffer distance: {buffer_distance_long}, Short buffer distance: {buffer_distance_short}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
                logging.info(f"[{symbol}] Best ask price: {best_ask_price}, Best bid price: {best_bid_price}")

                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                logging.info(f"[{symbol}] Outer price long: {outer_price_long}, Outer price short: {outer_price_short}")

                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price

                factors = np.linspace(0.0, 1.0, num=levels) ** strength
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

                # Check if long and short grid levels overlap
                if grid_levels_long[-1] >= grid_levels_short[0]:
                    logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                    # Adjust outer_price_distance to prevent overlap
                    outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                    # Recalculate grid levels
                    outer_price_long = current_price * (1 - outer_price_distance)
                    outer_price_short = current_price * (1 + outer_price_distance)
                    price_range_long = current_price - outer_price_long
                    price_range_short = outer_price_short - current_price
                    grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                    grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

                logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
                logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

                qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
                min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
                logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

                total_amount_long = self.calculate_total_amount_notional(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_long, "buy", levels, enforce_full_grid) if long_mode else 0
                total_amount_short = self.calculate_total_amount_notional(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_short, "sell", levels, enforce_full_grid) if short_mode else 0
                logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

                amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
                amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
                logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
                logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

                trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
                logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
                logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

                mfi_signal_long = mfirsi_signal.lower() == "long"
                mfi_signal_short = mfirsi_signal.lower() == "short"

                if self.should_reissue_orders(symbol, reissue_threshold):
                    open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                    #logging.info(f"Open orders for {symbol}: {open_orders}")

                    # Flags to check existence of buy or sell orders, excluding reduce-only orders
                    has_open_long_order = any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)
                    has_open_short_order = any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)

                    if not long_pos_qty and long_mode:  # Only enter this block if there are no long positions and long trading is enabled
                        if symbol in self.active_grids and "buy" in self.filled_levels[symbol] and has_open_long_order:
                            logging.info(f"[{symbol}] Reissuing long orders due to price movement beyond the threshold.")
                            self.cancel_grid_orders(symbol, "buy")
                            self.filled_levels[symbol]["buy"].clear()
                            logging.info(f"[{symbol}] Placing new long orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                        elif symbol not in self.active_grids:
                            logging.info(f"[{symbol}] No active long grid for the symbol. Skipping long grid reissue.")

                    if not short_pos_qty and short_mode:  # Only enter this block if there are no short positions and short trading is enabled
                        if symbol in self.active_grids and "sell" in self.filled_levels[symbol] and has_open_short_order:
                            logging.info(f"[{symbol}] Reissuing short orders due to price movement beyond the threshold.")
                            self.cancel_grid_orders(symbol, "sell")
                            self.filled_levels[symbol]["sell"].clear()
                            logging.info(f"[{symbol}] Placing new short orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                        elif symbol not in self.active_grids:
                            logging.info(f"[{symbol}] No active short grid for the symbol. Skipping short grid reissue.")
                            
                if symbol in open_symbols or trading_allowed:
                    if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                        if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                            if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)) or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing long and short grid orders if should_reissue or positions exist but no corresponding orders
                                self.cancel_grid_orders(symbol, "buy")
                                self.cancel_grid_orders(symbol, "sell")
                                self.filled_levels[symbol]["buy"].clear()
                                self.filled_levels[symbol]["sell"].clear()

                            # Place new long and short grid orders only if there are no existing orders (excluding TP orders) and no active grids
                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new long and short grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having active grids
                        else:
                            if long_mode and (mfi_signal_long or long_pos_qty > 0):
                                if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "buy")
                                    self.filled_levels[symbol]["buy"].clear()

                                # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                                if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                    logging.info(f"[{symbol}] Placing new long grid orders.")
                                    self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                    self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                            if short_mode and (mfi_signal_short or short_pos_qty > 0):
                                if should_reissue or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "sell")
                                    self.filled_levels[symbol]["sell"].clear()

                                # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                                if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                    logging.info(f"[{symbol}] Placing new short grid orders.")
                                    self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                    self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        if not self.auto_reduce_active_long.get(symbol, False):
                            logging.info(f"Auto-reduce for long position on {symbol} is not active")
                            if long_mode and (mfi_signal_long or (long_pos_qty > 0 and not long_grid_active)):
                                if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "buy")
                                    self.filled_levels[symbol]["buy"].clear()

                                # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                                if (not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active) or long_pos_qty > 0:
                                    if entry_during_autoreduce:
                                        logging.info(f"[{symbol}] Placing new long grid orders (entry during auto-reduce).")
                                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                                    else:
                                        logging.info(f"[{symbol}] Skipping new long grid orders due to active auto-reduce.")
                        else:
                            logging.info(f"Auto-reduce for long position on {symbol} is active, skipping entry")

                        if not self.auto_reduce_active_short.get(symbol, False):
                            logging.info(f"Auto-reduce for short position on {symbol} is not active")
                            if short_mode and (mfi_signal_short or (short_pos_qty > 0 and not short_grid_active)):
                                if should_reissue or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "sell")
                                    self.filled_levels[symbol]["sell"].clear()

                                # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                                if (not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active) or short_pos_qty > 0:
                                    if entry_during_autoreduce:
                                        logging.info(f"[{symbol}] Placing new short grid orders (entry during auto-reduce).")
                                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                                    else:
                                        logging.info(f"[{symbol}] Skipping new short grid orders due to active auto-reduce.")
                        else:
                            logging.info(f"Auto-reduce for short position on {symbol} is active, skipping entry")

                    # Check if there is room for trading new symbols
                    logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
                    if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                        logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                        # Place grid orders for the new symbol
                        if long_mode and mfi_signal_long:
                            if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active auto-reduce.")
                        if short_mode and mfi_signal_short:
                            if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active auto-reduce.")

                    short_take_profit = None
                    long_take_profit = None

                    # Calculate take profit for short and long positions using quickscalp method
                    short_take_profit = self.calculate_quickscalp_short_take_profit(short_pos_price, symbol, upnl_profit_pct)
                    long_take_profit = self.calculate_quickscalp_long_take_profit(long_pos_price, symbol, upnl_profit_pct)
                    
                    self.place_long_tp_order(
                        symbol,
                        best_ask_price,
                        long_pos_price,
                        long_pos_qty,
                        long_take_profit,
                        open_orders
                    )

                    self.place_short_tp_order(
                        symbol,
                        best_bid_price,
                        short_pos_price,
                        short_pos_qty,
                        short_take_profit,
                        open_orders
                    )

                    # Update TP for long position
                    if long_pos_qty > 0:
                        self.next_long_tp_update = self.update_quickscalp_tp(
                            symbol=symbol,
                            pos_qty=long_pos_qty,
                            upnl_profit_pct=upnl_profit_pct,
                            short_pos_price=short_pos_price,
                            long_pos_price=long_pos_price,
                            positionIdx=1,
                            order_side="sell",
                            last_tp_update=self.next_long_tp_update,
                            tp_order_counts=tp_order_counts
                        )

                    # Update TP for short position
                    if short_pos_qty > 0:
                        self.next_short_tp_update = self.update_quickscalp_tp(
                            symbol=symbol,
                            pos_qty=short_pos_qty,
                            upnl_profit_pct=upnl_profit_pct,
                            short_pos_price=short_pos_price,
                            long_pos_price=long_pos_price,
                            positionIdx=2,
                            order_side="buy",
                            last_tp_update=self.next_short_tp_update,
                            tp_order_counts=tp_order_counts
                        )

                else:
                    logging.info(f"[{symbol}] Trading not allowed. Skipping grid placement.")
                    time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in grid {e}")
            
            
    def linear_grid_handle_positions_mfirsi_persistent_notional(self, symbol: str, open_symbols: list, total_equity: float, long_pos_price: float, short_pos_price: float, long_pos_qty: float, short_pos_qty: float, levels: int, strength: float, outer_price_distance: float, reissue_threshold: float, wallet_exposure_limit: float, user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool, short_mode: bool, buffer_percentage: float, symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str, upnl_profit_pct: float, tp_order_counts: dict, entry_during_autoreduce: bool):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                should_reissue = self.should_reissue_orders(symbol, reissue_threshold)
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                #logging.info(f"Open orders: {open_orders}")

                # Initialize filled_levels dictionary for the current symbol if it doesn't exist
                if symbol not in self.filled_levels:
                    self.filled_levels[symbol] = {"buy": set(), "sell": set()}

                # Check if long and short grids are active separately
                long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
                short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"[{symbol}] Current price: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
                logging.info(f"[{symbol}] Best ask price: {best_ask_price}, Best bid price: {best_bid_price}")

                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                logging.info(f"[{symbol}] Outer price long: {outer_price_long}, Outer price short: {outer_price_short}")

                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price

                buffer_distance_long = current_price * buffer_percentage / 100
                buffer_distance_short = current_price * buffer_percentage / 100

                factors = np.linspace(0.0, 1.0, num=levels) ** strength
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

                # Check if long and short grid levels overlap
                if grid_levels_long[-1] >= grid_levels_short[0]:
                    logging.warning(f"[{symbol}] Long and short grid levels overlap. Adjusting outer_price_distance.")
                    # Adjust outer_price_distance to prevent overlap
                    outer_price_distance = (grid_levels_short[0] - grid_levels_long[-1]) / (2 * current_price)
                    # Recalculate grid levels
                    outer_price_long = current_price * (1 - outer_price_distance)
                    outer_price_short = current_price * (1 + outer_price_distance)
                    price_range_long = current_price - outer_price_long
                    price_range_short = outer_price_short - current_price
                    grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                    grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

                logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
                logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

                qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
                min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
                logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

                total_amount_long = self.calculate_total_amount_notional(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_long, "buy", levels, enforce_full_grid) if long_mode else 0
                total_amount_short = self.calculate_total_amount_notional(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_short, "sell", levels, enforce_full_grid) if short_mode else 0
                logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

                amounts_long = self.calculate_order_amounts_notional(symbol, total_amount_long, levels, strength, qty_precision, enforce_full_grid)
                amounts_short = self.calculate_order_amounts_notional(symbol, total_amount_short, levels, strength, qty_precision, enforce_full_grid)
                logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
                logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

                trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
                logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
                logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

                mfi_signal_long = mfirsi_signal.lower() == "long"
                mfi_signal_short = mfirsi_signal.lower() == "short"

                if symbol in open_symbols or trading_allowed:
                    if not self.auto_reduce_active_long.get(symbol, False) and not self.auto_reduce_active_short.get(symbol, False):
                        logging.info(f"Auto-reduce for long and short positions on {symbol} is not active")
                        if long_mode and short_mode and ((mfi_signal_long or long_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0)):
                            if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)) or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                # Cancel existing long and short grid orders if should_reissue or positions exist but no corresponding orders
                                self.cancel_grid_orders(symbol, "buy")
                                self.cancel_grid_orders(symbol, "sell")
                                self.filled_levels[symbol]["buy"].clear()
                                self.filled_levels[symbol]["sell"].clear()

                            # Place new long and short grid orders only if there are no existing orders (excluding TP orders) and no active grids
                            if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not long_grid_active and not short_grid_active:
                                logging.info(f"[{symbol}] Placing new long and short grid orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having active grids
                        else:
                            if long_mode and (mfi_signal_long or long_pos_qty > 0):
                                if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "buy")
                                    self.filled_levels[symbol]["buy"].clear()

                                # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                                if not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active:
                                    logging.info(f"[{symbol}] Placing new long grid orders.")
                                    self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                    self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                            if short_mode and (mfi_signal_short or short_pos_qty > 0):
                                if should_reissue or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "sell")
                                    self.filled_levels[symbol]["sell"].clear()

                                # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                                if not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active:
                                    logging.info(f"[{symbol}] Placing new short grid orders.")
                                    self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                    self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                    else:
                        if not self.auto_reduce_active_long.get(symbol, False):
                            logging.info(f"Auto-reduce for long position on {symbol} is not active")
                            if long_mode and (mfi_signal_long or (long_pos_qty > 0 and not long_grid_active)):
                                if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing long grid orders if should_reissue or long position exists but no buy orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "buy")
                                    self.filled_levels[symbol]["buy"].clear()

                                # Place new long grid orders if there are no existing buy orders (excluding TP orders) and no active long grid, or if there is a long position
                                if (not any(order['side'].lower() == 'buy' and not order['reduceOnly'] for order in open_orders) and not long_grid_active) or long_pos_qty > 0:
                                    if entry_during_autoreduce:
                                        logging.info(f"[{symbol}] Placing new long grid orders (entry during auto-reduce).")
                                        self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                                    else:
                                        logging.info(f"[{symbol}] Skipping new long grid orders due to active auto-reduce.")
                        else:
                            logging.info(f"Auto-reduce for long position on {symbol} is active, skipping entry")

                        if not self.auto_reduce_active_short.get(symbol, False):
                            logging.info(f"Auto-reduce for short position on {symbol} is not active")
                            if short_mode and (mfi_signal_short or (short_pos_qty > 0 and not short_grid_active)):
                                if should_reissue or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders)):
                                    # Cancel existing short grid orders if should_reissue or short position exists but no sell orders (excluding TP orders)
                                    self.cancel_grid_orders(symbol, "sell")
                                    self.filled_levels[symbol]["sell"].clear()

                                # Place new short grid orders if there are no existing sell orders (excluding TP orders) and no active short grid, or if there is a short position
                                if (not any(order['side'].lower() == 'sell' and not order['reduceOnly'] for order in open_orders) and not short_grid_active) or short_pos_qty > 0:
                                    if entry_during_autoreduce:
                                        logging.info(f"[{symbol}] Placing new short grid orders (entry during auto-reduce).")
                                        self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                        self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                                    else:
                                        logging.info(f"[{symbol}] Skipping new short grid orders due to active auto-reduce.")
                        else:
                            logging.info(f"Auto-reduce for short position on {symbol} is active, skipping entry")

                    # Check if there is room for trading new symbols
                    logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
                    if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                        logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                        # Place grid orders for the new symbol
                        if long_mode and mfi_signal_long:
                            if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                                logging.info(f"[{symbol}] Placing new long orders.")
                                self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new long orders due to active auto-reduce.")
                        if short_mode and mfi_signal_short:
                            if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
                                logging.info(f"[{symbol}] Placing new short orders.")
                                self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                                self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                            else:
                                logging.info(f"[{symbol}] Skipping new short orders due to active auto-reduce.")

                    # Update TP for long position
                    if long_pos_qty > 0:
                        self.next_long_tp_update = self.update_quickscalp_tp(
                            symbol=symbol,
                            pos_qty=long_pos_qty,
                            upnl_profit_pct=upnl_profit_pct,
                            short_pos_price=short_pos_price,
                            long_pos_price=long_pos_price,
                            positionIdx=1,
                            order_side="sell",
                            last_tp_update=self.next_long_tp_update,
                            tp_order_counts=tp_order_counts
                        )

                    # Update TP for short position
                    if short_pos_qty > 0:
                        self.next_short_tp_update = self.update_quickscalp_tp(
                            symbol=symbol,
                            pos_qty=short_pos_qty,
                            upnl_profit_pct=upnl_profit_pct,
                            short_pos_price=short_pos_price,
                            long_pos_price=long_pos_price,
                            positionIdx=2,
                            order_side="buy",
                            last_tp_update=self.next_short_tp_update,
                            tp_order_counts=tp_order_counts
                        )

                else:
                    logging.info(f"[{symbol}] Trading not allowed. Skipping grid placement.")
                    time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in grid {e}")

    def linear_grid_handle_positions_mfirsi(self, symbol: str, open_symbols: list, total_equity: float, long_pos_qty: float, short_pos_qty: float, levels: int, strength: float, outer_price_distance: float, reissue_threshold: float, wallet_exposure_limit: float, user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool, short_mode: bool, buffer_percentage: float, symbols_allowed: int, enforce_full_grid: bool, mfirsi_signal: str):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                should_reissue = self.should_reissue_orders(symbol, reissue_threshold)
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                # Initialize filled_levels dictionary for the current symbol if it doesn't exist
                if symbol not in self.filled_levels:
                    self.filled_levels[symbol] = {"buy": set(), "sell": set()}

                # Check if long and short grids are active separately
                long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
                short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

                if (long_grid_active or short_grid_active) and not should_reissue:
                    logging.info(f"[{symbol}] Grid already active and reissue threshold not met. Skipping grid placement.")
                    return

                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"[{symbol}] Current price: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
                logging.info(f"[{symbol}] Best ask price: {best_ask_price}, Best bid price: {best_bid_price}")

                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                logging.info(f"[{symbol}] Outer price long: {outer_price_long}, Outer price short: {outer_price_short}")

                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price

                buffer_distance_long = current_price * buffer_percentage / 100
                buffer_distance_short = current_price * buffer_percentage / 100

                factors = np.linspace(0.0, 1.0, num=levels) ** strength
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

                logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
                logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

                qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
                min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
                logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

                total_amount_long = self.calculate_total_amount(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_long, "buy", levels, min_qty, enforce_full_grid) if long_mode else 0
                total_amount_short = self.calculate_total_amount(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_short, "sell", levels, min_qty, enforce_full_grid) if short_mode else 0
                logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

                amounts_long = self.calculate_order_amounts(symbol, total_amount_long, levels, strength, qty_precision, min_qty, enforce_full_grid)
                amounts_short = self.calculate_order_amounts(symbol, total_amount_short, levels, strength, qty_precision, min_qty, enforce_full_grid)
                logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
                logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

                trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
                logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
                logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

                mfi_signal_long = mfirsi_signal.lower() == "long"
                mfi_signal_short = mfirsi_signal.lower() == "short"

                if symbol in open_symbols or trading_allowed:
                    if long_mode and (not long_grid_active or long_pos_qty > 0) and (mfi_signal_long or long_pos_qty > 0):
                        if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' for order in open_orders)):
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                    if short_mode and (not short_grid_active or short_pos_qty > 0) and (mfi_signal_short or short_pos_qty > 0):
                        if should_reissue or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' for order in open_orders)):
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                    # Check if there is room for trading new symbols
                    logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
                    if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                        logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                        # Place grid orders for the new symbol
                        if long_mode and mfi_signal_long:
                            logging.info(f"[{symbol}] Placing new long orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                        if short_mode and mfi_signal_short:
                            logging.info(f"[{symbol}] Placing new short orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                else:
                    logging.info(f"[{symbol}] Trading not allowed. Skipping grid placement.")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in grid {e}")
            
            
    def linear_grid_handle_positions(self, symbol: str, open_symbols: list, total_equity: float, long_pos_qty: float, short_pos_qty: float, levels: int, strength: float, outer_price_distance: float, reissue_threshold: float, wallet_exposure_limit: float, user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool, short_mode: bool, buffer_percentage: float, symbols_allowed: int, enforce_full_grid: bool):
        try:
            if symbol not in self.symbol_locks:
                self.symbol_locks[symbol] = threading.Lock()

            with self.symbol_locks[symbol]:
                should_reissue = self.should_reissue_orders(symbol, reissue_threshold)
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                # Initialize filled_levels dictionary for the current symbol if it doesn't exist
                if symbol not in self.filled_levels:
                    self.filled_levels[symbol] = {"buy": set(), "sell": set()}

                # Check if long and short grids are active separately
                long_grid_active = symbol in self.active_grids and "buy" in self.filled_levels[symbol]
                short_grid_active = symbol in self.active_grids and "sell" in self.filled_levels[symbol]

                if (long_grid_active or short_grid_active) and not should_reissue:
                    logging.info(f"[{symbol}] Grid already active and reissue threshold not met. Skipping grid placement.")
                    return

                current_price = self.exchange.get_current_price(symbol)
                logging.info(f"[{symbol}] Current price: {current_price}")

                order_book = self.exchange.get_orderbook(symbol)
                best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
                best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
                logging.info(f"[{symbol}] Best ask price: {best_ask_price}, Best bid price: {best_bid_price}")

                outer_price_long = current_price * (1 - outer_price_distance)
                outer_price_short = current_price * (1 + outer_price_distance)
                logging.info(f"[{symbol}] Outer price long: {outer_price_long}, Outer price short: {outer_price_short}")

                price_range_long = current_price - outer_price_long
                price_range_short = outer_price_short - current_price

                buffer_distance_long = current_price * buffer_percentage / 100
                buffer_distance_short = current_price * buffer_percentage / 100

                factors = np.linspace(0.0, 1.0, num=levels) ** strength
                grid_levels_long = [current_price - buffer_distance_long - price_range_long * factor for factor in factors]
                grid_levels_short = [current_price + buffer_distance_short + price_range_short * factor for factor in factors]

                logging.info(f"[{symbol}] Long grid levels: {grid_levels_long}")
                logging.info(f"[{symbol}] Short grid levels: {grid_levels_short}")

                qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
                min_qty = float(self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)["min_qty"])
                logging.info(f"[{symbol}] Quantity precision: {qty_precision}, Minimum quantity: {min_qty}")

                total_amount_long = self.calculate_total_amount(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_long, "buy", levels, min_qty, enforce_full_grid) if long_mode else 0
                total_amount_short = self.calculate_total_amount(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_short, "sell", levels, min_qty, enforce_full_grid) if short_mode else 0
                logging.info(f"[{symbol}] Total amount long: {total_amount_long}, Total amount short: {total_amount_short}")

                amounts_long = self.calculate_order_amounts(symbol, total_amount_long, levels, strength, qty_precision, min_qty, enforce_full_grid)
                amounts_short = self.calculate_order_amounts(symbol, total_amount_short, levels, strength, qty_precision, min_qty, enforce_full_grid)
                logging.info(f"[{symbol}] Long order amounts: {amounts_long}")
                logging.info(f"[{symbol}] Short order amounts: {amounts_short}")

                trading_allowed = self.can_trade_new_symbol(open_symbols, symbols_allowed, symbol)
                logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
                logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

                if symbol in open_symbols or trading_allowed:
                    if long_mode and not long_grid_active:
                        if should_reissue or (long_pos_qty > 0 and not any(order['side'].lower() == 'buy' for order in open_orders)):
                            logging.info(f"[{symbol}] Placing new long grid orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                    if short_mode and not short_grid_active:
                        if should_reissue or (short_pos_qty > 0 and not any(order['side'].lower() == 'sell' for order in open_orders)):
                            logging.info(f"[{symbol}] Placing new short grid orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid

                    # Check if there is room for trading new symbols
                    logging.info(f"[{symbol}] Number of open symbols: {len(open_symbols)}, Symbols allowed: {symbols_allowed}")
                    if len(open_symbols) < symbols_allowed and symbol not in self.active_grids:
                        logging.info(f"[{symbol}] No active grids. Checking for new symbols to trade.")
                        # Place grid orders for the new symbol
                        if long_mode:
                            logging.info(f"[{symbol}] Placing new long orders.")
                            self.issue_grid_orders(symbol, "buy", grid_levels_long, amounts_long, True, self.filled_levels[symbol]["buy"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                        if short_mode:
                            logging.info(f"[{symbol}] Placing new short orders.")
                            self.issue_grid_orders(symbol, "sell", grid_levels_short, amounts_short, False, self.filled_levels[symbol]["sell"])
                            self.active_grids.add(symbol)  # Mark the symbol as having an active grid
                else:
                    logging.info(f"[{symbol}] Trading not allowed. Skipping grid placement.")

                time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in grid {e}")

    def should_replace_grid_updated_buffer_dynamic(self, symbol: str, long_pos_price: float, short_pos_price: float, long_pos_qty: float, short_pos_qty: float, min_buffer_percentage: float, max_buffer_percentage: float) -> tuple:
        try:
            current_price = self.exchange.get_current_price(symbol)
            last_price = self.last_price.get(symbol)
            
            if last_price is None:
                self.last_price[symbol] = current_price
                logging.info(f"[{symbol}] Dynamic buffer: No last price recorded. Setting current price {current_price} as last price. No reissue required.")
                return False, False
            
            price_change_percentage = abs(current_price - last_price) / last_price * 100
            logging.info(f"[{symbol}] Dynamic buffer: Last recorded price: {last_price}, Current price: {current_price}, Price change: {price_change_percentage:.2f}%")
            
            replace_long_grid = False
            replace_short_grid = False
            
            if long_pos_qty > 0:
                if price_change_percentage >= min_buffer_percentage * 100:
                    replace_long_grid = True
                    logging.info(f"Dynamic buffer: [{symbol}] Price change {price_change_percentage:.2f}% exceeds minimum buffer percentage {min_buffer_percentage:.2%}. Replacing long grid.")
                else:
                    logging.info(f"[{symbol}] Dynamic Buffer: Price change {price_change_percentage:.2f}% does not exceed minimum buffer percentage {min_buffer_percentage:.2%}. No need to replace long grid.")
            
            if short_pos_qty > 0:
                if price_change_percentage >= min_buffer_percentage * 100:
                    replace_short_grid = True
                    logging.info(f"[{symbol}] Dynamic buffer: Price change {price_change_percentage:.2f}% exceeds minimum buffer percentage {min_buffer_percentage:.2%}. Replacing short grid.")
                else:
                    logging.info(f"[{symbol}] Dynamic buffer: Price change {price_change_percentage:.2f}% does not exceed minimum buffer percentage {min_buffer_percentage:.2%}. No need to replace short grid.")
            
            logging.info(f"[{symbol}] Dynamic buffer: Should replace long grid: {replace_long_grid}")
            logging.info(f"[{symbol}] Dynamic buffer: Should replace short grid: {replace_short_grid}")
            
            self.last_price[symbol] = current_price  # Update the last recorded price
            
            return replace_long_grid, replace_short_grid
        
        except Exception as e:
            logging.exception(f"Exception caught in should_replace_grid_updated_buffer: {e}")
            return False, False

    def should_replace_grid_updated_buffer(self, symbol: str, long_pos_price: float, short_pos_price: float, long_pos_qty: float, short_pos_qty: float, min_buffer_percentage: float, max_buffer_percentage: float) -> tuple:
        try:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")
            
            replace_long_grid = False
            replace_short_grid = False
            
            if long_pos_qty > 0:
                long_distance_from_entry = abs(current_price - long_pos_price) / long_pos_price
                buffer_percentage_long = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * long_distance_from_entry
                buffer_distance_long = long_pos_price * buffer_percentage_long
                
                logging.info(f"[{symbol}] Long position info:")
                logging.info(f"  - Long position price: {long_pos_price}")
                logging.info(f"  - Long position quantity: {long_pos_qty}")
                logging.info(f"  - Long distance from entry: {long_distance_from_entry}")
                logging.info(f"  - Long buffer percentage: {buffer_percentage_long}")
                logging.info(f"  - Long buffer distance: {buffer_distance_long}")
                
                if abs(current_price - long_pos_price) > buffer_distance_long:
                    replace_long_grid = True
                    logging.info(f"[{symbol}] Price change exceeds updated buffer distance for long position. Replacing long grid.")
                else:
                    logging.info(f"[{symbol}] Price change does not exceed updated buffer distance for long position. No need to replace long grid.")
            
            if short_pos_qty > 0:
                short_distance_from_entry = abs(current_price - short_pos_price) / short_pos_price
                buffer_percentage_short = min_buffer_percentage + (max_buffer_percentage - min_buffer_percentage) * short_distance_from_entry
                buffer_distance_short = short_pos_price * buffer_percentage_short
                
                logging.info(f"[{symbol}] Short position info:")
                logging.info(f"  - Short position price: {short_pos_price}")
                logging.info(f"  - Short position quantity: {short_pos_qty}")
                logging.info(f"  - Short distance from entry: {short_distance_from_entry}")
                logging.info(f"  - Short buffer percentage: {buffer_percentage_short}")
                logging.info(f"  - Short buffer distance: {buffer_distance_short}")
                
                if abs(current_price - short_pos_price) > buffer_distance_short:
                    replace_short_grid = True
                    logging.info(f"[{symbol}] Price change exceeds updated buffer distance for short position. Replacing short grid.")
                else:
                    logging.info(f"[{symbol}] Price change does not exceed updated buffer distance for short position. No need to replace short grid.")
            
            logging.info(f"[{symbol}] Should replace long grid: {replace_long_grid}")
            logging.info(f"[{symbol}] Should replace short grid: {replace_short_grid}")
            
            return replace_long_grid, replace_short_grid
        
        except Exception as e:
            logging.exception(f"Exception caught in should_replace_grid_updated_buffer: {e}")
            return False, False

    def should_replace_grid_updated_buffer_min_outerpricedist_v2(
        self, 
        symbol: str, 
        long_pos_price: float, 
        short_pos_price: float, 
        long_pos_qty: float, 
        short_pos_qty: float, 
        dynamic_outer_price_distance_long: float, 
        dynamic_outer_price_distance_short: float
    ) -> tuple:
        try:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Retrieve the last reissue prices, ensure they are floats
            last_reissue_price_long = self.last_reissue_price_long.get(symbol, long_pos_price)
            last_reissue_price_short = self.last_reissue_price_short.get(symbol, short_pos_price)

            # Safeguard against None values
            if last_reissue_price_long is None:
                logging.info(f"[{symbol}] Last reissue price (long) is None, setting it to long_pos_price: {long_pos_price}")
                last_reissue_price_long = long_pos_price

            if last_reissue_price_short is None:
                logging.info(f"[{symbol}] Last reissue price (short) is None, setting it to short_pos_price: {short_pos_price}")
                last_reissue_price_short = short_pos_price

            logging.info(f"[{symbol}] Last reissue price (long): {last_reissue_price_long}")
            logging.info(f"[{symbol}] Last reissue price (short): {last_reissue_price_short}")

            replace_long_grid = False
            replace_short_grid = False

            if long_pos_qty > 0 and last_reissue_price_long is not None:
                required_price_move_long_pct = dynamic_outer_price_distance_long * 100.0
                price_change_pct_long = abs(current_price - last_reissue_price_long) / last_reissue_price_long * 100.0

                logging.info(f"[{symbol}] Long position info:")
                logging.info(f"Dynamic outer price distance: {dynamic_outer_price_distance_long * 100.0:.2f}%")
                logging.info(f"  - Long position price: {long_pos_price}")
                logging.info(f"  - Long position quantity: {long_pos_qty}")
                logging.info(f"  - Required price move for reissue (long): {required_price_move_long_pct:.2f}%")
                logging.info(f"  - Current price change percentage: {price_change_pct_long:.2f}%")

                if price_change_pct_long > required_price_move_long_pct:
                    replace_long_grid = True
                    logging.info(f"[{symbol}] Price change exceeds dynamic outer price distance percentage for long position. Replacing long grid.")
                    self.last_reissue_price_long[symbol] = current_price  # Update last reissue price for long

            if short_pos_qty > 0 and last_reissue_price_short is not None:
                required_price_move_short_pct = dynamic_outer_price_distance_short * 100.0
                price_change_pct_short = abs(current_price - last_reissue_price_short) / last_reissue_price_short * 100.0

                logging.info(f"[{symbol}] Short position info:")
                logging.info(f"Dynamic outer price distance: {dynamic_outer_price_distance_short * 100.0:.2f}%")
                logging.info(f"  - Short position price: {short_pos_price}")
                logging.info(f"  - Short position quantity: {short_pos_qty}")
                logging.info(f"  - Required price move for reissue (short): {required_price_move_short_pct:.2f}%")
                logging.info(f"  - Current price change percentage: {price_change_pct_short:.2f}%")

                if price_change_pct_short > required_price_move_short_pct:
                    replace_short_grid = True
                    logging.info(f"[{symbol}] Price change exceeds dynamic outer price distance percentage for short position. Replacing short grid.")
                    self.last_reissue_price_short[symbol] = current_price  # Update last reissue price for short

            logging.info(f"[{symbol}] Should replace long grid: {replace_long_grid}")
            logging.info(f"[{symbol}] Should replace short grid: {replace_short_grid}")

            return replace_long_grid, replace_short_grid

        except Exception as e:
            logging.exception(f"Exception caught in should_replace_grid_updated_buffer_min_outerpricedist_v2: {e}")
            return False, False
            
    def should_replace_grid_updated_buffer_min_outerpricedist(self, symbol: str, long_pos_price: float, short_pos_price: float, long_pos_qty: float, short_pos_qty: float, min_outer_price_distance: float) -> tuple:
        try:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"[{symbol}] Current price: {current_price}")

            # Retrieve last recorded price
            last_price = self.last_price.get(symbol)
            if last_price is None:
                self.last_price[symbol] = current_price
                logging.info(f"[{symbol}] No last price recorded. Setting current price {current_price} as last price. No reissue required.")
                return False, False

            logging.info(f"[{symbol}] Last recorded price: {last_price}")

            replace_long_grid = False
            replace_short_grid = False

            if long_pos_qty > 0:
                required_price_move_long_pct = min_outer_price_distance * 100.0
                price_change_pct_long = abs(current_price - last_price) / last_price * 100.0

                logging.info(f"[{symbol}] Long position info:")
                logging.info(f"Minimum outer price distance: {min_outer_price_distance * 100.0:.2f}%")
                logging.info(f"  - Long position price: {long_pos_price}")
                logging.info(f"  - Long position quantity: {long_pos_qty}")
                logging.info(f"  - Required price move for reissue (long): {required_price_move_long_pct:.2f}%")
                logging.info(f"  - Current price change percentage: {price_change_pct_long:.2f}%")

                if price_change_pct_long > required_price_move_long_pct:
                    replace_long_grid = True
                    logging.info(f"[{symbol}] Price change exceeds minimum outer price distance percentage for long position. Replacing long grid.")
                    self.last_price[symbol] = current_price  # Update last price after condition is met
                else:
                    logging.info(f"[{symbol}] Price change does not exceed minimum outer price distance percentage for long position. No need to replace long grid.")

            if short_pos_qty > 0:
                required_price_move_short_pct = min_outer_price_distance * 100.0
                price_change_pct_short = abs(current_price - last_price) / last_price * 100.0

                logging.info(f"[{symbol}] Short position info:")
                logging.info(f"Minimum outer price distance: {min_outer_price_distance * 100.0:.2f}%")
                logging.info(f"  - Short position price: {short_pos_price}")
                logging.info(f"  - Short position quantity: {short_pos_qty}")
                logging.info(f"  - Required price move for reissue (short): {required_price_move_short_pct:.2f}%")
                logging.info(f"  - Current price change percentage: {price_change_pct_short:.2f}%")

                if price_change_pct_short > required_price_move_short_pct:
                    replace_short_grid = True
                    logging.info(f"[{symbol}] Price change exceeds minimum outer price distance percentage for short position. Replacing short grid.")
                    self.last_price[symbol] = current_price  # Update last price after condition is met
                else:
                    logging.info(f"[{symbol}] Price change does not exceed minimum outer price distance percentage for short position. No need to replace short grid.")

            logging.info(f"[{symbol}] Should replace long grid: {replace_long_grid}")
            logging.info(f"[{symbol}] Should replace short grid: {replace_short_grid}")

            return replace_long_grid, replace_short_grid

        except Exception as e:
            logging.exception(f"Exception caught in should_replace_grid_updated_buffer: {e}")
            return False, False

    def should_reissue_orders_revised(self, symbol: str, reissue_threshold: float, long_pos_qty: float, short_pos_qty: float, initial_entry_buffer_pct: float) -> tuple:
        try:
            current_price = self.exchange.get_current_price(symbol)
            last_price = self.last_price.get(symbol)
            
            if last_price is None:
                self.last_price[symbol] = current_price
                logging.info(f"[{symbol}] No last price recorded. Setting current price {current_price} as last price. No reissue required.")
                return False, False
            
            # Ensure last_price is not None before performing the subtraction
            if last_price is not None:
                price_change_percentage = abs(current_price - last_price) / last_price * 100
                logging.info(f"[{symbol}] Last recorded price: {last_price}, Current price: {current_price}, Price change: {price_change_percentage:.2f}%")
            else:
                logging.warning(f"[{symbol}] Last price is None, skipping reissue check.")
                return False, False
            
            # Adjust threshold by initial buffer percentage correctly
            adjusted_reissue_threshold = reissue_threshold + (reissue_threshold * initial_entry_buffer_pct / 100)
            
            # Define a small threshold to determine if the position is effectively zero
            position_threshold = 0.00001
            
            reissue_long = long_pos_qty < position_threshold and price_change_percentage >= adjusted_reissue_threshold and short_pos_qty < position_threshold
            reissue_short = short_pos_qty < position_threshold and price_change_percentage >= adjusted_reissue_threshold and long_pos_qty < position_threshold
            
            # Only update last price if a reissue is needed
            if reissue_long or reissue_short:
                self.last_price[symbol] = current_price

            if reissue_long:
                logging.info(f"[{symbol}] Price change ({price_change_percentage:.2f}%) exceeds adjusted reissue threshold ({adjusted_reissue_threshold:.2f}%). Reissuing long orders.")
            else:
                logging.info(f"[{symbol}] Price change ({price_change_percentage:.2f}%) does not exceed adjusted reissue threshold ({adjusted_reissue_threshold:.2f}%) or long position is open. No reissue required for long orders.")
            
            if reissue_short:
                logging.info(f"[{symbol}] Price change ({price_change_percentage:.2f}%) exceeds adjusted reissue threshold ({adjusted_reissue_threshold:.2f}%). Reissuing short orders.")
            else:
                logging.info(f"[{symbol}] Price change ({price_change_percentage:.2f}%) does not exceed adjusted reissue threshold ({adjusted_reissue_threshold:.2f}%) or short position is open. No reissue required for short orders.")
            
            return reissue_long, reissue_short
        
        except Exception as e:
            logging.exception(f"Exception caught in should_reissue_orders: {e}")
            return False, False


    def should_reissue_orders(self, symbol: str, reissue_threshold: float) -> bool:
        try:
            current_price = self.exchange.get_current_price(symbol)
            last_price = self.last_price.get(symbol)

            if last_price is None:
                self.last_price[symbol] = current_price
                logging.info(f"[{symbol}] No last price recorded. Current price {current_price} set as last price. No reissue required.")
                return False

            price_change_percentage = abs(current_price - last_price) / last_price * 100
            logging.info(f"[{symbol}] Last recorded price: {last_price}, Current price: {current_price}, Price change: {price_change_percentage:.2f}%")

            if price_change_percentage >= reissue_threshold * 100:
                self.last_price[symbol] = current_price
                logging.info(f"[{symbol}] Price change ({price_change_percentage:.2f}%) exceeds reissue threshold ({reissue_threshold*100:.2f}%). Reissuing orders.")
                return True
            else:
                logging.info(f"[{symbol}] Price change ({price_change_percentage:.2f}%) does not exceed reissue threshold ({reissue_threshold*100:.2f}%). No reissue required.")
                return False
        except Exception as e:
            logging.exception(f"Exception caught in should_reissue_orders: {e}")
            return False

    def clear_grid(self, symbol, side, max_retries=50, delay=2):
        """Clear all orders and internal states for a specific grid side with retries, excluding reduce-only orders."""
        logging.info(f"Clearing {side} grid for {symbol}")

        retry_counter = 0
        while retry_counter < max_retries:
            # Cancel all orders for the specified side
            self.cancel_grid_orders(symbol, side)

            # Re-fetch the open orders and filter out reduce-only orders
            open_orders_after_clear = self.retry_api_call(self.exchange.get_open_orders, symbol)
            lingering_orders = [o for o in open_orders_after_clear if o['info']['side'].lower() == side and not o['info'].get('reduceOnly', False)]

            if not lingering_orders:
                # Clear filled levels for the side when no more lingering orders
                self.filled_levels[symbol][side].clear()
                logging.info(f"Successfully cleared {side} grid for {symbol}.")
                
                # Mark the grid as cleared
                if side == 'buy':
                    self.grid_cleared_status[symbol]['long'] = True
                elif side == 'sell':
                    self.grid_cleared_status[symbol]['short'] = True

                break
            else:
                logging.warning(f"Lingering {side} orders detected for {symbol} (non-reduce-only): {lingering_orders}. Retrying...")
                retry_counter += 1
                time.sleep(delay)  # Wait before retrying

        if retry_counter == max_retries:
            logging.error(f"Failed to clear all {side} grid orders for {symbol} after {max_retries} attempts.")
        else:
            logging.info(f"{side.capitalize()} grid cleared after {retry_counter} retries for {symbol}.")


    # def clear_grid(self, symbol, side, max_retries=50, delay=2):
    #     """Clear all orders and internal states for a specific grid side with retries, excluding reduce-only orders."""
    #     logging.info(f"Clearing {side} grid for {symbol}")

    #     retry_counter = 0
    #     while retry_counter < max_retries:
    #         if side == 'buy':
    #             self.cancel_grid_orders(symbol, "buy")
    #         elif side == 'sell':
    #             self.cancel_grid_orders(symbol, "sell")

    #         # Re-fetch the open orders and filter out reduce-only orders
    #         open_orders_after_clear = self.retry_api_call(self.exchange.get_open_orders, symbol)
    #         lingering_orders = [o for o in open_orders_after_clear if o['info']['side'].lower() == side and not o['info'].get('reduceOnly', False)]

    #         if not lingering_orders:
    #             # Clear filled levels for the side when no more lingering orders
    #             if side == 'buy':
    #                 self.filled_levels[symbol]["buy"].clear()
    #                 logging.info(f"Successfully cleared long grid for {symbol}")
    #             elif side == 'sell':
    #                 self.filled_levels[symbol]["sell"].clear()
    #                 logging.info(f"Successfully cleared short grid for {symbol}")
    #             logging.info(f"{side.capitalize()} grid fully cleared for {symbol}.")
    #             break
    #         else:
    #             logging.warning(f"Lingering {side} orders detected for {symbol} (non-reduce-only): {lingering_orders}. Retrying...")
    #             retry_counter += 1
    #             time.sleep(delay)  # Wait before retrying

    #     if retry_counter == max_retries:
    #         logging.error(f"Failed to clear all {side} grid orders for {symbol} after {max_retries} attempts.")
    #     else:
    #         logging.info(f"{side.capitalize()} grid cleared after {retry_counter} retries for {symbol}.")

    # def clear_grid(self, symbol, side):
    #     """Clear all orders and internal states for a specific grid side."""
    #     if side == 'buy':
    #         self.cancel_grid_orders(symbol, "buy")
    #         self.filled_levels[symbol]["buy"].clear()
    #         #self.active_long_grids.discard(symbol)  # Remove the symbol from active long grids
    #     elif side == 'sell':
    #         self.cancel_grid_orders(symbol, "sell")
    #         self.filled_levels[symbol]["sell"].clear()
    #         #self.active_short_grids.discard(symbol)  # Remove the symbol from active short grids
        
    #     logging.info(f"Cleared {side} grid for {symbol}.")
    

    def generate_order_link_id(self, symbol, side, level):
        """
        Generates a unique, short, and descriptive OrderLinkedID for Bybit orders.
        """
        timestamp = int(time.time() * 1000) % 100000  # Use last 5 digits of current timestamp for uniqueness
        level_str = f"{level:.5f}".replace('.', '')[:5]  # Convert level to string, remove '.', and use first 5 characters
        unique_id = f"{symbol[:3]}_{side[0]}_{level_str}_{timestamp}"  # Build a compact OrderLinkedID
        return unique_id[:45]  # Ensure the ID does not exceed 45 characters

    def issue_grid_orders(self, symbol: str, side: str, grid_levels: list, amounts: list, is_long: bool, filled_levels: set):
        """
        Check the status of existing grid orders and place new orders for unfilled levels.
        """
        try:
            # Fetch open orders from the exchange
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            # Get the current price to update last reissue prices
            current_price = self.exchange.get_current_price(symbol)

            # Clear existing grid before placing new orders
            if is_long:
                logging.info(f"Clearing existing long grid for {symbol} before issuing new orders. (line: {inspect.currentframe().f_lineno})")
                self.clear_grid(symbol, 'buy')
                self.last_reissue_price_long[symbol] = current_price
                logging.info(f"Updated last reissue price for long orders of {symbol} to {current_price}")
            else:
                logging.info(f"Clearing existing short grid for {symbol} before issuing new orders. (line: {inspect.currentframe().f_lineno})")
                self.clear_grid(symbol, 'sell')
                self.last_reissue_price_short[symbol] = current_price
                logging.info(f"Updated last reissue price for short orders of {symbol} to {current_price}")

            # Clear the filled_levels set before placing new orders
            filled_levels.clear()

            # Add logging to verify types before the zip operation
            logging.info(f"Inside issue_grid_orders - Type of grid_levels: {type(grid_levels)}, Value: {grid_levels}")
            logging.info(f"Inside issue_grid_orders - Type of amounts: {type(amounts)}, Value: {amounts}")

            # Ensure grid_levels and amounts are lists
            assert isinstance(grid_levels, list), f"Expected grid_levels to be a list, but got {type(grid_levels)}"
            assert isinstance(amounts, list), f"Expected amounts to be a list, but got {type(amounts)}"

            # Ensure all elements within grid_levels and amounts are of correct type
            for level in grid_levels:
                assert isinstance(level, (float, int)), f"Each level in grid_levels should be a float or int, but got {type(level)}"
            
            for amount in amounts:
                assert isinstance(amount, (float, int)), f"Each amount in amounts should be a float or int, but got {type(amount)}"

            # Place new grid orders for unfilled levels
            for level, amount in zip(grid_levels, amounts):
                order_exists = any(order['price'] == level and order['side'].lower() == side.lower() for order in open_orders)
                if not order_exists:
                    order_link_id = self.generate_order_link_id(symbol, side, level)
                    position_idx = 1 if is_long else 2
                    try:
                        order = self.exchange.create_tagged_limit_order_bybit(symbol, side, amount, level, positionIdx=position_idx, orderLinkId=order_link_id)
                        if order and 'id' in order:
                            logging.info(f"Placed {side} order at level {level} for {symbol} with amount {amount}")
                            filled_levels.add(level)  # Add the level to filled_levels
                        else:
                            logging.info(f"Failed to place {side} order at level {level} for {symbol} with amount {amount}")
                    except Exception as e:
                        logging.error(f"Exception when placing {side} order at level {level} for {symbol}: {e}")
                else:
                    logging.info(f"Skipping {side} order at level {level} for {symbol} as it already exists.")

            logging.info(f"[{symbol}] {side.capitalize()} grid orders issued for unfilled levels.")
        except Exception as e:
            logging.error(f"Exception in issue_grid_orders: {e}")

    def cancel_grid_orders(self, symbol: str, side: str):
        try:
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            orders_canceled = 0
            for order in open_orders:
                if order['side'].lower() == side.lower():
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                    orders_canceled += 1
                    logging.info(f"Canceled order {order['id']} for {symbol}")

            if orders_canceled > 0:
                logging.info(f"Canceled {orders_canceled} {side} grid orders for {symbol}")
            else:
                logging.info(f"No {side} grid orders found to cancel for {symbol}")

            # Remove the symbol from active grids
            if side.lower() == 'buy' and orders_canceled > 0:
                self.active_long_grids.discard(symbol)
                logging.info(f"Removed {symbol} from active long grids")
            elif side.lower() == 'sell' and orders_canceled > 0:
                self.active_short_grids.discard(symbol)
                logging.info(f"Removed {symbol} from active short grids")

        except Exception as e:
            logging.error(f"Exception in cancel_grid_orders for {symbol} - {side}: {e}")
            logging.error("Traceback: %s", traceback.format_exc())

    def calculate_total_amount(self, symbol: str, total_equity: float, best_ask_price: float, best_bid_price: float, wallet_exposure_limit: float, user_defined_leverage: float, side: str, levels: int, min_qty: float, enforce_full_grid: bool) -> float:
        logging.info(f"Calculating total amount for {symbol} with total_equity: {total_equity}, best_ask_price: {best_ask_price}, best_bid_price: {best_bid_price}, wallet_exposure_limit: {wallet_exposure_limit}, user_defined_leverage: {user_defined_leverage}, side: {side}, levels: {levels}, min_qty: {min_qty}, enforce_full_grid: {enforce_full_grid}")
        
        # Fetch market data to get the minimum trade quantity for the symbol
        market_data = self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)
        logging.info(f"Minimum quantity for {symbol}: {min_qty}")
        
        # Calculate the minimum quantity in USD value based on the side
        if side == "buy":
            min_qty_usd_value = min_qty * best_ask_price
        elif side == "sell":
            min_qty_usd_value = min_qty * best_bid_price
        else:
            raise ValueError(f"Invalid side: {side}")
        logging.info(f"Minimum quantity USD value for {symbol}: {min_qty_usd_value}")
        
        # Calculate the maximum position value based on total equity, wallet exposure limit, and user-defined leverage
        max_position_value = total_equity * wallet_exposure_limit * user_defined_leverage
        logging.info(f"Maximum position value for {symbol}: {max_position_value}")
        
        if enforce_full_grid:
            # Calculate the total amount based on the maximum position value and number of levels
            total_amount = max(max_position_value // levels, min_qty_usd_value) * levels
        else:
            # Calculate the total amount as a multiple of the minimum quantity USD value
            total_amount = max(max_position_value // min_qty_usd_value, 1) * min_qty_usd_value
        
        logging.info(f"Calculated total amount for {symbol}: {total_amount}")
        
        return total_amount

    def calculate_order_amounts(self, symbol: str, total_amount: float, levels: int, strength: float, qty_precision: float, min_qty: float, enforce_full_grid: bool) -> List[float]:
        logging.info(f"Calculating order amounts for {symbol} with total_amount: {total_amount}, levels: {levels}, strength: {strength}, qty_precision: {qty_precision}, min_qty: {min_qty}, enforce_full_grid: {enforce_full_grid}")
        
        # Calculate the order amounts based on the strength
        amounts = []
        total_ratio = sum([(j + 1) ** strength for j in range(levels)])
        remaining_amount = total_amount
        logging.info(f"Total ratio: {total_ratio}, Remaining amount: {remaining_amount}")
        
        for i in range(levels):
            ratio = (i + 1) ** strength
            amount = total_amount * (ratio / total_ratio)
            logging.info(f"Level {i+1} - Ratio: {ratio}, Amount: {amount}")
            
            if enforce_full_grid:
                # Round the order amount to the nearest multiple of min_qty
                rounded_amount = round(amount / min_qty) * min_qty
            else:
                # Round the order amount to the nearest multiple of qty_precision or min_qty, whichever is larger
                rounded_amount = round(amount / max(qty_precision, min_qty)) * max(qty_precision, min_qty)
            
            logging.info(f"Level {i+1} - Rounded amount: {rounded_amount}")
            
            # Ensure the order amount is greater than or equal to the minimum quantity
            adjusted_amount = max(rounded_amount, min_qty * (i + 1))
            logging.info(f"Level {i+1} - Adjusted amount: {adjusted_amount}")
            
            amounts.append(adjusted_amount)
            remaining_amount -= adjusted_amount
            logging.info(f"Level {i+1} - Remaining amount: {remaining_amount}")
        
        # If enforce_full_grid is True and there is remaining amount, distribute it among the levels
        if enforce_full_grid and remaining_amount > 0:
            # Sort the amounts in ascending order
            sorted_amounts = sorted(amounts)
            
            # Iterate over the sorted amounts and add the remaining amount until it is fully distributed
            for i in range(len(sorted_amounts)):
                if remaining_amount <= 0:
                    break
                
                # Calculate the additional amount to add to the current level
                additional_amount = min(remaining_amount, min_qty)
                
                # Find the index of the current amount in the original amounts list
                index = amounts.index(sorted_amounts[i])
                
                # Update the amount in the original amounts list
                amounts[index] += additional_amount
                
                remaining_amount -= additional_amount
        
        logging.info(f"Calculated order amounts: {amounts}")
        return amounts

    def min_notional(self, symbol):
        base_notional_values = {"BTCUSDT": 100.5, "ETHUSDT": 20.1, "default": 6}
        return base_notional_values.get(symbol, base_notional_values["default"])

    def get_effective_leverage(self, user_defined_leverage, symbol, side):
        if user_defined_leverage in (0, None):
            # Log the defaulting action for clarity
            max_leverage = self.exchange.get_current_max_leverage_bybit(symbol, side)
            logging.info(f"No user-defined leverage specified for {symbol} on {side} side, using exchange max leverage: {max_leverage}")
            return max_leverage
        return user_defined_leverage
    
    def get_min_qty(self, symbol):
        market_data = self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)
        return float(market_data["min_qty"])

    def calculate_order_amounts_progressive_distribution(self, symbol: str, total_equity: float, 
                                                    best_ask_price: float, best_bid_price: float,
                                                    wallet_exposure_limit_long: float, 
                                                    wallet_exposure_limit_short: float, 
                                                    levels: int, qty_precision: float, 
                                                    side: str, strength: float, long_pos_qty=0, short_pos_qty=0) -> List[float]:
        logging.info(f"Calculating progressive drawdown order amounts for {symbol} with full position value distribution across {levels} levels.")
        
        # Determine the wallet exposure limit based on the side
        wallet_exposure_limit = wallet_exposure_limit_long if side == 'buy' else wallet_exposure_limit_short
        
        # Calculate the maximum position value for the symbol (no subtraction of current position)
        max_position_value = total_equity * wallet_exposure_limit
        logging.info(f"Maximum position value for {symbol}: {max_position_value}")
        
        # Get the minimum notional and quantity
        min_qty = self.get_min_qty(symbol)
        min_notional = self.min_notional(symbol)  # Use self.min_notional(symbol) here

        # Track the cumulative position value to ensure progressively larger orders
        cumulative_position_value = long_pos_qty * best_ask_price if side == 'buy' else short_pos_qty * best_bid_price

        # Distribute the full max position value across the specified number of levels
        amounts = []
        
        # Calculate the weighted distribution based on strength
        total_ratio = sum([(i + 1) ** strength for i in range(levels)])
        level_notional_factors = [(i + 1) ** strength / total_ratio for i in range(levels)]
        
        current_price = best_ask_price if side == 'buy' else best_bid_price

        for i in range(levels):
            # Calculate the notional amount for this level
            level_notional = max_position_value * level_notional_factors[i]
            quantity = level_notional / current_price
            
            # Ensure that the grid levels are larger than the current open position or previous levels
            if i == 0:
                # Ensure the first level is larger than the open position
                quantity = max(quantity, cumulative_position_value / current_price + (i + 1) * qty_precision)
            else:
                # Ensure each subsequent level is larger than the previous one
                if quantity <= amounts[-1]:
                    quantity = amounts[-1] + (i + 1) * qty_precision

            # Enforce the minimum notional or quantity
            notional_value = quantity * current_price
            if notional_value < min_notional:
                logging.info(f"Level {i+1}: Adjusting order to respect minimum notional requirement.")
                quantity = max(min_notional / current_price, min_qty)  # Ensure min_qty is respected
            
            # Round the quantity based on precision and ensure it's above the minimum quantity
            rounded_quantity = max(round(quantity / qty_precision) * qty_precision, min_qty)
            
            # Add the calculated and rounded quantity to the amounts list
            amounts.append(rounded_quantity)
            logging.info(f"Level {i+1} - Progressive drawdown order quantity: {rounded_quantity}")
            
            # Update cumulative position value after adding the order
            cumulative_position_value += rounded_quantity * current_price

        return amounts

    def calculate_order_amounts_aggressive_drawdown(self, symbol: str, total_equity: float, 
                                                    best_ask_price: float, best_bid_price: float,
                                                    wallet_exposure_limit_long: float, 
                                                    wallet_exposure_limit_short: float, 
                                                    levels: int, qty_precision: float, 
                                                    side: str, strength: float, long_pos_qty=0, short_pos_qty=0) -> List[float]:
        logging.info(f"Calculating aggressive drawdown order amounts for {symbol} with full position value distribution across {levels} levels.")
        
        # Convert long_pos_qty and short_pos_qty to float to ensure correct arithmetic
        try:
            long_pos_qty = float(long_pos_qty)
        except (ValueError, TypeError):
            logging.error(f"Invalid long_pos_qty value for {symbol}: {long_pos_qty}, setting to 0.")
            long_pos_qty = 0

        try:
            short_pos_qty = float(short_pos_qty)
        except (ValueError, TypeError):
            logging.error(f"Invalid short_pos_qty value for {symbol}: {short_pos_qty}, setting to 0.")
            short_pos_qty = 0

        # Determine the wallet exposure limit based on the side
        wallet_exposure_limit = wallet_exposure_limit_long if side == 'buy' else wallet_exposure_limit_short
        
        # Calculate the maximum position value for the symbol
        max_position_value = total_equity * wallet_exposure_limit
        logging.info(f"Maximum position value for {symbol}: {max_position_value}")

        # Calculate the current position value based on the side
        if side == 'buy':
            current_pos_value = long_pos_qty * best_ask_price
        else:
            current_pos_value = short_pos_qty * best_bid_price

        logging.info(f"Current position value for {symbol} on side {side}: {current_pos_value}")

        # Adjust the maximum position value by subtracting the current position value
        adjusted_max_position_value = max_position_value - current_pos_value
        logging.info(f"Adjusted maximum position value for {symbol} after considering current position: {adjusted_max_position_value}")

        # Distribute the full adjusted max position value across the specified number of levels
        amounts = []
        
        # Calculate the weighted distribution based on strength
        total_ratio = sum([(i + 1) ** strength for i in range(levels)])
        level_notional_factors = [(i + 1) ** strength / total_ratio for i in range(levels)]
        
        current_price = best_ask_price if side == 'buy' else best_bid_price
        
        for i in range(levels):
            level_notional = adjusted_max_position_value * level_notional_factors[i]
            quantity = level_notional / current_price
            rounded_quantity = max(round(quantity / qty_precision) * qty_precision, self.get_min_qty(symbol))
            amounts.append(rounded_quantity)
            logging.info(f"Level {i+1} - Aggressive drawdown order quantity: {rounded_quantity}")

        return amounts
    
    def calculate_total_amount_notional_ls_properdca(self, symbol, total_equity, best_ask_price, best_bid_price, 
                                                    wallet_exposure_limit_long, wallet_exposure_limit_short, 
                                                    side, levels, enforce_full_grid, 
                                                    long_pos_qty=0, short_pos_qty=0):
        logging.info(f"Calculating total amount for {symbol} with total_balance: {total_equity}, side: {side}, levels: {levels}, enforce_full_grid: {enforce_full_grid}")

        wallet_exposure_limit = wallet_exposure_limit_long if side == 'buy' else wallet_exposure_limit_short
        max_position_value = total_equity * wallet_exposure_limit
        logging.info(f"Maximum position value for {symbol}: {max_position_value}")

        # Convert to float to ensure correct type
        try:
            long_pos_qty = float(long_pos_qty)
            short_pos_qty = float(short_pos_qty)
            best_ask_price = float(best_ask_price)
            best_bid_price = float(best_bid_price)
        except ValueError as e:
            logging.error(f"Error converting values to float: {e}")
            raise

        # Type checking before multiplication
        assert isinstance(long_pos_qty, (int, float)), f"long_pos_qty is not a number: {long_pos_qty}"
        assert isinstance(short_pos_qty, (int, float)), f"short_pos_qty is not a number: {short_pos_qty}"
        assert isinstance(best_ask_price, (int, float)), f"best_ask_price is not a number: {best_ask_price}"
        assert isinstance(best_bid_price, (int, float)), f"best_bid_price is not a number: {best_bid_price}"

        if enforce_full_grid:
            required_notional = max_position_value / levels
        else:
            required_notional = max_position_value

        if side == 'buy':
            logging.info(f"For {symbol} Type of long_pos_qty before multiplication: {type(long_pos_qty)}, value: {long_pos_qty}")
            logging.info(f"For {symbol} Type of best_ask_price before multiplication: {type(best_ask_price)}, value: {best_ask_price}")
            current_pos_value = long_pos_qty * best_ask_price
        else:
            logging.info(f"For {symbol} Type of short_pos_qty before multiplication: {type(short_pos_qty)}, value: {short_pos_qty}")
            logging.info(f"For {symbol} Type of best_bid_price before multiplication: {type(best_bid_price)}, value: {best_bid_price}")
            current_pos_value = short_pos_qty * best_bid_price

        # Logging the values before calculation
        logging.info(f"For {symbol} Before adjustment: max_position_value: {max_position_value}, current_pos_value: {current_pos_value}")

        adjusted_max_position_value = max_position_value - current_pos_value
        
        # Logging the result of the adjustment
        logging.info(f"Adjusted max position value for {symbol}: {adjusted_max_position_value}")

        # Avoiding negative adjusted max position value
        adjusted_max_position_value = max(adjusted_max_position_value, 0)
        total_notional_amount = min(required_notional, adjusted_max_position_value)

        logging.info(f"Calculated total notional amount for {symbol}: {total_notional_amount}")
        return total_notional_amount

    def calculate_order_amounts_notional_properdca(self, symbol: str, total_amount: float, levels: int, 
                                                    strength: float, qty_precision: float, enforce_full_grid: bool,
                                                    long_pos_qty=0, short_pos_qty=0, side='buy') -> List[float]:
        # Logging the start of the calculation with details
        logging.info(f"Calculating order amounts for {symbol} with total_amount: {total_amount}, levels: {levels}, strength: {strength}, qty_precision: {qty_precision}, enforce_full_grid: {enforce_full_grid}")

        try:
            # Ensure qty_precision is a float
            qty_precision = float(qty_precision)

            # Get the current price of the symbol
            current_price = float(self.exchange.get_current_price(symbol))
            
            # Initialize the list to hold the calculated amounts
            amounts = []
            
            # Calculate the ratio for distributing the amounts across the levels
            total_ratio = sum([(i + 1) ** strength for i in range(levels)])
            level_notional = [(i + 1) ** strength for i in range(levels)]

            # Set the base notional amount
            base_notional = float(total_amount)

            # Retrieve the minimum quantity for the symbol and ensure it's a float
            min_qty = float(self.get_min_qty(symbol))

            # Determine the current position quantity based on the side (long or short)
            if side == 'buy':
                current_position_qty = float(long_pos_qty)
            else:
                current_position_qty = float(short_pos_qty)

            # Adjust the total amount based on the current position
            total_amount_adjusted = total_amount + (current_position_qty * current_price)
            logging.info(f"Total amount adjusted for current position: {total_amount_adjusted}")

            # Return 0 if the adjusted total amount is negative
            if total_amount_adjusted < 0:
                logging.info(f"Negative total_amount_adjusted for {symbol}: {total_amount_adjusted}. Side: {side}, total_amount: {total_amount}, current_position_qty: {current_position_qty}, current_price: {current_price}")
                return 0  # Return 0 if no further orders should be issued

            # Log the types and values of important variables for debugging
            logging.info(f"Type of total_amount: {type(total_amount)}, Value: {total_amount}")
            logging.info(f"Type of current_price: {type(current_price)}, Value: {current_price}")
            logging.info(f"Type of min_qty: {type(min_qty)}, Value: {min_qty}")

            # Loop through the levels to calculate the order amounts
            for i in range(levels):
                # Calculate the notional amount for the current level
                notional_amount = (level_notional[i] / total_ratio) * total_amount_adjusted
                # Calculate the quantity for the current level
                quantity = notional_amount / current_price
                logging.info(f"Level {i+1} - Initial quantity: {quantity}")
                
                # Determine the minimum quantity to use (either min_notional or min_qty)
                min_quantity = max(base_notional / current_price, min_qty)
                
                # Apply the minimum quantity requirement and round the quantity
                rounded_quantity = max(round(quantity / qty_precision) * qty_precision, min_quantity)
                logging.info(f"Level {i+1} - Rounded quantity: {rounded_quantity}")
                
                # Append the calculated quantity to the amounts list
                amounts.append(rounded_quantity)

            # Log the calculated amounts for the symbol
            logging.info(f"Calculated order amounts for {symbol}: {amounts}")
            
            # If enforce_full_grid is True and all amounts are the same, adjust them to ensure variation
            if enforce_full_grid and len(set(amounts)) == 1:
                logging.info("Adjusting amounts to ensure variation in enforce_full_grid mode")
                increment = qty_precision
                for i in range(1, levels):
                    amounts[i] += increment
                    increment += qty_precision

            # Return the final calculated amounts
            return amounts
        
        except Exception as e:
            # Log any exception that occurs during the calculation process, including the full traceback
            logging.error(f"Exception caught in calculate_order_amounts_notional_properdca for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return 0  # Return 0 if an exception occurs
        
    # def calculate_order_amounts_notional_properdca(self, symbol: str, total_amount: float, levels: int, 
    #                                                 strength: float, qty_precision: float, enforce_full_grid: bool,
    #                                                 long_pos_qty=0, short_pos_qty=0, side='buy') -> List[float]:
    #     logging.info(f"Calculating order amounts for {symbol} with total_amount: {total_amount}, levels: {levels}, strength: {strength}, qty_precision: {qty_precision}, enforce_full_grid: {enforce_full_grid}")
        
    #     current_price = self.exchange.get_current_price(symbol)
    #     amounts = []
    #     total_ratio = sum([(i + 1) ** strength for i in range(levels)])
    #     level_notional = [(i + 1) ** strength for i in range(levels)]

    #     base_notional = total_amount

    #     min_qty = self.get_min_qty(symbol)  # Retrieve the min_qty for the symbol

    #     if side == 'buy':
    #         current_position_qty = long_pos_qty
    #     else:
    #         current_position_qty = short_pos_qty

    #     # Adjust for current position
    #     total_amount_adjusted = total_amount + (current_position_qty * current_price)
    #     logging.info(f"Total amount adjusted for current position: {total_amount_adjusted}")

    #     for i in range(levels):
    #         notional_amount = (level_notional[i] / total_ratio) * total_amount_adjusted
    #         quantity = notional_amount / current_price
    #         logging.info(f"Level {i+1} - Initial quantity: {quantity}")
            
    #         # Determine the minimum quantity to use (either min_notional or min_qty)
    #         min_quantity = max(base_notional / current_price, min_qty)
            
    #         # Apply the minimum quantity requirement
    #         rounded_quantity = max(round(quantity / qty_precision) * qty_precision, min_quantity)
    #         logging.info(f"Level {i+1} - Rounded quantity: {rounded_quantity}")
            
    #         amounts.append(rounded_quantity)

    #     logging.info(f"Calculated order amounts for {symbol}: {amounts}")
        
    #     # Adjust amounts if they are all the same when enforce_full_grid is True
    #     if enforce_full_grid and len(set(amounts)) == 1:
    #         logging.info("Adjusting amounts to ensure variation in enforce_full_grid mode")
    #         increment = qty_precision
    #         for i in range(1, levels):
    #             amounts[i] += increment
    #             increment += qty_precision

    #     return amounts
    
    def calculate_max_positions(self, symbol, total_equity, current_price, max_qty_percent_long, max_qty_percent_short):
        leverage_long = self.get_effective_leverage(self.user_defined_leverage_long, symbol, 'buy')
        leverage_short = self.get_effective_leverage(self.user_defined_leverage_short, symbol, 'sell')

        max_qty_long = (total_equity * (max_qty_percent_long / 100) * leverage_long) / current_price
        max_qty_short = (total_equity * (max_qty_percent_short / 100) * leverage_short) / current_price

        return max_qty_long, max_qty_short
        
    def get_symbol_max_leverage(self, symbol):
        if symbol not in self.symbol_max_leverage:
            max_leverage = self.exchange.get_current_max_leverage_bybit(symbol)
            logging.info(f"Max leverage for {symbol} {max_leverage}")
            self.symbol_max_leverage[symbol] = max_leverage
            logging.info(f"Fetched and stored max leverage for {symbol}: {max_leverage}x")
        return self.symbol_max_leverage[symbol]

    def check_and_manage_positions(self, long_pos_qty, short_pos_qty, symbol, total_equity, current_price, wallet_exposure_limit_long, wallet_exposure_limit_short, max_qty_percent_long, max_qty_percent_short):
        try:
            logging.info(f"Checking and managing positions for {symbol}")

            # Calculate the maximum allowed positions based on the total equity and wallet exposure limits
            max_qty_long = (total_equity * wallet_exposure_limit_long) / current_price
            max_qty_short = (total_equity * wallet_exposure_limit_short) / current_price

            # Calculate the position exposure percentages
            long_pos_exposure_percent = (long_pos_qty * current_price / total_equity) * 100
            short_pos_exposure_percent = (short_pos_qty * current_price / total_equity) * 100

            # Log the position utilization and the actual utilization based on total equity
            logging.info(f"Position utilization for {symbol}:")
            logging.info(f"  - Long position exposure: {long_pos_exposure_percent:.2f}% of total equity")
            logging.info(f"  - Short position exposure: {short_pos_exposure_percent:.2f}% of total equity")

            # Log detailed information about the configuration parameters and maximum allowed positions
            logging.info(f"Configuration for {symbol}:")
            logging.info(f"  - Total equity: {total_equity:.2f} USD")
            logging.info(f"  - Current price: {current_price:.8f} USD")
            logging.info(f"  - Wallet exposure limit for long: {wallet_exposure_limit_long * 100:.2f}%")
            logging.info(f"  - Wallet exposure limit for short: {wallet_exposure_limit_short * 100:.2f}%")
            logging.info(f"  - Max quantity percentage for long: {max_qty_percent_long}%")
            logging.info(f"  - Max quantity percentage for short: {max_qty_percent_short}%")
            logging.info(f"Maximum allowed positions for {symbol}:")
            logging.info(f"  - Max quantity for long: {max_qty_long:.4f} units")
            logging.info(f"  - Max quantity for short: {max_qty_short:.4f} units")
            logging.info(f"{symbol} Long position quantity: {long_pos_qty}")
            logging.info(f"{symbol} Short position quantity: {short_pos_qty}")

            # Check if current positions exceed the maximum allowed quantities
            if long_pos_exposure_percent > max_qty_percent_long:
                logging.info(f"[{symbol}] Long position exposure exceeds the maximum allowed. Current long position exposure: {long_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_long}%. Clearing long grid.")
                self.clear_grid(symbol, 'buy')
                self.active_grids.discard(symbol)
                self.max_qty_reached_symbol_long.add(symbol)
            elif symbol in self.max_qty_reached_symbol_long and long_pos_exposure_percent <= max_qty_percent_long:
                logging.info(f"[{symbol}] Long position exposure is below the maximum allowed. Removing from max_qty_reached_symbol_long. Current long position exposure: {long_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_long}%.")
                self.max_qty_reached_symbol_long.remove(symbol)

            if short_pos_exposure_percent > max_qty_percent_short:
                logging.info(f"[{symbol}] Short position exposure exceeds the maximum allowed. Current short position exposure: {short_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_short}%. Clearing short grid.")
                self.clear_grid(symbol, 'sell')
                self.active_grids.discard(symbol)
                self.max_qty_reached_symbol_short.add(symbol)
            elif symbol in self.max_qty_reached_symbol_short and short_pos_exposure_percent <= max_qty_percent_short:
                logging.info(f"[{symbol}] Short position exposure is below the maximum allowed. Removing from max_qty_reached_symbol_short. Current short position exposure: {short_pos_exposure_percent:.2f}%, Max allowed: {max_qty_percent_short}%.")
                self.max_qty_reached_symbol_short.remove(symbol)

        except Exception as e:
            logging.error(f"Exception caught in check and manage positions: {e}")
            logging.info("Traceback:", traceback.format_exc())

    def calculate_total_amount_notional_ls(self, symbol, total_equity, best_ask_price, best_bid_price, 
                                            wallet_exposure_limit_long, wallet_exposure_limit_short, 
                                            side, levels, enforce_full_grid, 
                                            user_defined_leverage_long=None, user_defined_leverage_short=None):
        logging.info(f"Calculating total amount for {symbol} with total_equity: {total_equity}, side: {side}, levels: {levels}, enforce_full_grid: {enforce_full_grid}")

        leverage_used = user_defined_leverage_long if side == 'buy' and user_defined_leverage_long not in (0, None) else \
                        user_defined_leverage_short if side == 'sell' and user_defined_leverage_short not in (0, None) else \
                        self.exchange.get_current_max_leverage_bybit(symbol)
        logging.info(f"Using leverage for {symbol}: {leverage_used}")

        wallet_exposure_limit = wallet_exposure_limit_long if side == 'buy' else wallet_exposure_limit_short
        max_position_value = total_equity * wallet_exposure_limit * leverage_used
        logging.info(f"Maximum position value for {symbol}: {max_position_value}")

        base_notional = self.min_notional(symbol)
        required_notional = base_notional * levels if enforce_full_grid else max_position_value
        total_notional_amount = min(required_notional, max_position_value)

        logging.info(f"Calculated total notional amount for {symbol}: {total_notional_amount}")
        return total_notional_amount

    def calculate_order_amounts_notional(self, symbol: str, total_amount: float, levels: int, strength: float, qty_precision: float, enforce_full_grid: bool) -> List[float]:
        logging.info(f"Calculating order amounts for {symbol} with total_amount: {total_amount}, levels: {levels}, strength: {strength}, qty_precision: {qty_precision}, enforce_full_grid: {enforce_full_grid}")
        
        current_price = self.exchange.get_current_price(symbol)
        amounts = []
        total_ratio = sum([(i + 1) ** strength for i in range(levels)])
        level_notional = [(i + 1) ** strength for i in range(levels)]
        
        base_notional = self.min_notional(symbol)
        min_base_notional = base_notional / current_price
        
        for i in range(levels):
            notional_amount = (level_notional[i] / total_ratio) * total_amount
            quantity = notional_amount / current_price
            rounded_quantity = max(round(quantity / qty_precision) * qty_precision, min_base_notional)
            amounts.append(rounded_quantity)

        logging.info(f"Calculated order amounts for {symbol}: {amounts}")
        return amounts

    def initiate_spread_entry(self, symbol, open_orders, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty):
        order_book = self.exchange.get_orderbook(symbol)
        best_ask_price = order_book['asks'][0][0]
        best_bid_price = order_book['bids'][0][0]
        
        long_dynamic_amount = self.m_order_amount(symbol, "long", long_dynamic_amount)
        short_dynamic_amount = self.m_order_amount(symbol, "short", short_dynamic_amount)
        
        # Calculate order book imbalance
        depth = self.ORDER_BOOK_DEPTH
        top_bids = order_book['bids'][:depth]
        total_bids = sum([bid[1] for bid in top_bids])
        top_asks = order_book['asks'][:depth]
        total_asks = sum([ask[1] for ask in top_asks])
        
        if total_bids > total_asks:
            imbalance = "buy_wall"
        elif total_asks > total_bids:
            imbalance = "sell_wall"
        else:
            imbalance = "neutral"
        
        # Entry Logic
        if imbalance == "buy_wall" and not self.entry_order_exists(open_orders, "buy") and long_pos_qty <= 0:
            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
        elif imbalance == "sell_wall" and not self.entry_order_exists(open_orders, "sell") and short_pos_qty <= 0:
            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def get_order_book_imbalance(self, symbol):
        order_book = self.exchange.get_orderbook(symbol)
        
        depth = self.ORDER_BOOK_DEPTH
        top_bids = order_book['bids'][:depth]
        total_bids = sum([bid[1] for bid in top_bids])
        
        top_asks = order_book['asks'][:depth]
        total_asks = sum([ask[1] for ask in top_asks])
        
        if total_bids > total_asks:
            return "buy_wall"
        elif total_asks > total_bids:
            return "sell_wall"
        else:
            return "neutral"

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

    def bybit_turbocharged_entry_maker_walls(self, symbol, trend, mfi, one_minute_volume, five_minute_distance, min_vol, min_dist, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):
        if one_minute_volume is None or five_minute_distance is None or one_minute_volume <= min_vol or five_minute_distance <= min_dist:
            logging.warning(f"Either 'one_minute_volume' or 'five_minute_distance' does not meet the criteria for symbol {symbol}. Skipping current execution...")
            return

        order_book = self.exchange.get_orderbook(symbol)

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

    def bybit_turbocharged_entry_maker(self, open_orders, symbol, trend, mfi, one_minute_volume: float, five_minute_distance: float, min_vol, min_dist, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_long, should_add_to_long, should_short, should_add_to_short):

        if not (one_minute_volume and five_minute_distance) or one_minute_volume <= min_vol or five_minute_distance <= min_dist:
            logging.warning(f"Either 'one_minute_volume' or 'five_minute_distance' does not meet the criteria for symbol {symbol}. Skipping current execution...")
            return

        current_price = self.exchange.get_current_price(symbol)
        logging.info(f"[{symbol}] Current price: {current_price}")

        order_book = self.exchange.get_orderbook(symbol)
        best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
        best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
    
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
