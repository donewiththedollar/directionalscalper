from colorama import Fore
from sklearn.cluster import DBSCAN
from typing import Optional, Tuple, List, Dict, Union
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_DOWN
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
from sklearn.preprocessing import MinMaxScaler
from .logger import Logger
from datetime import datetime, timedelta
from threading import Thread, Lock

from ..bot_metrics import BotDatabase

from rate_limit import RateLimit


logging = Logger(logger_name="BaseStrategy", filename="BaseStrategy.log", stream=True)

class OrderBookAnalyzer:
    def __init__(self, exchange, symbol):
        self.exchange = exchange
        self.symbol = symbol

    def get_order_book(self):
        try:
            order_book = self.exchange.fetch_order_book(self.symbol)
            return order_book
        except Exception as e:
            logging.error(f"Error fetching order book for {self.symbol}: {e}")
            return None

    def get_best_prices(self):
        order_book = self.get_order_book()
        if order_book:
            best_bid_price = order_book['bids'][0][0] if order_book['bids'] else None
            best_ask_price = order_book['asks'][0][0] if order_book['asks'] else None
            return best_bid_price, best_ask_price
        return None, None

    def calculate_average_prices(self, top_n=5):
        order_book = self.get_order_book()
        if order_book:
            top_asks = order_book['asks'][:top_n]
            top_bids = order_book['bids'][:top_n]
            avg_top_asks = sum([ask[0] for ask in top_asks]) / len(top_asks) if top_asks else None
            avg_top_bids = sum([bid[0] for bid in top_bids]) / len(top_bids) if top_bids else None
            return avg_top_asks, avg_top_bids
        return None, None

    def identify_walls(self, order_book, side, threshold=0.5):
        """ Identify buy or sell walls based on a volume threshold """
        walls = []
        if side == "buy":
            orders = order_book['bids']
        elif side == "sell":
            orders = order_book['asks']
        else:
            logging.error("Invalid side specified for identifying walls. Choose 'buy' or 'sell'.")
            return walls

        total_volume = sum([order[1] for order in orders])
        cumulative_volume = 0

        for order in orders:
            price, volume = order
            cumulative_volume += volume
            if cumulative_volume / total_volume >= threshold:
                walls.append(price)
                break

        return walls

    def get_order_book_imbalance(self):
        order_book = self.get_order_book()
        if not order_book:
            return None

        total_bids = sum([bid[1] for bid in order_book['bids']])
        total_asks = sum([ask[1] for ask in order_book['asks']])

        if total_bids > total_asks * 1.5:
            return "buy_wall"
        elif total_asks > total_bids * 1.5:
            return "sell_wall"
        else:
            return "neutral"
        
class BaseStrategy:
    initialized_symbols = set()
    initialized_symbols_lock = threading.Lock()

    def __init__(self, exchange, config, manager, symbols_allowed=None):
        self.exchange = exchange
        self.config = config
        self.manager = manager
        # self.symbol = config.symbol
        self.symbols_allowed = symbols_allowed
        self.order_timestamps = {}
        self.entry_order_ids = {}
        self.long_dynamic_amount = {}
        self.short_dynamic_amount = {}
        self.printed_trade_quantities = False
        self.last_mfirsi_signal = None
        self.TAKER_FEE_RATE = Decimal("0.00055")
        self.taker_fee_rate = 0.055 / 100
        self.max_long_trade_qty = None
        self.max_short_trade_qty = None
        self.initial_max_long_trade_qty = None
        self.initial_max_short_trade_qty = None
        self.long_leverage_increased = False
        self.short_leverage_increased = False
        self.open_symbols_count = 0
        self.last_stale_order_check_time = time.time()
        self.helper_active = False
        self.helper_wall_size = 5
        self.helper_interval = 1  # Time interval between helper actions
        self.helper_duration = 5  # Helper duration in seconds
        self.LEVERAGE_STEP = 0.002
        #self.MAX_LEVERAGE = 0.1
        self.MAX_LEVERAGE = None
        self.QTY_INCREMENT = 0.01
        self.MAX_PCT_EQUITY = 0.1
        self.ORDER_BOOK_DEPTH = 10
        self.lock = threading.Lock()
        self.last_known_ask = {}
        self.last_known_bid = {}
        self.last_order_time = {}
        self.symbol_locks = {}
        self.order_ids = {}
        self.hedged_symbols = {}
        self.hedged_positions = {}
        self.blacklist = self.config.blacklist
        self.max_usd_value = self.config.max_usd_value
        self.auto_reduce_start_pct = self.config.auto_reduce_start_pct
        self.auto_reduce_maxloss_pct = self.config.auto_reduce_maxloss_pct
        self.max_pos_balance_pct = self.config.max_pos_balance_pct
        self.last_entries_cancel_time = 0
        self.MIN_RISK_LEVEL = 0.001
        self.MAX_RISK_LEVEL = 10
        self.auto_reduce_active_long = {}
        self.auto_reduce_active_short = {}
        self.auto_reduce_orders = {}
        self.auto_reduce_order_ids = {}
        self.previous_levels = {}
        self.max_long_trade_qty_per_symbol = {}
        self.max_short_trade_qty_per_symbol = {}
        self.initial_max_long_trade_qty_per_symbol = {}
        self.initial_max_short_trade_qty_per_symbol = {}
        self.long_pos_leverage_per_symbol = {}
        self.short_pos_leverage_per_symbol = {}
        self.dynamic_amount_per_symbol = {}
        self.max_trade_qty_per_symbol = {}
        self.last_auto_reduce_time = {}
        self.rate_limiter = RateLimit(10, 1)
        self.general_rate_limiter = RateLimit(50, 1)
        self.order_rate_limiter = RateLimit(5, 1) 
        self.last_known_mas = {}

        # self.bybit = self.Bybit(self)

    def dbscan_classification(self, ohlcv_data, zigzag_length, epsilon_deviation, aggregate_range):
        logging.info(f"Starting dbscan_classification with zigzag_length={zigzag_length}, epsilon_deviation={epsilon_deviation}, aggregate_range={aggregate_range}")

        # Extract highs and lows from the OHLCV data
        highs = np.array([candle['high'] for candle in ohlcv_data])
        lows = np.array([candle['low'] for candle in ohlcv_data])
        logging.info(f"Extracted highs: {highs}, lows: {lows}")

        peaks_and_troughs = []

        direction_up = False
        last_low = np.max(highs) * 100
        last_high = 0.0

        # Detect peaks and troughs
        for i in range(zigzag_length, len(ohlcv_data) - zigzag_length):
            h = np.max(highs[i - zigzag_length:i + zigzag_length + 1])
            l = np.min(lows[i - zigzag_length:i + zigzag_length + 1])
            logging.info(f"Evaluating at index {i}: high={h}, low={l}, direction_up={direction_up}")

            # Try a smaller zigzag_length
            zigzag_length = max(1, zigzag_length // 2)

            # Or adjust conditions in the loop
            if direction_up:
                if l < last_low:  # Less strict than 'l == ohlcv_data[i]['low']'
                    last_low = l
                    peaks_and_troughs.append(last_low)
                if h > last_high:  # Less strict than 'h == ohlcv_data[i]['high']'
                    last_high = h
                    direction_up = False
                    peaks_and_troughs.append(last_high)
            else:
                if h > last_high:
                    last_high = h
                    peaks_and_troughs.append(last_high)
                if l < last_low:
                    last_low = l
                    direction_up = True
                    peaks_and_troughs.append(last_low)

        # Convert peaks_and_troughs to a numpy array
        zigzag = np.array(peaks_and_troughs)
        logging.info(f"Generated zigzag array: {zigzag}")

        # Check if zigzag array is empty
        if zigzag.size == 0:
            logging.info("Zigzag array is empty. No peaks or troughs detected.")
            return []

        # Normalize the peaks and troughs
        min_price = np.min(zigzag)
        max_price = np.max(zigzag)
        logging.info(f"Zigzag min_price: {min_price}, max_price: {max_price}")

        normalized_zigzag = (zigzag - min_price) / (max_price - min_price)
        logging.info(f"Normalized zigzag array: {normalized_zigzag}")

        # Calculate the mean deviation
        mean = np.mean(normalized_zigzag)
        deviation = np.mean(np.abs(normalized_zigzag - mean))
        logging.info(f"Calculated mean: {mean}, deviation: {deviation}")

        # Define the epsilon value for DBSCAN
        epsilon = (deviation * epsilon_deviation) / 100.0
        logging.info(f"Calculated epsilon for DBSCAN: {epsilon}")

        # Prepare data points for DBSCAN
        data_points = normalized_zigzag.reshape(-1, 1)
        logging.info(f"Data points prepared for DBSCAN: {data_points}")

        # Run DBSCAN clustering
        dbscan = DBSCAN(eps=epsilon, min_samples=1, metric='euclidean')
        dbscan.fit(data_points)
        logging.info(f"DBSCAN labels: {dbscan.labels_}")

        # Extract clusters and noise
        clusters = []
        for label in set(dbscan.labels_):
            if label != -1:  # -1 means noise
                cluster = [i for i, l in enumerate(dbscan.labels_) if l == label]
                clusters.append(cluster)
                logging.info(f"Detected cluster with label {label}: {cluster}")

        noise = [i for i, l in enumerate(dbscan.labels_) if l == -1]
        logging.info(f"Detected noise points: {noise}")

        # Aggregate and filter clusters into significant levels
        support_resistance_levels = []
        for cluster in clusters:
            cluster_prices = zigzag[cluster]
            cluster_volumes = [ohlcv_data[i]['volume'] for i in cluster]
            median_price = np.median(cluster_prices)
            average_volume = np.mean(cluster_volumes)
            strength = len(cluster_prices)
            support_resistance_levels.append({
                'level': median_price,
                'strength': strength,
                'average_volume': average_volume
            })
            logging.info(f"Added support/resistance level: {median_price}, strength: {strength}, average volume: {average_volume}")

        # Add significant noise levels
        max_level = np.max([level['level'] for level in support_resistance_levels])
        min_level = np.min([level['level'] for level in support_resistance_levels])

        for i in noise:
            noise_level = zigzag[i]
            noise_volume = ohlcv_data[i]['volume']
            if noise_level > max_level and (noise_level - max_level) / max_level > aggregate_range / 100.0:
                support_resistance_levels.append({
                    'level': noise_level,
                    'strength': 1,
                    'average_volume': noise_volume
                })
                logging.info(f"Added significant noise level above max level: {noise_level}")
            elif noise_level < min_level and (min_level - noise_level) / min_level > aggregate_range / 100.0:
                support_resistance_levels.append({
                    'level': noise_level,
                    'strength': 1,
                    'average_volume': noise_volume
                })
                logging.info(f"Added significant noise level below min level: {noise_level}")

        # Sort the levels by price level in descending order
        support_resistance_levels.sort(key=lambda x: x['level'], reverse=True)
        logging.info(f"Sorted support/resistance levels: {support_resistance_levels}")

        # Filter out closely grouped levels
        filtered_levels = []
        i = 0
        while i < len(support_resistance_levels):
            current_group = [support_resistance_levels[i]]
            j = i + 1

            while j < len(support_resistance_levels) and \
                    abs(support_resistance_levels[j]['level'] - support_resistance_levels[i]['level']) / support_resistance_levels[i]['level'] <= aggregate_range / 100.0:
                current_group.append(support_resistance_levels[j])
                j += 1

            current_group.sort(key=lambda x: x['average_volume'], reverse=True)
            filtered_levels.append(current_group[0])
            i = j
            logging.info(f"Filtered level added: {current_group[0]}")

        # Finalize the levels by removing close duplicates
        final_levels = []
        for k in range(len(filtered_levels)):
            if len(final_levels) == 0 or \
                    abs(filtered_levels[k]['level'] - final_levels[-1]['level']) / final_levels[-1]['level'] > aggregate_range / 100.0:
                final_levels.append(filtered_levels[k])
                logging.info(f"Final level added: {filtered_levels[k]}")
            else:
                for m in range(k + 1, len(filtered_levels)):
                    if abs(filtered_levels[m]['level'] - final_levels[-1]['level']) / final_levels[-1]['level'] > aggregate_range / 100.0:
                        final_levels.append(filtered_levels[m])
                        k = m
                        logging.info(f"Final level added after checking close duplicates: {filtered_levels[m]}")
                        break

        # Sort final levels in descending order
        final_levels.sort(key=lambda x: x['level'], reverse=True)
        logging.info(f"Final sorted levels: {final_levels}")

        return final_levels

    def update_hedged_status(self, symbol, is_hedged):
        self.hedged_positions[symbol] = is_hedged

    def compute_average_daily_gain_percentage(self, initial_equity, current_equity, days_passed):
        """Compute average daily gain percentage."""
        if days_passed == 0:  # To prevent division by zero
            return 0
        gain = (current_equity - initial_equity) / initial_equity * 100
        avg_daily_gain = gain / days_passed
        return avg_daily_gain

    def convert_to_boolean(value):
        return value.lower() == "true"

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
            logging.info(f"Error in calculate_adg: {e}")
            return None

    def fetch_closed_trades_history(self, days):
        """
        Fetch the closed trades history for the specified number of days.
        This method should be implemented in the Exchange class.
        """
        # This is a placeholder. You need to implement this method based on your exchange's API.
        pass

    def get_position_balance(self, symbol, side, open_position_data):
        """
        Retrieves the position balance for a given symbol and side from open position data.

        :param symbol: The trading symbol (e.g., 'BTCUSDT').
        :param side: The side of the position ('Buy' for long, 'Sell' for short).
        :param open_position_data: The data containing information about open positions.
        :return: The position balance for the specified symbol and side, or 0 if not found.
        """
        for position in open_position_data:
            # Extract info from each position
            info = position.get('info', {})
            symbol_from_position = info.get('symbol', '').split(':')[0]
            side_from_position = info.get('side', '')
            position_balance = float(info.get('positionBalance', 0) or 0)

            # Check if the symbol and side match the requested ones
            if symbol_from_position == symbol and side_from_position == side:
                return position_balance

        # Return 0 if the symbol and side combination is not found
        return 0

    def get_open_symbols_long(self, open_position_data):
        """
        Get the symbols with open long positions and format them correctly.
        
        :param open_position_data: List of dictionaries containing open position data.
        :return: List of formatted symbols with open long positions.
        """
        long_symbols = []
        for position in open_position_data:
            if position['info']['side'].lower() == 'buy':
                symbol = position['symbol'].replace('/USDT:USDT', 'USDT')
                long_symbols.append(symbol)
        return long_symbols

    def get_open_symbols_short(self, open_position_data):
        """
        Get the symbols with open short positions and format them correctly.
        
        :param open_position_data: List of dictionaries containing open position data.
        :return: List of formatted symbols with open short positions.
        """
        short_symbols = []
        for position in open_position_data:
            if position['info']['side'].lower() == 'sell':
                symbol = position['symbol'].replace('/USDT:USDT', 'USDT')
                short_symbols.append(symbol)
        return short_symbols

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

    def fetch_historical_data(self, symbol, timeframe, limit=15):
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert columns to numeric
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric, errors='coerce')

        # Validate data
        if df[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
            logging.warning(f"Invalid data detected for {symbol} on timeframe {timeframe}. Data:\n{df}")
            # Handle invalid data here (e.g., skip the symbol, raise an error, etc.)

        return df

    def calculate_atr(self, df, period=14):
        # Drop rows with NaN values in 'high', 'low', or 'close' columns
        df = df.dropna(subset=['high', 'low', 'close'])

        # Check again if there are enough data points after dropping NaNs
        if len(df) < period:
            return None  # Not enough data points

        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.max([high_low, high_close, low_close], axis=0)

        # Calculate the ATR using np.nanmean to ignore any remaining NaNs
        atr = np.nanmean(tr[-period:])

        return atr if not np.isnan(atr) else None  # Return None if the result is NaN

    def initialize_trade_quantities(self, symbol, total_equity, best_ask_price, max_leverage):
        if symbol in self.initialized_symbols:
            return

        try:
            max_trade_qty = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
        except Exception as e:
            logging.info(f"Error calculating max trade quantity for {symbol}: {e}")
            return

        self.max_long_trade_qty_per_symbol[symbol] = max_trade_qty
        self.max_short_trade_qty_per_symbol[symbol] = max_trade_qty

        logging.info(f"For symbol {symbol} Calculated max_long_trade_qty: {max_trade_qty}, max_short_trade_qty: {max_trade_qty}")
        self.initialized_symbols.add(symbol)

    def get_all_moving_averages(self, symbol, max_retries=3, delay=5):
        with self.general_rate_limiter:
            for _ in range(max_retries):
                try:
                    m_moving_averages = self.manager.get_1m_moving_averages(symbol)
                    m5_moving_averages = self.manager.get_5m_moving_averages(symbol)

                    ma_6_high = m_moving_averages.get("MA_6_H")
                    ma_6_low = m_moving_averages.get("MA_6_L")
                    ma_3_low = m_moving_averages.get("MA_3_L")
                    ma_3_high = m_moving_averages.get("MA_3_H")
                    ma_1m_3_high = m_moving_averages.get("MA_3_H")
                    ma_5m_3_high = m5_moving_averages.get("MA_3_H")

                    # Check if the data is correct
                    if all(isinstance(value, (float, int, np.number)) for value in [ma_6_high, ma_6_low, ma_3_low, ma_3_high, ma_1m_3_high, ma_5m_3_high]):
                        self.last_known_mas[symbol] = {
                            "ma_6_high": ma_6_high,
                            "ma_6_low": ma_6_low,
                            "ma_3_low": ma_3_low,
                            "ma_3_high": ma_3_high,
                            "ma_1m_3_high": ma_1m_3_high,
                            "ma_5m_3_high": ma_5m_3_high,
                        }
                        return self.last_known_mas[symbol]

                    logging.warning(f"Invalid moving averages for {symbol}: {m_moving_averages}, {m5_moving_averages}. Retrying...")

                except Exception as e:
                    logging.error(f"Error fetching moving averages for {symbol}: {e}. Retrying...")

                # If the data is not correct, wait for a short delay
                time.sleep(delay)

            # If retries are exhausted, use the last known values
            if symbol in self.last_known_mas:
                logging.info(f"Using last known moving averages for {symbol}.")
                return self.last_known_mas[symbol]
            else:
                raise ValueError(f"Failed to fetch valid moving averages for {symbol} after multiple attempts and no fallback available.")


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
            logging.info(f"Failed to place market order: {e}")

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
            logging.info(f"Failed to place market close order: {e}")

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
            logging.info(f"Error fetching recent trades for {symbol}: {e}")
            return []
        
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
        for _ in range(max_retries):
            try:
                market_data = self.get_market_data_with_retry(symbol, max_retries, retry_delay)
                max_trade_qty = round(
                    (total_equity * wallet_exposure * max_leverage) / best_ask_price,
                    int(float(market_data["min_qty"]))
                )
                logging.info(f"Max trade qty for {symbol} calculated: {max_trade_qty}")
                return max_trade_qty
            except Exception as e:
                logging.info(f"An error occurred in calc_max_trade_qty: {e}. Retrying...")
                time.sleep(retry_delay)

        raise Exception("Failed to calculate maximum trade quantity after maximum retries.")


    # def calc_max_trade_qty(self, symbol, total_equity, best_ask_price, max_leverage, max_retries=5, retry_delay=5):
    #     wallet_exposure = self.config.wallet_exposure
    #     for i in range(max_retries):
    #         try:
    #             market_data = self.get_market_data_with_retry(symbol, max_retries = 5, retry_delay = 5)
    #             max_trade_qty = round(
    #                 (float(total_equity) * wallet_exposure / float(best_ask_price))
    #                 / (100 / max_leverage),
    #                 int(float(market_data["min_qty"])),
    #             )

    #             logging.info(f"Max trade qty for {symbol} calculated: {max_trade_qty} ")

    #             return max_trade_qty
    #         except TypeError as e:
    #             if total_equity is None:
    #                 print(f"Error: total_equity is None. Retrying in {retry_delay} seconds...")
    #             if best_ask_price is None:
    #                 print(f"Error: best_ask_price is None. Retrying in {retry_delay} seconds...")
    #         except Exception as e:
    #             print(f"An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...")
    #         time.sleep(retry_delay)

    #     raise Exception("Failed to calculate maximum trade quantity after maximum retries.")

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

    def calc_max_trade_qty_multi(self, symbol, total_equity, best_ask_price, max_leverage, max_retries=5, retry_delay=5):
        wallet_exposure = self.config.wallet_exposure
        for i in range(max_retries):
            try:
                market_data = self.exchange.get_market_data_bybit(symbol)
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

    def print_trade_quantities_once_bybit(self, symbol, total_equity, best_ask_price, max_leverage):
        # Fetch the best ask price
        order_book = self.exchange.get_orderbook(symbol)
        if 'asks' in order_book and order_book['asks']:
            best_ask_price = order_book['asks'][0][0]
        else:
            logging.warning(f"No ask orders available for {symbol}.")
            return

        # Ensure symbol is initialized
        if symbol not in self.initialized_symbols:
            max_trade_qty = self.calculate_max_trade_qty(symbol, total_equity, best_ask_price)
            self.max_trade_qty_per_symbol[symbol] = max_trade_qty
            self.initialized_symbols.add(symbol)

        if not self.printed_trade_quantities:
            print(f"Printing trade quantities for {symbol} at different leverage levels")
            for leverage in [0.001, 0.01, 0.1, 1, 2.5, 5]:
                # Temporarily set MAX_LEVERAGE to the current level for calculation
                original_max_leverage = max_leverage
                self.MAX_LEVERAGE = leverage
                dynamic_amount = self.calculate_dynamic_amount(symbol, total_equity, best_ask_price)
                print(f"Leverage {leverage}x: Trade Quantity = {dynamic_amount}")
                # Reset MAX_LEVERAGE to original
                self.MAX_LEVERAGE = original_max_leverage

            self.printed_trade_quantities = True

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

    def get_positions_bybit(self, symbol):
        position_data = self.exchange.get_positions_bybit(symbol)
        return position_data

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

    def check_short_long_conditions(self, best_bid_price, ma_3_high):
        should_short = best_bid_price > ma_3_high
        should_long = best_bid_price < ma_3_high
        return should_short, should_long

    def get_5m_averages(self, symbol):
        ma_values = self.manager.get_5m_moving_averages(symbol)
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

    def retry_api_call(self, function, *args, max_retries=100, base_delay=10, max_delay=60, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                with self.rate_limiter:
                    return function(*args, **kwargs)
            except ccxt.RateLimitExceeded as e:
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Rate limit exceeded: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            except Exception as e:
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Error occurred: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        raise Exception(f"Failed to execute the API function after {max_retries} retries.")

    # def retry_api_call(self, function, *args, max_retries=100, base_delay=10, max_delay=60, **kwargs):
    #     retries = 0
    #     while retries < max_retries:
    #         try:
    #             return function(*args, **kwargs)
    #         except Exception as e:  # Catch all exceptions
    #             retries += 1
    #             delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
    #             logging.info(f"Error occurred: {e}. Retrying in {delay:.2f} seconds...")
    #             time.sleep(delay)
    #     raise Exception(f"Failed to execute the API function after {max_retries} retries.")

    def can_trade_new_symbol(self, open_symbols: list, symbols_allowed: int, current_symbol: str) -> bool:
        """
        Checks if the bot can trade a given symbol.
        """
        unique_open_symbols = set(open_symbols)  # Convert to set to get unique symbols
        self.open_symbols_count = len(unique_open_symbols)  # Count unique symbols
        logging.info(f"Symbols allowed amount: {symbols_allowed}")
        logging.info(f"Open symbols count (unique): {self.open_symbols_count}")

        if symbols_allowed is None:
            logging.info(f"Symbols alloweed is none, defaulting to 10")
            symbols_allowed = 10  # Use a default value if symbols_allowed is not specified

        # If we haven't reached the symbol limit or the current symbol is already being traded, allow the trade
        if self.open_symbols_count < symbols_allowed or current_symbol in unique_open_symbols:
            logging.info(f"New symbol is allowed : Symbols allowed: {symbols_allowed} Open symbol count: {self.open_symbols_count}")
            return True
        else:
            return False
            
    # def can_trade_new_symbol(self, open_symbols: list, symbols_allowed: int, current_symbol: str) -> bool:
    #     """
    #     Checks if the bot can trade a given symbol.
    #     """
    #     unique_open_symbols = set(open_symbols)  # Convert to set to get unique symbols
    #     self.open_symbols_count = len(unique_open_symbols)  # Count unique symbols

    #     logging.info(f"Open symbols count (unique): {self.open_symbols_count}")

    #     if symbols_allowed is None:
    #         symbols_allowed = 10  # Use a default value if symbols_allowed is not specified

    #     # If the current symbol is already being traded, allow it
    #     if current_symbol in unique_open_symbols:
    #         return True

    #     # If we haven't reached the symbol limit, allow a new symbol to be traded
    #     if self.open_symbols_count < symbols_allowed:
    #         return True

    #     # If none of the above conditions are met, don't allow the new trade
    #     return False

    # Dashboard
    def update_shared_data(self, symbol_data: dict, open_position_data: dict, open_symbols_count: int):
        data_directory = "data"  # Define the data directory

        # Update and serialize symbol data
        with open(os.path.join(data_directory, "symbol_data.json"), "w") as f:
            json.dump(symbol_data, f)

        # Update and serialize open position data
        with open(os.path.join(data_directory, "open_positions_data.json"), "w") as f:
            json.dump(open_position_data, f)

        # Update and serialize count of open symbols
        with open(os.path.join(data_directory, "open_symbols_count.json"), "w") as f:
            json.dump({"count": open_symbols_count}, f)

    # def manage_liquidation_risk(self, long_pos_price, short_pos_price, long_liq_price, short_liq_price, symbol, amount):
    #     # Create some thresholds for when to act
    #     long_threshold = self.config.long_liq_pct
    #     short_threshold = self.config.short_liq_pct

    #     # Let's assume you have methods to get the best bid and ask prices
    #     best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
    #     best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]

    #     # Check if the long position is close to being liquidated
    #     if long_pos_price is not None and long_liq_price is not None:
    #         long_diff = abs(long_pos_price - long_liq_price) / long_pos_price
    #         if long_diff < long_threshold:
    #             # Place a post-only limit order to offset the risk
    #             self.postonly_limit_order_bybit(symbol, "buy", amount, best_bid_price, positionIdx=1, reduceOnly=False)
    #             logging.info(f"Placed a post-only limit order to offset long position risk on {symbol} at {best_bid_price}")

    #     # Check if the short position is close to being liquidated
    #     if short_pos_price is not None and short_liq_price is not None:
    #         short_diff = abs(short_pos_price - short_liq_price) / short_pos_price
    #         if short_diff < short_threshold:
    #             # Place a post-only limit order to offset the risk
    #             self.postonly_limit_order_bybit(symbol, "sell", amount, best_ask_price, positionIdx=2, reduceOnly=False)
    #             logging.info(f"Placed a post-only limit order to offset short position risk on {symbol} at {best_ask_price}")

    def get_active_order_count(self, symbol):
        try:
            active_orders = self.exchange.fetch_open_orders(symbol)
            return len(active_orders)
        except Exception as e:
            logging.warning(f"Could not fetch active orders for {symbol}: {e}")
            return 0

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
    
    def helperv2(self, symbol, short_dynamic_amount, long_dynamic_amount):
        if self.helper_active:
            # Fetch orderbook and positions
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = Decimal(orderbook['bids'][0][0])
            best_ask_price = Decimal(orderbook['asks'][0][0])

            open_position_data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)
            position_details = self.process_position_data(open_position_data)

            long_pos_qty = position_details.get(symbol, {}).get('long', {}).get('qty', 0)
            short_pos_qty = position_details.get(symbol, {}).get('short', {}).get('qty', 0)

            if short_pos_qty is None and long_pos_qty is None:
                logging.warning(f"Could not fetch position quantities for {symbol}. Skipping helper process.")
                return

            # Determine which position is larger
            larger_position = "long" if long_pos_qty > short_pos_qty else "short"

            # Adjust helper_wall_size based on the larger position
            base_helper_wall_size = self.helper_wall_size
            adjusted_helper_wall_size = base_helper_wall_size + 2

            # Initialize variables
            helper_orders = []

            # Dynamic safety_margin and base_gap based on asset's price
            safety_margin = best_ask_price * Decimal('0.0060')  # 0.10% of current price
            base_gap = best_ask_price * Decimal('0.0060')  # 0.10% of current price

            for i in range(adjusted_helper_wall_size):
                gap = base_gap + Decimal(i) * Decimal('0.002')  # Increasing gap for each subsequent order
                unique_id = int(time.time() * 1000) + i  # Generate a unique identifier

                if larger_position == "long":
                    # Calculate long helper price based on best ask price
                    helper_price_long = best_ask_price + gap + safety_margin
                    helper_price_long = helper_price_long.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    helper_order_long = self.exchange.create_tagged_limit_order_bybit(
                        symbol,
                        "sell",
                        long_dynamic_amount * 1.5,
                        helper_price_long,
                        positionIdx=2,
                        postOnly=True,
                        params={"orderLinkId": f"helperOrder_{symbol}_long_{unique_id}"}
                    )
                    helper_orders.append(helper_order_long)

                if larger_position == "short":
                    # Calculate short helper price based on best bid price
                    helper_price_short = best_bid_price - gap - safety_margin
                    helper_price_short = helper_price_short.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                    helper_order_short = self.exchange.create_tagged_limit_order_bybit(
                        symbol,
                        "buy",
                        short_dynamic_amount * 1.5,
                        helper_price_short,
                        positionIdx=1,
                        postOnly=True,
                        params={"orderLinkId": f"helperOrder_{symbol}_short_{unique_id}"}
                    )
                    helper_orders.append(helper_order_short)

            # Sleep for the helper duration and then cancel all placed orders
            time.sleep(self.helper_duration)

            # Cancel orders and handle errors
            for order in helper_orders:
                if 'id' in order:
                    logging.info(f"Helper order for {symbol}: {order}")
                    self.exchange.cancel_order_by_id(order['id'], symbol)
                else:
                    logging.warning(f"Could not place helper order for {symbol}: {order.get('error', 'Unknown error')}")

            # Deactivate helper for the next cycle
            self.helper_active = False

    def pm(self, symbol):
        # Fetch orderbook
        orderbook = self.exchange.get_orderbook(symbol)
        best_bid_price = Decimal(orderbook['bids'][0][0])
        best_ask_price = Decimal(orderbook['asks'][0][0])

        # Calculate target price movement
        target_price_increase = best_ask_price * Decimal('0.005')  # Target 0.5% price increase
        target_price = best_ask_price + target_price_increase

        # Initialize variables
        paint_orders = []

        # Start painting the market
        start_time = time.time()
        while time.time() - start_time < self.paint_duration:
            # Place buy orders to push the price up
            buy_price = best_ask_price + Decimal('0.001')  # Slightly above the best ask price
            buy_price = buy_price.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
            buy_order = self.exchange.create_market_order(symbol, "buy", self.trade_amount)
            paint_orders.append(buy_order)

            # Update best ask price
            orderbook = self.exchange.get_orderbook(symbol)
            best_ask_price = Decimal(orderbook['asks'][0][0])

            # Check if the target price has been reached
            if best_ask_price >= target_price:
                break

            # Short sleep to simulate real market activity
            time.sleep(1)

        # Log the painting activity
        logging.info(f"Market painting for {symbol} complete. Target price reached: {best_ask_price}")

        # Deactivate painter for the next cycle
        self.paint_active = False

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

    def improved_m_orders(self, symbol, short_amount, long_amount):
        # Retrieve order book
        order_book = self.exchange.get_orderbook(symbol)
        top_asks = order_book['asks'][:10]
        top_bids = order_book['bids'][:10]

        # Extract and update best ask/bid prices
        if 'asks' in order_book and len(order_book['asks']) > 0:
            best_ask_price = order_book['asks'][0][0]
        else:
            best_ask_price = self.last_known_ask.get(symbol)

        if 'bids' in order_book and len(order_book['bids']) > 0:
            best_bid_price = order_book['bids'][0][0]
        else:
            best_bid_price = self.last_known_bid.get(symbol)

        placed_orders = []  # Initialize the list to keep track of placed orders

        # Define buffer percentages
        BUFFER_PERCENTAGE = Decimal('0.0040')  # Use as a percentage, same as in helperv2

        # Determine the larger position
        larger_position = "long" if long_amount > short_amount else "short"

        # Calculate dynamic safety_margin and base_gap based on asset's price
        best_ask_price = Decimal(top_asks[0][0])
        best_bid_price = Decimal(top_bids[0][0])
        safety_margin = best_ask_price * BUFFER_PERCENTAGE
        base_gap = best_bid_price * BUFFER_PERCENTAGE

        # Place QS orders
        if random.randint(1, 10) > 8:
            for i in range(5):
                try:
                    gap = base_gap + Decimal(i) * BUFFER_PERCENTAGE  # Incremental gap for each subsequent order
                    price_adjustment = safety_margin + gap  # Combine safety margin and gap for price adjustment

                    order_price = best_bid_price - price_adjustment if larger_position == "long" else best_ask_price + price_adjustment
                    order_price = order_price.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)  # Adjust the price format if necessary

                    order_amount = long_amount if larger_position == "long" else short_amount
                    order_type = "buy" if larger_position == "long" else "sell"
                    order = self.limit_order_bybit(symbol, order_type, order_amount, order_price, postonly=True)
                    if order is not None:
                        placed_orders.append(order)
                except Exception as e:
                    logging.info(f"Error placing QS order: {e}")

        # Place L orders
        if random.randint(1, 10) > 7:
            for i in range(3):
                try:
                    gap = base_gap + Decimal(i) * BUFFER_PERCENTAGE  # Incremental gap for each subsequent order
                    price_adjustment = safety_margin + gap  # Combine safety margin and gap for price adjustment

                    order_price = best_bid_price - price_adjustment if larger_position == "long" else best_ask_price + price_adjustment
                    order_price = order_price.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)  # Adjust the price format if necessary

                    order_amount = long_amount * Decimal('1.5') if larger_position == "long" else short_amount * Decimal('1.5')
                    order_type = "buy" if larger_position == "long" else "sell"
                    order = self.limit_order_bybit(symbol, order_type, order_amount, order_price, reduceOnly=False)
                    if order is not None:
                        placed_orders.append(order)
                except Exception as e:
                    logging.info(f"Error placing L order: {e}")

        try:
            for _ in range(50):
                logging.info(f"QS for {symbol}")
                # Define the dynamic safety margin and base gap
                safety_margin = best_ask_price * Decimal('0.0040') if larger_position == "long" else best_bid_price * Decimal('0.0040')
                base_gap = safety_margin  # For simplicity, we're using the same value for base gap and safety margin here

                # Adjust the price based on the current market state
                gap = base_gap + Decimal('0.002')  # Incremental gap for each subsequent order, can be adjusted as needed
                stuffing_price_adjustment = gap + safety_margin
                stuffing_price = best_bid_price - stuffing_price_adjustment if larger_position == "long" else best_ask_price + stuffing_price_adjustment
                stuffing_price = stuffing_price.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)

                order_amount = long_amount if larger_position == "long" else short_amount
                # Include positionIdx in the order placement
                order = self.limit_order_bybit(symbol, "buy" if larger_position == "long" else "sell", order_amount, stuffing_price, positionIdx=1 if larger_position == "long" else 2, reduceOnly=False)
                self.exchange.cancel_order_by_id(order['order_id'], symbol)
        except Exception as e:
            logging.info(f"Error in quote stuffing: {e}")

        # Cancel orders
        for order in placed_orders:
            if order and 'id' in order:
                self.exchange.cancel_order_by_id(order['id'], symbol)

        return long_amount if larger_position == "long" else short_amount

    def e_m_d(self, symbol):
        while True:  # Continuous operation
            order_book = self.exchange.get_orderbook(symbol)
            top_asks = order_book['asks'][:10]
            top_bids = order_book['bids'][:10]

            # Generate extreme price adjustments
            price_adjustment = random.uniform(0.10, 0.50)  # 10% to 50% price adjustment
            amount_adjustment = random.uniform(100, 1000)  # Random order size

            # Place orders at extreme prices
            for _ in range(100):  # High frequency of orders
                try:
                    order_price = (top_bids[0][0] * (1 + price_adjustment)) if random.choice([True, False]) else (top_asks[0][0] * (1 - price_adjustment))
                    side = "buy" if order_price < top_bids[0][0] else "sell"
                    order = self.limit_order_bybit(symbol, side, amount_adjustment, order_price, positionIdx=1 if side == "buy" else 2, reduceOnly=False)
                    if order and 'id' in order:
                        self.exchange.cancel_order_by_id(order['id'], symbol)  # Immediate cancellation
                except Exception as e:
                    logging.info(f"Error in extreme market distortion: {e}")

            time.sleep(0.01)  # Minimal delay before next cycle

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
                    logging.info(f"Error placing order: {e}")

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
                    logging.info(f"Error placing order: {e}")

        # Cancel orders
        for order in placed_orders:
            if order and 'id' in order:
                self.exchange.cancel_order_by_id(order['id'], symbol)

        return amount

    def play_the_spread_entry_and_tp(self, symbol, open_orders, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):
        analyzer = OrderBookAnalyzer(self.exchange, symbol)

        imbalance = analyzer.get_order_book_imbalance()

        best_ask_price = analyzer.get_best_prices()[1]
        best_bid_price = analyzer.get_best_prices()[0]

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
        sell_walls = analyzer.identify_walls(order_book, "sell")
        buy_walls = analyzer.identify_walls(order_book, "buy")

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
            
    def set_spread_take_profits(self, symbol, open_orders, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):

        order_book = self.exchange.get_orderbook(symbol)
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

    def auto_hedge_orders_bybit_atr(self, symbol, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, best_ask_price, best_bid_price, hedge_ratio, atr, min_order_size):
        atr_multiplier = 1

        # Check and calculate dynamic thresholds based on ATR and the multiplier
        dynamic_threshold_long = (atr * atr_multiplier) / long_pos_price if long_pos_price and long_pos_price != 0 else float('inf')
        dynamic_threshold_short = (atr * atr_multiplier) / short_pos_price if short_pos_price and short_pos_price != 0 else float('inf')


        # # Check and calculate dynamic thresholds based on ATR and the multiplier
        # dynamic_threshold_long = (atr * atr_multiplier) / long_pos_price if long_pos_price != 0 else float('inf')
        # dynamic_threshold_short = (atr * atr_multiplier) / short_pos_price if short_pos_price != 0 else float('inf')

        # Auto-hedging logic for long position
        if long_pos_qty > 0:
            price_diff_percentage_long = abs(best_ask_price - long_pos_price) / long_pos_price if long_pos_price != 0 else float('inf')
            current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0

            if current_hedge_ratio_long < hedge_ratio:
                if price_diff_percentage_long >= dynamic_threshold_long:
                    additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                    if additional_hedge_needed_long > min_order_size:
                        order_response = self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)
                        logging.info(f"Auto-hedge long order placed for {symbol}: {order_response}")
                        time.sleep(5)

        # Auto-hedging logic for short position
        if short_pos_qty > 0:
            price_diff_percentage_short = abs(best_bid_price - short_pos_price) / short_pos_price if short_pos_price != 0 else float('inf')
            current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0

            if current_hedge_ratio_short < hedge_ratio:
                if price_diff_percentage_short >= dynamic_threshold_short:
                    additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                    if additional_hedge_needed_short > min_order_size:
                        order_response = self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)
                        logging.info(f"Auto-hedge short order placed for {symbol}: {order_response}")
                        time.sleep(5)

    def calculate_dynamic_hedge_threshold(self, symbol, long_pos_price, short_pos_price):
        if long_pos_price and short_pos_price:
            return abs(long_pos_price - short_pos_price) / min(long_pos_price, short_pos_price)
        else:
            return self.default_hedge_price_difference_threshold  # fallback to a default threshold

    def auto_hedge_orders_bybit(self, symbol, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, best_ask_price, best_bid_price, hedge_ratio, price_difference_threshold, min_order_size):
        # Auto-hedging logic for long position
        if long_pos_qty > 0:
            price_diff_percentage_long = abs(best_ask_price - long_pos_price) / long_pos_price
            current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0

            if price_diff_percentage_long >= price_difference_threshold and current_hedge_ratio_long < hedge_ratio:
                additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                if additional_hedge_needed_long > min_order_size:
                    order_response = self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"order_response: {order_response}")
                    logging.info(f"Auto-hedge long order placed for {symbol}: {order_response}")
                    time.sleep(5)
        # Auto-hedging logic for short position
        if short_pos_qty > 0:
            price_diff_percentage_short = abs(best_bid_price - short_pos_price) / short_pos_price
            current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0

            if price_diff_percentage_short >= price_difference_threshold and current_hedge_ratio_short < hedge_ratio:
                additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                if additional_hedge_needed_short > min_order_size:
                    order_response = self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"order_response: {order_response}")
                    logging.info(f"Auto-hedge short order placed for {symbol}: {order_response}")
                    time.sleep(5)

    def bybit_1m_walls_topbottom(self, open_orders: list, symbol: str, trend: str, hma_trend: str, eri_trend: str, top_signal: str, bottom_signal: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)

            # Extract and update best ask/bid prices
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else self.last_known_ask.get(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else self.last_known_bid.get(symbol)

            # Define variables for trend alignment
            trend_aligned_long = (eri_trend == "bullish" or trend.lower() == "long")
            trend_aligned_short = (eri_trend == "bearish" or trend.lower() == "short")

            # Long Entry Conditions
            if one_minute_volume > min_vol and ((should_long or should_add_to_long) and bottom_signal == 'True' and trend_aligned_long):
                if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            # Short Entry Conditions
            if one_minute_volume > min_vol and ((should_short or should_add_to_short) and top_signal == 'True' and trend_aligned_short):
                if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            # Order Book Wall Logic for Long Entries
            if largest_bid_wall and not self.entry_order_exists(open_orders, "buy") and ((should_long or should_add_to_long) and bottom_signal == 'True' and trend_aligned_long):
                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

            # Order Book Wall Logic for Short Entries
            if largest_ask_wall and not self.entry_order_exists(open_orders, "sell") and ((should_short or should_add_to_short) and top_signal == 'True' and trend_aligned_short):
                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            time.sleep(5)


    def bybit_1m_mfi_eri_walls_topbottom(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, top_signal: str, bottom_signal: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)

            # Extract and update best ask/bid prices
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else self.last_known_ask.get(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else self.last_known_bid.get(symbol)

            # Define variables for trend alignment
            trend_aligned_long = (eri_trend == "bullish" or trend.lower() == "long")
            trend_aligned_short = (eri_trend == "bearish" or trend.lower() == "short")

            # Define variables for MFI signals
            mfi_signal_long = mfi.lower() == "long"
            mfi_signal_short = mfi.lower() == "short"

            # Long Entry Conditions
            if one_minute_volume > min_vol and ((should_long or should_add_to_long) and bottom_signal == 'True' and trend_aligned_long) and mfi_signal_long:
                if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            # Short Entry Conditions
            if one_minute_volume > min_vol and ((should_short or should_add_to_short) and top_signal == 'True' and trend_aligned_short) and mfi_signal_short:
                if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            # Order Book Wall Logic for Long Entries
            if largest_bid_wall and not self.entry_order_exists(open_orders, "buy"):
                price_approaching_bid_wall = self.is_price_approaching_wall(current_price, largest_bid_wall[0], 'bid')

                if price_approaching_bid_wall and ((should_long or should_add_to_long) and bottom_signal == 'True' and trend_aligned_long):
                    logging.info(f"Approaching significant bid wall for long entry in {symbol}.")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

            # Order Book Wall Logic for Short Entries
            if largest_ask_wall and not self.entry_order_exists(open_orders, "sell"):
                price_approaching_ask_wall = self.is_price_approaching_wall(current_price, largest_ask_wall[0], 'ask')

                if price_approaching_ask_wall and ((should_short or should_add_to_short) and top_signal == 'True' and trend_aligned_short):
                    logging.info(f"Approaching significant ask wall for short entry in {symbol}.")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            time.sleep(5)

    def calculate_order_size_imbalance(self, order_book):
        total_bids = sum([amount for price, amount in order_book['bids'][:10]])
        total_asks = sum([amount for price, amount in order_book['asks'][:10]])
        imbalance = total_bids / total_asks if total_asks > 0 else 1
        return imbalance

    def adjust_dynamic_amounts_based_on_imbalance(self, imbalance, base_amount):
        if imbalance > 1.5:
            return base_amount * 1.5
        elif imbalance < 0.5:
            return base_amount * 1.5
        return base_amount

    def aggressive_entry_based_on_walls(self, current_price, largest_bid_wall, largest_ask_wall, should_long, should_short):
        if largest_bid_wall and should_long and current_price - largest_bid_wall[0] < current_price * 0.005:
            return True
        if largest_ask_wall and should_short and largest_ask_wall[0] - current_price < current_price * 0.005:
            return True
        return False

    def adjust_leverage_based_on_market_confidence(self, symbol, market_confidence):
        if market_confidence > 0.8:
            self.exchange.set_leverage_bybit(10, symbol)
        elif market_confidence < 0.3:
            self.exchange.set_leverage_bybit(5, symbol)

    def bybit_1m_mfi_eri_walls_imbalance(self, open_orders, symbol, mfi, eri_trend, one_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_long, should_short, should_add_to_long, should_add_to_short, fivemin_top_signal, fivemin_bottom_signal):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_significant_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            order_book = self.exchange.get_orderbook(symbol)

            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
            else:
                best_ask_price = self.last_known_ask.get(symbol)

            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
            else:
                best_bid_price = self.last_known_bid.get(symbol)

            eri_trend_aligned_long = eri_trend == "bullish"
            eri_trend_aligned_short = eri_trend == "bearish"

            mfi_signal_long = mfi.lower() == "long"
            mfi_signal_short = mfi.lower() == "short"
            mfi_signal_neutral = mfi.lower() == "neutral"

            imbalance = self.calculate_order_size_imbalance(order_book)
            long_dynamic_amount = self.adjust_dynamic_amounts_based_on_imbalance(imbalance, long_dynamic_amount)
            short_dynamic_amount = self.adjust_dynamic_amounts_based_on_imbalance(imbalance, short_dynamic_amount)

            aggressive_entry_signal = self.aggressive_entry_based_on_walls(current_price, largest_bid_wall, largest_ask_wall, should_long, should_short)

            if aggressive_entry_signal and one_minute_volume > min_vol:
                # Long Entry for Trend and MFI Signal
                if (should_long or should_add_to_long) and current_price >= qfl_base and eri_trend_aligned_long and mfi_signal_long:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(5)

                # Short Entry for Trend and MFI Signal
                if (should_short or should_add_to_short) and current_price <= qfl_ceiling and eri_trend_aligned_short and mfi_signal_short:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(5)

                # Order Book Wall Long Entry Logic
                if largest_bid_wall and not self.entry_order_exists(open_orders, "buy"):
                    price_approaching_bid_wall = self.is_price_approaching_wall(current_price, largest_bid_wall[0], 'bid')

                    # Check if the bottom signal is present for long entries
                    if price_approaching_bid_wall and (should_long or should_add_to_long) and eri_trend_aligned_long and mfi_signal_neutral and fivemin_bottom_signal:
                        logging.info(f"Price approaching significant buy wall and bottom signal detected for {symbol}. Placing long trade.")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)
                        time.sleep(5)

                # Order Book Wall Short Entry Logic
                if largest_ask_wall and not self.entry_order_exists(open_orders, "sell"):
                    price_approaching_ask_wall = self.is_price_approaching_wall(current_price, largest_ask_wall[0], 'ask')

                    # Check if the top signal is present for short entries
                    if price_approaching_ask_wall and (should_short or should_add_to_short) and eri_trend_aligned_short and mfi_signal_neutral and fivemin_top_signal:
                        logging.info(f"Price approaching significant sell wall and top signal detected for {symbol}. Placing short trade.")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)
                        time.sleep(5)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

    def get_mfirsi_ema_secondary_ema_bollinger(self, symbol: str, limit: int = 100, lookback: int = 1, ema_period: int = 5, secondary_ema_period: int = 3) -> str:
        # Fetch OHLCV data
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate MFI and RSI
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Calculate EMAs for MFI and RSI
        df['mfi_ema'] = df['mfi'].ewm(span=ema_period, adjust=False).mean()
        df['rsi_ema'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()

        # Calculate secondary EMAs for MFI and RSI
        df['mfi_ema_secondary'] = df['mfi'].ewm(span=secondary_ema_period, adjust=False).mean()
        df['rsi_ema_secondary'] = df['rsi'].ewm(span=secondary_ema_period, adjust=False).mean()

        # Calculate MACD
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_bbm'] = bollinger.bollinger_mavg()
        df['bb_bbh'] = bollinger.bollinger_hband()
        df['bb_bbl'] = bollinger.bollinger_lband()

        # Determine conditions using EMAs, MACD, and Bollinger Bands
        df['buy_condition'] = (
            (df['mfi_ema'] < 30) &
            (df['rsi_ema'] < 40) &
            (df['mfi_ema_secondary'] < df['mfi_ema']) &
            (df['rsi_ema_secondary'] < df['rsi_ema']) &
            (df['open'] < df['close']) &
            (df['macd'] > df['macd_signal']) &  # MACD condition
            (df['close'] < df['bb_bbl'])        # Bollinger Bands condition
        ).astype(int)
        
        df['sell_condition'] = (
            (df['mfi_ema'] > 70) &
            (df['rsi_ema'] > 60) &
            (df['mfi_ema_secondary'] > df['mfi_ema']) &
            (df['rsi_ema_secondary'] > df['rsi_ema']) &
            (df['open'] > df['close']) &
            (df['macd'] < df['macd_signal']) &  # MACD condition
            (df['close'] > df['bb_bbh'])        # Bollinger Bands condition
        ).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].sum() > 0:
            return 'long'
        elif recent_conditions['sell_condition'].sum() > 0:
            return 'short'
        else:
            return 'neutral'
            
    def get_mfirsi_ema_secondary_ema_highfreq(self, symbol: str, limit: int = 100, lookback: int = 1, ema_period: int = 5, secondary_ema_period: int = 3) -> str:
        # Fetch OHLCV data
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate MFI and RSI
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Calculate EMAs for MFI and RSI
        df['mfi_ema'] = df['mfi'].ewm(span=ema_period, adjust=False).mean()
        df['rsi_ema'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()

        # Calculate secondary EMAs for MFI and RSI
        df['mfi_ema_secondary'] = df['mfi'].ewm(span=secondary_ema_period, adjust=False).mean()
        df['rsi_ema_secondary'] = df['rsi'].ewm(span=secondary_ema_period, adjust=False).mean()

        # Determine conditions using EMAs and secondary EMAs
        df['buy_condition'] = (
            (df['mfi_ema'] < 35) &
            (df['rsi_ema'] < 45) &
            (df['mfi_ema_secondary'] < df['mfi_ema']) &
            (df['rsi_ema_secondary'] < df['rsi_ema'])
        ).astype(int)
        df['sell_condition'] = (
            (df['mfi_ema'] > 65) &
            (df['rsi_ema'] > 55) &
            (df['mfi_ema_secondary'] > df['mfi_ema']) &
            (df['rsi_ema_secondary'] > df['rsi_ema'])
        ).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].sum() > 0:
            return 'long'
        elif recent_conditions['sell_condition'].sum() > 0:
            return 'short'
        else:
            return 'neutral'
        
    def get_mfirsi_ema_secondary_ema(self, symbol: str, limit: int = 100, lookback: int = 1, ema_period: int = 5, secondary_ema_period: int = 3) -> str:
        # Fetch OHLCV data
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate MFI and RSI
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Calculate EMAs for MFI and RSI
        df['mfi_ema'] = df['mfi'].ewm(span=ema_period, adjust=False).mean()
        df['rsi_ema'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()

        # Calculate secondary EMAs for MFI and RSI
        df['mfi_ema_secondary'] = df['mfi'].ewm(span=secondary_ema_period, adjust=False).mean()
        df['rsi_ema_secondary'] = df['rsi'].ewm(span=secondary_ema_period, adjust=False).mean()

        # Determine conditions using EMAs and secondary EMAs
        df['buy_condition'] = (
            (df['mfi_ema'] < 30) &
            (df['rsi_ema'] < 40) &
            (df['mfi_ema_secondary'] < df['mfi_ema']) &
            (df['rsi_ema_secondary'] < df['rsi_ema']) &
            (df['open'] < df['close'])
        ).astype(int)
        df['sell_condition'] = (
            (df['mfi_ema'] > 70) &
            (df['rsi_ema'] > 60) &
            (df['mfi_ema_secondary'] > df['mfi_ema']) &
            (df['rsi_ema_secondary'] > df['rsi_ema']) &
            (df['open'] > df['close'])
        ).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].sum() > 0:
            return 'long'
        elif recent_conditions['sell_condition'].sum() > 0:
            return 'short'
        else:
            return 'neutral'

    def get_mfirsi_ema_secondary_ema_l(self, symbol: str, limit: int = 100, lookback: int = 6, ema_period: int = 6, secondary_ema_period: int = 4) -> str:
        # Fetch OHLCV data
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate MFI and RSI
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Calculate EMAs for MFI and RSI
        df['mfi_ema'] = df['mfi'].ewm(span=ema_period, adjust=False).mean()
        df['rsi_ema'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()

        # Calculate secondary EMAs for MFI and RSI
        df['mfi_ema_secondary'] = df['mfi'].ewm(span=secondary_ema_period, adjust=False).mean()
        df['rsi_ema_secondary'] = df['rsi'].ewm(span=secondary_ema_period, adjust=False).mean()

        # Determine conditions using EMAs and secondary EMAs
        df['buy_condition'] = (
            (df['mfi_ema'] < 33) &
            (df['rsi_ema'] < 43) &
            (df['mfi_ema_secondary'] < df['mfi_ema']) &
            (df['rsi_ema_secondary'] < df['rsi_ema']) &
            (df['open'] < df['close'])
        ).astype(int)
        df['sell_condition'] = (
            (df['mfi_ema'] > 67) &
            (df['rsi_ema'] > 57) &
            (df['mfi_ema_secondary'] > df['mfi_ema']) &
            (df['rsi_ema_secondary'] > df['rsi_ema']) &
            (df['open'] > df['close'])
        ).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].sum() > 0:
            return 'long'
        elif recent_conditions['sell_condition'].sum() > 0:
            return 'short'
        else:
            return 'neutral'

    def get_mfirsi_ema(self, symbol: str, limit: int = 100, lookback: int = 5, ema_period: int = 5) -> str:
        # Fetch OHLCV data
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate MFI and RSI
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)

        # Calculate EMAs for MFI and RSI
        df['mfi_ema'] = df['mfi'].ewm(span=ema_period, adjust=False).mean()
        df['rsi_ema'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()

        # Determine conditions using EMAs
        df['buy_condition'] = ((df['mfi_ema'] < 30) & (df['rsi_ema'] < 40) & (df['open'] < df['close'])).astype(int)
        df['sell_condition'] = ((df['mfi_ema'] > 80) & (df['rsi_ema'] > 70) & (df['open'] > df['close'])).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].any():
            return 'long'
        elif recent_conditions['sell_condition'].any():
            return 'short'
        else:
            return 'neutral'

    def get_mfirsi_volatility_ema(self, symbol: str, limit: int = 100, lookback: int = 5, ema_period: int = 5) -> str:
        # Fetch OHLCV data using CCXT
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate volatility (standard deviation of close prices)
        df['volatility'] = df['close'].rolling(window=14).std()

        # Determine MFI and RSI windows based on volatility
        high_volatility_threshold = 0.05
        mfi_window = 10 if df['volatility'].iloc[-1] > high_volatility_threshold else 20
        rsi_window = 10 if df['volatility'].iloc[-1] > high_volatility_threshold else 20

        # Calculate MFI and RSI with adaptive windows
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=mfi_window, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_window)

        # Calculate EMAs for MFI and RSI
        df['mfi_ema'] = df['mfi'].ewm(span=ema_period, adjust=False).mean()
        df['rsi_ema'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()

        # Determine conditions using EMAs
        df['buy_condition'] = (df['mfi_ema'].diff() > 0) & (df['rsi_ema'].diff() > 0) & (df['open'] < df['close'])
        df['sell_condition'] = (df['mfi_ema'].diff() < 0) & (df['rsi_ema'].diff() < 0) & (df['open'] > df['close'])

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].any():
            return 'long'
        elif recent_conditions['sell_condition'].any():
            return 'short'
        else:
            return 'neutral'


    def get_mfi_atr(self, symbol: str, limit: int = 100, lookback: int = 5) -> str:
        # Fetch 1-minute OHLCV data
        ohlcv_data_min = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df_min = pd.DataFrame(ohlcv_data_min, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Fetch 1-hour OHLCV data for ATR
        ohlcv_data_hour = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1h', limit=14)  # Last 14 hours
        df_hour = pd.DataFrame(ohlcv_data_hour, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate True Range and ATR on 1-hour data
        df_hour['tr'] = np.maximum((df_hour['high'] - df_hour['low']),
                                np.maximum(abs(df_hour['high'] - df_hour['close'].shift(1)),
                                            abs(df_hour['low'] - df_hour['close'].shift(1))))
        df_hour['atr'] = df_hour['tr'].rolling(window=14).mean()

        # Get the latest ATR value from 1-hour data
        latest_atr_value = df_hour['atr'].iloc[-1]

        # Determine Volatility Threshold
        volatility_threshold = df_hour['atr'].quantile(0.75)

        # Determine MFI and RSI windows based on ATR-based volatility
        mfi_window = 14 if latest_atr_value > volatility_threshold else 14
        rsi_window = 14 if latest_atr_value > volatility_threshold else 14

        # Calculate MFI and RSI with adaptive windows on 1-minute data
        df_min['mfi'] = ta.volume.MFIIndicator(high=df_min['high'], low=df_min['low'], close=df_min['close'], volume=df_min['volume'], window=mfi_window, fillna=False).money_flow_index()
        df_min['rsi'] = ta.momentum.rsi(df_min['close'], window=rsi_window)
        df_min['open_less_close'] = (df_min['open'] < df_min['close']).astype(int)

        # Adaptive thresholds based on volatility
        mfi_buy_threshold = 25 if latest_atr_value > volatility_threshold else 30
        mfi_sell_threshold = 85 if latest_atr_value > volatility_threshold else 80
        rsi_buy_threshold = 35 if latest_atr_value > volatility_threshold else 40
        rsi_sell_threshold = 75 if latest_atr_value > volatility_threshold else 70

        # Calculate conditions with adaptive thresholds
        df_min['buy_condition'] = ((df_min['mfi'] < mfi_buy_threshold) & (df_min['rsi'] < rsi_buy_threshold) & (df_min['open_less_close'] == 1)).astype(int)
        df_min['sell_condition'] = ((df_min['mfi'] > mfi_sell_threshold) & (df_min['rsi'] > rsi_sell_threshold) & (df_min['open_less_close'] == 0)).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df_min.iloc[-lookback:]
        if recent_conditions['buy_condition'].any():
            return 'long'
        elif recent_conditions['sell_condition'].any():
            return 'short'
        else:
            return 'neutral'

    def get_mfirsi(self, symbol: str, limit: int = 100, lookback: int = 5) -> str:
        # Fetch OHLCV data using CCXT
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate volatility (standard deviation of close prices)
        df['volatility'] = df['close'].rolling(window=14).std()

        # Determine MFI and RSI windows based on volatility
        high_volatility_threshold = 0.05
        mfi_window = 10 if df['volatility'].iloc[-1] > high_volatility_threshold else 20
        rsi_window = 10 if df['volatility'].iloc[-1] > high_volatility_threshold else 20

        # Calculate MFI and RSI with adaptive windows
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=mfi_window, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_window)
        df['open_less_close'] = (df['open'] < df['close']).astype(int)

        # Adaptive thresholds based on volatility
        mfi_buy_threshold = 25 if df['volatility'].iloc[-1] > high_volatility_threshold else 30
        mfi_sell_threshold = 85 if df['volatility'].iloc[-1] > high_volatility_threshold else 80
        rsi_buy_threshold = 35 if df['volatility'].iloc[-1] > high_volatility_threshold else 40
        rsi_sell_threshold = 75 if df['volatility'].iloc[-1] > high_volatility_threshold else 70

        # Calculate conditions with adaptive thresholds
        df['buy_condition'] = ((df['mfi'] < mfi_buy_threshold) & (df['rsi'] < rsi_buy_threshold) & (df['open_less_close'] == 1)).astype(int)
        df['sell_condition'] = ((df['mfi'] > mfi_sell_threshold) & (df['rsi'] > rsi_sell_threshold) & (df['open_less_close'] == 0)).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].any():
            return 'long'
        elif recent_conditions['sell_condition'].any():
            return 'short'
        else:
            return 'neutral'


    def get_mfirsi_v1(self, symbol: str, limit: int = 100, lookback: int = 5) -> str:
        # Fetch OHLCV data using CCXT
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate MFI and RSI
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['open_less_close'] = (df['open'] < df['close']).astype(int)

        # Calculate conditions
        df['buy_condition'] = ((df['mfi'] < 30) & (df['rsi'] < 40) & (df['open_less_close'] == 1)).astype(int)
        df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 70) & (df['open_less_close'] == 0)).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].any():
            return 'long'
        elif recent_conditions['sell_condition'].any():
            return 'short'
        else:
            return 'neutral'

    def calculate_volatility_metric(self, symbol, timeframe='1h', period=14, limit=15):
        """
        Calculate a volatility metric based on the Average True Range (ATR).
        :param symbol: Symbol for which to calculate the metric.
        :param timeframe: Timeframe for the historical data.
        :param period: Period for the ATR calculation.
        :param limit: Number of data points to fetch for the calculation.
        :return: A volatility metric.
        """
        # Fetch historical data
        df = self.fetch_historical_data(symbol, timeframe, limit)

        # Calculate ATR
        atr = self.calculate_atr(df, period)

        # Normalize or adjust the ATR to create a volatility metric
        # This could be as simple as using the ATR directly, or scaling it in some way
        volatility_metric = atr / df['close'].iloc[-1]  # Example: normalized by the latest closing price

        return volatility_metric
    
    def liq_stop_loss_logic(self, long_pos_qty, long_pos_price, long_liquidation_price, short_pos_qty, short_pos_price, short_liquidation_price, liq_stoploss_enabled, symbol, liq_price_stop_pct):
        if liq_stoploss_enabled:
            try:
                current_price = self.exchange.get_current_price(symbol)

                # Stop loss logic for long positions
                if long_pos_qty > 0 and long_liquidation_price:
                    # Convert to float if it's not None or empty string
                    long_liquidation_price = float(long_liquidation_price) if long_liquidation_price else None

                    if long_liquidation_price:
                        long_stop_loss_price = self.calculate_long_stop_loss_based_on_liq_price(
                            long_pos_price, long_liquidation_price, liq_price_stop_pct)
                        if long_stop_loss_price and current_price <= long_stop_loss_price:
                            # Place stop loss order for long position
                            logging.info(f"Placing long stop loss order for {symbol} at {long_stop_loss_price}")
                            self.postonly_limit_order_bybit_nolimit(symbol, "sell", long_pos_qty, long_stop_loss_price, positionIdx=1, reduceOnly=True)

                # Stop loss logic for short positions
                if short_pos_qty > 0 and short_liquidation_price:
                    # Convert to float if it's not None or empty string
                    short_liquidation_price = float(short_liquidation_price) if short_liquidation_price else None

                    if short_liquidation_price:
                        short_stop_loss_price = self.calculate_short_stop_loss_based_on_liq_price(
                            short_pos_price, short_liquidation_price, liq_price_stop_pct)
                        if short_stop_loss_price and current_price >= short_stop_loss_price:
                            # Place stop loss order for short position
                            logging.info(f"Placing short stop loss order for {symbol} at {short_stop_loss_price}")
                            self.postonly_limit_order_bybit_nolimit(symbol, "buy", short_pos_qty, short_stop_loss_price, positionIdx=2, reduceOnly=True)
            except Exception as e:
                logging.info(f"Exception caught in liquidation stop loss logic for {symbol}: {e}")


    def stop_loss_logic(self, long_pos_qty, long_pos_price, short_pos_qty, short_pos_price, stoploss_enabled, symbol, stoploss_upnl_pct):
        if stoploss_enabled:
            try:
                # Initial stop loss calculation
                initial_short_stop_loss = self.calculate_quickscalp_short_stop_loss(short_pos_price, symbol, stoploss_upnl_pct) if short_pos_price else None
                initial_long_stop_loss = self.calculate_quickscalp_long_stop_loss(long_pos_price, symbol, stoploss_upnl_pct) if long_pos_price else None

                current_price = self.exchange.get_current_price(symbol)
                order_book = self.exchange.get_orderbook(symbol)
                current_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else None
                current_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else None

                # Calculate and set stop loss for long positions
                if long_pos_qty > 0 and long_pos_price and initial_long_stop_loss:
                    threshold_for_long = long_pos_price - (long_pos_price - initial_long_stop_loss) * 0.1
                    if current_price <= threshold_for_long:
                        adjusted_long_stop_loss = initial_long_stop_loss if current_price > initial_long_stop_loss else current_bid_price
                        logging.info(f"Setting long stop loss for {symbol} at {adjusted_long_stop_loss}")
                        self.postonly_limit_order_bybit_nolimit(symbol, "sell", long_pos_qty, adjusted_long_stop_loss, positionIdx=1, reduceOnly=True)

                # Calculate and set stop loss for short positions
                if short_pos_qty > 0 and short_pos_price and initial_short_stop_loss:
                    threshold_for_short = short_pos_price + (initial_short_stop_loss - short_pos_price) * 0.1
                    if current_price >= threshold_for_short:
                        adjusted_short_stop_loss = initial_short_stop_loss if current_price < initial_short_stop_loss else current_ask_price
                        logging.info(f"Setting short stop loss for {symbol} at {adjusted_short_stop_loss}")
                        self.postonly_limit_order_bybit_nolimit(symbol, "buy", short_pos_qty, adjusted_short_stop_loss, positionIdx=2, reduceOnly=True)
            except Exception as e:
                logging.info(f"Exception caught in stop loss functionality for {symbol}: {e}")
                
    def auto_reduce_logic_grid_hardened(self, symbol, min_qty, long_pos_price, short_pos_price, 
                                            long_pos_qty, short_pos_qty, long_upnl, short_upnl,
                                            auto_reduce_enabled, total_equity, current_market_price,
                                            long_dynamic_amount, short_dynamic_amount, auto_reduce_start_pct,
                                            min_buffer_percentage_ar, max_buffer_percentage_ar,
                                            upnl_auto_reduce_threshold_long, upnl_auto_reduce_threshold_short, current_leverage):
            logging.info(f"Starting auto-reduce logic for symbol: {symbol}")
            if not auto_reduce_enabled:
                logging.info(f"Auto-reduce is disabled for {symbol}.")
                return

            try:
                long_upnl_pct_equity = (long_upnl / total_equity) * 100
                short_upnl_pct_equity = (short_upnl / total_equity) * 100

                logging.info(f"{symbol} Long uPNL % of Equity: {long_upnl_pct_equity:.2f}, Short uPNL % of Equity: {short_upnl_pct_equity:.2f}")

                long_loss_exceeded = long_pos_price is not None and long_pos_price != 0 and current_market_price < long_pos_price * (1 - auto_reduce_start_pct)
                short_loss_exceeded = short_pos_price is not None and short_pos_price != 0 and current_market_price > short_pos_price * (1 + auto_reduce_start_pct)

                logging.info(f"{symbol} Price Loss Exceeded - Long: {long_loss_exceeded}, Short: {short_loss_exceeded}")

                logging.info(f"Loss thresholds - Long: {upnl_auto_reduce_threshold_long}%, Short: {upnl_auto_reduce_threshold_short}%")

                upnl_long_exceeded = long_upnl_pct_equity < -upnl_auto_reduce_threshold_long
                upnl_short_exceeded = short_upnl_pct_equity < -upnl_auto_reduce_threshold_short

                logging.info(f"{symbol} UPnL Exceeded - Long: {upnl_long_exceeded}, Short: {upnl_short_exceeded}")

                trigger_auto_reduce_long = long_pos_qty > 0 and long_loss_exceeded and upnl_long_exceeded
                trigger_auto_reduce_short = short_pos_qty > 0 and short_loss_exceeded and upnl_short_exceeded

                logging.info(f"{symbol} Trigger Auto-Reduce - Long: {trigger_auto_reduce_long}, Short: {trigger_auto_reduce_short}")

                if trigger_auto_reduce_long:
                    logging.info(f"Executing auto-reduce for long position in {symbol}.")
                    self.auto_reduce_active_long[symbol] = True
                    self.execute_grid_auto_reduce_hardened('long', symbol, long_pos_qty, long_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty, min_buffer_percentage_ar, max_buffer_percentage_ar)
                else:
                    logging.info(f"No auto-reduce executed for long position in {symbol}.")
                    if symbol in self.auto_reduce_active_long:
                        del self.auto_reduce_active_long[symbol]

                if trigger_auto_reduce_short:
                    logging.info(f"Executing auto-reduce for short position in {symbol}.")
                    self.auto_reduce_active_short[symbol] = True
                    self.execute_grid_auto_reduce_hardened('short', symbol, short_pos_qty, short_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty, min_buffer_percentage_ar, max_buffer_percentage_ar)
                else:
                    logging.info(f"No auto-reduce executed for short position in {symbol}.")
                    if symbol in self.auto_reduce_active_short:
                        del self.auto_reduce_active_short[symbol]

            except Exception as e:
                logging.info(f"Error in auto-reduce logic for {symbol}: {e}")
                raise

    def auto_reduce_logic_grid(self, symbol, min_qty, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                                auto_reduce_enabled, total_equity, available_equity, current_market_price,
                                long_dynamic_amount, short_dynamic_amount, auto_reduce_start_pct,
                                max_pos_balance_pct, upnl_threshold_pct, shared_symbols_data):
        logging.info(f"Starting auto-reduce logic for symbol: {symbol}")
        if not auto_reduce_enabled:
            logging.info(f"Auto-reduce is disabled for {symbol}.")
            return

        try:
            #total_upnl = sum(data['long_upnl'] + data['short_upnl'] for data in shared_symbols_data.values())
            # Possibly has issues w/ calculation above

            # Testing fix

            total_upnl = sum(
                (data.get('long_upnl', 0) or 0) + (data.get('short_upnl', 0) or 0)
                for data in shared_symbols_data.values()
            )
            
            logging.info(f"Total uPNL : {total_upnl}")

            # Correct calculation for total UPnL percentage
            total_upnl_pct = total_upnl / total_equity if total_equity else 0

            # Correcting the UPnL threshold exceeded logic to compare absolute UPnL against the threshold value of total equity
            upnl_threshold_exceeded = abs(total_upnl) > (total_equity * upnl_threshold_pct)

            symbol_data = shared_symbols_data.get(symbol, {})
            long_position_value_pct = (symbol_data.get('long_pos_qty', 0) * current_market_price / total_equity) if total_equity else 0
            short_position_value_pct = (symbol_data.get('short_pos_qty', 0) * current_market_price / total_equity) if total_equity else 0

            long_loss_exceeded = long_pos_price is not None and current_market_price < long_pos_price * (1 - auto_reduce_start_pct)
            short_loss_exceeded = short_pos_price is not None and current_market_price > short_pos_price * (1 + auto_reduce_start_pct)

            trigger_auto_reduce_long = long_pos_qty > 0 and long_loss_exceeded and long_position_value_pct > max_pos_balance_pct and upnl_threshold_exceeded
            trigger_auto_reduce_short = short_pos_qty > 0 and short_loss_exceeded and short_position_value_pct > max_pos_balance_pct and upnl_threshold_exceeded

            logging.info(f"Total UPnL for all symbols: {total_upnl}, which is {total_upnl_pct * 100}% of total equity")
            logging.info(f"{symbol} Long Position Value %: {long_position_value_pct * 100}, Short Position Value %: {short_position_value_pct * 100}")
            logging.info(f"{symbol} Long Loss Exceeded: {long_loss_exceeded}, Short Loss Exceeded: {short_loss_exceeded}, UPnL Threshold Exceeded: {upnl_threshold_exceeded}")
            logging.info(f"{symbol} Trigger Auto-Reduce Long: {trigger_auto_reduce_long}, Trigger Auto-Reduce Short: {trigger_auto_reduce_short}")

            if trigger_auto_reduce_long:
                logging.info(f"Executing auto-reduce for long position in {symbol}.")
                self.execute_grid_auto_reduce('long', symbol, long_pos_qty, long_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty)
            else:
                logging.info(f"No auto-reduce executed for long position in {symbol}.")

            if trigger_auto_reduce_short:
                logging.info(f"Executing auto-reduce for short position in {symbol}.")
                self.execute_grid_auto_reduce('short', symbol, short_pos_qty, short_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty)
            else:
                logging.info(f"No auto-reduce executed for short position in {symbol}.")

        except Exception as e:
            logging.info(f"Error in auto-reduce logic for {symbol}: {e}")

    def execute_auto_reduce(self, position_type, symbol, pos_qty, dynamic_amount, market_price, total_equity, long_pos_price, short_pos_price, min_qty):
        # Fetch precision for the symbol
        amount_precision, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
        price_precision_level = -int(math.log10(price_precision))
        qty_precision_level = -int(math.log10(amount_precision))

        # Convert market_price to Decimal for consistent arithmetic operations
        market_price = Decimal(str(market_price))

        max_levels, price_interval = self.calculate_dynamic_auto_reduce_levels(symbol, pos_qty, market_price, total_equity, long_pos_price, short_pos_price)
        for i in range(1, max_levels + 1):
            # Calculate step price based on position type
            if position_type == 'long':
                step_price = market_price + (price_interval * i)
                # Ensure step price is greater than the market price for long positions
                if step_price <= market_price:
                    logging.warning(f"Skipping auto-reduce long order for {symbol} at {step_price} as it is not greater than the market price.")
                    continue
            else:  # position_type == 'short'
                step_price = market_price - (price_interval * i)
                # Ensure step price is less than the market price for short positions
                if step_price >= market_price:
                    logging.warning(f"Skipping auto-reduce short order for {symbol} at {step_price} as it is not less than the market price.")
                    continue

            # Round the step price to the correct precision
            step_price = round(step_price, price_precision_level)

            # Ensure dynamic_amount is at least the minimum required quantity and rounded to the correct precision
            adjusted_dynamic_amount = max(dynamic_amount, min_qty)
            adjusted_dynamic_amount = round(adjusted_dynamic_amount, qty_precision_level)

            # Attempt to place the auto-reduce order
            try:
                if position_type == 'long':
                    order_id = self.auto_reduce_long(symbol, adjusted_dynamic_amount, float(step_price))
                elif position_type == 'short':
                    order_id = self.auto_reduce_short(symbol, adjusted_dynamic_amount, float(step_price))

                # Initialize the symbol key if it doesn't exist
                if symbol not in self.auto_reduce_orders:
                    self.auto_reduce_orders[symbol] = []

                if order_id:
                    self.auto_reduce_orders[symbol].append(order_id)
                    logging.info(f"{symbol} {position_type.capitalize()} Auto-Reduce Order Placed at {step_price} with amount {adjusted_dynamic_amount}")
                else:
                    logging.warning(f"{symbol} {position_type.capitalize()} Auto-Reduce Order Not Filled Immediately at {step_price} with amount {adjusted_dynamic_amount}")
            except Exception as e:
                logging.info(f"Error in executing auto-reduce {position_type} order for {symbol}: {e}")
                logging.info("Traceback:", traceback.format_exc())

    def cancel_all_auto_reduce_orders_bybit(self, symbol: str) -> None:
        try:
            if symbol in self.auto_reduce_orders:
                for order_id in self.auto_reduce_orders[symbol]:
                    try:
                        self.exchange.cancel_order(order_id, symbol)
                        logging.info(f"Cancelling auto-reduce order: {order_id}")
                    except Exception as e:
                        logging.warning(f"An error occurred while cancelling auto-reduce order {order_id}: {e}")
                self.auto_reduce_orders[symbol].clear()  # Clear the list after cancellation
            else:
                logging.info(f"No auto-reduce orders found for {symbol}")

        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_all_auto_reduce_orders_bybit(): {e}")

    def auto_reduce_logic_simple(self, symbol, min_qty, long_pos_price, short_pos_price, long_pos_qty, short_pos_qty,
                                auto_reduce_enabled, total_equity, available_equity, current_market_price,
                                long_dynamic_amount, short_dynamic_amount, auto_reduce_start_pct,
                                max_pos_balance_pct, upnl_threshold_pct, shared_symbols_data):
        logging.info(f"Starting auto-reduce logic for symbol: {symbol}")
        if not auto_reduce_enabled:
            logging.info(f"Auto-reduce is disabled for {symbol}.")
            return

        try:
            #total_upnl = sum(data['long_upnl'] + data['short_upnl'] for data in shared_symbols_data.values())
            # Possibly has issues w/ calculation above

            # Testing fix

            total_upnl = sum(
                (data.get('long_upnl', 0) or 0) + (data.get('short_upnl', 0) or 0)
                for data in shared_symbols_data.values()
            )
            
            logging.info(f"Total uPNL : {total_upnl}")

            # Correct calculation for total UPnL percentage
            total_upnl_pct = total_upnl / total_equity if total_equity else 0

            # Correcting the UPnL threshold exceeded logic to compare absolute UPnL against the threshold value of total equity
            upnl_threshold_exceeded = abs(total_upnl) > (total_equity * upnl_threshold_pct)

            symbol_data = shared_symbols_data.get(symbol, {})
            long_position_value_pct = (symbol_data.get('long_pos_qty', 0) * current_market_price / total_equity) if total_equity else 0
            short_position_value_pct = (symbol_data.get('short_pos_qty', 0) * current_market_price / total_equity) if total_equity else 0

            long_loss_exceeded = long_pos_price is not None and current_market_price < long_pos_price * (1 - auto_reduce_start_pct)
            short_loss_exceeded = short_pos_price is not None and current_market_price > short_pos_price * (1 + auto_reduce_start_pct)

            trigger_auto_reduce_long = long_pos_qty > 0 and long_loss_exceeded and long_position_value_pct > max_pos_balance_pct and upnl_threshold_exceeded
            trigger_auto_reduce_short = short_pos_qty > 0 and short_loss_exceeded and short_position_value_pct > max_pos_balance_pct and upnl_threshold_exceeded

            logging.info(f"Total UPnL for all symbols: {total_upnl}, which is {total_upnl_pct * 100}% of total equity")
            logging.info(f"{symbol} Long Position Value %: {long_position_value_pct * 100}, Short Position Value %: {short_position_value_pct * 100}")
            logging.info(f"{symbol} Long Loss Exceeded: {long_loss_exceeded}, Short Loss Exceeded: {short_loss_exceeded}, UPnL Threshold Exceeded: {upnl_threshold_exceeded}")
            logging.info(f"{symbol} Trigger Auto-Reduce Long: {trigger_auto_reduce_long}, Trigger Auto-Reduce Short: {trigger_auto_reduce_short}")

            if trigger_auto_reduce_long:
                logging.info(f"Executing auto-reduce for long position in {symbol}.")
                self.execute_auto_reduce('long', symbol, long_pos_qty, long_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty)
            else:
                logging.info(f"No auto-reduce executed for long position in {symbol}.")

            if trigger_auto_reduce_short:
                logging.info(f"Executing auto-reduce for short position in {symbol}.")
                self.execute_auto_reduce('short', symbol, short_pos_qty, short_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty)
            else:
                logging.info(f"No auto-reduce executed for short position in {symbol}.")

        except Exception as e:
            logging.info(f"Error in auto-reduce logic for {symbol}: {e}")

    def failsafe_method(self, symbol, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price,
                        long_upnl, short_upnl, total_equity, current_price,
                        failsafe_enabled, long_failsafe_upnl_pct, short_failsafe_upnl_pct, failsafe_start_pct):
        if not failsafe_enabled:
            return

        try:

            logging.info(f"Failsafe method called for {symbol}")
            logging.info(f"Long position quantity: {long_pos_qty}")
            logging.info(f"Short position quantity: {short_pos_qty}")
            logging.info(f"Long position price: {long_pos_price}")
            logging.info(f"Short position price: {short_pos_price}")
            logging.info(f"Long UPNL: {long_upnl}")
            logging.info(f"Short UPNL: {short_upnl}")
            logging.info(f"Total equity: {total_equity}")
            logging.info(f"Current price: {current_price}")
            logging.info(f"Long failsafe UPNL %: {long_failsafe_upnl_pct}")
            logging.info(f"Short failsafe UPNL %: {short_failsafe_upnl_pct}")
            logging.info(f"Failsafe start %: {failsafe_start_pct}")
            
            long_upnl_pct_equity = (long_upnl / total_equity) * 100
            short_upnl_pct_equity = (short_upnl / total_equity) * 100

            logging.info(f"FAILSAFE: {symbol} Long UPNL % of Equity: {long_upnl_pct_equity:.2f}, Short UPNL % of Equity: {short_upnl_pct_equity:.2f}")

            long_failsafe_triggered = (
                long_pos_qty > 0
                and current_price < long_pos_price * (1 - failsafe_start_pct)
                and long_upnl_pct_equity < long_failsafe_upnl_pct
            )

            short_failsafe_triggered = (
                short_pos_qty > 0
                and current_price > short_pos_price * (1 + failsafe_start_pct)
                and short_upnl_pct_equity < -short_failsafe_upnl_pct
            )


            if long_failsafe_triggered:
                logging.info(f"Triggering failsafe for long position on {symbol}. Cutting position in half.")
                half_long_pos_qty = long_pos_qty // 2
                logging.info(f"Half position {symbol} long: {half_long_pos_qty}")
                #self.execute_failsafe_order(symbol, "long", half_long_pos_qty, current_price)

            if short_failsafe_triggered:
                logging.info(f"Triggering failsafe for short position on {symbol}. Cutting position in half.")
                half_short_pos_qty = short_pos_qty // 2
                logging.info(f"Half position {symbol} short: {half_short_pos_qty}")
                #self.execute_failsafe_order(symbol, "short", half_short_pos_qty, current_price)

        except Exception as e:
            logging.error(f"Error in failsafe_method for {symbol}: {e}")
            raise

    def failsafe_method_leveraged(self, symbol, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price,
                        long_upnl, short_upnl, total_equity, current_price,
                        failsafe_enabled, long_failsafe_upnl_pct, short_failsafe_upnl_pct, failsafe_start_pct):
        if not failsafe_enabled:
            return

        try:
            logging.info(f"Failsafe method called for {symbol}")
            logging.info(f"Long position quantity: {long_pos_qty}")
            logging.info(f"Short position quantity: {short_pos_qty}")
            logging.info(f"Long position price: {long_pos_price}")
            logging.info(f"Short position price: {short_pos_price}")
            logging.info(f"Long UPNL: {long_upnl}")
            logging.info(f"Short UPNL: {short_upnl}")
            logging.info(f"Total equity: {total_equity}")
            logging.info(f"Current price: {current_price}")
            logging.info(f"Long failsafe UPNL %: {long_failsafe_upnl_pct}")
            logging.info(f"Short failsafe UPNL %: {short_failsafe_upnl_pct}")
            logging.info(f"Failsafe start %: {failsafe_start_pct}")

            # Calculate UPNL as percentage of the total equity
            long_upnl_pct_equity = abs((long_upnl / total_equity) * 100) if total_equity else 0
            short_upnl_pct_equity = abs((short_upnl / total_equity) * 100) if total_equity else 0

            logging.info(f"FAILSAFE: {symbol} Long UPNL % of Total Equity: {long_upnl_pct_equity:.2f}, Short UPNL % of Total Equity: {short_upnl_pct_equity:.2f}")

            # Log the conditions for triggering the failsafe
            if long_pos_qty > 0:
                logging.info(f"Checking long failsafe for {symbol}: Current price {current_price}, Failsafe start price {long_pos_price * (1 - failsafe_start_pct)}, Long UPNL {long_upnl}, Long UPNL % {long_upnl_pct_equity}")
                long_failsafe_price = long_pos_price * (1 - (long_failsafe_upnl_pct / 100))
                logging.info(f"Long position would trigger failsafe at price: {long_failsafe_price}")
                if current_price < long_pos_price * (1 - failsafe_start_pct):
                    logging.info(f"Long position price is below failsafe start threshold for {symbol}")
                if long_upnl < -0.01:
                    logging.info(f"Long UPNL is significant for {symbol}")
                if long_upnl_pct_equity > long_failsafe_upnl_pct:
                    logging.info(f"Long UPNL % exceeds failsafe threshold for {symbol}")
                else:
                    logging.info(f"Long UPNL % does not exceed failsafe threshold for {symbol}")

            if short_pos_qty > 0:
                logging.info(f"Checking short failsafe for {symbol}: Current price {current_price}, Failsafe start price {short_pos_price * (1 + failsafe_start_pct)}, Short UPNL {short_upnl}, Short UPNL % {short_upnl_pct_equity}")
                short_failsafe_price = short_pos_price * (1 + (short_failsafe_upnl_pct / 100))
                logging.info(f"Short position would trigger failsafe at price: {short_failsafe_price}")
                if current_price > short_pos_price * (1 + failsafe_start_pct):
                    logging.info(f"Short position price is above failsafe start threshold for {symbol}")
                if short_upnl < -0.01:
                    logging.info(f"Short UPNL is significant for {symbol}")
                if short_upnl_pct_equity > short_failsafe_upnl_pct:
                    logging.info(f"Short UPNL % exceeds failsafe threshold for {symbol}")
                else:
                    logging.info(f"Short UPNL % does not exceed failsafe threshold for {symbol}")

            # Adjust failsafe trigger conditions to avoid triggering for very small UPNL values and to use absolute percentage values
            long_failsafe_triggered = (
                long_pos_qty > 0
                and current_price < long_pos_price * (1 - failsafe_start_pct)
                and long_upnl < -0.01  # Ensure there is a significant loss
                and long_upnl_pct_equity > long_failsafe_upnl_pct
            )

            short_failsafe_triggered = (
                short_pos_qty > 0
                and current_price > short_pos_price * (1 + failsafe_start_pct)
                and short_upnl < -0.01  # Ensure there is a significant loss
                and short_upnl_pct_equity > short_failsafe_upnl_pct
            )

            if long_failsafe_triggered:
                logging.info(f"Triggering failsafe for long position on {symbol}. Cutting position in half.")
                half_long_pos_qty = long_pos_qty // 2
                logging.info(f"Half position {symbol} long: {half_long_pos_qty}")
                self.execute_failsafe_order(symbol, "long", half_long_pos_qty, current_price)

            if short_failsafe_triggered:
                logging.info(f"Triggering failsafe for short position on {symbol}. Cutting position in half.")
                half_short_pos_qty = short_pos_qty // 2
                logging.info(f"Half position {symbol} short: {half_short_pos_qty}")
                self.execute_failsafe_order(symbol, "short", half_short_pos_qty, current_price)

        except Exception as e:
            logging.error(f"Error in failsafe_method for {symbol}: {e}")
            raise

    def get_user_defined_leverage(self, symbol, side):
        if side == 'long':
            leverage = self.user_defined_leverage_long if self.user_defined_leverage_long not in (0, None) else self.exchange.get_current_max_leverage_bybit(symbol)
            logging.info(f"User defined leverage long: {leverage}")
            return leverage
        elif side == 'short':
            leverage = self.user_defined_leverage_short if self.user_defined_leverage_short not in (0, None) else self.exchange.get_current_max_leverage_bybit(symbol)
            logging.info(f"User defined leverage short: {leverage}")
            return leverage
        return 1

    def execute_failsafe_order(self, symbol, position_type, pos_qty, market_price):
        amount_precision, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
        price_precision_level = -int(math.log10(price_precision))
        qty_precision_level = -int(math.log10(amount_precision))

        order_book = self.exchange.get_orderbook(symbol)
        best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, market_price)
        best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, market_price)

        order_price = best_bid_price if position_type == 'long' else best_ask_price
        order_price = round(order_price, price_precision_level)
        adjusted_pos_qty = round(pos_qty, qty_precision_level)

        positionIdx = 1 if position_type == 'long' else 2

        logging.info(f"Attempting to place failsafe order: Symbol={symbol}, Type={'sell' if position_type == 'long' else 'buy'}, Qty={adjusted_pos_qty}, Price={order_price}, PositionIdx={positionIdx}")

        try:
            # Place the reduce-only order
            logging.info(f"Would have placed failsafe order for {symbol}")
            # order_result = self.postonly_limit_order_bybit_nolimit(symbol, 'sell' if position type == 'long' else 'buy', adjusted_pos_qty, order_price, positionIdx, reduceOnly=True)
            # logging.info(f"Failsafe order placed successfully for {symbol}: {order_result}")
        except Exception as e:
            logging.info(f"Failed to place failsafe order for {symbol}: {e}")
            raise

    def fetch_profits(self, symbol):
        try:
            # Fetch trade history for the symbol to calculate profits
            trades = self.exchange.fetch_my_trades(symbol)
            logging.info(f"Trades from fetch_my_trades: {trades}")
            total_profit = sum(float(trade['info']['profit']) for trade in trades if 'profit' in trade['info'])
            logging.info(f"Total profit for {symbol}: {total_profit}")
            return total_profit
        except Exception as e:
            logging.error(f"Error fetching profits for {symbol}: {e}")
            return 0

    def create_reduce_order(self, symbol, position_type, pos_qty, market_price):
        try:
            amount_precision, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
            price_precision_level = -int(math.log10(price_precision))
            qty_precision_level = -int(math.log10(amount_precision))

            order_book = self.exchange.fetch_order_book(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else market_price
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else market_price

            order_price = best_bid_price if position_type == 'long' else best_ask_price
            order_price = round(order_price, price_precision_level)
            adjusted_pos_qty = round(pos_qty, qty_precision_level)

            positionIdx = 1 if position_type == 'long' else 2

            logging.info(f"Placing reduce-only order: Symbol={symbol}, Type={'sell' if position_type == 'long' else 'buy'}, Qty={adjusted_pos_qty}, Price={order_price}, PositionIdx={positionIdx}")

            # Place the reduce-only order
            #order_result = self.exchange.create_order(symbol, 'limit', 'sell' if position_type == 'long' else 'buy', adjusted_pos_qty, order_price, {'reduceOnly': True, 'positionIdx': positionIdx})
            #logging.info(f"Reduce-only order placed successfully for {symbol}: {order_result}")
            logging.info(f"This is where the auto reduce order would place for {symbol}")
        except Exception as e:
            logging.error(f"Failed to place reduce-only order for {symbol}: {e}")

    def autoreduce_method(self, symbol, auto_reduce_enabled, min_profit_pct, auto_reduce_cooldown_start_pct, upnl_auto_reduce_threshold_long, upnl_auto_reduce_threshold_short):
        if not auto_reduce_enabled:
            return

        try:
            logging.info(f"Autoreduction method called for {symbol}")

            # Fetch total profits for the symbol
            total_profit = self.fetch_profits(symbol)

            # Fetch current positions
            positions = self.exchange.fetch_positions(symbol)
            current_price = self.exchange.fetch_ticker(symbol)['last']
            long_pos_qty = positions['long']['contracts']
            short_pos_qty = positions['short']['contracts']
            long_pos_price = positions['long']['entryPrice']
            short_pos_price = positions['short']['entryPrice']

            # Fetch total equity
            total_equity = self.exchange.fetch_balance()['total']['USDT']
            profit_pct_equity = (total_profit / total_equity) * 100 if total_equity else 0

            # Calculate UPNL as percentage of the total equity
            long_upnl = (current_price - long_pos_price) * long_pos_qty
            short_upnl = (short_pos_price - current_price) * short_pos_qty
            long_upnl_pct_equity = abs((long_upnl / total_equity) * 100) if total_equity else 0
            short_upnl_pct_equity = abs((short_upnl / total_equity) * 100) if total_equity else 0

            logging.info(f"Profit % of Total Equity for {symbol}: {profit_pct_equity:.2f}")
            logging.info(f"FAILSAFE: {symbol} Long UPNL % of Total Equity: {long_upnl_pct_equity:.2f}, Short UPNL % of Total Equity: {short_upnl_pct_equity:.2f}")

            # Log the conditions for triggering the autoreduce
            if long_pos_qty > 0:
                logging.info(f"Checking long autoreduce for {symbol}: Current price {current_price}, Cooldown start price {long_pos_price * (1 - auto_reduce_cooldown_start_pct)}, Long UPNL {long_upnl}, Long UPNL % {long_upnl_pct_equity}")
                long_autoreduce_price = long_pos_price * (1 - (upnl_auto_reduce_threshold_long / 100))
                logging.info(f"Long position would trigger autoreduce at price: {long_autoreduce_price}")
                if current_price < long_pos_price * (1 - auto_reduce_cooldown_start_pct) and long_upnl_pct_equity > upnl_auto_reduce_threshold_long:
                    logging.info(f"Long UPNL % exceeds autoreduce threshold for {symbol}")
                    self.create_reduce_order(symbol, 'long', long_pos_qty // 2, current_price)

            if short_pos_qty > 0:
                logging.info(f"Checking short autoreduce for {symbol}: Current price {current_price}, Cooldown start price {short_pos_price * (1 + auto_reduce_cooldown_start_pct)}, Short UPNL {short_upnl}, Short UPNL % {short_upnl_pct_equity}")
                short_autoreduce_price = short_pos_price * (1 + (upnl_auto_reduce_threshold_short / 100))
                logging.info(f"Short position would trigger autoreduce at price: {short_autoreduce_price}")
                if current_price > short_pos_price * (1 + auto_reduce_cooldown_start_pct) and short_upnl_pct_equity > upnl_auto_reduce_threshold_short:
                    logging.info(f"Short UPNL % exceeds autoreduce threshold for {symbol}")
                    self.create_reduce_order(symbol, 'short', short_pos_qty // 2, current_price)

            if profit_pct_equity > min_profit_pct:
                logging.info(f"Profit percentage exceeds minimum threshold for {symbol}, initiating reduce-only orders")

                if long_pos_qty > 0:
                    logging.info(f"Creating reduce order for long position of {symbol}")
                    self.create_reduce_order(symbol, 'long', long_pos_qty // 2, current_price)

                if short_pos_qty > 0:
                    logging.info(f"Creating reduce order for short position of {symbol}")
                    self.create_reduce_order(symbol, 'short', short_pos_qty // 2, current_price)
            else:
                logging.info(f"Profit percentage does not exceed minimum threshold for {symbol}, no action taken")

        except Exception as e:
            logging.error(f"Error in autoreduction method for {symbol}: {e}")



    def calculate_dynamic_cooldown(self, current_price, entry_price, start_pct):
        trigger_price_long = entry_price * (1 - start_pct)
        trigger_price_short = entry_price * (1 + start_pct)
        distance_to_trigger_long = abs(current_price - trigger_price_long) / entry_price
        distance_to_trigger_short = abs(current_price - trigger_price_short) / entry_price
        distance_to_trigger = min(distance_to_trigger_long, distance_to_trigger_short)

        base_cooldown = 150  # Base cooldown of 5 minutes in seconds
        dynamic_cooldown = int(base_cooldown + (1 - distance_to_trigger) * 300)  # Scale up to 10 minutes
        logging.info(f"base cooldown: {base_cooldown}")
        logging.info(f"dynamic cooldown: {dynamic_cooldown}")
        return max(base_cooldown, dynamic_cooldown)

    def auto_reduce_logic_grid_hardened_cooldown(self, symbol, min_qty, long_pos_price, short_pos_price,
                                                long_pos_qty, short_pos_qty, long_upnl, short_upnl,
                                                auto_reduce_cooldown_enabled, total_equity, current_market_price,
                                                long_dynamic_amount, short_dynamic_amount, auto_reduce_cooldown_start_pct,
                                                min_buffer_percentage_ar, max_buffer_percentage_ar,
                                                upnl_auto_reduce_threshold_long, upnl_auto_reduce_threshold_short, current_leverage):
        logging.info(f"Starting auto-reduce logic for symbol: {symbol}")
        if not auto_reduce_cooldown_enabled:
            logging.info(f"Auto-reduce is disabled for {symbol}.")
            return

        key_long = f"{symbol}_long"
        key_short = f"{symbol}_short"
        current_time = time.time()

        try:
            long_upnl_pct_equity = (long_upnl / total_equity) * 100
            short_upnl_pct_equity = (short_upnl / total_equity) * 100

            logging.info(f"{symbol} Long uPNL % of Equity: {long_upnl_pct_equity:.2f}, Short uPNL % of Equity: {short_upnl_pct_equity:.2f}")

            long_loss_exceeded = long_pos_price is not None and long_pos_price != 0 and current_market_price < long_pos_price * (1 - auto_reduce_cooldown_start_pct)
            short_loss_exceeded = short_pos_price is not None and short_pos_price != 0 and current_market_price > short_pos_price * (1 + auto_reduce_cooldown_start_pct)

            logging.info(f"{symbol} Price Loss Exceeded - Long: {long_loss_exceeded}, Short: {short_loss_exceeded}")

            logging.info(f"Loss thresholds - Long: {upnl_auto_reduce_threshold_long}%, Short: {upnl_auto_reduce_threshold_short}%")

            upnl_long_exceeded = long_upnl_pct_equity < -upnl_auto_reduce_threshold_long
            upnl_short_exceeded = short_upnl_pct_equity < -upnl_auto_reduce_threshold_short

            logging.info(f"{symbol} UPnL Exceeded - Long: {upnl_long_exceeded}, Short: {upnl_short_exceeded}")

            # Calculate dynamic cooldown period only if there is a position and the position price is not zero
            cooldown_long = self.calculate_dynamic_cooldown(current_market_price, long_pos_price, auto_reduce_cooldown_start_pct) if long_pos_qty > 0 and long_pos_price > 0 else 1800
            cooldown_short = self.calculate_dynamic_cooldown(current_market_price, short_pos_price, auto_reduce_cooldown_start_pct) if short_pos_qty > 0 and short_pos_price > 0 else 1800

            logging.info(f"{symbol} Cooldown Long: {cooldown_long}, Cooldown Short: {cooldown_short}")
            logging.info(f"{symbol} Last Auto-Reduce Time Long: {self.last_auto_reduce_time.get(key_long, 0)}, Short: {self.last_auto_reduce_time.get(key_short, 0)}")
            logging.info(f"{symbol} Current Time: {current_time}")

            currently_auto_reducing_long = long_pos_qty > 0 and long_loss_exceeded and upnl_long_exceeded
            currently_auto_reducing_short = short_pos_qty > 0 and short_loss_exceeded and upnl_short_exceeded

            trigger_auto_reduce_long = currently_auto_reducing_long and (current_time - self.last_auto_reduce_time.get(key_long, 0) > cooldown_long)
            trigger_auto_reduce_short = currently_auto_reducing_short and (current_time - self.last_auto_reduce_time.get(key_short, 0) > cooldown_short)

            logging.info(f"{symbol} Trigger Auto-Reduce - Long: {trigger_auto_reduce_long}, Short: {trigger_auto_reduce_short}")

            if trigger_auto_reduce_long:
                logging.info(f"Executing auto-reduce for long position in {symbol}.")
                self.auto_reduce_active_long[symbol] = True
                self.execute_grid_auto_reduce_hardened('long', symbol, long_pos_qty, long_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty, min_buffer_percentage_ar, max_buffer_percentage_ar)
                self.last_auto_reduce_time[key_long] = current_time
            else:
                if currently_auto_reducing_long:
                    if current_time - self.last_auto_reduce_time.get(key_long, 0) <= cooldown_long:
                        logging.info(f"{symbol} Long position is still in the cooldown period. Time remaining: {cooldown_long - (current_time - self.last_auto_reduce_time.get(key_long, 0)):.2f} seconds.")
                    self.auto_reduce_active_long[symbol] = True
                else:
                    logging.info(f"No auto-reduce executed for long position in {symbol} because:")
                    if not long_loss_exceeded:
                        logging.info(f" - The current market price has not dropped below the threshold: {current_market_price} >= {long_pos_price * (1 - auto_reduce_cooldown_start_pct)}")
                    if not upnl_long_exceeded:
                        logging.info(f" - The long uPNL % of equity has not exceeded the threshold: {long_upnl_pct_equity:.2f} >= {-upnl_auto_reduce_threshold_long}")
                    if symbol in self.auto_reduce_active_long:
                        del self.auto_reduce_active_long[symbol]

            if trigger_auto_reduce_short:
                logging.info(f"Executing auto-reduce for short position in {symbol}.")
                self.auto_reduce_active_short[symbol] = True
                self.execute_grid_auto_reduce_hardened('short', symbol, short_pos_qty, short_dynamic_amount, current_market_price, total_equity, long_pos_price, short_pos_price, min_qty, min_buffer_percentage_ar, max_buffer_percentage_ar)
                self.last_auto_reduce_time[key_short] = current_time
            else:
                if currently_auto_reducing_short:
                    if current_time - self.last_auto_reduce_time.get(key_short, 0) <= cooldown_short:
                        logging.info(f"{symbol} Short position is still in the cooldown period. Time remaining: {cooldown_short - (current_time - self.last_auto_reduce_time.get(key_short, 0)):.2f} seconds.")
                    self.auto_reduce_active_short[symbol] = True
                else:
                    logging.info(f"No auto-reduce executed for short position in {symbol} because:")
                    if not short_loss_exceeded:
                        logging.info(f" - The current market price has not exceeded the threshold: {current_market_price} <= {short_pos_price * (1 + auto_reduce_cooldown_start_pct)}")
                    if not upnl_short_exceeded:
                        logging.info(f" - The short uPNL % of equity has not exceeded the threshold: {short_upnl_pct_equity:.2f} >= {-upnl_auto_reduce_threshold_short}")
                    if symbol in self.auto_reduce_active_short:
                        del self.auto_reduce_active_short[symbol]

        except Exception as e:
            logging.info(f"Error in auto-reduce logic for {symbol}: {e}")


    def execute_grid_auto_reduce_hardened(self, position_type, symbol, pos_qty, dynamic_amount, market_price, total_equity, long_pos_price, short_pos_price, min_qty, min_buffer_percentage_ar, max_buffer_percentage_ar):
        """
        Executes a single auto-reduction order for a position based on the best market price available,
        aiming to ensure a higher probability of order execution.
        """
        amount_precision, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
        price_precision_level = -int(math.log10(price_precision))
        qty_precision_level = -int(math.log10(amount_precision))

        # Fetch current best bid and ask prices to place the order as close as possible to the market
        # best_bid, best_ask = self.exchange.get_best_bid_ask(symbol)

        current_price = self.exchange.get_current_price(symbol)

        order_book = self.exchange.get_orderbook(symbol)
        best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol, current_price)
        best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol, current_price)
        
        # Determine the appropriate price to place the order based on position type
        order_price = best_bid_price if position_type == 'long' else best_ask_price
        order_price = round(order_price, price_precision_level)
        adjusted_dynamic_amount = max(dynamic_amount, min_qty)
        adjusted_dynamic_amount = round(adjusted_dynamic_amount, qty_precision_level)

        # Determine the positionIdx based on the position_type
        positionIdx = 1 if position_type == 'long' else 2

        logging.info(f"Attempting to place auto-reduce order: Symbol={symbol}, Type={'sell' if position_type == 'long' else 'buy'}, Qty={adjusted_dynamic_amount}, Price={order_price}")

        # Try placing the order using the provided utility method
        try:
            #order_result = self.postonly_limit_order_bybit_nolimit(symbol, 'sell' if position_type == 'long' else 'buy', adjusted_dynamic_amount, order_price, positionIdx, reduceOnly=True)
            order_result = self.limit_order_bybit_nolimit(symbol, 'sell' if position_type == 'long' else 'buy', adjusted_dynamic_amount, order_price, positionIdx, reduceOnly=True)
            logging.info(f"Auto-reduce order placed successfully: {order_result}")
        except Exception as e:
            logging.info(f"Failed to place auto-reduce order for {symbol}: {e}")
            raise

        # Log the order details for monitoring
        logging.info(f"Placed auto-reduce order for {symbol} at {order_price} for {adjusted_dynamic_amount}")


    def auto_reduce_marginbased_logic(self, auto_reduce_marginbased_enabled, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, symbol, total_equity, auto_reduce_wallet_exposure_pct, open_position_data, current_market_price, long_dynamic_amount, short_dynamic_amount, auto_reduce_start_pct, auto_reduce_maxloss_pct):
        if auto_reduce_marginbased_enabled:
            try:
                logging.info(f"Current market price for {symbol}: {current_market_price}")

                if symbol not in self.auto_reduce_orders:
                    self.auto_reduce_orders[symbol] = []

                active_auto_reduce_orders = []
                for order_id in self.auto_reduce_orders[symbol]:
                    order_status = self.exchange.get_order_status(order_id, symbol)
                    if order_status != 'canceled':
                        active_auto_reduce_orders.append(order_id)
                    else:
                        logging.info(f"Auto-reduce order {order_id} for {symbol} was canceled. Replacing it.")

                self.auto_reduce_orders[symbol] = active_auto_reduce_orders

                # Fetch open position data
                open_position_data = self.exchange.get_all_open_positions_bybit()

                # Initialize variables for used equity
                long_used_equity = 0
                short_used_equity = 0

                # Iterate through each position and calculate used equity
                for position in open_position_data:
                    info = position.get('info', {})

                    symbol_from_position = info.get('symbol', '').split(':')[0]
                    side_from_position = info.get('side', '')
                    position_balance = float(info.get('positionBalance', 0))

                    if symbol_from_position == symbol:
                        if side_from_position == 'Buy':
                            long_used_equity += position_balance
                        elif side_from_position == 'Sell':
                            short_used_equity += position_balance

                logging.info(f"Long used equity for {symbol} : {long_used_equity}")
                logging.info(f"Short used equity for {symbol} : {short_used_equity}")

                # Check if used equity exceeds the threshold for each side
                auto_reduce_triggered_long = long_used_equity > total_equity * auto_reduce_wallet_exposure_pct
                auto_reduce_triggered_short = short_used_equity > total_equity * auto_reduce_wallet_exposure_pct

                logging.info(f"Auto reduce trigger long for {symbol}: {auto_reduce_triggered_long}")
                logging.info(f"Auto reduce trigger short for {symbol}: {auto_reduce_triggered_short}")

                if long_pos_qty > 0 and long_pos_price is not None:
                    self.auto_reduce_active_long[symbol] = auto_reduce_triggered_long
                    if self.auto_reduce_active_long[symbol]:
                        max_levels, price_interval = self.calculate_auto_reduce_levels_long(
                            symbol,
                            current_market_price, long_pos_qty, long_dynamic_amount,
                            auto_reduce_start_pct, auto_reduce_maxloss_pct
                        )
                        for i in range(1, min(max_levels, 3) + 1):
                            step_price = current_market_price - (price_interval * i)
                            order_id = self.auto_reduce_long(symbol, long_dynamic_amount, step_price)
                            self.auto_reduce_orders[symbol].append(order_id)

                if short_pos_qty > 0 and short_pos_price is not None:
                    self.auto_reduce_active_short[symbol] = auto_reduce_triggered_short
                    if self.auto_reduce_active_short[symbol]:
                        max_levels, price_interval = self.calculate_auto_reduce_levels_short(
                            symbol,
                            current_market_price, short_pos_qty, short_dynamic_amount,
                            auto_reduce_start_pct, auto_reduce_maxloss_pct
                        )
                        for i in range(1, min(max_levels, 3) + 1):
                            step_price = current_market_price + (price_interval * i)
                            order_id = self.auto_reduce_short(symbol, short_dynamic_amount, step_price)
                            self.auto_reduce_orders[symbol].append(order_id)

            except Exception as e:
                logging.info(f"{symbol} Exception caught in margin auto reduce: {e}")

    def auto_reduce_percentile_logic(self, symbol, long_pos_qty, long_pos_price, short_pos_qty, short_pos_price, percentile_auto_reduce_enabled, auto_reduce_start_pct, auto_reduce_maxloss_pct, long_dynamic_amount, short_dynamic_amount):
        if percentile_auto_reduce_enabled:
            try:
                current_market_price = self.exchange.get_current_price(symbol)
                logging.info(f"Current market price for {symbol}: {current_market_price}")

                if symbol not in self.auto_reduce_orders:
                    self.auto_reduce_orders[symbol] = []

                active_auto_reduce_orders = []
                for order_id in self.auto_reduce_orders[symbol]:
                    order_status = self.exchange.get_order_status(order_id, symbol)
                    if order_status != 'canceled':
                        active_auto_reduce_orders.append(order_id)
                    else:
                        logging.info(f"Auto-reduce order {order_id} for {symbol} was canceled. Replacing it.")

                self.auto_reduce_orders[symbol] = active_auto_reduce_orders

                if long_pos_qty > 0 and long_pos_price is not None:
                    auto_reduce_start_price_long = long_pos_price * (1 - auto_reduce_start_pct)
                    self.auto_reduce_active_long[symbol] = current_market_price <= auto_reduce_start_price_long
                    if self.auto_reduce_active_long[symbol]:
                        max_levels, price_interval = self.calculate_auto_reduce_levels_long(
                            symbol,
                            current_market_price, long_pos_qty, long_dynamic_amount,
                            auto_reduce_start_pct, auto_reduce_maxloss_pct
                        )
                        for i in range(1, min(max_levels, 3) + 1):
                            step_price = current_market_price - (price_interval * i)
                            order_id = self.auto_reduce_long(symbol, long_dynamic_amount, step_price)
                            self.auto_reduce_orders[symbol].append(order_id)

                if short_pos_qty > 0 and short_pos_price is not None:
                    auto_reduce_start_price_short = short_pos_price * (1 + auto_reduce_start_pct)
                    self.auto_reduce_active_short[symbol] = current_market_price >= auto_reduce_start_price_short
                    if self.auto_reduce_active_short[symbol]:
                        max_levels, price_interval = self.calculate_auto_reduce_levels_short(
                            symbol,
                            current_market_price, short_pos_qty, short_dynamic_amount,
                            auto_reduce_start_pct, auto_reduce_maxloss_pct
                        )
                        for i in range(1, min(max_levels, 3) + 1):
                            step_price= current_market_price + (price_interval * i)
                            order_id = self.auto_reduce_short(symbol, short_dynamic_amount, step_price)
                            self.auto_reduce_orders[symbol].append(order_id)
            except Exception as e:
                logging.info(f"{symbol} Exception caught in auto reduce: {e}")

    def cancel_auto_reduce_orders_bybit(self, symbol, total_equity, max_pos_balance_pct, open_position_data, long_pos_qty, short_pos_qty):
        try:
            # Get current position balances
            long_position_balance = self.get_position_balance(symbol, 'Buy', open_position_data)
            short_position_balance = self.get_position_balance(symbol, 'Sell', open_position_data)
            long_position_balance_pct = (long_position_balance / total_equity) * 100
            short_position_balance_pct = (short_position_balance / total_equity) * 100

            # Cancel long auto-reduce orders if position balance is below max threshold and long position is open
            if long_pos_qty > 0 and long_position_balance_pct < max_pos_balance_pct and symbol in self.auto_reduce_orders:
                for order_id in self.auto_reduce_orders[symbol]:
                    try:
                        self.exchange.cancel_order_bybit(order_id, symbol)
                        logging.info(f"Cancelling long auto-reduce order: {order_id}")
                    except Exception as e:
                        logging.warning(f"An error occurred while cancelling auto-reduce order {order_id}: {e}")
                self.auto_reduce_orders[symbol].clear()  # Clear the list after cancellation

            # Cancel short auto-reduce orders if position balance is below max threshold and short position is open
            if short_pos_qty > 0 and short_position_balance_pct < max_pos_balance_pct and symbol in self.auto_reduce_orders:
                for order_id in self.auto_reduce_orders[symbol]:
                    try:
                        self.exchange.cancel_order_bybit(order_id, symbol)
                        logging.info(f"Cancelling short auto-reduce order: {order_id}")
                    except Exception as e:
                        logging.warning(f"An error occurred while cancelling auto-reduce order {order_id}: {e}")
                self.auto_reduce_orders[symbol].clear()  # Clear the list after cancellation

        except Exception as e:
            logging.info(f"An error occurred while canceling auto-reduce orders for {symbol}: {e}")
            

    def calculate_dynamic_auto_reduce_levels(self, symbol, pos_qty, market_price, total_equity, long_pos_price, short_pos_price):
        # Check if conditions have changed significantly to recalculate levels
        if symbol in self.previous_levels:
            last_market_price, last_max_levels, last_price_interval = self.previous_levels[symbol]
            if abs(last_market_price - market_price) < market_price * Decimal('0.01'):  # 1% change threshold
                # Return previous levels if market price change is within threshold
                return last_max_levels, last_price_interval

        volatility_metric = self.calculate_volatility_metric(symbol)
        risk_factor = abs(pos_qty / total_equity)

        # Dynamic scaling based on volatility and risk
        volatility_scale = min(max(1, volatility_metric * 10), 5)
        risk_scale = min(max(1, risk_factor * 10), 5)

        # Base number of levels
        base_levels = 10
        volatility_adjustment = int(volatility_metric * volatility_scale)
        risk_adjustment = int(risk_factor * risk_scale)

        # Calculate max_levels and ensure it stays within a reasonable range
        max_levels = base_levels + volatility_adjustment + risk_adjustment
        max_levels = min(max(max_levels, 5), 30)

        # Calculate price range dynamically based on the position price and current market price
        position_price = Decimal(str(long_pos_price if pos_qty > 0 else short_pos_price))
        total_price_range = abs(position_price - market_price)

        # Calculate price_interval dynamically
        price_interval = total_price_range / max_levels if max_levels > 1 else total_price_range

        # Store the calculated levels for future reference
        self.previous_levels[symbol] = (market_price, max_levels, price_interval)

        # Logging for debugging and analysis
        logging.info(f"Symbol: {symbol}, Volatility Metric: {volatility_metric}, Risk Factor: {risk_factor}")
        logging.info(f"Volatility Scale: {volatility_scale}, Risk Scale: {risk_scale}")
        logging.info(f"Volatility Adjustment: {volatility_adjustment}, Risk Adjustment: {risk_adjustment}")
        logging.info(f"Base Levels: {base_levels}, Max Levels: {max_levels}, Price Interval: {price_interval}, Total Price Range: {total_price_range}")

        return max_levels, price_interval

    def place_auto_reduce_order(self, symbol, step_price, dynamic_amount, position_type):
        try:
            if position_type == 'long':
                # Place a reduce-only sell order for long positions
                order = self.auto_reduce_long(symbol, step_price, dynamic_amount)
            elif position_type == 'short':
                # Place a reduce-only buy order for short positions
                order = self.auto_reduce_short(symbol, step_price, dynamic_amount)
            else:
                raise ValueError(f"Invalid position type: {position_type}")

            order_id = order.get('id', None) if order else None
            logging.info(f"Auto-reduce {position_type} order placed for {symbol} at {step_price} with amount {dynamic_amount}")
            return order_id
        except Exception as e:
            logging.info(f"Error in placing auto-reduce {position_type} order for {symbol}: {e}")
            return None

    # This worked until it does not. The max_loss_pct is used to calculate the grid and causes issues giving you further AR entries
    def auto_reduce_logic(self, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, auto_reduce_enabled, symbol, total_equity, auto_reduce_wallet_exposure_pct, open_position_data, current_market_price, long_dynamic_amount, short_dynamic_amount, auto_reduce_start_pct, auto_reduce_maxloss_pct):
        if auto_reduce_enabled:
            try:
                # Initialize variables for unrealized PnL
                long_unrealised_pnl = 0
                short_unrealised_pnl = 0

                # Iterate through each position and calculate unrealized PnL
                for position in open_position_data:
                    info = position.get('info', {})
                    symbol_from_position = info.get('symbol', '').split(':')[0]
                    side_from_position = info.get('side', '')
                    unrealised_pnl = float(info.get('unrealisedPnl', 0))

                    if symbol_from_position == symbol:
                        if side_from_position == 'Buy':
                            long_unrealised_pnl += unrealised_pnl
                        elif side_from_position == 'Sell':
                            short_unrealised_pnl += unrealised_pnl

                # Calculate PnL as a percentage of total equity
                long_pnl_percentage = (long_unrealised_pnl / total_equity) * 100
                short_pnl_percentage = (short_unrealised_pnl / total_equity) * 100

                # Determine how much more is needed to exceed the limit
                long_pnl_excess_needed = auto_reduce_wallet_exposure_pct - abs(long_pnl_percentage) if long_pnl_percentage < 0 else auto_reduce_wallet_exposure_pct - long_pnl_percentage
                short_pnl_excess_needed = auto_reduce_wallet_exposure_pct - abs(short_pnl_percentage) if short_pnl_percentage < 0 else auto_reduce_wallet_exposure_pct - short_pnl_percentage

                # Log the unrealized PnL percentage and excess needed
                logging.info(f"{symbol} Long unrealized PnL: {long_pnl_percentage:.2f}%, Excess needed to auto-reduce: {long_pnl_excess_needed:.2f}%")
                logging.info(f"{symbol} Short unrealized PnL: {short_pnl_percentage:.2f}%, Excess needed to auto-reduce: {short_pnl_excess_needed:.2f}%")

                # Check if unrealized PnL exceeds the threshold for each side
                auto_reduce_triggered_long = long_pnl_percentage > auto_reduce_wallet_exposure_pct
                auto_reduce_triggered_short = short_pnl_percentage > auto_reduce_wallet_exposure_pct

                # Long position auto-reduce check
                if long_pos_qty > 0 and long_pos_price is not None and auto_reduce_triggered_long:
                    if current_market_price >= long_pos_price * (1 + auto_reduce_start_pct):  # Position price threshold check
                        max_levels, price_interval = self.calculate_auto_reduce_levels_long(
                            symbol, current_market_price, long_pos_qty, long_dynamic_amount,
                            auto_reduce_start_pct, auto_reduce_maxloss_pct
                        )
                        for i in range(1, min(max_levels, 3) + 1):
                            step_price = current_market_price - (price_interval * i)
                            order_id = self.auto_reduce_long(symbol, long_dynamic_amount, step_price)
                            self.auto_reduce_orders[symbol].append(order_id)

                # Short position auto-reduce check
                if short_pos_qty > 0 and short_pos_price is not None and auto_reduce_triggered_short:
                    if current_market_price <= short_pos_price * (1 - auto_reduce_start_pct):  # Position price threshold check
                        max_levels, price_interval = self.calculate_auto_reduce_levels_short(
                            symbol, current_market_price, short_pos_qty, short_dynamic_amount,
                            auto_reduce_start_pct, auto_reduce_maxloss_pct
                        )
                        for i in range(1, min(max_levels, 3) + 1):
                            step_price = current_market_price + (price_interval * i)
                            order_id = self.auto_reduce_short(symbol, short_dynamic_amount, step_price)
                            self.auto_reduce_orders[symbol].append(order_id)

            except Exception as e:
                logging.info(f"{symbol} Exception caught in auto reduce: {e}")

    def calculate_auto_reduce_levels_long(self, symbol, current_market_price, long_pos_qty, long_dynamic_amount, auto_reduce_start_pct, max_loss_pct):
        try:
            if long_dynamic_amount <= 0:
                raise ValueError("Dynamic amount for long positions must be greater than zero.")

            # Calculate the number of levels, ensuring at least one level
            max_levels = max(int(long_pos_qty / long_dynamic_amount), 1)

            # Calculate the price difference for auto-reduce start and max loss
            price_diff_start = current_market_price * (1 - auto_reduce_start_pct)
            price_diff_max = current_market_price * (1 - max_loss_pct)
            total_price_range = price_diff_max - price_diff_start

            # Calculate the price interval between auto-reduce levels
            price_interval = total_price_range / max_levels if max_levels > 1 else total_price_range

            logging.info(f"Long Auto-Reduce for {symbol}: Price Start: {price_diff_start}, Price Max: {price_diff_max}, Total Range: {total_price_range}, Max Levels: {max_levels}, Price Interval: {price_interval}")

            return max_levels, price_interval
        except Exception as e:
            logging.info(f"Error calculating auto-reduce levels for long position in {symbol}: {e}")
            return None, None

    def calculate_auto_reduce_levels_short(self, symbol, current_market_price, short_pos_qty, short_dynamic_amount, auto_reduce_start_pct, max_loss_pct):
        try:
            if short_dynamic_amount <= 0:
                raise ValueError("Dynamic amount for short positions must be greater than zero.")

            # Calculate the number of levels, ensuring at least one level
            max_levels = max(int(short_pos_qty / short_dynamic_amount), 1)

            # Calculate the price difference for auto-reduce start and max loss
            price_diff_start = current_market_price * (1 + auto_reduce_start_pct)
            price_diff_max = current_market_price * (1 + max_loss_pct)
            total_price_range = price_diff_max - price_diff_start

            # Calculate the price interval between auto-reduce levels
            price_interval = total_price_range / max_levels if max_levels > 1 else total_price_range

            logging.info(f"Short Auto-Reduce for {symbol}: Price Start: {price_diff_start}, Price Max: {price_diff_max}, Total Range: {total_price_range}, Max Levels: {max_levels}, Price Interval: {price_interval}")

            return max_levels, price_interval
        except Exception as e:
            logging.info(f"Error calculating auto-reduce levels for short position in {symbol}: {e}")
            return None, None

    def auto_reduce_long(self, symbol, long_dynamic_amount, step_price):
        try:
            order = self.limit_order_bybit_reduce_nolimit(symbol, 'sell', long_dynamic_amount, float(step_price), positionIdx=1, reduceOnly=True)
            logging.info(f"Auto-reduce long order placed for {symbol} at {step_price} with amount {long_dynamic_amount}")
            return order.get('id', None) if order else None
        except Exception as e:
            logging.info(f"Error in auto-reduce long order for {symbol}: {e}")
            logging.info("Traceback:", traceback.format_exc())
            return None

    def auto_reduce_short(self, symbol, short_dynamic_amount, step_price):
        try:
            order = self.limit_order_bybit_reduce_nolimit(symbol, 'buy', short_dynamic_amount, float(step_price), positionIdx=2, reduceOnly=True)
            logging.info(f"Auto-reduce short order placed for {symbol} at {step_price} with amount {short_dynamic_amount}")
            return order.get('id', None) if order else None
        except Exception as e:
            logging.info(f"Error in auto-reduce short order for {symbol}: {e}")
            logging.info("Traceback:", traceback.format_exc())
            return None


    def calculate_long_stop_loss_based_on_liq_price(self, long_pos_price, long_liq_price, liq_price_stop_pct):
        if long_pos_price is None or long_liq_price is None:
            return None

        # Calculate the stop loss price as a percentage of the distance to the liquidation price
        stop_loss_distance = (long_liq_price - long_pos_price) * liq_price_stop_pct
        stop_loss_price = long_pos_price + stop_loss_distance

        logging.info(f"Stop loss distance: {stop_loss_distance}")
        logging.info(f"Stop loss price: {stop_loss_price}")
        return stop_loss_price

    def calculate_short_stop_loss_based_on_liq_price(self, short_pos_price, short_liq_price, liq_price_stop_pct):
        if short_pos_price is None or short_liq_price is None:
            return None

        # Calculate the stop loss price as a percentage of the distance to the liquidation price
        stop_loss_distance = (short_pos_price - short_liq_price) * liq_price_stop_pct
        stop_loss_price = short_pos_price - stop_loss_distance

        logging.info(f"Stop loss distance: {stop_loss_distance}")
        logging.info(f"Stop loss price: {stop_loss_price}")
        return stop_loss_price

    def calculate_quickscalp_long_stop_loss(self, long_pos_price, symbol, stoploss_upnl_pct):
        if long_pos_price is None:
            return None

        price_precision = int(self.exchange.get_price_precision(symbol))
        logging.info(f"Price precision for {symbol}: {price_precision}")

        # Calculate the stop loss price by reducing the long position price by the stop loss percentage
        stop_loss_price = Decimal(long_pos_price) * (1 - Decimal(stoploss_upnl_pct))

        # Quantize the stop loss price
        try:
            stop_loss_price = stop_loss_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_DOWN
            )
        except InvalidOperation as e:
            logging.info(f"Error when quantizing stop_loss_price. {e}")
            return None

        return float(stop_loss_price)

    def calculate_quickscalp_short_stop_loss(self, short_pos_price, symbol, stoploss_upnl_pct):
        if short_pos_price is None:
            return None

        price_precision = int(self.exchange.get_price_precision(symbol))
        logging.info(f"Price precision for {symbol}: {price_precision}")

        # Calculate the stop loss price by increasing the short position price by the stop loss percentage
        stop_loss_price = Decimal(short_pos_price) * (1 + Decimal(stoploss_upnl_pct))

        # Quantize the stop loss price
        try:
            stop_loss_price = stop_loss_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_DOWN
            )
        except InvalidOperation as e:
            logging.info(f"Error when quantizing stop_loss_price. {e}")
            return None

        return float(stop_loss_price)

    def quickscalp_mfi_handle_long_positions(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, long_pos_qty: float, long_pos_price: float):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"Current price for {symbol}: {current_price}")

            order_book = self.exchange.get_orderbook(symbol)
            # Extract and update best bid price
            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
            else:
                best_bid_price = self.last_known_bid.get(symbol)

            mfi_signal_long = mfirsi.lower() == "long"

            if one_minute_volume > min_vol:
                if long_pos_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial MFI-based long entry for {symbol} at {best_bid_price} with amount {long_dynamic_amount}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional MFI-based long entry for {symbol} at {best_bid_price} with amount {long_dynamic_amount}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            else:
                logging.info(f"Volume conditions not met for long position in {symbol}, skipping entry.")

            time.sleep(5)

    def quickscalp_mfi_handle_short_positions(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, short_dynamic_amount: float, short_pos_qty: float, short_pos_price: float):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"Current price for {symbol}: {current_price}")

            order_book = self.exchange.get_orderbook(symbol)
            # Extract and update best ask price
            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
            else:
                best_ask_price = self.last_known_ask.get(symbol)

            mfi_signal_short = mfirsi.lower() == "short"

            if one_minute_volume > min_vol:
                if short_pos_qty == 0 and mfi_signal_short and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial MFI-based short entry for {symbol} at {best_ask_price} with amount {short_dynamic_amount}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional MFI-based short entry for {symbol} at {best_ask_price} with amount {short_dynamic_amount}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume conditions not met for short position in {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_long_only_strategy(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, long_pos_qty: float, long_pos_price: float, entry_during_autoreduce: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"Current price for {symbol}: {current_price}")

            order_book = self.exchange.get_orderbook(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

            mfi_signal_long = mfirsi.lower() == "long"

            if one_minute_volume > min_vol:
                if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                    if long_pos_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(1)
                    elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(1)
            else:
                logging.info(f"Volume conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_1m_mfi_quickscalp_trend_dca(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, entry_during_autoreduce: bool, volume_check: bool):
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

            # Check if volume check is enabled or not
            if not volume_check or (one_minute_volume > min_vol):
                if not self.auto_reduce_active_long.get(symbol, False):
                    if long_pos_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(1)
                    elif long_pos_qty > 0 and mfi_signal_long:
                        # Calculate the DCA order size needed to bring the position to the current price
                        dca_order_size = self.calculate_dca_order_size(long_pos_qty, long_pos_price, current_price, symbol)
                        if dca_order_size > 0 and not self.entry_order_exists(open_orders, "buy"):
                            self.place_postonly_order_bybit(symbol, "buy", dca_order_size, best_bid_price, positionIdx=1, reduceOnly=False)
                        else:
                            # If DCA is not needed, check if additional entry is allowed during auto-reduce
                            if entry_during_autoreduce:
                                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            else:
                                logging.info(f"Skipping additional long entry for {symbol} due to active auto-reduce and entry_during_autoreduce set to False.")
                        time.sleep(1)

                if not self.auto_reduce_active_short.get(symbol, False):
                    if short_pos_qty == 0 and mfi_signal_short and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(1)
                    elif short_pos_qty > 0 and mfi_signal_short:
                        # Calculate the DCA order size needed to bring the position to the current price
                        dca_order_size = self.calculate_dca_order_size(short_pos_qty, short_pos_price, current_price, symbol)
                        if dca_order_size > 0 and not self.entry_order_exists(open_orders, "sell"):
                            self.place_postonly_order_bybit(symbol, "sell", dca_order_size, best_ask_price, positionIdx=2, reduceOnly=False)
                        else:
                            # If DCA is not needed, check if additional entry is allowed during auto-reduce
                            if entry_during_autoreduce:
                                self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                            else:
                                logging.info(f"Skipping additional short entry for {symbol} due to active auto-reduce and entry_during_autoreduce set to False.")
                        time.sleep(1)
            else:
                logging.info(f"Volume check is disabled or conditions not met for {symbol}, proceeding without volume check.")

            time.sleep(5)

    def calculate_dca_order_size(self, open_position_qty, open_position_avg_price, current_market_price, symbol):
        """
        Calculate the DCA order size needed to adjust the average price of the open position to the current market price.
        """
        if open_position_qty == 0:
            return 0  # No open position to adjust

        # Calculate the total cost of the current position
        total_position_cost = open_position_qty * open_position_avg_price

        logging.info(f"Total position cost for {symbol}: {total_position_cost}")

        # Calculate the quantity needed for DCA to achieve the current market price as the new average price
        dca_qty_needed = (total_position_cost - open_position_qty * current_market_price) / (current_market_price - open_position_avg_price)

        logging.info(f"DCA qty needed for {symbol}: {dca_qty_needed}")

        # Fetch the precision for the symbol to use in rounding
        _, price_precision = self.exchange.get_symbol_precision_bybit(symbol)
        qty_precision = -int(math.log10(price_precision))  # Assuming price precision is a good proxy for quantity precision

        # Adjust the DCA order size based on the symbol's quantity precision
        dca_order_size_adjusted = round(dca_qty_needed, qty_precision)

        logging.info(f"DCA order size for {symbol} is {dca_order_size_adjusted}")

        return max(0, dca_order_size_adjusted)  # Ensure the DCA quantity is non-negative

    def bybit_1m_mfi_quickscalp(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, eri_trend: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, uPNL_threshold: float, entry_during_autoreduce: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"Current price for {symbol}: {current_price}")

            order_book = self.exchange.get_orderbook(symbol)
            # Extract and update best ask/bid prices
            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
            else:
                best_ask_price = self.last_known_ask.get(symbol)

            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
            else:
                best_bid_price = self.last_known_bid.get(symbol)

            mfi_signal_long = mfirsi.lower() == "long"
            mfi_signal_short = mfirsi.lower() == "short"

            if one_minute_volume > min_vol:
                # Entry logic for initial and additional entries
                if not self.auto_reduce_active_long.get(symbol, False) or entry_during_autoreduce:
                    if long_pos_qty == 0 and mfi_signal_long and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial MFI-based long entry for {symbol} at {best_bid_price} with amount {long_dynamic_amount}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(1)
                    elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional MFI-based long entry for {symbol} at {best_bid_price} with amount {long_dynamic_amount}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(1)

                if not self.auto_reduce_active_short.get(symbol, False) or entry_during_autoreduce:
                    if short_pos_qty == 0 and mfi_signal_short and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial MFI-based short entry for {symbol} at {best_ask_price} with amount {short_dynamic_amount}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(1)
                    elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional MFI-based short entry for {symbol} at {best_ask_price} with amount {short_dynamic_amount}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(1)
            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_1m_mfi_eri_walls(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, fivemin_top_signal: bool, fivemin_bottom_signal: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_significant_order_book_walls(symbol)
            # bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            logging.info(f"Current price in autohedge: for {symbol} : {current_price}")

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)

            # Extract and update best ask/bid prices
            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
            else:
                best_ask_price = self.last_known_ask.get(symbol)

            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
            else:
                best_bid_price = self.last_known_bid.get(symbol)

            min_order_size = 1

            # Trend Alignment Checks
            trend_aligned_long = (eri_trend == "bullish" or trend.lower() == "long")
            trend_aligned_short = (eri_trend == "bearish" or trend.lower() == "short")

            eri_trend_aligned_long = eri_trend == "bullish"
            eri_trend_aligned_short = eri_trend == "bearish"

            mfi_signal_long = mfi.lower() == "long"
            mfi_signal_short = mfi.lower() == "short"
            mfi_signal_neutral = mfi.lower() == "neutral"

            if one_minute_volume > min_vol:
                # Long Entry for Trend and MFI Signal
                if eri_trend_aligned_long and mfi_signal_long:
                    if long_pos_qty == 0 and should_long and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(5)

                # Short Entry for Trend and MFI Signal
                if eri_trend_aligned_short and mfi_signal_short:
                    if short_pos_qty == 0 and should_short and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(5)

                # Order Book Wall Long Entry Logic
                if largest_bid_wall and not self.entry_order_exists(open_orders, "buy"):
                    price_approaching_bid_wall = self.is_price_approaching_wall(current_price, largest_bid_wall[0], 'bid')

                    # Check if the bottom signal is present for long entries
                    if price_approaching_bid_wall and (should_long or should_add_to_long) and eri_trend_aligned_long and mfi_signal_neutral and fivemin_bottom_signal:
                        logging.info(f"Price approaching significant buy wall and bottom signal detected for {symbol}. Placing long trade.")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)
                        time.sleep(5)

                # Order Book Wall Short Entry Logic
                if largest_ask_wall and not self.entry_order_exists(open_orders, "sell"):
                    price_approaching_ask_wall = self.is_price_approaching_wall(current_price, largest_ask_wall[0], 'ask')

                    # Check if the top signal is present for short entries
                    if price_approaching_ask_wall and (should_short or should_add_to_short) and eri_trend_aligned_short and mfi_signal_neutral and fivemin_top_signal:
                        logging.info(f"Price approaching significant sell wall and top signal detected for {symbol}. Placing short trade.")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)
                        time.sleep(5)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_1m_mfi_eri_walls(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, fivemin_top_signal: bool, fivemin_bottom_signal: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_significant_order_book_walls(symbol)
            # bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            logging.info(f"Current price in autohedge: for {symbol} : {current_price}")

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)

            # Extract and update best ask/bid prices
            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
            else:
                best_ask_price = self.last_known_ask.get(symbol)

            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
            else:
                best_bid_price = self.last_known_bid.get(symbol)

            min_order_size = 1

            # Trend Alignment Checks
            trend_aligned_long = (eri_trend == "bullish" or trend.lower() == "long")
            trend_aligned_short = (eri_trend == "bearish" or trend.lower() == "short")

            eri_trend_aligned_long = eri_trend == "bullish"
            eri_trend_aligned_short = eri_trend == "bearish"

            mfi_signal_long = mfi.lower() == "long"
            mfi_signal_short = mfi.lower() == "short"
            mfi_signal_neutral = mfi.lower() == "neutral"

            if one_minute_volume > min_vol:
                # Long Entry for Trend and MFI Signal
                if eri_trend_aligned_long and mfi_signal_long:
                    if long_pos_qty == 0 and should_long and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                        time.sleep(5)

                # Short Entry for Trend and MFI Signal
                if eri_trend_aligned_short and mfi_signal_short:
                    if short_pos_qty == 0 and should_short and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol}")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                        time.sleep(5)

                # Order Book Wall Long Entry Logic
                if largest_bid_wall and not self.entry_order_exists(open_orders, "buy"):
                    price_approaching_bid_wall = self.is_price_approaching_wall(current_price, largest_bid_wall[0], 'bid')

                    # Check if the bottom signal is present for long entries
                    if price_approaching_bid_wall and (should_long or should_add_to_long) and eri_trend_aligned_long and mfi_signal_neutral and fivemin_bottom_signal:
                        logging.info(f"Price approaching significant buy wall and bottom signal detected for {symbol}. Placing long trade.")
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)
                        time.sleep(5)

                # Order Book Wall Short Entry Logic
                if largest_ask_wall and not self.entry_order_exists(open_orders, "sell"):
                    price_approaching_ask_wall = self.is_price_approaching_wall(current_price, largest_ask_wall[0], 'ask')

                    # Check if the top signal is present for short entries
                    if price_approaching_ask_wall and (should_short or should_add_to_short) and eri_trend_aligned_short and mfi_signal_neutral and fivemin_top_signal:
                        logging.info(f"Price approaching significant sell wall and top signal detected for {symbol}. Placing short trade.")
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)
                        time.sleep(5)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_1m_mfi_eri_walls_atr_topbottom(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, hedge_ratio: float, atr: float, top_signal_short: bool, bottom_signal_long: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else self.last_known_ask.get(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else self.last_known_bid.get(symbol)

            min_order_size = 1

            # Call to your auto hedging function
            self.auto_hedge_orders_bybit_atr(symbol, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, best_ask_price, best_bid_price, hedge_ratio, atr, min_order_size)

            # Trend Alignment Checks based on ERI trend
            eri_trend_aligned_long = eri_trend == "bullish"
            eri_trend_aligned_short = eri_trend == "bearish"

            if one_minute_volume > min_vol:
                # Long Entry for Trend and MFI Signal
                mfi_signal_long = mfi.lower() == "long"
                if eri_trend_aligned_long and (should_long or should_add_to_long) and current_price >= qfl_base and mfi_signal_long:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Short Entry for Trend and MFI Signal
                mfi_signal_short = mfi.lower() == "short"
                if eri_trend_aligned_short and (should_short or should_add_to_short) and current_price <= qfl_ceiling and mfi_signal_short:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Order Book Wall Long Entry Logic
                if largest_bid_wall and eri_trend_aligned_long and should_add_to_long and not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                # Modified Order Book Wall Short Entry Logic
                if largest_ask_wall and eri_trend_aligned_short and should_add_to_short and not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_1m_mfi_eri_walls_autohedge(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, hedge_ratio: float, atr: float):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else self.last_known_ask.get(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else self.last_known_bid.get(symbol)

            min_order_size = 1

            # Call to your auto hedging function
            self.auto_hedge_orders_bybit_atr(symbol, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, best_ask_price, best_bid_price, hedge_ratio, atr, min_order_size)

            # Trend Alignment Checks based on ERI trend
            eri_trend_aligned_long = eri_trend == "bullish"
            eri_trend_aligned_short = eri_trend == "bearish"

            if one_minute_volume > min_vol:
                # Long Entry for Trend and MFI Signal
                mfi_signal_long = mfi.lower() == "long"
                if eri_trend_aligned_long and (should_long or should_add_to_long) and current_price >= qfl_base and mfi_signal_long:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Short Entry for Trend and MFI Signal
                mfi_signal_short = mfi.lower() == "short"
                if eri_trend_aligned_short and (should_short or should_add_to_short) and current_price <= qfl_ceiling and mfi_signal_short:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Order Book Wall Long Entry Logic
                if largest_bid_wall and eri_trend_aligned_long and should_add_to_long and not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                # Modified Order Book Wall Short Entry Logic
                if largest_ask_wall and eri_trend_aligned_short and should_add_to_short and not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)


    def bybit_1m_mfi_eri_walls_atr(self, min_qty, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, hedge_ratio: float, atr: float, fivemin_top_signal: bool, fivemin_bottom_signal: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else self.last_known_ask.get(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else self.last_known_bid.get(symbol)

            # Call to your auto hedging function
            self.auto_hedge_orders_bybit_atr(symbol, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, best_ask_price, best_bid_price, hedge_ratio, atr, min_order_size=min_qty)

            # Trend Alignment Checks based on ERI trend
            eri_trend_aligned_long = eri_trend == "bullish"
            eri_trend_aligned_short = eri_trend == "bearish"

            if one_minute_volume > min_vol:
                # Long Entry for Trend and MFI Signal
                mfi_signal_long = mfi.lower() == "long"
                if eri_trend_aligned_long and (should_long or should_add_to_long) and current_price >= qfl_base and mfi_signal_long:
                    if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Short Entry for Trend and MFI Signal
                mfi_signal_short = mfi.lower() == "short"
                if eri_trend_aligned_short and (should_short or should_add_to_short) and current_price <= qfl_ceiling and mfi_signal_short:
                    if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

                # Order Book Wall Long Entry Logic
                if largest_bid_wall and eri_trend_aligned_long and should_add_to_long and fivemin_bottom_signal and not self.entry_order_exists(open_orders, "buy"):
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

                # Modified Order Book Wall Short Entry Logic
                if largest_ask_wall and eri_trend_aligned_short and should_add_to_short and fivemin_top_signal and not self.entry_order_exists(open_orders, "sell"):
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

            else:
                logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

            time.sleep(5)

    def bybit_initial_entry_quickscalp(self, open_orders: list, symbol: str, mfi: str, one_minute_volume: float, min_vol: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Initial entry function with QFL, MFI, and ERI trend initialized for {symbol}")

            # Detecting order book walls
            bid_walls, ask_walls = self.detect_significant_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Process order book and update best ask/bid prices
            order_book = self.exchange.get_orderbook(symbol)
            # Extract and update best ask/bid prices
            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
            else:
                best_ask_price = self.last_known_ask.get(symbol)

            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
            else:
                best_bid_price = self.last_known_bid.get(symbol)

            # Trend and MFI Signal Checks
            mfi_signal_long = mfi.lower() == "long"
            mfi_signal_short = mfi.lower() == "short"
            mfi_signal_neutral = mfi.lower() == "neutral"

            if one_minute_volume > min_vol:
                # Long Entry Logic
                if long_pos_qty == 0 and mfi_signal_long:
                    if not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        entry_price = largest_bid_wall[0] if largest_bid_wall else best_bid_price
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, entry_price, positionIdx=1, reduceOnly=False)
                        time.sleep(5)

                # Short Entry Logic
                if short_pos_qty == 0 and mfi_signal_short:
                    if not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        entry_price = largest_ask_wall[0] if largest_ask_wall else best_ask_price
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, entry_price, positionIdx=2, reduceOnly=False)
                        time.sleep(5)


            time.sleep(5)

    def bybit_initial_entry_with_qfl_mfi_and_eri(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, eri_trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool, fivemin_top_signal: bool, fivemin_bottom_signal: bool):

        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            logging.info(f"Initial entry function with QFL, MFI, and ERI trend initialized for {symbol}")

            # Detecting order book walls
            bid_walls, ask_walls = self.detect_significant_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Process order book and update best ask/bid prices
            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else self.last_known_ask.get(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else self.last_known_bid.get(symbol)

            # Trend and MFI Signal Checks
            trend_aligned_long = (eri_trend == "bullish" or trend.lower() == "long") and mfi.lower() == "long"
            trend_aligned_short = (eri_trend == "bearish" or trend.lower() == "short") and mfi.lower() == "short"

            eri_trend_aligned_long = eri_trend == "bullish"
            eri_trend_aligned_short = eri_trend == "bearish"

            mfi_signal_long = mfi.lower() == "long"
            mfi_signal_short = mfi.lower() == "short"
            mfi_signal_neutral = mfi.lower() == "neutral"

            if one_minute_volume > min_vol:
                # Long Entry Logic
                if should_long and long_pos_qty == 0 and eri_trend_aligned_long and current_price >= qfl_base and mfi_signal_long:
                    if not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing initial long entry for {symbol}")
                        entry_price = largest_bid_wall[0] if largest_bid_wall else best_bid_price
                        self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, entry_price, positionIdx=1, reduceOnly=False)

                # Short Entry Logic
                if should_short and short_pos_qty == 0 and eri_trend_aligned_short and current_price <= qfl_ceiling and mfi_signal_short:
                    if not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing initial short entry for {symbol}")
                        entry_price = largest_ask_wall[0] if largest_ask_wall else best_ask_price
                        self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, entry_price, positionIdx=2, reduceOnly=False)


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

    def bybit_qs_entry_exit_eri(self, open_orders: list, symbol: str, trend: str, mfi: str, eri_trend: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool, hedge_ratio: float, price_difference_threshold: float):
        if five_minute_volume > min_vol:
            # Fetch necessary data
            bid_walls, ask_walls = self.detect_order_book_walls(symbol)
            largest_bid_wall = max(bid_walls, key=lambda x: x[1], default=None)
            largest_ask_wall = max(ask_walls, key=lambda x: x[1], default=None)

            qfl_base, qfl_ceiling = self.calculate_qfl_levels(symbol=symbol, timeframe='5m', lookback_period=12)
            current_price = self.exchange.get_current_price(symbol)

            # Fetch and process order book
            order_book = self.exchange.get_orderbook(symbol)

            # Extract and update best ask/bid prices
            if 'asks' in order_book and len(order_book['asks']) > 0:
                best_ask_price = order_book['asks'][0][0]
            else:
                best_ask_price = self.last_known_ask.get(symbol)

            if 'bids' in order_book and len(order_book['bids']) > 0:
                best_bid_price = order_book['bids'][0][0]
            else:
                best_bid_price = self.last_known_bid.get(symbol)

            min_order_size = 1

            # Auto-hedging logic for long position
            if long_pos_qty > 0:
                price_diff_percentage_long = abs(current_price - long_pos_price) / long_pos_price
                logging.info(f"Price difference long for {symbol}: {price_diff_percentage_long * 100:.2f}%")
                current_hedge_ratio_long = short_pos_qty / long_pos_qty if long_pos_qty > 0 else 0
                logging.info(f"Current hedge ratio long for {symbol}: {current_hedge_ratio_long:.2f}")

                if price_diff_percentage_long >= price_difference_threshold and current_hedge_ratio_long < hedge_ratio:
                    logging.info(f"Auto hedging for long position for {symbol}")
                    additional_hedge_needed_long = (long_pos_qty * hedge_ratio) - short_pos_qty
                    logging.info(f"Additional hedge needed long for {symbol}: {additional_hedge_needed_long}")

                    if additional_hedge_needed_long > min_order_size:
                        logging.info(f"Placing auto-hedge sell order for {symbol}: Amount: {additional_hedge_needed_long}, Price: {best_ask_price}")
                        order_response = self.place_postonly_order_bybit(symbol, "sell", additional_hedge_needed_long, best_ask_price, positionIdx=2, reduceOnly=False)
                        logging.info(f"Order response for {symbol} (Long Auto-Hedge): {order_response}")

            # Auto-hedging logic for short position
            if short_pos_qty > 0:
                price_diff_percentage_short = abs(current_price - short_pos_price) / short_pos_price
                logging.info(f"Price difference short for {symbol}: {price_diff_percentage_short * 100:.2f}%")
                current_hedge_ratio_short = long_pos_qty / short_pos_qty if short_pos_qty > 0 else 0
                logging.info(f"Current hedge ratio short for {symbol}: {current_hedge_ratio_short:.2f}")

                if price_diff_percentage_short >= price_difference_threshold and current_hedge_ratio_short < hedge_ratio:
                    logging.info(f"Auto hedging for short position for {symbol}")
                    additional_hedge_needed_short = (short_pos_qty * hedge_ratio) - long_pos_qty
                    logging.info(f"Additional hedge needed short for {symbol}: {additional_hedge_needed_short}")

                    if additional_hedge_needed_short > min_order_size:
                        logging.info(f"Placing auto-hedge buy order for {symbol}: Amount: {additional_hedge_needed_short}, Price: {best_bid_price}")
                        order_response = self.place_postonly_order_bybit(symbol, "buy", additional_hedge_needed_short, best_bid_price, positionIdx=1, reduceOnly=False)
                        logging.info(f"Order response for {symbol} (Short Auto-Hedge): {order_response}")

            # Long Entry based on trend and MFI
            trend_aligned_long = (eri_trend == "bullish" or trend.lower() == "long")
            mfi_signal_long = mfi.lower() == "long"
            if (should_long or should_add_to_long) and current_price >= qfl_base and trend_aligned_long and mfi_signal_long:
                self.process_long_entry_qs(symbol, long_pos_qty, open_orders, long_dynamic_amount, current_price, long_pos_price)

            # Short Entry based on trend and MFI
            trend_aligned_short = (eri_trend == "bearish" or trend.lower() == "short")
            mfi_signal_short = mfi.lower() == "short"
            if (should_short or should_add_to_short) and current_price <= qfl_ceiling and trend_aligned_short and mfi_signal_short:
                self.process_short_entry_qs(symbol, short_pos_qty, open_orders, short_dynamic_amount, current_price, short_pos_price)

            # Order Book Wall Logic
            if largest_bid_wall and current_price < largest_bid_wall[0] and not self.entry_order_exists(open_orders, "buy"):
                if (should_long or should_add_to_long) and current_price >= qfl_base:
                    logging.info(f"Placing additional long trade due to detected buy wall for {symbol}")
                    self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, largest_bid_wall[0], positionIdx=1, reduceOnly=False)

            if largest_ask_wall and current_price > largest_ask_wall[0] and not self.entry_order_exists(open_orders, "sell"):
                if (should_short or should_add_to_short) and current_price <= qfl_ceiling:
                    logging.info(f"Placing additional short trade due to detected sell wall for {symbol}")
                    self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, largest_ask_wall[0], positionIdx=2, reduceOnly=False)

        else:
            logging.info(f"Volume or distance conditions not met for {symbol}, skipping entry.")

        time.sleep(5)

    def process_long_entry_qs(self, symbol, long_pos_qty, open_orders, long_dynamic_amount, current_price, long_pos_price):
        # Logic for processing long entries
        if long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing initial long entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, current_price, positionIdx=1, reduceOnly=False)
        elif long_pos_qty > 0 and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Placing additional long entry for {symbol}")
            self.improved_m_orders(symbol, long_pos_qty, long_dynamic_amount)
            time.sleep(5)

    def process_short_entry_qs(self, symbol, short_pos_qty, open_orders, short_dynamic_amount, current_price, short_pos_price):
        # Logic for processing short entries
        if short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing initial short entry for {symbol}")
            self.place_postonly_order_bybit(symbol, "sell", short_dynamic_amount, current_price, positionIdx=2, reduceOnly=False)
        elif short_pos_qty > 0 and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Placing additional short entry for {symbol}")
            self.improved_m_orders(symbol, short_pos_qty, short_dynamic_amount)
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

    def update_quickscalp_take_profit_bybit(self, symbol, pos_qty, upnl_profit_pct, short_pos_price, long_pos_price, positionIdx, order_side, last_tp_update, max_retries=10):
        try:
            # Fetch the current open TP orders for the symbol
            long_tp_orders, short_tp_orders = self.exchange.get_open_tp_orders(symbol)

            # Calculate the original TP values using quickscalp method
            original_short_tp = self.calculate_quickscalp_short_take_profit(short_pos_price, symbol, upnl_profit_pct)
            original_long_tp = self.calculate_quickscalp_long_take_profit(long_pos_price, symbol, upnl_profit_pct)

            # Fetch the current best bid and ask prices
            order_book = self.exchange.get_orderbook(symbol)
            current_best_bid = order_book['bids'][0][0] if 'bids' in order_book and order_book['bids'] else None
            current_best_ask = order_book['asks'][0][0] if 'asks' in order_book and order_book['asks'] else None

            # Determine the new TP price based on the current market price
            new_tp_price = None
            if order_side == "sell" and current_best_bid > original_long_tp:
                new_tp_price = current_best_bid
            elif order_side == "buy" and current_best_ask < original_short_tp:
                new_tp_price = current_best_ask

            # Check if there's a need to update the TP orders
            relevant_tp_orders = long_tp_orders if order_side == "sell" else short_tp_orders
            orders_to_cancel = [order for order in relevant_tp_orders if order['qty'] != pos_qty or float(order['price']) != new_tp_price]

            now = datetime.now()
            update_now = now >= last_tp_update or orders_to_cancel
            orders_updated = False  # Flag to track if orders are updated

            if update_now and new_tp_price is not None:
                # Cancel mismatched or incorrectly priced TP orders if any
                for order in orders_to_cancel:
                    try:
                        self.exchange.cancel_order_by_id(order['id'], symbol)
                        logging.info(f"Cancelled TP order {order['id']} for update.")
                        time.sleep(0.05)  # Delay to ensure orders are cancelled
                        orders_updated = True
                    except Exception as e:
                        logging.info(f"Error in cancelling {order_side} TP order {order['id']}. Error: {e}")

                # Set new TP order at the updated market price
                try:
                    self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, new_tp_price, positionIdx=positionIdx, reduce_only=True)
                    logging.info(f"New {order_side.capitalize()} TP set at {new_tp_price}")
                    orders_updated = True
                except Exception as e:
                    logging.info(f"Failed to set new {order_side} TP for {symbol}. Error: {e}")

            if orders_updated:
                # Calculate and return the next update time
                return self.calculate_next_update_time()
            else:
                # Return the last update time if no orders were updated
                return last_tp_update
        except Exception as e:
            logging.info(f"Exception caught in update TP: {e}")
            return last_tp_update  # Return the last update time in case of exception

    def update_mfirsi_tp(self, symbol, pos_qty, mfirsi, current_market_price, positionIdx, last_tp_update, long_upnl):
        if mfirsi.lower() != 'short' or pos_qty <= 0 or long_upnl <= 0:
            logging.info(f"No update needed for TP for {symbol} as mfirsi is not 'short', no open long position, or position not in profit.")
            return last_tp_update

        # Fetch current open TP orders for the symbol
        long_tp_orders, _ = self.exchange.get_open_tp_orders(symbol)

        # Check if there's an existing TP order with a mismatched quantity or price
        mismatched_qty_orders = [order for order in long_tp_orders if order['qty'] != pos_qty or order['price'] != current_market_price]

        # Cancel mismatched TP orders if any
        for order in mismatched_qty_orders:
            try:
                self.exchange.cancel_order_by_id(order['id'], symbol)
                logging.info(f"Cancelled TP order {order['id']} for update.")
                time.sleep(0.05)
            except Exception as e:
                logging.info(f"Error in cancelling TP order {order['id']}. Error: {e}")

        now = datetime.now()
        if now >= last_tp_update or mismatched_qty_orders:
            # Place a new TP order with the current market price
            try:
                self.exchange.create_take_profit_order_bybit(symbol, "limit", "sell", pos_qty, current_market_price, positionIdx=positionIdx, reduce_only=True)
                logging.info(f"New sell TP set at current market price {current_market_price} for {symbol}")
            except Exception as e:
                logging.info(f"Failed to set new sell TP for {symbol}. Error: {e}")

            return datetime.now()
        else:
            logging.info(f"No immediate update needed for TP orders for {symbol}. Last update at: {last_tp_update}")
            return last_tp_update

    def update_take_profit_spread_bybit(self, symbol, pos_qty, short_take_profit, long_take_profit, short_pos_price, long_pos_price, positionIdx, order_side, next_tp_update, five_minute_distance, previous_five_minute_distance, tp_order_counts, max_retries=10):
        # Fetch the current open TP orders and TP order counts for the symbol
        long_tp_orders, short_tp_orders = self.exchange.get_open_tp_orders(symbol)
        #tp_order_counts = self.exchange.get_open_tp_order_count(symbol)
        long_tp_count = tp_order_counts['long_tp_count']
        short_tp_count = tp_order_counts['short_tp_count']

        # Calculate the TP values based on the current spread
        new_short_tp, new_long_tp = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)

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
        if now >= next_tp_update or mismatched_qty_orders:
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
            logging.info(f"Waiting for the next update time for TP orders.")
            return next_tp_update

    def is_hedge_order(self, symbol, order_side):
        hedge_info = self.hedged_positions.get(symbol)
        return hedge_info and hedge_info['type'] == order_side

    def mark_hedge_as_completed(self, symbol, order_side):
        if self.is_hedge_order(symbol, order_side):
            del self.hedged_positions[symbol]  # Remove the hedge flag as the hedge is completed

    def is_hedged_position(self, symbol):
        return symbol in self.hedged_positions

    def mark_hedge_tp_order(self, symbol, tp_order, order_side):
        if tp_order and 'id' in tp_order:
            # Storing order_side along with the TP order ID
            self.hedged_positions[symbol]['tp_order'] = {
                'id': tp_order['id'],
                'side': order_side
            }
            logging.info(f"Hedged TP order (side: {order_side}) placed for {symbol}, ID: {tp_order['id']}")
        else:
            logging.warning(f"Failed to mark TP order as hedge for {symbol}")


    # def mark_hedge_tp_order(self, symbol, tp_order, order_side):
    #     if tp_order and 'id' in tp_order:
    #         self.hedged_positions[symbol]['tp_order_id'] = tp_order['id']
    #         logging.info(f"Hedged TP order placed for {symbol}, ID: {tp_order['id']}")
    #     else:
    #         logging.warning(f"Failed to mark TP order as hedge for {symbol}")

    def bybit_hedge_placetp_maker_v2(self, symbol, pos_qty, take_profit_price, positionIdx, order_side, open_orders):
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
                tp_order = self.postonly_limit_order_bybit_nolimit(symbol, order_side, pos_qty, take_profit_price, positionIdx, reduceOnly=True)
                logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price}")

                # Mark the TP order for hedged positions
                if self.is_hedged_position(symbol):
                    self.mark_hedge_tp_order(symbol, tp_order, order_side)

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
                logging.info(f"Expected max_trade_qty to be float, got {type(max_trade_qty)}")
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
                logging.info(f"Expected max_trade_qty to be float, got {type(max_trade_qty)}")
            self.short_leverage_increased = False
            self.short_pos_leverage = 1.0
            logging.info(f"Short leverage returned to normal {self.short_pos_leverage}x")