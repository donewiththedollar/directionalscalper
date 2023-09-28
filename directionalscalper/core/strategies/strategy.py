from colorama import Fore
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
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
from .logger import Logger
from datetime import datetime, timedelta

logging = Logger(logger_name="Strategy", filename="Strategy.log", stream=True)

class Strategy:
    def __init__(self, exchange, config, manager, symbols_allowed=None):
    # def __init__(self, exchange, config, manager):
        self.exchange = exchange
        self.config = config
        self.manager = manager
        self.symbol = config.symbol
        self.symbols_allowed = symbols_allowed
        self.order_timestamps = {}
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
        self.MAX_LEVERAGE = 0.1 #0.3  # The maximum allowable leverage
        self.QTY_INCREMENT = 0.02 # How much your position size increases
        self.MAX_PCT_EQUITY = 0.5
        self.ORDER_BOOK_DEPTH = 10

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
        # Check if the symbol's trade quantities have been initialized
        if symbol not in self.max_long_trade_qty_per_symbol or symbol not in self.max_short_trade_qty_per_symbol:
            max_trade_qty = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
            
            self.max_long_trade_qty_per_symbol[symbol] = max_trade_qty
            self.max_short_trade_qty_per_symbol[symbol] = max_trade_qty
            
            logging.info(f"For symbol {symbol} Calculated max_long_trade_qty: {self.max_long_trade_qty_per_symbol[symbol]}, max_short_trade_qty: {self.max_short_trade_qty_per_symbol[symbol]}")

        # Check and set the initial max trade quantities if not set
        if symbol not in self.initial_max_long_trade_qty_per_symbol:
            self.initial_max_long_trade_qty_per_symbol[symbol] = self.max_long_trade_qty_per_symbol[symbol]
            logging.info(f"Initial max long trade qty set for {symbol} to {self.initial_max_long_trade_qty_per_symbol[symbol]}")

        if symbol not in self.initial_max_short_trade_qty_per_symbol:
            self.initial_max_short_trade_qty_per_symbol[symbol] = self.max_short_trade_qty_per_symbol[symbol]
            logging.info(f"Initial max short trade qty set for {symbol} to {self.initial_max_short_trade_qty_per_symbol[symbol]}")

    def calculate_dynamic_amount(self, symbol, market_data, total_equity, best_ask_price, max_leverage):

        self.initialize_trade_quantities(symbol, total_equity, best_ask_price, max_leverage)

        long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty_per_symbol[symbol]
        short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty_per_symbol[symbol]

        logging.info(f"Initial long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        # Cap the dynamic amount if it exceeds the maximum allowed
        max_allowed_dynamic_amount = (self.MAX_PCT_EQUITY / 100) * total_equity

        logging.info(f"Max allowed dynamic amount for {symbol} : {max_allowed_dynamic_amount}")

        min_qty = float(market_data["min_qty"])
        min_qty_str = str(min_qty)

        logging.info(f"min_qty: {min_qty}, min_qty_str: {min_qty_str}")

        logging.info(f"Original min_qty: {min_qty}")

        if ".0" in min_qty_str:
            precision_level = 0
        else:
            precision_level = len(min_qty_str.split(".")[1])

        logging.info(f"Calculated precision_level: {precision_level}")

        logging.info(f"Original long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        if precision_level > 0:
            long_dynamic_amount = Decimal(str(long_dynamic_amount)).quantize(Decimal('1e-{0}'.format(precision_level)), rounding=ROUND_HALF_UP)
            short_dynamic_amount = Decimal(str(short_dynamic_amount)).quantize(Decimal('1e-{0}'.format(precision_level)), rounding=ROUND_HALF_UP)
        else:
            long_dynamic_amount = int(Decimal(str(long_dynamic_amount)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            short_dynamic_amount = int(Decimal(str(short_dynamic_amount)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            
        logging.info(f"Rounded long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        long_dynamic_amount = min(long_dynamic_amount, max_allowed_dynamic_amount)
        short_dynamic_amount = min(short_dynamic_amount, max_allowed_dynamic_amount)

        logging.info(f"Forced min qty long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)
        self.check_amount_validity_once_bybit(short_dynamic_amount, symbol)

        if long_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for long_dynamic_amount")
            long_dynamic_amount = min_qty

        if short_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty for short_dynamic_amount")
            short_dynamic_amount = min_qty

        logging.info(f"Final long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

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

    def limit_order_bybit_unified(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit_unified(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

    def postonly_limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        # Check if we can place an order for the given symbol
        now = time.time()  # Current timestamp
        last_two_orders = self.order_timestamps.get(symbol, [])

        logging.info(f"Current timestamps for {symbol}: {last_two_orders}")  # Log the current timestamps for debugging

        if len(last_two_orders) >= 2 and (now - last_two_orders[0]) <= 300:  # If two orders were placed in the last 5 minutes
            logging.warning(f"Cannot place more than 2 orders per 5 minutes for {symbol}. Skipping order placement...")
            return None

        params = {"reduceOnly": reduceOnly, "postOnly": True}
        logging.info(f"Placing {side} limit order for {symbol} at {price} with qty {amount} and params {params}...")
        try:
            order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
            logging.info(f"Order result: {order}")
            
            # If order placement is successful, update the order timestamps
            if order:
                last_two_orders.append(now)  # Add the current timestamp
                if len(last_two_orders) > 2:  # Keep only the last two timestamps
                    last_two_orders.pop(0)
                self.order_timestamps[symbol] = last_two_orders
                logging.info(f"Updated timestamps for {symbol}: {last_two_orders}")  # Log the updated timestamps for debugging

            if order is None:
                logging.warning(f"Order result is None for {side} limit order on {symbol}")
            return order
        except Exception as e:
            logging.error(f"Error placing order: {str(e)}")
            logging.exception("Stack trace for error in placing order:")  # This will log the full stack trace

    # def postonly_limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
    #     # Check if we can place an order for the given symbol
    #     now = time.time()  # Current timestamp
    #     last_two_orders = self.order_timestamps.get(symbol, [])

    #     if len(last_two_orders) >= 2 and (now - last_two_orders[0]) <= 60:  # If two orders were placed in the last minute
    #         logging.warning(f"Cannot place more than 2 orders per minute for {symbol}. Skipping order placement...")
    #         return None

    #     params = {"reduceOnly": reduceOnly, "postOnly": True}
    #     logging.info(f"Placing {side} limit order for {symbol} at {price} with qty {amount} and params {params}...")
    #     try:
    #         order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
    #         logging.info(f"Order result: {order}")
            
    #         # If order placement is successful, update the order timestamps
    #         if order:
    #             last_two_orders.append(now)  # Add the current timestamp
    #             if len(last_two_orders) > 2:  # Keep only the last two timestamps
    #                 last_two_orders.pop(0)
    #             self.order_timestamps[symbol] = last_two_orders

    #         if order is None:
    #             logging.warning(f"Order result is None for {side} limit order on {symbol}")
    #         return order
    #     except Exception as e:
    #         logging.error(f"Error placing order: {str(e)}")
    #         logging.exception("Stack trace for error in placing order:")  # This will log the full stack trace

    def postonly_limit_order_bybit_nolimit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
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

    # def entry_order_exists(self, open_orders: list, side: str, symbol: str) -> bool:
    #     for order in open_orders:
    #         print(f"Open orders test {open_orders}")
    #         print(f"Open order test {order}")
    #         if order["symbol"] == symbol and order["side"].lower() == side and order["reduce_only"] == False:
    #             logging.info(f"An entry order for symbol {symbol} and side {side} already exists.")
    #             return True
    #     logging.info(f"No entry order found for symbol {symbol} and side {side}.")
    #     return False

    def entry_order_exists(self, open_orders: list, side: str) -> bool:
        for order in open_orders:
            if order["side"].lower() == side and order["reduce_only"] == False:
                logging.info(f"An entry order for side {side} already exists.")
                return True
        logging.info(f"No entry order found for side {side}.")
        return False
    
    # def entry_order_exists(self, open_orders: list, side: str) -> bool:
    #     for order in open_orders:
    #         if order["side"].lower() == side and order["reduce_only"] == False:
    #             return True
    #     return False
    
    def get_open_take_profit_order_quantity(self, orders, side):
        for order in orders:
            if order['side'].lower() == side.lower() and order['reduce_only']:
                return order['qty'], order['id']
        return None, None

    def get_open_take_profit_order_quantities(self, orders, side):
        take_profit_orders = []
        for order in orders:
            order_side = order.get('side')
            if order_side and isinstance(order_side, str) and order_side.lower() == side.lower() and order['reduce_only']:
                take_profit_orders.append((order['qty'], order['id']))
        return take_profit_orders

    # def get_open_take_profit_order_quantities(self, orders, side):
    #     take_profit_orders = []
    #     for order in orders:
    #         if order['side'].lower() == side.lower() and order['reduce_only']:
    #             take_profit_orders.append((order['qty'], order['id']))
    #     return take_profit_orders

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

    #             # Cap the max_trade_qty as a percentage of total_equity
    #             max_allowed_trade_qty = (self.MAX_PCT_EQUITY / 100) * total_equity
    #             max_trade_qty = min(max_trade_qty, max_allowed_trade_qty)
                
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

    def print_trade_quantities_once_bybit(self, symbol, max_trade_qty):
        if not self.printed_trade_quantities:
            wallet_exposure = self.config.wallet_exposure
            best_ask_price = self.exchange.get_orderbook(self.symbol)['asks'][0][0]
            #self.exchange.print_trade_quantities_bybit(max_trade_qty, [0.001, 0.01, 0.1, 1, 2.5, 5], wallet_exposure, best_ask_price)
            self.exchange.print_trade_quantities_bybit(self.max_long_trade_qty_per_symbol[symbol], [0.001, 0.01, 0.1, 1, 2.5, 5], wallet_exposure, best_ask_price)
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

    def get_positions_bybit(self):
        position_data = self.exchange.get_positions_bybit(self.symbol)
        return position_data

    # def calculate_next_update_time(self):
    #     # 5 min interval calc
    #     now = datetime.now()
    #     next_update_minute = (now.minute // 5 + 1) * 5
    #     if next_update_minute == 60:
    #         next_update_minute = 0
    #         now += timedelta(hours=1)
    #     return now.replace(minute=next_update_minute, second=0, microsecond=0)

    def calculate_next_update_time(self):
        # 6 min 19 seconds interval calc
        now = datetime.now()
        next_update_time = now + timedelta(minutes=1, seconds=19)
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
        if five_minute_distance != previous_five_minute_distance or short_take_profit is None or long_take_profit is None:
            short_take_profit = self.calculate_short_take_profit_spread_bybit(short_pos_price, symbol, five_minute_distance)
            long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, five_minute_distance)
        
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

    # def should_long_MFI(self, symbol):
    #     df = self.initialize_MFIRSI(symbol)
    #     return df.iloc[-1]['buy_condition'] == 1

    # def should_short_MFI(self, symbol):
    #     df = self.initialize_MFIRSI(symbol)
    #     return df.iloc[-1]['sell_condition'] == 1

    def parse_contract_code(self, symbol):
        parsed_symbol = symbol.split(':')[0]  # Remove ':USDT'
        parsed_symbol = parsed_symbol.replace('/', '-')  # Replace '/' with '-'
        return parsed_symbol

    def extract_symbols_from_positions_bybit(self, positions):
        symbols = [position['symbol'].split(':')[0] for position in positions]
        return list(set(symbols))

    def retry_api_call(self, function, *args, max_retries=5, delay=5, **kwargs):
        for i in range(max_retries):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                logging.info(f"Error occurred during API call: {e}. Retrying in {delay} seconds...")
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
        if self.open_symbols_count >= symbols_allowed:
            return False  # This restricts opening new positions if we have reached the symbols_allowed limit
        elif current_symbol in open_symbols:
            return True  # This allows new positions on already traded symbols
        else:
            return self.open_symbols_count < symbols_allowed  # This checks if we can trade a new symbol

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
       
    def spoofing_action(self, symbol, short_dynamic_amount, long_dynamic_amount):
        if self.spoofing_active:
            # Fetch orderbook and positions
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = Decimal(orderbook['bids'][0][0])
            best_ask_price = Decimal(orderbook['asks'][0][0])
            current_price = (best_bid_price + best_ask_price) / Decimal('2')  # Convert to Decimal

            position_data = self.exchange.get_positions_bybit(symbol)
            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            if short_pos_qty is None and long_pos_qty is None:
                logging.warning(f"Could not fetch position quantities for {symbol}. Skipping spoofing.")
                return

            # Initialize variables
            spoofing_orders = []
            larger_position = "long" if long_pos_qty > short_pos_qty else "short"
            safety_margin = Decimal('0.05')  # 1% safety margin
            base_gap = Decimal('0.005')  # Base gap for spoofing orders

            for i in range(self.spoofing_wall_size):
                gap = base_gap + Decimal(i) * Decimal('0.002')  # Increasing gap for each subsequent order

                # Calculate long spoof price based on best ask price (top of the order book)
                spoof_price_long = best_ask_price + gap + safety_margin
                spoof_price_long = spoof_price_long.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                
                if larger_position == "long":
                    spoof_price_long = spoof_price_long + current_price * Decimal('0.005')#('0.005')  # Adjust price in the direction of larger position
                
                spoof_order_long = self.limit_order_bybit(symbol, "sell", long_dynamic_amount, spoof_price_long, positionIdx=2, reduceOnly=False)
                spoofing_orders.append(spoof_order_long)

                # Calculate short spoof price based on best bid price (top of the order book)
                spoof_price_short = best_bid_price - gap - safety_margin
                spoof_price_short = spoof_price_short.quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
                
                if larger_position == "short":
                    spoof_price_short = spoof_price_short - current_price * Decimal('0.005')#('0.005')  # Adjust price in the direction of larger position
                
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

    def bybit_hedge_spoof_v1(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
                
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                if self.should_spoof:
                    self.spoofing_action(symbol, long_dynamic_amount, short_dynamic_amount)

                if self.should_place_spoof_scalping_orders:
                    self.spoof_scalping_strategy(
                        symbol, trend, mfi, one_minute_volume, five_minute_distance,
                        min_vol, min_dist, long_dynamic_amount, short_dynamic_amount,
                        long_pos_qty, short_pos_qty, long_pos_price, short_pos_price,
                        should_long, should_short, should_add_to_long, should_add_to_short
                    )

                if (trend.lower() == "long" and mfi.lower() == "long") and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                elif (trend.lower() == "long" and mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                if (trend.lower() == "short" and mfi.lower() == "short") and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry")

                elif (trend.lower() == "short" and mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)


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

    def bybit_hedge_entry_maker_v4(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):
        self.order_book_analyzer = self.OrderBookAnalyzer(self.exchange, symbol)
        order_book = self.order_book_analyzer.get_order_book()
        best_ask_price = order_book['asks'][0][0]
        best_bid_price = order_book['bids'][0][0]
        
        market_data = self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)
        min_qty = float(market_data["min_qty"])

        # Front-running strategy
        largest_bid = max(order_book['bids'], key=lambda x: x[1])
        largest_ask = min(order_book['asks'], key=lambda x: x[1])
        front_run_bid_price = largest_bid[0] + min_qty
        front_run_ask_price = largest_ask[0] - min_qty

        # Pressure Analysis Strategy
        if self.order_book_analyzer.buying_pressure():
            logging.info(f"Detected buying pressure!")
            # Modify the amount for aggressive buying based on buying pressure
            long_dynamic_amount *= 1.5
        elif self.order_book_analyzer.selling_pressure():
            logging.info(f"Detected selling pressure!")
            # Modify the amount for aggressive selling based on selling pressure
            short_dynamic_amount *= 1.5

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                if (trend.lower() == "long" or mfi.lower() == "long") and should_long and long_pos_qty == 0:
                    logging.info(f"Placing aggressive initial long entry using front-running strategy")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed aggressive initial long entry")
                else:
                    if (trend.lower() == "long" or mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price:
                        logging.info(f"Placing aggressive additional long entry using front-running strategy")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)

                if (trend.lower() == "short" or mfi.lower() == "short") and should_short and short_pos_qty == 0:
                    logging.info(f"Placing aggressive initial short entry using front-running strategy")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info("Placed aggressive initial short entry")
                else:
                    if (trend.lower() == "short" or mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price:
                        logging.info(f"Placing aggressive additional short entry using front-running strategy")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_entry_maker_v2_initial_entry(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, best_bid_price: float, best_ask_price: float, should_long: bool, should_short: bool):

        if one_minute_volume > min_vol and five_minute_distance > min_dist:
            open_orders = self.exchange.get_open_orders(symbol)
            
            if (trend.lower() == "long" or mfi.lower() == "long") and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                logging.info(f"Placing initial long entry")
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Placed initial long entry")

            if (trend.lower() == "short" or mfi.lower() == "short") and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                logging.info(f"Placing initial short entry")
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                logging.info("Placed initial short entry")

    def bybit_hedge_entry_maker_v2_additional_entry(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, best_bid_price: float, best_ask_price: float, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume > min_vol and five_minute_distance > min_dist:

            open_orders = self.exchange.get_open_orders(symbol)

            if (trend.lower() == "long" or mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                logging.info(f"Placing additional long entry")
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            if (trend.lower() == "short" or mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                logging.info(f"Placing additional short entry")
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            # Additional logic to check if we should initiate a new position on the other side
            if long_pos_qty == 0: # No existing long position
                if (trend.lower() == "long" or mfi.lower() == "long") and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Initiating new long position")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            if short_pos_qty == 0: # No existing short position
                if (trend.lower() == "short" or mfi.lower() == "short") and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Initiating new short position")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_entry_maker_v3_initial_entry(self, open_orders: list, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if trend is None or mfi is None:
            logging.warning(f"Either 'trend' or 'mfi' is None for symbol {symbol}. Skipping current execution...")
            return

        if one_minute_volume > min_vol and five_minute_distance > min_dist:

            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

            # Initial long entry
            if trend.lower() == "long" and mfi.lower() == "long" and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                if long_pos_qty <= self.max_long_trade_qty_per_symbol[symbol]:  # Check against max trade qty for the symbol
                    logging.info(f"Placing initial long entry for {symbol}")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                else:
                    logging.warning(f"Long position quantity for {symbol} exceeds maximum trade quantity. No trade placed.")

            # Initial short entry
            if trend.lower() == "short" and mfi.lower() == "short" and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                if short_pos_qty <= self.max_short_trade_qty_per_symbol[symbol]:  # Check against max trade qty for the symbol
                    logging.info(f"Placing initial short entry for {symbol}")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                else:
                    logging.warning(f"Short position quantity for {symbol} exceeds maximum trade quantity. No trade placed.")

    def bybit_hedge_additional_entry_maker_v3(self, open_orders: list, symbol: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):
        
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        #open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

        if should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Managing non-rotating symbol: Placing additional long entry for {symbol}")
            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        if should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Managing non-rotating symbol: Placing additional short entry for {symbol}")
            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

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
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            # Additional long entry
            elif trend.lower() == "long" and mfi.lower() == "long" and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                logging.info(f"Placing additional long entry for {symbol}")
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            # Initial short entry
            if trend.lower() == "short" and mfi.lower() == "short" and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
                logging.info(f"Placing initial short entry for {symbol}")
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

            # Additional short entry
            elif trend.lower() == "short" and mfi.lower() == "short" and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                logging.info(f"Placing additional short entry for {symbol}")
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

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

    def bybit_entry_mm_5m(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, five_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        logging.info(f"Hedge entry maker hma function hit")

        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        logging.info(f"Trend is {trend}")
        logging.info(f"MFI is {mfi}")
        logging.info(f"HMA is {hma_trend}")

        logging.info(f"Five min vol for {symbol} is {five_minute_volume}")
        logging.info(f"Five min dist for {symbol} is {five_minute_distance}")

        logging.info(f"Should long for {symbol} : {should_long}")
        logging.info(f"Should short for {symbol} : {should_short}")
        logging.info(f"Should add to long for {symbol} : {should_add_to_long}")
        logging.info(f"Should add to short for {symbol} : {should_add_to_short}")

        logging.info(f"Min dist: {min_dist}")
        logging.info(f"Min vol: {min_vol}")

        if five_minute_volume is not None and five_minute_distance is not None:
            if five_minute_volume > min_vol and five_minute_distance > min_dist:

                logging.info(f"Made it into the entry maker function for {symbol}")

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Before entering long
                logging.info(f"Checking for long entry conditions. max_long_trade_qty for {symbol}: {self.max_long_trade_qty_per_symbol.get(symbol, 0)}, Current long_pos_qty: {long_pos_qty}")

                # Check for long entry conditions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty_per_symbol.get(symbol, 0) and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                # Check for additional long entry conditions
                elif ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty_per_symbol.get(symbol, 0) and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Before entering short
                logging.info(f"Checking for short entry conditions. max_short_trade_qty for {symbol}: {self.max_short_trade_qty_per_symbol.get(symbol, 0)}, Current short_pos_qty: {short_pos_qty}")

                # Check for short entry conditions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol.get(symbol, 0) and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry")

                # Check for additional short entry conditions
                elif ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty_per_symbol.get(symbol, 0) and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
            
                # # Check for long entry conditions
                # if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and long_pos_qty < self.max_long_trade_qty and not self.entry_order_exists(open_orders, "buy"):
                #     logging.info(f"Placing initial long entry")
                #     self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                #     logging.info(f"Placed initial long entry")

                # # Check for additional long entry conditions
                # elif ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                #     logging.info(f"Placing additional long entry")
                #     self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # # Check for short entry conditions
                # if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty and not self.entry_order_exists(open_orders, "sell"):
                #     logging.info(f"Placing initial short entry")
                #     self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                #     logging.info(f"Placed initial short entry")

                # # Check for additional short entry conditions
                # elif ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                #     logging.info(f"Placing additional short entry")
                #     self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_initial_entry_mm_5m(self, open_orders: list, symbol: str, trend: str, hma_trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, should_long: bool, should_short: bool):

        if trend is None or mfi is None or hma_trend is None:
            logging.warning(f"Either 'trend', 'mfi', or 'hma_trend' is None for symbol {symbol}. Skipping current execution...")
            return

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                max_long_trade_qty_for_symbol = self.max_long_trade_qty_per_symbol.get(symbol, 0)  # Get value for symbol or default to 0
                max_short_trade_qty_for_symbol = self.max_short_trade_qty_per_symbol.get(symbol, 0)  # Get value for symbol or default to 0

                # Check for long entry conditions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_long and long_pos_qty == 0 and long_pos_qty < max_long_trade_qty_for_symbol and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                # Check for short entry conditions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and short_pos_qty < max_short_trade_qty_for_symbol and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry")

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
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        # Check for additional short entry conditions
        if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short:
            if short_pos_qty < max_short_trade_qty_for_symbol and best_ask_price > short_pos_price:
                if not self.entry_order_exists(open_orders, "sell"):
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

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
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                # Check for short entry conditions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_short and short_pos_qty == 0 and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
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

                # Check for additional long entry conditions
                if ((trend.lower() == "long" or hma_trend.lower() == "long") and mfi.lower() == "long") and should_add_to_long:
                    
                    if symbol in self.max_long_trade_qty_per_symbol and long_pos_qty >= self.max_long_trade_qty_per_symbol[symbol]:
                        logging.warning(f"Reached or exceeded max long trade qty for symbol: {symbol}. Current qty: {long_pos_qty}, Max allowed qty: {self.max_long_trade_qty_per_symbol[symbol]}. Skipping additional long entry.")
                    
                    elif best_bid_price < long_pos_price:
                        
                        if self.entry_order_exists(open_orders, "buy"):
                            logging.warning(f"Already have a buy order for symbol: {symbol}. Skipping additional long entry.")
                            for order in open_orders:
                                if order["symbol"] == symbol and order["side"] == "buy":
                                    logging.info(f"Existing buy order details: {order}")
                        else:
                            logging.info(f"Placing additional long entry")
                            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                # Check for additional short entry conditions
                if ((trend.lower() == "short" or hma_trend.lower() == "short") and mfi.lower() == "short") and should_add_to_short:
                    
                    if symbol in self.max_short_trade_qty_per_symbol and short_pos_qty >= self.max_short_trade_qty_per_symbol[symbol]:
                        logging.warning(f"Reached or exceeded max short trade qty for symbol: {symbol}. Current qty: {short_pos_qty}, Max allowed qty: {self.max_short_trade_qty_per_symbol[symbol]}. Skipping additional short entry.")
                    
                    elif best_ask_price > short_pos_price:
                        
                        if self.entry_order_exists(open_orders, "sell"):
                            logging.warning(f"Already have a sell order for symbol: {symbol}. Skipping additional short entry.")
                            for order in open_orders:
                                if order["symbol"] == symbol and order["side"] == "sell":
                                    logging.info(f"Existing sell order details: {order}")
                        else:
                            logging.info(f"Placing additional short entry")
                            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

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

    def update_take_profit_spread_bybit(self, symbol, pos_qty, take_profit_price, positionIdx, order_side, open_orders, next_tp_update, max_retries=10):
        existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
        total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
        logging.info(f"Existing {order_side} TPs: {existing_tps}")
        now = datetime.now()
        if now >= next_tp_update or not math.isclose(total_existing_tp_qty, pos_qty):
            try:
                logging.info(f"Next TP updating for {symbol} : {next_tp_update}")
                # Cancel the existing TP orders only when the time condition is met
                for qty, existing_tp_id in existing_tps:
                    self.exchange.cancel_order_by_id(existing_tp_id, symbol)
                    logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                    time.sleep(0.05)
                
                retries = 0
                success = False
                while retries < max_retries and not success:
                    try:
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, take_profit_price, positionIdx=positionIdx, reduce_only=True)
                        logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price}")
                        success = True
                    except Exception as e:
                        logging.error(f"Failed to set {order_side} TP. Retry {retries + 1}/{max_retries}. Error: {e}")
                        retries += 1
                        time.sleep(1)  # Wait for a moment before retrying
                
                next_tp_update = self.calculate_next_update_time()  # Calculate the next update time after placing the order
            except Exception as e:
                logging.info(f"Error in updating {order_side} TP: {e}")
        return next_tp_update

    # def update_take_profit_spread_bybit(self, symbol, pos_qty, take_profit_price, positionIdx, order_side, open_orders, next_tp_update):
    #     existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
    #     total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
    #     logging.info(f"Existing {order_side} TPs: {existing_tps}")
    #     #logging.info(f"Next TP update for {symbol} : {next_tp_update}")
    #     now = datetime.now()
    #     if now >= next_tp_update or not math.isclose(total_existing_tp_qty, pos_qty):
    #         try:
    #             logging.info(f"Next TP updating for {symbol} : {next_tp_update}")
    #             # Cancel the existing TP orders only when the time condition is met
    #             for qty, existing_tp_id in existing_tps:
    #                 self.exchange.cancel_order_by_id(existing_tp_id, symbol)
    #                 logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
    #                 time.sleep(0.05)
                
    #             self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, take_profit_price, positionIdx=positionIdx, reduce_only=True)
    #             logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price}")
    #             next_tp_update = self.calculate_next_update_time()  # Calculate the next update time after placing the order
    #         except Exception as e:
    #             logging.info(f"Error in updating {order_side} TP: {e}")
    #     return next_tp_update

# Bybit take profit placement based on 5 minute spread

    def bybit_hedge_placetp_maker(self, symbol, pos_qty, take_profit_price, positionIdx, order_side, open_orders):
        existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
        total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
        logging.info(f"Existing {order_side} TPs: {existing_tps}")
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

    def long_entry_maker_gs(self, symbol: str, trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, long_pos_qty: float, long_pos_price: float, should_add_to_long: bool):
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        
        if trend is not None and isinstance(trend, str) and trend.lower() == "long":
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                # Only placing additional long entries in GS mode
                if should_add_to_long and long_pos_qty < self.max_long_trade_qty and long_pos_price is not None:
                    if best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol} in GS mode")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

    def short_entry_maker_gs(self, symbol: str, trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, short_dynamic_amount: float, short_pos_qty: float, short_pos_price: float, should_add_to_short: bool):
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        
        if trend is not None and isinstance(trend, str) and trend.lower() == "short":
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)
                # Only placing additional short entries in GS mode
                if should_add_to_short and short_pos_qty < self.max_short_trade_qty and short_pos_price is not None:
                    if best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                        logging.info(f"Placing additional short entry for {symbol} in GS mode")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def long_entry_maker_gs_mfi(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, long_pos_qty: float, long_pos_price: float, should_add_to_long: bool):
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        
        if one_minute_volume > min_vol and five_minute_distance > min_dist:
            if (trend.lower() == "long" or mfi.lower() == "long") and should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price:
                logging.info(f"Placing additional long entry for {symbol} in GS mode using MFI signals")
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

    def short_entry_maker_gs_mfi(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, short_dynamic_amount: float, short_pos_qty: float, short_pos_price: float, should_add_to_short: bool):
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        
        if one_minute_volume > min_vol and five_minute_distance > min_dist:
            if (trend.lower() == "short" or mfi.lower() == "short") and should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price:
                logging.info(f"Placing additional short entry for {symbol} in GS mode using MFI signals")
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def manage_open_positions_aggressive(self, open_symbols, total_equity):
        # Get current rotator symbols
        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=self.whitelist, blacklist=self.blacklist, max_usd_value=self.max_usd_value)
        
        logging.info(f"open_symbols in manage_open_positions: {open_symbols}")
        for open_symbol in open_symbols:
            #logging.info(f"Type of open_symbols: {type(open_symbols)}")
            # Check if the open symbol is NOT in the current rotator symbols
            is_rotator_symbol = open_symbol in rotator_symbols

            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")
                
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            one_minute_distance = api_data['1mSpread']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']

            # Your existing code for each open_symbol
            # Fetch position data for the open symbol
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            min_qty = float(market_data["min_qty"])
            min_qty_str = str(min_qty)


            # Fetch the best ask and bid prices for the open symbol
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
            # Calculate the max trade quantities dynamically for this specific symbol
            # self.initial_max_long_trade_qty, self.initial_max_short_trade_qty = self.calc_max_trade_qty_multi(
            #     total_equity, best_ask_price_open_symbol, max_leverage)

            max_trade_qty_value = self.calc_max_trade_qty(open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            self.initial_max_long_trade_qty = max_trade_qty_value
            self.initial_max_short_trade_qty = max_trade_qty_value

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

            # # Calculate your take profit levels for each open symbol - with avoiding fees
            # short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["short"]["price"], short_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )
            # long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["long"]["price"], long_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )

            # Calculate your take profit levels for each open symbol.
            short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
                position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            )
            long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
                position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            )

            # Additional context-specific variables
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

            # Additional context-specific variables
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False

            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

            # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
            # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
            
            # Fetch open orders for the open symbol
            open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)

            all_open_orders = self.exchange.get_all_open_orders_bybit()

            logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")
            #print(f"All open orders {all_open_orders}")

            #print(f"Open orders open symbol {open_orders_open_symbol}")

            # Check for existing take profit orders before placing new ones
            # Existing long take profit orders
            existing_long_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'sell'
            ]

            # Existing short take profit orders
            existing_short_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'buy'
            ]

            # Call the function to update long take profit spread only if no existing take profit order
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None and not existing_long_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol
                )

            # Call the function to update short take profit spread only if no existing take profit order
            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None and not existing_short_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol
                )

            # Take profit spread replacement
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None:
                self.next_long_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update
                )

            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None:
                self.next_short_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update
                )

            if long_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if open_symbol in open_symbols:
                # Note: When calling the `bybit_turbocharged_entry_maker` function, make sure to use these updated, context-specific variables.
                if is_rotator_symbol: 
                    self.bybit_turbocharged_entry_maker(
                        open_orders_open_symbol,
                        open_symbol,
                        trend,
                        mfirsi_signal,
                        one_minute_volume,
                        five_minute_distance,
                        min_vol,
                        min_dist,
                        long_take_profit_open_symbol,
                        short_take_profit_open_symbol,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_long_open_symbol,
                        should_add_to_long_open_symbol,
                        should_short_open_symbol,
                        should_add_to_short_open_symbol
                    )
                else:
                    self.bybit_turbocharged_additional_entry_maker(
                        open_orders_open_symbol,
                        open_symbol,
                        trend,
                        mfirsi_signal,
                        one_minute_volume,
                        five_minute_distance,
                        min_dist,
                        min_vol,
                        long_take_profit_open_symbol,
                        short_take_profit_open_symbol,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

                self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

            # Cancel entries (Note: Replace this with the actual conditions for your open_symbol)
            #self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

    def gnifoops(self, open_symbols, total_equity):
        # Get current rotator symbols

        max_usd_value = self.config.max_usd_value
        whitelist = self.config.whitelist
        blacklist = self.config.blacklist

        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=whitelist, blacklist=blacklist, max_usd_value=max_usd_value)

        logging.info(f"open_symbols in manage_open_positions: {open_symbols}")
        for open_symbol in open_symbols:
            is_rotator_symbol = open_symbol in rotator_symbols

            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")
                
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            one_minute_distance = api_data['1mSpread']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']

            # Your existing code for each open_symbol
            # Fetch position data for the open symbol
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            min_qty = float(market_data["min_qty"])
            min_qty_str = str(min_qty)


            # Fetch the best ask and bid prices for the open symbol
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
            # Calculate the max trade quantities dynamically for this specific symbol
            max_trade_qty_value = self.calc_max_trade_qty(open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            self.initial_max_long_trade_qty = max_trade_qty_value
            self.initial_max_short_trade_qty = max_trade_qty_value

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

            # # Calculate your take profit levels for each open symbol - with avoiding fees
            # short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["short"]["price"], short_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )
            # long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["long"]["price"], long_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )

            # Calculate your take profit levels for each open symbol.
            short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
                position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            )
            long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
                position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            )

            # Additional context-specific variables
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

            # Additional context-specific variables
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False

            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

            # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
            # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
            
            # # Calculate your take profit levels for each open symbol.
            # short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
            #     position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            # )
            # long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
            #     position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            # )

            # Fetch open orders for the open symbol
            open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)

            all_open_orders = self.exchange.get_all_open_orders_bybit()

            logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")
            #print(f"All open orders {all_open_orders}")

            #print(f"Open orders open symbol {open_orders_open_symbol}")

            # Check for existing take profit orders before placing new ones
            # Existing long take profit orders
            existing_long_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'sell'
            ]

            # Existing short take profit orders
            existing_short_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'buy'
            ]

            # Call the function to update long take profit spread only if no existing take profit order
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None and not existing_long_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol
                )

            # Call the function to update short take profit spread only if no existing take profit order
            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None and not existing_short_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol
                )

            # Take profit spread replacement
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None:
                self.next_long_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update
                )

            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None:
                self.next_short_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update
                )

            if long_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if open_symbol in open_symbols:

                print(f"Symbols allowed from strategy file {self.symbols_allowed}")

                can_trade = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol)

                logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                
                # Check if the symbol is a rotator symbol
                if is_rotator_symbol:  

                    logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                    
                    # Add the following line before the initial entry trading logic to check symbol limits
                    if not self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol):
                        logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                        logging.info(f"Reached the symbol limit or already trading {open_symbol}. Skipping initial entry.")
                        
                    else:  # If we can trade, then proceed with initial entry
                        self.bybit_turbocharged_entry_maker(
                            open_orders_open_symbol,
                            open_symbol,
                            trend,
                            mfirsi_signal,
                            long_take_profit_open_symbol,
                            short_take_profit_open_symbol,
                            long_dynamic_amount_open_symbol,
                            short_dynamic_amount_open_symbol,
                            long_pos_qty_open_symbol,
                            short_pos_qty_open_symbol,
                            long_pos_price_open_symbol,
                            short_pos_price_open_symbol,
                            should_long_open_symbol,
                            should_add_to_long_open_symbol,
                            should_short_open_symbol,
                            should_add_to_short_open_symbol
                        )
                
                else:
                    self.bybit_turbocharged_additional_entry_maker(
                        open_orders_open_symbol,
                        open_symbol,
                        trend,
                        mfirsi_signal,
                        long_take_profit_open_symbol,
                        short_take_profit_open_symbol,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

                self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

                #self.update_take_profit_if_profitable_for_one_minute(open_symbol)

                current_time = time.time()
                # Check if it's time to perform spoofing
                if current_time - self.last_cancel_time >= self.spoofing_interval:
                    self.spoofing_active = True
                    self.spoofing_action(open_symbol, short_dynamic_amount_open_symbol, long_dynamic_amount_open_symbol)

    def manage_mm(self, open_symbols, total_equity):
        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=self.whitelist, blacklist=self.blacklist, max_usd_value=self.max_usd_value)

        logging.info(f"open_symbols in manage_open_positions: {open_symbols}")
        for open_symbol in open_symbols:
            is_rotator_symbol = open_symbol in rotator_symbols

            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")
                
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            one_minute_distance = api_data['1mSpread']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']

            # Your existing code for each open_symbol
            # Fetch position data for the open symbol
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            min_qty = float(market_data["min_qty"])
            min_qty_str = str(min_qty)


            # Fetch the best ask and bid prices for the open symbol
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
            # Calculate the max trade quantities dynamically for this specific symbol
            max_trade_qty_value = self.calc_max_trade_qty(open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            self.initial_max_long_trade_qty = max_trade_qty_value
            self.initial_max_short_trade_qty = max_trade_qty_value

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

            # Calculate your take profit levels for each open symbol.
            short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
                position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            )
            long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
                position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            )

            # Additional context-specific variables
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

            # Additional context-specific variables
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False

            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

            # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
            # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
            
            # Fetch open orders for the open symbol
            open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)

            all_open_orders = self.exchange.get_all_open_orders_bybit()

            logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")
            #print(f"All open orders {all_open_orders}")

            #print(f"Open orders open symbol {open_orders_open_symbol}")

            # Check for existing take profit orders before placing new ones
            # Existing long take profit orders
            existing_long_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'sell'
            ]

            # Existing short take profit orders
            existing_short_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'buy'
            ]

            # Call the function to update long take profit spread only if no existing take profit order
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None and not existing_long_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol
                )

            # Call the function to update short take profit spread only if no existing take profit order
            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None and not existing_short_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol
                )

            # Take profit spread replacement
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None:
                self.next_long_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update
                )

            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None:
                self.next_short_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update
                )

            if long_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if open_symbol in open_symbols:

                logging.info(f"Symbols allowed from strategy file {self.symbols_allowed}")

                can_trade = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol)

                logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                
                # Check if the symbol is a rotator symbol
                if is_rotator_symbol:  

                    logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                    
                    # Add the following line before the initial entry trading logic to check symbol limits
                    if not self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol):
                        logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                        logging.info(f"Reached the symbol limit or already trading {open_symbol}. Skipping initial entry.")
                        
                    else:  # If we can trade, then proceed with initial entry
                        self.bybit_hedge_entry_maker_v3(
                            open_orders_open_symbol,
                            open_symbol,
                            trend,
                            mfirsi_signal,
                            one_minute_volume,
                            five_minute_distance,
                            min_vol,
                            min_dist,
                            long_dynamic_amount_open_symbol,
                            short_dynamic_amount_open_symbol,
                            long_pos_qty_open_symbol,
                            short_pos_qty_open_symbol,
                            long_pos_price_open_symbol,
                            short_pos_price_open_symbol,
                            should_long_open_symbol,
                            should_short_open_symbol,
                            should_add_to_long_open_symbol,
                            should_add_to_short_open_symbol
                        )
                
                else:
                    self.bybit_hedge_additional_entry_maker_v3(
                        open_orders_open_symbol,
                        open_symbol,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

                self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

                #self.update_take_profit_if_profitable_for_one_minute(open_symbol)

                current_time = time.time()
                # Check if it's time to perform spoofing
                if current_time - self.last_cancel_time >= self.spoofing_interval:
                    self.spoofing_active = True
                    self.spoofing_action(open_symbol, short_dynamic_amount_open_symbol, long_dynamic_amount_open_symbol)

    def manage_non_rotator_symbols(self, open_symbols, total_equity):
        for open_symbol in open_symbols:

            previous_five_minute_distance = None

            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            hma_trend = api_data['HMA Trend']

            # Fetch necessary data
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            # MA data
            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]

            # modify leverage per symbol
            self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_bid_price_open_symbol, max_leverage)
            self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )
            
            # Calculate necessary parameters
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False
            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False


            # Place additional entry orders
            if should_add_to_long_open_symbol or should_add_to_short_open_symbol:
                open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)
                self.bybit_hedge_additional_entry_maker_hma(
                    open_orders_open_symbol,
                    open_symbol,
                    trend,
                    hma_trend,
                    mfirsi_signal,
                    one_minute_volume,
                    five_minute_distance,
                    min_vol,
                    min_dist,
                    long_dynamic_amount_open_symbol,
                    short_dynamic_amount_open_symbol,
                    long_pos_qty_open_symbol,
                    short_pos_qty_open_symbol,
                    long_pos_price_open_symbol,
                    short_pos_price_open_symbol,
                    should_add_to_long_open_symbol,
                    should_add_to_short_open_symbol
                )

                short_take_profit = None
                long_take_profit = None

                short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price_open_symbol, long_pos_price_open_symbol, open_symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)

                # Call the function to update long take profit spread
                if long_pos_qty_open_symbol > 0 and long_take_profit is not None:
                    self.bybit_hedge_placetp_maker(open_symbol, long_pos_qty_open_symbol, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol)

                # Call the function to update short take profit spread
                if short_pos_qty_open_symbol > 0 and short_take_profit is not None:
                    self.bybit_hedge_placetp_maker(open_symbol, short_pos_qty_open_symbol, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol)

                # Take profit spread replacement
                if long_pos_qty_open_symbol > 0 and long_take_profit is not None:
                    self.next_long_tp_update = self.update_take_profit_spread_bybit(open_symbol, long_pos_qty_open_symbol, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update)

                if short_pos_qty_open_symbol > 0 and short_take_profit is not None:
                    self.next_short_tp_update = self.update_take_profit_spread_bybit(open_symbol, short_pos_qty_open_symbol, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update)

        current_time = time.time()
        # Check if it's time to perform spoofing
        if current_time - self.last_cancel_time >= self.spoofing_interval:
            self.spoofing_active = True
            #self.spoofing_action(open_symbol, short_spoofing_amount_open_symbol, long_spoofing_amount_open_symbol)
            self.spoofing_action(open_symbol, short_dynamic_amount_open_symbol, long_dynamic_amount_open_symbol)

    def manage_non_rotator_symbols_5m(self, open_symbols, total_equity):
        for open_symbol in open_symbols:

            previous_five_minute_distance = None

            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            five_minute_volume = api_data['5mVol']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            hma_trend = api_data['HMA Trend']

            # Fetch necessary data
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            # MA data
            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]

            # modify leverage per symbol
            self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_bid_price_open_symbol, max_leverage)
            self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            # Logging the leverages for a given symbol
            if open_symbol in self.long_pos_leverage_per_symbol:
                logging.info(f"Current long leverage for {open_symbol}: {self.long_pos_leverage_per_symbol[open_symbol]}x")
            else:
                logging.info(f"No long leverage set for {open_symbol}")

            if open_symbol in self.short_pos_leverage_per_symbol:
                logging.info(f"Current short leverage for {open_symbol}: {self.short_pos_leverage_per_symbol[open_symbol]}x")
            else:
                logging.info(f"No short leverage set for {open_symbol}")
                
            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )
            
            # Calculate necessary parameters
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False
            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False


            # Place additional entry orders
            if should_add_to_long_open_symbol or should_add_to_short_open_symbol:
                open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)
                self.bybit_additional_entry_mm_5m(
                    open_orders_open_symbol,
                    open_symbol,
                    trend,
                    hma_trend,
                    mfirsi_signal,
                    five_minute_volume,
                    five_minute_distance,
                    min_vol,
                    min_dist,
                    long_dynamic_amount_open_symbol,
                    short_dynamic_amount_open_symbol,
                    long_pos_qty_open_symbol,
                    short_pos_qty_open_symbol,
                    long_pos_price_open_symbol,
                    short_pos_price_open_symbol,
                    should_add_to_long_open_symbol,
                    should_add_to_short_open_symbol
                )

                short_take_profit = None
                long_take_profit = None

                short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price_open_symbol, long_pos_price_open_symbol, open_symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)

                # Call the function to update long take profit spread
                if long_pos_qty_open_symbol > 0 and long_take_profit is not None:
                    self.bybit_hedge_placetp_maker(open_symbol, long_pos_qty_open_symbol, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol)

                # Call the function to update short take profit spread
                if short_pos_qty_open_symbol > 0 and short_take_profit is not None:
                    self.bybit_hedge_placetp_maker(open_symbol, short_pos_qty_open_symbol, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol)

                # Take profit spread replacement
                if long_pos_qty_open_symbol > 0 and long_take_profit is not None:
                    self.next_long_tp_update = self.update_take_profit_spread_bybit(open_symbol, long_pos_qty_open_symbol, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update)

                if short_pos_qty_open_symbol > 0 and short_take_profit is not None:
                    self.next_short_tp_update = self.update_take_profit_spread_bybit(open_symbol, short_pos_qty_open_symbol, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update)

        current_time = time.time()
        # Check if it's time to perform spoofing
        if current_time - self.last_cancel_time >= self.spoofing_interval:
            self.spoofing_active = True
            #self.spoofing_action(open_symbol, short_spoofing_amount_open_symbol, long_spoofing_amount_open_symbol)
            self.spoofing_action(open_symbol, short_dynamic_amount_open_symbol, long_dynamic_amount_open_symbol)


    def manage_mm_hma(self, open_symbols, total_equity):
        # Get current rotator symbols
        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=self.whitelist, blacklist=self.blacklist, max_usd_value=self.max_usd_value)

        #print(f"Rotator symbols debug from strategy manager: {rotator_symbols}")

        logging.info(f"Rotator symbols from strategy: {rotator_symbols}")
        
        logging.info(f"open_symbols in manage_open_positions: {open_symbols}")
        for open_symbol in open_symbols:
            is_rotator_symbol = open_symbol in rotator_symbols

            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")
                
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            one_minute_distance = api_data['1mSpread']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']
            hma_trend = api_data['HMA Trend']

            logging.info(f"Manager: One minute volume for {open_symbol} : {one_minute_volume}")
            logging.info(f"Manager: Five minute distance for {open_symbol}: {five_minute_distance}")
            logging.info(f"Manager: Trend: {trend} for symbol: {open_symbol}")
            logging.info(f"Manager: HMA Trend: {hma_trend} for symbol : {open_symbol}")
            
            # Your existing code for each open_symbol
            # Fetch position data for the open symbol
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            min_qty = float(market_data["min_qty"])
            min_qty_str = str(min_qty)


            # Fetch the best ask and bid prices for the open symbol
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
            # Calculate the max trade quantities dynamically for this specific symbol
            max_trade_qty_value = self.calc_max_trade_qty(open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            self.initial_max_long_trade_qty = max_trade_qty_value
            self.initial_max_short_trade_qty = max_trade_qty_value

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

            # Calculate your take profit levels for each open symbol.
            short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
                position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            )
            long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
                position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            )

            # Additional context-specific variables
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

            # Additional context-specific variables
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False

            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

            # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
            # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            long_spoofing_amount_open_symbol, short_spoofing_amount_open_symbol, = self.calculate_spoofing_amount(
                open_symbol, total_equity, best_ask_price_open_symbol, max_leverage
            )

            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
            
            # Fetch open orders for the open symbol
            open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)

            all_open_orders = self.exchange.get_all_open_orders_bybit()

            logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")
            #print(f"All open orders {all_open_orders}")

            #print(f"Open orders open symbol {open_orders_open_symbol}")

            # Check for existing take profit orders before placing new ones
            # Existing long take profit orders
            existing_long_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'sell'
            ]

            # Existing short take profit orders
            existing_short_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'buy'
            ]

            # Call the function to update long take profit spread only if no existing take profit order
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None and not existing_long_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol
                )

            # Call the function to update short take profit spread only if no existing take profit order
            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None and not existing_short_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol
                )

            # Take profit spread replacement
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None:
                self.next_long_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update
                )

            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None:
                self.next_short_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update
                )

            if long_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if open_symbol in open_symbols:

                logging.info(f"Symbols allowed from strategy file {self.symbols_allowed}")

                can_trade = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol)

                logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                
                # Check if the symbol is a rotator symbol
                if is_rotator_symbol:  

                    logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                    
                    # Add the following line before the initial entry trading logic to check symbol limits
                    if not self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol):
                        logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                        logging.info(f"Reached the symbol limit or already trading {open_symbol}. Skipping initial entry.")
                        
                    else:  # If we can trade, then proceed with initial entry
                        logging.info(f"Manager proceeding with initial entry for {open_symbol}")
                        self.bybit_hedge_entry_maker_hma(
                            open_orders_open_symbol,
                            open_symbol,
                            trend,
                            hma_trend,
                            mfirsi_signal,
                            one_minute_volume,
                            five_minute_distance,
                            min_vol,
                            min_dist,
                            long_dynamic_amount_open_symbol,
                            short_dynamic_amount_open_symbol,
                            long_pos_qty_open_symbol,
                            short_pos_qty_open_symbol,
                            long_pos_price_open_symbol,
                            short_pos_price_open_symbol,
                            should_long_open_symbol,
                            should_short_open_symbol,
                            should_add_to_long_open_symbol,
                            should_add_to_short_open_symbol
                        )
                    
                    # Additional entry for rotator symbols
                    logging.info(f"Manager proceeding with additional entry for {open_symbol}")
                    self.bybit_hedge_additional_entry_maker_hma(
                        open_orders_open_symbol,
                        open_symbol,
                        trend,
                        hma_trend,
                        mfirsi_signal,
                        one_minute_volume,
                        five_minute_distance,
                        min_vol,
                        min_dist,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

                else:
                    logging.info(f"Manager proceeding with additional entry for {open_symbol}")
                    self.bybit_hedge_additional_entry_maker_hma(
                        open_orders_open_symbol,
                        open_symbol,
                        trend,
                        hma_trend,
                        mfirsi_signal,
                        one_minute_volume,
                        five_minute_distance,
                        min_vol,
                        min_dist,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

            # if open_symbol in open_symbols:

            #     logging.info(f"Symbols allowed from strategy file {self.symbols_allowed}")

            #     can_trade = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol)

            #     logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                
            #     # Check if the symbol is a rotator symbol
            #     if is_rotator_symbol:  

            #         logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                    
            #         # Add the following line before the initial entry trading logic to check symbol limits
            #         if not self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol):
            #             logging.info(f"Config symbols allowed from strategy class: {can_trade}")
            #             logging.info(f"Reached the symbol limit or already trading {open_symbol}. Skipping initial entry.")
                        
            #         else:  # If we can trade, then proceed with initial entry
            #             logging.info(f"Manager proceeding with initial entry for {open_symbol}")
            #             self.bybit_hedge_entry_maker_hma(
            #                 open_orders_open_symbol,
            #                 open_symbol,
            #                 trend,
            #                 hma_trend,
            #                 mfirsi_signal,
            #                 one_minute_volume,
            #                 five_minute_distance,
            #                 min_vol,
            #                 min_dist,
            #                 long_dynamic_amount_open_symbol,
            #                 short_dynamic_amount_open_symbol,
            #                 long_pos_qty_open_symbol,
            #                 short_pos_qty_open_symbol,
            #                 long_pos_price_open_symbol,
            #                 short_pos_price_open_symbol,
            #                 should_long_open_symbol,
            #                 should_short_open_symbol,
            #                 should_add_to_long_open_symbol,
            #                 should_add_to_short_open_symbol
            #             )
                
            #     else:
            #         logging.info(f"Manager proceeding with additional entry for {open_symbol}")
            #         self.bybit_hedge_additional_entry_maker_hma(
            #             open_orders_open_symbol,
            #             open_symbol,
            #             trend,
            #             hma_trend,
            #             mfirsi_signal,
            #             one_minute_volume,
            #             five_minute_distance,
            #             min_vol,
            #             min_dist,
            #             long_dynamic_amount_open_symbol,
            #             short_dynamic_amount_open_symbol,
            #             long_pos_qty_open_symbol,
            #             short_pos_qty_open_symbol,
            #             long_pos_price_open_symbol,
            #             short_pos_price_open_symbol,
            #             should_add_to_long_open_symbol,
            #             should_add_to_short_open_symbol
            #         )

            #     self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

                #self.update_take_profit_if_profitable_for_one_minute(open_symbol)

                current_time = time.time()
                # Check if it's time to perform spoofing
                if current_time - self.last_cancel_time >= self.spoofing_interval:
                    self.spoofing_active = True
                    #self.spoofing_action(open_symbol, short_spoofing_amount_open_symbol, long_spoofing_amount_open_symbol)
                    self.spoofing_action(open_symbol, short_dynamic_amount_open_symbol, long_dynamic_amount_open_symbol)

    def manage_mm_ratio(self, open_symbols, total_equity):
        # Get current rotator symbols
        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=self.whitelist, blacklist=self.blacklist, max_usd_value=self.max_usd_value)
        
        logging.info(f"open_symbols in manage_open_positions: {open_symbols}")
        for open_symbol in open_symbols:
            is_rotator_symbol = open_symbol in rotator_symbols

            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")
                
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            one_minute_distance = api_data['1mSpread']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']

            # Your existing code for each open_symbol
            # Fetch position data for the open symbol
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            min_qty = float(market_data["min_qty"])
            min_qty_str = str(min_qty)


            # Fetch the best ask and bid prices for the open symbol
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
            # Calculate the max trade quantities dynamically for this specific symbol
            max_trade_qty_value = self.calc_max_trade_qty(open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            self.initial_max_long_trade_qty = max_trade_qty_value
            self.initial_max_short_trade_qty = max_trade_qty_value

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

            # Calculate your take profit levels for each open symbol.
            short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
                position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            )
            long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
                position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            )

            # Additional context-specific variables
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

            # Additional context-specific variables
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False

            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

            # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
            # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
            
            # Fetch open orders for the open symbol
            open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)

            all_open_orders = self.exchange.get_all_open_orders_bybit()

            logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")
            #print(f"All open orders {all_open_orders}")

            #print(f"Open orders open symbol {open_orders_open_symbol}")

            # Check for existing take profit orders before placing new ones
            # Existing long take profit orders
            existing_long_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'sell'
            ]

            # Existing short take profit orders
            existing_short_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'buy'
            ]

            # Call the function to update long take profit spread only if no existing take profit order
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None and not existing_long_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol
                )

            # Call the function to update short take profit spread only if no existing take profit order
            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None and not existing_short_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol
                )

            # Take profit spread replacement
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None:
                self.next_long_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update
                )

            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None:
                self.next_short_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update
                )

            if long_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if open_symbol in open_symbols:

                logging.info(f"Symbols allowed from strategy file {self.symbols_allowed}")

                can_trade = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol)

                logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                
                # Check if the symbol is a rotator symbol
                if is_rotator_symbol:  

                    logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                    
                    # Add the following line before the initial entry trading logic to check symbol limits
                    if not self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol):
                        logging.info(f"Config symbols allowed from strategy class: {can_trade}")
                        logging.info(f"Reached the symbol limit or already trading {open_symbol}. Skipping initial entry.")
                        
                    else:  # If we can trade, then proceed with initial entry
                        self.bybit_hedge_entry_maker_v3_ratio(
                            open_orders_open_symbol,
                            open_symbol,
                            trend,
                            mfirsi_signal,
                            one_minute_volume,
                            five_minute_distance,
                            min_vol,
                            min_dist,
                            long_dynamic_amount_open_symbol,
                            short_dynamic_amount_open_symbol,
                            long_pos_qty_open_symbol,
                            short_pos_qty_open_symbol,
                            long_pos_price_open_symbol,
                            short_pos_price_open_symbol,
                            should_long_open_symbol,
                            should_short_open_symbol,
                            should_add_to_long_open_symbol,
                            should_add_to_short_open_symbol
                        )
                
                else:
                    self.bybit_additional_entry_maker_v3_ratio(
                        open_orders_open_symbol,
                        open_symbol,
                        trend,
                        mfirsi_signal,
                        one_minute_volume,
                        five_minute_distance,
                        min_vol,
                        min_dist,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

                self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

                #self.update_take_profit_if_profitable_for_one_minute(open_symbol)

            current_time = time.time()
            # Check if it's time to perform spoofing
            if current_time - self.last_cancel_time >= self.spoofing_interval:
                self.spoofing_active = True
                self.spoofing_action(open_symbol, short_dynamic_amount_open_symbol, long_dynamic_amount_open_symbol)

    def manage_open_positions_v2(self, open_symbols, total_equity):
        # Get current rotator symbols
        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=self.whitelist, blacklist=self.blacklist, max_usd_value=self.max_usd_value)
        
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume

        for open_symbol in open_symbols:
            # Check if the open symbol is NOT in the current rotator symbols
            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")

            print(f"Symbols allowed: {self.symbols_allowed}")
            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']

            # Get Market Data
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)

            # Fetch the best ask and bid prices for the open symbol
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
            # Fetch position data for the open symbol
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

            max_trade_qty_value = self.calc_max_trade_qty(open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            self.initial_max_long_trade_qty = max_trade_qty_value
            self.initial_max_short_trade_qty = max_trade_qty_value

            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            # Adjust leverage based on the current position quantities
            if long_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

            # Additional context-specific variables
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_6_high_open_symbol is not None else False

            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

            # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
            # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

            # Log the dynamic amounts
            #print(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            #print(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
            logging.info(f"Long dynamic amount from manager for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount from manager for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
            logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")

            
            # # Calculate your take profit levels for each open symbol - with avoiding fees
            # short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["short"]["price"], short_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )
            # long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["long"]["price"], long_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )
            
            # Calculate your take profit levels for each open symbol.
            short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
                position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            )
            long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
                position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            )

            # Fetch open orders for the open symbol
            open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)

            all_open_orders = self.exchange.get_all_open_orders_bybit()

            #print(f"All open orders {all_open_orders}")

            #print(f"Open orders open symbol {open_orders_open_symbol}")

            # Check for existing take profit orders before placing new ones
            # Existing long take profit orders
            existing_long_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'sell'
            ]

            # Existing short take profit orders
            existing_short_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'buy'
            ]

            # Call the function to update long take profit spread only if no existing take profit order
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None and not existing_long_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol
                )

            # Call the function to update short take profit spread only if no existing take profit order
            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None and not existing_short_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol
                )

            # Take profit spread replacement
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None:
                self.next_long_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update
                )

            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None:
                self.next_short_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update
                )

            if open_symbol in open_symbols:
                is_rotator_symbol = open_symbol in rotator_symbols
                if is_rotator_symbol:

                    # Add the following line before the initial entry trading logic to check symbol limits
                    if not self.can_trade_new_symbol(open_symbols, self.symbols_allowed, open_symbol):
                        logging.info(f"Reached the symbol limit or already trading {open_symbol}. Skipping initial entry.")
                    
                    else:  # If we can trade, then proceed with initial entry
                        self.bybit_hedge_entry_maker_v3(
                            open_orders_open_symbol,
                            open_symbol,
                            trend,
                            mfirsi_signal,
                            one_minute_volume,
                            five_minute_distance,
                            min_vol,
                            min_dist,
                            long_dynamic_amount_open_symbol,
                            short_dynamic_amount_open_symbol,
                            long_pos_qty_open_symbol,
                            short_pos_qty_open_symbol,
                            long_pos_price_open_symbol,
                            short_pos_price_open_symbol,
                            should_long_open_symbol,
                            should_short_open_symbol,
                            should_add_to_long_open_symbol,
                            should_add_to_short_open_symbol
                        )
                
                else:
                    self.bybit_hedge_additional_entry_maker_v3(
                        open_orders_open_symbol,
                        open_symbol,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

                self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

                #self.update_take_profit_if_profitable_for_one_minute(open_symbol)
                
            # Log the dynamic amounts
            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")

    def manage_open_positions(self, open_symbols, total_equity):
        # Get current rotator symbols
        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=self.whitelist, blacklist=self.blacklist, max_usd_value=self.max_usd_value)
        
        logging.info(f"open_symbols in manage_open_positions: {open_symbols}")
        for open_symbol in open_symbols:
            #logging.info(f"Type of open_symbols: {type(open_symbols)}")

            is_rotator_symbol = open_symbol in rotator_symbols
            # Check if the open symbol is NOT in the current rotator symbols
            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")
                
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            # Get API data
            api_data = self.manager.get_api_data(open_symbol)
            one_minute_volume = api_data['1mVol']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']

            # Fetch position data for the open symbol
            market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
            max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            min_qty = float(market_data["min_qty"])
            min_qty_str = str(min_qty)

            # Fetch the best ask and bid prices for the open symbol
            best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
            best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
            # # Calculate the max trade quantities dynamically for this specific symbol
            # self.initial_max_long_trade_qty, self.initial_max_short_trade_qty = self.calc_max_trade_qty_multi(
            #     total_equity, best_ask_price_open_symbol, max_leverage)

            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            # long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount_multi(
            #     open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            # )

            self.bybit_reset_position_leverage_long(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            self.bybit_reset_position_leverage_short(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            
            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

            # # Calculate your take profit levels for each open symbol - with avoiding fees
            # short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["short"]["price"], short_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )
            # long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit_fees(
            #     position_data_open_symbol["long"]["price"], long_pos_qty_open_symbol, open_symbol, five_minute_distance
            # )

            # Calculate your take profit levels for each open symbol.
            short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
                position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
            )
            long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
                position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
            )

            # Additional context-specific variables
            long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
            short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

            # Additional context-specific variables
            should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
            should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False

            should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
            should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

            # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
            # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

            # Log the dynamic amounts
            #print(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            #print(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
            logging.info(f"Long dynamic amount from manager for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount from manager for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
            logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")
            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
            

            # Fetch open orders for the open symbol
            open_orders_open_symbol = self.retry_api_call(self.exchange.get_open_orders, open_symbol)

            all_open_orders = self.exchange.get_all_open_orders_bybit()

            #print(f"All open orders {all_open_orders}")

            #print(f"Open orders open symbol {open_orders_open_symbol}")

            # Check for existing take profit orders before placing new ones
            # Existing long take profit orders
            existing_long_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'sell'
            ]

            # Existing short take profit orders
            existing_short_tp_orders = [
                order for order in all_open_orders 
                if order['info'].get('orderType') == 'Take Profit' 
                and order['symbol'] == open_symbol 
                and order['side'] == 'buy'
            ]

            # Call the function to update long take profit spread only if no existing take profit order
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None and not existing_long_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol
                )

            # Call the function to update short take profit spread only if no existing take profit order
            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None and not existing_short_tp_orders:
                self.bybit_hedge_placetp_maker(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol
                )

            # Take profit spread replacement
            if long_pos_qty_open_symbol > 0 and long_take_profit_open_symbol is not None:
                self.next_long_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, long_pos_qty_open_symbol, long_take_profit_open_symbol, positionIdx=1, order_side="sell", open_orders=open_orders_open_symbol, next_tp_update=self.next_long_tp_update
                )

            if short_pos_qty_open_symbol > 0 and short_take_profit_open_symbol is not None:
                self.next_short_tp_update = self.update_take_profit_spread_bybit(
                    open_symbol, short_pos_qty_open_symbol, short_take_profit_open_symbol, positionIdx=2, order_side="buy", open_orders=open_orders_open_symbol, next_tp_update=self.next_short_tp_update
                )

            long_dynamic_amount_open_symbol = min_qty
            short_dynamic_amount_open_symbol = min_qty

            if open_symbol in open_symbols:
                if is_rotator_symbol:
                    self.bybit_hedge_entry_maker_v3(
                        open_orders_open_symbol,
                        open_symbol,
                        trend,
                        mfirsi_signal,
                        one_minute_volume,
                        five_minute_distance,
                        min_vol,
                        min_dist,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_long_open_symbol,
                        should_short_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )
                else:
                    self.bybit_hedge_additional_entry_maker_v3(
                        open_orders_open_symbol,
                        open_symbol,
                        long_dynamic_amount_open_symbol,
                        short_dynamic_amount_open_symbol,
                        long_pos_qty_open_symbol,
                        short_pos_qty_open_symbol,
                        long_pos_price_open_symbol,
                        short_pos_price_open_symbol,
                        should_add_to_long_open_symbol,
                        should_add_to_short_open_symbol
                    )

                self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)
                

            # Cancel entries (Note: Replace this with the actual conditions for your open_symbol)
            #self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

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

    def cancel_stale_orders_bybit(self):
        current_time = time.time()
        if current_time - self.last_stale_order_check_time < 3720:  # 3720 seconds = 1 hour 12 minutes
            return  # Skip the rest of the function if it's been less than 1 hour 12 minutes

        all_open_orders = self.exchange.get_all_open_orders_bybit()
        open_position_data = self.exchange.get_all_open_positions_bybit()
        open_symbols = self.extract_symbols_from_positions_bybit(open_position_data)
        open_symbols = [symbol.replace("/", "") for symbol in open_symbols]
        #rotator_symbols = self.manager.get_auto_rotate_symbols()
        rotator_symbols = self.manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=self.whitelist, blacklist=self.blacklist, max_usd_value=self.max_usd_value)
        all_open_order_symbols = [order['symbol'] for order in all_open_orders]
        orders_to_cancel = [order for order in all_open_order_symbols if order not in open_symbols and order not in rotator_symbols]

        for symbol in orders_to_cancel:
            self.exchange.cancel_all_open_orders_bybit(symbol)
            logging.info(f"Stale order for {symbol} canceled")

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

                if mfi is not None and isinstance(mfi, str):
                    if mfi.lower() == "long" and long_pos_qty == 0:
                        logging.info(f"Placing initial long entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1)
                        logging.info(f"Placed initial long entry with post-only order")
                    elif mfi.lower() == "long" and long_pos_qty < max_long_trade_qty and best_bid_price < long_pos_price:
                        logging.info(f"Placing additional long entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1)
                    elif mfi.lower() == "short" and short_pos_qty == 0:
                        logging.info(f"Placing initial short entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2)
                        logging.info(f"Placed initial short entry with post-only order")
                    elif mfi.lower() == "short" and short_pos_qty < max_short_trade_qty and best_ask_price > short_pos_price:
                        logging.info(f"Placing additional short entry with post-only order")
                        self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2)

    def adjust_leverage_and_qty(self, current_qty, initial_qty, current_leverage, max_leverage, increase=True):
        if increase:
            new_leverage = min(current_leverage + self.LEVERAGE_STEP, max_leverage, self.MAX_LEVERAGE)
            new_qty = initial_qty * (1 + self.QTY_INCREMENT)
        else:
            new_leverage = max(1.0, current_leverage - self.LEVERAGE_STEP)
            new_qty = initial_qty  # Reset to the initial maximum trade quantity

        return new_qty, new_leverage

    def bybit_reset_position_leverage_long_v3(self, symbol, long_pos_qty, total_equity, best_ask_price, max_leverage):
        if symbol not in self.initial_max_long_trade_qty_per_symbol:
            # Initialize for the symbol
            self.initial_max_long_trade_qty_per_symbol[symbol] = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
            self.long_pos_leverage_per_symbol[symbol] = 0.001  # starting leverage

        if long_pos_qty >= self.initial_max_long_trade_qty_per_symbol[symbol] and self.long_pos_leverage_per_symbol[symbol] < self.MAX_LEVERAGE:
            self.max_long_trade_qty_per_symbol[symbol], self.long_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                self.max_long_trade_qty_per_symbol.get(symbol, 0), 
                self.initial_max_long_trade_qty_per_symbol[symbol], 
                self.long_pos_leverage_per_symbol[symbol], 
                max_leverage, 
                increase=True
            )
            logging.info(f"Long leverage for {symbol} temporarily increased to {self.long_pos_leverage_per_symbol[symbol]}x")
        elif long_pos_qty < (self.max_long_trade_qty_per_symbol.get(symbol, 0) / 2) and self.long_pos_leverage_per_symbol[symbol] > 1.0:
            self.max_long_trade_qty_per_symbol[symbol], self.long_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                self.max_long_trade_qty_per_symbol.get(symbol, 0), 
                self.initial_max_long_trade_qty_per_symbol[symbol], 
                self.long_pos_leverage_per_symbol[symbol], 
                max_leverage, 
                increase=False
            )
            logging.info(f"Long leverage for {symbol} returned to normal {self.long_pos_leverage_per_symbol[symbol]}x")

    def bybit_reset_position_leverage_short_v3(self, symbol, short_pos_qty, total_equity, best_ask_price, max_leverage):
        # Initialize for the symbol if it's not already present
        if symbol not in self.initial_max_short_trade_qty_per_symbol:
            self.initial_max_short_trade_qty_per_symbol[symbol] = self.calc_max_trade_qty(symbol, total_equity, best_ask_price, max_leverage)
            self.short_pos_leverage_per_symbol[symbol] = 0.001  # starting leverage

        # Check conditions to increase leverage
        if short_pos_qty >= self.initial_max_short_trade_qty_per_symbol[symbol] and self.short_pos_leverage_per_symbol[symbol] < self.MAX_LEVERAGE:
            self.max_short_trade_qty_per_symbol[symbol], self.short_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                self.max_short_trade_qty_per_symbol.get(symbol, 0), 
                self.initial_max_short_trade_qty_per_symbol[symbol], 
                self.short_pos_leverage_per_symbol[symbol], 
                max_leverage, 
                increase=True
            )
            logging.info(f"Short leverage for {symbol} temporarily increased to {self.short_pos_leverage_per_symbol[symbol]}x")
        
        # Check conditions to reset leverage back to normal
        elif short_pos_qty < (self.max_short_trade_qty_per_symbol.get(symbol, 0) / 2) and self.short_pos_leverage_per_symbol[symbol] > 1.0:
            self.max_short_trade_qty_per_symbol[symbol], self.short_pos_leverage_per_symbol[symbol] = self.adjust_leverage_and_qty(
                self.max_short_trade_qty_per_symbol.get(symbol, 0), 
                self.initial_max_short_trade_qty_per_symbol[symbol], 
                self.short_pos_leverage_per_symbol[symbol], 
                max_leverage, 
                increase=False
            )
            logging.info(f"Short leverage for {symbol} returned to normal {self.short_pos_leverage_per_symbol[symbol]}x")

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
