from colorama import Fore
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
from ..logger import Logger
from datetime import datetime, timedelta
from threading import Thread, Lock

from ...bot_metrics import BotDatabase

from directionalscalper.core.strategies.base_strategy import BaseStrategy

logging = Logger(logger_name="BybitBaseStrategy", filename="BybitBaseStrategy.log", stream=True)

class BybitStrategy(BaseStrategy):
    def __init__(self, exchange, config, manager, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.linear_grid_orders = {} 
        # Bybit-specific initialization code
        pass

    TAKER_FEE_RATE = 0.00055

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
            logging.error(f"Error when quantizing target_profit_price. {e}")
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
            logging.error(f"Error when quantizing target_profit_price. {e}")
            return None

        return float(target_profit_price)
    
# price_precision, qty_precision = self.exchange.get_symbol_precision_bybit(symbol)
    def calculate_dynamic_long_take_profit(self, best_bid_price, long_pos_price, symbol, upnl_profit_pct, max_deviation_pct=0.0040):
        if long_pos_price is None:
            logging.error("Long position price is None for symbol: " + symbol)
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
            logging.error("Short position price is None for symbol: " + symbol)
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

    def adjust_risk_parameters(self, exchange_max_leverage):
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
            logging.error(f"Error placing order: {str(e)}")
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


    def bybit_1m_mfi_quickscalp_trend(self, open_orders: list, symbol: str, min_vol: float, one_minute_volume: float, mfirsi: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, entry_during_autoreduce: bool, volume_check: bool, long_take_profit: float, short_take_profit: float, upnl_profit_pct: float, tp_order_counts: dict):
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

                # Check if volume check is enabled or not
                if not volume_check or (one_minute_volume > min_vol):
                    if not self.auto_reduce_active_long.get(symbol, False):
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
                                    short_pos_price=short_pos_price,
                                    long_pos_price=long_pos_price,
                                    positionIdx=1,
                                    order_side="sell",
                                    last_tp_update=self.next_long_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                        elif long_pos_qty > 0 and mfi_signal_long and current_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                            if entry_during_autoreduce or not self.auto_reduce_active_long.get(symbol, False):
                                self.place_postonly_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                                time.sleep(1)
                                if long_pos_qty > 0:
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
                                logging.info(f"Skipping additional long entry for {symbol} due to active auto-reduce.")

                    if not self.auto_reduce_active_short.get(symbol, False):
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
                                    long_pos_price=long_pos_price,
                                    positionIdx=2,
                                    order_side="buy",
                                    last_tp_update=self.next_short_tp_update,
                                    tp_order_counts=tp_order_counts
                                )
                        elif short_pos_qty > 0 and mfi_signal_short and current_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                            if entry_during_autoreduce or not self.auto_reduce_active_short.get(symbol, False):
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
                                        long_pos_price=long_pos_price,
                                        positionIdx=2,
                                        order_side="buy",
                                        last_tp_update=self.next_short_tp_update,
                                        tp_order_counts=tp_order_counts
                                    )
                            else:
                                logging.info(f"Skipping additional short entry for {symbol} due to active auto-reduce.")
                else:
                    logging.info(f"Volume check is disabled or conditions not met for {symbol}, proceeding without volume check.")

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

    def linear_grid_handle_positions(self, symbol: str, total_equity: float, long_pos_qty: float, short_pos_qty: float, levels: int, strength: float, outer_price_distance: float, wallet_exposure_limit: float, user_defined_leverage_long: float, user_defined_leverage_short: float, long_mode: bool, short_mode: bool):
        if symbol not in self.symbol_locks:
            self.symbol_locks[symbol] = threading.Lock()

        with self.symbol_locks[symbol]:
            current_price = self.exchange.get_current_price(symbol)
            logging.info(f"Current price for {symbol}: {current_price}")

            order_book = self.exchange.get_orderbook(symbol)
            best_ask_price = order_book['asks'][0][0] if 'asks' in order_book else self.last_known_ask.get(symbol)
            best_bid_price = order_book['bids'][0][0] if 'bids' in order_book else self.last_known_bid.get(symbol)

            # Calculate outer prices
            outer_price_long = best_ask_price * (1 + outer_price_distance)
            outer_price_short = best_bid_price * (1 - outer_price_distance)

            # Calculate grid levels
            diff_long = outer_price_long - best_ask_price
            diff_short = best_bid_price - outer_price_short
            factors = np.linspace(0.0, 1.0, num=levels + 1) ** strength
            grid_levels_long = [best_ask_price + (diff_long * factor) for factor in factors[1:]]
            grid_levels_short = [best_bid_price - (diff_short * factor) for factor in factors[1:]]

            # Calculate total amount based on wallet exposure limit and user-defined leverage
            total_amount_long = self.calculate_total_amount(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_long, "buy") if long_mode else 0
            total_amount_short = self.calculate_total_amount(symbol, total_equity, best_ask_price, best_bid_price, wallet_exposure_limit, user_defined_leverage_short, "sell") if short_mode else 0
            amounts_long = self.calculate_order_amounts(total_amount_long, levels, strength)
            amounts_short = self.calculate_order_amounts(total_amount_short, levels, strength)

            if long_mode and long_pos_qty == 0:
                self.place_linear_grid_orders(symbol, "buy", grid_levels_long, amounts_long)
            elif short_mode and short_pos_qty == 0:
                self.place_linear_grid_orders(symbol, "sell", grid_levels_short, amounts_short)

            time.sleep(5)

    def calculate_total_amount(self, symbol: str, total_equity: float, best_ask_price: float, best_bid_price: float, wallet_exposure_limit: float, user_defined_leverage: float, side: str) -> float:
        # Fetch market data to get the minimum trade quantity for the symbol
        market_data = self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)
        min_qty = float(market_data["min_qty"])

        # Calculate the minimum quantity in USD value based on the side
        if side == "buy":
            min_qty_usd_value = min_qty * best_ask_price
        elif side == "sell":
            min_qty_usd_value = min_qty * best_bid_price
        else:
            raise ValueError(f"Invalid side: {side}")

        # Calculate the maximum position value based on total equity, wallet exposure limit, and user-defined leverage
        max_position_value = total_equity * wallet_exposure_limit * user_defined_leverage

        # Calculate the total amount considering the maximum position value and minimum quantity
        total_amount = max(max_position_value, min_qty_usd_value)

        # Adjusting total amount based on the symbol's minimum quantity precision
        qty_precision = self.exchange.get_symbol_precision_bybit(symbol)[1]
        total_amount_adjusted = round(total_amount, -int(math.log10(qty_precision)))

        return total_amount_adjusted

    def calculate_order_amounts(self, total_amount: float, levels: int, strength: float) -> List[float]:
        # Calculate the order amounts based on the strength (ratio_power)
        amounts = []
        for i in range(levels):
            ratio = (i + 1) ** strength
            amount = total_amount * (ratio / sum([(j + 1) ** strength for j in range(levels)]))
            amounts.append(amount)

        return amounts

    def place_linear_grid_orders(self, symbol: str, side: str, grid_levels: list, amounts: list):
        for level, amount in zip(grid_levels, amounts):
            order = self.exchange.create_order(symbol, 'limit', side, amount, level)
            self.linear_grid_orders.setdefault(symbol, []).append(order)
            logging.info(f"Placed {side} order at level {level} for {symbol} with amount {amount}")

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

