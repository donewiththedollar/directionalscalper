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

logging = Logger(logger_name="Strategy", filename="Strategy.log", stream=True)

class BybitStrategy(BaseStrategy):
    def __init__(self, exchange, config, manager, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.linear_grid_orders = {} 
        # Bybit-specific initialization code
        pass

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

    def limit_order_bybit_unified(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit_unified(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order
    
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

    def linear_grid_handle_positions(self, symbol: str, long_pos_qty: float, short_pos_qty: float, long_dynamic_amount: float, short_dynamic_amount: float, levels: int, strength: float, outer_price_distance: float):
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

            if long_pos_qty == 0:
                self.place_linear_grid_orders(symbol, "buy", grid_levels_long, long_dynamic_amount)
            elif short_pos_qty == 0:
                self.place_linear_grid_orders(symbol, "sell", grid_levels_short, short_dynamic_amount)

            time.sleep(5)

    def place_linear_grid_orders(self, symbol: str, side: str, grid_levels: list, dynamic_amount: float):
        for level in grid_levels:
            order = self.exchange.create_order(symbol, 'limit', side, dynamic_amount, level)
            self.linear_grid_orders.setdefault(symbol, []).append(order)
            logging.info(f"Placed {side} order at level {level} for {symbol} with amount {dynamic_amount}")