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

logging = Logger(logger_name="BybitStrategy", filename="BybitStrategy.log", stream=True)

class BinanceStrategy(BaseStrategy):
    def __init__(self, exchange, config, manager, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.linear_grid_orders = {} 
        # Bybit-specific initialization code
        pass

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
