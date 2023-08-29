from colorama import Fore
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
import time
import math
import numpy
import ta as ta
import os
import uuid
import logging
import json
from .logger import Logger
from datetime import datetime, timedelta

logging = Logger(logger_name="Strategy", filename="Strategy.log", stream=True)

class Strategy:
    LEVERAGE_STEP = 0.01  # The step at which to increase leverage
    MAX_LEVERAGE = 0.5  # The maximum allowable leverage
    QTY_INCREMENT = 0.1 # How much your position size increases
    def __init__(self, exchange, config, manager):
        self.exchange = exchange
        self.config = config
        self.manager = manager
        self.symbol = config.symbol
        self.printed_trade_quantities = False
        self.last_mfirsi_signal = None
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

    class OrderBookAnalyzer:
        def __init__(self, exchange, symbol):
            self.exchange = exchange
            self.symbol = symbol

        def get_order_book(self):
            return self.exchange.get_orderbook(self.symbol)

        def buying_pressure(self):
            order_book = self.get_order_book()
            total_bids = sum([bid[1] for bid in order_book['bids']])
            total_asks = sum([ask[1] for ask in order_book['asks']])
            
            return total_bids > total_asks

        def selling_pressure(self):
            return not self.buying_pressure()

    def toggle_spoof(self):
        # Legality issues
        self.should_spoof = not self.should_spoof

    def get_symbols_allowed(self, account_name):
        for exchange in self.config["exchanges"]:
            if exchange["account_name"] == account_name:
                return exchange.get("symbols_allowed", None)
        return None


    def calculate_dynamic_amount(self, symbol, market_data, total_equity, best_ask_price, max_leverage):
        
        logging.info(f"Calculating dynamic amount for symbol: {symbol}")

        # if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
        #     self.max_long_trade_qty, self.max_short_trade_qty = self.calc_max_trade_qty(total_equity,
        #                                                                                 best_ask_price,
        #                                                                                 max_leverage)

        if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
            max_trade_qty = self.calc_max_trade_qty(total_equity, best_ask_price, max_leverage)
            self.max_long_trade_qty = max_trade_qty
            self.max_short_trade_qty = max_trade_qty
            logging.info(f"Calculated max_long_trade_qty: {self.max_long_trade_qty}, max_short_trade_qty: {self.max_short_trade_qty}")

            if self.initial_max_long_trade_qty is None:
                self.initial_max_long_trade_qty = self.max_long_trade_qty
                logging.info(f"Initial max long trade qty set to {self.initial_max_long_trade_qty}")

            if self.initial_max_short_trade_qty is None:
                self.initial_max_short_trade_qty = self.max_short_trade_qty  
                logging.info(f"Initial max short trade qty set to {self.initial_max_short_trade_qty}") 

        long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty
        short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty

        logging.info(f"Initial long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

        min_qty = float(market_data["min_qty"])
        min_qty_str = str(min_qty)

        logging.info(f"min_qty: {min_qty}, min_qty_str: {min_qty_str}")

        if ".0" in min_qty_str:
            precision_level = 0
        else:
            precision_level = len(min_qty_str.split(".")[1])

        logging.info(f"Calculated precision_level: {precision_level}")

        long_dynamic_amount = round(long_dynamic_amount, precision_level)
        short_dynamic_amount = round(short_dynamic_amount, precision_level)

        logging.info(f"Rounded long_dynamic_amount: {long_dynamic_amount}, short_dynamic_amount: {short_dynamic_amount}")

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

    def calculate_dynamic_amount_multi(self, symbol, market_data, total_equity, best_ask_price, max_leverage):
        logging.info(f"Calculating dynamic amounts for {symbol}...")
        
        if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
            self.max_long_trade_qty, self.max_short_trade_qty = self.calc_max_trade_qty_multi(total_equity, best_ask_price, max_leverage)
            logging.info(f"Max long trade qty: {self.max_long_trade_qty}, Max short trade qty: {self.max_short_trade_qty}")

            if self.initial_max_long_trade_qty is None:
                self.initial_max_long_trade_qty = self.max_long_trade_qty
                logging.info(f"Initial max long trade qty set to {self.initial_max_long_trade_qty}")

            if self.initial_max_short_trade_qty is None:
                self.initial_max_short_trade_qty = self.max_short_trade_qty  
                logging.info(f"Initial max short trade qty set to {self.initial_max_short_trade_qty}") 

        long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty
        short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty

        min_qty = float(market_data["min_qty"])
        min_qty_str = str(min_qty)

        if ".0" in min_qty_str:
            precision_level = 0
        else:
            precision_level = len(min_qty_str.split(".")[1])

        long_dynamic_amount = round(long_dynamic_amount, precision_level)
        short_dynamic_amount = round(short_dynamic_amount, precision_level)

        self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)
        self.check_amount_validity_once_bybit(short_dynamic_amount, symbol)

        if long_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty")
            long_dynamic_amount = min_qty

        if short_dynamic_amount < min_qty:
            logging.info(f"Dynamic amount too small for 0.001x, using min_qty")
            short_dynamic_amount = min_qty

        logging.info(f"Calculated dynamic amounts: Long = {long_dynamic_amount}, Short = {short_dynamic_amount}")

        return long_dynamic_amount, short_dynamic_amount, min_qty

    # def calculate_dynamic_amount(self, symbol, market_data, total_equity, best_ask_price, max_leverage):
        
    #     if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
    #         self.max_long_trade_qty = self.max_short_trade_qty = self.calc_max_trade_qty(total_equity,
    #                                                                                     best_ask_price,
    #                                                                                     max_leverage)

    #         # Set initial quantities if they're None
    #         if self.initial_max_long_trade_qty is None:
    #             self.initial_max_long_trade_qty = self.max_long_trade_qty
    #             logging.info(f"Initial max trade qty set to {self.initial_max_long_trade_qty}")

    #         if self.initial_max_short_trade_qty is None:
    #             self.initial_max_short_trade_qty = self.max_short_trade_qty  
    #             logging.info(f"Initial trade qty set to {self.initial_max_short_trade_qty}") 

    #     long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty
    #     short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty

    #     min_qty = float(market_data["min_qty"])
    #     min_qty_str = str(min_qty)

    #     # # Get the precision level of the minimum quantity
    #     # if ".0" in min_qty_str:
    #     #     # The minimum quantity has a fractional part, get its precision level
    #     #     precision_level = len(min_qty_str.split(".")[1])
    #     # else:
    #     #     # The minimum quantity does not have a fractional part, precision is 0
    #     #     precision_level = 0

    #     if ".0" in min_qty_str:
    #         precision_level = 0
    #     else:
    #         precision_level = len(min_qty_str.split(".")[1])

    #     long_dynamic_amount = round(long_dynamic_amount, precision_level)
    #     short_dynamic_amount = round(short_dynamic_amount, precision_level)

    #     self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)
    #     self.check_amount_validity_once_bybit(short_dynamic_amount, symbol)

    #     if long_dynamic_amount < min_qty:
    #         logging.info(f"Dynamic amount too small for 0.001x, using min_qty")
    #         long_dynamic_amount = min_qty

    #     if short_dynamic_amount < min_qty:
    #         logging.info(f"Dynamic amount too small for 0.001x, using min_qty")
    #         short_dynamic_amount = min_qty

    #     return long_dynamic_amount, short_dynamic_amount, min_qty

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

    def limit_order_bybit_unified(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit_unified(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

    def postonly_limit_order_bybit(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
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

    def calc_max_trade_qty(self, total_equity, best_ask_price, max_leverage, max_retries=5, retry_delay=5):
        wallet_exposure = self.config.wallet_exposure
        for i in range(max_retries):
            try:
                market_data = self.exchange.get_market_data_bybit(self.symbol)
                max_trade_qty = round(
                    (float(total_equity) * wallet_exposure / float(best_ask_price))
                    / (100 / max_leverage),
                    int(float(market_data["min_qty"])),
                )
                # Return the same max_trade_qty for both long and short
                #return max_trade_qty, max_trade_qty
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

    # def calc_max_trade_qty(self, total_equity, best_ask_price, max_leverage, max_retries=5, retry_delay=5):
    #     wallet_exposure = self.config.wallet_exposure
    #     for i in range(max_retries):
    #         try:
    #             market_data = self.exchange.get_market_data_bybit(self.symbol)
    #             max_trade_qty = round(
    #                 (float(total_equity) * wallet_exposure / float(best_ask_price))
    #                 / (100 / max_leverage),
    #                 int(float(market_data["min_qty"])),
    #             )
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

    def print_trade_quantities_once_bybit(self, max_trade_qty):
        if not self.printed_trade_quantities:
            wallet_exposure = self.config.wallet_exposure
            best_ask_price = self.exchange.get_orderbook(self.symbol)['asks'][0][0]
            self.exchange.print_trade_quantities_bybit(max_trade_qty, [0.001, 0.01, 0.1, 1, 2.5, 5], wallet_exposure, best_ask_price)
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

    def calculate_next_update_time(self):
        # 5 min interval calc
        now = datetime.now()
        next_update_minute = (now.minute // 5 + 1) * 5
        if next_update_minute == 60:
            next_update_minute = 0
            now += timedelta(hours=1)
        return now.replace(minute=next_update_minute, second=0, microsecond=0)

    def calculate_short_take_profit_spread_bybit_fees(self, short_pos_price, quantity, symbol, decrease_percentage=0):
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

            if decrease_percentage is None:
                decrease_percentage = 0

            # Calculate the order value
            order_value = Decimal(quantity) / short_target_price
            # Calculate the trading fee for this order
            trading_fee = order_value * Decimal(self.taker_fee_rate)
            # Subtract the trading fee from the take profit target price
            short_target_price = short_target_price - trading_fee

            try:
                short_target_price = short_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing short_target_price. short_target_price={short_target_price}, price_precision={price_precision}")
                return None

            short_target_price -= short_target_price * Decimal(decrease_percentage) / 100
            short_profit_price = short_target_price

            return float(short_profit_price)

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

            # Calculate the order value
            order_value = Decimal(quantity) / long_target_price
            # Calculate the trading fee for this order
            trading_fee = order_value * Decimal(self.taker_fee_rate)
            # Add the trading fee to the take profit target price
            long_target_price = long_target_price + trading_fee

            try:
                long_target_price = long_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing long_target_price. long_target_price={long_target_price}, price_precision={price_precision}")
                return None

            long_target_price += long_target_price * Decimal(increase_percentage) / 100
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
        elif self.open_symbols_count >= symbols_allowed:
            return False  # This restricts opening new positions if we have reached the symbols_allowed limit
        elif current_symbol in open_symbols:
            return True  # This allows new positions on already traded symbols
        else:
            return self.open_symbols_count < symbols_allowed  # This checks if we can trade a new symbol

    # def can_trade_new_symbol(self, open_symbols: list, symbols_allowed: int, current_symbol: str) -> bool:
    #     """
    #     Checks if the bot can trade a given symbol.
    #     """
        
    #     self.open_symbols_count = len(open_symbols)  # Update the attribute with the current count

    #     logging.info(f"Open symbols count: {self.open_symbols_count}")
        
    #     if self.open_symbols_count >= symbols_allowed:
    #         return False  # This restricts opening new positions if we have reached the symbols_allowed limit
    #     elif current_symbol in open_symbols:
    #         return True  # This allows new positions on already traded symbols
    #     else:
    #         return self.open_symbols_count < symbols_allowed  # This checks if we can trade a new symbol

    # def can_trade_new_symbol(self, open_symbols: list, symbols_allowed: int, current_symbol: str) -> bool:
    #     """
    #     Checks if the bot can trade a given symbol.
    #     """
        
    #     self.open_symbols_count = len(open_symbols)  # Update the attribute with the current count

    #     logging.info(f"Open symbols count: {self.open_symbols_count}")
        
    #     if current_symbol in open_symbols:
    #         return True  # This allows new positions on already traded symbols
    #     else:
    #         return self.open_symbols_count < symbols_allowed  # This checks if we can trade a new symbol


    # def update_shared_data(self, symbol_data: dict, open_position_data: dict):
    #     # Update and serialize symbol data
    #     with open("symbol_data.json", "w") as f:
    #         json.dump(symbol_data, f)

    #     # Update and serialize open position data
    #     with open("open_positions_data.json", "w") as f:
    #         json.dump(open_position_data, f)

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

        # Check if the long position is close to being liquidated
        if long_pos_price is not None and long_liq_price is not None:
            long_diff = abs(long_pos_price - long_liq_price) / long_pos_price
            if long_diff < long_threshold:
                # Place a stop-loss order or open a position in the opposite direction to offset risk
                #self.place_stop_loss(symbol, "sell", amount, long_liq_price)  # Placeholder function, replace with your actual function
                logging.info(f"Placed a stop-loss order for long position on {symbol} at {long_liq_price}")

        # Check if the short position is close to being liquidated
        if short_pos_price is not None and short_liq_price is not None:
            short_diff = abs(short_pos_price - short_liq_price) / short_pos_price
            if short_diff < short_threshold:
                # Place a stop-loss order or open a position in the opposite direction to offset risk
                #self.place_stop_loss(symbol, "buy", amount, short_liq_price)  # Placeholder function, replace with your actual function
                logging.info(f"Placed a stop-loss order for short position on {symbol} at {short_liq_price}")

    def spoofing_action(self, symbol):
        if self.spoofing_active:
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = orderbook['bids'][0][0]
            best_ask_price = orderbook['asks'][0][0]

            spoofing_orders = []

            for i in range(self.spoofing_wall_size):
                spoof_price = best_bid_price - (i + 1) * 0.01  # Adjust the spoofing distance as needed
                spoof_price = Decimal(spoof_price).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
                spoof_amount = 0.1  # Adjust the spoofing order amount as needed
                spoof_order = self.limit_order(symbol, 'sell', spoof_amount, spoof_price)
                spoofing_orders.append(spoof_order)

                spoof_price = best_ask_price + (i + 1) * 0.01  # Adjust the spoofing distance as needed
                spoof_price = Decimal(spoof_price).quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)
                spoof_order = self.limit_order(symbol, 'buy', spoof_amount, spoof_price)
                spoofing_orders.append(spoof_order)

            time.sleep(self.spoofing_duration)

            for order in spoofing_orders:
                self.exchange.cancel_order_by_id(order['id'], symbol)

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
                    self.spoofing_action(symbol)

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

    def bybit_turbocharged_entry_maker_walls(self, symbol, trend, mfi, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):
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
            dynamic_long_amount = distance_to_tp_long * 10
            if trend.lower() == "long" and mfi.lower() == "long" and best_bid_price < long_pos_price:
                self.postonly_limit_order_bybit(symbol, "buy", dynamic_long_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Additional Long Entry Placed at {front_run_bid_price} with {dynamic_long_amount} amount!")

        # Check for short position and ensure take_profit_short is not None
        if short_pos_qty > 0 and take_profit_short:
            distance_to_tp_short = best_ask_price - take_profit_short
            dynamic_short_amount = distance_to_tp_short * 10
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

    def bybit_turbocharged_additional_entry_maker(self, symbol, trend, mfi, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_add_to_long, should_add_to_short):
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
            long_dynamic_amount += distance_to_tp_long * 10
            long_dynamic_amount = max(long_dynamic_amount, min_qty)

        if take_profit_short is not None:
            distance_to_tp_short = best_ask_price - take_profit_short
            short_dynamic_amount += distance_to_tp_short * 10
            short_dynamic_amount = max(short_dynamic_amount, min_qty)

        if long_pos_qty > 0 and take_profit_long:
            if trend.lower() == "long" and mfi.lower() == "long" and (long_pos_price is not None and best_bid_price < long_pos_price) and should_add_to_long:
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Additional Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        if short_pos_qty > 0 and take_profit_short:
            if trend.lower() == "short" and mfi.lower() == "short" and (short_pos_price is not None and best_ask_price > short_pos_price) and should_add_to_short:
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Additional Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")

    def bybit_turbocharged_entry_maker(self, symbol, trend, mfi, take_profit_long, take_profit_short, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_long, should_add_to_long, should_short, should_add_to_short):
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
            long_dynamic_amount += distance_to_tp_long * 10
            long_dynamic_amount = max(long_dynamic_amount, min_qty)

        if take_profit_short is not None:
            distance_to_tp_short = best_ask_price - take_profit_short
            short_dynamic_amount += distance_to_tp_short * 10
            short_dynamic_amount = max(short_dynamic_amount, min_qty)

        open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

        if long_pos_qty > 0 and take_profit_long:
            if trend.lower() == "long" and mfi.lower() == "long" and (long_pos_price is not None and best_bid_price < long_pos_price) and should_add_to_long and not self.entry_order_exists(open_orders, "buy"):
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Additional Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        if short_pos_qty > 0 and take_profit_short:
            if trend.lower() == "short" and mfi.lower() == "short" and (short_pos_price is not None and best_ask_price > short_pos_price) and should_add_to_short and not self.entry_order_exists(open_orders, "sell"):
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Additional Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")

        if long_pos_qty == 0:
            if trend.lower() == "long" and mfi.lower() == "long" and should_long and not self.entry_order_exists(open_orders, "buy"):
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        if short_pos_qty == 0:
            if trend.lower() == "short" and mfi.lower() == "short" and should_short and not self.entry_order_exists(open_orders, "sell"):
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")
                            
        # if long_pos_qty > 0 and take_profit_long:
        #     if trend.lower() == "long" and mfi.lower() == "long" and (long_pos_price is not None and best_bid_price < long_pos_price) and should_add_to_long:
        #         self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
        #         logging.info(f"Turbocharged Additional Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        # if short_pos_qty > 0 and take_profit_short:
        #     if trend.lower() == "short" and mfi.lower() == "short" and (short_pos_price is not None and best_ask_price > short_pos_price) and should_add_to_short:
        #         self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
        #         logging.info(f"Turbocharged Additional Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")

        # if long_pos_qty == 0:
        #     if trend.lower() == "long" and mfi.lower() == "long" and should_long:
        #         self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
        #         logging.info(f"Turbocharged Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        # if short_pos_qty == 0:
        #     if trend.lower() == "short" and mfi.lower() == "short" and should_short:
        #         self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
        #         logging.info(f"Turbocharged Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")

    def bybit_turbocharged_new_entry_maker(self, symbol, trend, mfi, long_dynamic_amount, short_dynamic_amount):
        self.order_book_analyzer = self.OrderBookAnalyzer(self.exchange, symbol)
        order_book = self.order_book_analyzer.get_order_book()

        best_ask_price = order_book['asks'][0][0]
        best_bid_price = order_book['bids'][0][0]

        market_data = self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)
        min_qty = float(market_data["min_qty"])

        largest_bid = max(order_book['bids'], key=lambda x: x[1])
        largest_ask = min(order_book['asks'], key=lambda x: x[1])
        
        spread = best_ask_price - best_bid_price
        front_run_bid_price = round(largest_bid[0] + (spread * 0.05), 4)  # front-run by 5% of the spread
        front_run_ask_price = round(largest_ask[0] - (spread * 0.05), 4)  # front-run by 5% of the spread

        position_data = self.exchange.get_positions_bybit(symbol)
        long_pos_qty = position_data["long"]["qty"]
        short_pos_qty = position_data["short"]["qty"]

        # Ensure the calculated amounts are not below the minimum order quantity
        long_dynamic_amount = max(long_dynamic_amount, min_qty)
        short_dynamic_amount = max(short_dynamic_amount, min_qty)

        # Entries for when there's no position yet
        if long_pos_qty == 0:
            if trend.lower() == "long" and mfi.lower() == "long":
                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, front_run_bid_price, positionIdx=1, reduceOnly=False)
                logging.info(f"Turbocharged Long Entry Placed at {front_run_bid_price} with {long_dynamic_amount} amount!")

        if short_pos_qty == 0:
            if trend.lower() == "short" and mfi.lower() == "short":
                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, front_run_ask_price, positionIdx=2, reduceOnly=False)
                logging.info(f"Turbocharged Short Entry Placed at {front_run_ask_price} with {short_dynamic_amount} amount!")

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

    def bybit_hedge_entry_maker_v3_initial_entry(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                open_orders = self.exchange.get_open_orders(symbol)

                if (trend.lower() == "long" and mfi.lower() == "long") and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                if (trend.lower() == "short" and mfi.lower() == "short") and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info("Placed initial short entry")

    # def bybit_hedge_additional_entry_maker_v3_multi(self, symbol: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

    #     max_long_qty, _ = self.max_long_trade_qty  # Unpacking tuple
    #     max_short_qty, _ = self.max_short_trade_qty  # Unpacking tuple
        
    #     best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
    #     best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
    #     open_orders = self.exchange.get_open_orders(symbol)

    #     if should_add_to_long and long_pos_qty < max_long_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
    #         logging.info(f"Managing non-rotating symbol: Placing additional long entry for {symbol}")
    #         self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

    #     if should_add_to_short and short_pos_qty < max_short_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
    #         logging.info(f"Managing non-rotating symbol: Placing additional short entry for {symbol}")
    #         self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
    def bybit_hedge_additional_entry_maker_v3_multi(self, symbol: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):

        if isinstance(self.max_long_trade_qty, tuple):
            max_long_qty, _ = self.max_long_trade_qty  # Unpacking tuple
        else:
            max_long_qty = self.max_long_trade_qty  # Assuming it's a float

        if isinstance(self.max_short_trade_qty, tuple):
            max_short_qty, _ = self.max_short_trade_qty  # Unpacking tuple
        else:
            max_short_qty = self.max_short_trade_qty  # Assuming it's a float

        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        open_orders = self.exchange.get_open_orders(symbol)

        if should_add_to_long and long_pos_qty < max_long_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Managing non-rotating symbol: Placing additional long entry for {symbol}")
            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        if should_add_to_short and short_pos_qty < max_short_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Managing non-rotating symbol: Placing additional short entry for {symbol}")
            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)


    def bybit_hedge_additional_entry_maker_v3(self, symbol: str, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_add_to_long: bool, should_add_to_short: bool):
        
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
        open_orders = self.exchange.get_open_orders(symbol)

        if should_add_to_long and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
            logging.info(f"Managing non-rotating symbol: Placing additional long entry for {symbol}")
            self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

        if should_add_to_short and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
            logging.info(f"Managing non-rotating symbol: Placing additional short entry for {symbol}")
            self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_entry_maker_v3_multi(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        max_long_qty, _ = self.max_long_trade_qty  # Unpacking tuple
        max_short_qty, _ = self.max_short_trade_qty  # Unpacking tuple

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
                
                open_orders = self.exchange.get_open_orders(symbol)

                # self.cancel_all_orders_for_symbol_bybit(symbol)
                
                if (trend.lower() == "long" and mfi.lower() == "long") and should_long and long_pos_qty == 0 and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing initial long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                    logging.info(f"Placed initial long entry")

                elif (trend.lower() == "long" and mfi.lower() == "long") and should_add_to_long and long_pos_qty < max_long_qty and best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                    logging.info(f"Placing additional long entry")
                    self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

                if (trend.lower() == "short" and mfi.lower() == "short") and should_short and short_pos_qty == 0 and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing initial short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)
                    logging.info(f"Placed initial short entry")

                elif (trend.lower() == "short" and mfi.lower() == "short") and should_add_to_short and short_pos_qty < max_short_qty and best_ask_price > short_pos_price and not self.entry_order_exists(open_orders, "sell"):
                    logging.info(f"Placing additional short entry")
                    self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2, reduceOnly=False)

    def bybit_hedge_entry_maker_v3(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:

                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
                
                open_orders = self.exchange.get_open_orders(symbol)

                # self.cancel_all_orders_for_symbol_bybit(symbol)
                
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

    # Revised consistent maker strategy using MA Trend OR MFI as well while maintaining same original MA logic
    def bybit_hedge_entry_maker_v2(self, symbol: str, trend: str, mfi: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, long_dynamic_amount: float, short_dynamic_amount: float, long_pos_qty: float, short_pos_qty: float, long_pos_price: float, short_pos_price: float, should_long: bool, should_short: bool, should_add_to_long: bool, should_add_to_short: bool):

        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                open_orders = self.exchange.get_open_orders(symbol)

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

# Bybit update take profit based on time and spread

    def update_take_profit_spread_bybit(self, symbol, pos_qty, take_profit_price, positionIdx, order_side, open_orders, next_tp_update):
        existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
        total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
        logging.info(f"Existing {order_side} TPs: {existing_tps}")
        now = datetime.now()
        if now >= next_tp_update or not math.isclose(total_existing_tp_qty, pos_qty):
            try:
                for qty, existing_tp_id in existing_tps:
                    self.exchange.cancel_order_by_id(existing_tp_id, symbol)
                    logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                    time.sleep(0.05)
                self.exchange.create_take_profit_order_bybit(symbol, "limit", order_side, pos_qty, take_profit_price, positionIdx=positionIdx, reduce_only=True)
                logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price}")
                next_tp_update = self.calculate_next_update_time()  # Calculate the next update time after placing the order
            except Exception as e:
                logging.info(f"Error in updating {order_side} TP: {e}")
        return next_tp_update

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
                self.postonly_limit_order_bybit(symbol, order_side, pos_qty, take_profit_price, positionIdx, reduceOnly=True)
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
                open_orders = self.exchange.get_open_orders(symbol)
                # Only placing additional long entries in GS mode
                if should_add_to_long and long_pos_qty < self.max_long_trade_qty and long_pos_price is not None:
                    if best_bid_price < long_pos_price and not self.entry_order_exists(open_orders, "buy"):
                        logging.info(f"Placing additional long entry for {symbol} in GS mode")
                        self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

    def short_entry_maker_gs(self, symbol: str, trend: str, one_minute_volume: float, five_minute_distance: float, min_vol: float, min_dist: float, short_dynamic_amount: float, short_pos_qty: float, short_pos_price: float, should_add_to_short: bool):
        best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
        
        if trend is not None and isinstance(trend, str) and trend.lower() == "short":
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                open_orders = self.exchange.get_open_orders(symbol)
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
        rotator_symbols = self.manager.get_auto_rotate_symbols()
        
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
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            eri_trend = api_data['ERI Trend']

            # Your existing code for each open_symbol
            # Fetch position data for the open symbol
            current_price = self.exchange.get_current_price(open_symbol)
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
            self.initial_max_long_trade_qty, self.initial_max_short_trade_qty = self.calc_max_trade_qty_multi(
                total_equity, best_ask_price_open_symbol, max_leverage)
            
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

            # # Replace the old function with the new function
            # long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount_multi(
            #     open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            # )
            
            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            # self.bybit_reset_position_leverage_long(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            # self.bybit_reset_position_leverage_short(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            # self.bybit_reset_position_leverage_long_v2(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            # self.bybit_reset_position_leverage_short_v2(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
        
            # Log the dynamic amounts
            # print(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            # print(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
            
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
                #self.bybit_reset_position_leverage_long_v2(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
                #self.bybit_reset_position_leverage_long(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            #self.bybit_reset_position_leverage_long_v2(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            #self.bybit_reset_position_leverage_short_v2(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
        
            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
                #self.bybit_reset_position_leverage_short_v2(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
                #self.bybit_reset_position_leverage_short(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            if open_symbol in open_symbols:
                # Note: When calling the `bybit_turbocharged_entry_maker` function, make sure to use these updated, context-specific variables.
                if is_rotator_symbol:  # Replace this with your own condition for switching between the two functions
                    self.bybit_turbocharged_entry_maker(
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


            # Cancel entries (Note: Replace this with the actual conditions for your open_symbol)
            #self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)

    # def manage_open_positions_multi(self, open_symbols, total_equity):
    #     # Get current rotator symbols
    #     rotator_symbols = self.manager.get_auto_rotate_symbols()
        
    #     logging.info(f"open_symbols in manage_open_positions: {open_symbols}")
        
    #     # Save the initial max trade quantities
    #     initial_max_long_trade_qty = self.max_long_trade_qty
    #     initial_max_short_trade_qty = self.max_short_trade_qty
        
    #     for open_symbol in open_symbols:
    #         # Check if the open symbol is in the rotator symbols
    #         is_rotator_symbol = open_symbol in rotator_symbols
            
    #         min_dist = self.config.min_distance
    #         min_vol = self.config.min_volume

    #         # Get API data
    #         api_data = self.manager.get_api_data(open_symbol)
    #         one_minute_volume = api_data['1mVol']
    #         five_minute_distance = api_data['5mSpread']
    #         trend = api_data['Trend']
    #         mfirsi_signal = api_data['MFI']
    #         eri_trend = api_data['ERI Trend']

    #         # Your existing code for each open_symbol
    #         current_price = self.exchange.get_current_price(open_symbol)
    #         market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
    #         max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
    #         position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
    #         long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
    #         short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]
            
    #         # Fetch the best ask and bid prices for the open symbol
    #         best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
    #         best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]
            
    #         # # Calculate the dynamic amounts for long and short positions
    #         # long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount_multi(
    #         #     open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
    #         # )
            
    #         long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
    #             open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
    #         )
            
    #         # Log the dynamic amounts
    #         logging.info(f"Long dynamic amount from manager for {open_symbol}: {long_dynamic_amount_open_symbol}")
    #         logging.info(f"Short dynamic amount from manager for {open_symbol}: {short_dynamic_amount_open_symbol}")

    #         # Restore the initial max trade quantities before updating leverages
    #         self.max_long_trade_qty = initial_max_long_trade_qty
    #         self.max_short_trade_qty = initial_max_short_trade_qty
            
    #         # Update the leverage for long and short positions
    #         self.bybit_reset_position_leverage_long_multi(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
    #         self.bybit_reset_position_leverage_short_multi(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            
    #         # Calculate your take profit levels for each open symbol (existing logic)
    #         long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
    #             position_data_open_symbol["long"]["price"], open_symbol
    #         )
    #         short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
    #             position_data_open_symbol["short"]["price"], open_symbol
    #         )

    #         current_price = self.exchange.get_current_price(open_symbol)
    #         market_data = self.get_market_data_with_retry(open_symbol, max_retries=5, retry_delay=5)
    #         max_leverage = self.exchange.get_max_leverage_bybit(open_symbol)
    #         position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
    #         long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
    #         short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

    #         min_qty = float(market_data["min_qty"])
    #         min_qty_str = str(min_qty)

    #         # Fetch the best ask and bid prices for the open symbol
    #         best_ask_price_open_symbol = self.exchange.get_orderbook(open_symbol)['asks'][0][0]
    #         best_bid_price_open_symbol = self.exchange.get_orderbook(open_symbol)['bids'][0][0]

    #         # Calculate moving averages for the open symbol
    #         moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)
    #         ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
    #         ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]
    #         ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
    #         ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
    #         ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
    #         ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
    #         ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
    #         ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

    #         # Calculate your take profit levels for each open symbol.
    #         short_take_profit_open_symbol = self.calculate_short_take_profit_spread_bybit(
    #             position_data_open_symbol["short"]["price"], open_symbol, five_minute_distance
    #         )
    #         long_take_profit_open_symbol = self.calculate_long_take_profit_spread_bybit(
    #             position_data_open_symbol["long"]["price"], open_symbol, five_minute_distance
    #         )

    #         # Additional context-specific variables
    #         long_pos_price_open_symbol = position_data_open_symbol["long"]["price"] if long_pos_qty_open_symbol > 0 else None
    #         short_pos_price_open_symbol = position_data_open_symbol["short"]["price"] if short_pos_qty_open_symbol > 0 else None

    #         # Additional context-specific variables
    #         should_long_open_symbol = self.long_trade_condition(best_bid_price_open_symbol, ma_3_low_open_symbol) if ma_3_low_open_symbol is not None else False
    #         should_short_open_symbol = self.short_trade_condition(best_ask_price_open_symbol, ma_6_high_open_symbol) if ma_3_high_open_symbol is not None else False

    #         should_add_to_long_open_symbol = (long_pos_price_open_symbol > ma_6_high_open_symbol) and should_long_open_symbol if long_pos_price_open_symbol is not None and ma_6_high_open_symbol is not None else False
    #         should_add_to_short_open_symbol = (short_pos_price_open_symbol < ma_6_low_open_symbol) and should_short_open_symbol if short_pos_price_open_symbol is not None and ma_6_low_open_symbol is not None else False

    #         # print(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")
    #         # logging.info(f"Calculating dynamic amount for {open_symbol} with market_data: {market_data}, total_equity: {total_equity}, best_ask_price_open_symbol: {best_ask_price_open_symbol}, max_leverage: {max_leverage}")

    #         # Log the dynamic amounts
    #         #print(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
    #         #print(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")
    #         logging.info(f"Long dynamic amount from manager for {open_symbol}: {long_dynamic_amount_open_symbol}")
    #         logging.info(f"Short dynamic amount from manager for {open_symbol}: {short_dynamic_amount_open_symbol}")
                                
    #         logging.info(f"Variables in manage_open_positions for {open_symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price_open_symbol={best_ask_price_open_symbol}, max_leverage={max_leverage}")

    #         # Force dynamic amounts to be min_qty
    #         long_dynamic_amount_open_symbol = min_qty
    #         short_dynamic_amount_open_symbol = min_qty

    #         # Entry logic based on whether the symbol is a rotator symbol or not
    #         if open_symbol in open_symbols:
    #             if is_rotator_symbol:
    #                 self.bybit_hedge_entry_maker_v3_multi(
    #                     open_symbol, trend, mfirsi_signal, one_minute_volume, five_minute_distance, min_vol, min_dist, 
    #                     long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, long_pos_qty_open_symbol, short_pos_qty_open_symbol, 
    #                     long_pos_price_open_symbol, short_pos_price_open_symbol, should_long_open_symbol, should_short_open_symbol, 
    #                     should_add_to_long_open_symbol, should_add_to_short_open_symbol
    #                 )
    #             else:
    #                 self.bybit_hedge_additional_entry_maker_v3_multi(
    #                     open_symbol, long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, long_pos_qty_open_symbol,
    #                     short_pos_qty_open_symbol, long_pos_price_open_symbol, short_pos_price_open_symbol, should_add_to_long_open_symbol,
    #                     should_add_to_short_open_symbol
    #                 )

    #             # Cancel existing orders based on your logic
    #             self.cancel_entries_bybit(open_symbol, best_ask_price_open_symbol, ma_1m_3_high_open_symbol, ma_5m_3_high_open_symbol)
                
    #     # At the end of the loop, you can optionally restore the initial values again, although it's not strictly necessary
    #     self.max_long_trade_qty = initial_max_long_trade_qty
    #     self.max_short_trade_qty = initial_max_short_trade_qty

    def manage_open_positions_v2(self, open_symbols, total_equity):
        # Get current rotator symbols
        rotator_symbols = self.manager.get_auto_rotate_symbols()
        
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume

        for open_symbol in open_symbols:
            # Check if the open symbol is NOT in the current rotator symbols
            if open_symbol not in rotator_symbols:
                logging.info(f"Symbol {open_symbol} is not in current rotator symbols. Managing it.")

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
            
            # Calculate dynamic amounts
            long_dynamic_amount_open_symbol, short_dynamic_amount_open_symbol, min_qty = self.calculate_dynamic_amount(
                open_symbol, market_data, total_equity, best_ask_price_open_symbol, max_leverage
            )

            # Fetch position data for the open symbol
            position_data_open_symbol = self.exchange.get_positions_bybit(open_symbol)
            long_pos_qty_open_symbol = position_data_open_symbol["long"]["qty"]
            short_pos_qty_open_symbol = position_data_open_symbol["short"]["qty"]

            # Adjust leverage based on the current position quantities
            if long_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_long_v3(open_symbol, long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
                #self.bybit_reset_position_leverage_long_v2(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
                #self.bybit_reset_position_leverage_long(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            #self.bybit_reset_position_leverage_long_v2(long_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
            #self.bybit_reset_position_leverage_short_v2(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
        
            if short_pos_qty_open_symbol > 0:
                self.bybit_reset_position_leverage_short_v3(open_symbol, short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
                #self.bybit_reset_position_leverage_short_v2(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)
                #self.bybit_reset_position_leverage_short(short_pos_qty_open_symbol, total_equity, best_ask_price_open_symbol, max_leverage)

            # Calculate moving averages for the open symbol
            moving_averages_open_symbol = self.get_all_moving_averages(open_symbol)

            ma_6_high_open_symbol = moving_averages_open_symbol["ma_6_high"]
            ma_6_low_open_symbol = moving_averages_open_symbol["ma_6_low"]
            ma_3_low_open_symbol = moving_averages_open_symbol["ma_3_low"]
            ma_3_high_open_symbol = moving_averages_open_symbol["ma_3_high"]
            ma_1m_3_high_open_symbol = moving_averages_open_symbol["ma_1m_3_high"]
            ma_5m_3_high_open_symbol = moving_averages_open_symbol["ma_5m_3_high"]

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

            # long_dynamic_amount_open_symbol = min_qty
            # short_dynamic_amount_open_symbol = min_qty

            if open_symbol in open_symbols:
                is_rotator_symbol = open_symbol in rotator_symbols
                if is_rotator_symbol:
                    self.bybit_hedge_entry_maker_v3(
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

            # Log the dynamic amounts
            logging.info(f"Long dynamic amount for {open_symbol}: {long_dynamic_amount_open_symbol}")
            logging.info(f"Short dynamic amount for {open_symbol}: {short_dynamic_amount_open_symbol}")

    def manage_open_positions(self, open_symbols, total_equity):
        # Get current rotator symbols
        rotator_symbols = self.manager.get_auto_rotate_symbols()
        
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

            # Your existing code for each open_symbol
            # Fetch position data for the open symbol
            current_price = self.exchange.get_current_price(open_symbol)
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
            self.initial_max_long_trade_qty, self.initial_max_short_trade_qty = self.calc_max_trade_qty_multi(
                total_equity, best_ask_price_open_symbol, max_leverage)

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

            long_dynamic_amount_open_symbol = min_qty
            short_dynamic_amount_open_symbol = min_qty

            if open_symbol in open_symbols:
                if is_rotator_symbol:
                    self.bybit_hedge_entry_maker_v3(
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

    def manage_gs_positions(self):
        quote_currency = "USDT"
        max_retries = 5
        retry_delay = 5

        while True:
            if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
                logging.info("Waiting for max trade quantities to be initialized...")
                time.sleep(10)  # Wait for 10 seconds before checking again
                continue

            # Get current rotator symbols and whitelist from config
            rotator_symbols = self.manager.get_auto_rotate_symbols()
            all_symbols = self.config.whitelist  # Adding this line to fetch the whitelist from config

            # Get all open positions
            open_positions = self.retry_api_call(self.exchange.get_all_open_positions_bybit)

            # Remove '/' from open symbols
            open_symbols = [symbol.replace('/', '') for symbol in self.extract_symbols_from_positions_bybit(open_positions)]

            # Only consider symbols that are both in the whitelist and have open positions
            symbols_to_check = [symbol for symbol in all_symbols if symbol in open_symbols]

            for symbol in symbols_to_check:
                if symbol not in rotator_symbols:
                    logging.info(f"Symbol {symbol} is no longer in rotation. Managing orders.")
                    market_data = self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)
                    
                    best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                    best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                    position_data = self.exchange.get_positions_bybit(symbol)
                    max_leverage = self.exchange.get_max_leverage_bybit(symbol)

                    data = self.manager.get_data()
                    one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
                    five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
                    trend = self.manager.get_asset_value(symbol, data, "Trend")
                    min_dist = self.config.min_distance
                    min_vol = self.config.min_volume
                    mfi = self.manager.get_asset_value(symbol, data, "MFI")

                    m_moving_averages = self.manager.get_1m_moving_averages(symbol)
                    m5_moving_averages = self.manager.get_5m_moving_averages(symbol)
                    ma_6_high = m_moving_averages["MA_6_H"]
                    ma_6_low = m_moving_averages["MA_6_L"]
                    ma_3_low = m_moving_averages["MA_3_L"]
                    ma_3_high = m_moving_averages["MA_3_H"]
                    ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
                    ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

                    should_short = self.short_trade_condition(best_ask_price, ma_3_high)
                    should_long = self.long_trade_condition(best_bid_price, ma_3_low)

                    short_pos_qty = position_data["short"]["qty"]
                    logging.info(f"GS Short pos qty: {short_pos_qty}")
                    long_pos_qty = position_data["long"]["qty"]
                    logging.info(f"GS Long pos qty: {long_pos_qty}")

                    short_pos_price = position_data["short"]["price"] if short_pos_qty > 0 else None
                    long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

                    if short_pos_price is not None:
                        should_add_to_short = short_pos_price < ma_6_low and self.short_trade_condition(best_ask_price, ma_6_high)

                    if long_pos_price is not None:
                        should_add_to_long = long_pos_price > ma_6_high and self.long_trade_condition(best_bid_price, ma_6_low)
                 
                    quote_currency = "USDT"

                    for i in range(max_retries):
                        try:
                            total_equity = self.exchange.get_balance_bybit(quote_currency)
                            break
                        except Exception as e:
                            if i < max_retries - 1:
                                logging.info(f"Error occurred while fetching balance: {e}. Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                raise e   
                            
                    min_qty = float(market_data["min_qty"])
                    min_qty_str = str(min_qty)

                    open_orders = self.exchange.get_open_orders(symbol)

                    order_side = None 

                    for side in ['long', 'short']:
                        if side == 'long' and long_pos_qty > 0:
                            current_pos_price = long_pos_price
                            current_pos_qty = long_pos_qty
                            order_side = "sell"
                            positionIdx = 1
                            self.long_entry_maker_gs(symbol, trend, one_minute_volume, five_minute_distance, min_vol, min_dist, min_qty, current_pos_qty, current_pos_price, should_add_to_long)
                            #self.long_entry_maker(symbol, trend, one_minute_volume, five_minute_distance, min_vol, min_dist, min_qty, current_pos_qty, current_pos_price, should_long, should_add_to_long)
                            take_profit_price = self.calculate_long_take_profit_spread_bybit(current_pos_price, symbol, five_minute_distance)
                        elif side == 'short' and short_pos_qty > 0:
                            current_pos_price = short_pos_price
                            current_pos_qty = short_pos_qty
                            order_side = "buy"
                            positionIdx = 2
                            self.short_entry_maker_gs(symbol, trend, one_minute_volume, five_minute_distance, min_vol, min_dist, min_qty, current_pos_qty, current_pos_price, should_add_to_short)
                            #self.short_entry_maker(symbol, trend, one_minute_volume, five_minute_distance, min_vol, min_dist, min_qty, current_pos_qty, current_pos_price, should_short, should_add_to_short)
                            take_profit_price = self.calculate_short_take_profit_spread_bybit(current_pos_price, symbol, five_minute_distance)
                        else:
                            continue

                        if take_profit_price and current_pos_qty is not None:
                            # Check for existing take profit orders
                            existing_tps = self.get_open_take_profit_order_quantities(open_orders, order_side)
                            total_existing_tp_qty = sum(qty for qty, _ in existing_tps)
                            logging.info(f"Existing {order_side} TPs: {existing_tps}")

                            # Cancel existing TP orders if their quantities do not match the current position quantity
                            for qty, existing_tp_id in existing_tps:
                                if not math.isclose(qty, current_pos_qty):
                                    try:
                                        self.exchange.cancel_order_by_id(existing_tp_id, symbol)
                                        logging.info(f"{order_side.capitalize()} take profit {existing_tp_id} canceled")
                                    except Exception as e:
                                        logging.info(f"Error in cancelling {order_side} TP orders: {e}")

                            # Place a new TP order if none exist
                            if len(existing_tps) < 1:
                                try:
                                    self.postonly_limit_order_bybit(symbol, order_side, current_pos_qty, take_profit_price, positionIdx, reduceOnly=True)
                                    logging.info(f"{order_side.capitalize()} take profit set at {take_profit_price}")
                                except Exception as e:
                                    logging.info(f"Error in placing {order_side} TP: {e}")

                            self.cancel_entries_bybit(symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high)
                    
            time.sleep(300)

    # Bybit cancel all entries
    def cancel_entries_bybit(self, symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high):
        # Cancel entries
        current_time = time.time()
        if current_time - self.last_cancel_time >= 60:  # Execute this block every 1 minute
            try:
                if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                    self.exchange.cancel_all_entries_bybit(symbol)
                    logging.info(f"Canceled entry orders for {symbol}")
                    time.sleep(0.05)
            except Exception as e:
                logging.info(f"An error occurred while canceling entry orders: {e}")

            self.last_cancel_time = current_time

    def cancel_stale_orders_bybit(self):
        current_time = time.time()
        if current_time - self.last_stale_order_check_time < 1800:  # 300 seconds = 5 minutes
            return  # Skip the rest of the function if it's been less than 5 minutes

        all_open_orders = self.exchange.get_all_open_orders_bybit()
        open_position_data = self.exchange.get_all_open_positions_bybit()
        open_symbols = self.extract_symbols_from_positions_bybit(open_position_data)
        open_symbols = [symbol.replace("/", "") for symbol in open_symbols]
        rotator_symbols = self.manager.get_auto_rotate_symbols()
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
            self.initial_max_long_trade_qty_per_symbol[symbol] = self.calc_max_trade_qty(total_equity, best_ask_price, max_leverage)
            self.long_pos_leverage_per_symbol[symbol] = 1.0  # starting leverage

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
            self.initial_max_short_trade_qty_per_symbol[symbol] = self.calc_max_trade_qty(total_equity, best_ask_price, max_leverage)
            self.short_pos_leverage_per_symbol[symbol] = 1.0  # starting leverage

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


    def bybit_reset_position_leverage_long_v2(self, long_pos_qty, total_equity, best_ask_price, max_leverage):
        if long_pos_qty >= self.initial_max_long_trade_qty and self.long_pos_leverage < self.MAX_LEVERAGE:
            self.max_long_trade_qty, self.long_pos_leverage = self.adjust_leverage_and_qty(
                self.max_long_trade_qty, self.initial_max_long_trade_qty, self.long_pos_leverage, max_leverage, increase=True
            )
            logging.info(f"Long leverage temporarily increased to {self.long_pos_leverage}x")
        elif long_pos_qty < (self.max_long_trade_qty / 2) and self.long_pos_leverage > 1.0:
            self.max_long_trade_qty, self.long_pos_leverage = self.adjust_leverage_and_qty(
                self.max_long_trade_qty, self.initial_max_long_trade_qty, self.long_pos_leverage, max_leverage, increase=False
            )
            logging.info(f"Long leverage returned to normal {self.long_pos_leverage}x")

    def bybit_reset_position_leverage_short_v2(self, short_pos_qty, total_equity, best_ask_price, max_leverage):
        if short_pos_qty >= self.initial_max_short_trade_qty and self.short_pos_leverage < self.MAX_LEVERAGE:
            self.max_short_trade_qty, self.short_pos_leverage = self.adjust_leverage_and_qty(
                self.max_short_trade_qty, self.initial_max_short_trade_qty, self.short_pos_leverage, max_leverage, increase=True
            )
            logging.info(f"Short leverage temporarily increased to {self.short_pos_leverage}x")
        elif short_pos_qty < (self.max_short_trade_qty / 2) and self.short_pos_leverage > 1.0:
            self.max_short_trade_qty, self.short_pos_leverage = self.adjust_leverage_and_qty(
                self.max_short_trade_qty, self.initial_max_short_trade_qty, self.short_pos_leverage, max_leverage, increase=False
            )
            logging.info(f"Short leverage returned to normal {self.short_pos_leverage}x")


# Bybit position leverage management

    def bybit_reset_position_leverage_long(self, long_pos_qty, total_equity, best_ask_price, max_leverage):
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
            max_trade_qty = self.calc_max_trade_qty(total_equity, best_ask_price, max_leverage)
            if isinstance(max_trade_qty, float):
                self.max_long_trade_qty = max_trade_qty
            else:
                logging.error(f"Expected max_trade_qty to be float, got {type(max_trade_qty)}")
            self.long_leverage_increased = False
            self.long_pos_leverage = 1.0
            logging.info(f"Long leverage returned to normal {self.long_pos_leverage}x")

    def bybit_reset_position_leverage_short(self, short_pos_qty, total_equity, best_ask_price, max_leverage):
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
            max_trade_qty = self.calc_max_trade_qty(total_equity, best_ask_price, max_leverage)
            if isinstance(max_trade_qty, float):
                self.max_short_trade_qty = max_trade_qty
            else:
                logging.error(f"Expected max_trade_qty to be float, got {type(max_trade_qty)}")
            self.short_leverage_increased = False
            self.short_pos_leverage = 1.0
            logging.info(f"Short leverage returned to normal {self.short_pos_leverage}x")

    # def bybit_reset_position_leverage_long(self, long_pos_qty, total_equity, best_ask_price, max_leverage):
    #     # Leverage increase logic for long positions
    #     if long_pos_qty >= self.initial_max_long_trade_qty and self.long_pos_leverage <= 1.0:
    #         self.max_long_trade_qty = 2 * self.initial_max_long_trade_qty  # double the maximum long trade quantity
    #         self.long_leverage_increased = True
    #         self.long_pos_leverage = 2.0
    #         logging.info(f"Long leverage for temporarily increased to {self.long_pos_leverage}x")
    #     elif long_pos_qty >= 2 * self.initial_max_long_trade_qty and self.long_pos_leverage <= 2.0:
    #         self.max_long_trade_qty = 3 * self.initial_max_long_trade_qty  # triple the maximum long trade quantity
    #         self.long_pos_leverage = 3.0
    #         logging.info(f"Long leverage temporarily increased to {self.long_pos_leverage}x")
    #     elif long_pos_qty < (self.max_long_trade_qty / 2) and self.long_pos_leverage > 1.0:
    #         self.max_long_trade_qty = self.calc_max_trade_qty(total_equity,
    #                                                         best_ask_price,
    #                                                         max_leverage)
    #         self.long_leverage_increased = False
    #         self.long_pos_leverage = 1.0
    #         logging.info(f"Long leverage returned to normal {self.long_pos_leverage}x")

    # def bybit_reset_position_leverage_short(self, short_pos_qty, total_equity, best_ask_price, max_leverage):
    #     # Leverage increase logic for short positions
    #     if short_pos_qty >= self.initial_max_short_trade_qty and self.short_pos_leverage <= 1.0:
    #         self.max_short_trade_qty = 2 * self.initial_max_short_trade_qty  # double the maximum short trade quantity
    #         self.short_leverage_increased = True
    #         self.short_pos_leverage = 2.0
    #         logging.info(f"Short leverage temporarily increased to {self.short_pos_leverage}x")
    #     elif short_pos_qty >= 2 * self.initial_max_short_trade_qty and self.short_pos_leverage <= 2.0:
    #         self.max_short_trade_qty = 3 * self.initial_max_short_trade_qty  # triple the maximum short trade quantity
    #         self.short_pos_leverage = 3.0
    #         logging.info(f"Short leverage temporarily increased to {self.short_pos_leverage}x")
    #     elif short_pos_qty < (self.max_short_trade_qty / 2) and self.short_pos_leverage > 1.0:
    #         self.max_short_trade_qty = self.calc_max_trade_qty(total_equity,
    #                                                         best_ask_price,
    #                                                         max_leverage)
    #         self.short_leverage_increased = False
    #         self.short_pos_leverage = 1.0
    #         logging.info(f"Short leverage returned to normal {self.short_pos_leverage}x")

    def bybit_reset_position_leverage_long_multi(self, long_pos_qty, total_equity, best_ask_price, max_leverage):
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
            self.max_long_trade_qty, _ = self.calc_max_trade_qty_multi(total_equity,
                                                                    best_ask_price,
                                                                    max_leverage)
            self.long_leverage_increased = False
            self.long_pos_leverage = 1.0
            logging.info(f"Long leverage returned to normal {self.long_pos_leverage}x")

    def bybit_reset_position_leverage_short_multi(self, short_pos_qty, total_equity, best_ask_price, max_leverage):
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
            _, self.max_short_trade_qty = self.calc_max_trade_qty_multi(total_equity,
                                                                        best_ask_price,
                                                                        max_leverage)
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
