import uuid
from .exchange import Exchange
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from ccxt.base.errors import RateLimitExceeded, NetworkError
import ccxt
import traceback
from directionalscalper.core.strategies.logger import Logger

from rate_limit import RateLimit

logging = Logger(logger_name="BlofinExchange", filename="BlofinExchange.log", stream=True)

class BlofinExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('blofin', api_key, secret_key, passphrase, market_type)
        
        self.max_retries = 100  # Maximum retries for rate-limited requests
        self.retry_wait = 5  # Seconds to wait between retries
        self.last_active_long_order_time = {}
        self.last_active_short_order_time = {}
        self.last_active_time = {}
        self.rate_limiter = RateLimit(10, 1)

    def log_order_active_times(self):
        try:
            current_time = time.time()
            for symbol, last_active_long in self.last_active_long_order_time.items():
                time_since_active_long = current_time - last_active_long
                logging.info(f"Long orders for symbol {symbol} were last active {time_since_active_long:.2f} seconds ago.")

            for symbol, last_active_short in self.last_active_short_order_time.items():
                time_since_active_short = current_time - last_active_short
                logging.info(f"Short orders for symbol {symbol} were last active {time_since_active_short:.2f} seconds ago.")
        except Exception as e:
            logging.info(f"Last order time exception {e}")

    def initialize_symbol_monitoring(self, symbol):
        if symbol not in self.last_active_time:
            self.last_active_time[symbol] = time.time()
            logging.info(f"Started monitoring {symbol} at {self.last_active_time[symbol]}")
     
    def get_symbol_info_and_positions(self, symbol: str):
        try:
            market = self.exchange.market(symbol)
            logging.info(f"Symbol: {market['symbol']}")
            logging.info(f"Base: {market['base']}")
            logging.info(f"Quote: {market['quote']}")
            logging.info(f"Type: {market['type']}")
            logging.info(f"Settle: {market['settle']}")

            positions = self.exchange.fetch_positions([symbol])
            for position in positions:
                logging.info(f"Position Info:")
                logging.info(f"Symbol: {position['symbol']}")
                logging.info(f"Side: {position['side']}")
                logging.info(f"Amount: {position['amount']}")
                logging.info(f"Entry Price: {position['entryPrice']}")
                logging.info(f"Unrealized PNL: {position['unrealizedPnl']}")
                logging.info(f"Leverage: {position['leverage']}")
                logging.info(f"Margin Type: {position['marginType']}")
                logging.info(f"Liquidation Price: {position['liquidationPrice']}")
            return positions
        except Exception as e:
            logging.info(f"Error fetching symbol info and positions: {e}")
            return []

    def get_market_data_blofin(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)
            if "info" in symbol_data:
                values["precision"] = symbol_data["precision"]["price"]
                values["min_qty"] = symbol_data["limits"]["amount"]["min"]

            positions = self.exchange.fetch_positions()
            for position in positions:
                if position['symbol'] == symbol:
                    values["leverage"] = float(position['leverage'])
        except Exception as e:
            logging.info(f"An unknown error occurred in get_market_data_blofin(): {e}")
        return values

    def get_best_bid_ask_blofin(self, symbol):
        try:
            orderbook = self.exchange.fetch_order_book(symbol)
            best_ask_price = orderbook['asks'][0][0] if orderbook['asks'] else None
            best_bid_price = orderbook['bids'][0][0] if orderbook['bids'] else None
            return best_bid_price, best_ask_price
        except Exception as e:
            logging.info(f"An unknown error occurred in get_best_bid_ask_blofin(): {e}")
            return None, None

    def get_all_open_orders_blofin(self):
        try:
            all_open_orders = self.exchange.fetch_open_orders()
            return all_open_orders
        except Exception as e:
            logging.info(f"An error occurred while fetching all open orders: {e}")
            return []

    def get_balance_blofin(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                balance_response = self.exchange.fetch_balance({'type': 'swap'})
                if quote in balance_response['total']:
                    total_balance = balance_response['total'][quote]
                    return total_balance
                else:
                    logging.info(f"Balance for {quote} not found in the response.")
            except Exception as e:
                logging.info(f"Error fetching balance from Blofin: {e}")
        return None

    def get_available_balance_blofin(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                balance_response = self.exchange.fetch_balance({'type': 'swap'})
                if 'free' in balance_response and quote in balance_response['free']:
                    return float(balance_response['free'][quote])
                else:
                    logging.warning(f"Available balance for {quote} not found in the response.")
            except Exception as e:
                logging.info(f"Error fetching available balance from Blofin: {e}")
        return None

    def get_balance_blofin_unified(self, quote):
        if self.exchange.has['fetchBalance']:
            balance = self.exchange.fetch_balance()
            unified_balance = balance.get('USDT', {})
            total_balance = unified_balance.get('total', None)
            if total_balance is not None:
                return float(total_balance)
        return None

    def create_limit_order_blofin(self, symbol: str, side: str, qty: float, price: float, positionIdx=0, params={}):
        try:
            if side == "buy" or side == "sell":
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=qty,
                    price=price,
                    params={**params, 'positionIdx': positionIdx}
                )
                return order
            else:
                logging.info(f"side {side} does not exist")
                return {"error": f"side {side} does not exist"}
        except Exception as e:
            logging.info(f"An unknown error occurred in create_limit_order_blofin() for {symbol}: {e}")
            return {"error": str(e)}

    def create_limit_order_blofin_spot(self, symbol: str, side: str, qty: float, price: float, isLeverage=0, orderLinkId=None):
        try:
            params = {
                'timeInForce': 'PostOnly',
                'isLeverage': isLeverage,
            }
            if orderLinkId:
                params['orderLinkId'] = orderLinkId
            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=qty,
                price=price,
                params=params
            )
            return order
        except Exception as e:
            logging.info(f"An error occurred while creating limit order on Blofin: {e}")
            return None

    def create_tagged_limit_order_blofin(self, symbol: str, side: str, qty: float, price: float, positionIdx=0, isLeverage=False, orderLinkId=None, postOnly=True, params={}):
        try:
            order_type = "limit"
            time_in_force = "PostOnly" if postOnly else "GTC"
            extra_params = {
                "positionIdx": positionIdx,
                "timeInForce": time_in_force
            }
            if isLeverage:
                extra_params["isLeverage"] = 1
            if orderLinkId:
                extra_params["orderLinkId"] = orderLinkId
            extra_params.update(params)
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=qty,
                price=price,
                params=extra_params
            )
            current_time = time.time()
            if side.lower() == 'buy':
                self.last_active_long_order_time[symbol] = current_time
                logging.info(f"Logged long order time for {symbol}")
            elif side.lower() == 'sell':
                self.last_active_short_order_time[symbol] = current_time
                logging.info(f"Logged short order time for {symbol}")
            return order
        except Exception as e:
            logging.info(f"An error occurred in create_tagged_limit_order_blofin() for {symbol}: {e}")
            return {"error": str(e)}

    def create_limit_order_blofin_unified(self, symbol: str, side: str, qty: float, price: float, positionIdx=0, params={}):
        try:
            if side == "buy" or side == "sell":
                order = self.exchange.create_unified_account_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=qty,
                    price=price,
                    params={**params, 'positionIdx': positionIdx}
                )
                return order
            else:
                logging.info(f"side {side} does not exist")
        except Exception as e:
            logging.info(f"An unknown error occurred in create_limit_order_blofin(): {e}")

    def create_market_order_blofin(self, symbol: str, side: str, qty: float, positionIdx=0, params={}):
        try:
            if side == "buy" or side == "sell":
                request = {
                    'symbol': symbol,
                    'type': 'market',
                    'side': side,
                    'qty': qty,
                    'positionIdx': positionIdx,
                    'closeOnTrigger': True
                }
                order = self.exchange.create_order(symbol, 'market', side, qty, params=request)
                return order
            else:
                logging.info(f"Side {side} does not exist")
        except Exception as e:
            logging.info(f"An unknown error occurred in create_market_order_blofin(): {e}")

    def cancel_all_open_orders_blofin(self, symbol=None, category="linear"):
        try:
            logging.info(f"cancel_all_open_orders_blofin called")
            params = {'category': category}
            if symbol is not None:
                market = self.exchange.market(symbol)
                params['symbol'] = market['id']
            response = self.exchange.cancel_all_orders(params=params)
            logging.info(f"Successfully cancelled orders {response}")
            return response
        except Exception as e:
            logging.info(f"Error cancelling orders: {e}")
            
    def cancel_order_blofin(self, order_id, symbol):
        try:
            response = self.exchange.cancel_order(order_id, symbol)
            logging.info(f"Order {order_id} for {symbol} cancelled successfully.")
            return response
        except Exception as e:
            logging.info(f"An error occurred while cancelling order {order_id} for {symbol}: {str(e)}")
            return None
        
    def get_precision_and_limits_blofin(self, symbol):
        markets = self.exchange.fetch_markets()
        for market in markets:
            if market['symbol'] == symbol:
                precision_amount = market['precision']['amount']
                precision_price = market['precision']['price']
                min_amount = market['limits']['amount']['min']
                return precision_amount, precision_price, min_amount
        return None, None, None

    def get_market_precision_data_blofin(self, symbol):
        markets = self.exchange.fetch_markets()
        logging.info(markets[0])
        for market in markets:
            if market['symbol'] == symbol:
                return market['precision']
        return None
