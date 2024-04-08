import uuid
from .exchange import Exchange
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from ccxt.base.errors import RateLimitExceeded
import math

class LBankExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('lbank', api_key, secret_key, passphrase, market_type)

    def get_balance_lbank(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                balance_response = self.exchange.fetch_balance()
                if quote in balance_response['total']:
                    total_balance = balance_response['total'][quote]
                    return total_balance
                else:
                    logging.info(f"Balance for {quote} not found in the response.")
            except Exception as e:
                logging.error(f"Error fetching balance from LBank: {e}")
        return None

    def create_limit_order_lbank(self, symbol: str, side: str, qty: float, price: float):
        try:
            if side == "buy" or side == "sell":
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=qty,
                    price=price
                )
                return order
            else:
                logging.info(f"side {side} does not exist")
                return {"error": f"side {side} does not exist"}
        except Exception as e:
            logging.info(f"An unknown error occurred in create_limit_order() for {symbol}: {e}")
            return {"error": str(e)}

    def create_market_order_lbank(self, symbol: str, side: str, qty: float):
        try:
            if side == "buy" or side == "sell":
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=qty
                )
                return order
            else:
                logging.info(f"Side {side} does not exist")
                return {"error": f"Side {side} does not exist"}
        except Exception as e:
            logging.info(f"An unknown error occurred in create_market_order(): {e}")
            return {"error": str(e)}

    def cancel_order_lbank(self, order_id, symbol):
        try:
            response = self.exchange.cancel_order(order_id, symbol)
            logging.info(f"Order {order_id} for {symbol} cancelled successfully.")
            return response
        except Exception as e:
            logging.error(f"An error occurred while cancelling order {order_id} for {symbol}: {str(e)}")
            return None

    def fetch_recent_trades_lbank(self, symbol, since=None, limit=100):
        try:
            self.exchange.load_markets()
            trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)
            return trades
        except Exception as e:
            logging.error(f"Error fetching recent trades for {symbol}: {e}")
            return []