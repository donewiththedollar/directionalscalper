import uuid
from .exchange import Exchange
import logging
import time

class HuobiExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('huobi', api_key, secret_key, passphrase, market_type)

    def fetch_balance_huobi(self, params={}):
        try:
            balance = self.exchange.fetch_balance(params)
            logging.info(f"Fetched balance from Huobi: {balance}")
            return balance
        except Exception as e:
            logging.error(f"Error occurred while fetching balance from Huobi: {e}")
            return None
        
    def cancel_close_huobi(self, symbol: str, side: str, offset: str) -> None:
        side_map = {"long": "buy", "short": "sell"}
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            print(f"Orders: {orders}")
            if orders:
                for order in orders:
                    order_info = order["info"]
                    order_id = order_info["order_id"]
                    order_status = order_info["status"]
                    order_direction = order_info["direction"]
                    order_offset = order_info["offset"]
                    reduce_only = order_info["reduce_only"]

                    if (
                        order_status == '3'  # Assuming '3' represents open orders
                        and order_direction == side_map[side]
                        and order_offset == offset
                        and reduce_only == '1'  # Assuming '1' represents reduce_only orders
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_close_huobi(): {e}")

    def create_take_profit_order_huobi(self, symbol, order_type, side, amount, price=None, reduce_only=False):
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")

            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")

            params = {"offset": "close" if reduce_only else "open"}
            return self.exchange.create_order(symbol, order_type, side, amount, price, params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def create_contract_order_huobi(self, symbol, order_type, side, amount, price=None, params={}):
        params = {'leverRate': 50}
        return self.exchange.create_contract_order(symbol, order_type, side, amount, price, params)

    def huobi_test_orders(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            print(orders)
        except Exception as e:
            print(f"Exception caught {e}")

    def cancel_all_entries_huobi(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            long_orders = 0
            short_orders = 0

            # Count the number of open long and short orders
            for order in orders:
                order_info = order["info"]
                order_status = str(order_info["status"])  # status seems to be a string of a number
                order_direction = order_info["direction"]
                order_offset = order_info["offset"]

                if order_status != "4" and order_status != "6":  # Assuming 4 is 'Filled' and 6 is 'Cancelled'
                    if order_direction == "buy" and order_offset == "open":
                        long_orders += 1
                    elif order_direction == "sell" and order_offset == "open":
                        short_orders += 1

            # Cancel extra long or short orders if more than one open order per side
            if long_orders > 1 or short_orders > 1:
                for order in orders:
                    order_info = order["info"]
                    order_id = order_info["order_id"]  # It's 'order_id' in Huobi, not 'orderId'
                    order_status = str(order_info["status"])
                    order_direction = order_info["direction"]
                    order_offset = order_info["offset"]

                    if order_status != "4" and order_status != "6":
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.info(f"An unknown error occurred in cancel_entry(): {e}")
