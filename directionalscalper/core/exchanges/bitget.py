import uuid
from .exchange import Exchange
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

class BitgetExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('bitget', api_key, secret_key, passphrase, market_type)
    
    def get_balance_bitget(self, quote, account_type='futures'):
        if account_type == 'futures':
            if self.exchange.has['fetchBalance']:
                # Fetch the balance
                balance = self.exchange.fetch_balance(params={'type': 'swap'})

                for currency_balance in balance['info']:
                    if currency_balance['marginCoin'] == quote:
                        return float(currency_balance['equity'])
        else:
            # Handle other account types or fallback to default behavior
            pass

    def get_open_orders_bitget(self, symbol: str) -> list:
        open_orders = []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            #print(f"Raw orders: {orders}")  # Add this line to print raw orders
            for order in orders:
                if "info" in order:
                    info = order["info"]
                    if "state" in info and info["state"] == "new":  # Change "status" to "state"
                        order_data = {
                            "id": info.get("orderId", ""),  # Change "order_id" to "orderId"
                            "price": info.get("price", 0.0),  # Use the correct field name
                            "qty": info.get("size", 0.0),  # Change "qty" to "size"
                            "side": info.get("side", ""),
                            "reduce_only": info.get("reduceOnly", False),
                        }
                        open_orders.append(order_data)
        except Exception as e:
            logging.info(f"An unknown error occurred in get_open_orders_debug(): {e}")
        return open_orders

    def get_max_leverage_bitget(self, symbol):
        try:
            # Fetch market leverage tiers
            leverage_tiers = self.exchange.fetch_market_leverage_tiers(symbol)

            # Extract maximum leverage from the tiers
            max_leverage = 0
            for tier in leverage_tiers:
                tier_leverage = tier['maxLeverage']
                if tier_leverage > max_leverage:
                    max_leverage = tier_leverage

            return max_leverage

        except Exception as e:
            logging.info(f"An error occurred while fetching max leverage: {e}")
            return None

    # Bitget
    def cancel_all_entries_bitget(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            long_orders = 0
            short_orders = 0

            # Count the number of open long and short orders
            for order in orders:
                order_info = order["info"]
                order_status = order_info["state"]
                order_side = order_info["side"]
                reduce_only = order_info["reduceOnly"]
                
                if order_status != "Filled" and order_status != "Cancelled" and not reduce_only:
                    if order_side == "open_long":
                        long_orders += 1
                    elif order_side == "open_short":
                        short_orders += 1

            # Cancel extra long or short orders if more than one open order per side
            if long_orders > 1 or short_orders > 1:
                for order in orders:
                    order_info = order["info"]
                    order_id = order_info["orderId"]
                    order_status = order_info["state"]
                    order_side = order_info["side"]
                    reduce_only = order_info["reduceOnly"]

                    if (
                        order_status != "Filled"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_entry(): {e}")

    def get_open_take_profit_order_quantity_bitget(self, orders, side):
        for order in orders:
            if order['side'] == side and order['params'].get('reduceOnly', False):
                return order['amount']
        return None

    # Bitget
    def get_order_status_bitget(self, symbol, side):
        open_orders = self.exchange.fetch_open_orders(symbol)

        for order in open_orders:
            if order['side'] == side:
                return order['status']

        return None

    # Bitget
    def cancel_entry_bitget(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            
            for order in orders:
                order_info = order["info"]
                order_id = order_info["orderId"]
                order_status = order_info["state"]
                order_side = order_info["side"]
                reduce_only = order_info["reduceOnly"]
                
                if (
                    order_status != "Filled"
                    and order_status != "Cancelled"
                    and not reduce_only
                ):
                    self.exchange.cancel_order(symbol=symbol, id=order_id)
                    logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_entry(): {e}")

    def cancel_close_bitget(self, symbol: str, side: str) -> None:
        side_map = {"long": "close_long", "short": "close_short"}
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        order_id = order["info"]["orderId"]
                        order_status = order["info"]["state"]
                        order_side = order["info"]["side"]
                        reduce_only = order["info"]["reduceOnly"]

                        if (
                            order_status != "filled"
                            and order_side == side_map[side]
                            and order_status != "canceled"
                            and reduce_only
                        ):
                            self.exchange.cancel_order(symbol=symbol, id=order_id)
                            logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_close_bitget(): {e}")

    def market_close_position_bitget(self, symbol, side, amount):
        """
        Close a position by creating a market order in the opposite direction.
        
        :param str symbol: Symbol of the market to create an order in.
        :param str side: Original side of the position. Either 'buy' (for long positions) or 'sell' (for short positions).
        :param float amount: The quantity of the position to close.
        """
        # Determine the side of the closing order based on the original side of the position
        if side == "buy":
            close_side = "sell"
        elif side == "sell":
            close_side = "buy"
        else:
            raise ValueError("Invalid order side. Must be either 'buy' or 'sell'.")

        # Create a market order in the opposite direction to close the position
        self.create_order(symbol, 'market', close_side, amount)
