import uuid
from .exchange import Exchange
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from ccxt.base.errors import RateLimitExceeded
import math

class BinanceExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('binance', api_key, secret_key, passphrase, market_type)

    def set_hedge_mode_binance(self):
        """
        set hedged to True for the account
        :returns dict: response from the exchange
        """
        try:
            response = self.exchange.set_position_mode(True)
            return response
        except Exception as e:
            print(f"An error occurred while setting position mode: {e}")

    # Binance
    def create_limit_order_binance(self, symbol: str, side: str, qty: float, price: float, params={}):
        try:
            if side == "buy" or side == "sell":
                user_position_side = "LONG"  # Replace this with the actual user's position side setting
                params["positionSide"] = user_position_side
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='LIMIT',
                    side=side,
                    amount=qty,
                    price=price,
                    params=params
                )
                return order
            else:
                logging.warning(f"side {side} does not exist")
        except Exception as e:
            logging.warning(f"An unknown error occurred in create_limit_order_binance(): {e}")

    # Binance
    def create_close_position_limit_order_binance(self, symbol: str, side: str, qty: float, price: float):
        try:
            if side == "buy" or side == "sell":
                position_side = "LONG" if side == "sell" else "SHORT"
                params = {
                    "positionSide": position_side,
                    "closePosition": True
                }
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='LIMIT',
                    side=side,
                    amount=qty,
                    price=price,
                    params=params
                )
                return order
            else:
                logging.warning(f"Invalid side: {side}")
        except Exception as e:
            logging.warning(f"An unknown error occurred in create_close_position_limit_order_binance(): {e}")

    # Binance
    def create_take_profit_order_binance(self, symbol, side, amount, price):
        if side not in ["buy", "sell"]:
            raise ValueError(f"Invalid side: {side}")

        params = {"closePosition": True}

        # Create the limit order for the take profit
        order = self.create_limit_order_binance(symbol, side, amount, price, params)

        return order

    def create_normal_take_profit_order_binance(self, symbol, side, quantity, price, stopPrice):
        params = {
            'stopPrice': stopPrice,  # the price at which the order is triggered
            'type': 'TAKE_PROFIT'  # specifies the type of the order
        }
        return self.exchange.create_order(symbol, 'limit', side, quantity, price, params)

    def binance_create_limit_order(self, symbol: str, side: str, amount: float, price: float, params={}):
        params["positionSide"] = "LONG" if side.lower() == "buy" else "SHORT"  # set positionSide parameter
        try:
            order = self.exchange.create_order(symbol, "limit", side, amount, price, params)
            return order
        except Exception as e:
            print(f"An error occurred while creating the limit order: {e}")

    def binance_create_limit_order_with_time_in_force(self, symbol: str, side: str, amount: float, price: float, time_in_force: str, params={}):
        params["positionSide"] = "LONG" if side.lower() == "buy" else "SHORT"  # set positionSide parameter
        params["timeInForce"] = time_in_force
        try:
            order = self.exchange.create_order(symbol, "limit", side, amount, price, params)
            return order
        except Exception as e:
            print(f"An error occurred while creating the limit order: {e}")

    def binance_create_take_profit_order(self, symbol: str, side: str, positionSide: str, amount: float, price: float, params={}):
        try:
            order_params = {
                'positionSide': positionSide,
                **params
            }
            order = self.exchange.create_order(symbol, "TAKE_PROFIT_MARKET", side, amount, price, order_params)
            return order
        except Exception as e:
            print(f"An error occurred while creating the take-profit order: {e}")

    def binance_create_limit_maker_order(self, symbol: str, side: str, amount: float, price: float):
        try:
            order_params = {
                'timeInForce': 'GTC'
            }
            order = self.exchange.create_order(symbol, "LIMIT", side, amount, price, order_params)
            return order
        except Exception as e:
            print(f"An error occurred while creating the limit maker order: {e}")

    def binance_create_take_profit_limit_maker_order(self, symbol: str, side: str, amount: float, stop_price: float, price: float):
        try:
            order_params = {
                'stopPrice': stop_price,
                'timeInForce': 'GTC'
            }
            order = self.exchange.create_order(symbol, "TAKE_PROFIT", side, amount, price, order_params)
            return order
        except Exception as e:
            print(f"An error occurred while creating the take profit limit maker order: {e}")

    def binance_create_reduce_only_limit_order(self, symbol: str, side: str, amount: float, price: float):
        try:
            order_params = {
                'reduceOnly': 'true',
                'timeInForce': 'GTC'
            }
            order = self.exchange.create_order(symbol, "LIMIT", side, amount, price, order_params)
            return order
        except Exception as e:
            raise Exception(f"An error occurred while creating the reduce only limit order: {e}") from e


    def get_market_data_binance(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0, "step_size": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)
                
            if "precision" in symbol_data:
                values["precision"] = symbol_data["precision"]["price"]
            if "limits" in symbol_data:
                values["min_qty"] = symbol_data["limits"]["amount"]["min"]

            # Fetch positions
            positions = self.exchange.fetch_positions()

            for position in positions:
                if position['symbol'] == symbol:
                    values["leverage"] = float(position['leverage'])

            # Fetch step size
            if "info" in symbol_data and "filters" in symbol_data["info"]:
                for filter in symbol_data["info"]["filters"]:
                    if filter['filterType'] == 'LOT_SIZE':
                        values["step_size"] = filter['stepSize']

        except Exception as e:
            logging.info(f"An unknown error occurred in get_market_data_binance(): {e}")
        return values
    
    def get_balance_binance(self, symbol: str):
        if self.exchange.has['fetchBalance']:
            # Fetch the balance
            balance = self.exchange.fetch_balance(params={'type': 'future'})
            #print(balance)

            # Find the symbol balance
            for currency_balance in balance['info']['assets']:
                if currency_balance['asset'] == symbol:
                    return float(currency_balance['walletBalance'])
        return None
    
    def print_positions_structure_binance(self):
        try:
            data = self.exchange.fetch_positions_risk()
            print(data)
        except Exception as e:
            logging.info(f"An unknown error occurred: {e}")

    # Binance
    def get_positions_binance(self, symbol):
        values = {
            "long": {
                "qty": 0.0,
                "price": 0.0,
                "realised": 0,
                "cum_realised": 0,
                "upnl": 0,
                "upnl_pct": 0,
                "liq_price": 0,
                "entry_price": 0,
            },
            "short": {
                "qty": 0.0,
                "price": 0.0,
                "realised": 0,
                "cum_realised": 0,
                "upnl": 0,
                "upnl_pct": 0,
                "liq_price": 0,
                "entry_price": 0,
            },
        }
        try:
            position_data = self.exchange.fetch_positions_risk([symbol])
            #print(position_data)
            if len(position_data) > 0:
                for position in position_data:
                    #print(position["info"])
                    position_side = position["info"]["positionSide"].lower()
                    if position_side == "both":
                        # Adjust for positions with side 'both'
                        long_qty = float(position["info"]["positionAmt"])
                        short_qty = -long_qty  # Assume opposite quantity for short side
                        position_side = "long"
                        # Update long side values
                        values[position_side]["qty"] = long_qty
                        values[position_side]["price"] = float(position["info"]["entryPrice"])
                        values[position_side]["realised"] = round(float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["cum_realised"] = round(float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl"] = round(float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl_pct"] = 0
                        values[position_side]["liq_price"] = float(position["info"]["liquidationPrice"] or 0)
                        values[position_side]["entry_price"] = float(position["info"]["entryPrice"])
                        # Update short side values
                        position_side = "short"
                        values[position_side]["qty"] = short_qty
                        values[position_side]["price"] = float(position["info"]["entryPrice"])
                        values[position_side]["realised"] = round(-float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["cum_realised"] = round(-float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl"] = round(-float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl_pct"] = 0
                        values[position_side]["liq_price"] = float(position["info"]["liquidationPrice"] or 0)
                        values[position_side]["entry_price"] = float(position["info"]["entryPrice"])
                    else:
                        # Directly update values when position_side is 'long' or 'short'
                        qty = float(position["info"]["positionAmt"]) if position["info"]["positionAmt"] else 0.0
                        entry_price = float(position["info"]["entryPrice"]) if position["info"]["entryPrice"] else 0.0
                        unrealized_profit = float(position["info"]["unRealizedProfit"]) if position["info"]["unRealizedProfit"] else 0.0
                        values[position_side]["qty"] = qty
                        values[position_side]["price"] = entry_price
                        values[position_side]["realised"] = round(unrealized_profit, 4)
                        values[position_side]["cum_realised"] = round(unrealized_profit, 4)
                        values[position_side]["upnl"] = round(unrealized_profit, 4)
                        values[position_side]["upnl_pct"] = 0
                        values[position_side]["liq_price"] = float(position["info"]["liquidationPrice"] or 0)
                        values[position_side]["entry_price"] = entry_price
        except Exception as e:
            logging.info(f"An unknown error occurred in get_positions_binance(): {e}")
        return values

    def cancel_order_by_id_binance(self, order_id, symbol):
        try:
            self.exchange.cancel_order(order_id, symbol)
            logging.info(f"Order with ID: {order_id} was successfully canceled")
        except Exception as e:
            logging.error(f"An error occurred while canceling the order: {e}")

    def cancel_take_profit_orders_binance(self, symbol, side):
        side = side.lower()
        
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            for order in open_orders:
                if (
                    order['side'].lower() == side
                    and order['reduce_only']  # Checking if the order is a reduce-only order
                    and order['type'] in ['TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']  # Checking if the order is a take profit order
                ):
                    order_id = order['id']
                    self.exchange.cancel_order(order_id, symbol)  # Cancel the order
                    logging.info(f"Canceled take profit order - ID: {order_id}")
        except Exception as e:
            print(f"An unknown error occurred in cancel_take_profit_orders_binance: {e}")
 
    def get_leverage_tiers_binance_binance(self):
        try:
            leverage_tiers = self.exchange.fetchLeverageTiers()
            print(f"Leverage tiers: {leverage_tiers}")
        except Exception as e:
            print(f"Error getting leverage tiers: {e}")
            
    def get_max_leverage_binance(self, symbol):
        # Split symbol into base and quote
        base = symbol[:-4]
        quote = symbol[-4:]
        formatted_symbol = f"{base}/{quote}:{quote}"
        
        try:
            leverage_tiers = self.exchange.fetchLeverageTiers()
            symbol_tiers = leverage_tiers.get(formatted_symbol)
            
            if not symbol_tiers:
                raise Exception(f"No leverage tier data available for symbol {formatted_symbol}")

            max_leverage = symbol_tiers[0]['maxLeverage']
            print(f"Max leverage for {formatted_symbol}: {max_leverage}")
            return max_leverage
        except Exception as e:
            print(f"Error getting max leverage: {e}")
            return None

    def binance_set_leverage(self, leverage, symbol: Optional[str] = None, params={}):
        # here we're assuming that maximum allowed leverage is 125 for the symbol
        # but the actual value can vary based on the symbol and the user's account
        max_leverage = 125 
        if leverage > max_leverage:
            print(f"Requested leverage of {leverage}x exceeds maximum allowed leverage of {max_leverage}x for {symbol}.")
            return None
        try:
            response = self.exchange.set_leverage(leverage, symbol, params)
            return response
        except Exception as e:
            print(f"An error occurred while setting the leverage: {e}")

    def binance_set_margin_mode(self, margin_mode: str, symbol: Optional[str] = None, params={}):
        if margin_mode not in ['ISOLATED', 'CROSSED']:
            print(f"Invalid margin mode {margin_mode} for {symbol}. Allowed modes are 'ISOLATED' and 'CROSSED'.")
            return None
        try:
            response = self.exchange.set_margin_mode(margin_mode, symbol, params)
            return response
        except Exception as e:
            print(f"An error occurred while setting the margin mode: {e}")
            
    def get_open_orders_binance(self, symbol: str) -> list:
        open_orders_list = []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            #print(orders)
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        order_info = {
                            "id": order["info"]["orderId"],
                            "price": order["info"]["price"],
                            "amount": float(order["info"]["origQty"]),
                            "status": order["info"]["status"],
                            "side": order["info"]["side"],
                            "reduce_only": order["info"]["reduceOnly"],
                            "type": order["info"]["type"]
                        }
                        open_orders_list.append(order_info)
        except Exception as e:
            logging.info(f"An unknown error occurred in get_open_orders_binance(): {e}")
        return open_orders_list

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
            