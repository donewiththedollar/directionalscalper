import uuid
from .exchange import Exchange
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import ccxt

class BybitExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('bybit', api_key, secret_key, passphrase, market_type)
        self.max_retries = 100  # Maximum retries for rate-limited requests
        self.retry_wait = 5  # Seconds to wait between retries

    
    def create_limit_order_bybit(self, symbol: str, side: str, qty: float, price: float, positionIdx=0, params={}):
        try:
            if side == "buy" or side == "sell":
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side=side,
                    amount=qty,
                    price=price,
                    params={**params, 'positionIdx': positionIdx}  # Pass the 'positionIdx' parameter here
                )
                return order
            else:
                logging.info(f"side {side} does not exist")
                return {"error": f"side {side} does not exist"}
        except Exception as e:
            logging.info(f"An unknown error occurred in create_limit_order() for {symbol}: {e}")
            return {"error": str(e)}

    def create_limit_order_bybit_spot(self, symbol: str, side: str, qty: float, price: float, isLeverage=0, orderLinkId=None):
        try:
            # Define the 'params' dictionary to include any additional parameters required by Bybit's v5 API
            params = {
                'timeInForce': 'PostOnly',  # Set the order as a PostOnly order
                'isLeverage': isLeverage,   # Specify whether to borrow for margin trading
            }
            
            # If 'orderLinkId' is provided, add it to the 'params' dictionary
            if orderLinkId:
                params['orderLinkId'] = orderLinkId

            # Create the limit order using CCXT's 'create_order' function
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
            logging.error(f"An error occurred while creating limit order on Bybit: {e}")
            return None

    def create_tagged_limit_order_bybit(self, symbol: str, side: str, qty: float, price: float, positionIdx=0, isLeverage=False, orderLinkId=None, postOnly=True, params={}):
        try:
            # Directly prepare the parameters required by the `create_order` method
            order_type = "limit"  # For limit orders
            time_in_force = "PostOnly" if postOnly else "GTC"
            
            # Include additional parameters
            extra_params = {
                "positionIdx": positionIdx,
                "timeInForce": time_in_force
            }
            if isLeverage:
                extra_params["isLeverage"] = 1
            if orderLinkId:
                extra_params["orderLinkId"] = orderLinkId
            
            # Merge any additional user-provided parameters
            extra_params.update(params)

            # Create the order
            order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=qty,
                price=price,
                params=extra_params  # Pass extra params here
            )
            return order
        except Exception as e:
            logging.info(f"An error occurred in create_tagged_limit_order_bybit() for {symbol}: {e}")
            return {"error": str(e)}
        
    def create_limit_order_bybit_unified(self, symbol: str, side: str, qty: float, price: float, positionIdx=0, params={}):
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
            logging.info(f"An unknown error occurred in create_limit_order(): {e}")

    def create_market_order_bybit(self, symbol: str, side: str, qty: float, positionIdx=0, params={}):
        try:
            if side == "buy" or side == "sell":
                request = {
                    'symbol': symbol,
                    'type': 'market',
                    'side': side,
                    'qty': qty,
                    'positionIdx': positionIdx,
                    'closeOnTrigger': True  # Set closeOnTrigger to True for market close order
                }
                order = self.exchange.create_contract_v3_order(symbol, 'market', side, qty, params=request)
                return order
            else:
                logging.info(f"Side {side} does not exist")
        except Exception as e:
            logging.info(f"An unknown error occurred in create_market_order(): {e}")

    def transfer_funds(self, code: str, amount: float, from_account: str, to_account: str, params={}):
        """
        Transfer funds between different account types under the same UID.
        """
        try:
            # Generate a unique transfer ID (UUID)
            transfer_id = str(uuid.uuid4())

            # Add the transfer ID to the params dictionary
            params['transferId'] = transfer_id

            # Use CCXT's transfer function to initiate the internal transfer
            transfer = self.exchange.transfer(code, amount, from_account, to_account, params)

            if transfer:
                logging.info(f"Funds transfer successful. Details: {transfer}")
                return transfer
            else:
                logging.error("Error occurred during funds transfer.")
                return None
        except Exception as e:
            logging.error(f"Error occurred during funds transfer: {e}")
            return None

    def get_bybit_wallet_balance(self, coin: str):
        """
        Fetch the Bybit wallet balance for a specific coin.
        """
        try:
            # Use the fetch_balance method from CCXT (inherited from the Exchange class)
            balance = self.exchange.fetch_balance()

            if coin in balance:
                return balance[coin]['free']
            else:
                logging.info(f"Coin {coin} not found in Bybit wallet balance.")
                return None
        except Exception as e:
            logging.error(f"Error occurred while fetching Bybit wallet balance: {e}")
            return None

    def get_futures_balance_bybit(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type if needed
                balance_response = self.exchange.fetch_balance({'type': 'swap'})

                # Log the raw response for debugging purposes
                #logging.info(f"Raw balance response from Bybit: {balance_response}")

                # Parse the balance
                if quote in balance_response['total']:
                    total_balance = balance_response['total'][quote]
                    return total_balance
                else:
                    logging.info(f"Balance for {quote} not found in the response.")
            except Exception as e:
                logging.error(f"Error fetching balance from Bybit: {e}")

        return None

    def get_available_balance_bybit(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type
                balance_response = self.exchange.fetch_balance({'type': 'swap'})

                # Log the raw response for debugging purposes
                #logging.info(f"Raw available balance response from Bybit: {balance_response}")

                # Check for the required keys in the response
                if 'free' in balance_response and quote in balance_response['free']:
                    # Return the available balance for the specified currency
                    return float(balance_response['free'][quote])
                else:
                    logging.info(f"Available balance for {quote} not found in the response.")

            except Exception as e:
                logging.error(f"Error fetching available balance from Bybit: {e}")

        return None
    
    def get_symbol_precision_bybit(self, symbol):
        try:
            # Use fetch_markets to retrieve data for all markets
            all_markets = self.exchange.fetch_markets()

            # Find the market data for the specific symbol
            market_data = next((market for market in all_markets if market['id'] == symbol), None)

            if market_data:
                # Extract precision data
                amount_precision = market_data['precision']['amount']
                price_precision = market_data['precision']['price']
                return amount_precision, price_precision
            else:
                print(f"Market data not found for {symbol}")
                return None, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None

    def get_positions_bybit(self, symbol, max_retries=100, retry_delay=5) -> dict:
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

        for i in range(max_retries):
            try:
                data = self.exchange.fetch_positions(symbol)
                if len(data) == 2:
                    sides = ["long", "short"]
                    for side in [0, 1]:
                        values[sides[side]]["qty"] = float(data[side]["contracts"])
                        values[sides[side]]["price"] = float(data[side]["entryPrice"] or 0)
                        values[sides[side]]["realised"] = round(float(data[side]["info"]["unrealisedPnl"] or 0), 4)
                        values[sides[side]]["cum_realised"] = round(float(data[side]["info"]["cumRealisedPnl"] or 0), 4)
                        values[sides[side]]["upnl"] = round(float(data[side]["info"]["unrealisedPnl"] or 0), 4)
                        values[sides[side]]["upnl_pct"] = round(float(data[side]["percentage"] or 0), 4)
                        values[sides[side]]["liq_price"] = float(data[side]["liquidationPrice"] or 0)
                        values[sides[side]]["entry_price"] = float(data[side]["entryPrice"] or 0)
                break  # If the fetch was successful, break out of the loop
            except Exception as e:
                if i < max_retries - 1:  # If not the last attempt
                    logging.info(f"An unknown error occurred in get_positions_bybit(): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.info(f"Failed to fetch positions after {max_retries} attempts: {e}")
                    raise e  # If it's still failing after max_retries, re-raise the exception.

        return values

    def cancel_all_orders_for_symbol_bybit(self, symbol):
        try:
            # Assuming 'self.exchange' is your initialized CCXT exchange instance
            cancel_result = self.exchange.cancel_all_orders(symbol)
            logging.info(f"All open orders for {symbol} have been cancelled. Result: {cancel_result}")
            return cancel_result
        except Exception as e:
            logging.error(f"Error cancelling open orders for {symbol}: {e}")
            return None
        
    def get_current_max_leverage_bybit(self, symbol):
        try:
            # Fetch leverage tiers for the symbol
            leverage_tiers = self.exchange.fetch_market_leverage_tiers(symbol)

            # Process leverage tiers to find the maximum leverage
            max_leverage = max([tier['maxLeverage'] for tier in leverage_tiers if 'maxLeverage' in tier])
            logging.info(f"Maximum leverage for symbol {symbol}: {max_leverage}")

            return max_leverage

        except Exception as e:
            logging.error(f"Error retrieving leverage tiers for {symbol}: {e}")
            return None

    def set_leverage_bybit(self, leverage, symbol):
        try:
            self.exchange.set_leverage(leverage, symbol)
            logging.info(f"Leverage set to {leverage} for symbol {symbol}")
        except Exception as e:
            logging.info(f"Error setting leverage: {e}")

    def set_symbol_to_cross_margin(self, symbol, leverage):
        """
        Set a specific symbol's margin mode to cross with specified leverage.
        """
        try:
            response = self.exchange.set_margin_mode('cross', symbol=symbol, params={'leverage': leverage})
            
            retCode = response.get('retCode') if isinstance(response, dict) else None

            if retCode == 110026:  # Margin mode is already set to cross
                logging.info(f"Symbol {symbol} is already set to cross margin mode. No changes made.")
                return {"status": "unchanged", "response": response}
            else:
                logging.info(f"Margin mode set to cross for symbol {symbol} with leverage {leverage}. Response: {response}")
                return {"status": "changed", "response": response}

        except Exception as e:
            logging.info(f"Failed to set margin mode or margin mode already set to cross for symbol {symbol} with leverage {leverage}: {e}")
            return {"status": "error", "message": str(e)}

    def setup_exchange_bybit(self, symbol) -> None:
        values = {"position": False, "leverage": False}
        try:
            # Set the position mode to hedge
            self.exchange.set_position_mode(hedged=True, symbol=symbol)
            values["position"] = True
        except Exception as e:
            logging.info(f"An unknown error occurred in with set_position_mode: {e}")

    def get_all_open_positions_bybit(self, retries=10, delay_factor=10, max_delay=60) -> List[dict]:
        now = datetime.now()

        # Check if the shared cache is still valid
        cache_duration = timedelta(seconds=30)  # Cache duration is 30 seconds
        if self.open_positions_shared_cache and self.last_open_positions_time_shared and now - self.last_open_positions_time_shared < cache_duration:
            return self.open_positions_shared_cache

        # Using a semaphore to limit concurrent API requests
        with self.open_positions_semaphore:
            # Double-checking the cache inside the semaphore to ensure no other thread has refreshed it in the meantime
            if self.open_positions_shared_cache and self.last_open_positions_time_shared and now - self.last_open_positions_time_shared < cache_duration:
                return self.open_positions_shared_cache

            for attempt in range(retries):
                try:
                    all_positions = self.exchange.fetch_positions() 
                    open_positions = [position for position in all_positions if float(position.get('contracts', 0)) != 0] 

                    # Update the shared cache with the new data
                    self.open_positions_shared_cache = open_positions
                    self.last_open_positions_time_shared = now

                    return open_positions
                except Exception as e:
                    is_rate_limit_error = "Too many visits" in str(e) or (hasattr(e, 'response') and e.response.status_code == 403)
                    
                    if is_rate_limit_error and attempt < retries - 1:
                        delay = min(delay_factor * (attempt + 1), max_delay)  # Exponential delay with a cap
                        logging.info(f"Rate limit on get_all_open_positions_bybit hit, waiting for {delay} seconds before retrying...")
                        time.sleep(delay)
                        continue
                    else:
                        logging.error(f"Error fetching open positions: {e}")
                        return []

    def fetch_leverage_tiers(self, symbol: str) -> dict:
        """
        Fetch leverage tiers for a given symbol using CCXT's fetch_market_leverage_tiers method.

        :param symbol: The trading symbol to fetch leverage tiers for.
        :return: A dictionary containing leverage tiers information if successful, None otherwise.
        """
        try:
            params = {'category': 'linear'}  # Adjust parameters based on the specific needs and API documentation
            leverage_tiers = self.exchange.fetch_derivatives_market_leverage_tiers(symbol, params)
            return leverage_tiers
        except Exception as e:
            logging.error(f"Error fetching leverage tiers for {symbol}: {e}")
            return None

    def get_open_take_profit_orders(self, symbol, side):
        """
        Fetches open take profit orders for the given symbol and side.
        
        :param str symbol: The trading pair symbol.
        :param str side: The side ("buy" or "sell") of the TP orders to fetch.
        :return: A list of take profit order structures.
        """
        # First, fetch the open orders
        response = self.exchange.get_open_orders(symbol)
        
        # Filter the orders for take profits (reduceOnly) and the specified side
        tp_orders = [
            order for order in response
            if order.get('info', {}).get('reduceOnly', False) and order.get('side', '').lower() == side.lower()
        ]
        
        # If necessary, you can further parse the orders here using self.parse_order or similar methods
        
        return tp_orders

    def get_open_orders(self, symbol):
        """Fetches open orders for the given symbol."""
        for _ in range(self.max_retries):
            try:
                open_orders = self.exchange.fetch_open_orders(symbol)
                logging.info(f"Open orders {open_orders}")
                return open_orders
            except ccxt.RateLimitExceeded:
                logging.info(f"Rate limit exceeded when fetching open orders for {symbol}. Retrying in {self.retry_wait} seconds...")
                time.sleep(self.retry_wait)
        logging.error(f"Failed to fetch open orders for {symbol} after {self.max_retries} retries.")
        return []
    
    def get_open_tp_orders(self, symbol):
        long_tp_orders = []
        short_tp_orders = []
        for _ in range(self.max_retries):
            try:
                all_open_orders = self.exchange.fetch_open_orders(symbol)
                #logging.info(f"All open orders for {symbol}: {all_open_orders}")
                
                for order in all_open_orders:
                    order_details = {
                        'id': order['id'],
                        'qty': float(order['info']['qty']),
                        'price': float(order['price'])  # Extracting the price
                    }
                    
                    if order['info'].get('reduceOnly', False):
                        if order['side'] == 'sell':
                            long_tp_orders.append(order_details)
                        elif order['side'] == 'buy':
                            short_tp_orders.append(order_details)
                
                return long_tp_orders, short_tp_orders
            except ccxt.RateLimitExceeded:
                logging.info(f"Rate limit exceeded when fetching TP orders for {symbol}. Retrying in {self.retry_wait} seconds...")
                time.sleep(self.retry_wait)
        logging.error(f"Failed to fetch TP orders for {symbol} after {self.max_retries} retries.")
        return long_tp_orders, short_tp_orders
    
    def get_open_tp_order_count(self, symbol):
        """
        Fetches the count of open take profit (TP) orders for the given symbol.
        
        :param str symbol: The trading pair symbol.
        :return: Dictionary with counts of long and short TP orders for the symbol.
        """
        long_tp_orders, short_tp_orders = self.get_open_tp_orders(symbol)
        return {
            'long_tp_count': len(long_tp_orders),
            'short_tp_count': len(short_tp_orders)
        }
