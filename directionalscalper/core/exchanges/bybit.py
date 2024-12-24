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

logging = Logger(logger_name="BybitExchange", filename="BybitExchange.log", stream=True)

class BybitExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap', collateral_currency='USDT'):
        """
        Initialize the BybitExchange class.

        :param api_key: API key for authentication with Bybit.
        :param secret_key: Secret key for authentication with Bybit.
        :param passphrase
        :param market_type: Type of market ('swap' or 'spot'). Default is 'swap'.
        :param collateral_currency: Currency used as collateral for trading. Default is 'USDT'. If set to 'all', it will use the total available balance.
        """

        if market_type == 'spot':
            super().__init__('bybit', api_key, secret_key, passphrase, market_type)
        else:
            super().__init__('bybit', api_key, secret_key, passphrase, market_type)

        self.max_retries = 100  # Maximum retries for rate-limited requests
        self.retry_wait = 5  # Seconds to wait between retries
        self.last_active_long_order_time = {}
        self.last_active_short_order_time = {}
        self.last_active_time = {}
        self.rate_limiter = RateLimit(10, 1)
        self.general_rate_limiter = RateLimit(50, 1)
        self.order_rate_limiter = RateLimit(5, 1) 
        self.collateral_currency = collateral_currency

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

    # Assuming you have an initializer or a specific method where symbols start being monitored
    def initialize_symbol_monitoring(self, symbol):
        if symbol not in self.last_active_time:
            self.last_active_time[symbol] = time.time()
            logging.info(f"Started monitoring {symbol} at {self.last_active_time[symbol]}")
     
    def get_symbol_info_and_positions(self, symbol: str):
        try:
            # Fetch the market info for the given symbol
            market = self.exchange.market(symbol)

            # Log the market info
            logging.info(f"Symbol: {market['symbol']}")
            logging.info(f"Base: {market['base']}")
            logging.info(f"Quote: {market['quote']}")
            logging.info(f"Type: {market['type']}")
            logging.info(f"Settle: {market['settle']}")

            # Fetch the positions for the given symbol
            positions = self.exchange.fetch_positions([symbol])

            # Log the positions
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

    def get_market_data_bybit(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            time.sleep(1)  # Adding a fixed delay of 1 second to avoid hitting the rate limit
            with self.general_rate_limiter:
                self.exchange.load_markets()
                symbol_data = self.exchange.market(symbol)
                
                if "info" in symbol_data:
                    values["precision"] = symbol_data["precision"]["price"]
                    values["min_qty"] = symbol_data["limits"]["amount"]["min"]

                # Fetch positions
                positions = self.exchange.fetch_positions()

                for position in positions:
                    if position['symbol'] == symbol:
                        values["leverage"] = float(position['leverage'])

        except Exception as e:
            logging.info(f"An unknown error occurred in get_market_data_bybit(): {e}")
            # Uncomment if you want to log the traceback for debugging
            #logging.info(f"Call Stack: {traceback.format_exc()}")
        
        return values       
    # def get_market_data_bybit(self, symbol: str) -> dict:
    #     values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
    #     try:
    #         self.exchange.load_markets()
    #         symbol_data = self.exchange.market(symbol)
            
    #         #print("Symbol data:", symbol_data)  # Debug print
            
    #         if "info" in symbol_data:
    #             values["precision"] = symbol_data["precision"]["price"]
    #             values["min_qty"] = symbol_data["limits"]["amount"]["min"]

    #         # Fetch positions
    #         positions = self.exchange.fetch_positions()

    #         for position in positions:
    #             if position['symbol'] == symbol:
    #                 values["leverage"] = float(position['leverage'])

    #     # except Exception as e:
    #     #     logging.info(f"An unknown error occurred in get_market_data_bybit(): {e}")
    #     #     logging.info(f"Call Stack: {traceback.format_exc()}")
    #     except Exception as e:
    #         logging.info(f"An unknown error occurred in get_market_data_bybit(): {e}")
    #     return values

    def get_best_bid_ask_bybit(self, symbol):
        orderbook = self.exchange.get_orderbook(symbol)
        try:
            best_ask_price = orderbook['asks'][0][0]
        except IndexError:
            best_ask_price = None
        try:
            best_bid_price = orderbook['bids'][0][0]
        except IndexError:
            best_bid_price = None

        return best_bid_price, best_ask_price

    def get_all_open_orders_bybit(self):
        """
        Fetch all open orders for all symbols from the Bybit API.
        
        :return: A list of open orders for all symbols.
        """
        try:
            all_open_orders = self.exchange.fetch_open_orders()
            return all_open_orders
        except Exception as e:
            print(f"An error occurred while fetching all open orders: {e}")
            return []

    def get_balance_bybit_spot(self, quote):
        if self.exchange.has['fetchBalance']:
            try:
                # Specify the type as 'spot' for spot trading
                balance_response = self.exchange.fetch_balance({'type': 'spot'})

                # Logging the raw response for debugging might be useful
                # logging.info(f"Raw balance response from Bybit: {balance_response}")

                # Parse the balance for the quote currency
                if quote in balance_response['total']:
                    total_balance = balance_response['total'][quote]
                    return total_balance
                else:
                    logging.info(f"Balance for {quote} not found in the response.")
            except Exception as e:
                logging.info(f"Error fetching balance from Bybit: {e}")

        return None

    def get_balance_bybit(self, quote):
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
                logging.info(f"Error fetching balance from Bybit: {e}")

        return None

    def get_available_balance_bybit(self):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type
                balance_response = self.exchange.fetch_balance({'type': 'swap'})

                # Log the raw response for debugging purposes
                #logging.info(f"Raw available balance response from Bybit: {balance_response}")

                if self.collateral_currency == 'all' and 'info' in balance_response:
                    logging.info("quote is not set - pulling available balance from total available")

                    available_balance = balance_response['info']['result']['list'][0]['totalAvailableBalance']
                    return float(available_balance)

                # Check for the required keys in the response
                if 'free' in balance_response and self.collateral_currency in balance_response['free']:
                    # Return the available balance for the specified currency
                    return float(balance_response['free'][self.collateral_currency])
                else:
                    logging.warning(f"Available balance for {self.collateral_currency} not found in the response.")

            except Exception as e:
                logging.info(f"Error fetching available balance from Bybit: {e}")

        return None
    

    def get_balance_bybit_unified(self, quote):
        if self.exchange.has['fetchBalance']:
            # Fetch the balance
            balance = self.exchange.fetch_balance()

            # Find the quote balance
            unified_balance = balance.get('USDT', {})
            total_balance = unified_balance.get('total', None)
            
            if total_balance is not None:
                return float(total_balance)

        return None
    
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
            logging.info(f"An error occurred while creating limit order on Bybit: {e}")
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

            # Log the time of order creation for side-specific tracking
            current_time = time.time()
            if side.lower() == 'buy':
                self.last_active_long_order_time[symbol] = current_time
                logging.info(f"Logged long order time for {symbol}")
            elif side.lower() == 'sell':
                self.last_active_short_order_time[symbol] = current_time
                logging.info(f"Logged short order time for {symbol}")

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

    def cancel_all_open_orders_bybit(self, symbol=None, category="linear"):
        """
        Cancels all open orders for a specific category. If a symbol is provided, only orders for that symbol are cancelled.
        :param symbol: Optional. The market symbol (e.g., 'BTC/USDT') for which to cancel orders. If None, all orders in the specified category are cancelled.
        :param category: The category of products for which to cancel orders (e.g., 'linear', 'inverse'). Default is 'linear'.
        :return: Response from the exchange indicating success or failure.
        """
        try:
            logging.info(f"cancel_all_open_orders_bybit called")
            params = {'category': category}  # Specify additional parameters as needed
            
            # Optionally, add symbol to request if provided
            if symbol is not None:
                market = self.exchange.market(symbol)
                params['symbol'] = market['id']

            response = self.exchange.cancel_all_orders(params=params)
            
            logging.info(f"Successfully cancelled orders {response}")
            return response
        except Exception as e:
            logging.info(f"Error cancelling orders: {e}")
            
    def cancel_order_bybit(self, order_id, symbol):
        """
        Wrapper function to cancel an order on the exchange using the CCXT instance.

        :param order_id: The ID of the order to cancel.
        :param symbol: The trading symbol of the market the order was made in.
        :return: The response from the exchange after attempting to cancel the order.
        """
        try:
            # Call the cancel_order method of the ccxt instance
            response = self.exchange.cancel_order(order_id, symbol)
            logging.info(f"Order {order_id} for {symbol} cancelled successfully.")
            return response
        except Exception as e:
            logging.info(f"An error occurred while cancelling order {order_id} for {symbol}: {str(e)}")
            # Handle the exception as needed (e.g., retry, raise, etc.)
            return None
        
    def get_precision_and_limits_bybit(self, symbol):
        # Fetch the market data
        markets = self.exchange.fetch_markets()

        # Filter for the specific symbol
        for market in markets:
            if market['symbol'] == symbol:
                precision_amount = market['precision']['amount']
                precision_price = market['precision']['price']
                min_amount = market['limits']['amount']['min']

                return precision_amount, precision_price, min_amount

        return None, None, None

    def get_market_precision_data_bybit(self, symbol):
        # Fetch the market data
        markets = self.exchange.fetch_markets()
        
        # Print the first market from the list
        logging.info(markets[0])

        # Filter for the specific symbol
        for market in markets:
            if market['symbol'] == symbol:
                return market['precision']
        
        return None
    
    def transfer_funds_bybit(self, code: str, amount: float, from_account: str, to_account: str, params={}):
        """
        Transfer funds between different account types under the same UID.

        :param str code: Unified currency code
        :param float amount: Amount to transfer
        :param str from_account: From account type (e.g., 'UNIFIED', 'CONTRACT')
        :param str to_account: To account type (e.g., 'SPOT', 'CONTRACT')
        :param dict params: Extra parameters specific to the exchange API endpoint
        :return: A transfer structure
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
                logging.info(f"Error occurred during funds transfer.")
                return None

        except Exception as e:
            logging.info(f"Error occurred during funds transfer: {e}")
            return None
        
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
                logging.info("Error occurred during funds transfer.")
                return None
        except Exception as e:
            logging.info(f"Error occurred during funds transfer: {e}")
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
            logging.info(f"Error occurred while fetching Bybit wallet balance: {e}")
            return None

    def get_futures_balance_bybit(self):
        if self.exchange.has['fetchBalance']:
            try:
                # Fetch the balance with params to specify the account type if needed
                balance_response = self.exchange.fetch_balance({'type': 'swap'})

                # Log the raw response for debugging purposes
                #logging.info(f"Raw balance response from Bybit: {balance_response}")

                if self.collateral_currency == 'all' and 'info' in balance_response:
                    logging.info("quote is not set - pulling total balance from total equity")

                    total_balance = balance_response['info']['result']['list'][0]['totalWalletBalance']
                    return total_balance

                # Parse the balance
                if self.collateral_currency in balance_response['total']:
                    total_balance = balance_response['total'][self.collateral_currency]
                    return total_balance
                else:
                    logging.info(f"Balance for {self.collateral_currency} not found in the response.")
            except Exception as e:
                logging.info(f"Error fetching balance from Bybit: {e}")

        return None

    # def get_symbol_precision_bybit(self, symbol):
    #     try:
    #         # Use fetch_markets to retrieve data for all markets
    #         all_markets = self.exchange.fetch_markets()

    #         # Find the market data for the specific symbol
    #         market_data = next((market for market in all_markets if market['id'] == symbol), None)

    #         if market_data:
    #             # Extract precision data
    #             amount_precision = market_data['precision']['amount']
    #             price_precision = market_data['precision']['price']
    #             return amount_precision, price_precision
    #         else:
    #             print(f"Market data not found for {symbol}")
    #             return None, None
    #     except Exception as e:
    #         logging.info(f"An error occurred in get_symbol_precision_bybit: {e}")
    #         logging.info("Traceback: %s", traceback.format_exc())
    #         return None, None

    def get_symbol_precision_bybit(self, symbol, max_retries=1000, retry_delay=5):
        for attempt in range(max_retries):
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
                    logging.info(f"Market data not found for {symbol}")
                    return None, None

            except Exception as e:
                logging.info(f"Attempt {attempt + 1}/{max_retries} failed in get_symbol_precision_bybit: {e}")
                logging.info("Traceback: %s", traceback.format_exc())

                if attempt < max_retries - 1:
                    # Wait before retrying
                    time.sleep(retry_delay)
                else:
                    # All attempts failed
                    logging.info(f"All retry attempts failed for get_symbol_precision_bybit({symbol}).")
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
            logging.info(f"All open orders for {symbol} have been cancelled.")
            #logging.info(f"Result: {cancel_result}")
            return cancel_result
        except Exception as e:
            logging.info(f"Error cancelling open orders for {symbol}: {e}")
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
            logging.info(f"Error retrieving leverage tiers for {symbol}: {e}")
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

    def get_all_open_positions_bybit_spot(self, retries=10, delay_factor=10, max_delay=60) -> List[dict]:
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
                    params = {
                        'type': 'spot',  # Set the type to 'spot' for Bybit Spot trading
                    }
                    all_positions = self.exchange.fetch_positions(params=params)
                    open_positions = [position for position in all_positions if float(position.get('contracts', position.get('size', 0))) != 0]
                    
                    # Update the shared cache with the new data
                    self.open_positions_shared_cache = open_positions
                    self.last_open_positions_time_shared = now
                    return open_positions
                except Exception as e:
                    is_rate_limit_error = "Too many visits" in str(e) or (hasattr(e, 'response') and e.response.status_code == 403)
                    if is_rate_limit_error and attempt < retries - 1:
                        delay = min(delay_factor * (attempt + 1), max_delay)  # Exponential delay with a cap
                        logging.info(f"Rate limit on get_all_open_positions_bybit_spot hit, waiting for {delay} seconds before retrying...")
                        time.sleep(delay)
                        continue
                    else:
                        logging.info(f"Error fetching open positions: {e}")
                        return []
                    
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
                    # all_positions = self.exchange.fetch_positions() 
                    all_positions = self.exchange.fetch_positions(params={'limit': 200})
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
                        logging.info(f"Error fetching open positions: {e}")
                        return []

    def get_all_open_positions_bybit_spot(self, retries=10, delay_factor=10, max_delay=60) -> List[dict]:
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
                    params = {
                        'type': 'spot',  # Set the type to 'spot' for Bybit Spot trading
                    }
                    all_positions = self.exchange.fetch_positions(params=params)
                    open_positions = [position for position in all_positions if float(position.get('contracts', position.get('size', 0))) != 0]
                    
                    # Update the shared cache with the new data
                    self.open_positions_shared_cache = open_positions
                    self.last_open_positions_time_shared = now
                    return open_positions
                except Exception as e:
                    is_rate_limit_error = "Too many visits" in str(e) or (hasattr(e, 'response') and e.response.status_code == 403)
                    if is_rate_limit_error and attempt < retries - 1:
                        delay = min(delay_factor * (attempt + 1), max_delay)  # Exponential delay with a cap
                        logging.info(f"Rate limit on get_all_open_positions_bybit_spot hit, waiting for {delay} seconds before retrying...")
                        time.sleep(delay)
                        continue
                    else:
                        logging.info(f"Error fetching open positions: {e}")
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
            logging.info(f"Error fetching leverage tiers for {symbol}: {e}")
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

    # def get_all_open_orders(self):
    #     """Fetches open orders for all symbols."""
    #     for _ in range(self.max_retries):
    #         try:
    #             open_orders = self.exchange.fetch_open_orders()
    #             return open_orders
    #         except ccxt.RateLimitExceeded:
    #             logging.info(f"Rate limit exceeded when fetching open orders. Retrying in {self.retry_wait} seconds...")
    #             time.sleep(self.retry_wait)
    #     logging.info(f"Failed to fetch open orders after {self.max_retries} retries.")
    #     return []

    # def get_open_orders(self, symbol):
    #     """Fetches open orders for the given symbol."""
    #     for _ in range(self.max_retries):
    #         try:
    #             open_orders = self.exchange.fetch_open_orders(symbol)
    #             #logging.info(f"Open orders {open_orders}")
    #             return open_orders
    #         except ccxt.RateLimitExceeded:
    #             logging.info(f"Rate limit exceeded when fetching open orders for {symbol}. Retrying in {self.retry_wait} seconds...")
    #             time.sleep(self.retry_wait)
    #     logging.info(f"Failed to fetch open orders for {symbol} after {self.max_retries} retries.")
    #     return []

    def get_all_open_orders(self):
        """Fetches open orders for all symbols."""
        for _ in range(self.max_retries):
            try:
                with self.rate_limiter:
                    open_orders = self.exchange.fetch_open_orders()
                return open_orders
            except RateLimitExceeded:
                logging.info(f"Rate limit exceeded when fetching open orders. Retrying in {self.retry_wait} seconds...")
                time.sleep(self.retry_wait)
            except Exception as e:
                logging.error(f"Error fetching open orders: {e}")
                logging.error(traceback.format_exc())
        logging.info(f"Failed to fetch open orders after {self.max_retries} retries.")
        return []

    def get_open_orders(self, symbol, max_retries=100, retry_wait=1):
        """Fetches open orders for the given symbol with exponential backoff."""
        backoff = retry_wait
        for attempt in range(max_retries):
            try:
                with self.rate_limiter:
                    open_orders = self.exchange.fetch_open_orders(symbol)
                return open_orders
            except RateLimitExceeded:
                logging.info(f"Rate limit exceeded when fetching open orders for {symbol}. Retrying in {retry_wait} seconds...")
                time.sleep(retry_wait)
            except NetworkError as e:
                logging.error(f"Network error fetching open orders for {symbol}: {e}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2  # Exponential backoff
            except Exception as e:
                logging.error(f"Error fetching open orders for {symbol}: {e}")
                logging.error(traceback.format_exc())
                break
        logging.info(f"Failed to fetch open orders for {symbol} after {max_retries} retries.")
        return []

    # def get_open_orders(self, symbol):
    #     """Fetches open orders for the given symbol."""
    #     for _ in range(self.max_retries):
    #         try:
    #             with self.rate_limiter:
    #                 open_orders = self.exchange.fetch_open_orders(symbol)
    #             return open_orders
    #         except RateLimitExceeded:
    #             logging.info(f"Rate limit exceeded when fetching open orders for {symbol}. Retrying in {self.retry_wait} seconds...")
    #             time.sleep(self.retry_wait)
    #         except Exception as e:
    #             logging.error(f"Error fetching open orders for {symbol}: {e}")
    #             logging.error(traceback.format_exc())
    #     logging.info(f"Failed to fetch open orders for {symbol} after {self.max_retries} retries.")
    #     return []



    def get_open_orders_bybit_unified(self, symbol: str) -> list:
        open_orders_list = []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            #print(orders)
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        order_info = {
                            "id": order["id"],
                            "price": float(order["price"]),
                            "qty": float(order["amount"]),
                            "order_status": order["status"],
                            "side": order["side"],
                            "reduce_only": order["reduceOnly"],
                            "position_idx": int(order["info"]["positionIdx"])
                        }
                        open_orders_list.append(order_info)
        except Exception as e:
            logging.info(f"An unknown error occurred in get_open_orders(): {e}")
        return open_orders_list

    def get_open_tp_orders(self, open_orders):
        long_tp_orders = []
        short_tp_orders = []

        for order in open_orders:
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


    def get_open_tp_order_count(self, open_orders):
        """
        Fetches the count of open take profit (TP) orders from the given open orders.

        :param list open_orders: The list of open orders.
        :return: Dictionary with counts of long and short TP orders.
        """
        long_tp_orders, short_tp_orders = self.get_open_tp_orders(open_orders)
        return {
            'long_tp_count': len(long_tp_orders),
            'short_tp_count': len(short_tp_orders)
        }

    def cancel_close_bybit(self, symbol: str, side: str, max_retries: int = 3) -> None:
        side = side.lower()
        side_map = {"long": "buy", "short": "sell"}
        side = side_map.get(side, side)
        
        position_idx_map = {"buy": 1, "sell": 2}
        
        retries = 0
        while retries < max_retries:
            try:
                orders = self.exchange.fetch_open_orders(symbol)
                if len(orders) > 0:
                    for order in orders:
                        if "info" in order:
                            order_id = order["info"]["orderId"]
                            order_status = order["info"]["orderStatus"]
                            order_side = order["info"]["side"]
                            reduce_only = order["info"]["reduceOnly"]
                            position_idx = order["info"]["positionIdx"]

                            if (
                                order_status != "Filled"
                                and order_side.lower() == side
                                and order_status != "Cancelled"
                                and reduce_only
                                and position_idx == position_idx_map[side]
                            ):
                                # use the new cancel_derivatives_order function
                                self.exchange.cancel_derivatives_order(order_id, symbol)
                                logging.info(f"Cancelling order: {order_id}")
                # If the code reaches this point without an exception, break out of the loop
                break

            except (RateLimitExceeded, NetworkError) as e:
                retries += 1
                logging.info(f"Encountered an error in cancel_close_bybit(). Retrying... {retries}/{max_retries}")
                time.sleep(5)  # Pause before retrying, this can be adjusted

            except Exception as e:
                # For other exceptions, log and break out of the loop
                logging.info(f"An unknown error occurred in cancel_close_bybit(): {e}")
                break

    def create_take_profit_order_bybit(self, symbol, order_type, side, amount, price=None, positionIdx=1, reduce_only=True):
        logging.info(f"Calling create_take_profit_order_bybit with symbol={symbol}, order_type={order_type}, side={side}, amount={amount}, price={price}")
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")
            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")
            params = {"reduceOnly": reduce_only, "postOnly": True}  # Add postOnly parameter
            return self.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
    def create_normal_take_profit_order_bybit(self, symbol, order_type, side, amount, price=None, positionIdx=1, reduce_only=True):
        logging.info(f"Calling create_take_profit_order_bybit with symbol={symbol}, order_type={order_type}, side={side}, amount={amount}, price={price}")
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")

            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")

            params = {"reduceOnly": reduce_only}
            return self.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def postonly_create_take_profit_order_bybit(self, symbol, order_type, side, amount, price=None, positionIdx=1, reduce_only=True, post_only=True):
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")

            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")

            params = {"reduceOnly": reduce_only, "postOnly": post_only}
            return self.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
    def create_market_order_bybit_spot(self, symbol: str, side: str, qty: float, marketUnit=None, isLeverage=0, orderLinkId=None, orderFilter=None, takeProfit=None, stopLoss=None, tpOrderType=None, slOrderType=None, tpLimitPrice=None, slLimitPrice=None):
        try:
            # Define the 'params' dictionary to include additional parameters
            params = {
                'isLeverage': isLeverage,
            }

            # Add optional parameters to the 'params' dictionary if provided
            if marketUnit:
                params['marketUnit'] = marketUnit
            if orderLinkId:
                params['orderLinkId'] = orderLinkId
            if orderFilter:
                params['orderFilter'] = orderFilter
            if takeProfit:
                params['takeProfit'] = str(takeProfit)
            if stopLoss:
                params['stopLoss'] = str(stopLoss)
            if tpOrderType:
                params['tpOrderType'] = tpOrderType
            if slOrderType:
                params['slOrderType'] = slOrderType
            if tpLimitPrice:
                params['tpLimitPrice'] = str(tpLimitPrice)
            if slLimitPrice:
                params['slLimitPrice'] = str(slLimitPrice)

            # Create the market order using CCXT's 'create_order' function
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=qty,
                params=params
            )

            return order
        except Exception as e:
            logging.info(f"An error occurred while creating market order on Bybit: {e}")
            return None
        
    def cancel_order_by_id(self, order_id, symbol):
        try:
            # Call the updated cancel_order method
            result = self.exchange.cancel_order(id=order_id, symbol=symbol)
            logging.info(f"Canceled order - ID: {order_id}, Response: {result}")
        except Exception as e:
            logging.info(f"Error occurred in cancel_order_by_id: {e}")
           
    def cancel_take_profit_orders_bybit(self, symbol, side):
        side = side.lower()
        side_map = {"long": "buy", "short": "sell"}
        side = side_map.get(side, side)
        
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            position_idx_map = {"buy": 1, "sell": 2}

            for order in open_orders:
                if (
                    order['side'].lower() == side
                    and order['info'].get('reduceOnly')
                    and order['info'].get('positionIdx') == position_idx_map[side]
                ):
                    order_id = order['id']  # Assuming 'id' is the standard format expected by cancel_order
                    self.exchange.cancel_order(order_id, symbol)
                    logging.info(f"Canceled take profit order - ID: {order_id}")

        except Exception as e:
            logging.info(f"An unknown error occurred in cancel_take_profit_orders: {e}")

    def get_take_profit_order_quantity_bybit(self, symbol, side):
        side = side.lower()
        side_map = {"long": "buy", "short": "sell"}
        side = side_map.get(side, side)
        total_qty = 0
        
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            position_idx_map = {"buy": 1, "sell": 2}

            for order in open_orders:
                if (
                    order['side'].lower() == side
                    and order['info'].get('reduceOnly')
                    and order['info'].get('positionIdx') == position_idx_map[side]
                ):
                    total_qty += order.get('amount', 0)  # Assuming 'amount' contains the order quantity
        except Exception as e:
            logging.info(f"An unknown error occurred in get_take_profit_order_quantity_bybit: {e}")

        return total_qty

    def retry_api_call(self, function, *args, max_retries=100, base_delay=10, max_delay=60, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                return function(*args, **kwargs)
            except Exception as e:  # Catch all exceptions
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Error occurred: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        raise Exception(f"Failed to execute the API function after {max_retries} retries.")

    def get_contract_size_bybit(self, symbol):
        positions = self.exchange.fetch_derivatives_positions([symbol])
        return positions[0]['contractSize']

    def get_max_leverage_bybit(self, symbol, max_retries=10, backoff_factor=0.5):
        #logging.info(f"Called get_max_leverage_bybit with symbol: {symbol}")
        for retry in range(max_retries):
            try:
                tiers = self.exchange.fetch_derivatives_market_leverage_tiers(symbol)
                for tier in tiers:
                    info = tier.get('info', {})
                    if info.get('symbol') == symbol:
                        return float(info.get('maxLeverage'))
                return None  # If symbol not found in tiers

            except (RateLimitExceeded, NetworkError) as e:  # Include NetworkError
                # Log the exception
                logging.info(f"An error occurred while fetching max leverage: {str(e)}")

                # Wait and retry if not the last attempt
                if retry < max_retries - 1:  
                    sleep_time = backoff_factor * (2 ** retry)  # Exponential backoff
                    time.sleep(sleep_time)

            except Exception as e:
                # For any other types of exceptions, log and re-raise.
                logging.info(f"An unknown error occurred: {str(e)}")
                raise e

        raise Exception(f"Failed to get max leverage for {symbol} after {max_retries} retries.")

    def print_trade_quantities_bybit(self, max_trade_qty, leverage_sizes, wallet_exposure, best_ask_price):
        sorted_leverage_sizes = sorted(leverage_sizes)  # Sort leverage sizes in ascending order

        for leverage in sorted_leverage_sizes:
            trade_qty = max_trade_qty * leverage  # Calculate trade quantity based on leverage
            print(f"Leverage: {leverage}x, Trade Quantity: {trade_qty}")

    # Bybit calc lot size based on spread
    def spread_based_entry_size_bybit(self, symbol, spread, min_order_qty):
        current_price = self.get_current_price(symbol)
        logging.info(f"Current price debug: {current_price}")
        entry_amount = min_order_qty + (spread * current_price) / 100

        return entry_amount

    def bybit_fetch_precision(self, symbol):
        try:
            markets = self.exchange.fetch_derivatives_markets()
            for market in markets['result']['list']:
                if market['symbol'] == symbol:
                    qty_step = market['lotSizeFilter']['qtyStep']
                    self.market_precisions[symbol] = {'amount': float(qty_step)}
                    break
        except Exception as e:
            logging.info(f"Exception in bybit_fetch_precision: {e}")

    def get_market_tick_size_bybit(self, symbol):
        # Fetch the market data
        markets = self.exchange.fetch_markets()

        # Filter for the specific symbol
        for market in markets:
            if market['symbol'] == symbol:
                tick_size = market['info']['priceFilter']['tickSize']
                return tick_size
        
        return None

    def fetch_recent_trades(self, symbol, since=None, limit=100):
        """
        Fetch recent trades for a given symbol.

        :param str symbol: The trading pair symbol.
        :param int since: Timestamp in milliseconds for fetching trades since this time.
        :param int limit: The maximum number of trades to fetch.
        :return: List of recent trades.
        """
        try:
            # Ensure the markets are loaded
            self.exchange.load_markets()

            # Fetch trades using ccxt
            trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)
            return trades
        except Exception as e:
            logging.info(f"Error fetching recent trades for {symbol}: {e}")
            return []

    def fetch_unrealized_pnl(self, symbol):
        """
        Fetches the unrealized profit and loss (PNL) for both long and short positions of a given symbol.
        :param symbol: The trading pair symbol.
        :return: A dictionary containing the unrealized PNL for long and short positions.
                The dictionary has keys 'long' and 'short' with corresponding PNL values.
                Returns None for a position if it's not open or there's an error.
                Returns 0.0 if the 'unrealisedPnl' key is present but its value is an empty string.
        """
        # Fetch positions for the symbol
        response = self.exchange.fetch_positions([symbol])
        #logging.info(f"Response from unrealized pnl: {response}")

        unrealized_pnl = {'long': None, 'short': None}

        # Loop through each position in the response
        for pos in response:
            side = pos['info'].get('side', '').lower()
            pnl = pos['info'].get('unrealisedPnl', '')  # Default to empty string if 'unrealisedPnl' is not present

            if pnl == '':
                # If 'unrealisedPnl' is an empty string, set the PNL value to 0.0
                if side == 'buy':
                    unrealized_pnl['long'] = 0.0
                elif side == 'sell':
                    unrealized_pnl['short'] = 0.0
            else:
                try:
                    pnl = float(pnl)
                    if side == 'buy':
                        # Long position
                        unrealized_pnl['long'] = pnl
                    elif side == 'sell':
                        # Short position
                        unrealized_pnl['short'] = pnl
                    else:
                        logging.warning(f"Unknown side value for {symbol}: {side}")
                except (ValueError, TypeError) as e:
                    logging.info(f"Error converting unrealisedPnl to float for {symbol}: {e}")
                    # Set the PNL value to None if there's an error
                    if side == 'buy':
                        unrealized_pnl['long'] = None
                    elif side == 'sell':
                        unrealized_pnl['short'] = None

        return unrealized_pnl

    def process_position_data(self, open_position_data):
        position_details = {}

        for position in open_position_data:
            info = position.get('info', {})
            symbol = info.get('symbol', '').split(':')[0]  # Splitting to get the base symbol

            # Ensure 'size', 'side', and 'avgPrice' keys exist in the info dictionary
            if 'size' in info and 'side' in info and 'avgPrice' in info:
                size = float(info['size'])
                side = info['side'].lower()
                avg_price = float(info['avgPrice'])

                # Initialize the nested dictionary if the symbol is not already in position_details
                if symbol not in position_details:
                    position_details[symbol] = {'long': {'qty': 0, 'avg_price': None}, 'short': {'qty': 0, 'avg_price': None}}

                # Update the quantities and average prices based on the side of the position
                if side == 'buy':
                    position_details[symbol]['long']['qty'] += size
                    position_details[symbol]['long']['avg_price'] = avg_price
                elif side == 'sell':
                    position_details[symbol]['short']['qty'] += size
                    position_details[symbol]['short']['avg_price'] = avg_price

        return position_details