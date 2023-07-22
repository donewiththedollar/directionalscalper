import os
import logging
import time
import ccxt
import pandas as pd
import json
import requests, hmac, hashlib
import urllib.parse
from typing import Optional, Tuple
from ccxt.base.errors import RateLimitExceeded
from .strategies.logger import Logger

logging = Logger(filename="exchange.log", stream=True)

class Exchange:
    def __init__(self, exchange_id, api_key, secret_key, passphrase=None):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.name = exchange_id
        self.initialise()
        self.symbols = self._get_symbols()
        self.market_precisions = {}

    def initialise(self):
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange_params = {
            "apiKey": self.api_key,
            "secret": self.secret_key,
        }
        if os.environ.get('HTTP_PROXY') and os.environ.get('HTTPS_PROXY'):
            exchange_params["proxies"] = {
                'http': os.environ.get('HTTP_PROXY'),
                'https': os.environ.get('HTTPS_PROXY'),
            }
            
        if self.passphrase:
            exchange_params["password"] = self.passphrase

        exchange_id_lower = self.exchange_id.lower()

        if exchange_id_lower == 'huobi':
            exchange_params['options'] = {
                'defaultType': 'swap',
                'defaultSubType': 'linear',
            }
        elif exchange_id_lower == 'bybit_spot':
            exchange_params['options'] = {
                'defaultType': 'spot',
            }
        elif exchange_id_lower == 'binance':
            exchange_params['options'] = {
                'defaultType': 'future',
            }

        # if self.exchange_id.lower() == 'bybit':
        #     exchange_params['urls'] = {
        #         'api': 'https://api-testnet.bybit.com',
        #         'public': 'https://api-testnet.bybit.com',
        #         'private': 'https://api-testnet.bybit.com',
        #     }

        self.exchange = exchange_class(exchange_params)
        #print(self.exchange.describe())  # Print the exchange properties

    def _get_symbols(self):
        while True:
            try:
                #self.exchange.set_sandbox_mode(True)
                markets = self.exchange.load_markets()
                symbols = [market['symbol'] for market in markets.values()]
                return symbols
            except ccxt.errors.RateLimitExceeded as e:
                logging.info(f"Rate limit exceeded: {e}, retrying in 10 seconds...")
                time.sleep(10)
            except Exception as e:
                logging.info(f"An error occurred while fetching symbols: {e}, retrying in 10 seconds...")
                time.sleep(10)

    def check_account_type_huobi(self):
        if self.exchange_id.lower() != 'huobi':
            logging.info("This operation is only available for Huobi.")
            return

        response = self.exchange.contractPrivateGetLinearSwapApiV3SwapUnifiedAccountType()
        return response
    
    def switch_account_type_huobi(self, account_type: int):
        if self.exchange_id.lower() != 'huobi':
            logging.info("This operation is only available for Huobi.")
            return

        body = {
            "account_type": account_type
        }

        response = self.exchange.contractPrivatePostLinearSwapApiV3SwapSwitchAccountType(body)
        return response

    def calculate_max_trade_quantity(self, symbol, leverage, wallet_exposure, best_ask_price):
        # Fetch necessary data from the exchange
        market_data = self.get_market_data_bybit(symbol)
        dex_equity = self.get_balance_bybit('USDT')

        # Calculate the max trade quantity based on leverage and equity
        max_trade_qty = round(
            (float(dex_equity) * wallet_exposure / float(best_ask_price))
            / (100 / leverage),
            int(float(market_data['min_qty'])),
        )

        return max_trade_qty
    
    def get_market_tick_size_bybit(self, symbol):
        # Fetch the market data
        markets = self.exchange.fetch_markets()

        # Filter for the specific symbol
        for market in markets:
            if market['symbol'] == symbol:
                tick_size = market['info']['priceFilter']['tickSize']
                return tick_size
        
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

    # Bybit
    def calculate_trade_quantity(self, symbol, leverage, asset_wallet_exposure, best_ask_price):
        dex_equity = self.get_balance_bybit('USDT')
        asset_exposure = dex_equity * asset_wallet_exposure / 100.0
        trade_qty = asset_exposure / float(best_ask_price) / leverage
        return trade_qty

    # Bybit
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

    def debug_derivatives_positions(self, symbol):
        try:
            positions = self.exchange.fetch_derivatives_positions([symbol])
            logging.info(f"Debug positions: {positions}")
        except Exception as e:
            logging.info(f"Exception in debug derivs func: {e}")

    def debug_derivatives_markets_bybit(self):
        try:
            markets = self.exchange.fetch_derivatives_markets({'category': 'linear'})
            logging.info(f"Debug markets: {markets}")
        except Exception as e:
            logging.info(f"Exception in debug_derivatives_markets_bybit: {e}")

    # Bybit
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


    # # Bybit
    # def bybit_fetch_precision(self, symbol):
    #     market_data = self.exchange.fetch_derivatives_markets(symbol)
    #     self.market_precisions[symbol] = market_data['precision']

    # def bybit_fetch_precision(self, symbol):
    #     market_data = self.exchange.fetch_derivatives_markets([symbol])
    #     self.market_precisions[symbol] = market_data['precision']

    # def bybit_fetch_precision(self, symbol):
    #     market_data = self.exchange.fetch_derivatives_markets({'symbol': symbol})
    #     self.market_precisions[symbol] = market_data['precision']

    # def bybit_fetch_precision(self, symbol):
    #     try:
    #         market_data = self.exchange.fetch_derivatives_markets(symbol)
    #         print("Market Data:", market_data)
    #     except Exception as e:
    #         print(f"Error in fetching precision: {e}")


    # Bybit
    def get_current_leverage_bybit(self, symbol):
        try:
            positions = self.exchange.fetch_derivatives_positions([symbol])
            if len(positions) > 0:
                position = positions[0]
                leverage = position['leverage']
                logging.info(f"Current leverage for symbol {symbol}: {leverage}")
            else:
                logging.info(f"No positions found for symbol {symbol}")
        except Exception as e:
            logging.info(f"Error retrieving current leverage: {e}")
            
    # Bybit
    def set_leverage_bybit(self, leverage, symbol):
        try:
            self.exchange.set_leverage(leverage, symbol)
            logging.info(f"Leverage set to {leverage} for symbol {symbol}")
        except Exception as e:
            logging.info(f"Error setting leverage: {e}")

    # Bybit
    def setup_exchange_bybit(self, symbol) -> None:
        values = {"position": False, "leverage": False}
        try:
            # Set the position mode to hedge
            self.exchange.set_position_mode(hedged=True, symbol=symbol)
            values["position"] = True
        except Exception as e:
            logging.info(f"An unknown error occurred in with set_position_mode: {e}")

        market_data = self.get_market_data_bybit(symbol=symbol)
        try:
            # Set the margin mode to cross
            self.exchange.set_derivatives_margin_mode(marginMode="cross", symbol=symbol)

        except Exception as e:
            logging.info(f"An unknown error occurred in with set_derivatives_margin_mode: {e}")

        # log.info(values)

    def parse_trading_fee(self, fee_data):
        maker_fee = float(fee_data.get('makerFeeRate', '0'))
        taker_fee = float(fee_data.get('takerFeeRate', '0'))
        return {
            'maker_fee': maker_fee,
            'taker_fee': taker_fee
        }
    
    # Mexc
    def get_market_data_mexc(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)

            # Extract the desired values from symbol_data
            if "precision" in symbol_data:
                values["precision"] = symbol_data["precision"]["price"]
            if "limits" in symbol_data:
                values["min_qty"] = symbol_data["limits"]["amount"]["min"]
            # Note that leverage is not available in the provided symbol_data for the mexc exchange
            values["leverage"] = None

        except Exception as e:
            logging.info(f"An unknown error occurred in get_market_data_mexc(): {e}")
        return values

    # Bitget
    def get_current_candle_bitget(self, symbol: str, timeframe='1m', retries=3, delay=60):
        """
        Fetches the current candle for a given symbol and timeframe from Bitget.

        :param str symbol: unified symbol of the market to fetch OHLCV data for
        :param str timeframe: the length of time each candle represents
        :returns [int]: A list representing the current candle [timestamp, open, high, low, close, volume]
        """
        for _ in range(retries):
            try:
                # Fetch the most recent 2 candles
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=2)

                # The last element in the list is the current (incomplete) candle
                current_candle = ohlcv[-1]

                return current_candle

            except RateLimitExceeded:
                print("Rate limit exceeded... sleeping for {} seconds".format(delay))
                time.sleep(delay)
        
        raise RateLimitExceeded("Failed to fetch candle data after {} retries".format(retries))

    # Bitget 
    def set_leverage_bitget(self, symbol, leverage, params={}):
        """
        Set the level of leverage for a market.

        :param str symbol: unified market symbol
        :param float leverage: the rate of leverage
        :param dict params: extra parameters specific to the Bitget API endpoint
        :returns dict: response from the exchange
        """
        try:
            if hasattr(self.exchange, 'set_leverage'):
                return self.exchange.set_leverage(leverage, symbol, params)
            else:
                print(f"The {self.exchange_id} exchange doesn't support setting leverage.")
                return None
        except ccxt.BaseError as e:
            print(f"An error occurred while setting leverage: {e}")
            return None


    # Bitget
    def get_market_data_bitget(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)

            if self.exchange.id == 'bybit':
                if "info" in symbol_data:
                    values["precision"] = symbol_data["info"]["price_scale"]
                    values["leverage"] = symbol_data["info"]["leverage_filter"][
                        "max_leverage"
                    ]
                    values["min_qty"] = symbol_data["info"]["lot_size_filter"][
                        "min_trading_qty"
                    ]
            elif self.exchange.id == 'bitget':
                if "precision" in symbol_data:
                    values["precision"] = symbol_data["precision"]["price"]
                if "limits" in symbol_data:
                    values["min_qty"] = symbol_data["limits"]["amount"]["min"]
            else:
                logging.info("Exchange not recognized for fetching market data.")

        except Exception as e:
            logging.info(f"An unknown error occurred in get_market_data(): {e}")
        return values

    # Bybit
    def get_market_data_bybit(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)
            
            #print("Symbol data:", symbol_data)  # Debug print
            
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
        return values

    # Huobi
    # def fetch_max_leverage_huobi(self, symbol):
    #     """
    #     Retrieve the maximum leverage for a given symbol
    #     :param str symbol: unified market symbol
    #     :returns int: maximum leverage for the symbol
    #     """
    #     leverage_tiers = self.exchange.fetch_leverage_tiers([symbol])
    #     if symbol in leverage_tiers:
    #         symbol_tiers = leverage_tiers[symbol]
    #         print(symbol_tiers)  # print the content of symbol_tiers
    #         max_leverage = max([tier['lever_rate'] for tier in symbol_tiers])
    #         return max_leverage
    #     else:
    #         return None

    def fetch_max_leverage_huobi(self, symbol):
        """
        Retrieve the maximum leverage for a given symbol
        :param str symbol: unified market symbol
        :returns int: maximum leverage for the symbol
        """
        leverage_tiers = self.exchange.fetch_leverage_tiers([symbol])
        if symbol in leverage_tiers:
            symbol_tiers = leverage_tiers[symbol]
            #print(symbol_tiers)  # print the content of symbol_tiers
            #max_leverage = max([tier['lever_rate'] for tier in symbol_tiers])
            max_leverage = max([tier['maxLeverage'] for tier in symbol_tiers])
            return max_leverage
        else:
            return None

        
    # Huobi
    def get_market_data_huobi(self, symbol: str) -> dict:
        values = {"precision": 0.0, "min_qty": 0.0, "leverage": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)
            
            if "precision" in symbol_data:
                values["precision"] = symbol_data["precision"]["price"]
            if "limits" in symbol_data:
                values["min_qty"] = symbol_data["limits"]["amount"]["min"]
            if "info" in symbol_data and "leverage-ratio" in symbol_data["info"]:
                values["leverage"] = float(symbol_data["info"]["leverage-ratio"])
        except Exception as e:
            logging.info(f"An unknown error occurred in get_market_data_huobi(): {e}")
        return values

    # Bybit
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

    # Bybit
    def get_balance_bybit(self, quote):
        if self.exchange.has['fetchBalance']:
            # Fetch the balance
            balance = self.exchange.fetch_balance()

            # Find the quote balance
            for currency_balance in balance['info']['result']['list']:
                if currency_balance['coin'] == quote:
                    return float(currency_balance['equity'])
        return None

    # Bybit
    def get_available_balance_bybit(self, quote):
        if self.exchange.has['fetchBalance']:
            # Fetch the balance
            balance = self.exchange.fetch_balance()

            # Find the quote balance
            try:
                for currency_balance in balance['info']['result']['list']:
                    if currency_balance['coin'] == quote:
                        return float(currency_balance['availableBalance'])
            except KeyError as e:
                print(f"KeyError: {e}")
                print(balance)  # Print the balance if there was a KeyError
        return None

    def get_available_balance_huobi(self, symbol):
        try:
            balance_data = self.exchange.fetch_balance()
            contract_details = balance_data.get('info', {}).get('data', [{}])[0].get('futures_contract_detail', [])
            for contract in contract_details:
                if contract['contract_code'] == symbol:
                    return float(contract['margin_available'])
            return "No contract found for symbol " + symbol
        except Exception as e:
            return f"An error occurred while fetching balance: {str(e)}"


    def debug_print_balance_huobi(self):
        try:
            balance = self.exchange.fetch_balance()
            print("Full balance data:")
            print(balance)
        except Exception as e:
            print(f"An error occurred while fetching balance: {str(e)}")



    # Binance
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

    def get_balance_mexc(self, quote, market_type='swap'):
        if self.exchange.has['fetchBalance']:
            # Fetch the balance
            balance = self.exchange.fetch_balance(params={"type": market_type})

            # Find the quote balance
            if quote in balance['total']:
                return float(balance['total'][quote])
        return None

    def get_balance_huobi(self, quote, type='spot', subType='linear', marginMode='cross'):
        if self.exchange.has['fetchBalance']:
            params = {
                'type': type,
                'subType': subType,
                'marginMode': marginMode,
                'unified': False
            }
            balance = self.exchange.fetch_balance(params)
            if quote in balance:
                return balance[quote]['free']
        return None

    def get_balance_huobi_unified(self, quote, type='spot', subType='linear', marginMode='cross'):
        if self.exchange.has['fetchBalance']:
            params = {
                'type': type,
                'subType': subType,
                'marginMode': marginMode,
                'unified': True
            }
            balance = self.exchange.fetch_balance(params)
            if quote in balance:
                return balance[quote]['free']
        return None

    def cancel_order_huobi(self, id: str, symbol: Optional[str] = None, params={}):
        try:
            result = self.exchange.cancel_order(id, symbol, params)
            return result
        except Exception as e:
            # Log exception details, if any
            print(f"Failed to cancel order: {str(e)}")
            return None

    def get_contract_orders_huobi(self, symbol, status='open', type='limit', limit=20):
        if self.exchange.has['fetchOrders']:
            params = {
                'symbol': symbol,
                'status': status,  # 'open', 'closed', or 'all'
                'type': type,  # 'limit' or 'market'
                'limit': limit,  # maximum number of orders
            }
            return self.exchange.fetch_orders(params)
        return None
    
    def fetch_balance_huobi(self, params={}):
        # Call base class's fetch_balance for spot balances
        spot_balance = self.exchange.fetch_balance(params)
        print("Spot Balance:", spot_balance)

        # Fetch margin balances
        margin_balance = self.fetch_margin_balance_huobi(params)
        print("Margin Balance:", margin_balance)

        # Fetch futures balances
        futures_balance = self.fetch_futures_balance_huobi(params)
        print("Futures Balance:", futures_balance)

        # Fetch swap balances
        swaps_balance = self.fetch_swaps_balance_huobi(params)
        print("Swaps Balance:", swaps_balance)

        # Combine balances
        total_balance = self.exchange.deep_extend(spot_balance, margin_balance, futures_balance, swaps_balance)

        # Remove the unnecessary information
        parsed_balance = {}
        for currency in total_balance:
            parsed_balance[currency] = {
                'free': total_balance[currency]['free'],
                'used': total_balance[currency]['used'],
                'total': total_balance[currency]['total']
            }

        return parsed_balance

    def fetch_margin_balance_huobi(self, params={}):
        response = self.exchange.private_get_margin_accounts_balance(params)
        return self._parse_huobi_balance(response)

    def fetch_futures_balance_huobi(self, params={}):
        response = self.exchange.linearGetV2AccountInfo(params)
        return self._parse_huobi_balance(response)

    def fetch_swaps_balance_huobi(self, params={}):
        response = self.exchange.swapGetSwapBalance(params)
        return self._parse_huobi_balance(response)

    def _parse_huobi_balance(self, response):
        if 'data' in response:
            balance_data = response['data']
            parsed_balance = {}
            for currency_data in balance_data:
                currency = currency_data['currency']
                parsed_balance[currency] = {
                    'free': float(currency_data.get('available', 0)),
                    'used': float(currency_data.get('frozen', 0)),
                    'total': float(currency_data.get('balance', 0))
                }
            return parsed_balance
        else:
            return {}


    def get_precision_ultimate_bybit(self, symbol: str) -> Tuple[int, int]:
        try:
            market = self.exchange.market(symbol)

            smallest_increment_price = market['precision']['price']
            price_precision = len(str(smallest_increment_price).split('.')[-1])

            quantity_precision = int(market['precision']['amount'])

            return price_precision, quantity_precision
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None


    def get_symbol_precision_bybit(self, symbol: str) -> Tuple[int, int]:
        try:
            market = self.exchange.market(symbol)
            price_precision = int(market['precision']['price'])
            quantity_precision = int(market['precision']['amount'])
            return price_precision, quantity_precision
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None

    def get_price_precision(self, symbol):
        market = self.exchange.market(symbol)
        smallest_increment = market['precision']['price']
        price_precision = len(str(smallest_increment).split('.')[-1])
        return price_precision

    def get_precision_bybit(self, symbol):
        markets = self.exchange.fetch_derivatives_markets()
        for market in markets:
            if market['symbol'] == symbol:
                return market['precision']
        return None


    def get_balance(self, quote: str) -> dict:
        values = {
            "available_balance": 0.0,
            "pnl": 0.0,
            "upnl": 0.0,
            "wallet_balance": 0.0,
            "equity": 0.0,
        }
        try:
            data = self.exchange.fetch_balance()
            if "info" in data:
                if "result" in data["info"]:
                    if quote in data["info"]["result"]:
                        values["available_balance"] = float(
                            data["info"]["result"][quote]["available_balance"]
                        )
                        values["pnl"] = float(
                            data["info"]["result"][quote]["realised_pnl"]
                        )
                        values["upnl"] = float(
                            data["info"]["result"][quote]["unrealised_pnl"]
                        )
                        values["wallet_balance"] = round(
                            float(data["info"]["result"][quote]["wallet_balance"]), 2
                        )
                        values["equity"] = round(
                            float(data["info"]["result"][quote]["equity"]), 2
                        )
        except Exception as e:
            logging.info(f"An unknown error occurred in get_balance(): {e}")
        return values
    
    # Universal
    def fetch_ohlcv(self, symbol, timeframe='1d'):
        """
        Fetch OHLCV data for the given symbol and timeframe.

        :param symbol: Trading symbol.
        :param timeframe: Timeframe string.
        :return: DataFrame with OHLCV data.
        """
        try:
            # Fetch the OHLCV data from the exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            df.set_index('timestamp', inplace=True)

            return df

        except ccxt.BaseError as e:
            print(f'Failed to fetch OHLCV data: {e}')
            return pd.DataFrame()
        
    def get_orderbook(self, symbol, max_retries=3, retry_delay=5) -> dict:
        values = {"bids": [], "asks": []}
        
        for i in range(max_retries):
            try:
                data = self.exchange.fetch_order_book(symbol)
                if "bids" in data and "asks" in data:
                    if len(data["bids"]) > 0 and len(data["asks"]) > 0:
                        if len(data["bids"][0]) > 0 and len(data["asks"][0]) > 0:
                            values["bids"] = data["bids"]
                            values["asks"] = data["asks"]
                break  # if the fetch was successful, break out of the loop
            except Exception as e:
                if i < max_retries - 1:  # if not the last attempt
                    logging.info(f"An unknown error occurred in get_orderbook(): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.info(f"Failed to fetch order book after {max_retries} attempts: {e}")
                    raise e  # If it's still failing after max_retries, re-raise the exception.
        
        return values

    # def get_orderbook(self, symbol) -> dict:
    #     values = {"bids": [], "asks": []}
    #     try:
    #         data = self.exchange.fetch_order_book(symbol)
    #         if "bids" in data and "asks" in data:
    #             if len(data["bids"]) > 0 and len(data["asks"]) > 0:
    #                 if len(data["bids"][0]) > 0 and len(data["asks"][0]) > 0:
    #                     values["bids"] = data["bids"]
    #                     values["asks"] = data["asks"]
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in get_orderbook(): {e}")
    #     return values

    # Bitget
    def get_positions_bitget(self, symbol) -> dict:
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
            data = self.exchange.fetch_positions([symbol])
            for position in data:
                side = position["side"]
                values[side]["qty"] = float(position["contracts"])  # Use "contracts" instead of "contractSize"
                values[side]["price"] = float(position["entryPrice"])
                values[side]["realised"] = round(float(position["info"]["achievedProfits"]), 4)
                values[side]["upnl"] = round(float(position["unrealizedPnl"]), 4)
                if position["liquidationPrice"] is not None:
                    values[side]["liq_price"] = float(position["liquidationPrice"])
                else:
                    print(f"Warning: liquidationPrice is None for {side} position")
                    values[side]["liq_price"] = None
                values[side]["entry_price"] = float(position["entryPrice"])
        except Exception as e:
            logging.info(f"An unknown error occurred in get_positions_bitget(): {e}")
        return values

    # Bybit
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


    # Bybit 
    def get_positions_bybit(self, symbol, max_retries=3, retry_delay=5) -> dict:
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
            if len(position_data) > 0:
                for position in position_data:
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





    # def get_positions_binance(self, symbol) -> dict:
    #     values = {
    #         "long": {
    #             "qty": 0.0,
    #             "price": 0.0,
    #             "realised": 0,
    #             "cum_realised": 0,
    #             "upnl": 0,
    #             "upnl_pct": 0,
    #             "liq_price": 0,
    #             "entry_price": 0,
    #         },
    #         "short": {
    #             "qty": 0.0,
    #             "price": 0.0,
    #             "realised": 0,
    #             "cum_realised": 0,
    #             "upnl": 0,
    #             "upnl_pct": 0,
    #             "liq_price": 0,
    #             "entry_price": 0,
    #         },
    #     }
    #     try:
    #         data = self.exchange.fetch_positions_risk([symbol])
    #         print(data)
    #         if len(data) > 0:
    #             for position in data:
    #                 position_side = position["positionSide"].lower()
    #                 values[position_side]["qty"] = float(position["positionAmt"])
    #                 values[position_side]["price"] = float(position["entryPrice"] or 0)
    #                 values[position_side]["realised"] = round(float(position["unRealizedProfit"] or 0), 4)
    #                 values[position_side]["cum_realised"] = round(float(position["unRealizedProfit"] or 0), 4)
    #                 values[position_side]["upnl"] = round(float(position["unRealizedProfit"] or 0), 4)
    #                 values[position_side]["upnl_pct"] = 0  # Binance does not provide the unrealized PnL percentage
    #                 values[position_side]["liq_price"] = float(position["liquidationPrice"] or 0)
    #                 values[position_side]["entry_price"] = float(position["entryPrice"] or 0)
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in get_positions(): {e}")
    #     return values

    
    # # Huobi
    # def safe_order_operation(self, operation, *args, **kwargs):
    #     while True:
    #         try:
    #             return operation(*args, **kwargs)
    #         except ccxt.BaseError as e:
    #             if 'In settlement' in str(e) or 'In delivery' in str(e):
    #                 print(f"Contract is in settlement or delivery. Cannot perform operation currently. Retrying in 10 seconds...")
    #                 time.sleep(10)
    #             else:
    #                 raise

    # # Huobi
    # def safe_order_operation(self, operation, *args, **kwargs):
    #     while True:
    #         try:
    #             return operation(*args, **kwargs)
    #         except ccxt.BaseError as e:
    #             e_str = str(e)
    #             if 'In settlement' in e_str or 'In delivery' in e_str:
    #                 print(f"Contract is in settlement or delivery. Cannot perform operation currently. Retrying in 10 seconds...")
    #                 time.sleep(10)
    #             elif 'Insufficient close amount available' in e_str:
    #                 print(f"Insufficient close amount available. Retrying in 5 seconds...")
    #                 time.sleep(5)
    #             else:
    #                 raise

    #Huobi 
    def safe_order_operation(self, operation, *args, **kwargs):
        while True:
            try:
                return operation(*args, **kwargs)
            except ccxt.BaseError as e:
                e_str = str(e)
                if 'In settlement' in e_str or 'In delivery' in e_str or 'Settling. Unable to place/cancel orders currently.' in e_str:
                    print(f"Contract is in settlement or delivery. Cannot perform operation currently. Retrying in 10 seconds...")
                    time.sleep(10)
                elif 'Insufficient close amount available' in e_str:
                    print(f"Insufficient close amount available. Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    raise


    # Huobi                
    def get_contract_size_huobi(self, symbol):
        markets = self.exchange.fetch_markets_by_type_and_sub_type('swap', 'linear')
        for market in markets:
            if market['symbol'] == symbol:
                return market['contractSize']
        return None
    

    # Huobi
    def get_positions_huobi(self, symbol) -> dict:
        print(f"Symbol received in get_positions_huobi: {symbol}")
        self.exchange.load_markets()
        if symbol not in self.exchange.markets:
            print(f"Market symbol {symbol} not found in Huobi markets.")
            return None
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
            data = self.exchange.fetch_positions([symbol])
            for position in data:
                if "info" not in position or "direction" not in position["info"]:
                    continue
                side = "long" if position["info"]["direction"] == "buy" else "short"
                values[side]["qty"] = float(position["info"]["volume"])  # Updated to use 'volume'
                values[side]["price"] = float(position["info"]["cost_open"])
                values[side]["realised"] = float(position["info"]["profit"])
                values[side]["cum_realised"] = float(position["info"]["profit"])  # Huobi API doesn't seem to provide cumulative realised profit
                values[side]["upnl"] = float(position["info"]["profit_unreal"])
                values[side]["upnl_pct"] = float(position["info"]["profit_rate"])
                values[side]["liq_price"] = 0.0  # Huobi API doesn't seem to provide liquidation price
                values[side]["entry_price"] = float(position["info"]["cost_open"])
        except Exception as e:
            logging.info(f"An unknown error occurred in get_positions_huobi(): {e}")
        return values

    # Huobi debug
    def get_positions_debug(self):
        try:
            positions = self.exchange.fetch_positions()
            print(f"{positions}")
        except Exception as e:
            print(f"Error is {e}")

    def get_positions(self, symbol) -> dict:
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
            data = self.exchange.fetch_positions([symbol])
            if len(data) == 2:
                sides = ["long", "short"]
                for side in [0, 1]:
                    values[sides[side]]["qty"] = float(data[side]["contracts"])
                    values[sides[side]]["price"] = float(data[side]["entryPrice"])
                    values[sides[side]]["realised"] = round(
                        float(data[side]["info"]["realised_pnl"]), 4
                    )
                    values[sides[side]]["cum_realised"] = round(
                        float(data[side]["info"]["cum_realised_pnl"]), 4
                    )
                    if data[side]["info"]["unrealised_pnl"] is not None:
                        values[sides[side]]["upnl"] = round(
                            float(data[side]["info"]["unrealised_pnl"]), 4
                        )
                    if data[side]["precentage"] is not None:
                        values[sides[side]]["upnl_pct"] = round(
                            float(data[side]["precentage"]), 4
                        )
                    if data[side]["liquidationPrice"] is not None:
                        values[sides[side]]["liq_price"] = float(
                            data[side]["liquidationPrice"]
                        )
                    if data[side]["entryPrice"] is not None:
                        values[sides[side]]["entry_price"] = float(
                            data[side]["entryPrice"]
                        )
        except Exception as e:
            logging.info(f"An unknown error occurred in get_positions(): {e}")
        return values

    # Universal
    def get_current_price(self, symbol: str) -> float:
        current_price = 0.0
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            if "bid" in ticker and "ask" in ticker:
                current_price = (ticker["bid"] + ticker["ask"]) / 2
        except Exception as e:
            logging.info(f"An unknown error occurred in get_positions(): {e}")
        return current_price

    def get_moving_averages(self, symbol: str, timeframe: str = "1m", num_bars: int = 20, max_retries=3, retry_delay=5) -> dict:
        values = {"MA_3_H": 0.0, "MA_3_L": 0.0, "MA_6_H": 0.0, "MA_6_L": 0.0}

        for i in range(max_retries):
            try:
                bars = self.exchange.fetch_ohlcv(
                    symbol=symbol, timeframe=timeframe, limit=num_bars
                )
                df = pd.DataFrame(
                    bars, columns=["Time", "Open", "High", "Low", "Close", "Volume"]
                )
                df["Time"] = pd.to_datetime(df["Time"], unit="ms")
                df["MA_3_High"] = df.High.rolling(3).mean()
                df["MA_3_Low"] = df.Low.rolling(3).mean()
                df["MA_6_High"] = df.High.rolling(6).mean()
                df["MA_6_Low"] = df.Low.rolling(6).mean()
                values["MA_3_H"] = df["MA_3_High"].iat[-1]
                values["MA_3_L"] = df["MA_3_Low"].iat[-1]
                values["MA_6_H"] = df["MA_6_High"].iat[-1]
                values["MA_6_L"] = df["MA_6_Low"].iat[-1]
                break  # If the fetch was successful, break out of the loop
            except Exception as e:
                if i < max_retries - 1:  # If not the last attempt
                    logging.info(f"An unknown error occurred in get_moving_averages(): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.info(f"Failed to fetch moving averages after {max_retries} attempts: {e}")
                    raise e  # If it's still failing after max_retries, re-raise the exception.
        
        return values

    # def get_moving_averages(
    #     self, symbol: str, timeframe: str = "1m", num_bars: int = 20
    # ) -> dict:
    #     values = {"MA_3_H": 0.0, "MA_3_L": 0.0, "MA_6_H": 0.0, "MA_6_L": 0.0}
    #     try:
    #         bars = self.exchange.fetch_ohlcv(
    #             symbol=symbol, timeframe=timeframe, limit=num_bars
    #         )
    #         df = pd.DataFrame(
    #             bars, columns=["Time", "Open", "High", "Low", "Close", "Volume"]
    #         )
    #         df["Time"] = pd.to_datetime(df["Time"], unit="ms")
    #         df["MA_3_High"] = df.High.rolling(3).mean()
    #         df["MA_3_Low"] = df.Low.rolling(3).mean()
    #         df["MA_6_High"] = df.High.rolling(6).mean()
    #         df["MA_6_Low"] = df.Low.rolling(6).mean()
    #         values["MA_3_H"] = df["MA_3_High"].iat[-1]
    #         values["MA_3_L"] = df["MA_3_Low"].iat[-1]
    #         values["MA_6_H"] = df["MA_6_High"].iat[-1]
    #         values["MA_6_L"] = df["MA_6_Low"].iat[-1]
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in get_moving_averages(): {e}")
    #     return values

    def get_open_orders(self, symbol: str) -> list:
        open_orders_list = []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            #print(orders)
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        #print(f"Order info: {order['info']}")  # Debug print
                        order_info = {
                            "id": order["info"]["orderId"],
                            "price": float(order["info"]["price"]),
                            "qty": float(order["info"]["qty"]),
                            "order_status": order["info"]["orderStatus"],
                            "side": order["info"]["side"],
                            "reduce_only": order["info"]["reduceOnly"],  # Update this line
                            "position_idx": int(order["info"]["positionIdx"])  # Add this line
                        }
                        open_orders_list.append(order_info)
        except Exception as e:
            logging.info(f"An unknown error occurred in get_open_orders(): {e}")
        return open_orders_list

    # Binance
    def get_open_orders_binance(self, symbol: str) -> list:
        open_orders_list = []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        order_info = {
                            "id": order["id"],
                            "price": float(order["price"]),
                            "qty": float(order["amount"]),
                            "order_status": order["status"],
                            "side": order["side"],
                            "reduce_only": False,  # Binance does not have a "reduceOnly" field
                            "position_idx": None  # Binance does not have a "positionIdx" field
                        }
                        open_orders_list.append(order_info)
        except Exception as e:
            logging.info(f"An unknown error occurred in get_open_orders(): {e}")
        return open_orders_list


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

    def get_open_orders_huobi(self, symbol: str) -> list:
        open_orders_list = []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            #print(f"Debug: {orders}")
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        try:
                            # Extracting the necessary fields for Huobi orders
                            order_info = {
                                "id": order["info"]["order_id"],
                                "price": float(order["info"]["price"]),
                                "qty": float(order["info"]["volume"]),
                                "order_status": order["info"]["status"],
                                "side": order["info"]["direction"],  # assuming 'direction' indicates 'buy' or 'sell'
                            }
                            open_orders_list.append(order_info)
                        except KeyError as e:
                            logging.info(f"Key {e} not found in order info.")
        except Exception as e:
            logging.info(f"An unknown error occurred in get_open_orders_huobi(): {e}")
        return open_orders_list

    def debug_open_orders(self, symbol: str) -> None:
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            logging.info(open_orders)
        except:
            logging.info(f"Fuck")

    def cancel_long_entry(self, symbol: str) -> None:
        self._cancel_entry(symbol, order_side="Buy")

    def cancel_short_entry(self, symbol: str) -> None:
        self._cancel_entry(symbol, order_side="Sell")

    def _cancel_entry(self, symbol: str, order_side: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            for order in orders:
                if "info" in order:
                    order_id = order["info"]["order_id"]
                    order_status = order["info"]["order_status"]
                    side = order["info"]["side"]
                    reduce_only = order["info"]["reduce_only"]
                    if (
                        order_status != "Filled"
                        and side == order_side
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling {order_side} order: {order_id}")
        except Exception as e:
            logging.info(f"An unknown error occurred in _cancel_entry(): {e}")

    # Binance
    def cancel_all_entries_binance(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            long_orders = 0
            short_orders = 0

            # Count the number of open long and short orders
            for order in orders:
                order_status = order["status"]
                order_side = order["side"]
                reduce_only = False  # Binance does not have a "reduceOnly" field
                position_idx = None  # Binance does not have a "positionIdx" field

                if order_status != "closed" and not reduce_only:
                    if position_idx == 1 and order_side == "buy":
                        long_orders += 1
                    elif position_idx == 2 and order_side == "sell":
                        short_orders += 1

            # Cancel extra long or short orders if more than one open order per side
            if long_orders > 1 or short_orders > 1:
                for order in orders:
                    order_id = order["id"]
                    order_status = order["status"]
                    order_side = order["side"]
                    reduce_only = False  # Binance does not have a "reduceOnly" field
                    position_idx = None  # Binance does not have a "positionIdx" field

                    if order_status != "closed" and not reduce_only:
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
                        # log.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.info(f"An unknown error occurred in cancel_all_entries_binance(): {e}")

    def cancel_all_entries_bybit(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            long_orders = 0
            short_orders = 0

            # Count the number of open long and short orders
            for order in orders:
                order_info = order["info"]
                order_status = order_info["orderStatus"]
                order_side = order_info["side"]
                reduce_only = order_info["reduceOnly"]
                position_idx = int(order_info["positionIdx"])

                if order_status != "Filled" and order_status != "Cancelled" and not reduce_only:
                    if position_idx == 1 and order_side == "Buy":
                        long_orders += 1
                    elif position_idx == 2 and order_side == "Sell":
                        short_orders += 1

            # Cancel extra long or short orders
            if long_orders > 0 or short_orders > 0:
                for order in orders:
                    order_info = order["info"]
                    order_id = order_info["orderId"]
                    order_status = order_info["orderStatus"]
                    order_side = order_info["side"]
                    reduce_only = order_info["reduceOnly"]
                    position_idx = int(order_info["positionIdx"])

                    if (
                        order_status != "Filled"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_all_entries_bybit(): {e}")

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
            logging.warning(f"An unknown error occurred in cancel_entry(): {e}")


    # Binance
    def get_max_leverage_binance(self, symbol):
        if self.exchange.has['fetchLeverageTiers']:
            tiers = self.exchange.fetch_leverage_tiers()
            if symbol in tiers:
                brackets = tiers[symbol].get('brackets', [])
                if len(brackets) > 0:
                    maxLeverage = brackets[0].get('initialLeverage')
                    if maxLeverage is not None:
                        return float(maxLeverage)
        return None

    # Bybit
    def get_contract_size_bybit(self, symbol):
        positions = self.exchange.fetch_derivatives_positions([symbol])
        return positions[0]['contractSize']

    # Bybit
    def get_max_leverage_bybit(self, symbol):
        tiers = self.exchange.fetch_derivatives_market_leverage_tiers(symbol)
        for tier in tiers:
            info = tier.get('info', {})
            if info.get('symbol') == symbol:
                return float(info.get('maxLeverage'))
        return None
    
    # Bitget 
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


    def cancel_entry(self, symbol: str) -> None:
        try:
            order = self.exchange.fetch_open_orders(symbol)
            #print(f"Open orders for {symbol}: {order}")
            if len(order) > 0:
                if "info" in order[0]:
                    order_id = order[0]["info"]["order_id"]
                    order_status = order[0]["info"]["order_status"]
                    order_side = order[0]["info"]["side"]
                    reduce_only = order[0]["info"]["reduce_only"]
                    if (
                        order_status != "Filled"
                        and order_side == "Buy"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
                    elif (
                        order_status != "Filled"
                        and order_side == "Sell"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_entry(): {e}")

    # Bybit
    def cancel_take_profit_orders_bybit(self, symbol, side):
        side = side.lower()
        side_map = {"long": "buy", "short": "sell"}
        side = side_map.get(side, side)
        
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            position_idx_map = {"buy": 1, "sell": 2}
            #print("Open Orders:", open_orders)
            #print("Position Index Map:", position_idx_map)
            for order in open_orders:
                if (
                    order['side'].lower() == side
                    and order['info'].get('reduceOnly')
                    and order['info'].get('positionIdx') == position_idx_map[side]
                ):
                    order_id = order['info']['orderId']
                    self.exchange.cancel_derivatives_order(order_id, symbol)
                    logging.info(f"Canceled take profit order - ID: {order_id}")
        except Exception as e:
            print(f"An unknown error occurred in cancel_take_profit_orders: {e}")

    # Bybit
    def cancel_order_by_id(self, order_id, symbol):
        try:
            self.exchange.cancel_derivatives_order(order_id, symbol)
            logging.info(f"Canceled take profit order - ID: {order_id}")
        except Exception as e:
            logging.info(f"An unknown error occurred in cancel_take_profit_orders: {e}")

    # def cancel_take_profit_orders_bybit(self, symbol, side):
    #     try:
    #         open_orders = self.exchange.fetch_open_orders(symbol)
    #         position_idx_map = {"long": 1, "short": 2}
    #         position_idx_map = {"buy": 1, "sell": 2}
    #         for order in open_orders:
    #             if (
    #                 order['side'].lower() == side.lower()
    #                 and order['info'].get('reduceOnly')
    #                 and order['info'].get('positionIdx') == position_idx_map[side]
    #             ):
    #                 order_id = order['info']['orderId']
    #                 self.exchange.cancel_derivatives_order(order_id, symbol)
    #                 print(f"Canceled take profit order - ID: {order_id}")
    #     except Exception as e:
    #         print(f"An unknown error occurred in cancel_take_profit_orders: {e}")

    def cancel_close_bybit(self, symbol: str, side: str) -> None:
        side = side.lower()
        side_map = {"long": "buy", "short": "sell"}
        side = side_map.get(side, side)
        
        position_idx_map = {"buy": 1, "sell": 2}
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
        except Exception as e:
            logging.warning(f"An unknown error occurred in cancel_close_bybit(): {e}")

    def huobi_test_orders(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            print(orders)
        except Exception as e:
            print(f"Exception caught {e}")

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

    # def cancel_close_huobi(self, symbol: str, side: str) -> None:
    #     side_map = {"long": "buy", "short": "sell"}
    #     offset = "close"
    #     try:
    #         orders = self.exchange.fetch_open_orders(symbol)
    #         if orders:
    #             for order in orders:
    #                 order_info = order["info"]
    #                 order_id = order_info["order_id"]
    #                 order_status = order_info["status"]
    #                 order_direction = order_info["direction"]
    #                 order_offset = order_info["offset"]
    #                 reduce_only = order_info["reduce_only"]

    #                 if (
    #                     order_status == '3'  # Assuming '3' represents open orders
    #                     and order_direction == side_map[side]
    #                     and order_offset == offset
    #                     and reduce_only == '1'  # Assuming '1' represents reduce_only orders
    #                 ):
    #                     self.exchange.cancel_order(symbol=symbol, id=order_id)
    #                     log.info(f"Cancelling order: {order_id}")
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in cancel_close_huobi(): {e}")

    # def cancel_close_huobi(self, symbol: str, side: str) -> None:
    #     side_map = {"long": "sell", "short": "buy"}
    #     offset_map = {"long": "close", "short": "close"}
    #     try:
    #         orders = self.exchange.fetch_open_orders(symbol)
    #         if len(orders) > 0:
    #             for order in orders:
    #                 order_info = order["info"]
    #                 order_id = order_info["order_id"]
    #                 order_status = str(order_info["status"])  # status seems to be a string of a number
    #                 order_direction = order_info["direction"]
    #                 order_offset = order_info["offset"]

    #                 if (
    #                     order_status != "4"  # Assuming 4 is 'Filled'
    #                     and order_direction == side_map[side]
    #                     and order_offset == offset_map[side]
    #                     and order_status != "6"  # Assuming 6 is 'Cancelled'
    #                     # There's no 'reduceOnly' equivalent in Huobi from the provided data
    #                 ):
    #                     self.exchange.cancel_order(symbol=symbol, id=order_id)
    #                     log.info(f"Cancelling order: {order_id}")
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in cancel_close_huobi(): {e}")

    def cancel_close(self, symbol: str, side: str) -> None:
        try:
            order = self.exchange.fetch_open_orders(symbol)
            if len(order) > 0:
                if "info" in order[0]:
                    order_id = order[0]["info"]["order_id"]
                    order_status = order[0]["info"]["order_status"]
                    order_side = order[0]["info"]["side"]
                    reduce_only = order[0]["info"]["reduce_only"]
                    if (
                        order_status != "Filled"
                        and order_side == "Buy"
                        and side == "long"
                        and order_status != "Cancelled"
                        and reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
                    elif (
                        order_status != "Filled"
                        and order_side == "Sell"
                        and side == "short"
                        and order_status != "Cancelled"
                        and reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.warning(f"{e}")

    # Bybit
    def create_take_profit_order_bybit(self, symbol, order_type, side, amount, price=None, positionIdx=1, reduce_only=True):
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")

            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")

            params = {"reduceOnly": reduce_only}
            return self.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    # Bybit
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

    def create_take_profit_order(self, symbol, order_type, side, amount, price=None, reduce_only=False):
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")

            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")

            params = {"reduceOnly": reduce_only}
            return self.exchange.create_order(symbol, order_type, side, amount, price, params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    # Huobi
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


    def create_market_order(self, symbol: str, side: str, amount: float, params={}, close_position: bool = False) -> None:
        try:
            if side not in ["buy", "sell"]:
                logging.warning(f"side {side} does not exist")
                return

            order_type = "market"

            # Determine the correct order side for closing positions
            if close_position:
                market = self.exchange.market(symbol)
                if market['type'] in ['swap', 'future']:
                    if side == "buy":
                        side = "close_short"
                    elif side == "sell":
                        side = "close_long"

            response = self.exchange.create_order(symbol, order_type, side, amount, params=params)
            return response
        except Exception as e:
            logging.warning(f"An unknown error occurred in create_market_order(): {e}")

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
                logging.warning(f"side {side} does not exist")
        except Exception as e:
            logging.warning(f"An unknown error occurred in create_limit_order(): {e}")

    # Bybit
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
                logging.warning(f"side {side} does not exist")
        except Exception as e:
            logging.warning(f"An unknown error occurred in create_limit_order(): {e}")

    # # Binance
    # def create_take_profit_order_binance(self, symbol, side, amount, price):
    #     if side not in ["buy", "sell"]:
    #         raise ValueError(f"Invalid side: {side}")
        
    #     params={"reduceOnly": True}

    #     # Create the limit order for the take profit
    #     order = self.create_limit_order_binance(symbol, side, amount, price, params)

    #     return order
    
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

    def create_limit_order(self, symbol, side, amount, price, reduce_only=False, **params):
        if side == "buy":
            return self.create_limit_buy_order(symbol, amount, price, reduce_only=reduce_only, **params)
        elif side == "sell":
            return self.create_limit_sell_order(symbol, amount, price, reduce_only=reduce_only, **params)
        else:
            raise ValueError(f"Invalid side: {side}")

    def create_limit_buy_order(self, symbol: str, qty: float, price: float, **params) -> None:
        self.exchange.create_order(
            symbol=symbol,
            type='limit',
            side='buy',
            amount=qty,
            price=price,
            **params
        )

    def create_limit_sell_order(self, symbol: str, qty: float, price: float, **params) -> None:
        self.exchange.create_order(
            symbol=symbol,
            type='limit',
            side='sell',
            amount=qty,
            price=price,
            **params
        )

    def create_order(self, symbol, order_type, side, amount, price=None, reduce_only=False, **params):
        if reduce_only:
            params.update({'reduceOnly': 'true'})

        if self.exchange_id == 'bybit':
            order = self.exchange.create_order(symbol, order_type, side, amount, price, params=params)
        else:
            if order_type == 'limit':
                if side == "buy":
                    order = self.create_limit_buy_order(symbol, amount, price, **params)
                elif side == "sell":
                    order = self.create_limit_sell_order(symbol, amount, price, **params)
                else:
                    raise ValueError(f"Invalid side: {side}")
            elif order_type == 'market':
                #... handle market orders if necessary
                order = self.create_market_order(symbol, side, amount, params)
            else:
                raise ValueError("Invalid order type. Use 'limit' or 'market'.")

        return order

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
                logging.warning(f"Side {side} does not exist")
        except Exception as e:
            logging.warning(f"An unknown error occurred in create_market_order(): {e}")

    def create_contract_order_huobi(self, symbol, order_type, side, amount, price=None, params={}):
        params = {'leverRate': 50}
        return self.exchange.create_contract_order(symbol, order_type, side, amount, price, params)


    # def get_symbol_precision_bybit(self, symbol: str) -> Tuple[int, int]:
    #     try:
    #         market = self.exchange.market(symbol)
    #         price_precision = int(market['precision']['price'])
    #         quantity_precision = int(market['precision']['amount'])
    #         return price_precision, quantity_precision
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return None, None

    # def get_symbol_precision_bybit(self, symbol: str) -> Tuple[int, int]:
    #     market = self.exchange.market(symbol)
    #     price_precision = int(market['precision']['price'])
    #     quantity_precision = int(market['precision']['amount'])
    #     return price_precision, quantity_precision

    # def _get_symbols(self):
    #     markets = self.exchange.load_markets()
    #     symbols = [market['symbol'] for market in markets.values()]
    #     return symbols