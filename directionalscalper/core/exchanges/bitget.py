import uuid
from .exchange import Exchange
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from ccxt.base.errors import RateLimitExceeded
import traceback

class BitgetExchange(Exchange):
    def __init__(self, api_key: str, secret_key: str, passphrase: Optional[str] = None, market_type: str = 'swap'):
        """
        Initialize the Bitget exchange.
        
        Args:
            api_key (str): API key for Bitget.
            secret_key (str): Secret key for Bitget.
            passphrase (Optional[str]): Passphrase for Bitget API, if required.
            market_type (str): Market type ('swap' or 'spot'). Defaults to 'swap'.
        """
        super().__init__('bitget', api_key, secret_key, passphrase, market_type)
    
    def get_balance_bitget(self, quote: str, account_type: str = 'futures') -> float:
        """
        Fetch the balance for a specific quote currency in the given account type.
        
        Args:
            quote (str): The currency to fetch balance for (e.g., 'USDT').
            account_type (str): Account type ('futures' or 'spot'). Defaults to 'futures'.
        
        Returns:
            float: The total balance for the quote currency, or 0.0 if not found or on error.
        
        Raises:
            ValueError: If quote or account_type is invalid.
        """
        if not quote:
            raise ValueError("Quote currency cannot be empty")
        if account_type not in ['futures', 'spot']:
            raise ValueError("Account type must be 'futures' or 'spot'")
        
        try:
            with self.rate_limiter:
                params = {'type': 'swap'} if account_type == 'futures' else {'type': 'spot'}
                balance = self.exchange.fetch_balance(params=params)
                if quote in balance:
                    return float(balance[quote].get('total', 0.0))
                logging.warning(f"No balance found for {quote} in {account_type} account")
                return 0.0
        except Exception as e:
            logging.error(f"Error fetching Bitget balance for {quote}: {e}")
            logging.error(traceback.format_exc())
            return 0.0

    def get_open_orders_bitget(self, symbol: str) -> List[Dict]:
        """
        Fetch open orders for a given symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        
        Returns:
            List[Dict]: List of open order dictionaries with id, price, qty, side, and reduce_only.
        
        Raises:
            ValueError: If symbol is empty or invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        try:
            with self.rate_limiter:
                orders = self.exchange.fetch_open_orders(symbol)
                open_orders = []
                for order in orders:
                    order_data = {
                        "id": self.exchange.safe_string(order, "id", ""),
                        "price": self.exchange.safe_float(order, "price", 0.0),
                        "qty": self.exchange.safe_float(order, "amount", 0.0),
                        "side": self.exchange.safe_string(order, "side", ""),
                        "reduce_only": self.exchange.safe_value(order, "reduceOnly", False)
                    }
                    open_orders.append(order_data)
                return open_orders
        except Exception as e:
            logging.error(f"Error fetching open orders for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return []

    def get_max_leverage_bitget(self, symbol: str) -> Optional[int]:
        """
        Fetch the maximum leverage for a given symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        
        Returns:
            Optional[int]: The maximum leverage, or None if not found or on error.
        
        Raises:
            ValueError: If symbol is empty or invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        try:
            with self.rate_limiter:
                leverage_tiers = self.exchange.fetch_market_leverage_tiers(symbol)
                if not leverage_tiers:
                    logging.warning(f"No leverage tiers found for {symbol}")
                    return None
                max_leverage = max(self.exchange.safe_integer(tier, 'maxLeverage', 1) for tier in leverage_tiers)
                return max_leverage
        except Exception as e:
            logging.error(f"Error fetching max leverage for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return None

    def cancel_all_entries_bitget(self, symbol: str) -> bool:
        """
        Cancel all non-reduce-only open orders for a symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        
        Returns:
            bool: True if cancellation was successful, False otherwise.
        
        Raises:
            ValueError: If symbol is empty or invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        try:
            with self.rate_limiter:
                orders = self.exchange.fetch_open_orders(symbol)
                for order in orders:
                    if not self.exchange.safe_value(order, 'reduceOnly', False) and order['status'] == 'open':
                        self.exchange.cancel_order(order['id'], symbol)
                        logging.info(f"Cancelled order: {order['id']}")
                return True
        except Exception as e:
            logging.error(f"Error cancelling entries for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return False

    def cancel_entry_bitget(self, symbol: str) -> bool:
        """
        Cancel non-filled, non-reduce-only orders for a symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        
        Returns:
            bool: True if cancellation was successful, False otherwise.
        
        Raises:
            ValueError: If symbol is empty or invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        try:
            with self.rate_limiter:
                orders = self.exchange.fetch_open_orders(symbol)
                for order in orders:
                    order_info = order.get("info", {})
                    order_id = self.exchange.safe_string(order_info, "orderId")
                    order_status = self.exchange.safe_string(order_info, "state")
                    reduce_only = self.exchange.safe_value(order_info, "reduceOnly", False)
                    
                    if order_status not in ["filled", "cancelled"] and not reduce_only:
                        self.exchange.cancel_order(order_id, symbol)
                        logging.info(f"Cancelled order: {order_id}")
                return True
        except Exception as e:
            logging.error(f"Error cancelling entry orders for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return False

    def get_open_take_profit_order_quantity_bitget(self, symbol: str, side: str) -> Optional[float]:
        """
        Fetch the total quantity of open take-profit orders for a symbol and side.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            side (str): Order side ('buy' or 'sell').
        
        Returns:
            Optional[float]: Total quantity of open take-profit orders, or None if none exist.
        
        Raises:
            ValueError: If symbol or side is invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        try:
            with self.rate_limiter:
                orders = self.exchange.fetch_open_orders(symbol)
                total_quantity = 0.0
                for order in orders:
                    if (self.exchange.safe_string(order, 'side') == side and 
                        self.exchange.safe_value(order, 'reduceOnly', False) and 
                        order['status'] == 'open'):
                        total_quantity += self.exchange.safe_float(order, 'amount', 0.0)
                return total_quantity if total_quantity > 0 else None
        except Exception as e:
            logging.error(f"Error fetching TP order quantity for {symbol}, side {side}: {e}")
            logging.error(traceback.format_exc())
            return None

    def get_order_status_bitget(self, symbol: str, side: str) -> Optional[str]:
        """
        Fetch the status of the most recent open order for a symbol and side.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            side (str): Order side ('buy' or 'sell').
        
        Returns:
            Optional[str]: Status of the most recent open order, or None if none exist.
        
        Raises:
            ValueError: If symbol or side is invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        
        try:
            with self.rate_limiter:
                orders = self.exchange.fetch_open_orders(symbol)
                for order in orders:
                    if self.exchange.safe_string(order, 'side') == side:
                        return self.exchange.safe_string(order, 'status')
                return None
        except Exception as e:
            logging.error(f"Error fetching order status for {symbol}, side {side}: {e}")
            logging.error(traceback.format_exc())
            return None

    def cancel_close_bitget(self, symbol: str, side: str) -> bool:
        """
        Cancel reduce-only orders for closing positions for a symbol and side.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            side (str): Position side ('long' or 'short').
        
        Returns:
            bool: True if cancellation was successful, False otherwise.
        
        Raises:
            ValueError: If symbol or side is invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if side not in ['long', 'short']:
            raise ValueError("Side must be 'long' or 'short'")
        
        close_side = 'sell' if side == 'long' else 'buy'  # Closing a long is sell, short is buy
        try:
            with self.rate_limiter:
                orders = self.exchange.fetch_open_orders(symbol)
                for order in orders:
                    order_info = order.get("info", {})
                    order_id = self.exchange.safe_string(order_info, "orderId")
                    order_status = self.exchange.safe_string(order_info, "state")
                    order_side = self.exchange.safe_string(order_info, "side")
                    reduce_only = self.exchange.safe_value(order_info, "reduceOnly", False)
                    
                    if (order_status not in ['filled', 'cancelled'] and 
                        reduce_only and 
                        order_side == close_side):
                        self.exchange.cancel_order(order_id, symbol)
                        logging.info(f"Cancelled order: {order_id}")
                return True
        except Exception as e:
            logging.error(f"Error cancelling close orders for {symbol}, side {side}: {e}")
            logging.error(traceback.format_exc())
            return False

    def market_close_position_bitget(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Close a position by creating a market order in the opposite direction.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            side (str): Original position side ('buy' for long, 'sell' for short).
            amount (float): The quantity to close.
        
        Returns:
            Dict: The created order details.
        
        Raises:
            ValueError: If symbol, side, or amount is invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        close_side = 'sell' if side == 'buy' else 'buy'
        try:
            with self.rate_limiter:
                order = self.exchange.create_order(
                    symbol, 'market', close_side, amount, params={'reduceOnly': True}
                )
                logging.info(f"Closed position: {symbol}, side: {side}, amount: {amount}")
                return order
        except Exception as e:
            logging.error(f"Error closing position for {symbol}, side {side}: {e}")
            logging.error(traceback.format_exc())
            raise

    def get_current_candle_bitget(self, symbol: str, timeframe: str = '1m', retries: int = 3, delay: int = 60) -> List:
        """
        Fetch the current candlestick for a symbol and timeframe.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            timeframe (str): The candlestick timeframe (e.g., '1m', '5m'). Defaults to '1m'.
            retries (int): Number of retries for rate limit errors. Defaults to 3.
            delay (int): Delay in seconds between retries. Defaults to 60.
        
        Returns:
            List: Current candle [timestamp, open, high, low, close, volume].
        
        Raises:
            ValueError: If symbol or timeframe is invalid.
            RateLimitExceeded: If retries are exhausted.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if not timeframe:
            raise ValueError("Timeframe cannot be empty")
        
        for attempt in range(retries):
            try:
                with self.rate_limiter:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=2)
                    return ohlcv[-1]
            except RateLimitExceeded:
                if attempt < retries - 1:
                    logging.info(f"Rate limit exceeded, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise RateLimitExceeded(f"Failed to fetch candle data for {symbol} after {retries} retries")
            except Exception as e:
                logging.error(f"Error fetching candle for {symbol}: {e}")
                logging.error(traceback.format_exc())
                raise

    def set_leverage_bitget(self, symbol: str, leverage: float, params: Dict = {}) -> Optional[Dict]:
        """
        Set the leverage for a market.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            leverage (float): The leverage value (e.g., 10.0).
            params (Dict): Extra parameters for the Bitget API.
        
        Returns:
            Optional[Dict]: Response from the exchange, or None if unsupported.
        
        Raises:
            ValueError: If symbol or leverage is invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if leverage <= 0:
            raise ValueError("Leverage must be positive")
        
        try:
            with self.rate_limiter:
                if hasattr(self.exchange, 'set_leverage'):
                    return self.exchange.set_leverage(leverage, symbol, params)
                logging.warning(f"{self.exchange_id} does not support setting leverage")
                return None
        except Exception as e:
            logging.error(f"Error setting leverage for {symbol}: {e}")
            logging.error(traceback.format_exc())
            return None

    def get_market_data_bitget(self, symbol: str) -> Dict:
        """
        Fetch market data for a symbol, including precision, leverage, and minimum quantity.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        
        Returns:
            Dict: Dictionary with precision, leverage, and min_qty.
        
        Raises:
            ValueError: If symbol is empty or invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            with self.rate_limiter:
                self.exchange.load_markets()
                symbol_data = self.exchange.market(symbol)
                values["precision"] = self.exchange.safe_float(symbol_data.get("precision", {}), "price", 0.0)
                values["min_qty"] = self.exchange.safe_float(symbol_data.get("limits", {}).get("amount", {}), "min", 0.0)
                leverage_tiers = self.exchange.fetch_market_leverage_tiers(symbol)
                if leverage_tiers:
                    values["leverage"] = max(self.exchange.safe_integer(tier, 'maxLeverage', 1) for tier in leverage_tiers)
        except Exception as e:
            logging.error(f"Error fetching market data for {symbol}: {e}")
            logging.error(traceback.format_exc())
        return values

    def get_positions_bitget(self, symbol: str) -> Dict:
        """
        Fetch long and short position details for a symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
        
        Returns:
            Dict: Dictionary with long and short position details.
        
        Raises:
            ValueError: If symbol is empty or invalid.
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        values = {
            "long": {
                "qty": 0.0, "price": 0.0, "realised": 0.0, "cum_realised": 0.0,
                "upnl": 0.0, "upnl_pct": 0.0, "liq_price": 0.0, "entry_price": 0.0
            },
            "short": {
                "qty": 0.0, "price": 0.0, "realised": 0.0, "cum_realised": 0.0,
                "upnl": 0.0, "upnl_pct": 0.0, "liq_price": 0.0, "entry_price": 0.0
            }
        }
        try:
            with self.rate_limiter:
                positions = self.exchange.fetch_positions([symbol])
                for position in positions:
                    side = self.exchange.safe_string(position, "side")
                    if side in values:
                        values[side]["qty"] = self.exchange.safe_float(position, "contracts", 0.0)
                        values[side]["price"] = self.exchange.safe_float(position, "entryPrice", 0.0)
                        values[side]["realised"] = round(self.exchange.safe_float(position.get("info", {}), "achievedProfits", 0.0), 4)
                        values[side]["upnl"] = round(self.exchange.safe_float(position, "unrealizedPnl", 0.0), 4)
                        values[side]["liq_price"] = self.exchange.safe_float(position, "liquidationPrice", 0.0)
                        values[side]["entry_price"] = self.exchange.safe_float(position, "entryPrice", 0.0)
        except Exception as e:
            logging.error(f"Error fetching positions for {symbol}: {e}")
            logging.error(traceback.format_exc())
        return values