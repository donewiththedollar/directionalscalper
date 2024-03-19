import uuid
from .exchange import Exchange
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from ccxt.base.errors import RateLimitExceeded

class MexcExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('bitget', api_key, secret_key, passphrase, market_type)
    
    def get_balance_mexc(self, quote, market_type='swap'):
        if self.exchange.has['fetchBalance']:
            # Fetch the balance
            balance = self.exchange.fetch_balance(params={"type": market_type})

            # Find the quote balance
            if quote in balance['total']:
                return float(balance['total'][quote])
        return None
    
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