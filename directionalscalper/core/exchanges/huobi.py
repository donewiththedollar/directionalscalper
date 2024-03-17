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
        
