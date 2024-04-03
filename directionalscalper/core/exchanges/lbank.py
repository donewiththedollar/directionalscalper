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
        super().__init__('binance', api_key, secret_key, passphrase, market_type)