import time
from .strategy import Strategy

class HuobiHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config)
        self.manager = manager

    def parse_symbol(self, symbol):
        if "huobi" in self.exchange.name.lower():
            base_currency = symbol[:-4]  # Extracts everything except last 4 characters
            quote_currency = symbol[-4:]  # Extracts last 4 characters
            return f"{base_currency}/{quote_currency}"
        return symbol


    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure

        while True:
            print(f"Huobi strategy running")

            parsed_symbol = self.parse_symbol(symbol)

            quote = 'USDT'

            total_equity = self.exchange.get_balance_huobi_unified(quote, 'swap', 'linear')

            print(f"Huobi balance: {total_equity}")

            # Orderbook data
            orderbook = self.exchange.get_orderbook(parsed_symbol)
            best_bid_price = orderbook['bids'][0][0]
            best_ask_price = orderbook['asks'][0][0]

            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")

            time.sleep(30)
            