import time
from .strategy import Strategy

class BybitHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config)
        self.manager = manager

    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure

        while True:
            print(f"Bybit strategy running")

            quote_currency = "USDT"
            dex_equity = self.exchange.get_balance_bybit(quote_currency)

            print(f"Total equity: {dex_equity}")
            
            market_data = self.exchange.get_market_data_bybit(symbol)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]

            leverage = float(market_data["leverage"]) if market_data["leverage"] !=0 else 50.0

            max_trade_qty = round(
                (float(dex_equity) * wallet_exposure / float(best_ask_price))
                / (100 / leverage),
                int(float(market_data["min_qty"])),
            )            
            
            print(f"Max trade quantity for {symbol}: {max_trade_qty}")



            time.sleep(30)
            




