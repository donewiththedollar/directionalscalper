import time
from ..strategy import Strategy

class MEXCHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager

    def run(self, symbol, amount):
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        wallet_exposure = self.config.wallet_exposure

        while True:
            print(f"MEXC strategy running")
            time.sleep(0.05)

            quote_currency = "USDT"
            total_equity = self.exchange.get_balance_mexc(quote_currency)

            print(f"Total equity: {total_equity}")
            
            # Orderbook data
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = orderbook['bids'][0][0]
            best_ask_price = orderbook['asks'][0][0]

            print(f"Bid: {best_bid_price}")
            print(f"Ask: {best_ask_price}")

            market_data = self.exchange.get_market_data_mexc(symbol)

            print(f"Market data: {market_data}")

            leverage = (
                float(market_data["leverage"])
                if market_data["leverage"] is not None and market_data["leverage"] != 0
                else 50.0
            )

            max_trade_qty = round(
                (float(total_equity) * wallet_exposure / float(best_ask_price))
                / (100 / leverage),
                int(float(market_data["min_qty"])),
            )

            print(f"Max trade quantity for {symbol}: {max_trade_qty}")


            time.sleep(30)
            