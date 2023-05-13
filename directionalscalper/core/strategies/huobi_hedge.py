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
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        wallet_exposure = self.config.wallet_exposure
        min_order_value = 6
        max_retries = 5
        retry_delay = 5

        while True:
            print(f"Huobi strategy running")

            parsed_symbol = self.parse_symbol(symbol)

            quote = 'USDT'

            for i in range(max_retries):
                try:
                    total_equity = self.exchange.get_balance_huobi_unified(quote, 'swap', 'linear')
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"Error occurred while fetching balance: {e}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise e

            print(f"Current balance: {total_equity}")

            # Orderbook data
            orderbook = self.exchange.get_orderbook(parsed_symbol)
            best_bid_price = orderbook['bids'][0][0]
            best_ask_price = orderbook['asks'][0][0]

            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")

            market_data = self.exchange.get_market_data_huobi(parsed_symbol)

            #print(f"{market_data}")

            price_precision = market_data["precision"]

            #print(f"{price_precision}")

            leverage = market_data["leverage"] if market_data["leverage"] != 0 else 50.0

            #print(f"{leverage}")

            max_trade_qty = round(
                (float(total_equity) * wallet_exposure / float(best_ask_price))
                / (100 / leverage),
                int(float(market_data["min_qty"])),
            )

            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            current_price = self.exchange.get_current_price(parsed_symbol)

            print(f"Current price: {current_price}")

            print(f"Entry size: {amount}")

            min_qty_huobi = float(market_data["min_qty"])

            print(f"Min trade quantitiy for {parsed_symbol}: {min_qty_huobi}")
            print(f"Min volume: {min_vol}")
            print(f"Min distance: {min_dist}")

            # Get data from manager
            data = self.manager.get_data()

            # Data we need from API
            one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
            five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
            trend = self.manager.get_asset_value(symbol, data, "Trend")
            print(f"1m Volume: {one_minute_volume}")
            print(f"5m Spread: {five_minute_distance}")
            print(f"Trend: {trend}")

            print(f"Parsed symbol: {parsed_symbol}")
            print(f"Regular symbol: {symbol}")

            position_data = self.exchange.get_positions_huobi("DOGE")

            print(f"{position_data}")

            self.exchange.get_positions_debug()

            time.sleep(30)
            