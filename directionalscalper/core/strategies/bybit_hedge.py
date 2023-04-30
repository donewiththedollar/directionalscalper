import time
from .strategy import Strategy

class BybitHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config)
        self.manager = manager

    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure

        while True:
            print(f"Bybit hedge strategy running")
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume
            print(f"Min volume: {min_vol}")
            print(f"Min distance: {min_dist}")

            # Get API data
            data = self.manager.get_data()
            one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
            five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
            trend = self.manager.get_asset_value(symbol, data, "Trend")
            print(f"1m Volume: {one_minute_volume}")
            print(f"5m Spread: {five_minute_distance}")
            print(f"Trend: {trend}")

            quote_currency = "USDT"
            dex_equity = self.exchange.get_balance_bybit(quote_currency)

            print(f"Total equity: {dex_equity}")
            
            market_data = self.exchange.get_market_data_bybit(symbol)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")

            leverage = float(market_data["leverage"]) if market_data["leverage"] !=0 else 50.0

            max_trade_qty = round(
                (float(dex_equity) * wallet_exposure / float(best_ask_price))
                / (100 / leverage),
                int(float(market_data["min_qty"])),
            )            
            
            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            min_qty_bybit = market_data["min_qty"]
            print(f"Min qty: {min_qty_bybit}")

            if float(amount) < min_qty_bybit:
                print(f"The amount you entered ({amount}) is less than the minimum required by Bitget for {symbol}: {min_qty_bybit}.")
                break
            else:
                print(f"The amount you entered ({amount}) is valid for {symbol}")

            # Get the 1-minute moving averages
            print(f"Fetching MA data")
            m_moving_averages = self.manager.get_1m_moving_averages(symbol)
            m5_moving_averages = self.manager.get_5m_moving_averages(symbol)

            # Define MAs for ease of use
            ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

            print(f"1m MAs: {m_moving_averages}")
            print(f"5m MAs: {m5_moving_averages}")
            print(f"1m MA3 HIGH: {ma_1m_3_high}")
            print(f"5m MA3 HIGH: {ma_5m_3_high}")

            # Hedge logic starts here
            # if trend is not None and isinstance(trend, str):
            #     if one_minute_volume is not None and five_minute_distance is not None:
            #         if one_minute_volume > min_vol and five_minute_distance > min_dist:

            #             if trend.lower() == "long" and should_long and long_pos_qty == 0:

            #                 #self.limit_order(symbol, "buy", amount, bid_price, reduce_only=False)
            #                 print(f"Placed initial long entry")
            #             else:
            #                 if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty:
            #                     print(f"Placed additional long entry")
            #                     self.limit_order(symbol, "buy", amount, bid_price, reduce_only=False)

            #             if trend.lower() == "short" and should_short and short_pos_qty == 0:

            #                 #self.limit_order(symbol, "sell", amount, ask_price, reduce_only=False)
            #                 print("Placed initial short entry")
            #             else:
            #                 if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty:
            #                     print(f"Placed additional short entry")
            #                     self.limit_order(symbol, "sell", amount, ask_price, reduce_only=False)

            time.sleep(30)
            




