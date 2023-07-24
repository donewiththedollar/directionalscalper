import time
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
from ..strategy import Strategy
from typing import Tuple
import threading
import os
import logging
from ..logger import Logger

logging = Logger(filename="bybitlongonly.log", stream=True)

class BybitLongStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.last_cancel_time = 0
        self.current_wallet_exposure = 1.0
        self.printed_trade_quantities = False

    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        max_leverage = self.exchange.get_max_leverage_bybit(symbol)
        retry_delay = 5
        max_retries = 5

        print("Setting up exchange")
        self.exchange.setup_exchange_bybit(symbol)

        print("Setting leverage")
        if current_leverage != max_leverage:
            print(f"Current leverage is not at maximum. Setting leverage to maximum. Maximum is {max_leverage}")
            self.exchange.set_leverage_bybit(max_leverage, symbol)

        # # Create the strategy table
        # strategy_table = create_strategy_table(symbol, total_equity, long_upnl, short_upnl, short_pos

        while True:
            print(f"Bybit long only strategy running")
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

            price_precision = int(self.exchange.get_price_precision(symbol))

            print(f"Precision: {price_precision}")

            quote_currency = "USDT"

            for i in range(max_retries):
                try:
                    total_equity = self.exchange.get_balance_bybit(quote_currency)
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"Error occurred while fetching balance: {e}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise e
                    
            print(f"Total equity: {total_equity}")

            current_price = self.exchange.get_current_price(symbol)
            market_data = self.get_market_data_with_retry(symbol, max_retries = 5, retry_delay = 5)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")
            print(f"Current price: {current_price}")

            max_trade_qty = self.calc_max_trade_qty(total_equity,
                                                     best_ask_price,
                                                     max_leverage)
            
            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            min_qty_bybit = market_data["min_qty"]
            print(f"Min qty: {min_qty_bybit}")

            self.check_amount_validity_bybit(amount, symbol)

            self.print_trade_quantities_once_bybit(max_trade_qty)

            # Get the 1-minute moving averages
            print(f"Fetching MA data")
            m_moving_averages = self.manager.get_1m_moving_averages(symbol)
            m5_moving_averages = self.manager.get_5m_moving_averages(symbol)
            ma_6_high = m_moving_averages["MA_6_H"]
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

            position_data = self.exchange.get_positions_bybit(symbol)

            #print(f"Bybit pos data: {position_data}")

            long_pos_qty = position_data["long"]["qty"]

            print(f"Long pos qty: {long_pos_qty}")

            long_upnl = position_data["long"]["upnl"]

            print(f"Long uPNL: {long_upnl}")

            cum_realised_pnl_long = position_data["long"]["cum_realised"]

            print(f"Long cum. PNL: {cum_realised_pnl_long}")

            long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

            print(f"Long pos price {long_pos_price}")

            # Take profit calc
            long_take_profit = self.calculate_long_take_profit_bybit(long_pos_price, symbol)

            should_long = self.long_trade_condition(best_bid_price, ma_3_low)

            should_add_to_short = False
            should_add_to_long = False
            
            if long_pos_price is not None:
                should_add_to_long = long_pos_price > ma_6_high
                long_tp_distance_percent = ((long_take_profit - best_bid_price) / best_bid_price) * 100
                print(f"Long TP price: {long_take_profit}, TP distance in percent: {long_tp_distance_percent:.2f}%")

            print(f"Long condition: {should_long}")
            print(f"Add long condition: {should_add_to_long}")

            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:
                            print(f"Placing initial long entry")
                            self.limit_order_bybit(symbol, "buy", amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            print(f"Placed initial long entry")
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long entry")
                                self.limit_order_bybit(symbol, "buy", amount, best_bid_price, positionIdx=1, reduceOnly=False)

            open_orders = self.exchange.get_open_orders(symbol)

            if long_pos_qty > 0 and long_take_profit is not None:
                existing_long_tps = self.get_open_take_profit_order_quantities(open_orders, "sell")
                total_existing_long_tp_qty = sum(qty for qty, _ in existing_long_tps)
                if not math.isclose(total_existing_long_tp_qty, long_pos_qty):
                    try:
                        for _, existing_long_tp_id in existing_long_tps:
                            self.exchange.cancel_take_profit_orders_bybit(symbol, "sell")  # Corrected side value to "sell"
                            print(f"Long take profit canceled")
                            time.sleep(0.05)

                        #print(f"Debug: Long Position Quantity {long_pos_qty}, Long Take Profit {long_take_profit}")
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", "sell", long_pos_qty, long_take_profit, positionIdx=1, reduce_only=True)
                        print(f"Long take profit set at {long_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing long TP: {e}")

            # Cancel all entries routinely
            self.cancel_entries_bybit(symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high)

            time.sleep(30)