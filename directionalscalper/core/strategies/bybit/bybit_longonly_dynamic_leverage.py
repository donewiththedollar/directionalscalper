import time
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
from ..strategy import Strategy
from typing import Tuple
import logging
from ..logger import Logger

logging = Logger(filename="bybithedgedynamicleverage.log", stream=True)

class BybitLongOnlyDynamicLeverage(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.last_cancel_time = 0
        self.current_wallet_exposure = 1.0
        self.printed_trade_quantities = False
        self.checked_amount_validity = False
        self.max_long_trade_qty = None
        self.max_short_trade_qty = None
        self.initial_max_long_trade_qty = None
        self.initial_max_short_trade_qty = None
        self.long_leverage_increased = False

    def run(self, symbol):
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

        previous_five_minute_distance = None
        previous_thirty_minute_distance = None
        previous_one_hour_distance = None
        previous_four_hour_distance = None

        while True:
            print(f"Bybit longonly dynamic leverage strategy running")
            print(f"Min volume: {min_vol}")
            print(f"Min distance: {min_dist}")

            # Get API data
            data = self.manager.get_data()
            one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
            one_minute_distance = self.manager.get_asset_value(symbol, data, "1mSpread")
            five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
            thirty_minute_distance = self.manager.get_asset_value(symbol, data, "30mSpread")
            one_hour_distance = self.manager.get_asset_value(symbol, data, "1hSpread")
            four_hour_distance = self.manager.get_asset_value(symbol, data, "4hSpread")
            trend = self.manager.get_asset_value(symbol, data, "Trend")
            print(f"1m Volume: {one_minute_volume}")
            print(f"1m Spread: {one_minute_distance}")
            print(f"5m Spread: {five_minute_distance}")
            print(f"30m Spread: {thirty_minute_distance}")
            print(f"1h Spread: {one_hour_distance}")
            print(f"4h Spread: {four_hour_distance}")
            print(f"Trend: {trend}")

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

            for i in range(max_retries):
                try:
                    available_equity = self.exchange.get_available_balance_bybit(quote_currency)
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"Error occurred while fetching available balance: {e}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise e
            
            print(f"Available equity: {available_equity}")

            current_price = self.exchange.get_current_price(symbol)
            market_data = self.get_market_data_with_retry(symbol, max_retries = 5, retry_delay = 5)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")
            print(f"Current price: {current_price}")

            if self.max_long_trade_qty is None:
                self.max_long_trade_qty = self.calc_max_trade_qty(total_equity,
                                                                                             best_ask_price,
                                                                                             max_leverage)

                # Set initial quantities if they're None
                if self.initial_max_long_trade_qty is None:
                    self.initial_max_long_trade_qty = self.max_long_trade_qty
                    print(f"Initial max trade qty set to {self.initial_max_long_trade_qty}")

            print(f"Max long trade quantity for {symbol}: {self.max_long_trade_qty}")
            print(f"Max short trade quantity for {symbol}: {self.max_short_trade_qty}")
            print(f"Initial long trade qty locked: {self.initial_max_long_trade_qty}")
            
            max_trade_qty = self.calc_max_trade_qty(total_equity,
                                                     best_ask_price,
                                                     max_leverage)   
            
            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            # Calculate the dynamic amount
            long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty

            min_qty = float(market_data["min_qty"])
            min_qty_str = str(min_qty)

            # Get the precision level of the minimum quantity
            if ".0" in min_qty_str:
                # The minimum quantity does not have a fractional part, precision is 0
                precision_level = 0
            else:
                # The minimum quantity has a fractional part, get its precision level
                precision_level = len(min_qty_str.split(".")[1])

            # Round the amount to the precision level of the minimum quantity
            long_dynamic_amount = round(long_dynamic_amount, precision_level)

            print(f"Long dynamic amount: {long_dynamic_amount}")

            self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)

            # Check if the amount is less than the minimum quantity allowed by the exchange
            if long_dynamic_amount < min_qty:
                print(f"Dynamic amount too small for 0.001x, using min_qty")
                long_dynamic_amount = min_qty

            print(f"Min qty: {min_qty}")

            self.check_amount_validity_bybit(long_dynamic_amount, symbol)

            self.print_trade_quantities_once_bybit(self.max_long_trade_qty)

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

            if long_pos_qty >= self.max_long_trade_qty and self.long_pos_leverage <= 1.0:
                self.max_long_trade_qty *= 2  # double the maximum long trade quantity
                self.long_leverage_increased = True
                self.long_pos_leverage = 2.0
                logging.info(f"Long leverage temporarily increased to {self.long_pos_leverage}x")
            elif long_pos_qty < (self.max_long_trade_qty / 2) and self.long_pos_leverage > 1.0:
                self.max_long_trade_qty = self.calc_max_trade_qty(total_equity,
                                                                best_ask_price,
                                                                max_leverage)
                self.long_leverage_increased = False
                self.long_pos_leverage = 1.0
                logging.info(f"Long leverage returned to normal {self.long_pos_leverage}x")

            if self.long_leverage_increased:
                print(f"Long position currently increased to 2x")
            else:
                print(f"Long position currently at normal leverage 1x")
            
            long_upnl = position_data["long"]["upnl"]

            print(f"Long uPNL: {long_upnl}")

            cum_realised_pnl_long = position_data["long"]["cum_realised"]

            print(f"Long cum. PNL: {cum_realised_pnl_long}")

            long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

            print(f"Long pos price {long_pos_price}")

            long_take_profit = None

            if five_minute_distance != previous_five_minute_distance:
                long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, five_minute_distance)
            else:
                if long_take_profit is None:
                    long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, five_minute_distance)
                    
            previous_five_minute_distance = five_minute_distance

            print(f"Long TP: {long_take_profit}")

            should_long = self.long_trade_condition(best_bid_price, ma_3_low)

            should_add_to_long = False
             
            if long_pos_price is not None:
                should_add_to_long = long_pos_price > ma_6_high
                long_tp_distance_percent = ((long_take_profit - long_pos_price) / long_pos_price) * 100
                long_expected_profit_usdt = long_tp_distance_percent / 100 * long_pos_price * long_pos_qty
                print(f"Long TP price: {long_take_profit}, TP distance in percent: {long_tp_distance_percent:.2f}%, Expected profit: {long_expected_profit_usdt:.2f} USDT")


            print(f"Long condition: {should_long}")
            print(f"Add long condition: {should_add_to_long}")

            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:
                            print(f"Placing initial long entry")
                            self.limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            print(f"Placed initial long entry")
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long entry")
                                self.limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1, reduceOnly=False)

            open_orders = self.exchange.get_open_orders(symbol)

            if long_pos_qty > 0 and long_take_profit is not None:
                existing_long_tps = self.get_open_take_profit_order_quantities(open_orders, "sell")
                total_existing_long_tp_qty = sum(qty for qty, _ in existing_long_tps)
                print(f"Existing long TPs: {existing_long_tps}")
                if not math.isclose(total_existing_long_tp_qty, long_pos_qty):
                    try:
                        for qty, existing_long_tp_id in existing_long_tps:
                            if not math.isclose(qty, long_pos_qty):
                                self.exchange.cancel_order_by_id(existing_long_tp_id, symbol)
                                print(f"Long take profit {existing_long_tp_id} canceled")
                                time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in cancelling long TP orders {e}")

                if len(existing_long_tps) < 1:
                    try:
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", "sell", long_pos_qty, long_take_profit, positionIdx=1, reduce_only=True)
                        print(f"Long take profit set at {long_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing long TP: {e}")

            # Cancel entries
            current_time = time.time()
            if current_time - self.last_cancel_time >= 60:  # Execute this block every 1 minute
                try:
                    if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                        self.exchange.cancel_all_entries_bybit(symbol)
                        print(f"Canceled entry orders for {symbol}")
                        time.sleep(0.05)
                except Exception as e:
                    print(f"An error occurred while canceling entry orders: {e}")

                self.last_cancel_time = current_time  # Update the last cancel time

            time.sleep(30)