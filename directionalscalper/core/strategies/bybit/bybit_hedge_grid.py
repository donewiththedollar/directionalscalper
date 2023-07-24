import time
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
from ..strategy import Strategy
from typing import Tuple
import threading
import os

class BybitHedgeGridStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.last_cancel_time = 0
        self.current_wallet_exposure = 1.0
        self.printed_trade_quantities = False

    # def create_long_grid_entries(self, symbol, amount, price, distance):
    #     # Create long grid entries
    #     entry_prices = [round(price / distance * (1 - i), 8) for i in range(3)]
    #     for entry_price in entry_prices:
    #         print(f"Long Grid Entry Price: {entry_price}")
    #         self.limit_order(symbol, "buy", amount, entry_price, positionIdx=1, reduceOnly=False)

    # def create_short_grid_entries(self, symbol, amount, price, distance):
    #     # Create short grid entries
    #     entry_prices = [round(price / distance * (1 + i), 8) for i in range(3)]
    #     for entry_price in entry_prices:
    #         print(f"Short Grid Entry Price: {entry_price}")
    #         self.limit_order(symbol, "sell", amount, entry_price, positionIdx=2, reduceOnly=False)

    # def create_long_grid_takeprofits(self, symbol, quantity, price, distance):
    #     # Create long grid take profits
    #     take_profit_prices = [round(price + distance * i, 8) for i in range(3)]
    #     for take_profit_price in take_profit_prices:
    #         print(f"Long Grid Take Profit Price: {take_profit_price}")
    #         self.create_take_profit_order(symbol, "sell", quantity, take_profit_price)

    # def create_short_grid_takeprofits(self, symbol, quantity, price, distance):
    #     # Create short grid take profits
    #     take_profit_prices = [round(price - distance * i, 8) for i in range(3)]
    #     for take_profit_price in take_profit_prices:
    #         print(f"Short Grid Take Profit Price: {take_profit_price}")
    #         self.create_take_profit_order(symbol, "buy", quantity, take_profit_price)

    def place_grid_orders(self, symbol, position, amount, best_price, grid_span_percentage, max_orders):
        # Calculate the price increment based on the grid span percentage
        price_increment = best_price * grid_span_percentage / 100

        # Initialize the order price
        order_price = best_price

        # Place the grid orders
        for i in range(max_orders):
            if position.lower() == "long":
                self.limit_order_bybit(symbol, "buy", amount, order_price, positionIdx=1, reduceOnly=False)
            elif position.lower() == "short":
                self.limit_order_bybit(symbol, "sell", amount, order_price, positionIdx=2, reduceOnly=False)
            else:
                print(f"Invalid position type: {position}")
                return

            # Update the order price for the next order
            order_price += price_increment

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
            print(f"Bybit hedge grid strategy running")
            print(f"Min volume: {min_vol}")
            print(f"Min distance: {min_dist}")

            # Get API data
            data = self.manager.get_data()
            one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
            five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
            thirty_minute_distance = self.manager.get_asset_value(symbol, data, "30mSpread")
            four_hour_distance = self.manager.get_asset_value(symbol, data, "4hSpread")
            trend = self.manager.get_asset_value(symbol, data, "Trend")
            print(f"1m Volume: {one_minute_volume}")
            print(f"5m Spread: {five_minute_distance}")
            print(f"30m Spread: {thirty_minute_distance}")
            print(f"4h spread: {four_hour_distance}")
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
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

            position_data = self.exchange.get_positions_bybit(symbol)

            #print(f"Bybit pos data: {position_data}")

            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            print(f"Short pos qty: {short_pos_qty}")
            print(f"Long pos qty: {long_pos_qty}")

            short_upnl = position_data["short"]["upnl"]
            long_upnl = position_data["long"]["upnl"]

            print(f"Short uPNL: {short_upnl}")
            print(f"Long uPNL: {long_upnl}")

            cum_realised_pnl_long = position_data["long"]["cum_realised"]
            cum_realised_pnl_short = position_data["short"]["cum_realised"]

            print(f"Short cum. PNL: {cum_realised_pnl_short}")
            print(f"Long cum. PNL: {cum_realised_pnl_long}")

            short_pos_price = position_data["short"]["price"] if short_pos_qty > 0 else None
            long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

            print(f"Long pos price {long_pos_price}")
            print(f"Short pos price {short_pos_price}")

            # Take profit calc
            short_take_profit = self.calculate_short_take_profit_bybit(short_pos_price, symbol)
            long_take_profit = self.calculate_long_take_profit_bybit(long_pos_price, symbol)


            should_short = best_bid_price > ma_3_high
            should_long = best_bid_price < ma_3_high

            should_add_to_short = False
            should_add_to_long = False
            
            if short_pos_price is not None:
                should_add_to_short = short_pos_price < ma_6_low
                short_tp_distance_percent = ((short_take_profit - best_ask_price) / best_ask_price) * 100
                print(f"Short TP price: {short_take_profit}, TP distance in percent: {short_tp_distance_percent:.2f}%")
                
            if long_pos_price is not None:
                should_add_to_long = long_pos_price > ma_6_low
                long_tp_distance_percent = ((long_take_profit - best_bid_price) / best_bid_price) * 100
                print(f"Long TP price: {long_take_profit}, TP distance in percent: {long_tp_distance_percent:.2f}%")

            print(f"Short condition: {should_short}")
            print(f"Long condition: {should_long}")
            print(f"Add short condition: {should_add_to_short}")
            print(f"Add long condition: {should_add_to_long}")

            grid_span_percentage = 0.5
            max_orders = 5

            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:
                            print(f"Placing initial long grid")
                            # self.create_long_grid_entries(symbol, amount, best_bid_price, four_hour_distance)
                            self.place_grid_orders(symbol, "long", amount, best_bid_price, grid_span_percentage, max_orders)
                            print(f"Placed initial long grid")
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long grid")
                                self.place_grid_orders(symbol, "long", amount, best_bid_price, grid_span_percentage, max_orders)

                        if trend.lower() == "short" and should_short and short_pos_qty == 0:
                            print(f"Placing initial short grid")
                            #self.create_short_grid_entries(symbol, amount, best_ask_price, four_hour_distance)
                            self.place_grid_orders(symbol, "short", amount, best_bid_price, grid_span_percentage, max_orders)
                            print("Placed initial short grid")
                        else:
                            if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty and best_ask_price > short_pos_price:
                                print(f"Placed additional short grid")
                                self.place_grid_orders(symbol, "short", amount, best_bid_price, grid_span_percentage, max_orders)
        
            open_orders = self.exchange.get_open_orders(symbol)

            if long_pos_qty > 0 and long_take_profit is not None:
                existing_long_tps = self.get_open_take_profit_order_quantities(open_orders, "sell")
                total_existing_long_tp_qty = sum(qty for qty, _ in existing_long_tps)
                if not math.isclose(total_existing_long_tp_qty, long_pos_qty):
                    try:
                        for _, existing_long_tp_id in existing_long_tps:
                            self.exchange.cancel_order_by_id(symbol, "sell")  # Corrected side value to "sell"
                            print(f"Long take profit canceled")
                            time.sleep(0.05)

                        #print(f"Debug: Long Position Quantity {long_pos_qty}, Long Take Profit {long_take_profit}")
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", "sell", long_pos_qty, long_take_profit, positionIdx=1, reduce_only=True)
                        print(f"Long take profit set at {long_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing long TP: {e}")

            if short_pos_qty > 0 and short_take_profit is not None:
                existing_short_tps = self.get_open_take_profit_order_quantities(open_orders, "buy")
                total_existing_short_tp_qty = sum(qty for qty, _ in existing_short_tps)
                if not math.isclose(total_existing_short_tp_qty, short_pos_qty):
                    try:
                        for _, existing_short_tp_id in existing_short_tps:
                            self.exchange.cancel_order_by_id(symbol, "buy")  # Corrected side value to "buy"
                            print(f"Short take profit canceled")
                            time.sleep(0.05)

                        #print(f"Debug: Short Position Quantity {short_pos_qty}, Short Take Profit {short_take_profit}")
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", "buy", short_pos_qty, short_take_profit, positionIdx=2, reduce_only=True)
                        print(f"Short take profit set at {short_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing short TP: {e}")

            # Cancel all entries routinely
            self.cancel_entries_bybit(symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high)

            time.sleep(30)