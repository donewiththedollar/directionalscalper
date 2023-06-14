import time
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
from ..strategy import Strategy
from typing import Tuple
#from ...tables import create_strategy_table, start_live_table
#from directionalscalper.core.tables import create_strategy_table, start_live_table
import threading
import os

class BybitVolatilityHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.last_cancel_time = 0
        self.wallet_exposure_limit = self.config.wallet_exposure_limit
        self.current_wallet_exposure = 1.0
        self.printed_trade_quantities = False

    # def format_symbol(self, symbol):
    #     base = symbol[:-4]  # Get all characters in symbol except last 4 ('USDT')
    #     return f"{base}/USD"

    def format_symbol(self, symbol):
        """
        Format the given symbol string to include a '/' between the base and quote currencies.
        The function handles base currencies of 3 to 4 characters and quote currencies of 3 to 4 characters.
        """
        quote_currencies = ["USDT", "USD", "BTC", "ETH"]
        for quote in quote_currencies:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return base + '/' + quote
        return None

    def calculate_trade_quantity(self, symbol, leverage):
        dex_equity = self.exchange.get_balance_bybit('USDT')
        trade_qty = (float(dex_equity) * self.current_wallet_exposure) / leverage
        return trade_qty

    def adjust_position_wallet_exposure(self, symbol):
        if self.current_wallet_exposure > self.wallet_exposure_limit:
            desired_wallet_exposure = self.wallet_exposure_limit
            # Calculate the necessary position size to achieve the desired wallet exposure
            max_trade_qty = self.calculate_trade_quantity(symbol, 1)
            current_trade_qty = self.calculate_trade_quantity(symbol, 1 / self.current_wallet_exposure)
            reduction_qty = current_trade_qty - max_trade_qty
            # Reduce the position to the desired wallet exposure level
            self.exchange.reduce_position_bybit(symbol, reduction_qty)

    def truncate(self, number: float, precision: int) -> float:
        return float(Decimal(number).quantize(Decimal('0.' + '0'*precision), rounding=ROUND_DOWN))

    def limit_order(self, symbol, side, amount, price, positionIdx, reduceOnly=False):
        params = {"reduceOnly": reduceOnly}
        #print(f"Symbol: {symbol}, Side: {side}, Amount: {amount}, Price: {price}, Params: {params}")
        order = self.exchange.create_limit_order_bybit(symbol, side, amount, price, positionIdx=positionIdx, params=params)
        return order

    def get_open_take_profit_order_quantity(self, orders, side):
        for order in orders:
            if order['side'].lower() == side.lower() and order['reduce_only']:
                return order['qty'], order['id']
        return None, None

    def get_open_take_profit_order_quantities(self, orders, side):
        take_profit_orders = []
        for order in orders:
            if order['side'].lower() == side.lower() and order['reduce_only']:
                take_profit_orders.append((order['qty'], order['id']))
        return take_profit_orders

    def cancel_take_profit_orders(self, symbol, side):
        self.exchange.cancel_close_bybit(symbol, side)

    def calculate_short_take_profit(self, short_pos_price, symbol):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        #print("Debug: Price Precision for Symbol (", symbol, "):", price_precision)

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating short_target_price. short_pos_price={short_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            try:
                short_target_price = short_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing short_target_price. short_target_price={short_target_price}, price_precision={price_precision}")
                return None

            #print("Debug: Short Target Price:", short_target_price)

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None

    def calculate_long_take_profit(self, long_pos_price, symbol):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        #print("Debug: Price Precision for Symbol (", symbol, "):", price_precision)

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            try:
                long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
            except InvalidOperation as e:
                print(f"Error: Invalid operation when calculating long_target_price. long_pos_price={long_pos_price}, ma_6_high={ma_6_high}, ma_6_low={ma_6_low}")
                return None

            try:
                long_target_price = long_target_price.quantize(
                    Decimal('1e-{}'.format(price_precision)),
                    rounding=ROUND_HALF_UP
                )
            except InvalidOperation as e:
                print(f"Error: Invalid operation when quantizing long_target_price. long_target_price={long_target_price}, price_precision={price_precision}")
                return None

            print("Debug: Long Target Price:", long_target_price)

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def calculate_quantity(self, pos_price, take_profit, best_ask_price, amount, symbol, four_hour_distance, thirty_minute_distance):
        # Fetch the precision for the symbol
        precision_info = self.exchange.get_market_precision_bybit(symbol)
        quantity_decimal_places = precision_info['amount']

        # Calculate the ratio between the position price and the sum of four-hour distance and thirty-minute distance
        combined_distance = float(four_hour_distance or 0) + float(thirty_minute_distance or 0)
        ratio = float(pos_price) / combined_distance if combined_distance else 1.0
        # Scale the quantity inversely with the ratio. The larger the distance, the smaller the quantity.
        scale_factor = 1 / ratio
        qty = abs(scale_factor * (float(take_profit) - float(pos_price)) / float(best_ask_price))
        # Add the user's amount to the calculated quantity
        qty += float(amount)
        # Round the quantity to the allowed number of decimal places
        qty = round(qty, quantity_decimal_places)
        return qty

    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        max_leverage = self.exchange.get_max_leverage_bybit(symbol)

        print("Setting up exchange")
        self.exchange.setup_exchange_bybit(symbol)

        print("Setting leverage")
        if current_leverage != max_leverage:
            print(f"Current leverage is not at maximum. Setting leverage to maximum. Maximum is {max_leverage}")
            self.exchange.set_leverage_bybit(max_leverage, symbol)

        # # Create the strategy table
        # strategy_table = create_strategy_table(symbol, total_equity, long_upnl, short_upnl, short_pos

        while True:
            print(f"Bybit hedge volatility strategy running")
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
            print(f"4h Spread: {four_hour_distance}")
            print(f"Trend: {trend}")

            quote_currency = "USDT"
            total_equity = self.exchange.get_balance_bybit(quote_currency)

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

            self.check_amount_validity_bybit(amount, symbol)
            self.print_trade_quantities_once_bybit(max_trade_qty)

            # Take profit calc
            short_take_profit = self.calculate_short_take_profit(short_pos_price, symbol)
            long_take_profit = self.calculate_long_take_profit(long_pos_price, symbol)


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

            price_precision = int(self.exchange.get_price_precision(symbol))

            print(f"Precision: {price_precision}")


            if short_pos_price is not None:
                combined_distance = float(four_hour_distance or 0) + float(thirty_minute_distance or 0)
                average_distance = combined_distance / 2.0 if combined_distance else 0.0
                print(f"Average distance short: {average_distance}")
                distance_factor = average_distance / 100.0  # scale down the distance to a reasonable factor. Change the denominator based on your requirement.
                
                ratio = float(short_pos_price) / combined_distance if combined_distance else 1.0
                scale_factor = 1 / ratio
                ob_short_qty = abs(scale_factor * (float(short_take_profit) - float(short_pos_price)) / float(best_ask_price))
                ob_short_qty += float(amount)
                ob_short_qty *= (1.0 + distance_factor)  # Increase order quantity based on distance factor
                ob_short_qty = round(ob_short_qty / min_qty_bybit) * min_qty_bybit  # ensure it's a multiple of min_qty
                print(f"Short QTY: {ob_short_qty}")

            if long_pos_price is not None:
                combined_distance = float(four_hour_distance or 0) + float(thirty_minute_distance or 0)
                average_distance = combined_distance / 2.0 if combined_distance else 0.0
                print(f"Average distance long: {average_distance}")
                distance_factor = average_distance / 100.0  # scale down the distance to a reasonable factor. Change the denominator based on your requirement.
                
                ratio = float(long_pos_price) / combined_distance if combined_distance else 1.0
                scale_factor = 1 / ratio
                ob_long_qty = abs(scale_factor * (float(long_take_profit) - float(long_pos_price)) / float(best_ask_price))
                ob_long_qty += float(amount)
                ob_long_qty *= (1.0 + distance_factor)  # Increase order quantity based on distance factor
                ob_long_qty = round(ob_long_qty / min_qty_bybit) * min_qty_bybit  # ensure it's a multiple of min_qty
                print(f"Long QTY: {ob_long_qty}")


            print(f"Short condition: {should_short}")
            print(f"Long condition: {should_long}")
            print(f"Add short condition: {should_add_to_short}")
            print(f"Add long condition: {should_add_to_long}")

            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:
                            print(f"Placing initial long entry")
                            self.limit_order(symbol, "buy", amount, best_bid_price, positionIdx=1, reduceOnly=False)
                            print(f"Placed initial long entry")
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long entry")
                                self.limit_order(symbol, "buy", ob_long_qty, best_bid_price, positionIdx=1, reduceOnly=False)

                        if trend.lower() == "short" and should_short and short_pos_qty == 0:
                            print(f"Placing initial short entry")
                            self.limit_order(symbol, "sell", amount, best_ask_price, positionIdx=2, reduceOnly=False)
                            print("Placed initial short entry")
                        else:
                            if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty and best_ask_price > short_pos_price:
                                print(f"Placed additional short entry")
                                self.limit_order(symbol, "sell", ob_short_qty, best_bid_price, positionIdx=2, reduceOnly=False)
        
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

            if short_pos_qty > 0 and short_take_profit is not None:
                existing_short_tps = self.get_open_take_profit_order_quantities(open_orders, "buy")
                total_existing_short_tp_qty = sum(qty for qty, _ in existing_short_tps)
                if not math.isclose(total_existing_short_tp_qty, short_pos_qty):
                    try:
                        for _, existing_short_tp_id in existing_short_tps:
                            self.exchange.cancel_take_profit_orders_bybit(symbol, "buy")  # Corrected side value to "buy"
                            print(f"Short take profit canceled")
                            time.sleep(0.05)

                        #print(f"Debug: Short Position Quantity {short_pos_qty}, Short Take Profit {short_take_profit}")
                        self.exchange.create_take_profit_order_bybit(symbol, "limit", "buy", short_pos_qty, short_take_profit, positionIdx=2, reduce_only=True)
                        print(f"Short take profit set at {short_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing short TP: {e}")

            if short_upnl < -0.6 * total_equity:
                print("Negative UPNL condition met: Market closing position")
                self.exchange.create_market_order_bybit(symbol, "sell", short_pos_qty, positionIdx=2)

            if long_upnl < -0.6 * total_equity:
                print("Negative UPNL condition met: Market closing position")
                self.exchange.create_market_order_bybit(symbol, "buy", long_pos_qty, positionIdx=1)

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
            