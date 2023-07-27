import time
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
from ..strategy import Strategy
from typing import Tuple
import threading
import os

class BinanceHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.last_cancel_time = 0
        self.current_wallet_exposure = 1.0
        self.printed_trade_quantities = False

    def calculate_trade_quantity(self, symbol, leverage):
        dex_equity = self.exchange.get_balance_bybit('USDT')
        trade_qty = (float(dex_equity) * self.current_wallet_exposure) / leverage
        return trade_qty

    def truncate(self, number: float, precision: int) -> float:
        return float(Decimal(number).quantize(Decimal('0.' + '0'*precision), rounding=ROUND_DOWN))

    def limit_order(self, symbol, side, amount, price, reduceOnly=False):
        try:
            params = {"reduceOnly": reduceOnly}
            order = self.exchange.create_limit_order_binance(symbol, side, amount, price, params=params)
            return order
        except Exception as e:
            print(f"An error occurred in limit_order(): {e}")

    def get_open_take_profit_order_quantity_binance(self, orders, side):
        for order in orders:
            if order['side'].lower() == side.lower() and order.get('reduce_only', False):
                return order['origQty'], order['orderId']
        return None, None

    def get_open_take_profit_order_quantities_binance(self, orders, side):
        take_profit_orders = []
        for order in orders:
            if order['side'].lower() == side.lower() and order.get('reduce_only', False):
                take_profit_orders.append((order['origQty'], order['orderId']))
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

            #print("Debug: Long Target Price:", long_target_price)

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        #current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        true_max_leverage = self.exchange.get_max_leverage_binance(symbol)
        print(f"True max leverage: {true_max_leverage}")
        max_leverage = 20.0
        min_notional = 5.0
        print(f"Max leverage {max_leverage}")

        # print("Setting up exchange")
        # self.exchange.setup_exchange_bybit(symbol)

        # print("Setting leverage")
        # if current_leverage != max_leverage:
        #     print(f"Current leverage is not at maximum. Setting leverage to maximum. Maximum is {max_leverage}")
        #     self.exchange.set_leverage_bybit(max_leverage, symbol)

        while True:
            print(f"Binance hedge strategy running")
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
            total_equity = self.exchange.get_balance_binance(quote_currency)

            print(f"Total equity: {total_equity}")

            current_price = self.exchange.get_current_price_binance(symbol)
            print(f"Current price: {current_price}")
            
            market_data = self.exchange.get_market_data_bybit(symbol)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]


            print(f"Market data: {market_data}")

            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")

            min_qty = min_notional / current_price  # Compute minimum quantity

            max_trade_qty = round(
                (float(total_equity) * wallet_exposure / float(best_ask_price))
                / (100 / max_leverage),
                int(float(market_data["min_qty"])),
            )            
            
            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            min_qty_binance = market_data["min_qty"]
            print(f"Min qty: {min_qty_binance}")

            print(f"Min qty based on notional: {min_qty}")

            if float(amount) < min_qty:
                print(f"The amount you entered ({amount}) is less than the minimum required by Binance for {symbol}: {min_qty}.")
                break
            else:
                print(f"The amount you entered ({amount}) is valid for {symbol}")

            if not self.printed_trade_quantities:
                self.exchange.print_trade_quantities_bybit(max_trade_qty, [0.001, 0.01, 0.1, 1, 2.5, 5], wallet_exposure, best_ask_price)
                self.printed_trade_quantities = True

            # Get the 1-minute moving averages
            print(f"Fetching MA data")
            m_moving_averages = self.manager.get_1m_moving_averages(symbol)
            m5_moving_averages = self.manager.get_5m_moving_averages(symbol)
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

            position_data = self.exchange.get_positions_binance(symbol)

            print(position_data)

            #self.exchange.print_positions_structure_binance()

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

            # Precision is annoying

            # price_precision = int(self.exchange.get_price_precision(symbol))

            # print(f"Price Precision: {price_precision}")

            # Precision
            #price_precision, quantity_precision = self.exchange.get_symbol_precision_bybit(symbol)

            # Take profit calc
            short_take_profit = self.calculate_short_take_profit(short_pos_price, symbol)
            long_take_profit = self.calculate_long_take_profit(long_pos_price, symbol)

            print(f"Long take profit: {long_take_profit}")
            print(f"Short take profit: {short_take_profit}")


            should_short = best_bid_price > ma_3_high
            should_long = best_bid_price < ma_3_high

            should_add_to_short = False
            should_add_to_long = False
            
            if short_pos_price is not None:
                should_add_to_short = short_pos_price < ma_6_low

            if long_pos_price is not None:
                should_add_to_long = long_pos_price > ma_6_low

            print(f"Short condition: {should_short}")
            print(f"Long condition: {should_long}")
            print(f"Add short condition: {should_add_to_short}")
            print(f"Add long condition: {should_add_to_long}")

            #self.exchange.create_limit_order_binance(symbol, "buy", amount, best_bid_price)
            #self.limit_order(symbol, "buy", amount, best_bid_price, reduceOnly=False)

            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:
                            print(f"Placing initial long entry")
                            self.exchange.binance_create_limit_order(symbol, "buy", amount, best_bid_price)
                            print(f"Placed initial long entry")
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long entry")
                                self.exchange.binance_create_limit_order(symbol, "buy", amount, best_bid_price)

                        if trend.lower() == "short" and should_short and short_pos_qty == 0:
                            print(f"Placing initial short entry")
                            self.exchange.binance_create_limit_order(symbol, "sell", amount, best_ask_price)
                            print("Placed initial short entry")
                        else:
                            if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty and best_ask_price > short_pos_price:
                                print(f"Placed additional short entry")
                                self.exchange.binance_create_limit_order(symbol, "sell", amount, best_ask_price)
        
            open_orders = self.exchange.get_open_orders_binance(symbol)

            print(open_orders)

            # # Call the get_open_take_profit_order_quantity function for the 'buy' side
            # buy_qty, buy_id = self.get_open_take_profit_order_quantity(open_orders, 'buy')

            # # Call the get_open_take_profit_order_quantity function for the 'sell' side
            # sell_qty, sell_id = self.get_open_take_profit_order_quantity(open_orders, 'sell')

            # # Print the results
            # print("Buy Take Profit Order - Quantity: ", buy_qty, "ID: ", buy_id)
            # print("Sell Take Profit Order - Quantity: ", sell_qty, "ID: ", sell_id)

            if long_pos_qty > 0 and long_take_profit is not None:
                existing_long_tps = self.get_open_take_profit_order_quantities_binance(open_orders, "sell")
                total_existing_long_tp_qty = sum(qty for qty, _ in existing_long_tps)
                if not math.isclose(total_existing_long_tp_qty, long_pos_qty):
                    try:
                        for _, existing_long_tp_id in existing_long_tps:
                            self.exchange.cancel_take_profit_orders_bybit(symbol, "sell")  # Corrected side value to "sell"
                            print(f"Long take profit canceled")
                            time.sleep(0.05)

                        #print(f"Debug: Long Position Quantity {long_pos_qty}, Long Take Profit {long_take_profit}")
                        self.exchange.create_close_position_limit_order_binance(symbol, "sell", long_pos_qty, long_take_profit)
                        print(f"Long take profit set at {long_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing long TP: {e}")

            if short_pos_qty > 0 and short_take_profit is not None:
                existing_short_tps = self.get_open_take_profit_order_quantities_binance(open_orders, "buy")
                total_existing_short_tp_qty = sum(qty for qty, _ in existing_short_tps)
                if not math.isclose(total_existing_short_tp_qty, short_pos_qty):
                    try:
                        for _, existing_short_tp_id in existing_short_tps:
                            self.exchange.cancel_take_profit_orders_bybit(symbol, "buy")  # Corrected side value to "buy"
                            print(f"Short take profit canceled")
                            time.sleep(0.05)

                        #print(f"Debug: Short Position Quantity {short_pos_qty}, Short Take Profit {short_take_profit}")
                        self.exchange.create_close_position_limit_order_binance(symbol, "limit", "buy", short_pos_qty, short_take_profit)
                        print(f"Short take profit set at {short_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing short TP: {e}")

            # Cancel entries
            current_time = time.time()
            if current_time - self.last_cancel_time >= 60:  # Execute this block every 1 minute
                try:
                    if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                        self.exchange.cancel_all_entries_binance(symbol)
                        print(f"Canceled entry orders for {symbol}")
                        time.sleep(0.05)
                except Exception as e:
                    print(f"An error occurred while canceling entry orders: {e}")

                self.last_cancel_time = current_time  # Update the last cancel time

            time.sleep(30)