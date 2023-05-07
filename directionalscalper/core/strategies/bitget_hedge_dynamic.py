import time, math
from decimal import Decimal, ROUND_HALF_UP
from .strategy import Strategy

class BitgetDynamicHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config)
        self.manager = manager

    def limit_order(self, symbol, side, amount, price, reduce_only=False):
        min_qty_usd = 5
        current_price = self.exchange.get_current_price(symbol)
        min_qty_bitget = min_qty_usd / current_price

        print(f"Min trade quantitiy for {symbol}: {min_qty_bitget}")

        if float(amount) < min_qty_bitget:
            print(f"The amount you entered ({amount}) is less than the minimum required by Bitget for {symbol}: {min_qty_bitget}.")
            return
        order = self.exchange.create_order(symbol, 'limit', side, amount, price, reduce_only=reduce_only)
        return order

    def take_profit_order(self, symbol, side, amount, price, reduce_only=True):
        min_qty_usd = 5
        current_price = self.exchange.get_current_price(symbol)
        min_qty_bitget = min_qty_usd / current_price

        print(f"Min trade quantitiy for {symbol}: {min_qty_bitget}")

        if float(amount) < min_qty_bitget:
            print(f"The amount you entered ({amount}) is less than the minimum required by Bitget for {symbol}: {min_qty_bitget}.")
            return
        order = self.exchange.create_order(symbol, 'limit', side, amount, price, reduce_only=reduce_only)
        return order

    def close_position(self, symbol, side, amount):
        try:
            self.exchange.create_market_order(symbol, side, amount)
            print(f"Closed {side} position for {symbol} with amount {amount}")
        except Exception as e:
            print(f"An error occurred while closing the position: {e}")

    def get_open_take_profit_order_quantity(self, orders, side):
        for order in orders:
            if order['side'] == side and order['reduce_only']:
                return order['qty'], order['id']
        return None, None

    def parse_symbol(self, symbol):
        if "bitget" in self.exchange.name.lower():
            if symbol == "PEPEUSDT" or symbol == "PEPEUSDT_UMCBL":
                return "1000PEPEUSDT"
            return symbol.replace("_UMCBL", "")
        return symbol

    def cancel_take_profit_orders(self, symbol, side):
        self.exchange.cancel_close_bitget(symbol, side)

    def has_open_orders(self, symbol):
        open_orders = self.exchange.get_open_orders(symbol)
        return len(open_orders) > 0

    def calculate_short_take_profit(self, short_pos_price, symbol):
        if short_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            short_target_price = Decimal(short_pos_price) - (ma_6_high - ma_6_low)
            short_target_price = short_target_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_UP
            )

            short_profit_price = short_target_price

            return float(short_profit_price)
        return None

    def calculate_long_take_profit(self, long_pos_price, symbol):
        if long_pos_price is None:
            return None

        five_min_data = self.manager.get_5m_moving_averages(symbol)
        price_precision = int(self.exchange.get_price_precision(symbol))

        if five_min_data is not None:
            ma_6_high = Decimal(five_min_data["MA_6_H"])
            ma_6_low = Decimal(five_min_data["MA_6_L"])

            long_target_price = Decimal(long_pos_price) + (ma_6_high - ma_6_low)
            long_target_price = long_target_price.quantize(
                Decimal('1e-{}'.format(price_precision)),
                rounding=ROUND_HALF_UP
            )

            long_profit_price = long_target_price

            return float(long_profit_price)
        return None

    def run(self, symbol):
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        wallet_exposure = self.config.wallet_exposure
        min_order_value = 5

        while True:
            # Max trade qty calculation
            quote_currency = "USDT"  # Change this to your desired quote currency
            dex_equity = self.exchange.get_balance_bitget(quote_currency)

            market_data = self.exchange.get_market_data_bitget(symbol)

            # Orderbook data
            orderbook = self.exchange.get_orderbook(symbol)
            best_bid_price = orderbook['bids'][0][0]
            best_ask_price = orderbook['asks'][0][0]

            leverage = float(market_data["leverage"]) if market_data["leverage"] != 0 else 50.0

            max_trade_qty = round(
                (float(dex_equity) * wallet_exposure / float(best_ask_price))
                / (100 / leverage),
                int(float(market_data["min_qty"])),
            )

            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            # min_qty_bitget = market_data["min_qty"]
            current_price = self.exchange.get_current_price(symbol)

            amount = min_order_value / current_price

            print(f"Current price: {current_price}")
            print(f"Dynamic amount: {amount}")

            # # Update the amount based on the current price
            # dynamic_amount = max(amount, min_order_value / current_price)
            # dynamic_amount = round(dynamic_amount, int(float(market_data["min_qty"])))

            min_qty_bitget = min_order_value / current_price

            print(f"Min trade quantitiy for {symbol}: {min_qty_bitget}")
                
            if float(amount) < min_qty_bitget:
                print(f"The amount you entered ({amount}) is less than the minimum required by Bitget for {symbol}: {min_qty_bitget}.")
                break
            else:
                print(f"The amount you entered ({amount}) is valid for {symbol}")

            print(f"Min volume: {min_vol}")
            print(f"Min distance: {min_dist}")

            # Get data from manager
            data = self.manager.get_data()

            # Parse the symbol according to the exchange being used
            parsed_symbol = self.parse_symbol(symbol)

            # Data we need from API
            one_minute_volume = self.manager.get_asset_value(parsed_symbol, data, "1mVol")
            five_minute_distance = self.manager.get_asset_value(parsed_symbol, data, "5mSpread")
            trend = self.manager.get_asset_value(parsed_symbol, data, "Trend")
            print(f"1m Volume: {one_minute_volume}")
            print(f"5m Spread: {five_minute_distance}")
            print(f"Trend: {trend}")


            # Hedge logic starts

            # data = self.exchange.exchange.fetch_positions([symbol])
            # print(f"Bitget positions response: {data}")   
 
            # Get pos data from exchange
            position_data = self.exchange.get_positions_bitget(symbol) 
            print(f"Fetching position data")
            #print(f"Raw position data: {position_data}")

            # Extract short and long position prices
            # short_pos_price = position_data["short"]["price"]
            # long_pos_price = position_data["long"]["price"]

            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            print(f"Short pos qty: {short_pos_qty}")
            print(f"Long pos qty: {long_pos_qty}")

            short_pos_price = position_data["short"]["price"] if short_pos_qty > 0 else None
            long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

            print(f"Short pos price: {short_pos_price}")
            print(f"Long pos price: {long_pos_price}")

            # Get the 1-minute moving averages
            print(f"Fetching MA data")
            m_moving_averages = self.manager.get_1m_moving_averages(symbol)
            m5_moving_averages = self.manager.get_5m_moving_averages(symbol)
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

            # Take profit calc
            short_take_profit = self.calculate_short_take_profit(short_pos_price, symbol)
            long_take_profit = self.calculate_long_take_profit(long_pos_price, symbol)

            print(f"Short take profit: {short_take_profit}")
            print(f"Long take profit: {long_take_profit}")

            price_precision = market_data["precision"]

            if long_take_profit is not None:
                precise_long_take_profit = round(long_take_profit, int(-math.log10(price_precision)))

            if short_take_profit is not None:
                precise_short_take_profit = round(short_take_profit, int(-math.log10(price_precision)))

            # Trade conditions 
            # should_short = self.short_trade_condition(best_ask_price, m_moving_averages["MA_3_H"])
            # should_long = self.long_trade_condition(best_bid_price, m_moving_averages["MA_3_L"])
            should_short = self.short_trade_condition(best_bid_price, ma_3_high)
            should_long = self.long_trade_condition(best_bid_price, ma_3_high)

            should_add_to_short = self.add_short_trade_condition(short_pos_price, ma_6_low)
            should_add_to_long = self.add_long_trade_condition(long_pos_price, ma_6_low)

            print(f"Short condition: {should_short}")
            print(f"Long condition: {should_long}")
            print(f"Add short condition: {should_add_to_short}")
            print(f"Add long condition: {should_add_to_long}")

            close_short_position = short_pos_qty > 0 and current_price <= short_take_profit
            close_long_position = long_pos_qty > 0 and current_price >= long_take_profit

            print(f"Current price: {current_price}")
            print(f"Close short position condition: {close_short_position}")
            print(f"Close long position condition: {close_long_position}")

            # New hedge logic
            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:

                            self.limit_order(symbol, "buy", amount, best_bid_price, reduce_only=False)
                            print(f"Placed initial long entry")
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long entry")
                                self.limit_order(symbol, "buy", amount, best_bid_price, reduce_only=False)

                        if trend.lower() == "short" and should_short and short_pos_qty == 0:

                            self.limit_order(symbol, "sell", amount, best_ask_price, reduce_only=False)
                            print("Placed initial short entry")
                        else:
                            if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty and best_ask_price > short_pos_price:
                                print(f"Placed additional short entry")
                                self.limit_order(symbol, "sell", amount, best_ask_price, reduce_only=False)
            
            open_orders = self.exchange.get_open_orders_bitget(symbol)

            if long_pos_qty > 0 and long_take_profit is not None:
                existing_long_tp_qty, existing_long_tp_id = self.get_open_take_profit_order_quantity(open_orders, "close_long")
                if existing_long_tp_qty is None or existing_long_tp_qty != long_pos_qty:
                    try:
                        if existing_long_tp_id is not None:
                            self.cancel_take_profit_orders(symbol, "long")
                            print(f"Long take profit canceled")
                            time.sleep(0.05)

                        self.exchange.create_take_profit_order(symbol, "limit", "sell", long_pos_qty, long_take_profit, reduce_only=True)
                        print(f"Long take profit set at {long_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing long TP: {e}")

            if short_pos_qty > 0 and short_take_profit is not None:
                existing_short_tp_qty, existing_short_tp_id = self.get_open_take_profit_order_quantity(open_orders, "close_short")
                if existing_short_tp_qty is None or existing_short_tp_qty != short_pos_qty:
                    try:
                        if existing_short_tp_id is not None:
                            self.cancel_take_profit_orders(symbol, "short")
                            print(f"Short take profit canceled")
                            time.sleep(0.05)

                        self.exchange.create_take_profit_order(symbol, "limit", "buy", short_pos_qty, short_take_profit, reduce_only=True)
                        print(f"Short take profit set at {short_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing short TP: {e}")

            # Cancel entries
            try:
                if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                    self.exchange.cancel_all_entries(symbol)
                    print(f"Canceled entry orders for {symbol}")
                    time.sleep(0.05)
            except Exception as e:
                print(f"An error occurred while canceling entry orders: {e}")

            time.sleep(30)
