import time, math
from decimal import Decimal, ROUND_HALF_UP
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

    def parse_symbol_swap(self, symbol):
        if "huobi" in self.exchange.name.lower():
            base_currency = symbol[:-4]
            quote_currency = symbol[-4:] 
            return f"{base_currency}/{quote_currency}:{quote_currency}"
        return symbol

    def limit_order(self, symbol, side, amount, price, reduce_only=False):
        order = self.exchange.create_order(symbol, 'limit', side, amount, price, reduce_only=reduce_only)
        return order

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

    def run(self, symbol, amount):
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        wallet_exposure = self.config.wallet_exposure
        min_order_value = 6
        max_retries = 5
        retry_delay = 5

        while True:
            print(f"Huobi strategy running")

            try:
                self.exchange.switch_account_type_huobi(1)
                time.sleep(0.05)
                print(f"Changed account type")
            except Exception as e:
                print(f"Error in switching account type {e}")

            parsed_symbol = self.parse_symbol(symbol)

            quote = 'USDT'

            for i in range(max_retries):
                try:
                    total_equity = self.exchange.get_balance_huobi(quote, 'swap', 'linear')
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

            price_precision = market_data["precision"]

            leverage = market_data["leverage"] if market_data["leverage"] != 0 else 50.0

            max_trade_qty = round(
                (float(total_equity) * wallet_exposure / float(best_ask_price))
                / (100 / leverage),
                int(float(market_data["min_qty"])),
            )

            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            current_price = self.exchange.get_current_price(parsed_symbol)

            print(f"Current price: {current_price}")

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
            print(f"Entry size: {amount}")

            print(f"Parsed symbol: {parsed_symbol}")
            print(f"Regular symbol: {symbol}")

            parsed_symbol_swap = self.parse_symbol_swap(symbol)

            position_data = self.exchange.get_positions_huobi(parsed_symbol_swap)
            print(f"Fetched position data for {parsed_symbol_swap}")

            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            print(f"Long pos qty: {long_pos_qty}")
            print(f"Short pos qty: {short_pos_qty}")

            short_pos_price = position_data["short"]["price"]
            long_pos_price = position_data["long"]["price"]

            print(f"Long pos price: {long_pos_price}")
            print(f"Short pos price: {short_pos_price}")

            # Get the 1-minute moving averages
            print(f"Fetching MA data")
            m_moving_averages = self.manager.get_1m_moving_averages(parsed_symbol_swap)
            m5_moving_averages = self.manager.get_5m_moving_averages(parsed_symbol_swap)
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(parsed_symbol_swap)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(parsed_symbol_swap)["MA_3_H"]
            print(f"Done fetching MA data")

            # Take profit calc
            if short_pos_price is not None:
                short_take_profit = self.calculate_short_take_profit(short_pos_price, parsed_symbol_swap)
                print(f"Short take profit: {short_take_profit}")
            if long_pos_price is not None:
                long_take_profit = self.calculate_long_take_profit(long_pos_price, parsed_symbol_swap)
                print(f"Long take profit: {long_take_profit}")

            # Trade conditions for strategy
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

            # print(f"Testing trade")
            # self.exchange.create_contract_order_huobi(parsed_symbol_swap, 'limit', 'buy', amount, price=best_bid_price)

            # New hedge logic
            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:

                            # self.limit_order(symbol, "buy", amount, best_bid_price, reduce_only=False)
                            self.exchange.create_contract_order_huobi(parsed_symbol_swap, 'limit', 'buy', amount, price=best_bid_price)
                            print(f"Placed initial long entry")
                            time.sleep(0.05)
                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long entry")
                                self.exchange.create_contract_order_huobi(parsed_symbol_swap, 'limit', 'buy', amount, price=best_bid_price)
                                time.sleep(0.05)

                        if trend.lower() == "short" and should_short and short_pos_qty == 0:

                            self.exchange.create_contract_order_huobi(parsed_symbol_swap, 'limit', 'sell', amount, price=best_ask_price)
                            print("Placed initial short entry")
                            time.sleep(0.05)
                        else:
                            if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty and best_ask_price > short_pos_price:
                                print(f"Placed additional short entry")
                                self.exchange.create_contract_order_huobi(parsed_symbol_swap, 'limit', 'sell', amount, price=best_ask_price)
                                time.sleep(0.05)


            time.sleep(30)
            