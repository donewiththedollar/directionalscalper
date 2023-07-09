import time, math
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_DOWN
from ..strategy import Strategy

class HuobiHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.last_cancel_time = 0
        self.long_entry_order_ids = set()
        self.short_entry_order_ids = set()

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
                current_account_type = self.exchange.check_account_type_huobi()
                print(f"Current account type at start: {current_account_type}")
                if current_account_type['data']['account_type'] != '1':
                    self.exchange.switch_account_type_huobi(1)
                    time.sleep(0.05)
                    print(f"Changed account type")
                else:
                    print(f"Account type is already 1")
            except Exception as e:
                print(f"Error in switching account type {e}")

            # Annoying symbol parsing
            #parsed_symbol = self.parse_symbol(symbol)
            parsed_symbol_swap = self.parse_symbol_swap_huobi(symbol)

            min_contract_size = self.exchange.get_contract_size_huobi(parsed_symbol_swap)

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
            orderbook = self.exchange.get_orderbook(parsed_symbol_swap)
            best_bid_price = orderbook['bids'][0][0]
            best_ask_price = orderbook['asks'][0][0]

            print(f"Best bid: {best_bid_price}")
            print(f"Best ask: {best_ask_price}")

            market_data = self.exchange.get_market_data_huobi(parsed_symbol_swap)

            print(market_data)

            price_precision = market_data["precision"]

            leverage = market_data["leverage"] if market_data["leverage"] != 0 else 50.0

            max_trade_qty = round(
                (float(total_equity) * wallet_exposure / float(best_ask_price))
                / (100 / 50),
                int(float(min_contract_size)),
            )

            print(f"Max trade quantity for {symbol}: {max_trade_qty}")

            current_price = self.exchange.get_current_price(parsed_symbol_swap)

            print(f"Current price: {current_price}")

            #min_qty_huobi = float(market_data["min_qty"])

            #print(f"Min trade quantity for {parsed_symbol_swap}: {min_qty_huobi}")
            print(f"Min volume: {min_vol}")
            print(f"Min distance: {min_dist}")
            print(f"Min contract size: {min_contract_size}")

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
            print(f"Parsed symbol: {parsed_symbol_swap}")


            for i in range(max_retries):
                try:
                    position_data = self.exchange.safe_order_operation(lambda: self.exchange.get_positions_huobi(parsed_symbol_swap))
                    print(f"Fetched position data for {parsed_symbol_swap}")
                    break
                except Exception as e:
                    if i < max_retries - 1:
                        print(f"Error occurred while fetching balance: {e}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise e

            #print(f"Debug position data: {position_data}")

            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            print(f"Long contract qty: {long_pos_qty}")
            print(f"Short contract qty: {short_pos_qty}")

            short_pos_actual_qty = self.calculate_actual_quantity_huobi(short_pos_qty, parsed_symbol_swap)
            long_pos_actual_qty = self.calculate_actual_quantity_huobi(long_pos_qty, parsed_symbol_swap)

            short_pos_price = position_data["short"]["price"]
            long_pos_price = position_data["long"]["price"]

            print(f"Long pos price: {long_pos_price}")
            print(f"Short pos price: {short_pos_price}")

            long_upnl = position_data["long"]["upnl"]
            short_upnl = position_data["short"]["upnl"]
            print(f"Long uPNL: {long_upnl}")
            print(f"Short uPNL: {short_upnl}")

            # Get the 1-minute moving averages
            print(f"Fetching MA data")
            m_moving_averages = self.manager.get_1m_moving_averages(parsed_symbol_swap)
            m5_moving_averages = self.manager.get_5m_moving_averages(parsed_symbol_swap)
            ma_6_high = m_moving_averages["MA_6_H"]
            ma_6_low = m_moving_averages["MA_6_L"]
            ma_3_low = m_moving_averages["MA_3_L"]
            ma_3_high = m_moving_averages["MA_3_H"]
            ma_1m_3_high = self.manager.get_1m_moving_averages(parsed_symbol_swap)["MA_3_H"]
            ma_5m_3_high = self.manager.get_5m_moving_averages(parsed_symbol_swap)["MA_3_H"]
            print(f"Done fetching MA data")

            # Take profit calc
            if short_pos_price is not None:
                short_take_profit = self.calculate_short_take_profit_huobi(short_pos_price, parsed_symbol_swap)
                print(f"Short take profit: {short_take_profit}")
            if long_pos_price is not None:
                long_take_profit = self.calculate_long_take_profit_huobi(long_pos_price, parsed_symbol_swap)
                print(f"Long take profit: {long_take_profit}")

            # Trade conditions for strategy
            should_short = self.short_trade_condition(best_bid_price, ma_3_high)
            should_long = self.long_trade_condition(best_bid_price, ma_3_low)

            should_add_to_short = False
            should_add_to_long = False

            if short_pos_price is not None:
                should_add_to_short = short_pos_price < ma_6_low

            if long_pos_price is not None:
                should_add_to_long = long_pos_price > ma_6_high

            print(f"Short condition: {should_short}")
            print(f"Long condition: {should_long}")
            print(f"Add short condition: {should_add_to_short}")
            print(f"Add long condition: {should_add_to_long}")
            
            # New hedge logic

            if trend is not None and isinstance(trend, str):
                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist:

                        if trend.lower() == "long" and should_long and long_pos_qty == 0:
                            for i in range(max_retries):
                                try:
                                    order = self.exchange.safe_order_operation(
                                        self.exchange.create_contract_order_huobi, parsed_symbol_swap, 'limit', 'buy', amount, price=best_bid_price
                                    )
                                    self.long_entry_order_ids.add(order['id'])
                                    print(f"Placed initial long entry")
                                    time.sleep(0.05)
                                    break
                                except Exception as e:
                                    if i < max_retries - 1:  # if not the last try
                                        print(f"Error occurred while placing an order: {e}. Retrying in {retry_delay} seconds...")
                                        time.sleep(retry_delay)
                                    else:
                                        raise e 

                        else:
                            if trend.lower() == "long" and should_add_to_long and long_pos_qty < max_trade_qty and best_bid_price < long_pos_price:
                                print(f"Placed additional long entry")
                                for i in range(max_retries):
                                    try:
                                        order = self.exchange.safe_order_operation(
                                            self.exchange.create_contract_order_huobi, parsed_symbol_swap, 'limit', 'buy', amount, price=best_bid_price
                                        )
                                        self.long_entry_order_ids.add(order['id'])
                                        time.sleep(0.05)
                                        break
                                    except Exception as e:
                                        if i < max_retries - 1:  # if not the last try
                                            print(f"Error occurred while placing an order: {e}. Retrying in {retry_delay} seconds...")
                                            time.sleep(retry_delay)
                                        else:
                                            raise e 

                        if trend.lower() == "short" and should_short and short_pos_qty == 0:
                            for i in range(max_retries):
                                try:
                                    order = self.exchange.safe_order_operation(
                                        self.exchange.create_contract_order_huobi, parsed_symbol_swap, 'limit', 'sell', amount, price=best_ask_price
                                    )
                                    self.short_entry_order_ids.add(order['id'])
                                    print("Placed initial short entry")
                                    time.sleep(0.05)
                                    break
                                except Exception as e:
                                    if i < max_retries - 1:  # if not the last try
                                        print(f"Error occurred while placing an order: {e}. Retrying in {retry_delay} seconds...")
                                        time.sleep(retry_delay)
                                    else:
                                        raise e 

                        else:
                            if trend.lower() == "short" and should_add_to_short and short_pos_qty < max_trade_qty and best_ask_price > short_pos_price:
                                print(f"Placed additional short entry")
                                for i in range(max_retries):
                                    try:
                                        order = self.exchange.safe_order_operation(
                                            self.exchange.create_contract_order_huobi, parsed_symbol_swap, 'limit', 'sell', amount, price=best_ask_price
                                        )
                                        self.short_entry_order_ids.add(order['id'])
                                        time.sleep(0.05)
                                        break
                                    except Exception as e:
                                        if i < max_retries - 1:  # if not the last try
                                            print(f"Error occurred while placing an order: {e}. Retrying in {retry_delay} seconds...")
                                            time.sleep(retry_delay)
                                        else:
                                            raise e 

            open_orders = self.exchange.get_open_orders_huobi(parsed_symbol_swap)

            #print(f"Open orders: {open_orders}")

            if long_pos_qty > 0 and long_take_profit is not None:
                existing_long_tps = self.get_open_take_profit_order_quantities_huobi(open_orders, "sell")
                total_existing_long_tp_qty = sum(qty for qty, _ in existing_long_tps)
                print(f"Existing long TPs: {existing_long_tps}")
                if not math.isclose(total_existing_long_tp_qty, long_pos_qty):
                    try:
                        for qty, existing_long_tp_id in existing_long_tps:
                            if not math.isclose(qty, long_pos_qty):
                                self.exchange.safe_order_operation(
                                    self.exchange.cancel_order_huobi, id=existing_long_tp_id, symbol=parsed_symbol_swap
                                )
                                print(f"Long take profit {existing_long_tp_id} canceled")
                                time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in cancelling long TP orders {e}")

                if not any(math.isclose(qty, long_pos_actual_qty) for qty, _ in existing_long_tps):
                    try:
                        self.exchange.safe_order_operation(
                            self.exchange.create_take_profit_order,
                            parsed_symbol_swap, "limit", "sell", long_pos_actual_qty, long_take_profit, reduce_only=True
                        )
                        print(f"Long take profit set at {long_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing long TP: {e}")

            if short_pos_qty > 0 and short_take_profit is not None:
                existing_short_tps = self.get_open_take_profit_order_quantities_huobi(open_orders, "buy")
                total_existing_short_tp_qty = sum(qty for qty, _ in existing_short_tps)
                print(f"Existing short TPs: {existing_short_tps}")
                if not math.isclose(total_existing_short_tp_qty, short_pos_qty):
                    try:
                        for qty, existing_short_tp_id in existing_short_tps:
                            if not math.isclose(qty, short_pos_qty):
                                self.exchange.safe_order_operation(
                                    self.exchange.cancel_order_huobi, id=existing_short_tp_id, symbol=parsed_symbol_swap
                                )
                                print(f"Short take profit {existing_short_tp_id} canceled")
                                time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in cancelling short TP orders: {e}")

                if not any(math.isclose(qty, short_pos_qty) for qty, _ in existing_short_tps):
                    try:
                        self.exchange.safe_order_operation(
                            self.exchange.create_take_profit_order,
                            parsed_symbol_swap, "limit", "buy", short_pos_qty, short_take_profit, reduce_only=True
                        )
                        print(f"Short take profit set at {short_take_profit}")
                        time.sleep(0.05)
                    except Exception as e:
                        print(f"Error in placing short TP: {e}")

            # Cancel entries
            current_time = time.time()
            if current_time - self.last_cancel_time >= 60:  # Execute this block every 1 minute
                try:
                    if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                        self.exchange.cancel_all_entries_huobi(parsed_symbol_swap)
                        print(f"Canceled entry orders for {parsed_symbol_swap}")
                        time.sleep(0.05)
                except Exception as e:
                    print(f"An error occurred while canceling entry orders: {e}")

                self.last_cancel_time = current_time  # Update the last cancel time

            time.sleep(30)
            