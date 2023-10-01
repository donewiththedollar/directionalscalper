import time
import json
import os
import copy
import pytz
from threading import Thread
from datetime import datetime
from ...strategy import Strategy
from ...logger import Logger
from ....bot_metrics import BotDatabase
from live_table_manager import shared_symbols_data

logging = Logger(logger_name="Bybitfivemin", filename="Bybitfivemin.log", stream=True)

class BybitMMFiveMinute(Strategy):
    def __init__(self, exchange, manager, config, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.symbols_allowed = symbols_allowed
        self.manager = manager
        self.last_health_check_time = time.time()
        self.health_check_interval = 600
        self.bot_db = BotDatabase(exchange=self.exchange)
        self.bot_db.create_tables_if_not_exists()
        self.max_long_trade_qty = None
        self.max_short_trade_qty = None
        self.initial_max_long_trade_qty = None
        self.initial_max_short_trade_qty = None
        self.long_leverage_increased = False
        self.short_leverage_increased = False
        self.last_stale_order_check_time = time.time()
        self.initial_max_long_trade_qty_per_symbol = {}
        self.initial_max_short_trade_qty_per_symbol = {}
        self.long_pos_leverage_per_symbol = {}
        self.short_pos_leverage_per_symbol = {}
        self.last_long_tp_update = datetime.now()
        self.last_short_tp_update = datetime.now()
        self.next_long_tp_update = self.calculate_next_update_time()
        self.next_short_tp_update = self.calculate_next_update_time()
        self.last_cancel_time = 0
        self.spoofing_active = False
        self.spoofing_wall_size = 5
        self.spoofing_duration = 5
        self.spoofing_interval = 1

    def run(self, symbol):
        threads = [
            Thread(target=self.run_single_symbol, args=(symbol,))
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def run_single_symbol(self, symbol):

        print(f"Initializing default values")

        # Initialize potentially missing values
        min_qty = None
        current_price = None
        total_equity = None
        available_equity = None
        one_minute_volume = None
        five_minute_distance = None
        trend = None
        long_pos_qty = 0
        short_pos_qty = 0
        long_upnl = 0
        short_upnl = 0
        cum_realised_pnl_long = 0
        cum_realised_pnl_short = 0
        long_pos_price = None
        short_pos_price = None

        print(f"Running for symbol (inside run_single_symbol method): {symbol}")

        quote_currency = "USDT"
        max_retries = 5
        retry_delay = 5
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        MaxAbsFundingRate = self.config.MaxAbsFundingRate
        current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        max_leverage = self.exchange.get_max_leverage_bybit(symbol)

        if self.config.dashboard_enabled:
            dashboard_path = os.path.join(self.config.shared_data_path, "shared_data.json")

        logging.info("Setting up exchange")
        self.exchange.setup_exchange_bybit(symbol)

        logging.info("Setting leverage")
        if current_leverage != max_leverage:
            logging.info(f"Current leverage is not at maximum. Setting leverage to maximum. Maximum is {max_leverage}")
            self.exchange.set_leverage_bybit(max_leverage, symbol)

        previous_five_minute_distance = None

        while True:
            rotator_symbols = self.manager.get_auto_rotate_symbols()
            if symbol not in rotator_symbols:
                logging.info(f"Symbol {symbol} not in rotator symbols. Waiting for it to reappear.")
                time.sleep(60)
                continue

            should_exit = False
            rotator_symbols = self.manager.get_auto_rotate_symbols()
            if symbol not in rotator_symbols:
                logging.info(f"Symbol {symbol} no longer in rotator symbols. Stopping operations for this symbol.")
                should_exit = True

            whitelist = self.config.whitelist
            blacklist = self.config.blacklist
            if symbol not in whitelist or symbol in blacklist:
                logging.info(f"Symbol {symbol} is no longer allowed based on whitelist/blacklist. Stopping operations for this symbol.")
                should_exit = True

            if should_exit:
                break

            api_data = self.manager.get_api_data(symbol)
            one_minute_volume = api_data['1mVol']
            five_minute_volume = api_data['5mVol']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            funding_rate = api_data['Funding']
            hma_trend = api_data['HMA Trend']

            logging.info(f"One minute volume for {symbol} : {one_minute_volume}")
            logging.info(f"Five minute distance for {symbol} : {five_minute_distance}")

            funding_check = self.is_funding_rate_acceptable(symbol)

            logging.info(f"Funding check on {symbol} : {funding_check}")

            total_equity = self.retry_api_call(self.exchange.get_balance_bybit, quote_currency)
            available_equity = self.retry_api_call(self.exchange.get_available_balance_bybit, quote_currency)
            current_price = self.exchange.get_current_price(symbol)
            market_data = self.get_market_data_with_retry(symbol, max_retries=5, retry_delay=5)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

            logging.info(f"Variables in main loop for {symbol}: market_data={market_data}, total_equity={total_equity}, best_ask_price={best_ask_price}, max_leverage={max_leverage}")

            moving_averages = self.get_all_moving_averages(symbol)
            position_data = self.retry_api_call(self.exchange.get_positions_bybit, symbol)

            logging.info(f"Position data for {symbol} : {position_data}")

            open_position_data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)

            open_symbols = self.extract_symbols_from_positions_bybit(open_position_data)
            open_symbols = [symbol.replace("/", "") for symbol in open_symbols]

            logging.info(f"Open symbols: {open_symbols}")
            
            rotator_symbols = self.manager.get_auto_rotate_symbols()
            logging.info(f"HMA Current rotator symbols: {rotator_symbols}")

            symbols_to_manage = [s for s in open_symbols if s not in rotator_symbols]

            logging.info(f"Symbols to manage {symbols_to_manage}")

            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            logging.info(f"Open orders: {open_orders}")

            self.manage_non_rotator_symbols_5m(symbols_to_manage, total_equity, open_orders, market_data, position_data)

            can_open_new_position = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, symbol)

            logging.info(f"Can open new position: {can_open_new_position}")

            # longtps = self.exchange.get_take_profit_order_quantity_bybit(symbol, 'sell')
            # shorttps = self.exchange.get_take_profit_order_quantity_bybit(symbol, 'buy')

            # logging.info(f"Symbol: {symbol} Long TPs: {longtps}")
            # logging.info(f"Symbol: {symbol} Short TPs: {shorttps}")

            if symbol in open_symbols:


                short_pos_qty = position_data["short"]["qty"]
                long_pos_qty = position_data["long"]["qty"]

                logging.info(f"Long pos qty {long_pos_qty} for {symbol}")
                logging.info(f"Short pos qty {short_pos_qty} for {symbol}")

                short_liq_price = position_data["short"]["liq_price"]
                long_liq_price = position_data["long"]["liq_price"]

                self.bybit_reset_position_leverage_long_v3(symbol, long_pos_qty, total_equity, best_ask_price, max_leverage)
                self.bybit_reset_position_leverage_short_v3(symbol, short_pos_qty, total_equity, best_ask_price, max_leverage)

                long_dynamic_amount, short_dynamic_amount, min_qty = self.calculate_dynamic_amount(symbol, market_data, total_equity, best_ask_price, max_leverage)
                logging.info(f"Long dynamic amount: {long_dynamic_amount} for {symbol}")
                logging.info(f"Short dynamic amount: {short_dynamic_amount} for {symbol}")

                self.print_trade_quantities_once_bybit(symbol, self.max_long_trade_qty)
                self.print_trade_quantities_once_bybit(symbol, self.max_short_trade_qty)

                short_upnl = position_data["short"]["upnl"]
                long_upnl = position_data["long"]["upnl"]
                cum_realised_pnl_long = position_data["long"]["cum_realised"]
                cum_realised_pnl_short = position_data["short"]["cum_realised"]

                short_pos_price = position_data["short"]["price"] if short_pos_qty > 0 else None
                long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

                short_take_profit = None
                long_take_profit = None

                short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)
                previous_five_minute_distance = five_minute_distance

                should_short = self.short_trade_condition(best_ask_price, moving_averages["ma_3_high"])
                should_long = self.long_trade_condition(best_bid_price, moving_averages["ma_3_low"])
                should_add_to_short = False
                should_add_to_long = False

                if short_pos_price is not None:
                    should_add_to_short = short_pos_price < moving_averages["ma_6_low"] and self.short_trade_condition(best_ask_price, moving_averages["ma_6_high"])

                if long_pos_price is not None:
                    should_add_to_long = long_pos_price > moving_averages["ma_6_high"] and self.long_trade_condition(best_bid_price, moving_averages["ma_6_low"])


                # open_tp_orders = self.exchange.bybit.get_open_tp_orders(symbol)

                # logging.info(f"Open TP Orders for {symbol} {open_tp_orders}")

                open_tp_order_count = self.exchange.bybit.get_open_tp_order_count(symbol)

                logging.info(f"Open TP order count {open_tp_order_count}")

                current_time = time.time()
                if current_time - self.last_cancel_time >= self.spoofing_interval:
                    self.spoofing_active = True
                    self.spoofing_action(symbol, short_dynamic_amount, long_dynamic_amount)

                self.bybit_entry_mm_5m(open_orders, symbol, trend, hma_trend, mfirsi_signal, five_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_long, should_short, should_add_to_long, should_add_to_short)

                # # Check for existing TP orders
                # existing_long_tp_count = self.exchange.bybit.get_open_tp_order_count(symbol)
                # existing_short_tp_count = self.exchange.bybit.get_open_tp_order_count(symbol)

                # logging.info(f"Existing long tps: {existing_long_tp_count}")
                # logging.info(f"Existing short tps: {existing_short_tp_count}")

                if long_pos_qty > 0 and long_take_profit is not None:
                        self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

                if short_pos_qty > 0 and short_take_profit is not None:
                        self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)

                tp_order_counts = self.exchange.bybit.get_open_tp_order_count(symbol)

                long_tp_counts = tp_order_counts['long_tp_count']
                short_tp_counts = tp_order_counts['short_tp_count']

                logging.info(f"Long tp counts: {long_tp_counts}")
                logging.info(f"Short tp counts: {short_tp_counts}")

                # Place long TP order if there are no existing long TP orders
                if long_pos_qty > 0 and long_take_profit is not None and tp_order_counts['long_tp_count'] == 0:
                    self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

                # Place short TP order if there are no existing short TP orders
                if short_pos_qty > 0 and short_take_profit is not None and tp_order_counts['short_tp_count'] == 0:
                    self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)
                    
                # # Check if there's no existing take profit order before placing a new one
                # if open_tp_order_count == 0:
                #     if long_pos_qty > 0 and long_take_profit is not None:
                #         self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

                #     if short_pos_qty > 0 and short_take_profit is not None:
                #         self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)
                # else:
                #     logging.info(f"Skipping TP placement for {symbol} as it already exists.")

                current_time = datetime.now()
                
                # Check for long positions
                if current_time >= self.next_long_tp_update and long_take_profit is not None:
                    self.next_long_tp_update = self.update_take_profit_spread_bybit(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", next_tp_update=self.next_long_tp_update)

                # Check for short positions
                if current_time >= self.next_short_tp_update and short_take_profit is not None:
                    self.next_short_tp_update = self.update_take_profit_spread_bybit(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", next_tp_update=self.next_short_tp_update)


                self.cancel_entries_bybit(symbol, best_ask_price, moving_averages["ma_1m_3_high"], moving_averages["ma_5m_3_high"])
                self.cancel_stale_orders_bybit()

            elif can_open_new_position:
                open_symbols_count = len(open_symbols)

                if open_symbols_count < self.symbols_allowed:
                    long_dynamic_amount, short_dynamic_amount, min_qty = self.calculate_dynamic_amount(symbol, market_data, total_equity, best_ask_price, max_leverage)

                    short_pos_qty = position_data["short"]["qty"]
                    long_pos_qty = position_data["long"]["qty"]

                    should_short = self.short_trade_condition(best_ask_price, moving_averages["ma_3_high"])
                    should_long = self.long_trade_condition(best_bid_price, moving_averages["ma_3_low"])
                    
                    self.bybit_initial_entry_mm_5m(open_orders, symbol, trend, hma_trend, mfirsi_signal, five_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, should_long, should_short)

            symbol_data = {
                'symbol': symbol,
                'min_qty': min_qty,
                'current_price': current_price,
                'balance': total_equity,
                'available_bal': available_equity,
                'volume': one_minute_volume,
                'spread': five_minute_distance,
                'trend': trend,
                'long_pos_qty': long_pos_qty,
                'short_pos_qty': short_pos_qty,
                'long_upnl': long_upnl,
                'short_upnl': short_upnl,
                'long_cum_pnl': cum_realised_pnl_long,
                'short_cum_pnl': cum_realised_pnl_short,
                'long_pos_price': long_pos_price,
                'short_pos_price': short_pos_price
            }

            shared_symbols_data[symbol] = symbol_data

            if self.config.dashboard_enabled:
                data_to_save = copy.deepcopy(shared_symbols_data)
                with open(dashboard_path, "w") as f:
                    json.dump(data_to_save, f)
                self.update_shared_data(symbol_data, open_position_data, len(open_symbols))

            avg_daily_gain = self.bot_db.compute_average_daily_gain()
            logging.info(f"Average Daily Gain Percentage: {avg_daily_gain}%")


            time.sleep(15)
