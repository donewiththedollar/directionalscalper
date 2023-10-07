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

logging = Logger(logger_name="BybitOBStrength", filename="BybitOBStrength.log", stream=True)

class BybitOBStrength(Strategy):
    def __init__(self, exchange, manager, config, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        # Removed redundant initializations (they are already done in the parent class)
        self.last_health_check_time = time.time()
        self.health_check_interval = 600
        self.bot_db = BotDatabase(exchange=self.exchange)
        self.bot_db.create_tables_if_not_exists()
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
        logging.info(f"Initializing default values")
        # [Initialization code remains unchanged]
        # Initialize potentially missing values
        min_qty = None
        current_price = None
        total_equity = None
        available_equity = None
        one_minute_volume = None
        five_minute_volume = None
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

        # Check leverages only at startup
        self.current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        self.max_leverage = self.exchange.get_max_leverage_bybit(symbol)

        # Set the leverage to max if it's not already
        if self.current_leverage != self.max_leverage:
            logging.info(f"Current leverage is not at maximum. Setting leverage to maximum. Maximum is {self.max_leverage}")
            self.exchange.set_leverage_bybit(self.max_leverage, symbol)

        logging.info(f"Running for symbol (inside run_single_symbol method): {symbol}")

        quote_currency = "USDT"
        max_retries = 5
        retry_delay = 5
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        MaxAbsFundingRate = self.config.MaxAbsFundingRate
        # current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        # max_leverage = self.exchange.get_max_leverage_bybit(symbol)

        if self.config.dashboard_enabled:
            dashboard_path = os.path.join(self.config.shared_data_path, "shared_data.json")

        logging.info("Setting up exchange")
        self.exchange.setup_exchange_bybit(symbol)

        # logging.info("Setting leverage")
        # if current_leverage != max_leverage:
        #     logging.info(f"Current leverage is not at maximum. Setting leverage to maximum. Maximum is {max_leverage}")
        #     self.exchange.set_leverage_bybit(max_leverage, symbol)

        previous_five_minute_distance = None

        while True:
            rotator_symbols = self.manager.get_auto_rotate_symbols()
            open_position_data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)
            open_symbols = self.extract_symbols_from_positions_bybit(open_position_data)
            open_symbols = [symbol.replace("/", "") for symbol in open_symbols]
            api_data = self.manager.get_api_data(symbol)
            position_data = self.retry_api_call(self.exchange.get_positions_bybit, symbol)
            total_equity = self.retry_api_call(self.exchange.get_balance_bybit, quote_currency)
            available_equity = self.retry_api_call(self.exchange.get_available_balance_bybit, quote_currency)
            open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

            whitelist = self.config.whitelist
            blacklist = self.config.blacklist
            if symbol not in whitelist or symbol in blacklist:
                logging.info(f"Symbol {symbol} is no longer allowed based on whitelist/blacklist. Stopping operations for this symbol.")
                break

            one_minute_volume = api_data['1mVol']
            five_minute_volume = api_data['5mVol']
            five_minute_distance = api_data['5mSpread']
            trend = api_data['Trend']
            mfirsi_signal = api_data['MFI']
            funding_rate = api_data['Funding']
            hma_trend = api_data['HMA Trend']

            logging.info(f"One minute volume for {symbol} : {one_minute_volume}")
            logging.info(f"Five minute volume for {symbol} : {five_minute_volume}")
            logging.info(f"Five minute distance for {symbol} : {five_minute_distance}")

            funding_check = self.is_funding_rate_acceptable(symbol)
            logging.info(f"Funding check on {symbol} : {funding_check}")

            current_price = self.exchange.get_current_price(symbol)
            best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
            best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

            moving_averages = self.get_all_moving_averages(symbol)
            logging.info(f"Position data for {symbol} : {position_data}")
            
            logging.info(f"Open symbols: {open_symbols}")
            logging.info(f"HMA Current rotator symbols: {rotator_symbols}")
            symbols_to_manage = [s for s in open_symbols if s not in rotator_symbols]
            logging.info(f"Symbols to manage {symbols_to_manage}")
            
            logging.info(f"Open orders: {open_orders}")

            logging.info(f"Symbols allowed: {self.symbols_allowed}")

            trading_allowed = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, symbol)
            logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")

            # if symbol not in open_symbols and not trading_allowed:
            #     logging.warning(f"Skipping actions for symbol {symbol} as it's not tradable. due to trading allowed: {trading_allowed}")
            #     continue  # This will skip the rest of the loop for this iteration

            short_pos_qty = position_data["short"]["qty"]
            long_pos_qty = position_data["long"]["qty"]

            if symbol in rotator_symbols and (symbol in open_symbols or trading_allowed):

                logging.info(f"Rotator symbols: {rotator_symbols}")
                logging.info(f"Open symbols: {open_symbols}")

                logging.info(f"Long pos qty {long_pos_qty} for {symbol}")
                logging.info(f"Short pos qty {short_pos_qty} for {symbol}")

                short_liq_price = position_data["short"]["liq_price"]
                long_liq_price = position_data["long"]["liq_price"]

                self.set_position_leverage_long_bybit(symbol, long_pos_qty, total_equity, best_ask_price, self.max_leverage)
                self.set_position_leverage_short_bybit(symbol, short_pos_qty, total_equity, best_ask_price, self.max_leverage)


                long_dynamic_amount, short_dynamic_amount, min_qty = self.calculate_dynamic_amount_obstrength(symbol, total_equity, best_ask_price, self.max_leverage)
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


                logging.info(f"Short take profit: {short_take_profit}")
                logging.info(f"Long take profit: {long_take_profit}")

                should_short = self.short_trade_condition(best_ask_price, moving_averages["ma_3_high"])
                should_long = self.long_trade_condition(best_bid_price, moving_averages["ma_3_low"])
                should_add_to_short = False
                should_add_to_long = False

                if short_pos_price is not None:
                    should_add_to_short = short_pos_price < moving_averages["ma_6_low"] and self.short_trade_condition(best_ask_price, moving_averages["ma_6_high"])

                if long_pos_price is not None:
                    should_add_to_long = long_pos_price > moving_averages["ma_6_high"] and self.long_trade_condition(best_bid_price, moving_averages["ma_6_low"])

                open_tp_order_count = self.exchange.bybit.get_open_tp_order_count(symbol)

                logging.info(f"Open TP order count {open_tp_order_count}")

                current_time = time.time()
                if current_time - self.last_cancel_time >= self.spoofing_interval:
                    self.spoofing_active = True
                    self.spoofing_action(symbol, short_dynamic_amount, long_dynamic_amount)

                # Entries, and additional entries
                self.bybit_hedge_entry_maker_v3(open_orders, symbol, trend, mfirsi_signal, one_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_long, should_short, should_add_to_long, should_add_to_short)

                tp_order_counts = self.exchange.bybit.get_open_tp_order_count(symbol)

                long_tp_counts = tp_order_counts['long_tp_count']
                short_tp_counts = tp_order_counts['short_tp_count']

                logging.info(f"Long tp counts: {long_tp_counts}")
                logging.info(f"Short tp counts: {short_tp_counts}")

                logging.info(f"Long pos qty {long_pos_qty} for {symbol}")
                logging.info(f"Short pos qty {short_pos_qty} for {symbol}")

                logging.info(f"Long take profit {long_take_profit} for {symbol}")
                logging.info(f"Short take profit {short_take_profit} for {symbol}")

                # Place long TP order if there are no existing long TP orders
                if long_pos_qty > 0 and long_take_profit is not None and tp_order_counts['long_tp_count'] == 0:
                    self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

                # Place short TP order if there are no existing short TP orders
                if short_pos_qty > 0 and short_take_profit is not None and tp_order_counts['short_tp_count'] == 0:
                    self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)
                    
                current_time = datetime.now()

                # Check for long positions
                if current_time >= self.next_long_tp_update and long_take_profit is not None:
                    self.next_long_tp_update = self.update_take_profit_spread_bybit(
                        symbol=symbol, 
                        pos_qty=long_pos_qty, 
                        positionIdx=1, 
                        order_side="sell", 
                        next_tp_update=self.next_long_tp_update,
                        five_minute_distance=five_minute_distance, 
                        previous_five_minute_distance=previous_five_minute_distance
                    )

                # Check for short positions
                if current_time >= self.next_short_tp_update and short_take_profit is not None:
                    self.next_short_tp_update = self.update_take_profit_spread_bybit(
                        symbol=symbol, 
                        pos_qty=short_pos_qty, 
                        positionIdx=2, 
                        order_side="buy", 
                        next_tp_update=self.next_short_tp_update,
                        five_minute_distance=five_minute_distance, 
                        previous_five_minute_distance=previous_five_minute_distance
                    )


                self.cancel_entries_bybit(symbol, best_ask_price, moving_averages["ma_1m_3_high"], moving_averages["ma_5m_3_high"])
                self.cancel_stale_orders_bybit()

                # time.sleep(5)

            elif symbol not in rotator_symbols and symbol in open_symbols:
                logging.info(f"Managing open symbols not in rotator_symbols")

                short_pos_price = position_data["short"]["price"] if short_pos_qty > 0 else None
                long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

                logging.info(f"Symbol manager: Short pos price: {short_pos_price}")
                logging.info(f"Symbol manager: Long pos price: {long_pos_price}")

                logging.info(f"Symbol manager: Long pos qty: {long_pos_qty}")
                logging.info(f"Symbol manager: Short pos qty: {short_pos_qty}")

                tp_order_counts = self.exchange.bybit.get_open_tp_order_count(symbol)

                short_take_profit = None
                long_take_profit = None

                short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)
                previous_five_minute_distance = five_minute_distance

                logging.info(f"Short take profit for managed symbol {symbol}: {short_take_profit}")
                logging.info(f"Long take profit for managed symbol {symbol} : {long_take_profit}")

                self.set_position_leverage_long_bybit(symbol, long_pos_qty, total_equity, best_ask_price, self.max_leverage)
                self.set_position_leverage_short_bybit(symbol, short_pos_qty, total_equity, best_ask_price, self.max_leverage)

                long_dynamic_amount, short_dynamic_amount, min_qty = self.calculate_dynamic_amount_obstrength(symbol, total_equity, best_ask_price, self.max_leverage)

                # [Rest of the logic for symbols not in open_positions]
                # Place long TP order if there are no existing long TP orders
                if long_pos_qty > 0 and long_take_profit is not None and tp_order_counts['long_tp_count'] == 0:
                    self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

                # Place short TP order if there are no existing short TP orders
                if short_pos_qty > 0 and short_take_profit is not None and tp_order_counts['short_tp_count'] == 0:
                    self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)

                current_time = datetime.now()

                # Check for long positions
                if current_time >= self.next_long_tp_update and long_take_profit is not None:
                    self.next_long_tp_update = self.update_take_profit_spread_bybit(
                        symbol=symbol, 
                        pos_qty=long_pos_qty, 
                        positionIdx=1, 
                        order_side="sell", 
                        next_tp_update=self.next_long_tp_update,
                        five_minute_distance=five_minute_distance, 
                        previous_five_minute_distance=previous_five_minute_distance
                    )

                # Check for short positions
                if current_time >= self.next_short_tp_update and short_take_profit is not None:
                    self.next_short_tp_update = self.update_take_profit_spread_bybit(
                        symbol=symbol, 
                        pos_qty=short_pos_qty, 
                        positionIdx=2, 
                        order_side="buy", 
                        next_tp_update=self.next_short_tp_update,
                        five_minute_distance=five_minute_distance, 
                        previous_five_minute_distance=previous_five_minute_distance
                    )

                self.cancel_entries_bybit(symbol, best_ask_price, moving_averages["ma_1m_3_high"], moving_averages["ma_5m_3_high"])

                current_time = time.time()
                if current_time - self.last_cancel_time >= self.spoofing_interval:
                    self.spoofing_active = True
                    self.spoofing_action(symbol, short_dynamic_amount, long_dynamic_amount)

                time.sleep(15)

            elif symbol in rotator_symbols and symbol not in open_symbols and trading_allowed:

                logging.info(f"Managing new rotator symbol {symbol} not in open symbols")

                if trading_allowed:
                    logging.info(f"New position allowed {symbol}")
                    self.set_position_leverage_long_bybit(symbol, long_pos_qty, total_equity, best_ask_price, self.max_leverage)
                    self.set_position_leverage_short_bybit(symbol, short_pos_qty, total_equity, best_ask_price, self.max_leverage)

                    long_dynamic_amount, short_dynamic_amount, min_qty = self.calculate_dynamic_amount_obstrength(symbol, total_equity, best_ask_price, self.max_leverage)

                    short_pos_qty = position_data["short"]["qty"]
                    long_pos_qty = position_data["long"]["qty"]

                    should_short = self.short_trade_condition(best_ask_price, moving_averages["ma_3_high"])
                    should_long = self.long_trade_condition(best_bid_price, moving_averages["ma_3_low"])
                            
                    self.bybit_hedge_entry_maker_v3_initial_entry(open_orders, symbol, trend, mfirsi_signal, one_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, should_long, should_short)

                    # time.sleep(15)
                else:
                    logging.warning(f"Potential trade for {symbol} skipped as max symbol limit reached.")


            symbol_data = {
                'symbol': symbol,
                'min_qty': min_qty,
                'current_price': current_price,
                'balance': total_equity,
                'available_bal': available_equity,
                'volume': five_minute_volume,
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

