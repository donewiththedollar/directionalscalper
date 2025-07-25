import time
import json
import os
import copy
import pytz
import threading
import traceback
from threading import Thread, Lock
from datetime import datetime, timedelta

from directionalscalper.core.config_initializer import ConfigInitializer
from directionalscalper.core.strategies.bybit.bybit_strategy import BybitStrategy
from directionalscalper.core.exchanges.bybit import BybitExchange
from directionalscalper.core.strategies.logger import Logger
from live_table_manager import shared_symbols_data
from rate_limit import RateLimit
logging = Logger(logger_name="LinearGridBase", filename="LinearGridBase.log", stream=True)

symbol_locks = {}

class LinearGridBaseFutures(BybitStrategy):
    def __init__(self, exchange, manager, config, symbols_allowed=None, rotator_symbols_standardized=None, mfirsi_signal=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.rate_limiter = RateLimit(25, 1)
        self.general_rate_limiter = RateLimit(100, 1)
        self.order_rate_limiter = RateLimit(20, 1) 
        self.mfirsi_signal = mfirsi_signal
        self.is_order_history_populated = False
        self.last_health_check_time = time.time()
        self.health_check_interval = 600
        self.last_helper_order_cancel_time = 0
        self.helper_active = False
        self.helper_wall_size = 5
        self.helper_duration = 5
        self.helper_interval = 1
        self.running_long = False
        self.running_short = False
        self.last_known_equity = 0.0
        self.last_known_upnl = {}
        self.last_known_mas = {}
        ConfigInitializer.initialize_config_attributes(self, config)
        self._initialize_symbol_locks(rotator_symbols_standardized)

    def _initialize_symbol_locks(self, symbols):
        for symbol in symbols or []:
            standardized_symbol = symbol.upper()
            if standardized_symbol not in symbol_locks:
                symbol_locks[standardized_symbol] = {'long': threading.Lock(), 'short': threading.Lock()}

    def run(self, symbol, rotator_symbols_standardized=None, mfirsi_signal=None, action=None):
        standardized_symbol = symbol.upper()
        logging.info(f"[{standardized_symbol}] Entering run() with action='{action}'")

        # # ── EARLY EXIT FOR XGRIDT ────────────────────────────────────────────────
        # try:
        #     grid_behavior = self.config.linear_grid.get("grid_behavior", "")
        #     if grid_behavior == "xgridt":
        #         sig = self.generate_xgridt_signal(symbol)
        #         logging.info(f"[{symbol}] xgridt early-check: signal='{sig}' vs action='{action}'")
        #         if sig != action:
        #             logging.info(f"[{symbol}] xgridt → no matching signal, skipping run()")
        #             return
        #         # else—we’ll proceed under the lock
        # except Exception as e:
        #     logging.error(f"[{symbol}] xgridt early-check error: {e}", exc_info=True)
        #     return

        # ── LOCKING & DISPATCH ────────────────────────────────────────────────────
        current_thread_id = threading.get_ident()
        if standardized_symbol not in symbol_locks:
            symbol_locks[standardized_symbol] = {'long': threading.Lock(), 'short': threading.Lock()}

        lock = symbol_locks[standardized_symbol][action]
        if not lock.acquire(blocking=False):
            logging.info(f"[{symbol}] Failed to acquire lock for action='{action}' by thread {current_thread_id}")
            return

        logging.info(f"[{symbol}] Lock acquired for action='{action}' by thread {current_thread_id}")
        try:
            if action == "long":
                self.running_long = True
                self.run_single_symbol(standardized_symbol, rotator_symbols_standardized, mfirsi_signal, "long")
            elif action == "short":
                self.running_short = True
                self.run_single_symbol(standardized_symbol, rotator_symbols_standardized, mfirsi_signal, "short")
            else:
                logging.warning(f"[{symbol}] Unknown action '{action}' passed to run()")
        finally:
            lock.release()
            logging.info(f"[{symbol}] Lock released for action='{action}' by thread {current_thread_id}")

    def run_long_trades(self, symbol, rotator_symbols_standardized=None, mfirsi_signal=None):
        self.running_long = True
        self.run_single_symbol(symbol, rotator_symbols_standardized, mfirsi_signal, "long")

    def run_short_trades(self, symbol, rotator_symbols_standardized=None, mfirsi_signal=None):
        self.running_short = True
        self.run_single_symbol(symbol, rotator_symbols_standardized, mfirsi_signal, "short")

    def run_single_symbol(self, symbol, rotator_symbols_standardized=None, mfirsi_signal=None, action=None):
        try:

            current_signal = mfirsi_signal

            logging.info(f"Starting to process symbol: {symbol}")
            logging.info(f"Initializing default values for symbol: {symbol}")

            previous_long_pos_qty = 0
            previous_short_pos_qty = 0

            min_qty = None
            current_price = None
            total_equity = None
            available_equity = None
            one_minute_volume = None
            five_minute_volume = None
            one_minute_distance = None
            five_minute_distance = None
            ma_trend = 'neutral'  # Initialize with default value
            ema_trend = 'undefined'  # Initialize with default value
            long_pos_qty = 0
            short_pos_qty = 0
            long_upnl = 0
            short_upnl = 0
            cum_realised_pnl_long = 0
            cum_realised_pnl_short = 0
            long_pos_price = None
            short_pos_price = None
            
            # Track whether a position was ever opened for each side (to fix thread exit bug)
            long_position_ever_opened = False
            short_position_ever_opened = False
            
            # Track iterations to allow proper order placement before breaking
            iteration_count = 0

            # Initializing time trackers for less frequent API calls
            last_equity_fetch_time = 0
            equity_refresh_interval = 30  # 30 minutes in seconds

            fetched_total_equity = None

            # # Clean out orders
            # self.exchange.cancel_all_orders_for_symbol_bybit(symbol)
            # logging.info(f"Canceled all orders for {symbol}")

            # Check leverages only at startup
            self.current_leverage = self.exchange.get_current_max_leverage_bybit(symbol)
            self.max_leverage = self.exchange.get_current_max_leverage_bybit(symbol)

            logging.info(f"Current leverage: {self.current_leverage}")

            logging.info(f"Max leverage for {symbol}: {self.max_leverage}")

            # self.adjust_risk_parameters(exchange_max_leverage=self.max_leverage)

            self.exchange.set_leverage_bybit(self.max_leverage, symbol)
            self.exchange.set_symbol_to_cross_margin(symbol, self.max_leverage)

            logging.info(f"Running for symbol (inside run_single_symbol method): {symbol}")

            # Definitions
            max_retries = 5
            retry_delay = 5


            test_orders_enabled = self.config.test_orders_enabled
            

            levels = self.config.linear_grid['levels']
            strength = self.config.linear_grid['strength']
            long_mode = self.config.linear_grid['long_mode']
            short_mode = self.config.linear_grid['short_mode']
            reissue_threshold = self.config.linear_grid['reissue_threshold']
            buffer_percentage = self.config.linear_grid['buffer_percentage']
            enforce_full_grid = self.config.linear_grid['enforce_full_grid']
            auto_graceful_stop = self.config.linear_grid['auto_graceful_stop']
            target_coins_mode = self.config.linear_grid['target_coins_mode']
            initial_entry_buffer_pct = self.config.linear_grid['initial_entry_buffer_pct']
            min_buffer_percentage = self.config.linear_grid['min_buffer_percentage']
            max_buffer_percentage = self.config.linear_grid['max_buffer_percentage']
            wallet_exposure_limit_long = self.config.linear_grid['wallet_exposure_limit_long']
            wallet_exposure_limit_short = self.config.linear_grid['wallet_exposure_limit_short']
            max_qty_percent_long = self.config.linear_grid['max_qty_percent_long']
            max_qty_percent_short = self.config.linear_grid['max_qty_percent_short']
            max_usd_position_value_long = self.config.linear_grid.get('max_usd_position_value_long', None)
            max_usd_position_value_short = self.config.linear_grid.get('max_usd_position_value_short', None)
            one_symbol_optimization = self.config.linear_grid.get('one_symbol_optimization', False)
            min_outer_price_distance = self.config.linear_grid['min_outer_price_distance']
            min_outer_price_distance_long = self.config.linear_grid['min_outer_price_distance_long']
            min_outer_price_distance_short = self.config.linear_grid['min_outer_price_distance_short']
            max_outer_price_distance_long = self.config.linear_grid['max_outer_price_distance_long']
            max_outer_price_distance_short = self.config.linear_grid['max_outer_price_distance_short']
            graceful_stop_long = self.config.linear_grid['graceful_stop_long']
            graceful_stop_short = self.config.linear_grid['graceful_stop_short']
            additional_entries_from_signal = self.config.linear_grid['additional_entries_from_signal']
            stop_loss_long = self.config.linear_grid['stop_loss_long']
            stop_loss_short = self.config.linear_grid['stop_loss_short']
            stop_loss_enabled = self.config.linear_grid['stop_loss_enabled']
            auto_hedge_enabled = self.config.linear_grid['auto_hedge_enabled']
            auto_hedge_ratio = self.config.linear_grid['auto_hedge_ratio']
            auto_hedge_min_position_size = self.config.linear_grid['auto_hedge_min_position_size']
            auto_hedge_price_diff_threshold = self.config.linear_grid['auto_hedge_price_diff_threshold']
            disable_grid_on_hedge_side = self.config.linear_grid['disable_grid_on_hedge_side']
            hedge_with_grid = self.config.linear_grid['hedge_with_grid']
            forcibly_close_hedge = self.config.linear_grid['forcibly_close_hedge']
            auto_shift_hedge = self.config.linear_grid['auto_shift_hedge']
            side_with_grid = self.config.linear_grid['side_with_grid']

            grid_behavior = self.config.linear_grid.get('grid_behavior', 'infinite')
            drawdown_behavior = self.config.linear_grid.get('drawdown_behavior', 'maxqtypercent')

            hedge_stop_loss = self.config.linear_grid['hedge_stop_loss']

            dynamic_grid = self.config.linear_grid['dynamic_grid']

            enable_max_qty_stop_loss = self.config.linear_grid['enable_max_qty_stop_loss']

            max_qty_percent_long_stop_loss = self.config.linear_grid['max_qty_percent_long_stop_loss']
            max_qty_percent_short_stop_loss = self.config.linear_grid['max_qty_percent_short_stop_loss']

            # reissue_threshold_inposition = self.config.linear_grid['reissue_threshold_inposition']

            signal_flip_close_at_loss = self.config.linear_grid['signal_flip_close_at_loss']

            DRAWDOWN_CLOSE_THRESHOLD = self.config.linear_grid['drawdown_close_threshold']

            # Sticky size parameters
            sticky_size_enabled = self.config.linear_grid.get('sticky_size_enabled', False)
            sticky_size_aggressiveness = self.config.linear_grid.get('sticky_size_aggressiveness', 1.0)
            sticky_size_max_multiplier = self.config.linear_grid.get('sticky_size_max_multiplier', 5.0)
            sticky_size_target_profit = self.config.linear_grid.get('sticky_size_target_profit', 0.001)
            sticky_size_use_orderbook = self.config.linear_grid.get('sticky_size_use_orderbook', True)
            sticky_size_min_volume_ratio = self.config.linear_grid.get('sticky_size_min_volume_ratio', 0.2)

            volume_check = self.config.volume_check
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            upnl_threshold_pct = self.config.upnl_threshold_pct
            upnl_profit_pct = self.config.upnl_profit_pct
            max_upnl_profit_pct = self.config.max_upnl_profit_pct


            entry_during_autoreduce = self.config.entry_during_autoreduce

        
            # Funding
            MaxAbsFundingRate = self.config.MaxAbsFundingRate
            
            # Hedge ratio
            hedge_ratio = self.config.hedge_ratio

            # Hedge price diff
            price_difference_threshold = self.config.hedge_price_difference_threshold


                    
            logging.info("Setting up exchange")
            self.exchange.setup_exchange_bybit(symbol)

            previous_one_minute_distance = None
            previous_five_minute_distance = None

            since_timestamp = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)  # 24 hours ago in milliseconds
            # recent_trades = self.fetch_recent_trades_for_symbol(symbol, since=since_timestamp, limit=20)

            # #logging.info(f"Recent trades for {symbol} : {recent_trades}")

            # # Check if there are any trades in the last 24 hours
            # recent_activity = any(trade['timestamp'] >= since_timestamp for trade in recent_trades)
            # if recent_activity:
            #     logging.info(f"Recent trading activity detected for {symbol}")
            # else:
            #     logging.info(f"No recent trading activity for {symbol} in the last 24 hours")


            while self.running_long or self.running_short:
                iteration_count += 1
                logging.info(f"Trading {symbol} in while loop iteration {iteration_count} with long: {self.running_long}, short: {self.running_short}")

                # Example condition to stop the loop
                if action == "long" and not self.running_long:
                    logging.info(f"Killing thread for {symbol} because not running long")
                    break
                if action == "short" and not self.running_short:
                    logging.info(f"Killing thread for {symbol} because not running short")
                    break

                current_time = time.time()

                iteration_start_time = time.time()

                leverage_tiers = self.exchange.fetch_leverage_tiers(symbol)

                # if leverage_tiers:
                #     logging.info(f"Leverage tiers for {symbol}: {leverage_tiers}")
                # else:
                #     logging.error(f"Failed to fetch leverage tiers for {symbol}.")


                logging.info(f"Max USD value: {self.max_usd_value}")
            
                # Log which thread is running this part of the code
                thread_id = threading.get_ident()
                logging.info(f"[Thread ID: {thread_id}] In while true loop {symbol}")

                # Fetch open symbols every loop
                open_position_data = self.retry_api_call(self.exchange.get_all_open_positions_bybit)

                
                #logging.info(f"Open position data for {symbol}: {open_position_data}")

                position_details = {}

                for position in open_position_data:
                    info = position.get('info', {})
                    position_symbol = info.get('symbol', '').split(':')[0]  # Use a different variable name

                    # Ensure 'size', 'side', 'avgPrice', and 'liqPrice' keys exist in the info dictionary
                    if 'size' in info and 'side' in info and 'avgPrice' in info and 'liqPrice' in info:
                        size = float(info['size'])
                        side = info['side'].lower()
                        avg_price = float(info['avgPrice'])
                        liq_price = info.get('liqPrice', None)  # Default to None if not available

                        # Initialize the nested dictionary if the position_symbol is not already in position_details
                        if position_symbol not in position_details:
                            position_details[position_symbol] = {
                                'long': {'qty': 0, 'avg_price': 0, 'liq_price': None}, 
                                'short': {'qty': 0, 'avg_price': 0, 'liq_price': None}
                            }

                        # Update the quantities, average prices, and liquidation prices based on the side of the position
                        if side == 'buy':
                            position_details[position_symbol]['long']['qty'] += size
                            position_details[position_symbol]['long']['avg_price'] = avg_price
                            position_details[position_symbol]['long']['liq_price'] = liq_price
                        elif side == 'sell':
                            position_details[position_symbol]['short']['qty'] += size
                            position_details[position_symbol]['short']['avg_price'] = avg_price
                            position_details[position_symbol]['short']['liq_price'] = liq_price
                    else:
                        logging.warning(f"Missing required keys in position info for {position_symbol}")

                open_symbols = self.extract_symbols_from_positions_bybit(open_position_data)
                open_symbols = [symbol.replace("/", "") for symbol in open_symbols]
                logging.info(f"Open symbols: {open_symbols}")
                open_orders = self.retry_api_call(self.exchange.get_open_orders, symbol)

                #logging.info(f"Open symbols: {open_symbols}")

                #logging.info(f"Open orders: {open_orders}")

                market_data = self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)
                min_qty = float(market_data["min_qty"])

                # position_last_update_time = self.get_position_update_time(symbol)

                # logging.info(f"{symbol} last update time: {position_last_update_time}")

                # Fetch equity data
                fetched_total_equity = self.retry_api_call(self.exchange.get_futures_balance_bybit)

                logging.info(f"Fetched total equity: {fetched_total_equity}")

                # Attempt to convert fetched_total_equity to a float
                try:
                    fetched_total_equity = float(fetched_total_equity)
                except (ValueError, TypeError):
                    logging.warning(f"Fetched total equity could not be converted to float: {fetched_total_equity}. Resorting to last known equity.")
                    fetched_total_equity = None

                # Refresh equity if interval passed or fetched equity is 0.0
                if current_time - last_equity_fetch_time > equity_refresh_interval or fetched_total_equity == 0.0:
                    if fetched_total_equity is not None and fetched_total_equity > 0.0:
                        total_equity = fetched_total_equity
                        self.last_known_equity = total_equity  # Update the last known equity
                    else:
                        logging.warning("Failed to fetch valid total_equity or received 0.0. Using last known value.")
                        total_equity = self.last_known_equity  # Use last known equity

                    available_equity = self.retry_api_call(self.exchange.get_available_balance_bybit)
                    last_equity_fetch_time = current_time

                    logging.info(f"Total equity: {total_equity}")
                    logging.info(f"Available equity: {available_equity}")
                    
                    # Log the type of total_equity
                    logging.info(f"Type of total_equity: {type(total_equity)}")

                    # If total_equity is still None (which it shouldn't be), log an error and skip the iteration
                    if total_equity is None:
                        logging.error("This should not happen as total_equity should never be None. Skipping this iteration.")
                        time.sleep(10)  # wait for a short period before retrying
                        continue
                    
                blacklist = self.config.blacklist
                if symbol in blacklist:
                    logging.info(f"Symbol {symbol} is in the blacklist. Stopping operations for this symbol.")
                    break

                funding_check = self.is_funding_rate_acceptable(symbol)
                logging.info(f"Funding check on {symbol} : {funding_check}")

                current_price = self.exchange.get_current_price(symbol)

                order_book = self.exchange.get_orderbook(symbol)
                # best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                # best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                # Handling best ask price with fallback mechanism
                if 'asks' in order_book and len(order_book['asks']) > 0:
                    best_ask_price = order_book['asks'][0][0]
                    self.last_known_ask[symbol] = best_ask_price  # Update last known ask price
                else:
                    best_ask_price = self.last_known_ask.get(symbol)  # Use last known ask price
                    if best_ask_price is None:
                        logging.warning(f"Best ask price is not available for {symbol}. Defaulting to last known ask price, which is also None.")
                        best_ask_price = 0.0  # Default to 0.0 if None

                # Convert best_ask_price to float to ensure no type mismatches
                best_ask_price = float(best_ask_price)

                # Handling best bid price with fallback mechanism
                if 'bids' in order_book and len(order_book['bids']) > 0:
                    best_bid_price = order_book['bids'][0][0]
                    self.last_known_bid[symbol] = best_bid_price  # Update last known bid price
                else:
                    best_bid_price = self.last_known_bid.get(symbol)  # Use last known bid price
                    if best_bid_price is None:
                        logging.warning(f"Best bid price is not available for {symbol}. Defaulting to last known bid price, which is also None.")
                        best_bid_price = 0.0  # Default to 0.0 if None

                # Convert best_bid_price to float to ensure no type mismatches
                best_bid_price = float(best_bid_price)
                best_ask_price = float(best_ask_price)

                logging.info(f"Best bid price: {best_bid_price}")
                logging.info(f"Best ask price: {best_ask_price}")

                # Fetch moving averages with fallback mechanism
                try:
                    moving_averages = self.get_all_moving_averages(symbol)
                except ValueError as e:
                    logging.info(f"Failed to get new moving averages for {symbol}, using last known values: {e}")
                    moving_averages = self.last_known_mas.get(symbol, {})  # Continue using last known values if an error occurs


                # Ensure the moving averages are valid, fallback to last known if not present
                ma_3_high = moving_averages.get("ma_3_high", self.last_known_mas.get(symbol, {}).get("ma_3_high", 0.0))
                ma_3_low = moving_averages.get("ma_3_low", self.last_known_mas.get(symbol, {}).get("ma_3_low", 0.0))
                ma_6_high = moving_averages.get("ma_6_high", self.last_known_mas.get(symbol, {}).get("ma_6_high", 0.0))
                ma_6_low = moving_averages.get("ma_6_low", self.last_known_mas.get(symbol, {}).get("ma_6_low", 0.0))

                # Convert moving averages to float to ensure no type mismatches
                ma_3_high = float(ma_3_high)
                ma_3_low = float(ma_3_low)
                ma_6_high = float(ma_6_high)
                ma_6_low = float(ma_6_low)

                # Log warnings if any of the moving averages are missing
                if None in [ma_3_high, ma_3_low, ma_6_high, ma_6_low]:
                    logging.info(f"Missing moving averages for {symbol}. Using fallback values.")


                # moving_averages = self.get_all_moving_averages(symbol)

                logging.info(f"Open symbols: {open_symbols}")
                logging.info(f"Current rotator symbols: {rotator_symbols_standardized}")
                symbols_to_manage = [s for s in open_symbols if s not in rotator_symbols_standardized]
                logging.info(f"Symbols to manage {symbols_to_manage}")
                
                #logging.info(f"Open orders for {symbol}: {open_orders}")

                logging.info(f"Symbols allowed: {self.symbols_allowed}")

                trading_allowed = self.can_trade_new_symbol(open_symbols, self.symbols_allowed, symbol)
                logging.info(f"Checking trading for symbol {symbol}. Can trade: {trading_allowed}")
                logging.info(f"Symbol: {symbol}, In open_symbols: {symbol in open_symbols}, Trading allowed: {trading_allowed}")

                # self.adjust_risk_parameters()

                # self.initialize_symbol(symbol, total_equity, best_ask_price, self.max_leverage)

                # Log the currently initialized symbols
                logging.info(f"Initialized symbols: {list(self.initialized_symbols)}")

                # self.check_for_inactivity(long_pos_qty, short_pos_qty)

                # self.print_trade_quantities_once_bybit(symbol, total_equity, best_ask_price)

                logging.info(f"Rotator symbols standardized: {rotator_symbols_standardized}")

                symbol_precision = self.exchange.get_symbol_precision_bybit(symbol)

                logging.info(f"Symbol precision for {symbol} : {symbol_precision}")


                long_pos_qty = position_details.get(symbol, {}).get('long', {}).get('qty', 0)
                short_pos_qty = position_details.get(symbol, {}).get('short', {}).get('qty', 0)
                
                # Track if positions were ever opened (to fix thread exit bug)
                if long_pos_qty > 0:
                    long_position_ever_opened = True
                if short_pos_qty > 0:
                    short_position_ever_opened = True


                # Position side for symbol recently closed
                logging.info(f"Previous long pos qty for {symbol} : {previous_long_pos_qty}")
                logging.info(f"Previous short pos qty for {symbol} : {previous_short_pos_qty}")

                logging.info(f"Current long pos qty for {symbol} {long_pos_qty}")
                logging.info(f"Current short pos qty for {symbol} {short_pos_qty}")
            
                # if previous_long_pos_qty > 0 and long_pos_qty == 0:
                #     logging.info(f"Long position closed for {symbol}. Canceling long grid orders.")
                #     self.cancel_grid_orders(symbol, "buy")
                #     self.active_long_grids.discard(symbol)
                #     if short_pos_qty == 0:
                #         logging.info(f"No open positions for {symbol}. Removing from shared symbols data.")
                #         shared_symbols_data.pop(symbol, None)
                #     # don’t break here—keep the thread alive to reissue on neutral
                #     continue

                # elif previous_short_pos_qty > 0 and short_pos_qty == 0:
                #     logging.info(f"Short position closed for {symbol}. Canceling short grid orders.")
                #     self.cancel_grid_orders(symbol, "sell")
                #     self.active_short_grids.discard(symbol)
                #     if long_pos_qty == 0:
                #         logging.info(f"No open positions for {symbol}. Removing from shared symbols data.")
                #         shared_symbols_data.pop(symbol, None)
                #     # don’t break here either
                #     continue


                if previous_long_pos_qty > 0 and long_pos_qty == 0:
                    logging.info(f"Long position closed for {symbol}. Canceling long grid orders.")
                    if grid_behavior == "xgridt" or grid_behavior == "xgrid_highfrequency":
                        # Force cancel all grid orders immediately, bypassing timing protection
                        self.force_cancel_grid_orders(symbol, "buy")
                        logging.info(f"Force canceled {grid_behavior} orders for {symbol} long side (position closed)")
                    else:
                        self.cancel_grid_orders(symbol, "buy")
                    self.active_long_grids.discard(symbol)
                    if short_pos_qty == 0:
                        logging.info(f"No open positions for {symbol}. Removing from shared symbols data.")
                        shared_symbols_data.pop(symbol, None)
                    break  # Exit the while loop, thus ending the thread

                elif previous_short_pos_qty > 0 and short_pos_qty == 0:
                    logging.info(f"Short position closed for {symbol}. Canceling short grid orders.")
                    if grid_behavior == "xgridt" or grid_behavior == "xgrid_highfrequency":
                        # Force cancel all grid orders immediately, bypassing timing protection
                        self.force_cancel_grid_orders(symbol, "sell")
                        logging.info(f"Force canceled {grid_behavior} orders for {symbol} short side (position closed)")
                    else:
                        self.cancel_grid_orders(symbol, "sell")
                    self.active_short_grids.discard(symbol)
                    if long_pos_qty == 0:
                        logging.info(f"No open positions for {symbol}. Removing from shared symbols data.")
                        shared_symbols_data.pop(symbol, None)
                    break  # Exit the while loop, thus ending the thread


                try:
                    logging.info(f"Checking position inactivity")
                    # Check for position inactivity
                    inactive_pos_time_threshold = 60 
                    if self.check_position_inactivity(symbol, inactive_pos_time_threshold, long_pos_qty, short_pos_qty, previous_long_pos_qty, previous_short_pos_qty):
                        logging.info(f"No open positions for {symbol} in the last {inactive_pos_time_threshold} seconds. Terminating the thread.")
                        shared_symbols_data.pop(symbol, None)
                        break
                except Exception as e:
                    logging.info(f"Exception caught in check_position_inactivity {e}")
       
                # Optionally, break out of the loop if all trading sides are closed
                if not self.running_long and not self.running_short:
                    shared_symbols_data.pop(symbol, None)
                    self.cancel_grid_orders(symbol, "buy")
                    self.cancel_grid_orders(symbol, "sell")
                    self.active_long_grids.discard(symbol)
                    self.active_short_grids.discard(symbol)
                    self.cleanup_before_termination(symbol)
                    logging.info("Both long and short operations have terminated. Exiting the loop.")
                    break
                
                # Determine if positions have just been closed
                if previous_long_pos_qty > 0 and long_pos_qty == 0:
                    logging.info(f"All long positions for {symbol} were recently closed. Checking for inactivity.")
                    inactive_long = True
                else:
                    inactive_long = False

                if previous_short_pos_qty > 0 and short_pos_qty == 0:
                    logging.info(f"All short positions for {symbol} were recently closed. Checking for inactivity.")
                    inactive_short = True
                else:
                    inactive_short = False

                # Update previous quantities for the next iteration
                previous_long_pos_qty = long_pos_qty
                previous_short_pos_qty = short_pos_qty

                if not self.running_long and not self.running_short:
                    logging.info("Both long and short operations have ended. Preparing to exit loop.")
                    shared_symbols_data.pop(symbol, None)  # Remove the symbol from shared_symbols_data

                # Reduced sleep for xgrid high-frequency trading
                if self.config.linear_grid.get("grid_behavior") == "xgridt":
                    time.sleep(0.1)
                else:
                    time.sleep(2)

                # If the symbol is in rotator_symbols and either it's already being traded or trading is allowed.
                if symbol in rotator_symbols_standardized or (symbol in open_symbols or trading_allowed): # and instead of or

                    # Fetch the API data
                    api_data = self.manager.get_api_data(symbol)

                    # Extract the required metrics using the new implementation
                    metrics = self.manager.extract_metrics(api_data, symbol)

                    # Assign the metrics to the respective variables
                    one_minute_volume = metrics['1mVol']
                    five_minute_volume = metrics['5mVol']
                    one_minute_distance = metrics['1mSpread']
                    five_minute_distance = metrics['5mSpread']
                    ma_trend = metrics['MA Trend']
                    ema_trend = metrics['EMA Trend']

                    #mfirsi_signal = metrics['MFI']
                    #mfirsi_signal = self.get_mfirsi_ema(symbol, limit=100, lookback=5, ema_period=5)
                    #mfirsi_signal = self.get_mfirsi_ema_secondary_ema(symbol, limit=100, lookback=1, ema_period=5, secondary_ema_period=3)

                    # mfirsi_signal = self.exchange.generate_l_signals(symbol)



                    funding_rate = metrics['Funding']
                    hma_trend = metrics['HMA Trend']
                    eri_trend = metrics['ERI Trend']

                    logging.info(f"{symbol} ERI Trend: {eri_trend}")

                    logging.info(f"{symbol} MFIRSI Signal: {mfirsi_signal}")

                    fivemin_top_signal = metrics['Top Signal 5m']
                    fivemin_bottom_signal = metrics['Bottom Signal 5m']

                    onemin_top_signal = metrics['Top Signal 1m']
                    onemin_bottom_signal = metrics['Bottom Signal 1m']

                    position_data = self.retry_api_call(self.exchange.get_positions_bybit, symbol)

                    long_liquidation_price = position_details.get(symbol, {}).get('long', {}).get('liq_price')
                    short_liquidation_price = position_details.get(symbol, {}).get('short', {}).get('liq_price')

                    logging.info(f"Long liquidation price for {symbol}: {long_liquidation_price}")
                    logging.info(f"Short liquidation price for {symbol}: {short_liquidation_price}")

                    logging.info(f"Rotator symbol trading: {symbol}")
                                
                    logging.info(f"Rotator symbols: {rotator_symbols_standardized}")
                    logging.info(f"Open symbols: {open_symbols}")

                    logging.info(f"Long pos qty {long_pos_qty} for {symbol}")
                    logging.info(f"Short pos qty {short_pos_qty} for {symbol}")

                    # short_liq_price = position_data["short"]["liq_price"]
                    # long_liq_price = position_data["long"]["liq_price"]


                    # Adjust risk parameters based on the maximum leverage allowed by the exchange
                    # self.adjust_risk_parameters(exchange_max_leverage=self.max_leverage)

                    # Calculate dynamic entry sizes for long and short positions
                    long_dynamic_amount, short_dynamic_amount = self.calculate_dynamic_amounts_notional(
                        symbol=symbol,
                        total_equity=total_equity,
                        best_ask_price=best_ask_price,
                        best_bid_price=best_bid_price,
                        wallet_exposure_limit_long=wallet_exposure_limit_long,
                        wallet_exposure_limit_short=wallet_exposure_limit_short
                    )

                    logging.info(f"Long dynamic amount: {long_dynamic_amount} for {symbol}")
                    logging.info(f"Short dynamic amount: {short_dynamic_amount} for {symbol}")

                    long_dynamic_amount_helper, short_dynamic_amount_helper = self.calculate_dynamic_amounts_notional_nowelimit(
                        symbol=symbol,
                        total_equity=total_equity,
                        best_bid_price=best_bid_price,
                        best_ask_price=best_ask_price
                    )

                    logging.info(f"Long dynamic amount helper: {long_dynamic_amount} for {symbol}")
                    logging.info(f"Short dynamic amount helper: {short_dynamic_amount} for {symbol}")

                    cum_realised_pnl_long = position_data["long"]["cum_realised"]
                    cum_realised_pnl_short = position_data["short"]["cum_realised"]

                    # Get the average price for long and short positions
                    long_pos_price = position_details.get(symbol, {}).get('long', {}).get('avg_price', None)
                    short_pos_price = position_details.get(symbol, {}).get('short', {}).get('avg_price', None)

                    short_take_profit = None
                    long_take_profit = None

                    # Calculate take profit for short and long positions using quickscalp method
                    short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
                    long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

                    short_stop_loss = None
                    long_stop_loss = None

                    initial_short_stop_loss = None
                    initial_long_stop_loss = None

                    # Failsafe method disabled for xgrid + momentum_scalping
                    try:
                        pass  # Failsafe functionality removed
                    except Exception as e:
                        logging.info(f"Failsafe failed: {e}")

                    # Auto-reduce logic disabled for xgrid + momentum_scalping
                    try:
                        pass  # Auto-reduce functionality removed
                    except Exception as e:
                        logging.info(f"Auto-reduce exception caught {e}")


                    # short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, one_minute_distance, previous_one_minute_distance, short_take_profit, long_take_profit)
                    #short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)
                    previous_five_minute_distance = five_minute_distance

                    previous_one_minute_distance = one_minute_distance


                    logging.info(f"Short take profit for {symbol}: {short_take_profit}")
                    logging.info(f"Long take profit for {symbol}: {long_take_profit}")

                    # Handling best ask price with fallback and type conversion
                    if 'asks' in order_book and len(order_book['asks']) > 0:
                        best_ask_price = float(order_book['asks'][0][0])
                        self.last_known_ask[symbol] = best_ask_price  # Update last known ask price
                    else:
                        best_ask_price = self.last_known_ask.get(symbol, 0.0)  # Fallback to 0.0 if not available
                        if best_ask_price == 0.0:
                            logging.warning(f"Best ask price is not available for {symbol}. Defaulting to 0.0.")

                    # Handling best bid price with fallback and type conversion
                    if 'bids' in order_book and len(order_book['bids']) > 0:
                        best_bid_price = float(order_book['bids'][0][0])
                        self.last_known_bid[symbol] = best_bid_price  # Update last known bid price
                    else:
                        best_bid_price = self.last_known_bid.get(symbol, 0.0)  # Fallback to 0.0 if not available
                        if best_bid_price == 0.0:
                            logging.warning(f"Best bid price is not available for {symbol}. Defaulting to 0.0.")

                    # Ensure valid decisions on whether to short or long based on conditions
                    should_short = self.short_trade_condition(best_ask_price, ma_3_high)
                    should_long = self.long_trade_condition(best_bid_price, ma_3_low)

                    # Determine if additional shorts should be added
                    should_add_to_short = False
                    if short_pos_price is not None:
                        should_add_to_short = short_pos_price < ma_6_low and self.short_trade_condition(best_ask_price, ma_6_high)

                    # Determine if additional longs should be added
                    should_add_to_long = False
                    if long_pos_price is not None:
                        should_add_to_long = long_pos_price > ma_6_high and self.long_trade_condition(best_bid_price, ma_6_low)

                    logging.info(f"Five minute volume for {symbol} : {five_minute_volume}")
                        
                    # historical_data = self.fetch_historical_data(
                    #     symbol,
                    #     timeframe='4h'
                    # )

                    # one_hour_atr_value = self.calculate_atr(historical_data)

                    # logging.info(f"ATR for {symbol} : {one_hour_atr_value}")

                    tp_order_counts = self.exchange.get_open_tp_order_count(open_orders)

                    logging.info(f"Open TP order count {tp_order_counts}")

                    # Check for long position
                    if long_pos_qty > 0:
                        try:
                            unrealized_pnl = self.exchange.fetch_unrealized_pnl(symbol)
                            long_upnl = unrealized_pnl.get('long')
                            self.last_known_upnl[symbol] = self.last_known_upnl.get(symbol, {})
                            self.last_known_upnl[symbol]['long'] = long_upnl  # Store the last known long uPNL
                            logging.info(f"Long UPNL for {symbol}: {long_upnl}")
                        except Exception as e:
                            # Fallback to last known uPNL if an exception occurs
                            long_upnl = self.last_known_upnl.get(symbol, {}).get('long', 0.0)
                            logging.info(f"Exception fetching Long UPNL for {symbol}: {e}. Using last known UPNL: {long_upnl}")

                    # Check for short position
                    if short_pos_qty > 0:
                        try:
                            unrealized_pnl = self.exchange.fetch_unrealized_pnl(symbol)
                            short_upnl = unrealized_pnl.get('short')
                            self.last_known_upnl[symbol] = self.last_known_upnl.get(symbol, {})
                            self.last_known_upnl[symbol]['short'] = short_upnl  # Store the last known short uPNL
                            logging.info(f"Short UPNL for {symbol}: {short_upnl}")
                        except Exception as e:
                            # Fallback to last known uPNL if an exception occurs
                            short_upnl = self.last_known_upnl.get(symbol, {}).get('short', 0.0)
                            logging.info(f"Exception fetching Short UPNL for {symbol}: {e}. Using last known UPNL: {short_upnl}")

                    long_tp_counts = tp_order_counts['long_tp_count']
                    short_tp_counts = tp_order_counts['short_tp_count']

                    # ────────────────────────────────────────────────────────────────
                    # 1) Run one full grid pass
                    try:
                        self.lineargrid_base(
                            symbol,
                            open_symbols,
                            total_equity,
                            long_pos_price,
                            short_pos_price,
                            long_pos_qty,
                            short_pos_qty,
                            levels,
                            min_outer_price_distance,
                            min_outer_price_distance_long,
                            min_outer_price_distance_short,
                            max_outer_price_distance_long,
                            max_outer_price_distance_short,
                            reissue_threshold,
                            wallet_exposure_limit_long,
                            wallet_exposure_limit_short,
                            long_mode,
                            short_mode,
                            initial_entry_buffer_pct,
                            min_buffer_percentage,
                            max_buffer_percentage,
                            self.symbols_allowed,
                            enforce_full_grid,
                            current_signal,
                            upnl_profit_pct,
                            max_upnl_profit_pct,
                            tp_order_counts,
                            entry_during_autoreduce,
                            max_qty_percent_long,
                            max_qty_percent_short,
                            max_usd_position_value_long,
                            max_usd_position_value_short,
                            graceful_stop_long,
                            graceful_stop_short,
                            additional_entries_from_signal,
                            open_position_data,
                            drawdown_behavior,
                            grid_behavior,
                            stop_loss_long,
                            stop_loss_short,
                            stop_loss_enabled,
                            auto_hedge_enabled,
                            auto_hedge_ratio,
                            auto_hedge_min_position_size,
                            auto_hedge_price_diff_threshold,
                            disable_grid_on_hedge_side,
                            hedge_with_grid,
                            forcibly_close_hedge,
                            auto_shift_hedge,
                            side_with_grid,
                            hedge_stop_loss,
                            enable_max_qty_stop_loss,
                            max_qty_percent_long_stop_loss,
                            max_qty_percent_short_stop_loss,
                            dynamic_grid,
                            signal_flip_close_at_loss,
                            DRAWDOWN_CLOSE_THRESHOLD,
                            one_symbol_optimization,
                            sticky_size_enabled,
                            sticky_size_aggressiveness,
                            sticky_size_max_multiplier,
                            sticky_size_target_profit,
                            sticky_size_use_orderbook,
                            sticky_size_min_volume_ratio
                        )
                    except Exception as e:
                        logging.info(f"Something went wrong in lineargrid_base: {e}")

                    # ────────────────────────────────────────────────────────────────
                    # 2) Build and publish the table row *before* any break
                    symbol_data = {
                        'symbol':         symbol,
                        'min_qty':        min_qty,
                        'current_price':  current_price,
                        'balance':        total_equity,
                        'available_bal':  available_equity,
                        'volume':         five_minute_volume,
                        'spread':         five_minute_distance,
                        'ma_trend':       ma_trend,
                        'ema_trend':      ema_trend,
                        'long_pos_qty':   long_pos_qty,
                        'short_pos_qty':  short_pos_qty,
                        'long_upnl':      long_upnl,
                        'short_upnl':     short_upnl,
                        'long_cum_pnl':   cum_realised_pnl_long,
                        'short_cum_pnl':  cum_realised_pnl_short,
                        'long_pos_price': long_pos_price,
                        'short_pos_price':short_pos_price
                    }
                    shared_symbols_data[symbol] = symbol_data
                    self.update_shared_data(symbol_data, open_position_data, len(open_symbols))

                    # ────────────────────────────────────────────────────────────────
                    # 3) xgridt/xgrid_highfrequency mode: handle individual side exits for fresh signal entry
                    if self.config.linear_grid.get("grid_behavior") in ["xgridt", "xgrid_highfrequency"]:
                        # If running specific action and that side WAS opened but is NOW closed, exit to allow fresh entry
                        if action == "long" and long_position_ever_opened and long_pos_qty == 0:
                            logging.info(f"[{symbol}] {grid_behavior} mode: long position was opened but now closed, canceling orders and exiting long thread for fresh signal entry")
                            self.force_cancel_grid_orders(symbol, "buy")
                            self.active_long_grids.discard(symbol)
                            break
                        elif action == "short" and short_position_ever_opened and short_pos_qty == 0:
                            logging.info(f"[{symbol}] {grid_behavior} mode: short position was opened but now closed, canceling orders and exiting short thread for fresh signal entry")
                            self.force_cancel_grid_orders(symbol, "sell")
                            self.active_short_grids.discard(symbol)
                            break
                        # If both sides were opened but are now closed, exit completely
                        elif long_position_ever_opened and short_position_ever_opened and long_pos_qty == 0 and short_pos_qty == 0:
                            logging.info(f"[{symbol}] {grid_behavior} mode: all positions were opened but now closed, canceling all orders and exiting thread for fresh signal entry")
                            self.force_cancel_grid_orders(symbol, "buy")
                            self.force_cancel_grid_orders(symbol, "sell")
                            self.active_long_grids.discard(symbol)
                            self.active_short_grids.discard(symbol)
                            shared_symbols_data.pop(symbol, None)
                            break
                        # NEW: In xgridt mode with single-sided action, allow few iterations then exit if no position opened
                        elif iteration_count > 10:  # Allow at least 10 iterations for order placement and fills
                            if action == "long" and not long_position_ever_opened and long_pos_qty == 0:
                                # If we're running long action but no long position ever opened after sufficient iterations, exit
                                logging.info(f"[{symbol}] {grid_behavior} mode: long action but no long position opened after {iteration_count} iterations, exiting long thread for fresh signal entry")
                                self.force_cancel_grid_orders(symbol, "buy")
                                self.active_long_grids.discard(symbol)
                                break
                            elif action == "short" and not short_position_ever_opened and short_pos_qty == 0:
                                # If we're running short action but no short position ever opened after sufficient iterations, exit
                                logging.info(f"[{symbol}] {grid_behavior} mode: short action but no short position opened after {iteration_count} iterations, exiting short thread for fresh signal entry")
                                self.force_cancel_grid_orders(symbol, "sell")
                                self.active_short_grids.discard(symbol)
                                break
                        else:
                            if long_pos_qty == 0 and short_pos_qty == 0:
                                logging.info(f"[{symbol}] {grid_behavior} mode: no positions opened yet, continuing to monitor for fills (long_ever_opened: {long_position_ever_opened}, short_ever_opened: {short_position_ever_opened})")
                            else:
                                logging.info(f"[{symbol}] {grid_behavior} mode: completed one grid pass, continuing to monitor open positions (long: {long_pos_qty}, short: {short_pos_qty})")

                    logging.info(f"Long tp counts: {long_tp_counts}")
                    logging.info(f"Short tp counts: {short_tp_counts}")

                    logging.info(f"Long pos qty {long_pos_qty} for {symbol}")
                    logging.info(f"Short pos qty {short_pos_qty} for {symbol}")

                    logging.info(f"Long take profit {long_take_profit} for {symbol}")
                    logging.info(f"Short take profit {short_take_profit} for {symbol}")

                    logging.info(f"Long TP order count for {symbol} is {tp_order_counts['long_tp_count']}")
                    logging.info(f"Short TP order count for {symbol} is {tp_order_counts['short_tp_count']}")

                    current_latest_time = datetime.now()
                    logging.info(f"Current time: {current_latest_time}")
                    logging.info(f"Next long TP update time: {self.next_long_tp_update}")
                    logging.info(f"Next short TP update time: {self.next_short_tp_update}")

                    # Calculate take profit for short and long positions using quickscalp method
                    short_take_profit = self.calculate_quickscalp_short_take_profit_dynamic_distance(short_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)
                    long_take_profit = self.calculate_quickscalp_long_take_profit_dynamic_distance(long_pos_price, symbol, min_upnl_profit_pct=upnl_profit_pct, max_upnl_profit_pct=max_upnl_profit_pct)

                    # Update TP for long position
                    if long_pos_qty > 0:
                        new_long_tp_min, new_long_tp_max = self.calculate_quickscalp_long_take_profit_dynamic_distance(
                            long_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                        )
                        if new_long_tp_min is not None and new_long_tp_max is not None:
                            self.next_long_tp_update = self.update_quickscalp_tp_dynamic(
                                symbol=symbol,
                                pos_qty=long_pos_qty,
                                upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                                max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                                short_pos_price=None,  # Not relevant for long TP settings
                                long_pos_price=long_pos_price,
                                positionIdx=1,
                                order_side="sell",
                                last_tp_update=self.next_long_tp_update,
                                tp_order_counts=tp_order_counts,
                                open_orders=open_orders
                            )

                    if short_pos_qty > 0:
                        new_short_tp_min, new_short_tp_max = self.calculate_quickscalp_short_take_profit_dynamic_distance(
                            short_pos_price, symbol, upnl_profit_pct, max_upnl_profit_pct
                        )
                        if new_short_tp_min is not None and new_short_tp_max is not None:
                            self.next_short_tp_update = self.update_quickscalp_tp_dynamic(
                                symbol=symbol,
                                pos_qty=short_pos_qty,
                                upnl_profit_pct=upnl_profit_pct,  # Minimum desired profit percentage
                                max_upnl_profit_pct=max_upnl_profit_pct,  # Maximum desired profit percentage for scaling
                                short_pos_price=short_pos_price,
                                long_pos_price=None,  # Not relevant for short TP settings
                                positionIdx=2,
                                order_side="buy",
                                last_tp_update=self.next_short_tp_update,
                                tp_order_counts=tp_order_counts,
                                open_orders=open_orders
                            )
                            

                    if self.test_orders_enabled and current_time - self.last_helper_order_cancel_time >= self.helper_interval:
                        if symbol in open_symbols:
                            self.helper_active = True
                            self.helperv2(symbol, short_dynamic_amount_helper, long_dynamic_amount_helper)
                        else:
                            logging.info(f"Skipping test orders for {symbol} as it's not in open symbols list.")
                            
                    # # Check if the symbol should terminate
                    # if self.should_terminate_full(symbol, current_time, previous_long_pos_qty, long_pos_qty, previous_short_pos_qty, short_pos_qty):
                    #     self.cleanup_before_termination(symbol)
                    #     break  # Exit the while loop, thus ending the thread

                    # Check to terminate the loop if both long and short are no longer running
                    if not self.running_long and not self.running_short:
                        logging.info("Both long and short operations have ended. Preparing to exit loop.")
                        shared_symbols_data.pop(symbol, None)  # Remove the symbol from shared symbols data
                        # This will cause the loop condition to fail naturally without a break, making the code flow cleaner
                
                    # self.cancel_entries_bybit(symbol, best_ask_price, moving_averages["ma_1m_3_high"], moving_averages["ma_5m_3_high"])
                    # self.cancel_stale_orders_bybit(symbol)
                    
                # Reduced sleep for xgrid high-frequency trading
                if self.config.linear_grid.get("grid_behavior") == "xgridt":
                    time.sleep(0.5)
                else:
                    time.sleep(5)

                dashboard_path = os.path.join(self.config.shared_data_path, "shared_data.json")
                
                symbol_data = {
                    'symbol': symbol,
                    'min_qty': min_qty,
                    'current_price': current_price,
                    'balance': total_equity,
                    'available_bal': available_equity,
                    'volume': five_minute_volume,
                    'spread': five_minute_distance,
                    'ma_trend': ma_trend,
                    'ema_trend': ema_trend,
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
                    try:
                        data_to_save = copy.deepcopy(shared_symbols_data)
                        with open(dashboard_path, "w") as f:
                            json.dump(data_to_save, f)
                        self.update_shared_data(symbol_data, open_position_data, len(open_symbols))
                    except Exception as e:
                        logging.info(f"Dashboard saving is not working properly {e}")

                if self.config.dashboard_enabled:
                    try:
                        dashboard_path = os.path.join(self.config.shared_data_path, "shared_data.json")
                        logging.info(f"Dashboard path: {dashboard_path}")

                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
                        logging.info(f"Directory created: {os.path.dirname(dashboard_path)}")

                        if os.path.exists(dashboard_path):
                            with open(dashboard_path, "r") as file:
                                # Read or process file data
                                data = json.load(file)
                                logging.info("Loaded existing data from shared_data.json")
                        else:
                            logging.warning("shared_data.json does not exist. Creating a new file.")
                            data = {}  # Initialize data as an empty dictionary

                        # Save the updated data to the JSON file
                        with open(dashboard_path, "w") as file:
                            json.dump(data, file)
                            logging.info("Data saved to shared_data.json")

                    except FileNotFoundError:
                        logging.info(f"File not found: {dashboard_path}")
                        # Handle the absence of the file, e.g., by creating it or using default data
                    except IOError as e:
                        logging.info(f"I/O error occurred: {e}")
                        # Handle other I/O errors
                    except Exception as e:
                        logging.info(f"An unexpected error occurred in saving json: {e}")
                        
                iteration_end_time = time.time()  # Record the end time of the iteration
                iteration_duration = iteration_end_time - iteration_start_time
                logging.info(f"Iteration for symbol {symbol} took {iteration_duration:.2f} seconds")

                # # ────────────────────────────────────────────────────────────────
                # # after we've updated the dashboard for this symbol,
                # # if we're in xgridt mode we can now safely exit
                # if self.config.linear_grid.get("grid_behavior") == "xgridt":
                #     logging.info(f"[{symbol}] xgridt mode: final table update done → exiting thread")
                #     break
                # # ────────────────────────────────────────────────────────────────

                # Reduced sleep for xgrid high-frequency trading
                if self.config.linear_grid.get("grid_behavior") == "xgridt":
                    time.sleep(0.3)
                else:
                    time.sleep(3)
        except Exception as e:
            traceback_info = traceback.format_exc()  # Get the full traceback
            logging.info(f"Exception caught in quickscalp strategy '{symbol}': {e}\nTraceback:\n{traceback_info}")