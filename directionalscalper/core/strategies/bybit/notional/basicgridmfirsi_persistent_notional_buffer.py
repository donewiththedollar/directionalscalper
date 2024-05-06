import time
import json
import os
import copy
import pytz
import threading
import traceback
from threading import Thread, Lock
from datetime import datetime, timedelta

from directionalscalper.core.strategies.bybit.bybit_strategy import BybitStrategy
from directionalscalper.core.exchanges.bybit import BybitExchange
from directionalscalper.core.strategies.logger import Logger
from live_table_manager import shared_symbols_data
logging = Logger(logger_name="BybitBasicGridBuffered", filename="BybitBasicGridBuffered.log", stream=True)

symbol_locks = {}

class BybitBasicGridBuffered(BybitStrategy):
    def __init__(self, exchange, manager, config, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)
        self.is_order_history_populated = False
        self.last_health_check_time = time.time()
        self.health_check_interval = 600
        self.last_long_tp_update = datetime.now()
        self.last_short_tp_update = datetime.now()
        self.next_long_tp_update = datetime.now() - timedelta(seconds=1)
        self.next_short_tp_update = datetime.now() - timedelta(seconds=1)
        self.last_helper_order_cancel_time = 0
        self.helper_active = False
        self.helper_wall_size = 5
        self.helper_duration = 5
        self.helper_interval = 1
        try:
            self.wallet_exposure_limit = self.config.wallet_exposure_limit
            self.user_defined_leverage_long = self.config.user_defined_leverage_long
            self.user_defined_leverage_short = self.config.user_defined_leverage_short
            self.levels = self.config.linear_grid['levels']
            self.strength = self.config.linear_grid['strength']
            self.outer_price_distance = self.config.linear_grid['outer_price_distance']
            self.long_mode = self.config.linear_grid['long_mode']
            self.short_mode = self.config.linear_grid['short_mode']
            self.reissue_threshold = self.config.linear_grid['reissue_threshold']
            self.buffer_percentage = self.config.linear_grid['buffer_percentage']
            self.enforce_full_grid = self.config.linear_grid['enforce_full_grid']
            self.min_buffer_percentage = self.config.linear_grid['min_buffer_percentage']
            self.max_buffer_percentage = self.config.linear_grid['max_buffer_percentage']
            # self.reissue_threshold_inposition = self.config.linear_grid['reissue_threshold_inposition']
            self.upnl_threshold_pct = self.config.upnl_threshold_pct
            self.volume_check = self.config.volume_check
            self.max_usd_value = self.config.max_usd_value
            self.blacklist = self.config.blacklist
            try:
                self.test_orders_enabled = self.config.test_orders_enabled
                # Initialization of other attributes
            except AttributeError as e:
                logging.error(f"Failed to initialize attributes from config: {e}")
            self.upnl_profit_pct = self.config.upnl_profit_pct
            self.stoploss_enabled = self.config.stoploss_enabled
            self.stoploss_upnl_pct = self.config.stoploss_upnl_pct
            self.liq_stoploss_enabled = self.config.liq_stoploss_enabled
            self.liq_price_stop_pct = self.config.liq_price_stop_pct
            self.auto_reduce_enabled = self.config.auto_reduce_enabled
            self.auto_reduce_start_pct = self.config.auto_reduce_start_pct
            self.auto_reduce_maxloss_pct = self.config.auto_reduce_maxloss_pct
            self.entry_during_autoreduce = self.config.entry_during_autoreduce
            self.auto_reduce_marginbased_enabled = self.config.auto_reduce_marginbased_enabled
            self.auto_reduce_wallet_exposure_pct = self.config.auto_reduce_wallet_exposure_pct
            self.percentile_auto_reduce_enabled = self.config.percentile_auto_reduce_enabled
            self.max_pos_balance_pct = self.config.max_pos_balance_pct
            self.auto_leverage_upscale = self.config.auto_leverage_upscale
        except AttributeError as e:
            logging.error(f"Failed to initialize attributes from config: {e}")


    def run(self, symbol, rotator_symbols_standardized=None):
        try:
            standardized_symbol = symbol.upper()  # Standardize the symbol name
            logging.info(f"Standardized symbol: {standardized_symbol}")
            current_thread_id = threading.get_ident()

            if standardized_symbol not in symbol_locks:
                symbol_locks[standardized_symbol] = threading.Lock()

            if symbol_locks[standardized_symbol].acquire(blocking=False):
                logging.info(f"Lock acquired for symbol {standardized_symbol} by thread {current_thread_id}")
                try:
                    self.run_single_symbol(standardized_symbol, rotator_symbols_standardized)
                finally:
                    symbol_locks[standardized_symbol].release()
                    logging.info(f"Lock released for symbol {standardized_symbol} by thread {current_thread_id}")
            else:
                logging.info(f"Failed to acquire lock for symbol: {standardized_symbol}")
        except Exception as e:
            logging.info(f"Exception in run function {e}")

    def run_single_symbol(self, symbol, rotator_symbols_standardized=None):
        try:
            logging.info(f"Starting to process symbol: {symbol}")
            logging.info(f"Initializing default values for symbol: {symbol}")

            self.running_long = True
            self.running_short = True
            
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

            # Initializing time trackers for less frequent API calls
            last_equity_fetch_time = 0
            equity_refresh_interval = 1800  # 30 minutes in seconds

            # Clean out orders
            # self.exchange.cancel_all_orders_for_symbol_bybit(symbol)
            # logging.info(f"Canceled all orders for {symbol}")

            # Check leverages only at startup
            self.current_leverage = self.exchange.get_current_max_leverage_bybit(symbol)
            self.max_leverage = self.exchange.get_current_max_leverage_bybit(symbol)

            logging.info(f"Max leverage for {symbol}: {self.max_leverage}")

            self.adjust_risk_parameters(exchange_max_leverage=self.max_leverage)

            self.exchange.set_leverage_bybit(self.max_leverage, symbol)
            self.exchange.set_symbol_to_cross_margin(symbol, self.max_leverage)

            logging.info(f"Running for symbol (inside run_single_symbol method): {symbol}")

            # Definitions
            quote_currency = "USDT"
            max_retries = 5
            retry_delay = 5


            test_orders_enabled = self.config.test_orders_enabled
            

            levels = self.config.linear_grid['levels']
            strength = self.config.linear_grid['strength']
            outer_price_distance = self.config.linear_grid['outer_price_distance']
            long_mode = self.config.linear_grid['long_mode']
            short_mode = self.config.linear_grid['short_mode']
            reissue_threshold = self.config.linear_grid['reissue_threshold']
            buffer_percentage = self.config.linear_grid['buffer_percentage']
            enforce_full_grid = self.config.linear_grid['enforce_full_grid']
            min_buffer_percentage = self.config.linear_grid['min_buffer_percentage']
            max_buffer_percentage = self.config.linear_grid['max_buffer_percentage']
            # reissue_threshold_inposition = self.config.linear_grid['reissue_threshold_inposition']

            volume_check = self.config.volume_check
            min_dist = self.config.min_distance
            min_vol = self.config.min_volume

            upnl_threshold_pct = self.config.upnl_threshold_pct
            upnl_profit_pct = self.config.upnl_profit_pct

            # Stop loss
            stoploss_enabled = self.config.stoploss_enabled
            stoploss_upnl_pct = self.config.stoploss_upnl_pct
            # Liq based stop loss
            liq_stoploss_enabled = self.config.liq_stoploss_enabled
            liq_price_stop_pct = self.config.liq_price_stop_pct

            # Auto reduce
            auto_reduce_enabled = self.config.auto_reduce_enabled
            auto_reduce_start_pct = self.config.auto_reduce_start_pct

            auto_reduce_maxloss_pct = self.config.auto_reduce_maxloss_pct

            entry_during_autoreduce = self.config.entry_during_autoreduce

            auto_reduce_marginbased_enabled = self.config.auto_reduce_marginbased_enabled

            auto_reduce_wallet_exposure_pct = self.config.auto_reduce_wallet_exposure_pct

            percentile_auto_reduce_enabled = self.config.percentile_auto_reduce_enabled
        
            max_pos_balance_pct = self.config.max_pos_balance_pct

            auto_leverage_upscale = self.config.auto_leverage_upscale

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
            recent_trades = self.fetch_recent_trades_for_symbol(symbol, since=since_timestamp, limit=20)

            logging.info(f"Recent trades for {symbol} : {recent_trades}")

            # Check if there are any trades in the last 24 hours
            recent_activity = any(trade['timestamp'] >= since_timestamp for trade in recent_trades)
            if recent_activity:
                logging.info(f"Recent trading activity detected for {symbol}")
            else:
                logging.info(f"No recent trading activity for {symbol} in the last 24 hours")

            long_pos_qty = position_details.get(symbol, {}).get('long', {}).get('qty', 0)
            short_pos_qty = position_details.get(symbol, {}).get('short', {}).get('qty', 0)

            terminate_side = self.should_terminate_open_orders(symbol, long_pos_qty, short_pos_qty, open_position_data, open_orders, current_time)

            logging.info(f"Terminate side: {terminate_side}")

            while self.running_long or self.running_short:
                current_time = time.time()

                iteration_start_time = time.time()

                leverage_tiers = self.exchange.fetch_leverage_tiers(symbol)

                if leverage_tiers:
                    logging.info(f"Leverage tiers for {symbol}: {leverage_tiers}")
                else:
                    logging.error(f"Failed to fetch leverage tiers for {symbol}.")


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

                logging.info(f"Open symbols: {open_symbols}")

                logging.info(f"Open orders: {open_orders}")

                market_data = self.get_market_data_with_retry(symbol, max_retries=100, retry_delay=5)
                min_qty = float(market_data["min_qty"])

                # position_last_update_time = self.get_position_update_time(symbol)

                # logging.info(f"{symbol} last update time: {position_last_update_time}")

                # Fetch equity data less frequently or if it's not available yet
                if current_time - last_equity_fetch_time > equity_refresh_interval or total_equity is None:
                    total_equity = self.retry_api_call(self.exchange.get_futures_balance_bybit, quote_currency)
                    available_equity = self.retry_api_call(self.exchange.get_available_balance_bybit, quote_currency)
                    last_equity_fetch_time = current_time

                    logging.info(f"Total equity: {total_equity}")
                    logging.info(f"Available equity: {available_equity}")
                    
                    # Log the type of total_equity
                    logging.info(f"Type of total_equity: {type(total_equity)}")
                    
                    # If total_equity is still None after fetching, log a warning and skip to the next iteration
                    if total_equity is None:
                        logging.warning("Failed to fetch total_equity. Skipping this iteration.")
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

                # Handling best ask price
                if 'asks' in order_book and len(order_book['asks']) > 0:
                    best_ask_price = order_book['asks'][0][0]
                    self.last_known_ask[symbol] = best_ask_price  # Update last known ask price
                else:
                    best_ask_price = self.last_known_ask.get(symbol)  # Use last known ask price

                # Handling best bid price
                if 'bids' in order_book and len(order_book['bids']) > 0:
                    best_bid_price = order_book['bids'][0][0]
                    self.last_known_bid[symbol] = best_bid_price  # Update last known bid price
                else:
                    best_bid_price = self.last_known_bid.get(symbol)  # Use last known bid price
                                
                moving_averages = self.get_all_moving_averages(symbol)

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
                # Update the previous position quantities
                previous_long_pos_qty = long_pos_qty
                previous_short_pos_qty = short_pos_qty

                terminate_long, terminate_short = self.should_terminate_open_orders(symbol, long_pos_qty, short_pos_qty, open_position_data, open_orders, current_time)
                logging.info(f"Terminate long: {terminate_long}, Terminate short: {terminate_short}")

                global thread_termination_status

                try:
                    if terminate_long:
                        logging.info(f"Should terminate long orders for {symbol}")
                        self.cancel_grid_orders(symbol, "buy")
                        self.cleanup_before_termination(symbol)
                        self.running_long = False  # Set the flag to stop the long thread
                        thread_termination_status[symbol] = True  # Update the termination status

                    if terminate_short:
                        logging.info(f"Should terminate short orders for {symbol}")
                        self.cancel_grid_orders(symbol, "sell")
                        self.cleanup_before_termination(symbol)
                        self.running_short = False  # Set the flag to stop the short thread
                        thread_termination_status[symbol] = True  # Update the termination status

                except Exception as e:
                    logging.info(f"Exception caught in termination {e}")

                if not self.running_long and not self.running_short:
                    logging.info("Both long and short operations have ended. Preparing to exit loop.")
                    shared_symbols_data.pop(symbol, None)  # Remove the symbol from shared_symbols_data

                    # Check if a position has been closed
                    if previous_long_pos_qty > 0 and long_pos_qty == 0:
                        logging.info(f"Long position closed for {symbol}. Canceling long grid orders.")
                        self.cancel_grid_orders(symbol, "buy")
                        self.cleanup_before_termination(symbol)
                        self.running_long = False  # Set the flag to stop the long thread

                    if previous_short_pos_qty > 0 and short_pos_qty == 0:
                        logging.info(f"Short position closed for {symbol}. Canceling short grid orders.")
                        self.cancel_grid_orders(symbol, "sell")
                        self.cleanup_before_termination(symbol)
                        self.running_short = False  # Set the flag to stop the short thread
                        
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
                    mfirsi_signal = self.get_mfirsi_ema_secondary_ema(symbol, limit=100, lookback=1, ema_period=5, secondary_ema_period=3)

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

                    long_pos_qty = position_details.get(symbol, {}).get('long', {}).get('qty', 0)
                    short_pos_qty = position_details.get(symbol, {}).get('short', {}).get('qty', 0)

                    # Update the previous position quantities
                    previous_long_pos_qty = long_pos_qty
                    previous_short_pos_qty = short_pos_qty

                    logging.info(f"Rotator symbol trading: {symbol}")
                                
                    logging.info(f"Rotator symbols: {rotator_symbols_standardized}")
                    logging.info(f"Open symbols: {open_symbols}")

                    logging.info(f"Long pos qty {long_pos_qty} for {symbol}")
                    logging.info(f"Short pos qty {short_pos_qty} for {symbol}")

                    # short_liq_price = position_data["short"]["liq_price"]
                    # long_liq_price = position_data["long"]["liq_price"]


                    # Adjust risk parameters based on the maximum leverage allowed by the exchange
                    self.adjust_risk_parameters(exchange_max_leverage=self.max_leverage)

                    # Calculate dynamic entry sizes for long and short positions
                    long_dynamic_amount, short_dynamic_amount = self.calculate_dynamic_amounts(
                        symbol=symbol,
                        total_equity=total_equity,
                        best_ask_price=best_ask_price,
                        best_bid_price=best_bid_price
                    )

                    logging.info(f"Long dynamic amount: {long_dynamic_amount} for {symbol}")
                    logging.info(f"Short dynamic amount: {short_dynamic_amount} for {symbol}")

                    cum_realised_pnl_long = position_data["long"]["cum_realised"]
                    cum_realised_pnl_short = position_data["short"]["cum_realised"]

                    # Get the average price for long and short positions
                    long_pos_price = position_details.get(symbol, {}).get('long', {}).get('avg_price', None)
                    short_pos_price = position_details.get(symbol, {}).get('short', {}).get('avg_price', None)

                    short_take_profit = None
                    long_take_profit = None

                    # Calculate take profit for short and long positions using quickscalp method
                    short_take_profit = self.calculate_quickscalp_short_take_profit(short_pos_price, symbol, upnl_profit_pct)
                    long_take_profit = self.calculate_quickscalp_long_take_profit(long_pos_price, symbol, upnl_profit_pct)

                    short_stop_loss = None
                    long_stop_loss = None

                    initial_short_stop_loss = None
                    initial_long_stop_loss = None

                    try:
                        self.auto_reduce_logic_grid(
                            symbol,
                            min_qty,
                            long_pos_price,
                            short_pos_price,
                            long_pos_qty,
                            short_pos_qty,
                            auto_reduce_enabled,
                            total_equity,
                            available_equity,
                            current_market_price=current_price,
                            long_dynamic_amount=long_dynamic_amount,
                            short_dynamic_amount=short_dynamic_amount,
                            auto_reduce_start_pct=auto_reduce_start_pct,
                            max_pos_balance_pct=max_pos_balance_pct,
                            upnl_threshold_pct=upnl_threshold_pct,
                            shared_symbols_data=shared_symbols_data
                        )
                    except Exception as e:
                        logging.info(f"Exception caught in auto_reduce_logic_grid {e}")

                    self.auto_reduce_percentile_logic(
                        symbol,
                        long_pos_qty,
                        long_pos_price,
                        short_pos_qty,
                        short_pos_price,
                        percentile_auto_reduce_enabled,
                        auto_reduce_start_pct,
                        auto_reduce_maxloss_pct,
                        long_dynamic_amount,
                        short_dynamic_amount
                    )

                    self.liq_stop_loss_logic(
                        long_pos_qty,
                        long_pos_price,
                        long_liquidation_price,
                        short_pos_qty,
                        short_pos_price,
                        short_liquidation_price,
                        liq_stoploss_enabled,
                        symbol,
                        liq_price_stop_pct
                    )

                    self.stop_loss_logic(
                        long_pos_qty,
                        long_pos_price,
                        short_pos_qty,
                        short_pos_price,
                        stoploss_enabled,
                        symbol,
                        stoploss_upnl_pct
                    )

                    self.auto_reduce_marginbased_logic(
                        auto_reduce_marginbased_enabled,
                        long_pos_qty,
                        short_pos_qty,
                        long_pos_price,
                        short_pos_price,
                        symbol,
                        total_equity,
                        auto_reduce_wallet_exposure_pct,
                        open_position_data,
                        current_price,
                        long_dynamic_amount,
                        short_dynamic_amount,
                        auto_reduce_start_pct,
                        auto_reduce_maxloss_pct
                    )

                    self.cancel_auto_reduce_orders_bybit(
                        symbol,
                        total_equity,
                        max_pos_balance_pct,
                        open_position_data,
                        long_pos_qty,
                        short_pos_qty
                    )


                    # short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, one_minute_distance, previous_one_minute_distance, short_take_profit, long_take_profit)
                    #short_take_profit, long_take_profit = self.calculate_take_profits_based_on_spread(short_pos_price, long_pos_price, symbol, five_minute_distance, previous_five_minute_distance, short_take_profit, long_take_profit)
                    previous_five_minute_distance = five_minute_distance

                    previous_one_minute_distance = one_minute_distance


                    logging.info(f"Short take profit for {symbol}: {short_take_profit}")
                    logging.info(f"Long take profit for {symbol}: {long_take_profit}")

                    should_short = self.short_trade_condition(best_ask_price, moving_averages["ma_3_high"])
                    should_long = self.long_trade_condition(best_bid_price, moving_averages["ma_3_low"])
                    should_add_to_short = False
                    should_add_to_long = False

                    if short_pos_price is not None:
                        should_add_to_short = short_pos_price < moving_averages["ma_6_low"] and self.short_trade_condition(best_ask_price, moving_averages["ma_6_high"])

                    if long_pos_price is not None:
                        should_add_to_long = long_pos_price > moving_averages["ma_6_high"] and self.long_trade_condition(best_bid_price, moving_averages["ma_6_low"])

                    open_tp_order_count = self.exchange.get_open_tp_order_count(symbol)

                    logging.info(f"Open TP order count {open_tp_order_count}")

                    logging.info(f"Five minute volume for {symbol} : {five_minute_volume}")
                        
                    historical_data = self.fetch_historical_data(
                        symbol,
                        timeframe='4h'
                    )

                    one_hour_atr_value = self.calculate_atr(historical_data)

                    logging.info(f"ATR for {symbol} : {one_hour_atr_value}")

                    tp_order_counts = self.exchange.get_open_tp_order_count(symbol)
                    #print(type(tp_order_counts))

                    # Check for long position
                    if long_pos_qty > 0:
                        try:
                            unrealized_pnl = self.exchange.fetch_unrealized_pnl(symbol)
                            long_upnl = unrealized_pnl.get('long')
                            logging.info(f"Long UPNL for {symbol}: {long_upnl}")
                        except Exception as e:
                            logging.info(f"Exception fetching Long UPNL for {symbol}: {e}")

                    # Check for short position
                    if short_pos_qty > 0:
                        try:
                            unrealized_pnl = self.exchange.fetch_unrealized_pnl(symbol)
                            short_upnl = unrealized_pnl.get('short')
                            logging.info(f"Short UPNL for {symbol}: {short_upnl}")
                        except Exception as e:
                            logging.info(f"Exception fetching Short UPNL for {symbol}: {e}")


                    long_tp_counts = tp_order_counts['long_tp_count']
                    short_tp_counts = tp_order_counts['short_tp_count']

                    try:
                        self.linear_grid_handle_positions_mfirsi_persistent_notional_dynamic_buffer(
                            symbol,
                            open_symbols,
                            total_equity,
                            long_pos_price,
                            short_pos_price,
                            long_pos_qty,
                            short_pos_qty,
                            levels,
                            strength,
                            outer_price_distance,
                            reissue_threshold,
                            self.wallet_exposure_limit,
                            self.user_defined_leverage_long,
                            self.user_defined_leverage_short,
                            long_mode,
                            short_mode,
                            min_buffer_percentage,
                            max_buffer_percentage,
                            self.symbols_allowed,
                            enforce_full_grid,
                            mfirsi_signal,
                            upnl_profit_pct,
                            tp_order_counts,
                            entry_during_autoreduce
                        )
                    except Exception as e:
                        logging.info("Something is up with variables for the grid")


                    logging.info(f"Long tp counts: {long_tp_counts}")
                    logging.info(f"Short tp counts: {short_tp_counts}")

                    logging.info(f"Long pos qty {long_pos_qty} for {symbol}")
                    logging.info(f"Short pos qty {short_pos_qty} for {symbol}")

                    logging.info(f"Long take profit {long_take_profit} for {symbol}")
                    logging.info(f"Short take profit {short_take_profit} for {symbol}")

                    logging.info(f"Long TP order count for {symbol} is {tp_order_counts['long_tp_count']}")
                    logging.info(f"Short TP order count for {symbol} is {tp_order_counts['short_tp_count']}")

                    self.place_long_tp_order(
                        symbol,
                        best_ask_price,
                        long_pos_price,
                        long_pos_qty,
                        long_take_profit,
                        open_orders
                    )

                    self.place_short_tp_order(
                        symbol,
                        best_bid_price,
                        short_pos_price,
                        short_pos_qty,
                        short_take_profit,
                        open_orders
                    )

                    current_latest_time = datetime.now()
                    logging.info(f"Current time: {current_latest_time}")
                    logging.info(f"Next long TP update time: {self.next_long_tp_update}")
                    logging.info(f"Next short TP update time: {self.next_short_tp_update}")

                    # Check for long positions
                    if long_pos_qty > 0:
                        if current_latest_time >= self.next_long_tp_update:
                            self.next_long_tp_update = self.update_quickscalp_tp(
                                symbol=symbol, 
                                pos_qty=long_pos_qty, 
                                upnl_profit_pct=upnl_profit_pct,  # Add the quickscalp percentage
                                short_pos_price=short_pos_price,
                                long_pos_price=long_pos_price,
                                positionIdx=1, 
                                order_side="sell", 
                                last_tp_update=self.next_long_tp_update,
                                tp_order_counts=tp_order_counts
                            )

                    # Check for short positions
                    if short_pos_qty > 0:
                        if current_latest_time >= self.next_short_tp_update:
                            self.next_short_tp_update = self.update_quickscalp_tp(
                                symbol=symbol, 
                                pos_qty=short_pos_qty, 
                                upnl_profit_pct=upnl_profit_pct,  # Add the quickscalp percentage
                                short_pos_price=short_pos_price,
                                long_pos_price=long_pos_price,
                                positionIdx=2, 
                                order_side="buy", 
                                last_tp_update=self.next_short_tp_update,
                                tp_order_counts=tp_order_counts
                            )

                    if self.test_orders_enabled and current_time - self.last_helper_order_cancel_time >= self.helper_interval:
                        if symbol in open_symbols:
                            self.helper_active = True
                            self.helperv2(symbol, short_dynamic_amount, long_dynamic_amount)
                        else:
                            logging.info(f"Skipping test orders for {symbol} as it's not in open symbols list.")
                            
                    # Check if the symbol should terminate
                    if self.should_terminate_full(symbol, current_time, previous_long_pos_qty, long_pos_qty, previous_short_pos_qty, short_pos_qty):
                        self.cleanup_before_termination(symbol)
                        break  # Exit the while loop, thus ending the thread
                    
                    # Check if a position has been closed
                    if previous_long_pos_qty > 0 and long_pos_qty == 0:
                        logging.info(f"Long position closed for {symbol}. Canceling long grid orders.")
                        self.cancel_grid_orders(symbol, "buy")
                        self.cleanup_before_termination(symbol)
                        break  # Exit the while loop, thus ending the thread

                    if previous_short_pos_qty > 0 and short_pos_qty == 0:
                        logging.info(f"Short position closed for {symbol}. Canceling short grid orders.")
                        self.cancel_grid_orders(symbol, "sell")
                        self.cleanup_before_termination(symbol)
                        break  # Exit the while loop, thus ending the thread

                    # self.cancel_entries_bybit(symbol, best_ask_price, moving_averages["ma_1m_3_high"], moving_averages["ma_5m_3_high"])
                    # self.cancel_stale_orders_bybit(symbol)

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

                time.sleep(3)
        except Exception as e:
            traceback_info = traceback.format_exc()  # Get the full traceback
            logging.error(f"Exception caught in quickscalp strategy '{symbol}': {e}\nTraceback:\n{traceback_info}")