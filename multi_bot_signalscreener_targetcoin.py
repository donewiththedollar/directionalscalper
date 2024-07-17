import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import Future
from collections import defaultdict
import threading
from threading import Thread
import random
import colorama
from colorama import Fore, Style
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent)
print("Project directory:", project_dir)
sys.path.insert(0, project_dir)

import traceback

import inquirer
from rich.live import Live
import argparse
from pathlib import Path
from config import load_config, Config, VERSION
from api.manager import Manager

from directionalscalper.core.exchanges import *

import directionalscalper.core.strategies.bybit.notional.instantsignals as instant_signals
import directionalscalper.core.strategies.bybit.notional as bybit_notional
import directionalscalper.core.strategies.bybit.scalping as bybit_scalping
import directionalscalper.core.strategies.bybit.hedging as bybit_hedging
from directionalscalper.core.strategies.binance import *
from directionalscalper.core.strategies.huobi import *

from live_table_manager import LiveTableManager, shared_symbols_data

from directionalscalper.core.strategies.logger import Logger

from rate_limit import RateLimit

from collections import deque

general_rate_limiter = RateLimit(50, 1)
order_rate_limiter = RateLimit(5, 1) 

thread_management_lock = threading.Lock()
thread_to_symbol = {}
thread_to_symbol_lock = threading.Lock()
active_symbols = set()
active_threads = defaultdict(dict)
long_threads = {}
short_threads = {}

threads = {}  # Threads for each symbol
thread_start_time = {}  # Dictionary to track the start time for each symbol's thread
symbol_last_started_time = {}

extra_symbols = set()  # To track symbols opened past the limit
under_review_symbols = set()

latest_rotator_symbols = set()
last_rotator_update_time = time.time()
tried_symbols = set()

logging = Logger(logger_name="MultiBot", filename="MultiBot.log", stream=True)

colorama.init()

def print_cool_trading_info(symbol, exchange_name, strategy_name, account_name):
    ascii_art = r"""
    ______  _____ 
    |  _  \/  ___|
    | | | |\ `--. 
    | | | | `--. \
    | |/ / /\__/ /
    |___/  \____/ 
                 
        Created by Tyler Simpson
    """
    print(Fore.GREEN + ascii_art)
    print(Style.BRIGHT + Fore.YELLOW + "DirectionalScalper is trading..")
    print(Fore.CYAN + f"Trading symbol: {symbol}")
    print(Fore.MAGENTA + f"Exchange name: {exchange_name}")
    print(Fore.BLUE + f"Strategy name: {strategy_name}")
    print(Fore.GREEN + f"Account name: {account_name}")
    print(Style.RESET_ALL)

def standardize_symbol(symbol):
    return symbol.replace('/', '').split(':')[0]

def get_available_strategies():
    return [
        'qsgridob',
        'qsgridoblsignal',
        # 'qstrendobdynamictp',
        # 'qsgridinstantsignal',
        # 'qsgridobtight',
        # 'qsgriddynamicstatic',
        # 'qsgridobdca',
        # 'qsgriddynmaicgridspaninstant',
        # 'qsdynamicgridspan',
        # 'qsgriddynamictplinspaced',
        # 'dynamicgridob',
        # 'dynamicgridobsratrp',
        # 'qsgriddynamictp',
        # 'qsgriddynamic',
        # 'qsgridbasic',
        # 'basicgridpersist',
        # 'qstrend',
        # 'qstrendob',
        # 'qstrenderi',
        # 'qstrendemas',
        # 'qstrend',
        # 'qsematrend',
        # 'qstrendemas',
        # 'mfieritrend',
        # 'qstrendlongonly',
        # 'qstrendshortonly',
        # 'qstrend_unified',
        # 'qstrendspot',
    ]

def choose_strategy():
    questions = [
        inquirer.List('strategy',
                      message='Which strategy would you like to run?',
                      choices=get_available_strategies())
    ]
    answers = inquirer.prompt(questions)
    return answers['strategy']

def get_available_exchanges():
    return ['bybit', 'hyperliquid']

def ask_for_missing_arguments(args):
    questions = []
    if not args.exchange:
        questions.append(inquirer.List('exchange', message="Which exchange do you want to use?", choices=get_available_exchanges()))
    if not args.strategy:
        questions.append(inquirer.List('strategy', message="Which strategy do you want to use?", choices=get_available_strategies()))
    if not args.account_name:
        questions.append(inquirer.Text('account_name', message="Please enter the name of the account:"))

    if questions:
        answers = inquirer.prompt(questions)
        args.exchange = args.exchange or answers.get('exchange')
        args.strategy = args.strategy or answers.get('strategy')
        args.account_name = args.account_name or answers.get('account_name')

    return args

class DirectionalMarketMaker:
    def __init__(self, config: Config, exchange_name: str, account_name: str):
        self.config = config
        self.exchange_name = exchange_name
        self.account_name = account_name

        exchange_config = next((exch for exch in config.exchanges if exch.name == exchange_name and exch.account_name == account_name), None)
        
        if not exchange_config:
            raise ValueError(f"Exchange {exchange_name} with account {account_name} not found in the configuration file.")

        api_key = exchange_config.api_key
        secret_key = exchange_config.api_secret
        passphrase = getattr(exchange_config, 'passphrase', None)  # Use getattr to get passphrase if it exists
        
        exchange_classes = {
            'bybit': BybitExchange,
            'bybit_spot': BybitExchange,
            'hyperliquid': HyperLiquidExchange,
            'huobi': HuobiExchange,
            'bitget': BitgetExchange,
            'binance': BinanceExchange,
            'mexc': MexcExchange,
            'lbank': LBankExchange,
            'blofin': BlofinExchange
        }

        exchange_class = exchange_classes.get(exchange_name.lower(), Exchange)

        # Initialize the exchange based on whether a passphrase is required
        if exchange_name.lower() in ['bybit', 'binance']:  # Add other exchanges here that do not require a passphrase
            self.exchange = exchange_class(api_key, secret_key)
        elif exchange_name.lower() == 'bybit_spot':
            self.exchange = exchange_class(api_key, secret_key, 'spot')
        else:
            self.exchange = exchange_class(api_key, secret_key, passphrase)

    def run_strategy(self, symbol, strategy_name, config, account_name, symbols_to_trade=None, rotator_symbols_standardized=None, mfirsi_signal=None, action=None):
        logging.info(f"Received rotator symbols in run_strategy for {symbol}: {rotator_symbols_standardized}")
        
        symbols_allowed = next((exch.symbols_allowed for exch in config.exchanges if exch.name == self.exchange_name and exch.account_name == account_name), None)

        logging.info(f"Matched exchange: {self.exchange_name}, account: {account_name}. Symbols allowed: {symbols_allowed}")

        if symbols_to_trade:
            logging.info(f"Calling run method with symbols: {symbols_to_trade}")
            try:
                print_cool_trading_info(symbol, self.exchange_name, strategy_name, account_name)
                logging.info(f"Printed trading info for {symbol}")
            except Exception as e:
                logging.error(f"Error in printing info: {e}")

        strategy_classes = {
            'bybit_1m_qfl_mfi_eri_walls': bybit_scalping.BybitMMOneMinuteQFLMFIERIWalls,
            'bybit_1m_qfl_mfi_eri_autohedge_walls_atr': bybit_hedging.BybitMMOneMinuteQFLMFIERIAutoHedgeWallsATR,
            'bybit_mfirsi_imbalance': bybit_scalping.BybitMFIRSIERIOBImbalance,
            'bybit_mfirsi_quickscalp': bybit_scalping.BybitMFIRSIQuickScalp,
            'qsematrend': bybit_scalping.BybitQuickScalpEMATrend,
            'qstrend_dca': bybit_scalping.BybitQuickScalpTrendDCA,
            'mfieritrend': bybit_scalping.BybitMFIERILongShortTrend,
            'qstrendlongonly': bybit_scalping.BybitMFIRSIQuickScalpLong,
            'qstrendshortonly': bybit_scalping.BybitMFIRSIQuickScalpShort,
            'qstrend_unified': bybit_scalping.BybitQuickScalpUnified,
            'basicgrid': bybit_scalping.BybitBasicGrid,
            'qstrendspot': bybit_scalping.BybitQuickScalpTrendSpot,
            'qsgridinstantsignal': instant_signals.BybitDynamicGridSpanOBSRStaticIS,
            'qsgriddynmaicgridspaninstant': instant_signals.BybitDynamicGridSpanIS,
            'qsgridob': instant_signals.BybitDynamicGridSpanOBLevels,
            'qstrendobdynamictp': instant_signals.BybitQuickScalpTrendDynamicTP,
            'qsgridoblsignal': instant_signals.BybitDynamicGridSpanOBLevelsLSignal
        }

        strategy_class = strategy_classes.get(strategy_name.lower())
        if strategy_class:
            strategy = strategy_class(self.exchange, self.manager, config.bot, symbols_allowed)
            try:
                logging.info(f"Running strategy for symbol {symbol} with action {action}")
                if action == "long":
                    future_long = Future()
                    Thread(target=self.run_with_future, args=(strategy, symbol, rotator_symbols_standardized, mfirsi_signal, "long", future_long)).start()
                    return future_long
                elif action == "short":
                    future_short = Future()
                    Thread(target=self.run_with_future, args=(strategy, symbol, rotator_symbols_standardized, mfirsi_signal, "short", future_short)).start()
                    return future_short
                else:
                    future = Future()
                    future.set_result(True)
                    return future
            except Exception as e:
                future = Future()
                future.set_exception(e)
                return future
        else:
            logging.error(f"Strategy {strategy_name} not found.")
            future = Future()
            future.set_exception(ValueError(f"Strategy {strategy_name} not found."))
            return future


    def run_with_future(self, strategy, symbol, rotator_symbols_standardized, mfirsi_signal, action, future):
        try:
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized, mfirsi_signal=mfirsi_signal, action=action)
            future.set_result(True)
        except Exception as e:
            future.set_exception(e)

    def get_balance(self, quote, market_type=None, sub_type=None):
        if self.exchange_name == 'bitget':
            return self.exchange.get_balance_bitget(quote)
        elif self.exchange_name == 'bybit':
            #self.exchange.retry_api_call(self.exchange.get_balance_bybit, quote)
            # return self.exchange.retry_api_call(self.exchange.get_balance_bybit(quote))
            return self.exchange.get_balance_bybit(quote)
        elif self.exchange_name == 'bybit_unified':
            return self.exchange.retry_api_call(self.exchange.get_balance_bybit(quote))
        elif self.exchange_name == 'mexc':
            return self.exchange.get_balance_mexc(quote, market_type='swap')
        elif self.exchange_name == 'huobi':
            print("Huobi starting..")
        elif self.exchange_name == 'okx':
            print(f"Unsupported for now")
        elif self.exchange_name == 'binance':
            return self.exchange.get_balance_binance(quote)
        elif self.exchange_name == 'phemex':
            print(f"Unsupported for now")

    def create_order(self, symbol, order_type, side, amount, price=None):
        return self.exchange.create_order(symbol, order_type, side, amount, price)

    def get_symbols(self):
        with general_rate_limiter:
            return self.exchange._get_symbols()

    def format_symbol_bybit(self, symbol):
        return f"{symbol[:3]}/{symbol[3:]}:USDT"

    def is_valid_symbol_bybit(self, symbol):
        valid_symbols = self.get_symbols()
        
        # Check for SYMBOL/USDT:USDT format
        if f"{symbol[:3]}/{symbol[3:]}:USDT" in valid_symbols:
            return True
        
        # Check for SYMBOL/USD:SYMBOL format
        if f"{symbol[:3]}/USD:{symbol[:3]}" in valid_symbols:
            return True
        
        # Check for SYMBOL/USDC:USDC format
        if f"{symbol}/USDC:USDC" in valid_symbols:
            return True
        
        # Check for SYMBOL/USDC:USDC-YYMMDD format
        for valid_symbol in valid_symbols:
            if valid_symbol.startswith(f"{symbol}/USDC:USDC-"):
                return True
        
        # Check for SYMBOL/USDC:USDC-YYMMDD-STRIKE-C/P format
        for valid_symbol in valid_symbols:
            if valid_symbol.startswith(f"{symbol}/USDC:USDC-") and valid_symbol.endswith(("-C", "-P")):
                return True
        
        logging.info(f"Invalid symbol type for some reason according to bybit but is probably valid symbol: {symbol}")
        return True

    def fetch_open_orders(self, symbol):
        with general_rate_limiter:
            return self.exchange.retry_api_call(self.exchange.get_open_orders, symbol)

    def fetch_open_positions(self):
        with general_rate_limiter:
            return getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()

    def generate_l_signals(self, symbol):
        with general_rate_limiter:
            return self.exchange.generate_l_signals(symbol)

    def get_mfirsi_signal(self, symbol):
        # Retrieve the MFI/RSI signal
        with general_rate_limiter:
            return self.exchange.get_mfirsi_ema_secondary_ema(symbol, limit=100, lookback=1, ema_period=5, secondary_ema_period=3)

BALANCE_REFRESH_INTERVAL = 600  # in seconds

orders_canceled = False

def monitor_threads():
    """Function to monitor and restart threads if needed."""
    while True:
        with thread_management_lock:
            for symbol, thread_info in list(active_threads.items()):
                for action, thread_data in list(thread_info.items()):
                    thread, thread_completed = thread_data
                    if not thread.is_alive() and not thread_completed.is_set():
                        logging.info(f"Thread for symbol {symbol} with action {action} is not alive and not marked as completed. Restarting the thread.")
                        try:
                            thread_completed.set()  # Mark the thread as completed
                            thread.join()  # Ensure the thread has terminated
                            logging.info(f"Successfully joined the thread for symbol {symbol} with action {action}.")

                            # Restart the thread
                            with general_rate_limiter:
                                mfirsi_signal = market_maker.generate_l_signals(symbol)
                            new_thread_completed = threading.Event()
                            new_thread = threading.Thread(target=run_bot, args=(
                                symbol, args, market_maker, manager, args.account_name, symbols_allowed,
                                latest_rotator_symbols, new_thread_completed, mfirsi_signal, action))
                            active_threads[symbol][action] = (new_thread, new_thread_completed)
                            new_thread.start()
                            logging.info(f"Successfully restarted the thread for symbol {symbol} with action {action}.")
                        except Exception as e:
                            logging.error(f"Error while restarting the thread for symbol {symbol} with action {action}: {e}")
                            logging.debug(traceback.format_exc())
                    else:
                        logging.debug(f"Thread for symbol {symbol} with action {action} is alive or already marked as completed.")
        time.sleep(10)  # Monitor interval

def run_bot(symbol, args, market_maker, manager, account_name, symbols_allowed, rotator_symbols_standardized, thread_completed, mfirsi_signal, action):
    global orders_canceled
    current_thread = threading.current_thread()
    try:
        with thread_to_symbol_lock:
            thread_to_symbol[current_thread] = symbol
            active_symbols.add(symbol)  # Add symbol to active_symbols when the thread starts

        if not args.config.startswith('configs/'):
            config_file_path = Path('configs/' + args.config)
        else:
            config_file_path = Path(args.config)

        logging.info(f"Loading config from: {config_file_path}")

        account_file_path = Path('configs/account.json')  # Define the account file path
        config = load_config(config_file_path, account_file_path)  # Pass both file paths to load_config

        exchange_name = args.exchange
        strategy_name = args.strategy
        account_name = args.account_name

        logging.info(f"Trading symbol: {symbol}")
        logging.info(f"Exchange name: {exchange_name}")
        logging.info(f"Strategy name: {strategy_name}")
        logging.info(f"Account name: {account_name}")

        market_maker.manager = manager

        try:
            if not orders_canceled and hasattr(market_maker.exchange, 'cancel_all_open_orders_bybit'):
                market_maker.exchange.cancel_all_open_orders_bybit()
                logging.info(f"Cleared all open orders on the exchange upon initialization.")
                orders_canceled = True
        except Exception as e:
            logging.error(f"Exception caught while cancelling orders: {e}")

        logging.info(f"Rotator symbols in run_bot: {rotator_symbols_standardized}")
        logging.info(f"Latest rotator symbols in run bot: {latest_rotator_symbols}")

        time.sleep(2)

        with general_rate_limiter:
            future = market_maker.run_strategy(symbol, args.strategy, config, account_name, symbols_to_trade=symbols_allowed, rotator_symbols_standardized=latest_rotator_symbols, mfirsi_signal=mfirsi_signal, action=action)
            future.result()  # Wait for the strategy to complete

    except Exception as e:
        logging.error(f"An error occurred in run_bot for symbol {symbol}: {e}")
        logging.debug(traceback.format_exc())
    finally:
        with thread_to_symbol_lock:
            if current_thread in thread_to_symbol:
                del thread_to_symbol[current_thread]
            active_symbols.discard(symbol)  # Remove symbol from active_symbols when the thread completes
        logging.info(f"Thread for symbol {symbol} with action {action} has completed.")
        thread_completed.set()

def bybit_auto_rotation_spot(args, market_maker, manager, symbols_allowed):
    global latest_rotator_symbols, active_symbols, last_rotator_update_time

    # Set max_workers to the number of CPUs
    max_workers_signals = 1
    max_workers_trading = 1

    signal_executor = ThreadPoolExecutor(max_workers=max_workers_signals)
    trading_executor = ThreadPoolExecutor(max_workers=max_workers_trading)

    logging.info(f"Initialized signal executor with max workers: {max_workers_signals}")
    logging.info(f"Initialized trading executor with max workers: {max_workers_trading}")

    config_file_path = Path('configs/' + args.config) if not args.config.startswith('configs/') else Path(args.config)
    account_file_path = Path('configs/account.json')
    config = load_config(config_file_path, account_file_path)

    market_maker.manager = manager

    long_mode = config.bot.linear_grid['long_mode']
    short_mode = config.bot.linear_grid['short_mode']

    logging.info(f"Long mode: {long_mode}")
    logging.info(f"Short mode: {short_mode}")

    def fetch_open_positions():
        with general_rate_limiter:
            return getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()

    def process_futures(futures):
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Exception in thread: {e}")
                logging.debug(traceback.format_exc())

    while True:
        try:
            current_time = time.time()
            open_position_data = fetch_open_positions()
            open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
            logging.info(f"Open position symbols: {open_position_symbols}")

            if not latest_rotator_symbols or current_time - last_rotator_update_time >= 60:
                with general_rate_limiter:
                    latest_rotator_symbols = fetch_updated_symbols(args, manager)
                last_rotator_update_time = current_time
                logging.info(f"Refreshed latest rotator symbols: {latest_rotator_symbols}")
            else:
                logging.debug(f"No refresh needed yet. Last update was at {last_rotator_update_time}, less than 60 seconds ago.")

            update_active_symbols(open_position_symbols)
            logging.info(f"Active symbols: {active_symbols}")
            logging.info(f"Active symbols updated. Symbols allowed: {symbols_allowed}")

            with thread_management_lock:
                # update_active_symbols(open_position_symbols)
                # logging.info(f"Active symbols updated. Symbols allowed: {symbols_allowed}")

                open_position_futures = []
                for symbol in open_position_symbols:
                    with general_rate_limiter:
                        mfirsi_signal = market_maker.generate_l_signals(symbol)
                    has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
                    open_position_futures.append(trading_executor.submit(start_thread_for_open_symbol_spot, symbol, args, manager, mfirsi_signal, has_open_long, long_mode, short_mode))
                    logging.info(f"Submitted thread for symbol {symbol}. MFIRSI signal: {mfirsi_signal}. Has open long: {has_open_long}.")

                signal_futures = [signal_executor.submit(process_signal_for_open_position_spot, symbol, args, manager, symbols_allowed, open_position_data, long_mode, short_mode)
                                for symbol in open_position_symbols]
                logging.info(f"Submitted signal processing for open position symbols: {open_position_symbols}.")

                if len(active_symbols) < symbols_allowed:
                    for symbol in latest_rotator_symbols:
                        signal_futures.append(signal_executor.submit(process_signal_spot, symbol, args, manager, symbols_allowed, open_position_data, False, long_mode, short_mode))
                        logging.info(f"Submitted signal processing for new rotator symbol {symbol}.")
                        time.sleep(2)

                process_futures(open_position_futures + signal_futures)

                completed_symbols = []
                for symbol, (thread, thread_completed) in long_threads.items():
                    if thread_completed.is_set():
                        thread.join()
                        completed_symbols.append(symbol)

                for symbol in completed_symbols:
                    active_symbols.discard(symbol)
                    if symbol in long_threads:
                        del long_threads[symbol]
                    logging.info(f"Thread and symbol management completed for: {symbol}")

        except Exception as e:
            logging.error(f"Exception caught in bybit_auto_rotation_spot: {str(e)}")
            logging.debug(traceback.format_exc())
        time.sleep(1)

def bybit_auto_rotation(args, market_maker, manager, symbols_allowed):
    global latest_rotator_symbols, long_threads, short_threads, active_symbols, last_rotator_update_time

    max_workers_signals = 1
    max_workers_trading = 1

    signal_executor = ThreadPoolExecutor(max_workers=max_workers_signals)
    trading_executor = ThreadPoolExecutor(max_workers=max_workers_trading)
    
    logging.info(f"Initialized signal executor with max workers: {max_workers_signals}")
    logging.info(f"Initialized trading executor with max workers: {max_workers_trading}")

    config_file_path = Path('configs/' + args.config) if not args.config.startswith('configs/') else Path(args.config)
    account_file_path = Path('configs/account.json')
    config = load_config(config_file_path, account_file_path)

    market_maker.manager = manager

    long_mode = config.bot.linear_grid['long_mode']
    short_mode = config.bot.linear_grid['short_mode']

    logging.info(f"Long mode: {long_mode}")
    logging.info(f"Short mode: {short_mode}")

    # Get the whitelisted symbol
    whitelist = config.bot.whitelist
    if not whitelist:
        logging.error("No symbol in whitelist. Please add a symbol to the whitelist.")
        return
    whitelisted_symbol = whitelist[0]  # Assume we're using the first symbol in the whitelist

    def fetch_open_positions():
        with general_rate_limiter:
            return getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()

    # Start threads for open positions at startup
    open_position_data = fetch_open_positions()
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
    
    for symbol in open_position_symbols:
        has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
        has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
        
        with general_rate_limiter:
            mfirsi_signal = market_maker.generate_l_signals(symbol)
        
        if has_open_long:
            start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "long")
        if has_open_short:
            start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "short")

    while True:
        try:
            current_time = time.time()
            
            # Always process the whitelisted symbol
            with general_rate_limiter:
                mfirsi_signal = market_maker.generate_l_signals(whitelisted_symbol)
            
            logging.info(f"Processing signal for whitelisted symbol {whitelisted_symbol}. MFIRSI signal: {mfirsi_signal}")

            # Check if there's an open position for the whitelisted symbol
            open_position_data = fetch_open_positions()
            has_open_long = any(pos['side'].lower() == 'long' and standardize_symbol(pos['symbol']) == whitelisted_symbol for pos in open_position_data)
            has_open_short = any(pos['side'].lower() == 'short' and standardize_symbol(pos['symbol']) == whitelisted_symbol for pos in open_position_data)

            # Process the signal for the whitelisted symbol
            action_taken = handle_signal(whitelisted_symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, True, long_mode, short_mode)

            if action_taken:
                logging.info(f"Action taken for whitelisted symbol {whitelisted_symbol}.")
            else:
                logging.info(f"No action taken for whitelisted symbol {whitelisted_symbol}.")

            # Manage existing threads
            manage_threads(market_maker, args, manager)

            time.sleep(1)  # Adjust this sleep time as needed to control how often you check for signals

        except Exception as e:
            logging.error(f"Exception caught in bybit_auto_rotation: {str(e)}")
            logging.debug(traceback.format_exc())

def manage_threads(market_maker, args, manager):
    """Function to manage and restart threads if needed."""
    try:
        logging.info("Starting to manage threads.")

        # Fetch open positions at regular intervals
        open_position_data = market_maker.fetch_open_positions()
        open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
        
        # Logging the open position symbols
        logging.info(f"Fetched open position symbols: {open_position_symbols}")

        with thread_management_lock:
            logging.info(f"Active long threads: {list(long_threads.keys())}")
            logging.info(f"Active short threads: {list(short_threads.keys())}")

            # Check for missing threads for open positions
            for symbol in open_position_symbols:
                has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
                has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)

                with general_rate_limiter:
                    mfirsi_signal = market_maker.generate_l_signals(symbol)

                if has_open_long:
                    if symbol not in long_threads or not long_threads[symbol][0].is_alive():
                        logging.info(f"Starting missing long thread for symbol {symbol}.")
                        start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "long")
                    else:
                        logging.info(f"Long thread already active for symbol {symbol}.")
                
                if has_open_short:
                    if symbol not in short_threads or not short_threads[symbol][0].is_alive():
                        logging.info(f"Starting missing short thread for symbol {symbol}.")
                        start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "short")
                    else:
                        logging.info(f"Short thread already active for symbol {symbol}.")

            # Stop threads for symbols that are no longer open
            for symbol in list(long_threads.keys()):
                if symbol not in open_position_symbols:
                    logging.info(f"Stopping long thread for symbol {symbol} as it is no longer open.")
                    stop_thread_for_symbol(symbol, "long")
                else:
                    logging.info(f"Long thread still needed and active for symbol {symbol}.")

            for symbol in list(short_threads.keys()):
                if symbol not in open_position_symbols:
                    logging.info(f"Stopping short thread for symbol {symbol} as it is no longer open.")
                    stop_thread_for_symbol(symbol, "short")
                else:
                    logging.info(f"Short thread still needed and active for symbol {symbol}.")

    except Exception as e:
        logging.error(f"Exception caught in manage_threads: {e}")
        logging.debug(traceback.format_exc())

def stop_thread_for_symbol(symbol, action):
    """Stop the thread for a given symbol and action."""
    if action == "long" and symbol in long_threads:
        thread, thread_completed = long_threads[symbol]
        thread_completed.set()
        thread.join()
        del long_threads[symbol]
        logging.info(f"Stopped and removed long thread for symbol {symbol}.")
    elif action == "short" and symbol in short_threads:
        thread, thread_completed = short_threads[symbol]
        thread_completed.set()
        thread.join()
        del short_threads[symbol]
        logging.info(f"Stopped and removed short thread for symbol {symbol}.")

def process_signal_for_open_position(symbol, args, market_maker, manager, symbols_allowed, open_position_data, long_mode, short_mode):
    market_maker.manager = manager

    with general_rate_limiter:
        mfirsi_signal = market_maker.generate_l_signals(symbol)
    logging.info(f"Processing signal for open position symbol {symbol}. MFIRSI signal: {mfirsi_signal}")

    action_taken = handle_signal(symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, True, long_mode, short_mode)

    if action_taken:
        logging.info(f"Action taken for open position symbol {symbol}.")
    else:
        logging.info(f"No action taken for open position symbol {symbol}.")


def process_signal(symbol, args, market_maker, manager, symbols_allowed, open_position_data, is_open_position, long_mode, short_mode):
    market_maker.manager = manager

    mfirsi_signal = market_maker.generate_l_signals(symbol)
    logging.info(f"Processing signal for {'open position' if is_open_position else 'new rotator'} symbol {symbol}. MFIRSI signal: {mfirsi_signal}")

    action_taken = handle_signal(symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, is_open_position, long_mode, short_mode)

    if action_taken:
        logging.info(f"Action taken for {'open position' if is_open_position else 'new rotator'} symbol {symbol}.")
    else:
        logging.info(f"No action taken for {'open position' if is_open_position else 'new rotator'} symbol {symbol}.")

def handle_signal(symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, is_open_position, long_mode, short_mode):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
    logging.info(f"Open position symbols: {open_position_symbols}")

    mfi_signal_long = mfirsi_signal.lower() == "long"
    mfi_signal_short = mfirsi_signal.lower() == "short"

    current_long_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'long')
    current_short_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'short')

    unique_open_symbols = len(open_position_symbols)

    logging.info(f"Handling signal for whitelisted symbol {symbol}. Current long positions: {current_long_positions}. Current short positions: {current_short_positions}. Unique open symbols: {unique_open_symbols}")

    has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
    has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)

    logging.info(f"Whitelisted symbol {symbol} - Has open long: {has_open_long}, Has open short: {has_open_short}")
    logging.info(f"MFIRSI Signal: {mfirsi_signal}, Long Mode: {long_mode}, Short Mode: {short_mode}")

    action_taken_long = False
    action_taken_short = False

    # Always attempt to start a new long position if the signal is long
    if mfi_signal_long and long_mode:
        logging.info(f"Starting long thread for symbol {symbol}.")
        action_taken_long = start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "long")
    else:
        logging.info(f"Long signal not triggered or long mode not enabled for symbol {symbol}.")
        logging.info(f"MFIRSI Signal: {mfirsi_signal}")
        logging.info(f"Long mode: {long_mode}")
        logging.info(f"Has open long: {has_open_long}")

    # Always attempt to start a new short position if the signal is short
    if mfi_signal_short and short_mode:
        logging.info(f"Starting short thread for symbol {symbol}.")
        action_taken_short = start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "short")
    else:
        logging.info(f"Short signal not triggered or short mode not enabled for symbol {symbol}.")
        logging.info(f"MFIRSI Signal: {mfirsi_signal}")
        logging.info(f"Short mode: {short_mode}")
        logging.info(f"Has open short: {has_open_short}")

    if action_taken_long or action_taken_short:
        logging.info(f"Action taken for whitelisted symbol {symbol}.")
    else:
        logging.info(f"No action taken for whitelisted symbol {symbol} due to lack of clear signal.")

    return action_taken_long or action_taken_short

def handle_signal_spot(symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, is_open_position, long_mode, short_mode):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
    logging.info(f"Open position symbols: {open_position_symbols}")

    mfi_signal_long = mfirsi_signal.lower() == "long"
    mfi_signal_short = mfirsi_signal.lower() == "short"

    current_long_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'long')
    unique_open_symbols = len(open_position_symbols)

    logging.info(f"Handling signal for {'open position' if is_open_position else 'new rotator'} symbol {symbol}. Current long positions: {current_long_positions}. Unique open symbols: {unique_open_symbols}")

    has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)

    logging.info(f"{'Open position' if is_open_position else 'New rotator'} symbol {symbol} - Has open long: {has_open_long}")

    action_taken_long = False

    if mfi_signal_long and long_mode and not has_open_long:
        logging.info(f"Starting long thread for symbol {symbol}.")
        action_taken_long = start_thread_for_symbol_spot(symbol, args, manager, mfirsi_signal, "long")
    else:
        logging.info(f"Long thread already running or long position already open for symbol {symbol}. Skipping.")

    if mfi_signal_short and short_mode and has_open_long:
        logging.info(f"Starting short (sell) thread for symbol {symbol}.")
        action_taken_long = start_thread_for_symbol_spot(symbol, args, manager, mfirsi_signal, "short")
    else:
        logging.info(f"Short thread (sell order) already running or no long position open for symbol {symbol}. Skipping.")

    if action_taken_long:
        logging.info(f"Action taken for {'open position' if is_open_position else 'new rotator'} symbol {symbol}.")
    else:
        logging.info(f"Evaluated action for {'open position' if is_open_position else 'new rotator'} symbol {symbol}: No action due to existing position or lack of clear signal.")

    return action_taken_long

def process_signal_for_open_position_spot(symbol, args, market_maker, manager, symbols_allowed, open_position_data, long_mode, short_mode):
    market_maker.manager = manager
    with general_rate_limiter:
        mfirsi_signal = market_maker.generate_l_signals(symbol)
    logging.info(f"Processing signal for open position symbol {symbol}. MFIRSI signal: {mfirsi_signal}")

    action_taken = handle_signal_spot(symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, True, long_mode, short_mode)

    if action_taken:
        logging.info(f"Action taken for open position symbol {symbol}.")
    else:
        logging.info(f"No action taken for open position symbol {symbol}.")

def process_signal_spot(symbol, args, market_maker, manager, symbols_allowed, open_position_data, is_open_position, long_mode, short_mode):
    market_maker.manager = manager
    mfirsi_signal = market_maker.generate_l_signals(symbol)
    logging.info(f"Processing signal for {'open position' if is_open_position else 'new rotator'} symbol {symbol}. MFIRSI signal: {mfirsi_signal}")

    action_taken = handle_signal_spot(symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, is_open_position, long_mode, short_mode)

    if action_taken:
        logging.info(f"Action taken for {'open position' if is_open_position else 'new rotator'} symbol {symbol}.")
    else:
        logging.info(f"No action taken for {'open position' if is_open_position else 'new rotator'} symbol {symbol}.")

def start_thread_for_open_symbol_spot(symbol, args, manager, mfirsi_signal, has_open_long, long_mode, short_mode):
    action_taken = False
    if long_mode and (has_open_long or mfirsi_signal.lower() == "long"):
        action_taken |= start_thread_for_symbol_spot(symbol, args, manager, mfirsi_signal, "long")
        logging.info(f"[DEBUG] Started long thread for open symbol {symbol}")
    if short_mode and (has_open_long and mfirsi_signal.lower() == "short"):
        action_taken |= start_thread_for_symbol_spot(symbol, args, manager, mfirsi_signal, "short")
        logging.info(f"[DEBUG] Started short (sell) thread for open symbol {symbol}")
    return action_taken

def start_thread_for_symbol_spot(symbol, args, manager, mfirsi_signal, action):
    if action == "long":
        if symbol in long_threads and long_threads[symbol][0].is_alive():
            logging.info(f"Long thread already running for symbol {symbol}. Skipping.")
            return False
    elif action == "short":
        if symbol in long_threads and long_threads[symbol][0].is_alive():
            logging.info(f"Short thread (sell order) already running for symbol {symbol}. Skipping.")
            return False
    elif action == "neutral":
        logging.info(f"Start thread function hit for {symbol} but signal is {mfirsi_signal}")

    thread_completed = threading.Event()
    thread = threading.Thread(target=run_bot, args=(symbol, args, market_maker, manager, args.account_name, symbols_allowed, latest_rotator_symbols, thread_completed, mfirsi_signal, action))

    if action == "long":
        long_threads[symbol] = (thread, thread_completed)
    elif action == "short":
        long_threads[symbol] = (thread, thread_completed)

    thread.start()
    logging.info(f"Started thread for symbol {symbol} with action {action} based on MFIRSI signal.")
    return True

def update_active_symbols(open_position_symbols):
    global active_symbols
    active_symbols = open_position_symbols
    logging.info(f"Updated active symbols: {active_symbols}")

def manage_rotator_symbols(rotator_symbols, args, manager, symbols_allowed):
    global active_symbols, latest_rotator_symbols

    logging.info(f"Starting symbol management. Total symbols allowed: {symbols_allowed}. Current active symbols: {len(active_symbols)}")

    open_position_data = getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
    logging.info(f"Currently open positions: {open_position_symbols}")

    current_long_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'long')
    current_short_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'short')
    logging.info(f"Current long positions: {current_long_positions}, Current short positions: {current_short_positions}")

    random_rotator_symbols = list(rotator_symbols)
    random.shuffle(random_rotator_symbols)
    logging.info(f"Shuffled rotator symbols for processing: {random_rotator_symbols}")

    for symbol in open_position_symbols:
        process_signal(symbol, args, market_maker, manager, symbols_allowed, open_position_data, True)

    for symbol in random_rotator_symbols:
        if len(open_position_symbols) >= symbols_allowed:
            logging.info("Maximum number of open positions reached.")
            break
        process_signal(symbol, args, manager, market_maker, symbols_allowed, open_position_data, False)

    manage_excess_threads(symbols_allowed)
    time.sleep(5)

def manage_excess_threads(symbols_allowed):
    global active_symbols
    long_positions = {symbol for symbol in active_symbols if is_long_position(symbol)}
    short_positions = {symbol for symbol in active_symbols if is_short_position(symbol)}

    logging.info(f"Managing excess threads. Total long positions: {len(long_positions)}, Total short positions: {len(short_positions)}")

    excess_long_count = len(long_positions) - symbols_allowed
    excess_short_count = len(short_positions) - symbols_allowed

    while excess_long_count > 0:
        symbol_to_remove = long_positions.pop()
        remove_thread_for_symbol(symbol_to_remove)
        logging.info(f"Removed excess long thread for symbol: {symbol_to_remove}")
        excess_long_count -= 1

    while excess_short_count > 0:
        symbol_to_remove = short_positions.pop()
        remove_thread_for_symbol(symbol_to_remove)
        logging.info(f"Removed excess short thread for symbol: {symbol_to_remove}")
        excess_short_count -= 1

def is_long_position(symbol):
    pos_data = getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()
    is_long = any(standardize_symbol(pos['symbol']) == symbol and pos['side'].lower() == 'long' for pos in pos_data)
    logging.debug(f"Checked if {symbol} is a long position: {is_long}")
    return is_long

def is_short_position(symbol):
    pos_data = getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()
    is_short = any(standardize_symbol(pos['symbol']) == symbol and pos['side'].lower() == 'short' for pos in pos_data)
    logging.debug(f"Checked if {symbol} is a short position: {is_short}")
    return is_short

def remove_thread_for_symbol(symbol):
    if symbol in long_threads:
        thread, thread_completed = long_threads[symbol]
    elif symbol in short_threads:
        thread, thread_completed = short_threads[symbol]
    else:
        return

    if thread:
        thread_completed.set()
        thread.join()
        logging.info(f"Removed thread for symbol {symbol}.")

    if symbol in long_threads:
        del long_threads[symbol]
    if symbol in short_threads:
        del short_threads[symbol]

def start_thread_for_open_symbol(symbol, args, manager, mfirsi_signal, has_open_long, has_open_short, long_mode, short_mode):
    action_taken = False
    if long_mode and (has_open_long or mfirsi_signal.lower() == "long"):
        action_taken |= start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "long")
        logging.info(f"[DEBUG] Started long thread for open symbol {symbol}")
    if short_mode and (has_open_short or mfirsi_signal.lower() == "short"):
        action_taken |= start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "short")
        logging.info(f"[DEBUG] Started short thread for open symbol {symbol}")
    return action_taken

def start_thread_for_symbol(symbol, args, manager, mfirsi_signal, action):
    if action == "long":
        if symbol in long_threads and long_threads[symbol][0].is_alive():
            logging.info(f"Long thread already running for symbol {symbol}. Skipping.")
            return False
    elif action == "short":
        if symbol in short_threads and short_threads[symbol][0].is_alive():
            logging.info(f"Short thread already running for symbol {symbol}. Skipping.")
            return False
    elif action == "neutral":
        logging.info(f"Start thread function hit for {symbol} but signal is {mfirsi_signal}")

    thread_completed = threading.Event()
    thread = threading.Thread(
        target=run_bot,
        args=(
            symbol, args, market_maker, manager, args.account_name,
            symbols_allowed, latest_rotator_symbols, thread_completed,
            mfirsi_signal, action
        )
    )

    if action == "long":
        long_threads[symbol] = (thread, thread_completed)
    elif action == "short":
        short_threads[symbol] = (thread, thread_completed)

    thread.start()
    logging.info(f"Started thread for symbol {symbol} with action {action} based on MFIRSI signal.")
    return True

def fetch_updated_symbols(args, manager):
    strategy = args.strategy.lower()
    potential_symbols = []

    if strategy == 'basicgrid':
        potential_bullish_symbols = manager.get_bullish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_bearish_symbols = manager.get_bearish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_symbols = potential_bullish_symbols + potential_bearish_symbols
    elif strategy == 'basicgridmfirsi':
        potential_bullish_symbols = manager.get_bullish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_bearish_symbols = manager.get_bearish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_symbols = potential_bullish_symbols + potential_bearish_symbols
    elif strategy == 'basicgridmfipersist':
        potential_bullish_symbols = manager.get_bullish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_bearish_symbols = manager.get_bearish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_symbols = potential_bullish_symbols + potential_bearish_symbols
    elif strategy == 'basicgridpersistnotional':
        potential_bullish_symbols = manager.get_bullish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_bearish_symbols = manager.get_bearish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        potential_symbols = potential_bullish_symbols + potential_bearish_symbols
    elif strategy == 'qstrendlongonly':
        potential_symbols = manager.get_bullish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
    elif strategy == 'qstrendshortonly':
        potential_symbols = manager.get_bearish_rotator_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
    else:
        potential_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)

    logging.info(f"Potential symbols for {strategy}: {potential_symbols}")
    return set(standardize_symbol(sym) for sym in potential_symbols)

def log_symbol_details(strategy, symbols):
    logging.info(f"Potential symbols for {strategy}: {symbols}")

def blofin_auto_rotation(args, market_maker, manager, symbols_allowed):
    market_maker.manager = manager
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in market_maker.exchange.get_all_open_positions_blofin()}
    logging.info(f"Open position symbols: {open_position_symbols}")

def hyperliquid_auto_rotation(args, manager, symbols_allowed):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in market_maker.exchange.get_all_open_positions_hyperliquid()}
    logging.info(f"Open position symbols: {open_position_symbols}")

def huobi_auto_rotation(args, manager, symbols_allowed):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in market_maker.exchange.get_all_open_positions_huobi()}
    logging.info(f"Open position symbols: {open_position_symbols}")

def bitget_auto_rotation(args, manager, symbols_allowed):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in market_maker.exchange.get_all_open_positions_bitget()}
    logging.info(f"Open position symbols: {open_position_symbols}")

def binance_auto_rotation(args, manager, symbols_allowed):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in market_maker.exchange.get_all_open_positions_binance()}
    logging.info(f"Open position symbols: {open_position_symbols}")

def mexc_auto_rotation(args, manager, symbols_allowed):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in market_maker.exchange.get_all_open_positions_binance()}
    logging.info(f"Open position symbols: {open_position_symbols}")

def lbank_auto_rotation(args, manager, symbols_allowed):
    open_position_symbols = {standardize_symbol(pos['symbol']) for pos in market_maker.exchange.get_all_open_positions_binance()}
    logging.info(f"Open position symbols: {open_position_symbols}")

if __name__ == '__main__':
    sword = "====||====>"

    print("\n" + "=" * 50)
    print(f"DirectionalScalper {VERSION}".center(50))
    print(f"Developed by Tyler Simpson and contributors at Quantum Void Labs".center(50))
    print("=" * 50 + "\n")

    print("Initializing", end="")
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")

    print("Battle-Ready Algorithm".center(50))
    print(sword.center(50) + "\n")

    parser = argparse.ArgumentParser(description='DirectionalScalper')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')
    parser.add_argument('--account_name', type=str, help='The name of the account to use')
    parser.add_argument('--exchange', type=str, help='The name of the exchange to use')
    parser.add_argument('--strategy', type=str, help='The name of the strategy to use')
    parser.add_argument('--symbol', type=str, help='The trading symbol to use')
    parser.add_argument('--amount', type=str, help='The size to use')

    args = parser.parse_args()
    args = ask_for_missing_arguments(args)

    print(f"DirectionalScalper {VERSION} Initialized Successfully!".center(50))
    print("=" * 50 + "\n")

    config_file_path = Path(args.config)
    account_path = Path('configs/account.json')

    try:
        config = load_config(config_file_path, account_path)
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        logging.error(f"There is probably an issue with your path. Try using --config configs/config.json")
        sys.exit(1)

    exchange_name = args.exchange
    try:
        market_maker = DirectionalMarketMaker(config, exchange_name, args.account_name)
    except Exception as e:
        logging.error(f"Failed to initialize market maker: {str(e)}")
        sys.exit(1)

    manager = Manager(
        market_maker.exchange,
        exchange_name=args.exchange,
        data_source_exchange=config.api.data_source_exchange,
        api=config.api.mode,
        path=Path("data", config.api.filename),
        url=f"{config.api.url}{config.api.filename}"
    )

    whitelist = config.bot.whitelist
    blacklist = config.bot.blacklist
    max_usd_value = config.bot.max_usd_value

    for exch in config.exchanges:
        if exch.name == exchange_name and exch.account_name == args.account_name:
            logging.info(f"Symbols allowed changed to symbols_allowed from config")
            symbols_allowed = exch.symbols_allowed
            break
    else:
        logging.info(f"Symbols allowed defaulted to 10")
        symbols_allowed = 10

    table_manager = LiveTableManager()
    display_thread = threading.Thread(target=table_manager.display_table)
    display_thread.daemon = True
    display_thread.start()

    # Removed redundant calls and initialization
    while True:
        try:
            whitelist = config.bot.whitelist
            blacklist = config.bot.blacklist
            max_usd_value = config.bot.max_usd_value

            match exchange_name.lower():
                case 'bybit':
                    bybit_auto_rotation(args, market_maker, manager, symbols_allowed)
                case 'bybit_spot':
                    bybit_auto_rotation_spot(args, market_maker, manager, symbols_allowed)
                case 'blofin':
                    blofin_auto_rotation(args, market_maker, manager, symbols_allowed)
                case 'hyperliquid':
                    hyperliquid_auto_rotation(args, market_maker, manager, symbols_allowed)
                case 'huobi':
                    huobi_auto_rotation(args, market_maker, manager, symbols_allowed)
                case 'bitget':
                    bitget_auto_rotation(args, market_maker, manager, symbols_allowed)
                case 'binance':
                    binance_auto_rotation(args, market_maker, manager, symbols_allowed)
                case 'mexc':
                    mexc_auto_rotation(args, market_maker, manager, symbols_allowed)
                case 'lbank':
                    lbank_auto_rotation(args, market_maker, manager, symbols_allowed)
                case _:
                    logging.warning(f"Auto-rotation not implemented for exchange: {exchange_name}")

            logging.info(f"Active symbols: {active_symbols}")
            logging.info(f"Total active symbols: {len(active_symbols)}")

            time.sleep(10)
        except Exception as e:
            logging.error(f"Exception caught in main loop: {e}")
            logging.debug(traceback.format_exc())