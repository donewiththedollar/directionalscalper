import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
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
import config
from config import load_config, Config
from config import VERSION
from api.manager import Manager

from directionalscalper.core.exchanges.blofin import BlofinExchange
from directionalscalper.core.exchanges.lbank import LBankExchange
from directionalscalper.core.exchanges.mexc import MexcExchange
from directionalscalper.core.exchanges.huobi import HuobiExchange
from directionalscalper.core.exchanges.bitget import BitgetExchange
from directionalscalper.core.exchanges.binance import BinanceExchange
from directionalscalper.core.exchanges.hyperliquid import HyperLiquidExchange
from directionalscalper.core.exchanges.bybit import BybitExchange
from directionalscalper.core.exchanges.exchange import Exchange

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
active_threads = []
long_threads = {}
short_threads = {}

threads = {}  # Threads for each symbol
thread_start_time = {}  # Dictionary to track the start time for each symbol's thread
symbol_last_started_time = {}

extra_symbols = set()  # To track symbols opened past the limit
under_review_symbols = set()

latest_rotator_symbols = set()
# last_rotator_update_time = 0
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
        'qstrendobdynamictp',
        'qsgridinstantsignal',
        'qsgridobtight',
        'qsgriddynamicstatic',
        'qsgridobdca',
        'qsgriddynmaicgridspaninstant',
        'qsdynamicgridspan',
        'qsgriddynamictplinspaced',
        'dynamicgridob',
        'dynamicgridobsratrp',
        'qsgriddynamictp',
        'qsgriddynamic',
        'qsgridbasic',
        'basicgridpersist',
        'qstrend',
        'qstrendob',
        'qstrenderi',
        'qstrendemas',
        'qstrend',
        'qsematrend',
        'qstrendemas',
        'mfieritrend',
        'qstrendlongonly',
        'qstrendshortonly',
        'qstrend_unified',
        'qstrendspot',
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


    def run_strategy(self, symbol, strategy_name, config, account_name, symbols_to_trade=None, rotator_symbols_standardized=None, mfirsi_signal=None):
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
            'qstrend': bybit_notional.BybitQuickScalpTrendNotional,
            'qsematrend': bybit_scalping.BybitQuickScalpEMATrend,
            'qstrend_dca': bybit_scalping.BybitQuickScalpTrendDCA,
            'mfieritrend': bybit_scalping.BybitMFIERILongShortTrend,
            'qstrendlongonly': bybit_scalping.BybitMFIRSIQuickScalpLong,
            'qstrendshortonly': bybit_scalping.BybitMFIRSIQuickScalpShort,
            'qstrend_unified': bybit_scalping.BybitQuickScalpUnified,
            'qstrendemas': bybit_notional.BybitQSTrendDoubleMANotional,
            'basicgrid': bybit_scalping.BybitBasicGrid,
            'qstrendspot': bybit_scalping.BybitQuickScalpTrendSpot,
            'basicgridpersist': bybit_notional.BybitBasicGridMFIRSIPersisentNotional,
            'qstrenderi': bybit_notional.BybitQuickScalpTrendERINotional,
            'qsgridnotional': bybit_notional.BybitQSGridNotional,
            'qsgridbasic': bybit_notional.BybitBasicGridBuffered,
            'qsgriddynamic': bybit_notional.BybitBasicGridBufferedQS,
            'qstrendob': bybit_notional.BybitQuickScalpTrendOBNotional,
            'qsgriddynamictp': bybit_notional.BybitBasicGridBufferedQSDTP,
            'qsgriddynamictplinspaced': bybit_notional.BybitDynamicGridDynamicTPLinSpaced,
            'qsdynamicgridspan': bybit_notional.BybitDynamicGridSpan,
            'dynamicgridob': bybit_notional.BybitDynamicGridSpanOB,
            'dynamicgridobsratrp': bybit_notional.BybitDynamicGridSpanOBSRATRP,
            'qsgriddynamicstatic': bybit_notional.BybitDynamicGridSpanOBSRStatic,
            'qsgridobdca': bybit_notional.BybitDynamicGridOBDCA,
            'qsgridinstantsignal': instant_signals.BybitDynamicGridSpanOBSRStaticIS,
            'qsgriddynmaicgridspaninstant' : instant_signals.BybitDynamicGridSpanIS,
            'qsgridobtight' : instant_signals.BybitDynamicGridSpanOBTight,
            'qsgridob' : instant_signals.BybitDynamicGridSpanOBLevels,
            'qstrendobdynamictp' : instant_signals.BybitQuickScalpTrendDynamicTP,
        }

        strategy_class = strategy_classes.get(strategy_name.lower())
        if strategy_class:
            strategy = strategy_class(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized, mfirsi_signal=mfirsi_signal)
        else:
            logging.error(f"Strategy {strategy_name} not found.")
            
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
            return self.exchange.symbols

    def get_mfirsi_signal(self, symbol):
        # Retrieve the MFI/RSI signal
        with general_rate_limiter:
            return self.exchange.get_mfirsi_ema_secondary_ema(symbol, limit=100, lookback=1, ema_period=5, secondary_ema_period=3)

BALANCE_REFRESH_INTERVAL = 600  # in seconds

orders_canceled = False

def run_bot(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized, thread_completed, mfirsi_signal):
    global orders_canceled
    current_thread = threading.current_thread()
    try:
        with thread_to_symbol_lock:
            thread_to_symbol[current_thread] = symbol

        if not args.config.startswith('configs/'):
            config_file_path = Path('configs/' + args.config)
        else:
            config_file_path = Path(args.config)

        print("Loading config from:", config_file_path)
        config = load_config(config_file_path)

        exchange_name = args.exchange
        strategy_name = args.strategy
        account_name = args.account_name

        logging.info(f"Trading symbol: {symbol}")
        logging.info(f"Exchange name: {exchange_name}")
        logging.info(f"Strategy name: {strategy_name}")
        logging.info(f"Account name: {account_name}")

        market_maker = DirectionalMarketMaker(config, exchange_name, account_name)
        market_maker.manager = manager

        try:
            if not orders_canceled and hasattr(market_maker.exchange, 'cancel_all_open_orders_bybit'):
                market_maker.exchange.cancel_all_open_orders_bybit()
                logging.info(f"Cleared all open orders on the exchange upon initialization.")
                orders_canceled = True
        except Exception as e:
            logging.info(f"Exception caught {e}")

        logging.info(f"Rotator symbols in run_bot {rotator_symbols_standardized}")
        logging.info(f"Latest rotator symbols in run bot {latest_rotator_symbols}")

        time.sleep(2)

        with general_rate_limiter:
            market_maker.run_strategy(symbol, args.strategy, config, account_name, symbols_to_trade=symbols_allowed, rotator_symbols_standardized=latest_rotator_symbols, mfirsi_signal=mfirsi_signal)

    except Exception as e:
        logging.info(f"An error occurred in run_bot for symbol {symbol}: {e}")
    finally:
        with thread_to_symbol_lock:
            if current_thread in thread_to_symbol:
                del thread_to_symbol[current_thread]
        logging.info(f"Thread for symbol {symbol} has completed.")
        thread_completed.set()

def bybit_auto_rotation(args, manager, symbols_allowed):
    global latest_rotator_symbols, long_threads, short_threads, active_symbols, last_rotator_update_time

    # Set max_workers to the number of CPUs
    max_workers_signals = 1
    max_workers_trading = 1

    signal_executor = ThreadPoolExecutor(max_workers=max_workers_signals)
    trading_executor = ThreadPoolExecutor(max_workers=max_workers_trading)
    
    logging.info(f"Initialized signal executor with max workers: {max_workers_signals}")
    logging.info(f"Initialized trading executor with max workers: {max_workers_trading}")

    config_file_path = Path('configs/' + args.config) if not args.config.startswith('configs/') else Path(args.config)
    config = load_config(config_file_path)
    logging.info(f"Loaded configuration from {config_file_path}.")

    market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name)
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

            current_long_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'long')
            current_short_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'short')
            logging.info(f"Current long positions: {current_long_positions}, Current short positions: {current_short_positions}")

            if not latest_rotator_symbols or current_time - last_rotator_update_time >= 60:
                with general_rate_limiter:
                    latest_rotator_symbols = fetch_updated_symbols(args, manager)
                last_rotator_update_time = current_time
                logging.info(f"Refreshed latest rotator symbols: {latest_rotator_symbols}")
            else:
                logging.debug(f"No refresh needed yet. Last update was at {last_rotator_update_time}, less than 60 seconds ago.")

            with thread_management_lock:
                update_active_symbols(open_position_symbols)
                logging.info(f"Active symbols updated. Symbols allowed: {symbols_allowed}")

                open_position_futures = []
                for symbol in open_position_symbols:
                    if symbol not in long_threads and symbol not in short_threads:
                        with general_rate_limiter:
                            mfirsi_signal = market_maker.get_mfirsi_signal(symbol)
                        has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
                        has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
                        open_position_futures.append(trading_executor.submit(start_thread_for_open_symbol, symbol, args, manager, mfirsi_signal, has_open_long, has_open_short, long_mode, short_mode))
                        logging.info(f"Submitted thread for symbol {symbol}. MFIRSI signal: {mfirsi_signal}. Has open long: {has_open_long}. Has open short: {has_open_short}.")

                signal_futures = [signal_executor.submit(process_signal_for_open_position, symbol, args, manager, symbols_allowed, open_position_data, long_mode, short_mode)
                                for symbol in open_position_symbols]
                logging.info(f"Submitted signal processing for open position symbols: {open_position_symbols}.")

                if len(active_symbols) < symbols_allowed:
                    for symbol in latest_rotator_symbols:
                        signal_futures.append(signal_executor.submit(process_signal, symbol, args, manager, symbols_allowed, open_position_data, False, long_mode, short_mode))
                        logging.info(f"Submitted signal processing for new rotator symbol {symbol}.")

                        time.sleep(2)

                process_futures(open_position_futures + signal_futures)

                completed_symbols = []
                for symbol, (thread, thread_completed) in {**long_threads, **short_threads}.items():
                    if thread_completed.is_set():
                        thread.join()
                        completed_symbols.append(symbol)

                for symbol in completed_symbols:
                    active_symbols.discard(symbol)
                    if symbol in long_threads:
                        del long_threads[symbol]
                    if symbol in short_threads:
                        del short_threads[symbol]
                    logging.info(f"Thread and symbol management completed for: {symbol}")

        except Exception as e:
            logging.error(f"Exception caught in bybit_auto_rotation: {str(e)}")
            logging.debug(traceback.format_exc())
        time.sleep(1)

def process_signal_for_open_position(symbol, args, manager, symbols_allowed, open_position_data, long_mode, short_mode):
    market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name)
    market_maker.manager = manager
    with general_rate_limiter:
        mfirsi_signal = market_maker.get_mfirsi_signal(symbol)
    logging.info(f"Processing signal for open position symbol {symbol}. MFIRSI signal: {mfirsi_signal}")

    action_taken = handle_signal(symbol, args, manager, mfirsi_signal, open_position_data, symbols_allowed, True, long_mode, short_mode)

    if action_taken:
        logging.info(f"Action taken for open position symbol {symbol}.")
    else:
        logging.info(f"No action taken for open position symbol {symbol}.")

def process_signal(symbol, args, manager, symbols_allowed, open_position_data, is_open_position, long_mode, short_mode):
    market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name)
    market_maker.manager = manager
    mfirsi_signal = market_maker.get_mfirsi_signal(symbol)
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

    logging.info(f"Handling signal for {'open position' if is_open_position else 'new rotator'} symbol {symbol}. Current long positions: {current_long_positions}. Current short positions: {current_short_positions}. Unique open symbols: {unique_open_symbols}")

    has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
    has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)

    logging.info(f"{'Open position' if is_open_position else 'New rotator'} symbol {symbol} - Has open long: {has_open_long}, Has open short: {has_open_short}")

    if is_open_position:
        if mfi_signal_long and not has_open_long and long_mode:
            return start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "long")
        elif mfi_signal_short and not has_open_short and short_mode:
            return start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "short")
    else:
        if unique_open_symbols < symbols_allowed:
            if mfi_signal_long and long_mode:
                return start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "long")
            elif mfi_signal_short and short_mode:
                return start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "short")

    logging.info(f"Evaluated action for {'open position' if is_open_position else 'new rotator'} symbol {symbol}: No action due to existing position or lack of clear signal.")
    return False

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
        process_signal(symbol, args, manager, symbols_allowed, open_position_data, True)

    for symbol in random_rotator_symbols:
        if len(open_position_symbols) >= symbols_allowed:
            logging.info("Maximum number of open positions reached.")
            break
        process_signal(symbol, args, manager, symbols_allowed, open_position_data, False)

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
    if has_open_long and long_mode:
        start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "long")
    if has_open_short and short_mode:
        start_thread_for_symbol(symbol, args, manager, mfirsi_signal, "short")

def start_thread_for_symbol(symbol, args, manager, mfirsi_signal, action):
    if action == "long":
        if symbol in long_threads and long_threads[symbol][0].is_alive():
            logging.info(f"Long thread already running for symbol {symbol}. Skipping.")
            return False
    elif action == "short":
        if symbol in short_threads and short_threads[symbol][0].is_alive():
            logging.info(f"Short thread already running for symbol {symbol}. Skipping.")
            return False

    thread_completed = threading.Event()
    thread = threading.Thread(target=run_bot, args=(symbol, args, manager, args.account_name, symbols_allowed, latest_rotator_symbols, thread_completed, mfirsi_signal))

    if action == "long":
        long_threads[symbol] = (thread, thread_completed)
    elif action == "short":
        short_threads[symbol] = (thread, thread_completed)

    thread.start()
    logging.info(f"Started thread for symbol {symbol} with action {action} based on MFIRSI signal.")
    return True

def fetch_updated_symbols(args, manager):
    """Fetches and logs potential symbols based on the current trading strategy."""
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

def bybit_spot_auto_rotation(args, manager, symbols_allowed):
    global latest_rotator_symbols, long_threads, short_threads, active_symbols, last_rotator_update_time

    # Set max_workers to the number of CPUs
    max_workers_signals = 1
    max_workers_trading = 1

    signal_executor = ThreadPoolExecutor(max_workers=max_workers_signals)
    trading_executor = ThreadPoolExecutor(max_workers=max_workers_trading)
    
    logging.info(f"Initialized signal executor with max workers: {max_workers_signals}")
    logging.info(f"Initialized trading executor with max workers: {max_workers_trading}")

    config_file_path = Path('configs/' + args.config) if not args.config.startswith('configs/') else Path(args.config)
    config = load_config(config_file_path)
    logging.info(f"Loaded configuration from {config_file_path}.")

    market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name)
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

            current_long_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'long')
            current_short_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'short')
            logging.info(f"Current long positions: {current_long_positions}, Current short positions: {current_short_positions}")

            if not latest_rotator_symbols or current_time - last_rotator_update_time >= 60:
                with general_rate_limiter:
                    latest_rotator_symbols = fetch_updated_symbols(args, manager)
                last_rotator_update_time = current_time
                logging.info(f"Refreshed latest rotator symbols: {latest_rotator_symbols}")
            else:
                logging.debug(f"No refresh needed yet. Last update was at {last_rotator_update_time}, less than 60 seconds ago.")

            with thread_management_lock:
                update_active_symbols(open_position_symbols)
                logging.info(f"Active symbols updated. Symbols allowed: {symbols_allowed}")

                open_position_futures = []
                for symbol in open_position_symbols:
                    if symbol not in long_threads and symbol not in short_threads:
                        with general_rate_limiter:
                            mfirsi_signal = market_maker.get_mfirsi_signal(symbol)
                        has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
                        has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
                        open_position_futures.append(trading_executor.submit(start_thread_for_open_symbol, symbol, args, manager, mfirsi_signal, has_open_long, has_open_short, long_mode, short_mode))
                        logging.info(f"Submitted thread for symbol {symbol}. MFIRSI signal: {mfirsi_signal}. Has open long: {has_open_long}. Has open short: {has_open_short}.")

                signal_futures = [signal_executor.submit(process_signal_for_open_position, symbol, args, manager, symbols_allowed, open_position_data, long_mode, short_mode)
                                for symbol in open_position_symbols]
                logging.info(f"Submitted signal processing for open position symbols: {open_position_symbols}.")

                if len(active_symbols) < symbols_allowed:
                    for symbol in latest_rotator_symbols:
                        signal_futures.append(signal_executor.submit(process_signal, symbol, args, manager, symbols_allowed, open_position_data, False, long_mode, short_mode))
                        logging.info(f"Submitted signal processing for new rotator symbol {symbol}.")

                        time.sleep(2)

                process_futures(open_position_futures + signal_futures)

                completed_symbols = []
                for symbol, (thread, thread_completed) in {**long_threads, **short_threads}.items():
                    if thread_completed.is_set():
                        thread.join()
                        completed_symbols.append(symbol)

                for symbol in completed_symbols:
                    active_symbols.discard(symbol)
                    if symbol in long_threads:
                        del long_threads[symbol]
                    if symbol in short_threads:
                        del short_threads[symbol]
                    logging.info(f"Thread and symbol management completed for: {symbol}")

        except Exception as e:
            logging.error(f"Exception caught in bybit_spot_auto_rotation: {str(e)}")
            logging.debug(traceback.format_exc())
        time.sleep(1)

def blofin_auto_rotation(args, manager, symbols_allowed):
    market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name)
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

    if not args.config.startswith('configs/'):
        config_file_path = Path('configs/' + args.config)
    else:
        config_file_path = Path(args.config)

    config = load_config(config_file_path)
    exchange_name = args.exchange
    market_maker = DirectionalMarketMaker(config, exchange_name, args.account_name)

    manager = Manager(
        market_maker.exchange,
        exchange_name=args.exchange,
        data_source_exchange=config.api.data_source_exchange,
        api=config.api.mode,
        path=Path("data", config.api.filename),
        url=f"{config.api.url}{config.api.filename}"
    )

    print(f"Using exchange {config.api.data_source_exchange} for API data")

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

    all_symbols_standardized = [standardize_symbol(symbol) for symbol in manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)]
    open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    open_positions_symbols = [standardize_symbol(position['symbol']) for position in open_position_data]

    print(f"Open positions symbols: {open_positions_symbols}")
    symbols_to_trade = list(set(open_positions_symbols + all_symbols_standardized[:symbols_allowed]))

    print(f"Symbols to trade: {symbols_to_trade}")

    while True:
        try:
            whitelist = config.bot.whitelist
            blacklist = config.bot.blacklist
            max_usd_value = config.bot.max_usd_value

            if exchange_name.lower() == 'bybit':
                bybit_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'bybit_spot':
                bybit_spot_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'blofin':
                blofin_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'hyperliquid':
                hyperliquid_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'huobi':
                huobi_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'bitget':
                bitget_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'binance':
                binance_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'mexc':
                mexc_auto_rotation(args, manager, symbols_allowed)
            elif exchange_name.lower() == 'lbank':
                lbank_auto_rotation(args, manager, symbols_allowed)
            else:
                logging.warning(f"Auto-rotation not implemented for exchange: {exchange_name}")

            logging.info(f"Active symbols: {active_symbols}")
            logging.info(f"Total active symbols: {len(active_symbols)}")

            time.sleep(10)
        except Exception as e:
            logging.info(f"Exception caught in main loop: {e}")
            logging.info(traceback.format_exc())
