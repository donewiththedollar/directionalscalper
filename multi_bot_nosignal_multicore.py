import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import colorama
from colorama import Fore, Style
from pathlib import Path
import os

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

from directionalscalper.core.exchanges.lbank import LBankExchange
from directionalscalper.core.exchanges.mexc import MexcExchange
from directionalscalper.core.exchanges.huobi import HuobiExchange
from directionalscalper.core.exchanges.bitget import BitgetExchange
from directionalscalper.core.exchanges.binance import BinanceExchange
from directionalscalper.core.exchanges.hyperliquid import HyperLiquidExchange
from directionalscalper.core.exchanges.bybit import BybitExchange
from directionalscalper.core.exchanges.exchange import Exchange


import directionalscalper.core.strategies.bybit.nosignal.hotkeys_base_strategy as hotkeysbase
import directionalscalper.core.strategies.bybit.nosignal.dynamicgrid_oblevels_nosignal as nosignalob
import directionalscalper.core.strategies.bybit.notional as bybit_notional
import directionalscalper.core.strategies.bybit.scalping as bybit_scalping
import directionalscalper.core.strategies.bybit.hedging as bybit_hedging
from directionalscalper.core.strategies.binance import *
from directionalscalper.core.strategies.huobi import *

from live_table_manager import LiveTableManager, shared_symbols_data


from directionalscalper.core.strategies.logger import Logger

from collections import deque

thread_management_lock = threading.Lock()
thread_to_symbol = {}
thread_to_symbol_lock = threading.Lock()
active_symbols = set()
active_threads = []

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
        'longonlyhftob',
        'hotkeysmanual',
        'qsgridnosignalstatic',
        'qsgriddynamicstatic',
        'qsgridobdca',
        'qsdynamicgridspan',
        'qsgriddynamictplinspaced',
        'dynamicgridob',
        'dynamicgridobsr',
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
        exchange_config = None

        for exch in config.exchanges:
            if exch.name == exchange_name and exch.account_name == account_name:  # Check both fields
                exchange_config = exch
                break

        if not exchange_config:
            raise ValueError(f"Exchange {exchange_name} with account {account_name} not found in the configuration file.")
        
        api_key = exchange_config.api_key
        secret_key = exchange_config.api_secret
        passphrase = exchange_config.passphrase
        
        if exchange_name.lower() == 'bybit':
            market_type = 'swap'
            self.exchange = BybitExchange(api_key, secret_key, passphrase, market_type)
        elif exchange_name.lower() == 'bybit_spot':
            market_type = 'spot'
            self.exchange = BybitExchange(api_key, secret_key, passphrase, market_type)
        elif exchange_name.lower() == 'hyperliquid':
            self.exchange = HyperLiquidExchange(api_key, secret_key, passphrase)
        elif exchange_name.lower() == 'huobi':
            self.exchange = HuobiExchange(api_key, secret_key, passphrase)
        elif exchange_name.lower() == 'bitget':
            self.exchange = BitgetExchange(api_key, secret_key, passphrase)
        elif exchange_name.lower() == 'binance':
            self.exchange = BinanceExchange(api_key, secret_key, passphrase)
        elif exchange_name.lower() == 'mexc':
            self.exchange = MexcExchange(api_key, secret_key, passphrase)
        elif exchange_name.lower() == 'lbank':
            self.exchange = LBankExchange(api_key, secret_key, passphrase)
        else:
            self.exchange = Exchange(self.exchange_name, api_key, secret_key, passphrase)

    def run_strategy(self, symbol, strategy_name, config, account_name, symbols_to_trade=None, rotator_symbols_standardized=None):
        logging.info(f"Received rotator symbols in run_strategy for {symbol}: {rotator_symbols_standardized}")
        symbols_allowed = None
        for exch in config.exchanges:
            if exch.name == self.exchange_name and exch.account_name == account_name:
                symbols_allowed = exch.symbols_allowed
                print(f"Matched exchange: {exchange_name}, account: {args.account_name}. Symbols allowed: {symbols_allowed}")
                break

        print(f"Multibot.py: symbols_allowed from config: {symbols_allowed}")
        
        if symbols_to_trade:
            print(f"Calling run method with symbols: {symbols_to_trade}")

            try:
                print_cool_trading_info(symbol, exchange_name, strategy_name, account_name)
                logging.info(f"Printed trading info for {symbol}")
            except Exception as e:
                logging.info(f"Error in printing info: {e}")

        if strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_walls':
            strategy = bybit_scalping.BybitMMOneMinuteQFLMFIERIWalls(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_autohedge_walls_atr':
            strategy = bybit_hedging.BybitMMOneMinuteQFLMFIERIAutoHedgeWallsATR(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mfirsi_imbalance':
            strategy = bybit_scalping.BybitMFIRSIERIOBImbalance(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mfirsi_quickscalp':
            strategy = bybit_scalping.BybitMFIRSIQuickScalp(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrend':
            strategy = bybit_notional.BybitQuickScalpTrendNotional(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsematrend':
            strategy = bybit_scalping.BybitQuickScalpEMATrend(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrend_dca':
            strategy = bybit_scalping.BybitQuickScalpTrendDCA(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'mfieritrend':
            strategy = bybit_scalping.BybitMFIERILongShortTrend(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrendlongonly':
            strategy = bybit_scalping.BybitMFIRSIQuickScalpLong(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrendshortonly':
            strategy = bybit_scalping.BybitMFIRSIQuickScalpShort(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrend_unified':
            strategy = bybit_scalping.BybitQuickScalpUnified(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrendemas':
            strategy = bybit_notional.BybitQSTrendDoubleMANotional(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'basicgrid':
            strategy = bybit_scalping.BybitBasicGrid(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrendspot':
            strategy = bybit_scalping.BybitQuickScalpTrendSpot(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'basicgridpersist':
            strategy = bybit_notional.BybitBasicGridMFIRSIPersisentNotional(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrenderi':
            strategy = bybit_notional.BybitQuickScalpTrendERINotional(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgridnotional':
            strategy = bybit_notional.BybitQSGridNotional(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgridbasic':
            strategy = bybit_notional.BybitBasicGridBuffered(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgriddynamic':
            strategy = bybit_notional.BybitBasicGridBufferedQS(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qstrendob':
            strategy = bybit_notional.BybitQuickScalpTrendOBNotional(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgriddynamictp':
            strategy = bybit_notional.BybitBasicGridBufferedQSDTP(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgriddynamictplinspaced':
            strategy = bybit_notional.BybitDynamicGridDynamicTPLinSpaced(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsdynamicgridspan':
            strategy = bybit_notional.BybitDynamicGridSpan(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'dynamicgridob':
            strategy = bybit_notional.BybitDynamicGridSpanOB(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'dynamicgridobsr':
            strategy = bybit_notional.BybitDynamicGridSpanOBSR(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'dynamicgridobsratrp':
            strategy = bybit_notional.BybitDynamicGridSpanOBSRATRP(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgriddynamicstatic':
            strategy = bybit_notional.BybitDynamicGridSpanOBSRStatic(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgridobdca':
            strategy = bybit_notional.BybitDynamicGridOBDCA(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'qsgridnosignalstatic':
            strategy = bybit_notional.BybitDynamicGridSpanOBSRStaticNoSignal(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'longonlyhftob':
            strategy = nosignalob.BybitDynamicGridSpanOBLevelsNoSignal(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'hotkeysmanual':
            strategy = hotkeysbase.BybitHotkeysBaseStrategy(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
                          
    def get_balance(self, quote, market_type=None, sub_type=None):
        if self.exchange_name == 'bitget':
            return self.exchange.get_balance_bitget(quote)
        elif self.exchange_name == 'bybit':
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
        return self.exchange.symbols

    def get_mfirsi_signal(self, symbol):
        return self.exchange.get_mfirsi_ema_secondary_ema(symbol, limit=100, lookback=1, ema_period=5, secondary_ema_period=3)



BALANCE_REFRESH_INTERVAL = 600  # in seconds

orders_canceled = False

def run_bot(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized, thread_completed):
    global orders_canceled
    current_thread = threading.current_thread()
    try:
        with thread_to_symbol_lock:
            thread_to_symbol[current_thread] = symbol
            time.sleep(1)

        if not args.config.startswith('configs/'):
            config_file_path = Path('configs/' + args.config)
        else:
            config_file_path = Path(args.config)

        print("Loading config from:", config_file_path)
        config = load_config(config_file_path)

        exchange_name = args.exchange
        strategy_name = args.strategy
        account_name = args.account_name

        print(f"Trading symbol: {symbol}")
        print(f"Exchange name: {exchange_name}")
        print(f"Strategy name: {strategy_name}")
        print(f"Account name: {account_name}")

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

        market_maker.run_strategy(symbol, args.strategy, config, account_name, symbols_to_trade=symbols_allowed, rotator_symbols_standardized=latest_rotator_symbols)

        thread_completed.set()

    except Exception as e:
        logging.info(f"An error occurred in run_bot for symbol {symbol}: {e}")
        thread_completed.set()

    finally:
        with thread_to_symbol_lock:
            if current_thread in thread_to_symbol:
                del thread_to_symbol[current_thread]
        logging.info(f"Thread for symbol {symbol} has completed.")

def bybit_spot_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value):
    global latest_rotator_symbols, threads, active_symbols, last_rotator_update_time

    max_workers_signals = 1
    max_workers_trading = 1

    signal_executor = ThreadPoolExecutor(max_workers=max_workers_signals)
    trading_executor = ThreadPoolExecutor(max_workers=max_workers_trading)

    while True:
        try:
            current_time = time.time()

            open_position_data = manager.exchange.get_all_open_positions_bybit_spot()
            open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
            logging.info(f"Open position symbols: {open_position_symbols}")

            if not latest_rotator_symbols or current_time - last_rotator_update_time >= 60:
                latest_rotator_symbols = fetch_updated_symbols(args, manager, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
                last_rotator_update_time = current_time
                logging.info(f"Refreshed latest rotator symbols: {latest_rotator_symbols}")
            else:
                logging.info(f"No refresh needed yet. Last update was at {last_rotator_update_time}, less than 60 seconds ago.")

            with thread_management_lock:
                update_active_symbols()

                logging.info(f"Symbols allowed: {symbols_allowed}")

                open_position_futures = []
                for symbol in open_position_symbols:
                    open_position_futures.append(trading_executor.submit(start_thread_for_symbol, symbol, args, manager))

                for future in as_completed(open_position_futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.info(f"Exception in thread: {e}")
                        logging.info(traceback.format_exc())

                signal_futures = []
                for symbol in latest_rotator_symbols:
                    signal_futures.append(signal_executor.submit(process_signal, symbol, args, manager, symbols_allowed, open_position_data))

                for future in as_completed(signal_futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.info(f"Exception in signal processing: {e}")
                        logging.info(traceback.format_exc())

                completed_symbols = []
                for symbol, (thread, thread_completed) in threads.items():
                    if thread_completed.is_set():
                        thread.join()
                        completed_symbols.append(symbol)

                for symbol in completed_symbols:
                    active_symbols.discard(symbol)
                    del threads[symbol]
                    del thread_start_time[symbol]
                    logging.info(f"Thread and symbol management completed for: {symbol}")

        except Exception as e:
            logging.info(f"Exception caught in bybit_spot_auto_rotation: {str(e)}")
            logging.info(traceback.format_exc())
        time.sleep(1)

def bybit_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value):
    global latest_rotator_symbols, threads, active_symbols, last_rotator_update_time

    max_workers_signals = 1
    max_workers_trading = 1

    signal_executor = ThreadPoolExecutor(max_workers=max_workers_signals)
    trading_executor = ThreadPoolExecutor(max_workers=max_workers_trading)

    while True:
        try:
            current_time = time.time()

            open_position_data = getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()
            open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}
            logging.info(f"Open position symbols: {open_position_symbols}")

            if not latest_rotator_symbols or current_time - last_rotator_update_time >= 60:
                latest_rotator_symbols = fetch_updated_symbols(args, manager, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
                last_rotator_update_time = current_time
                logging.info(f"Refreshed latest rotator symbols: {latest_rotator_symbols}")
            else:
                logging.info(f"No refresh needed yet. Last update was at {last_rotator_update_time}, less than 60 seconds ago.")

            with thread_management_lock:
                update_active_symbols()

                logging.info(f"Symbols allowed: {symbols_allowed}")

                open_position_futures = []
                for symbol in open_position_symbols:
                    open_position_futures.append(trading_executor.submit(start_thread_for_symbol, symbol, args, manager))

                for future in as_completed(open_position_futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.info(f"Exception in thread: {e}")
                        logging.info(traceback.format_exc())

                signal_futures = []
                for symbol in latest_rotator_symbols:
                    signal_futures.append(signal_executor.submit(process_signal, symbol, args, manager, symbols_allowed, open_position_data))

                for future in as_completed(signal_futures):
                    try:
                        future.result()
                    except Exception as e:
                        logging.info(f"Exception in signal processing: {e}")
                        logging.info(traceback.format_exc())

                completed_symbols = []
                for symbol, (thread, thread_completed) in threads.items():
                    if thread_completed.is_set():
                        thread.join()
                        completed_symbols.append(symbol)

                for symbol in completed_symbols:
                    active_symbols.discard(symbol)
                    del threads[symbol]
                    del thread_start_time[symbol]
                    logging.info(f"Thread and symbol management completed for: {symbol}")

        except Exception as e:
            logging.info(f"Exception caught in bybit_auto_rotation: {str(e)}")
            logging.info(traceback.format_exc())
        time.sleep(1)

def process_signal(symbol, args, manager, symbols_allowed, open_position_data):
    market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name)
    market_maker.manager = manager
    mfirsi_signal = market_maker.get_mfirsi_signal(symbol)
    logging.info(f"MFIRSI signal for {symbol}: {mfirsi_signal}")

    mfi_signal_long = mfirsi_signal.lower() == "long"
    mfi_signal_short = mfirsi_signal.lower() == "short"

    current_long_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'long')
    current_short_positions = sum(1 for pos in open_position_data if pos['side'].lower() == 'short')
    logging.info(f"Current long positions: {current_long_positions}, Current short positions: {current_short_positions}")

    has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
    has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)

    logging.info(f"{symbol} - Open Long: {has_open_long}, Open Short: {has_open_short}")

    action_taken = False
    if not has_open_long and current_long_positions < symbols_allowed:
        action = "long"
        action_desc = "starting a new long position"
        current_long_positions += 1
    elif not has_open_short and current_short_positions < symbols_allowed:
        action = "short"
        action_desc = "starting a new short position"
        current_short_positions += 1
    else:
        action = None
        action_desc = "no action due to existing position or lack of clear signal"

    logging.info(f"Evaluated action for {symbol}: {action_desc}")

    if action and (symbol not in threads or not threads[symbol][0].is_alive()):
        if start_thread_for_symbol(symbol, args, manager):
            active_symbols.add(symbol)
            logging.info(f"Successfully started thread for {symbol} based on '{action}' signal.")
            action_taken = True
    else:
        logging.info(f"No thread started for {symbol}: {action_desc}")

    if not action_taken:
        logging.info(f"No action taken for {symbol}.")

def update_active_symbols():
    global active_symbols
    active_symbols = {symbol for symbol in active_symbols if symbol in threads and threads[symbol][0].is_alive()}

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

    def process_symbol(symbol):
        try:
            nonlocal current_long_positions, current_short_positions

            market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name)
            market_maker.manager = manager
            mfirsi_signal = market_maker.get_mfirsi_signal(symbol)
            logging.info(f"MFIRSI signal for {symbol}: {mfirsi_signal}")

            mfi_signal_long = mfirsi_signal.lower() == "long"
            mfi_signal_short = mfirsi_signal.lower() == "short"

            has_open_long = any(pos['side'].lower() == 'long' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)
            has_open_short = any(pos['side'].lower() == 'short' for pos in open_position_data if standardize_symbol(pos['symbol']) == symbol)

            logging.info(f"{symbol} - Open Long: {has_open_long}, Open Short: {has_open_short}")

            action_taken = False
            if not has_open_long and current_long_positions + current_short_positions < symbols_allowed * 2:
                action = "long"
                action_desc = "starting a new long position"
                current_long_positions += 1
            else:
                action = None
                action_desc = "no action due to existing position or lack of clear signal"

            logging.info(f"Evaluated action for {symbol}: {action_desc}")

            if action and (symbol not in threads or not threads[symbol][0].is_alive()):
                if start_thread_for_symbol(symbol, args, manager):
                    active_symbols.add(symbol)
                    logging.info(f"Successfully started thread for {symbol} based on '{action}' signal.")
                    action_taken = True
            else:
                logging.info(f"No thread started for {symbol}: {action_desc}")

            if not action_taken:
                logging.info(f"No action taken for {symbol}.")
        except Exception as e:
            logging.info(f"Exception caught in process symbol {e}")
            
        while True:
            open_position_data = getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()
            open_position_symbols = {standardize_symbol(pos['symbol']) for pos in open_position_data}

            if current_long_positions + current_short_positions < symbols_allowed * 2:
                for symbol in open_position_symbols:
                    process_symbol(symbol)

            for symbol in random_rotator_symbols:
                if current_long_positions + current_short_positions >= symbols_allowed * 2:
                    logging.info("Maximum number of long and short positions reached.")
                    break
                process_symbol(symbol)

            manage_excess_threads(symbols_allowed)
            
            time.sleep(3)

def manage_excess_threads(symbols_allowed):
    global active_symbols
    long_positions = {symbol for symbol in active_symbols if is_long_position(symbol)}
    short_positions = {symbol for symbol in active_symbols if is_short_position(symbol)}

    logging.info(f"Manage excess threads: Total long positions: {len(long_positions)}, Total short positions: {len(short_positions)}")

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
    for pos in pos_data:
        if standardize_symbol(pos['symbol']) == symbol and pos['side'].lower() == 'long':
            return True
    return False

def is_short_position(symbol):
    pos_data = getattr(manager.exchange, f"get_all_open_positions_{args.exchange.lower()}")()
    for pos in pos_data:
        if standardize_symbol(pos['symbol']) == symbol and pos['side'].lower() == 'short':
            return True
    return False

def remove_thread_for_symbol(symbol):
    thread, thread_completed = threads.get(symbol, (None, None))
    if thread:
        thread_completed.set()
        thread.join()
    threads.pop(symbol, None)

def start_thread_for_symbol(symbol, args, manager):
    logging.info(f"Starting thread for symbol: {symbol}")
    try:
        thread_completed = threading.Event()
        new_thread = threading.Thread(target=run_bot, args=(symbol, args, manager, args.account_name, symbols_allowed, latest_rotator_symbols, thread_completed))
        new_thread.start()
        threads[symbol] = (new_thread, thread_completed)
        thread_start_time[symbol] = time.time()
        return True
    except Exception as e:
        logging.error(f"Error starting thread for symbol {symbol}: {e}")
        return False

def fetch_updated_symbols(args, manager, blacklist=None, whitelist=None, max_usd_value=None):
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
    elif strategy == 'longonlyhftob':
        potential_symbols = manager.get_everything(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)
        logging.info(f"Potential symbols everything: {potential_symbols}")
        logging.info(f"Getting everything for all symbols")
    else:
        potential_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, whitelist=whitelist, max_usd_value=max_usd_value)

    logging.info(f"Potential symbols for {strategy}: {potential_symbols}")
    return set(standardize_symbol(sym) for sym in potential_symbols)

def log_symbol_details(strategy, symbols):
    if strategy in ['basicgrid', 'basicgridmfirsi', 'basicgridmfipersist', 'basicgridpersistnotional', 'qstrendlongonly', 'qstrendshortonly']:
        logging.info(f"Potential symbols for {strategy}: {symbols}")
    else:
        logging.info(f"Other strategy symbols: {symbols}")
        
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
                bybit_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            elif exchange_name.lower() == 'bybit_spot':
                bybit_spot_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            elif exchange_name.lower() == 'hyperliquid':
                hyperliquid_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            elif exchange_name.lower() == 'huobi':
                huobi_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            elif exchange_name.lower() == 'bitget':
                bitget_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            elif exchange_name.lower() == 'binance':
                binance_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            elif exchange_name.lower() == 'mexc':
                mexc_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            elif exchange_name.lower() == 'lbank':
                lbank_auto_rotation(args, manager, symbols_allowed, whitelist, blacklist, max_usd_value)
            else:
                logging.warning(f"Auto-rotation not implemented for exchange: {exchange_name}")

            logging.info(f"Active symbols: {active_symbols}")
            logging.info(f"Total active symbols: {len(active_symbols)}")

            time.sleep(5)
        except Exception as e:
            logging.info(f"Exception caught in main loop: {e}")
            logging.info(traceback.format_exc())
