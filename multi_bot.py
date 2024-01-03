import sys
import time
import threading
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent)
print("Project directory:", project_dir)
sys.path.insert(0, project_dir)

import inquirer
from rich.live import Live
import argparse
from pathlib import Path
import config
from config import load_config, Config
from config import VERSION
from api.manager import Manager
from directionalscalper.core.exchange import Exchange
from directionalscalper.core.strategies.strategy import Strategy


import directionalscalper.core.strategies.bybit.scalping as bybit_scalping
import directionalscalper.core.strategies.bybit.hedging as bybit_hedging
from directionalscalper.core.strategies.binance import *
from directionalscalper.core.strategies.huobi import *

from live_table_manager import LiveTableManager, shared_symbols_data


from directionalscalper.core.strategies.logger import Logger

thread_to_symbol = {}
thread_to_symbol_lock = threading.Lock()
active_symbols = set()
active_threads = []


logging = Logger(logger_name="MultiBot", filename="MultiBot.log", stream=True)

def standardize_symbol(symbol):
    return symbol.replace('/', '').split(':')[0]

def get_available_strategies():
    return [
        'bybit_1m_qfl_mfi_eri_walls',
        'bybit_1m_qfl_mfi_eri_autohedge_walls_atr',
        'bybit_mfirsi_imbalance',
        'bybit_mfirsi_quickscalp'
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
    return ['bybit', 'bitget', 'mexc', 'huobi', 'okx', 'binance', 'phemex']

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
        self.exchange = Exchange(self.exchange_name, api_key, secret_key, passphrase)

    def run_strategy(self, symbol, strategy_name, config, account_name, symbols_to_trade=None, rotator_symbols_standardized=None):
        symbols_allowed = None
        for exch in config.exchanges:
            #print(f"Checking: {exch.name} vs {self.exchange_name} and {exch.account_name} vs {account_name}")
            if exch.name == self.exchange_name and exch.account_name == account_name:
                symbols_allowed = exch.symbols_allowed
                print(f"Matched exchange: {exchange_name}, account: {args.account_name}. Symbols allowed: {symbols_allowed}")
                break

        print(f"Multibot.py: symbols_allowed from config: {symbols_allowed}")
        
        if symbols_to_trade:
            print(f"Calling run method with symbols: {symbols_to_trade}")

        # Pass symbols_allowed to the strategy constructors
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
        return self.exchange.symbols


BALANCE_REFRESH_INTERVAL = 600  # in seconds

def run_bot(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized):
    current_thread = threading.current_thread()

    try:
        with thread_to_symbol_lock:
            thread_to_symbol[current_thread] = symbol

            time.sleep(1)

        # Correct the path for the configuration file
        if not args.config.startswith('configs/'):
            config_file_path = Path('configs/' + args.config)
        else:
            config_file_path = Path(args.config)
        print("Loading config from:", config_file_path)
        config = load_config(config_file_path)

        # Initialize balance cache and last fetch time at the beginning
        cached_balance = None
        last_balance_fetch_time = 0

        exchange_name = args.exchange  # These are now guaranteed to be non-None
        strategy_name = args.strategy
        account_name = args.account_name  # Get the account_name from args

        print(f"Symbol: {symbol}")
        print(f"Exchange name: {exchange_name}")
        print(f"Strategy name: {strategy_name}")
        print(f"Account name: {account_name}") 

        # Pass account_name to DirectionalMarketMaker constructor
        market_maker = DirectionalMarketMaker(config, exchange_name, account_name)
        market_maker.manager = manager
        
        # Pass rotator_symbols_standardized to the run_strategy method
        market_maker.run_strategy(symbol, strategy_name, config, account_name, symbols_to_trade=symbols_allowed, rotator_symbols_standardized=rotator_symbols_standardized)

        quote = "USDT"
        current_time = time.time()
        if current_time - last_balance_fetch_time > BALANCE_REFRESH_INTERVAL or not cached_balance:
            if exchange_name.lower() == 'huobi':
                print(f"Loading huobi strategy..")
            elif exchange_name.lower() == 'mexc':
                cached_balance = market_maker.get_balance(quote, type='swap')
                print(f"Futures balance: {cached_balance}")
            else:
                cached_balance = market_maker.get_balance(quote)
                print(f"Futures balance: {cached_balance}")
            last_balance_fetch_time = current_time
    except Exception as e:
        logging.error(f"An error occurred in run_bot for symbol {symbol}: {e}")

    finally:
        with thread_to_symbol_lock:
            if current_thread in thread_to_symbol:
                del thread_to_symbol[current_thread]
        logging.info(f"Thread for symbol {symbol} has completed.")

def start_thread_for_symbol(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized):
    if symbol in active_symbols:
        logging.info(f"Symbol {symbol} is already being processed by another thread.")
        return

    thread = threading.Thread(target=run_bot, args=(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized))
    thread.start()
    active_threads.append(thread)
    thread_to_symbol[thread] = symbol  # Associate the thread with the symbol
    active_symbols.add(symbol)  # Mark the symbol as active
    logging.info(f"Started thread for symbol: {symbol}")


def start_threads_for_new_symbols(new_symbols, args, manager, account_name, symbols_allowed, rotator_symbols_standardized):
    for symbol in new_symbols:
        start_thread_for_symbol(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized)

if __name__ == '__main__':
    # ASCII Art and Text
    sword = "====||====>"

    print("\n" + "=" * 50)
    print(f"DirectionalScalper {VERSION}".center(50))
    print(f"Developed by Tyler Simpson and contributors at TradeSimple".center(50))
    print("=" * 50 + "\n")

    print("Initializing", end="")
    # Loading animation
    for i in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print("\n")

    # Display the ASCII art
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

    # Correct the path for the configuration file
    if not args.config.startswith('configs/'):
        config_file_path = Path('configs/' + args.config)
    else:
        config_file_path = Path(args.config)

    config = load_config(config_file_path)

    # config_file_path = Path('configs/' + args.config)
    # config = load_config(config_file_path)

    exchange_name = args.exchange  # Now it will have a value
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

    # whitelist = config.bot.whitelist
    blacklist = config.bot.blacklist
    max_usd_value = config.bot.max_usd_value

    # symbols_allowed = config.bot.symbols_allowed
  
    # Loop through the exchanges to find the correct exchange and account name
    for exch in config.exchanges:
        if exch.name == exchange_name and exch.account_name == args.account_name:
            logging.info(f"Symbols allowed changed to symbols_allowed from config")
            symbols_allowed = exch.symbols_allowed
            break
    else:
        # Default to a reasonable value if symbols_allowed is None
        logging.info(f"Symbols allowed defaulted to 10")
        symbols_allowed = 10  # You can choose an appropriate default value

    ### ILAY ###
    table_manager = LiveTableManager()
    display_thread = threading.Thread(target=table_manager.display_table)
    display_thread.daemon = True
    display_thread.start()
    ### ILAY ###

    # Fetch all symbols that meet your criteria and standardize them
    all_symbols_standardized = [standardize_symbol(symbol) for symbol in manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)]

    # Get symbols with open positions and standardize them
    open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    open_positions_symbols = [standardize_symbol(position['symbol']) for position in open_position_data]

    print(f"Open positions symbols: {open_positions_symbols}")

    # Combine open positions symbols with potential new symbols
    symbols_to_trade = list(set(open_positions_symbols + all_symbols_standardized[:symbols_allowed]))

    print(f"Symbols to trade: {symbols_to_trade}")

    symbol_last_started_time = {}
    extra_symbols = set()  # To track symbols opened past the limit

    while True:
        # Update active threads and symbols
        completed_threads = [t for t in active_threads if not t.is_alive()]
        for t in completed_threads:
            symbol = thread_to_symbol.pop(t, None)
            if symbol:
                active_symbols.discard(symbol)
                extra_symbols.discard(symbol)
            active_threads.remove(t)

        # Fetch and process open position data
        open_position_data = market_maker.exchange.get_all_open_positions_bybit()
        unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}

        # Update the active symbols and extra symbols based on unique open position symbols
        active_symbols = active_symbols.intersection(unique_open_position_symbols)
        extra_symbols = extra_symbols.intersection(unique_open_position_symbols)

        # Fetch rotator symbols
        rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
        rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

        # Start threads for symbols in unique open positions
        for symbol in unique_open_position_symbols:
            if symbol not in active_symbols and len(active_symbols) + len(extra_symbols) < symbols_allowed:
                start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
                active_symbols.add(symbol)

        # Handle extra symbols
        for symbol in extra_symbols:
            if symbol not in active_symbols and len(active_symbols) < symbols_allowed:
                start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
                active_symbols.add(symbol)
                extra_symbols.discard(symbol)

        # Start threads for rotator symbols if slots are available
        available_slots = max(0, symbols_allowed - len(active_symbols) - len(extra_symbols))
        for symbol in rotator_symbols_standardized:
            if symbol not in active_symbols and available_slots > 0:
                start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
                active_symbols.add(symbol)
                available_slots -= 1

        logging.info(f"Open symbols: {unique_open_position_symbols}")
        logging.info(f"Active symbols: {active_symbols}")
        logging.info(f"Extra symbols: {extra_symbols}")
        logging.info(f"Total active symbols: {len(active_symbols)}")
        time.sleep(60)
        
    # while True:
    #     # Update active threads and symbols
    #     completed_threads = [t for t in active_threads if not t.is_alive()]
    #     for t in completed_threads:
    #         symbol = thread_to_symbol.pop(t, None)
    #         if symbol:
    #             active_symbols.discard(symbol)  # Remove symbol if present
    #         active_threads.remove(t)

    #     # Fetch and process open position data
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}

    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     # Update the active symbols set based on unique open position symbols
    #     active_symbols = active_symbols.intersection(unique_open_position_symbols)

    #     # Fetch and standardize rotator symbols
    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     # Start or maintain threads for all unique open position symbols
    #     for symbol in unique_open_position_symbols:
    #         if symbol not in active_symbols:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)

    #     # Determine available slots for new symbols from rotator list
    #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

    #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

    #     # Start new threads for additional symbols within available slots
    #     for symbol in rotator_symbols_standardized:
    #         if symbol not in active_symbols and available_new_symbol_slots > 0:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)
    #             available_new_symbol_slots -= 1

    #     logging.info(f"Open symbols: {unique_open_position_symbols}")
    #     logging.info(f"Active symbols: {active_symbols}")
    #     logging.info(f"Total active symbols: {len(active_symbols)}")
    #     time.sleep(60)


# A LOT OF COMMENTS WILL REMOVE LATER AFTER WE ARE CERTAIN
                
    # symbol_start_time = {}
    # symbol_lock = threading.Lock()

    # TIMEOUT_THRESHOLD = 120  # 2 minutes in seconds

    # while True:
    #     current_time = time.time()

    #     with symbol_lock:
    #         # Fetch unique open position symbols
    #         open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #         unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
    #         logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #         # Handle completed and timed-out threads
    #         for t in list(active_threads):
    #             symbol = thread_to_symbol.get(t)
    #             if symbol:
    #                 elapsed_time = current_time - symbol_start_time.get(symbol, 0)
    #                 if not t.is_alive() or (elapsed_time > TIMEOUT_THRESHOLD and symbol not in unique_open_position_symbols):
    #                     active_symbols.discard(symbol)
    #                     thread_to_symbol.pop(t, None)
    #                     active_threads.remove(t)
    #                     symbol_start_time.pop(symbol, None)
    #                     reason = "Completed" if not t.is_alive() else "Timeout and not in open positions"
    #                     logging.info(f"Thread removed for symbol {symbol}. Reason: {reason}")

    #         # Fetch rotator symbols
    #         rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #         rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]
    #         logging.info(f"Rotator symbols: {rotator_symbols_standardized}")

    #         # Start or maintain threads for all unique open position symbols
    #         for symbol in unique_open_position_symbols:
    #             if symbol not in active_symbols:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for open position symbol: {symbol}")

    #         # Discard excess symbols if over the limit
    #         symbols_discarded = 0
    #         while len(active_symbols) > symbols_allowed:
    #             symbol_to_discard = next((s for s in active_symbols if s not in unique_open_position_symbols), None)
    #             if symbol_to_discard:
    #                 active_symbols.discard(symbol_to_discard)
    #                 symbols_discarded += 1
    #                 logging.info(f"Discarded thread for symbol: {symbol_to_discard} to maintain limit")
    #             else:
    #                 break
    #         if symbols_discarded > 0:
    #             logging.info(f"Discarded {symbols_discarded} symbols to maintain the limit of {symbols_allowed}")

    #         # Start new threads for additional symbols within available slots
    #         available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
    #         logging.info(f"Attempting to start new threads for symbols. Available slots: {available_new_symbol_slots}")
    #         for symbol in rotator_symbols_standardized:
    #             if symbol not in active_symbols and available_new_symbol_slots > 0:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for new symbol within limit: {symbol}")
    #                 available_new_symbol_slots -= 1

    #     logging.info(f"Actively managing threads for {active_symbols}")
    #     logging.info(f"Total active symbols currently managed: {len(active_symbols)}")

    #     time.sleep(60)


    # symbol_start_time = {}
    # symbol_lock = threading.Lock()

    # while True:
    #     with symbol_lock:
    #         # Handle completed threads and update active symbols
    #         completed_threads = [t for t in active_threads if not t.is_alive()]
    #         for t in completed_threads:
    #             symbol = thread_to_symbol.pop(t, None)
    #             if symbol:
    #                 active_symbols.discard(symbol)
    #                 if symbol in symbol_start_time:
    #                     del symbol_start_time[symbol]
    #                 logging.info(f"Thread completed and removed for symbol: {symbol}")
    #             active_threads.remove(t)

    #     current_time = time.time()

    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     logging.info(f"Rotator symbols: {rotator_symbols_standardized}")

    #     with symbol_lock:
    #         # Start or maintain threads for all unique open position symbols
    #         for symbol in unique_open_position_symbols:
    #             if symbol not in active_symbols:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for open position symbol: {symbol}")

    #         # Discard excess symbols if over the limit
    #         symbols_discarded = 0
    #         while len(active_symbols) > symbols_allowed:
    #             symbol_to_discard = next((s for s in active_symbols if s not in unique_open_position_symbols), None)
    #             if symbol_to_discard:
    #                 active_symbols.discard(symbol_to_discard)
    #                 symbols_discarded += 1
    #                 logging.info(f"Discarded thread for symbol: {symbol_to_discard} to maintain limit")
    #             else:
    #                 break
    #         if symbols_discarded > 0:
    #             logging.info(f"Discarded {symbols_discarded} symbols to maintain the limit of {symbols_allowed}")

    #         # Start new threads for additional symbols within available slots
    #         available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
    #         logging.info(f"Attempting to start new threads for symbols. Available slots: {available_new_symbol_slots}")
    #         for symbol in rotator_symbols_standardized:
    #             if symbol not in active_symbols and symbol not in unique_open_position_symbols and available_new_symbol_slots > 0:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for new symbol within limit: {symbol}")
    #                 available_new_symbol_slots -= 1
    #             else:
    #                 logging.info(f"Skipped starting thread for {symbol} - either active or exceeds limit")

    #         logging.info(f"Actively managing threads for {active_symbols}")
    #         logging.info(f"Total active symbols currently managed: {len(active_symbols)}")

    #     time.sleep(60)


    # A lot of comments left behind until this is 100% how I want it

    # symbol_start_time = {}
    # symbol_lock = threading.Lock()

    # while True:
    #     with symbol_lock:
    #         # Handle completed threads and update active symbols
    #         completed_threads = [t for t in active_threads if not t.is_alive()]
    #         for t in completed_threads:
    #             symbol = thread_to_symbol.pop(t, None)
    #             if symbol:
    #                 active_symbols.discard(symbol)
    #                 if symbol in symbol_start_time:
    #                     del symbol_start_time[symbol]
    #                 logging.info(f"Thread completed and removed for symbol: {symbol}")
    #             active_threads.remove(t)

    #     current_time = time.time()

    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     logging.info(f"Rotator symbols: {rotator_symbols_standardized}")

    #     with symbol_lock:
    #         # Start or maintain threads for all unique open position symbols
    #         for symbol in unique_open_position_symbols:
    #             if symbol not in active_symbols:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for open position symbol: {symbol}")

    #         # Check for symbols that have not opened a position within 1 minute
    #         for symbol in list(active_symbols):
    #             if symbol in rotator_symbols_standardized and current_time - symbol_start_time.get(symbol, 0) > 60 and symbol not in unique_open_position_symbols:
    #                 active_symbols.discard(symbol)
    #                 del symbol_start_time[symbol]
    #                 logging.info(f"No open position for {symbol} within 1 minute. Thread closed.")

    #         # Discard excess symbols if over the limit
    #         symbols_discarded = 0
    #         while len(active_symbols) > symbols_allowed:
    #             symbol_to_discard = next((s for s in active_symbols if s not in unique_open_position_symbols), None)
    #             if symbol_to_discard:
    #                 active_symbols.discard(symbol_to_discard)
    #                 symbols_discarded += 1
    #                 logging.info(f"Discarded thread for symbol: {symbol_to_discard} to maintain limit")
    #             else:
    #                 break
    #         if symbols_discarded > 0:
    #             logging.info(f"Discarded {symbols_discarded} symbols to maintain the limit of {symbols_allowed}")

    #         # Start new threads for additional symbols within available slots
    #         available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
    #         logging.info(f"Attempting to start new threads for symbols. Available slots: {available_new_symbol_slots}")
    #         for symbol in rotator_symbols_standardized:
    #             if symbol not in active_symbols and available_new_symbol_slots > 0:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for new symbol within limit: {symbol}")
    #                 available_new_symbol_slots -= 1
    #             else:
    #                 logging.info(f"Skipped starting thread for {symbol} - either active or exceeds limit")

    #         logging.info(f"Total active symbols currently managed: {len(active_symbols)}")

    #     time.sleep(60)

    # symbol_start_time = {}
    # symbol_lock = threading.Lock()

    # while True:
    #     with symbol_lock:
    #         # Handle completed threads and update active symbols
    #         completed_threads = [t for t in active_threads if not t.is_alive()]
    #         for t in completed_threads:
    #             symbol = thread_to_symbol.pop(t, None)
    #             if symbol:
    #                 active_symbols.discard(symbol)
    #                 if symbol in symbol_start_time:
    #                     del symbol_start_time[symbol]
    #                 logging.info(f"Thread completed and removed for symbol: {symbol}")
    #             active_threads.remove(t)

    #     current_time = time.time()

    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     logging.info(f"Rotator symbols: {rotator_symbols_standardized}")

    #     with symbol_lock:
    #         # Start or maintain threads for all unique open position symbols
    #         for symbol in unique_open_position_symbols:
    #             if symbol not in active_symbols:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for open position symbol: {symbol}")

    #         # Check for symbols that have not opened a position within 1 minute
    #         for symbol in list(active_symbols):
    #             if symbol in rotator_symbols_standardized and current_time - symbol_start_time.get(symbol, 0) > 60 and symbol not in unique_open_position_symbols:
    #                 active_symbols.discard(symbol)
    #                 del symbol_start_time[symbol]
    #                 logging.info(f"No open position for {symbol} within 1 minute. Thread closed.")

    #         # Discard excess symbols if over the limit
    #         symbols_discarded = 0
    #         while len(active_symbols) > symbols_allowed:
    #             symbol_to_discard = next((s for s in active_symbols if s not in unique_open_position_symbols), None)
    #             if symbol_to_discard:
    #                 active_symbols.discard(symbol_to_discard)
    #                 symbols_discarded += 1
    #                 logging.info(f"Discarded thread for symbol: {symbol_to_discard} to maintain limit")
    #             else:
    #                 break
    #         if symbols_discarded > 0:
    #             logging.info(f"Discarded {symbols_discarded} symbols to maintain the limit of {symbols_allowed}")

    #         # Start new threads for additional symbols within available slots
    #         available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
    #         logging.info(f"Attempting to start new threads for symbols. Available slots: {available_new_symbol_slots}")
    #         for symbol in rotator_symbols_standardized:
    #             if symbol not in active_symbols and available_new_symbol_slots > 0:
    #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                 active_symbols.add(symbol)
    #                 symbol_start_time[symbol] = current_time
    #                 logging.info(f"Started thread for new symbol within limit: {symbol}")
    #                 available_new_symbol_slots -= 1
    #             else:
    #                 logging.info(f"Skipped starting thread for {symbol} - either active or exceeds limit")

    #         logging.info(f"Total active symbols currently managed: {len(active_symbols)}")

    #     time.sleep(60)


    # Symbols rotate slowly
    # while True:
    #     # Update active threads and symbols
    #     completed_threads = [t for t in active_threads if not t.is_alive()]
    #     for t in completed_threads:
    #         symbol = thread_to_symbol.pop(t, None)
    #         if symbol:
    #             active_symbols.discard(symbol)  # Remove symbol if present
    #         active_threads.remove(t)

    #     # Fetch and process open position data
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     # Fetch and standardize rotator symbols
    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     # Start or maintain threads for all unique open position symbols, respecting symbols_allowed limit
    #     for symbol in unique_open_position_symbols:
    #         if symbol not in active_symbols and len(active_symbols) < symbols_allowed:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)

    #     # Update available new symbol slots considering open positions
    #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

    #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

    #     # Start new threads for additional symbols within available slots
    #     for symbol in rotator_symbols_standardized:
    #         if symbol not in active_symbols and symbol not in unique_open_position_symbols and available_new_symbol_slots > 0:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)
    #             available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))  # Recalculate after each addition

    #     logging.info(f"Total active symbols: {len(active_symbols)}")
    #     time.sleep(60)

    #     symbol_start_time = {}
    #     symbol_lock = threading.Lock()

    #     while True:
    #         with symbol_lock:
    #             # Handle completed threads and update active symbols
    #             completed_threads = [t for t in active_threads if not t.is_alive()]
    #             for t in completed_threads:
    #                 symbol = thread_to_symbol.pop(t, None)
    #                 if symbol:
    #                     active_symbols.discard(symbol)
    #                     if symbol in symbol_start_time:
    #                         del symbol_start_time[symbol]
    #                     logging.info(f"Thread completed and removed for symbol: {symbol}")
    #                 active_threads.remove(t)

    #         current_time = time.time()

    #         open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #         unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
    #         logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #         rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #         rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #         with symbol_lock:
    #             # Start or maintain threads for all unique open position symbols
    #             for symbol in unique_open_position_symbols:
    #                 if symbol not in active_symbols:
    #                     start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                     active_symbols.add(symbol)
    #                     symbol_start_time[symbol] = current_time
    #                     logging.info(f"Started thread for open position symbol: {symbol}")

    #             # Discard threads for symbols that are not in open positions and exceed the limit
    #             if len(active_symbols) > symbols_allowed:
    #                 for symbol in list(active_symbols):
    #                     if symbol not in unique_open_position_symbols and len(active_symbols) > symbols_allowed:
    #                         active_symbols.discard(symbol)
    #                         logging.info(f"Discarded thread for symbol: {symbol} to maintain limit")

    #             # Start new threads for additional symbols within available slots
    #             available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
    #             for symbol in rotator_symbols_standardized:
    #                 if symbol not in active_symbols and symbol not in unique_open_position_symbols and available_new_symbol_slots > 0:
    #                     start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #                     active_symbols.add(symbol)
    #                     symbol_start_time[symbol] = current_time
    #                     logging.info(f"Started thread for new symbol within limit: {symbol}")
    #                     available_new_symbol_slots -= 1

    #             logging.info(f"Total active symbols currently managed: {len(active_symbols)}")

    #         time.sleep(60)
            
        # symbol_start_time = {}
        # symbol_lock = threading.Lock()

        # while True:
        #     with symbol_lock:
        #         # Handle completed threads and update active symbols
        #         completed_threads = [t for t in active_threads if not t.is_alive()]
        #         for t in completed_threads:
        #             symbol = thread_to_symbol.pop(t, None)
        #             if symbol:
        #                 active_symbols.discard(symbol)
        #                 if symbol in symbol_start_time:
        #                     del symbol_start_time[symbol]
        #                 logging.info(f"Thread completed and removed for symbol: {symbol}")
        #             active_threads.remove(t)

        #     current_time = time.time()
        #     rotation_threshold = 60

        #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
        #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
        #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

        #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
        #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

        #     with symbol_lock:
        #         # Rotate out symbols not being traded
        #         for symbol in list(active_symbols):
        #             if symbol not in unique_open_position_symbols and symbol in symbol_start_time:
        #                 if current_time - symbol_start_time[symbol] > rotation_threshold:
        #                     active_symbols.discard(symbol)
        #                     del symbol_start_time[symbol]
        #                     logging.info(f"Rotated out symbol due to inactivity: {symbol}")

        #         # Manage all unique open position symbols and newly opened positions
        #         for symbol in unique_open_position_symbols.union(rotator_symbols_standardized):
        #             if symbol not in active_symbols and (len(active_symbols) < symbols_allowed or symbol in unique_open_position_symbols):
        #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #                 active_symbols.add(symbol)
        #                 symbol_start_time[symbol] = current_time
        #                 logging.info(f"Started or restarted thread for symbol: {symbol}")

        #         # Recalculate available slots for new symbols, prioritizing open positions
        #         available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
        #         logging.info(f"Active symbols after updates: {active_symbols}")
        #         logging.info(f"Available slots for new symbols: {available_new_symbol_slots}")
        #         logging.info(f"Total active symbols currently managed: {len(active_symbols)}")

        #     time.sleep(60)


        # symbol_start_time = {}
        # symbol_lock = threading.Lock()

        # while True:
        #     with symbol_lock:
        #         # Handle completed threads and update active symbols
        #         completed_threads = [t for t in active_threads if not t.is_alive()]
        #         for t in completed_threads:
        #             symbol = thread_to_symbol.pop(t, None)
        #             if symbol:
        #                 active_symbols.discard(symbol)
        #                 if symbol in symbol_start_time:
        #                     del symbol_start_time[symbol]
        #             active_threads.remove(t)

        #     current_time = time.time()
        #     rotation_threshold = 60

        #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
        #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
        #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

        #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
        #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

        #     with symbol_lock:
        #         # Rotate out symbols not being traded
        #         for symbol in list(active_symbols):
        #             if symbol not in unique_open_position_symbols and symbol in symbol_start_time:
        #                 if current_time - symbol_start_time[symbol] > rotation_threshold:
        #                     active_symbols.discard(symbol)
        #                     del symbol_start_time[symbol]
        #                     logging.info(f"Rotated out symbol: {symbol}")

        #         # Start or maintain threads for all unique open position symbols, up to symbols_allowed
        #         for symbol in unique_open_position_symbols:
        #             if symbol not in active_symbols and len(active_symbols) < symbols_allowed:
        #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #                 active_symbols.add(symbol)
        #                 symbol_start_time[symbol] = current_time

        #         # Recalculate available slots for new symbols
        #         available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
        #         logging.info(f"Active symbols: {active_symbols}")
        #         logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

        #         # Start new threads for additional symbols within available slots
        #         for symbol in rotator_symbols_standardized:
        #             if symbol not in active_symbols and symbol not in unique_open_position_symbols and available_new_symbol_slots > 0:
        #                 start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #                 active_symbols.add(symbol)
        #                 symbol_start_time[symbol] = current_time
        #                 available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

        #     logging.info(f"Total active symbols: {len(active_symbols)}")
        #     time.sleep(60)

            
            
        # symbol_start_time = {}

        # while True:
        #     completed_threads = [t for t in active_threads if not t.is_alive()]
        #     for t in completed_threads:
        #         symbol = thread_to_symbol.pop(t, None)
        #         if symbol:
        #             active_symbols.discard(symbol)
        #             if symbol in symbol_start_time:
        #                 del symbol_start_time[symbol]
        #         active_threads.remove(t)

        #     current_time = time.time()
        #     rotation_threshold = 60

        #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
        #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
        #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

        #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
        #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

        #     for symbol in list(active_symbols):
        #         if symbol not in unique_open_position_symbols and symbol in symbol_start_time:
        #             if current_time - symbol_start_time[symbol] > rotation_threshold:
        #                 active_symbols.discard(symbol)
        #                 del symbol_start_time[symbol]
        #                 logging.info(f"Rotated out symbol: {symbol}")

        #     for symbol in unique_open_position_symbols:
        #         if symbol not in active_symbols and len(active_symbols) < symbols_allowed:
        #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #             active_symbols.add(symbol)
        #             symbol_start_time[symbol] = current_time

        #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
        #     logging.info(f"Active symbols: {active_symbols}")
        #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

        #     for symbol in rotator_symbols_standardized:
        #         if symbol not in active_symbols and symbol not in unique_open_position_symbols and available_new_symbol_slots > 0:
        #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #             active_symbols.add(symbol)
        #             symbol_start_time[symbol] = current_time
        #             available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

        #     logging.info(f"Total active symbols: {len(active_symbols)}")
        #     time.sleep(60)


        # symbol_start_time = {}

        # while True:
        #     completed_threads = [t for t in active_threads if not t.is_alive()]
        #     for t in completed_threads:
        #         symbol = thread_to_symbol.pop(t, None)
        #         if symbol:
        #             active_symbols.discard(symbol)
        #             if symbol in symbol_start_time:
        #                 del symbol_start_time[symbol]
        #         active_threads.remove(t)

        #     current_time = time.time()
        #     rotation_threshold = 60

        #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
        #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
        #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

        #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
        #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

        #     for symbol in list(active_symbols):
        #         if symbol not in unique_open_position_symbols and symbol in symbol_start_time:
        #             if current_time - symbol_start_time[symbol] > rotation_threshold:
        #                 active_symbols.discard(symbol)
        #                 del symbol_start_time[symbol]
        #                 logging.info(f"Rotated out symbol: {symbol}")

        #     for symbol in unique_open_position_symbols:
        #         if symbol not in active_symbols and len(active_symbols) < symbols_allowed:
        #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #             active_symbols.add(symbol)
        #             symbol_start_time[symbol] = current_time

        #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
        #     logging.info(f"Active symbols: {active_symbols}")
        #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

        #     for symbol in rotator_symbols_standardized:
        #         if symbol not in active_symbols and symbol not in unique_open_position_symbols and available_new_symbol_slots > 0:
        #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #             active_symbols.add(symbol)
        #             symbol_start_time[symbol] = current_time
        #             available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

        #     logging.info(f"Total active symbols: {len(active_symbols)}")
        #     time.sleep(60)


        # # Outside the while loop
        # symbol_start_time = {}

        # while True:
        #     # Update active threads and symbols
        #     completed_threads = [t for t in active_threads if not t.is_alive()]
        #     for t in completed_threads:
        #         symbol = thread_to_symbol.pop(t, None)
        #         if symbol:
        #             active_symbols.discard(symbol)  # Remove symbol if present
        #             if symbol in symbol_start_time:
        #                 del symbol_start_time[symbol]  # Remove the start time tracking
        #         active_threads.remove(t)

        #     current_time = time.time()
        #     rotation_threshold = 60  # 1 minute threshold

        #     # Fetch and process open position data
        #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
        #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
        #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

        #     # Fetch and standardize rotator symbols
        #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
        #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

        #     # Rotate out symbols not being traded
        #     for symbol in list(active_symbols):
        #         if symbol not in unique_open_position_symbols and symbol in symbol_start_time:
        #             if current_time - symbol_start_time[symbol] > rotation_threshold:
        #                 active_symbols.discard(symbol)
        #                 del symbol_start_time[symbol]
        #                 logging.info(f"Rotated out symbol: {symbol}")

        #     # Start or maintain threads for all unique open position symbols
        #     for symbol in unique_open_position_symbols:
        #         if symbol not in active_symbols and len(active_symbols) < symbols_allowed:
        #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #             active_symbols.add(symbol)
        #             symbol_start_time[symbol] = current_time  # Track start time

        #     # Update available new symbol slots considering open positions
        #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

        #     logging.info(f"Active symbols: {active_symbols}")
        #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

        #     # Start new threads for additional symbols within available slots
        #     for symbol in rotator_symbols_standardized:
        #         if symbol not in active_symbols and symbol not in unique_open_position_symbols and available_new_symbol_slots > 0:
        #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
        #             active_symbols.add(symbol)
        #             symbol_start_time[symbol] = current_time  # Track start time
        #             available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))  # Recalculate after each addition

        #     logging.info(f"Total active symbols: {len(active_symbols)}")
        #     time.sleep(60)

    # while True:
    #     # Update active threads and symbols
    #     completed_threads = [t for t in active_threads if not t.is_alive()]
    #     for t in completed_threads:
    #         symbol = thread_to_symbol.pop(t, None)
    #         if symbol:
    #             active_symbols.discard(symbol)  # Remove symbol if present
    #         active_threads.remove(t)

    #     # Fetch and process open position data
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}

    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     # Update the active symbols set based on unique open position symbols
    #     active_symbols = active_symbols.intersection(unique_open_position_symbols)

    #     # Fetch and standardize rotator symbols
    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     # Start or maintain threads for all unique open position symbols
    #     for symbol in unique_open_position_symbols:
    #         if symbol not in active_symbols:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)

    #     # Determine available slots for new symbols from rotator list
    #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

    #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

    #     # Start new threads for additional symbols within available slots
    #     for symbol in rotator_symbols_standardized:
    #         if symbol not in active_symbols and available_new_symbol_slots > 0:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)
    #             available_new_symbol_slots -= 1

    #     logging.info(f"Total active symbols: {len(active_symbols)}")
    #     time.sleep(60)

    # while True:
    #     # Update active threads and symbols
    #     completed_threads = [t for t in active_threads if not t.is_alive()]
    #     for t in completed_threads:
    #         symbol = thread_to_symbol.pop(t, None)
    #         if symbol:
    #             active_symbols.discard(symbol)  # Remove symbol if present
    #         active_threads.remove(t)

    #     # Fetch and process open position data
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}
    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     # Update the active symbols set based on unique open position symbols
    #     active_symbols = active_symbols.intersection(unique_open_position_symbols)

    #     # Fetch and standardize rotator symbols
    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     # Start or maintain threads for all unique open position symbols
    #     for symbol in unique_open_position_symbols:
    #         if symbol not in active_symbols and len(active_symbols) < symbols_allowed:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)

    #     # Recalculate available new symbol slots after updating active_symbols
    #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))
    #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

    #     # Start new threads for additional symbols within available slots
    #     for symbol in rotator_symbols_standardized:
    #         if symbol not in active_symbols and available_new_symbol_slots > 0:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)
    #             available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))  # Recalculate after each addition

    #     logging.info(f"Total active symbols: {len(active_symbols)}")
    #     time.sleep(60)

    # while True:
    #     # Update active threads and symbols
    #     completed_threads = [t for t in active_threads if not t.is_alive()]
    #     for t in completed_threads:
    #         symbol = thread_to_symbol.pop(t, None)
    #         if symbol:
    #             active_symbols.discard(symbol)  # Remove symbol if present
    #         active_threads.remove(t)

    #     # Fetch and process open position data
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}

    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     # Fetch and standardize rotator symbols
    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]


    #     # Start or maintain threads for all unique open position symbols
    #     for symbol in unique_open_position_symbols:
    #         if symbol not in active_symbols:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)

    #     # Recalculate available new symbol slots after updating active_symbols
    #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

    #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

    #     # Start new threads for additional symbols within available slots
    #     for symbol in rotator_symbols_standardized:
    #         if symbol not in active_symbols and available_new_symbol_slots > 0:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)
    #             available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))  # Recalculate after each addition

    #     logging.info(f"Total active symbols: {len(active_symbols)}")
    #     time.sleep(60)
        
    # while True:
    #     # Update active threads and symbols
    #     completed_threads = [t for t in active_threads if not t.is_alive()]
    #     for t in completed_threads:
    #         symbol = thread_to_symbol.pop(t, None)
    #         if symbol:
    #             active_symbols.discard(symbol)  # Remove symbol if present
    #         active_threads.remove(t)

    #     # Fetch and process open position data
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     unique_open_position_symbols = {standardize_symbol(position['symbol']) for position in open_position_data}

    #     logging.info(f"Unique open position symbols: {unique_open_position_symbols}")

    #     # Update the active symbols set based on unique open position symbols
    #     active_symbols = active_symbols.intersection(unique_open_position_symbols)

    #     # Fetch and standardize rotator symbols
    #     rotator_symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in rotator_symbols]

    #     # Start or maintain threads for all unique open position symbols
    #     for symbol in unique_open_position_symbols:
    #         if symbol not in active_symbols:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)

    #     # Determine available slots for new symbols from rotator list
    #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

    #     logging.info(f"Available new slots for rotator symbols: {available_new_symbol_slots}")

    #     # Start new threads for additional symbols within available slots
    #     for symbol in rotator_symbols_standardized:
    #         if symbol not in active_symbols and available_new_symbol_slots > 0:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             active_symbols.add(symbol)
    #             available_new_symbol_slots -= 1

    #     logging.info(f"Total active symbols: {len(active_symbols)}")
    #     time.sleep(60)




    # Saving this because it works but not for long & short accounted, 
    # will remove after further testing
        
    # while True:
    #     # Update active threads and symbols
    #     completed_threads = [t for t in active_threads if not t.is_alive()]
    #     for t in completed_threads:
    #         symbol = thread_to_symbol.pop(t, None)
    #         if symbol:
    #             active_symbols.remove(symbol)
    #         active_threads.remove(t)

    #     # Fetch updated symbols list
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)]
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     open_positions_symbols = [standardize_symbol(position['symbol']) for position in open_position_data]

    #     logging.info(f"Open positions symbols: {open_positions_symbols}")

    #     # Check for symbols with active threads but no open positions
    #     for symbol in list(active_symbols):
    #         if symbol not in open_positions_symbols:
    #             # This symbol no longer has open positions, consider it completed
    #             active_symbols.remove(symbol)
    #             # Optional: Signal the corresponding thread to terminate, if necessary

    #     # Start or maintain threads for all open positions
    #     for symbol in open_positions_symbols:
    #         if symbol not in active_symbols:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)

    #     # Calculate available new symbol slots
    #     available_new_symbol_slots = max(0, symbols_allowed - len(active_symbols))

    #     logging.info(f"Available new symbol slots: {available_new_symbol_slots}")

    #     # Start threads for new rotator symbols within the available slots
    #     new_symbols = [s for s in rotator_symbols_standardized if s not in active_symbols][:available_new_symbol_slots]
    #     for symbol in new_symbols:
    #         if available_new_symbol_slots > 0:
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
    #             available_new_symbol_slots -= 1

    #     time.sleep(60)