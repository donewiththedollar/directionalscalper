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
# Bybit rotator
from directionalscalper.core.strategies.bybit.multi.bybit_mfirsi_trend_rotator import BybitMFIRSITrendRotator
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fiveminute_qfl_mfi import BybitMMFiveMinuteQFLMFI
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fivemin_qfl_mfi_eri_walls_autohedge_v2 import BybitMMFiveMinuteQFLMFIERIAutoHedgeWallsV2
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fivemin_qfl_mfi_eri_walls_autohedge_v3 import BybitMMFiveMinuteQFLMFIERIAutoHedgeWallsV3
from directionalscalper.core.strategies.bybit.multi.bybit_mm_onemin_qfl_mfi_eri_walls_autohedge import BybitMMOneMinuteQFLMFIERIAutoHedgeWallsAutoHedge
from directionalscalper.core.strategies.bybit.multi.bybit_mm_onemin_qfl_mfi_eri_walls_autohedge_atr import BybitMMOneMinuteQFLMFIERIAutoHedgeWallsATR
from directionalscalper.core.strategies.bybit.multi.bybit_mm_onemin_qfl_mfi_eri_walls_autohedge_atr_topbottom import BybitMMOneMinuteQFLMFIERIAutoHedgeWallsATRTopBottom
from directionalscalper.core.strategies.bybit.multi.bybit_mm_onemin_qfl_mfi_eri_walls import BybitMMOneMinuteQFLMFIERIAutoHedgeWalls
from directionalscalper.core.strategies.bybit.multi.bybit_mm_onemin_qfl_mfi_eri_walls_allowedfix import BybitMMOneMinuteQFLMFIERIAutoHedgeWallsAllowedFix
from directionalscalper.core.strategies.bybit.multi.bybit_mm_onemin_qfl_mfi_eri_walls_tb import BybitMMOneMinuteQFLMFIERIAutoHedgeWallsTB
from directionalscalper.core.strategies.bybit.multi.bybit_mm_onemin_eri_tb import BybitMMOneMinERITB
from directionalscalper.core.strategies.bybit.multi.bybit_qs import BybitQSStrategy
from directionalscalper.core.strategies.bybit.multi.bybit_mfirsi import BybitAutoRotatorMFIRSI
from live_table_manager import LiveTableManager, shared_symbols_data


from directionalscalper.core.strategies.logger import Logger

thread_to_symbol = {}
thread_to_symbol_lock = threading.Lock()


logging = Logger(logger_name="MultiBot", filename="MultiBot.log", stream=True)

def standardize_symbol(symbol):
    return symbol.replace('/', '').split(':')[0]

def get_available_strategies():
    return [
        'bybit_1m_qfl_mfi_eri_walls',
        'bybit_1m_qfl_mfi_eri_autohedge_walls_atr',
        # 'bybit_1m_eri_tb',
        # 'bybit_1m_qfl_mfi_eri_walls_tb',
        # 'bybit_1m_qfl_mfi_eri_walls_allowedfix',
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
        if strategy_name.lower() == 'bybit_mm_mfirsi':
            strategy = BybitAutoRotatorMFIRSI(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qfl_mfi':
            strategy = BybitMMFiveMinuteQFLMFI(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_walls':
            strategy = BybitMMOneMinuteQFLMFIERIAutoHedgeWalls(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_walls_allowedfix':
            strategy = BybitMMOneMinuteQFLMFIERIAutoHedgeWallsAllowedFix(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_walls_tb':
            strategy = BybitMMOneMinuteQFLMFIERIAutoHedgeWallsTB(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_eri_tb':
            strategy = BybitMMOneMinERITB(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_autohedge_walls_autohedge':
            strategy = BybitMMOneMinuteQFLMFIERIAutoHedgeWallsAutoHedge(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_autohedge_walls_atr':
            strategy = BybitMMOneMinuteQFLMFIERIAutoHedgeWallsATR(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_1m_qfl_mfi_eri_autohedge_walls_atr_top_bottom':
            strategy = BybitMMOneMinuteQFLMFIERIAutoHedgeWallsATRTopBottom(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qfl_mfi_eri_autohedge_walls_v2':
            strategy = BybitMMFiveMinuteQFLMFIERIAutoHedgeWallsV2(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qfl_mfi_eri_autohedge_walls_v3':
            strategy = BybitMMFiveMinuteQFLMFIERIAutoHedgeWallsV3(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qs':
            strategy = BybitQSStrategy(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mfirsi_trend':
            strategy = BybitMFIRSITrendRotator(self.exchange, self.manager, config.bot, symbols_allowed)
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
    thread = threading.Thread(target=run_bot, args=(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized))
    thread.start()
    active_threads.append(thread)
    logging.info(f"Started thread for symbol: {symbol}")

def start_threads_for_new_symbols(new_symbols, args, manager, account_name, symbols_allowed, rotator_symbols_standardized):
    for symbol in new_symbols:
        start_thread_for_symbol(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized)


def start_threads_for_symbols(symbols, args, manager, account_name, symbols_allowed, rotator_symbols_standardized):
    active_threads = []
    for symbol in symbols:
        with thread_to_symbol_lock:
            if symbol not in thread_to_symbol.values() and len(active_threads) < symbols_allowed:
                thread = threading.Thread(target=run_bot, args=(symbol, args, manager, account_name, symbols_allowed, rotator_symbols_standardized))
                thread.start()
                active_threads.append(thread)
                logging.info(f"Started thread for symbol: {symbol}")
    return active_threads

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

    active_threads = start_threads_for_symbols(symbols_to_trade, args, manager, args.account_name, symbols_allowed, all_symbols_standardized)

    while True:
        active_threads = [t for t in active_threads if t is not None and t.is_alive()]
        rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)]

        open_position_data = market_maker.exchange.get_all_open_positions_bybit()
        open_positions_symbols = [standardize_symbol(position['symbol']) for position in open_position_data]

        logging.info(f"Open positions symbols : {open_positions_symbols}")

        # Start or maintain threads for all open positions
        for symbol in open_positions_symbols:
            if symbol not in thread_to_symbol.values():
                start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)

        # Calculate available slots for new symbols AFTER accommodating open positions
        available_new_symbol_slots = max(0, symbols_allowed - len(thread_to_symbol.values()))

        # Start threads for new rotator symbols within the available slots
        new_symbols = [s for s in rotator_symbols_standardized if s not in thread_to_symbol.values()][:available_new_symbol_slots]
        for symbol in new_symbols:
            if available_new_symbol_slots > 0:
                start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)
                available_new_symbol_slots -= 1

        time.sleep(60)

# All of these comments are staying for now because symbols_allowed is being that annoying yes
                
    # while True:
    #     # Active threads update
    #     active_threads = [t for t in active_threads if t is not None and t.is_alive()]
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)]

    #     # Fetch open position symbols
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     open_positions_symbols = [standardize_symbol(position['symbol']) for position in open_position_data]

    #     # Start or maintain threads for open positions
    #     for symbol in open_positions_symbols:
    #         if symbol not in thread_to_symbol.values():
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)

    #     # Calculate available slots after accommodating open positions
    #     available_slots = max(0, symbols_allowed - len(thread_to_symbol.values()))

    #     # Start threads for new rotator symbols within the available slots
    #     new_symbols = [s for s in rotator_symbols_standardized if s not in thread_to_symbol.values()][:available_slots]
    #     for symbol in new_symbols:
    #         start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)

    #     time.sleep(60)

    # while True:
    #     # Active threads and rotator symbols update
    #     active_threads = [t for t in active_threads if t.is_alive()]
    #     rotator_symbols_standardized = [standardize_symbol(symbol) for symbol in manager.get_auto_rotate_symbols(min_qty_threshold=None, blacklist=blacklist, max_usd_value=max_usd_value)]

    #     # Update and prioritize open position symbols
    #     open_position_data = market_maker.exchange.get_all_open_positions_bybit()
    #     open_positions_symbols = [standardize_symbol(position['symbol']) for position in open_position_data]

    #     # Start or maintain threads for open positions regardless of symbols_allowed limit
    #     for symbol in open_positions_symbols:
    #         if symbol not in thread_to_symbol.values():
    #             start_thread_for_symbol(symbol, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)

    #     # Start threads for rotator symbols if under the limit, excluding already handled open positions
    #     new_symbols = [s for s in rotator_symbols_standardized if s not in thread_to_symbol.values() and s not in open_positions_symbols]
    #     if len(active_threads) < symbols_allowed:
    #         start_threads_for_new_symbols(new_symbols, args, manager, args.account_name, symbols_allowed, rotator_symbols_standardized)

    #     time.sleep(60)