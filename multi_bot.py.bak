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
from directionalscalper.core.strategies.bybit.multi.bybit_auto_rotator import BybitAutoRotator
from directionalscalper.core.strategies.bybit.multi.bybit_mfirsi_trend_rotator import BybitMFIRSITrendRotator
from directionalscalper.core.strategies.bybit.multi.bybit_mm_oneminute import BybitMMOneMinute
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fiveminute import BybitMMFiveMinute
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fiveminute_walls import BybitMMFiveMinuteWalls
from directionalscalper.core.strategies.bybit.multi.bybit_mm_oneminute_walls import BybitMMOneMinuteWalls
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fiveminute_qfl_mfi import BybitMMFiveMinuteQFLMFI
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fiveminute_qfl_mfi_autohedge import BybitMMFiveMinuteQFLMFIAutoHedge
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fiveminute_qfl_mfi_eri_autohedge import BybitMMFiveMinuteQFLMFIERIAutoHedge
from directionalscalper.core.strategies.bybit.multi.bybit_mm_fiveminute_qfl_mfi_eri_autohedge_unstuck import BybitMMFiveMinuteQFLMFIERIAutoHedgeUnstuck
from directionalscalper.core.strategies.bybit.multi.bybit_qs import BybitQSStrategy
from directionalscalper.core.strategies.bybit.multi.bybit_obstrength import BybitOBStrength
from directionalscalper.core.strategies.bybit.multi.bybit_mfirsi import BybitAutoRotatorMFIRSI
from directionalscalper.core.strategies.bybit.multi.bybit_mm_playthespread import BybitMMPlayTheSpread
from directionalscalper.core.strategies.bybit.multi.bybit_obstrength_random import BybitOBStrengthRandom
from live_table_manager import LiveTableManager, shared_symbols_data


from directionalscalper.core.strategies.logger import Logger

logging = Logger(logger_name="MultiBot", filename="MultiBot.log", stream=True)


# Function to standardize symbols
def standardize_symbol(symbol):
    return symbol.replace('/', '').split(':')[0]

# Function to choose a strategy via the command line
def choose_strategy():
    questions = [
        inquirer.List('strategy',
                      message='Which strategy would you like to run?',
                      choices=[
                          'bybit_mm_mfirsi',
                          'bybit_mm_fivemin',
                          'bybit_mm_onemin',
                          'bybit_mfirsi_trend',
                          'bybit_obstrength',
                          'bybit_mm_fivemin_walls',
                          'bybit_mm_onemin_walls',
                          'bybit_mm_qfl_mfi',
                          'bybit_mm_qfl_mfi_autohedge',
                          'bybit_mm_qs',
                          'bybit_mm_qfl_mfi_eri_autohedge',
                          'bybit_mm_qfl_mfi_eri_autohedge_unstuck',
                      ])
    ]
    answers = inquirer.prompt(questions)
    return answers['strategy']

class DirectionalMarketMaker:
    def __init__(self, config: Config, exchange_name: str, account_name: str, manager):
        self.config = config
        self.exchange_name = exchange_name
        self.account_name = account_name
        self.manager = manager
        exchange_config = self.get_exchange_config(config, exchange_name, account_name)
        self.exchange = Exchange(exchange_name, exchange_config.api_key, exchange_config.api_secret, exchange_config.passphrase)
    
    def get_exchange_config(self, config, exchange_name, account_name):
        for exch in config.exchanges:
            if exch.name == exchange_name and exch.account_name == account_name:
                return exch
        raise ValueError(f"Exchange {exchange_name} with account {account_name} not found in the configuration file.")

    def run_strategy(self, symbol, strategy_name, config, account_name, symbols_to_trade=None, rotator_symbols_standardized=None):
        symbols_allowed = None
        for exch in config.exchanges:
            #print(f"Checking: {exch.name} vs {self.exchange_name} and {exch.account_name} vs {account_name}")
            if exch.name == self.exchange_name and exch.account_name == account_name:
                symbols_allowed = exch.symbols_allowed
                print(f"Matched exchange: {self.exchange_name}, account: {args.account_name}. Symbols allowed: {symbols_allowed}")
                break

        print(f"Multibot.py: symbols_allowed from config: {symbols_allowed}")
        
        if symbols_to_trade:
            print(f"Calling run method with symbols: {symbols_to_trade}")

        # Pass symbols_allowed to the strategy constructors
        if strategy_name.lower() == 'bybit_mm_mfirsi':
            strategy = BybitAutoRotatorMFIRSI(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_onemin':
            strategy = BybitMMOneMinute(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_fivemin':
            strategy = BybitMMFiveMinute(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_fivemin_walls':
            strategy = BybitMMFiveMinuteWalls(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_onemin_walls':
            strategy = BybitMMOneMinuteWalls(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qfl_mfi':
            strategy = BybitMMFiveMinuteQFLMFI(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qfl_mfi_autohedge':
            strategy = BybitMMFiveMinuteQFLMFIAutoHedge(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qfl_mfi_eri_autohedge':
            strategy = BybitMMFiveMinuteQFLMFIERIAutoHedge(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qfl_mfi_eri_autohedge_unstuck':
            strategy = BybitMMFiveMinuteQFLMFIERIAutoHedgeUnstuck(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mm_qs':
            strategy = BybitQSStrategy(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_mfirsi_trend':
            strategy = BybitMFIRSITrendRotator(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_obstrength':
            strategy = BybitOBStrength(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_pts':
            strategy = BybitMMPlayTheSpread(self.exchange, self.manager, config.bot, symbols_allowed)
            strategy.run(symbol, rotator_symbols_standardized=rotator_symbols_standardized)
        elif strategy_name.lower() == 'bybit_obstrength_random':
            strategy = BybitOBStrengthRandom(self.exchange, self.manager, config.bot, symbols_allowed)
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
    
# Function to run the bot
def run_bot(market_maker, symbol, args, account_name, symbols_allowed, rotator_symbols_standardized):
    # Identify the current thread for future reference
    current_thread = threading.current_thread()
    thread_to_symbol[current_thread] = symbol

    # Log basic information about the operation
    print(f"Running strategy for Symbol: {symbol}")
    print(f"Exchange name: {market_maker.exchange_name}")
    print(f"Strategy name: {args.strategy}")
    print(f"Account name: {account_name}")

    # Pass rotator_symbols_standardized to the run_strategy method
    market_maker.run_strategy(symbol, args.strategy, market_maker.config, account_name, symbols_allowed, rotator_symbols_standardized)

    # Implement logic for balance check and other operations as necessary
    # For example, checking the balance at regular intervals
    quote = "USDT"
    current_time = time.time()
    balance_refresh_interval = 600  # seconds
    cached_balance = None
    last_balance_fetch_time = 0

    if current_time - last_balance_fetch_time > balance_refresh_interval or not cached_balance:
        cached_balance = market_maker.get_balance(quote)
        print(f"Futures balance: {cached_balance}")
        last_balance_fetch_time = current_time

# Function to start threads for each symbol
# Ensure to update start_threads_for_symbols to pass market_maker instead of individual params
def start_threads_for_symbols(market_maker, symbols, args, manager, account_name, symbols_allowed, rotator_symbols_standardized):
    threads = []
    for symbol in symbols:
        thread = threading.Thread(target=run_bot, args=(market_maker, symbol, args, account_name, symbols_allowed, rotator_symbols_standardized))
        threads.append(thread)
        thread.start()
    return threads

# Main execution block
if __name__ == '__main__':
    # ASCII Art and Text
    sword = "====||====>"

    print("\n" + "=" * 50)
    print(f"DirectionalScalper {VERSION}".center(50))
    print("=" * 50 + "\n")

    print("Initializing", end="")
    # Loading animation
    for i in range(3):
        time.sleep(0.5)
        print(".", end="")
    print("\n")

    # Display the ASCII art
    print("Battle-Ready Algorithm".center(50))
    print("Developed by TradeSimple Foundation".center(50))
    print(sword.center(50) + "\n")

    parser = argparse.ArgumentParser(description='DirectionalScalper')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')
    parser.add_argument('--account_name', type=str, required=True, help='The name of the account to use')
    parser.add_argument('--exchange', type=str, required=True, help='The name of the exchange to use')
    parser.add_argument('--strategy', type=str, required=True, help='The name of the strategy to use')
    args = parser.parse_args()

    config_file_path = Path('configs/' + args.config)
    config = load_config(config_file_path)

    # Initialize the Manager object
    manager = Manager(
        None,  # Temporarily set to None
        args.exchange,
        config.api.data_source_exchange,
        config.api.mode,
        cache_life_seconds=240,
        asset_value_cache_life_seconds=10,
        path=Path("data", config.api.filename),
        url=f"{config.api.url}{config.api.filename}"
    )

    # Create the DirectionalMarketMaker object
    market_maker = DirectionalMarketMaker(config, args.exchange, args.account_name, manager)
    manager.exchange = market_maker.exchange  # Update exchange in manager

    # Access blacklist and max_usd_value from config
    blacklist = config.bot.blacklist
    max_usd_value = config.bot.max_usd_value

    ### ILAY ###
    table_manager = LiveTableManager()
    display_thread = threading.Thread(target=table_manager.display_table, daemon=True)
    display_thread.start()
    ### ILAY ###

    thread_to_symbol = {}

    while True:
        # Fetch and standardize all symbols from the manager
        all_symbols_standardized = [standardize_symbol(symbol) for symbol in manager.get_auto_rotate_symbols()]

        # Retrieve symbols_allowed for the specific exchange and account
        symbols_allowed = None
        for exch in config.exchanges:
            if exch.name == args.exchange and exch.account_name == args.account_name:
                symbols_allowed = exch.symbols_allowed
                break

        if symbols_allowed is None:
            raise ValueError(f"Symbols allowed not set for exchange {args.exchange} and account {args.account_name}")

        # Remove dead threads from the thread_to_symbol mapping
        thread_to_symbol = {t: s for t, s in thread_to_symbol.items() if t.is_alive()}

        # Filter out blacklisted symbols
        tradeable_symbols = [symbol for symbol in all_symbols_standardized if symbol not in blacklist]

        # Get currently active symbols
        active_symbols = [thread_to_symbol[t] for t in thread_to_symbol]

        # Determine new symbols to trade
        new_symbols = [symbol for symbol in tradeable_symbols if symbol not in active_symbols]
        num_symbols_to_add = min(symbols_allowed - len(active_symbols), len(new_symbols))
        new_symbols = new_symbols[:num_symbols_to_add]

        # Start threads for new symbols
        for symbol in new_symbols:
            thread = threading.Thread(target=run_bot, args=(market_maker, symbol, args, args.account_name, symbols_allowed, all_symbols_standardized))
            thread.start()
            thread_to_symbol[thread] = symbol
            logging.info(f"Started new thread for {symbol}")

        # Log the number of active threads
        logging.info(f"Active threads: {len(thread_to_symbol)}")

        time.sleep(60)