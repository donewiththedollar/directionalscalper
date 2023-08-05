import sys
import traceback
import threading
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent)
print("Project directory:", project_dir)
sys.path.insert(0, project_dir)

from rich.live import Live
import argparse
from pathlib import Path
import config
from config import load_config, Config
import ccxt
from api.manager import Manager
from directionalscalper.core.exchange import Exchange
from directionalscalper.core.strategies.strategy import Strategy
# Bybit rotator
from directionalscalper.core.strategies.bybit.bybit_auto_rotator import BybitAutoRotator
from directionalscalper.core.strategies.bybit.bybit_auto_rotator_mfirsi import BybitAutoRotatorMFIRSI
from directionalscalper.core.strategies.bybit.bybit_auto_hedge_maker_mfirsi_rotator import BybitAutoHedgeStrategyMakerMFIRSIRotator

class DirectionalMarketMaker:
    def __init__(self, config: Config, exchange_name: str):
        self.config = config
        self.exchange_name = exchange_name
        exchange_config = None

        for exch in config.exchanges:
            if exch.name == exchange_name:
                exchange_config = exch
                break

        if not exchange_config:
            raise ValueError(f"Exchange {exchange_name} not found in the configuration file.")

        api_key = exchange_config.api_key
        secret_key = exchange_config.api_secret
        passphrase = exchange_config.passphrase
        self.exchange = Exchange(self.exchange_name, api_key, secret_key, passphrase)

    def run_strategy(self, symbol, strategy_name, config):
        if strategy_name.lower() == 'bybit_hedge_rotator':
            strategy = BybitAutoRotator(self.exchange, self.manager, config.bot)
            print(f"Calling run method with symbols: {symbols}")
            strategy.run(symbol)
        elif strategy_name.lower() == 'bybit_hedge_rotator_mfirsi':
            strategy = BybitAutoRotatorMFIRSI(self.exchange, self.manager, config.bot)
            strategy.run(symbol)
        elif strategy_name.lower() == 'bybit_auto_hedge_mfi_rotator':
            strategy = BybitAutoRotatorMFIRSI(self.exchange, self.manager, config.bot)
            strategy.run(symbol)

    def get_balance(self, quote, market_type=None, sub_type=None):
        if self.exchange_name == 'bitget':
            return self.exchange.get_balance_bitget(quote)
        elif self.exchange_name == 'bybit':
            return self.exchange.get_balance_bybit(quote)
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

def run_bot(symbol, args, manager):
    config_file_path = Path('configs/' + args.config)
    print("Loading config from:", config_file_path)
    config = load_config(config_file_path)

    exchange_name = args.exchange
    strategy_name = args.strategy
    amount = args.amount

    print(f"Symbol: {symbol}")
    print(f"Exchange name: {exchange_name}")
    print(f"Strategy name: {strategy_name}")

    market_maker = DirectionalMarketMaker(config, exchange_name)
    market_maker.manager = manager  # Use the passed-in manager instance

    quote = "USDT"
    if exchange_name.lower() == 'huobi':
        print(f"Loading huobi strategy..")
    elif exchange_name.lower() == 'mexc':
        balance = market_maker.get_balance(quote, type='swap')
        print(f"Futures balance: {balance}")
    else:
        balance = market_maker.get_balance(quote)
        print(f"Futures balance: {balance}")

    market_maker.run_strategy(symbol, strategy_name, config)  # Calling the run_strategy method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DirectionalScalper')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')
    parser.add_argument('--exchange', type=str, help='The name of the exchange to use')
    parser.add_argument('--strategy', type=str, help='The name of the strategy to use')
    parser.add_argument('--symbol', type=str, help='The trading symbol to use')
    parser.add_argument('--amount', type=str, help='The size to use')
    args = parser.parse_args()
    config_file_path = Path('configs/' + args.config)
    config = load_config(config_file_path)

    exchange_name = args.exchange
    market_maker = DirectionalMarketMaker(config, exchange_name)
    manager = Manager(market_maker.exchange, api=config.api.mode, path=Path("data", config.api.filename), url=f"{config.api.url}{config.api.filename}")
    

    # whitelist = ['ORDIUSDT', '1000PEPEUSDT', 'MINAUSDT', 'ILVUSDT', 'YGGUSDT', 'XRPUSDT, MATICUSDT, INJUSDT, LTCUSDT, AVAXUSDT, DOTUSDT, ATOMUSDT, ETCUSDT, SHIB1000USDT, UNIUSDT, FILUSDT, APTUSDT, ARBUSDT, XLMUSDT, NEARUSDT, OPUSDT, ALGOUSDT, SANDUSDT, MANAUSDT, FTMUSDT, CTSIUSDT', 'COMPUSDT', 'STMXUSDT', 'APEUSDT']  # Symbols that you want to include]
    # blacklist = ['BTCUSDT', 'ETHUSDT']  # Symbols that you want to exclude

    whitelist = config.bot.whitelist
    blacklist = config.bot.blacklist

    symbols = manager.get_auto_rotate_symbols(min_qty_threshold=None, whitelist=whitelist, blacklist=blacklist)

    #symbols = manager.get_auto_rotate_symbols()  # Assuming this is the method to get symbols

    threads = [threading.Thread(target=run_bot, args=(symbol, args, manager)) for symbol in symbols]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()