import sys
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent)
print("Project directory:", project_dir)
sys.path.insert(0, project_dir)
# project_dir = str(Path(__file__).resolve().parents[1])
# sys.path.insert(0, project_dir)
import argparse
from pathlib import Path
from config import load_config, Config
import ccxt
import config
from api.manager import Manager
from directionalscalper.core.exchange import Exchange
from directionalscalper.core.strategies.strategy import Strategy
from directionalscalper.core.strategies.hedge import HedgeStrategy

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

    def get_balance(self, quote):
        if self.exchange_name == 'bitget':
            return self.exchange.get_balance_bitget(quote)
        else:
            return self.exchange.get_balance(quote)

    def create_order(self, symbol, order_type, side, amount, price=None):
        return self.exchange.create_order(symbol, order_type, side, amount, price)

    def get_symbols(self):
        return self.exchange.symbols

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DirectionalScalper')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--exchange', type=str, help='The name of the exchange to use')
    parser.add_argument('--strategy', type=str, help='The name of the strategy to use')
    parser.add_argument('--symbol', type=str, help='The trading symbol to use')
    parser.add_argument('--amount', type=str, help='The size to use')
    args = parser.parse_args()

    config_file_path = Path(args.config)
    config = load_config(config_file_path)

    exchange_name = args.exchange
    strategy_name = args.strategy
    symbol = args.symbol
    amount = args.amount
    print(f"Exchange name: {exchange_name}")
    print(f"Strategy name: {strategy_name}")
    print(f"Symbol: {symbol}")

    market_maker = DirectionalMarketMaker(config, exchange_name)  # Initialize the DirectionalMarketMaker object

    manager = Manager(market_maker.exchange, api=config.api.mode, path=Path("data", config.api.filename), url=f"{config.api.url}{config.api.filename}")
    market_maker.manager = manager  # Assign the `Manager` object to the `DirectionalMarketMaker`

    quote = "USDT"
    balance = market_maker.get_balance(quote)
    print(f"Balance: {balance}")

    try:
        if strategy_name.lower() == 'hedge':
            strategy = HedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)
        else:
            print("Strategy not recognized. Please choose a valid strategy.")
    except ccxt.ExchangeError as e:
        print(f"An error occurred while executing the strategy: {e}")
