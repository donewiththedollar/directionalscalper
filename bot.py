import sys
import traceback
from pathlib import Path

project_dir = str(Path(__file__).resolve().parent)
print("Project directory:", project_dir)
sys.path.insert(0, project_dir)

from rich.live import Live
import argparse
from pathlib import Path
from config import load_config, Config
import ccxt
import config
from api.manager import Manager
from directionalscalper.core.exchanges.exchange import Exchange
from directionalscalper.core.strategies.strategy import Strategy
# BITGET
from directionalscalper.core.strategies.bitget.bitget_hedge import BitgetHedgeStrategy
from directionalscalper.core.strategies.bitget.bitget_hedge_dynamic import BitgetDynamicHedgeStrategy
from directionalscalper.core.strategies.bitget.bitget_longonly_dynamic import BitgetLongOnlyDynamicStrategy
from directionalscalper.core.strategies.bitget.bitget_shortonly_dynamic import BitgetShortOnlyDynamicStrategy
from directionalscalper.core.strategies.bitget.bitget_auctionbased_dynamic import BitgetDynamicAuctionBasedStrategy
from directionalscalper.core.strategies.bitget.bitget_grid_dynamic import BitgetGridStrategy
from directionalscalper.core.strategies.bitget.bitget_fivemin import BitgetFiveMinuteStrategy
# OKX
from directionalscalper.core.strategies.okx.okx_hedge import OKXHedgeStrategy
# BYBIT
from directionalscalper.core.strategies.bybit.single.bybit_hedge import BybitHedgeStrategy
from directionalscalper.core.strategies.bybit.single.bybit_violent import BybitViolentHedgeStrategy
from directionalscalper.core.strategies.bybit.single.bybit_hedge_unified import BybitHedgeUnifiedStrategy
from directionalscalper.core.strategies.bybit.single.bybit_longonly import BybitLongStrategy
from directionalscalper.core.strategies.bybit.single.bybit_shortonly import BybitShortStrategy
from directionalscalper.core.strategies.bybit.single.bybit_longonly_dynamic_leverage import BybitLongOnlyDynamicLeverage
from directionalscalper.core.strategies.bybit.single.bybit_shortonly_dynamic_leverage import BybitShortOnlyDynamicLeverage
from directionalscalper.core.strategies.bybit.single.bybit_auto_hedge import BybitAutoHedgeStrategy
from directionalscalper.core.strategies.bybit.single.bybit_auto_hedge_maker import BybitAutoHedgeStrategyMaker
from directionalscalper.core.strategies.bybit.single.bybit_auto_hedge_maker_mfirsi import BybitAutoHedgeStrategyMakerMFIRSI
from directionalscalper.core.strategies.bybit.single.bybit_auto_hedge_MFIRSI import BybitAutoHedgeStrategyMFIRSI
from directionalscalper.core.strategies.bybit.single.bybit_auto_hedge_maker_eri_trend import BybitAutoHedgeStrategyMakerERITrend
from directionalscalper.core.strategies.bybit.single.bybit_hedge_mfirsi_trigger import BybitHedgeMFIRSITrigger
from directionalscalper.core.strategies.bybit.single.bybit_hedge_mfirsi_trenderi_maker import BybitAutoHedgeMFIRSIPostOnly
from directionalscalper.core.strategies.bybit.single.bybit_hedge_mfirsi_trend_maker import BybitAutoHedgeMFIRSITrendMaker
from directionalscalper.core.strategies.bybit.single.bybit_hedge_mfirsi_trigger_postonly import BybitHedgeMFIRSITriggerPostOnly
from directionalscalper.core.strategies.bybit.single.bybit_hedge_mfirsi_trigger_postonly_avoidfees import BybitHedgeMFIRSITriggerPostOnlyAvoidFees
# HUOBI
from directionalscalper.core.strategies.huobi.huobi_auto_hedge import HuobiAutoHedgeStrategy
# BINANCE
from directionalscalper.core.strategies.binance.binance_auto_hedge import BinanceAutoHedgeStrategy
from directionalscalper.core.strategies.binance.binance_auto_hedge_maker import BinanceAutoHedgeMakerStrategy
# PHEMEX
from directionalscalper.core.strategies.phemex.phemex_hedge import PhemexHedgeStrategy
# MEXC
from directionalscalper.core.strategies.mexc.mexc_hedge import MEXCHedgeStrategy

class DirectionalMarketMaker:
    def __init__(self, config: Config, exchange_name: str, account_name: str):
        self.config = config
        self.exchange_name = exchange_name
        self.account_name = account_name
        exchange_config = None

        for exch in config.exchanges:
            if exch.name == exchange_name and exch.account_name == account_name:
                exchange_config = exch
                break

        if not exchange_config:
            raise ValueError(f"Exchange {exchange_name} with account {account_name} not found in the configuration file.")

        api_key = exchange_config.api_key
        secret_key = exchange_config.api_secret
        passphrase = exchange_config.passphrase
        self.exchange = Exchange(self.exchange_name, api_key, secret_key, passphrase)

    def get_balance(self, quote, market_type=None, sub_type=None):
        if self.exchange_name == 'bitget':
            return self.exchange.get_balance_bitget(quote)
        elif self.exchange_name == 'bybit':
            return self.exchange.get_balance_bybit(quote)
        elif self.exchange_name == 'mexc':
            return self.exchange.get_balance_mexc(quote, market_type='swap')
        elif self.exchange_name == 'huobi':
            print("Huobi starting..")
            #return self.exchange.get_balance_huobi(quote, type=market_type, subType=sub_type)
        elif self.exchange_name == 'okx':
            #return self.exchange.get_balance_okx(quote)
            print(f"Unsupported for now")
        elif self.exchange_name == 'binance':
            return self.exchange.get_balance_binance(quote)
        elif self.exchange_name == 'phemex':
            print(f"Unsupported for now")

    def create_order(self, symbol, order_type, side, amount, price=None):
        return self.exchange.create_order(symbol, order_type, side, amount, price)

    def get_symbols(self):
        return self.exchange.symbols

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DirectionalScalper')
    parser.add_argument('--config', type=str, default='configs/config.json', help='Path to the configuration file')
    parser.add_argument('--exchange', type=str, help='The name of the exchange to use')
    parser.add_argument('--account_name', type=str, help='The name of the account to use')
    parser.add_argument('--strategy', type=str, help='The name of the strategy to use')
    parser.add_argument('--symbol', type=str, help='The trading symbol to use')
    parser.add_argument('--amount', type=str, help='The size to use')
    args = parser.parse_args()

    config_file_path = Path('configs/' + args.config)
    config = load_config(config_file_path)

    exchange_name = args.exchange
    account_name = args.account_name
    strategy_name = args.strategy
    symbol = args.symbol
    amount = args.amount

    print(f"Exchange name: {exchange_name}")
    print(f"Account name: {account_name}")
    print(f"Strategy name: {strategy_name}")
    print(f"Symbol: {symbol}")

    market_maker = DirectionalMarketMaker(config, exchange_name, account_name)

    manager = Manager(market_maker.exchange, api=config.api.mode, path=Path("data", config.api.filename), url=f"{config.api.url}{config.api.filename}")
    market_maker.manager = manager 

    quote = "USDT"
    balance = market_maker.get_balance(quote)
    print(f"Futures balance: {balance}")

    # quote = "USDT"
    # if exchange_name.lower() == 'huobi':
    #     #balance = market_maker.get_balance(quote, 'swap', 'linear')
    #     #print(f"Futures balance: {balance}")
    #     print(f"Loading huobi strategy..")
    # elif exchange_name.lower() == 'mexc':
    #     balance = market_maker.get_balance(quote, type='swap')
    #     print(f"Futures balance: {balance}")
    # else:
    #     balance = market_maker.get_balance(quote)
    #     print(f"Futures balance: {balance}")

    try:
        # Bitget strategies
        if strategy_name.lower() == 'bitget_hedge':
            strategy = BitgetHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)

        elif strategy_name.lower() == 'bitget_hedge_dynamic':
            strategy = BitgetDynamicHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bitget_timebased_dynamic':
            strategy = BitgetDynamicAuctionBasedStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bitget_fiveminute':
            strategy = BitgetFiveMinuteStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bitget_longonly_dynamic':
            strategy = BitgetLongOnlyDynamicStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bitget_shortonly_dynamic':
            strategy = BitgetShortOnlyDynamicStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bitget_hedge_grid_dynamic':
            strategy = BitgetGridStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        # Bybit strategies
        elif strategy_name.lower() == 'bybit_hedge':
            strategy = BybitHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)

        elif strategy_name.lower() == 'bybit_longonly':
            strategy = BybitLongStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)

        elif strategy_name.lower() == 'bybit_shortonly':
            strategy = BybitShortStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)
            
        elif strategy_name.lower() == 'bybit_longonly_dynamic_leverage':
            strategy = BybitLongOnlyDynamicLeverage(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_shortonly_dynamic_leverage':
            strategy = BybitShortOnlyDynamicLeverage(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_hedge_unified':
            strategy = BybitHedgeUnifiedStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)

        elif strategy_name.lower() == 'bybit_hedge_violent':
            strategy = BybitViolentHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)

        elif strategy_name.lower() == 'bybit_auto_hedge':
            strategy = BybitAutoHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_auto_hedge_maker':
            strategy = BybitAutoHedgeStrategyMaker(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_auto_hedge_maker_v2':
            strategy = BybitAutoHedgeStrategyMakerMFIRSI(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_auto_hedge_maker_eritrend':
            strategy = BybitAutoHedgeStrategyMakerERITrend(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_hedge_maker_mfirsi_trend':
            strategy = BybitAutoHedgeMFIRSITrendMaker(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_auto_hedge_mfi':
            strategy = BybitAutoHedgeStrategyMFIRSI(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_auto_mfirsi_trigger':
            strategy = BybitHedgeMFIRSITrigger(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_hedge_mfirsionly_maker':
            strategy = BybitHedgeMFIRSITriggerPostOnly(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)
    
        elif strategy_name.lower() == 'bybit_hedge_mfirsionly_maker_avoidfees':
            strategy = BybitHedgeMFIRSITriggerPostOnlyAvoidFees(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'bybit_hedge_mfirsi_maker':
            strategy = BybitAutoHedgeMFIRSIPostOnly(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        # Huobi strategies
        elif strategy_name.lower() == 'huobi_auto_hedge':
            strategy = HuobiAutoHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot, config, symbol)
            strategy.run(symbol, amount)

        # Mexc Strategies
        elif strategy_name.lower() == 'mexc_hedge':
            strategy = MEXCHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)

        # OKX strategies
        elif strategy_name.lower() == 'okx_hedge':
            strategy = OKXHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)

        # Binance Strategies
        elif strategy_name.lower() == 'binance_auto_hedge':
            strategy = BinanceAutoHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)

        elif strategy_name.lower() == 'binance_auto_hedge_maker':
            strategy = BinanceAutoHedgeMakerStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol)
        
        # Phemex strategies
        elif strategy_name.lower() == 'phemex_hedge':
            strategy = PhemexHedgeStrategy(market_maker.exchange, market_maker.manager, config.bot)
            strategy.run(symbol, amount)
        
        else:
            print("Strategy not recognized. Please choose a valid strategy.")
    except ccxt.ExchangeError as e:
        print(f"An error occurred while executing the strategy: {e}")
