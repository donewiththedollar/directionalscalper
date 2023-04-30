import time
from .strategy import Strategy

class HuobiHedgeStrategy(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config)
        self.manager = manager

    def run(self, symbol, amount):
        wallet_exposure = self.config.wallet_exposure

        while True:
            print(f"Huobi strategy running")

            time.sleep(30)
            