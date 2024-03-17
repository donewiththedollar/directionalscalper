import uuid
from .exchange import Exchange
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

class BinanceExchange(Exchange):
    def __init__(self, api_key, secret_key, passphrase=None, market_type='swap'):
        super().__init__('binance', api_key, secret_key, passphrase, market_type)
    
    def get_market_data_binance(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0, "step_size": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)
                
            if "precision" in symbol_data:
                values["precision"] = symbol_data["precision"]["price"]
            if "limits" in symbol_data:
                values["min_qty"] = symbol_data["limits"]["amount"]["min"]

            # Fetch positions
            positions = self.exchange.fetch_positions()

            for position in positions:
                if position['symbol'] == symbol:
                    values["leverage"] = float(position['leverage'])

            # Fetch step size
            if "info" in symbol_data and "filters" in symbol_data["info"]:
                for filter in symbol_data["info"]["filters"]:
                    if filter['filterType'] == 'LOT_SIZE':
                        values["step_size"] = filter['stepSize']

        except Exception as e:
            logging.info(f"An unknown error occurred in get_market_data_binance(): {e}")
        return values
    
    def get_balance_binance(self, symbol: str):
        if self.exchange.has['fetchBalance']:
            # Fetch the balance
            balance = self.exchange.fetch_balance(params={'type': 'future'})
            #print(balance)

            # Find the symbol balance
            for currency_balance in balance['info']['assets']:
                if currency_balance['asset'] == symbol:
                    return float(currency_balance['walletBalance'])
        return None
    
    def print_positions_structure_binance(self):
        try:
            data = self.exchange.fetch_positions_risk()
            print(data)
        except Exception as e:
            logging.info(f"An unknown error occurred: {e}")

    # Binance
    def get_positions_binance(self, symbol):
        values = {
            "long": {
                "qty": 0.0,
                "price": 0.0,
                "realised": 0,
                "cum_realised": 0,
                "upnl": 0,
                "upnl_pct": 0,
                "liq_price": 0,
                "entry_price": 0,
            },
            "short": {
                "qty": 0.0,
                "price": 0.0,
                "realised": 0,
                "cum_realised": 0,
                "upnl": 0,
                "upnl_pct": 0,
                "liq_price": 0,
                "entry_price": 0,
            },
        }
        try:
            position_data = self.exchange.fetch_positions_risk([symbol])
            #print(position_data)
            if len(position_data) > 0:
                for position in position_data:
                    #print(position["info"])
                    position_side = position["info"]["positionSide"].lower()
                    if position_side == "both":
                        # Adjust for positions with side 'both'
                        long_qty = float(position["info"]["positionAmt"])
                        short_qty = -long_qty  # Assume opposite quantity for short side
                        position_side = "long"
                        # Update long side values
                        values[position_side]["qty"] = long_qty
                        values[position_side]["price"] = float(position["info"]["entryPrice"])
                        values[position_side]["realised"] = round(float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["cum_realised"] = round(float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl"] = round(float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl_pct"] = 0
                        values[position_side]["liq_price"] = float(position["info"]["liquidationPrice"] or 0)
                        values[position_side]["entry_price"] = float(position["info"]["entryPrice"])
                        # Update short side values
                        position_side = "short"
                        values[position_side]["qty"] = short_qty
                        values[position_side]["price"] = float(position["info"]["entryPrice"])
                        values[position_side]["realised"] = round(-float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["cum_realised"] = round(-float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl"] = round(-float(position["info"]["unRealizedProfit"]), 4)
                        values[position_side]["upnl_pct"] = 0
                        values[position_side]["liq_price"] = float(position["info"]["liquidationPrice"] or 0)
                        values[position_side]["entry_price"] = float(position["info"]["entryPrice"])
                    else:
                        # Directly update values when position_side is 'long' or 'short'
                        qty = float(position["info"]["positionAmt"]) if position["info"]["positionAmt"] else 0.0
                        entry_price = float(position["info"]["entryPrice"]) if position["info"]["entryPrice"] else 0.0
                        unrealized_profit = float(position["info"]["unRealizedProfit"]) if position["info"]["unRealizedProfit"] else 0.0
                        values[position_side]["qty"] = qty
                        values[position_side]["price"] = entry_price
                        values[position_side]["realised"] = round(unrealized_profit, 4)
                        values[position_side]["cum_realised"] = round(unrealized_profit, 4)
                        values[position_side]["upnl"] = round(unrealized_profit, 4)
                        values[position_side]["upnl_pct"] = 0
                        values[position_side]["liq_price"] = float(position["info"]["liquidationPrice"] or 0)
                        values[position_side]["entry_price"] = entry_price
        except Exception as e:
            logging.info(f"An unknown error occurred in get_positions_binance(): {e}")
        return values
