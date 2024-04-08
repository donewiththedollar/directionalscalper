import logging
import math
from directionalscalper.core.strategies.base_strategy import BaseStrategy
from directionalscalper.core.logger import Logger

logging = Logger(logger_name="LBankStrategy", filename="LBankStrategy.log", stream=True)

class LBankStrategy(BaseStrategy):
    def __init__(self, exchange, config, manager, symbols_allowed=None):
        super().__init__(exchange, config, manager, symbols_allowed)

    def calculate_dynamic_amounts(self, symbol, total_equity, best_ask_price, best_bid_price):
        # Fetch market data to get the minimum trade quantity for the symbol
        market_data = self.exchange.get_market_data_lbank(symbol)
        min_qty = float(market_data["min_qty"])
        min_qty_usd_value = min_qty * best_ask_price

        # Calculate dynamic entry sizes based on risk parameters
        max_equity_for_long_trade = total_equity * self.wallet_exposure_limit
        max_long_position_value = max_equity_for_long_trade * self.user_defined_leverage_long
        long_entry_size = max(max_long_position_value / best_ask_price, min_qty_usd_value / best_ask_price)

        max_equity_for_short_trade = total_equity * self.wallet_exposure_limit
        max_short_position_value = max_equity_for_short_trade * self.user_defined_leverage_short
        short_entry_size = max(max_short_position_value / best_bid_price, min_qty_usd_value / best_bid_price)

        # Adjusting entry sizes based on the symbol's minimum quantity precision
        qty_precision = self.exchange.get_symbol_precision_lbank(symbol)[1]
        if qty_precision is None:
            long_entry_size_adjusted = round(long_entry_size)
            short_entry_size_adjusted = round(short_entry_size)
        else:
            long_entry_size_adjusted = round(long_entry_size, -int(math.log10(qty_precision)))
            short_entry_size_adjusted = round(short_entry_size, -int(math.log10(qty_precision)))

        return long_entry_size_adjusted, short_entry_size_adjusted

    def lbank_hedge_entry_maker(self, symbol, trend, mfi, one_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price):
        if one_minute_volume is not None and five_minute_distance is not None:
            if one_minute_volume > min_vol and five_minute_distance > min_dist:
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                if trend.lower() == "long" and mfi.lower() == "long" and long_pos_qty == 0:
                    logging.info(f"Placing initial long entry")
                    self.exchange.create_limit_order_lbank(symbol, "buy", long_dynamic_amount, best_bid_price)
                    logging.info(f"Placed initial long entry")
                elif trend.lower() == "long" and mfi.lower() == "long" and long_pos_qty < self.max_long_trade_qty_per_symbol[symbol] and best_bid_price < long_pos_price:
                    logging.info(f"Placing additional long entry")
                    self.exchange.create_limit_order_lbank(symbol, "buy", long_dynamic_amount, best_bid_price)

                if trend.lower() == "short" and mfi.lower() == "short" and short_pos_qty == 0:
                    logging.info(f"Placing initial short entry")
                    self.exchange.create_limit_order_lbank(symbol, "sell", short_dynamic_amount, best_ask_price)
                    logging.info("Placed initial short entry")
                elif trend.lower() == "short" and mfi.lower() == "short" and short_pos_qty < self.max_short_trade_qty_per_symbol[symbol] and best_ask_price > short_pos_price:
                    logging.info(f"Placing additional short entry")
                    self.exchange.create_limit_order_lbank(symbol, "sell", short_dynamic_amount, best_ask_price)