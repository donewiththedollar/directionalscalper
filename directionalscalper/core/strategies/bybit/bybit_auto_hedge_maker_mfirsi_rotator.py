import time
import math
from threading import Thread, Lock
from ..strategy import Strategy
from datetime import datetime, timedelta
from typing import Tuple
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box
import pandas as pd
import ta
import logging
from ..logger import Logger

logging = Logger(logger_name="BybitAutoRotatorMFIRSIRotator", filename="BybitAutoRotatorMFIRSIRotator.log", stream=True)

class BybitAutoHedgeStrategyMakerMFIRSIRotator(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.all_symbol_data = {}
        self.last_long_tp_update = datetime.now()
        self.last_short_tp_update = datetime.now()
        self.next_long_tp_update = self.calculate_next_update_time()
        self.next_short_tp_update = self.calculate_next_update_time()
        self.last_cancel_time = 0
        self.current_wallet_exposure = 1.0
        self.short_tp_distance_percent = 0.0
        self.short_expected_profit_usdt = 0.0
        self.long_tp_distance_percent = 0.0
        self.long_expected_profit_usdt = 0.0
        self.printed_trade_quantities = False
        self.checked_amount_validity = False
        self.long_pos_leverage = 1.0
        self.short_pos_leverage = 1.0
        self.max_long_trade_qty = None
        self.max_short_trade_qty = None
        self.initial_max_long_trade_qty = None
        self.initial_max_short_trade_qty = None
        self.long_leverage_increased = False
        self.short_leverage_increased = False
        self.version = "2.0.6"
        self.lock = Lock()
        self.rows = {}
        # Recreate the table
        self.table = Table(header_style="bold magenta", title=f"Directional Scalper MFIRSI {self.version}")
        self.table.add_column("Symbol")
        self.table.add_column("Min. Qty")
        self.table.add_column("Price")
        self.table.add_column("Balance")
        self.table.add_column("Available Bal.")
        self.table.add_column("1m Vol")
        self.table.add_column("5m Spread")
        self.table.add_column("Trend")
        self.table.add_column("Long Pos. Qty")
        self.table.add_column("Short Pos. Qty")
        self.table.add_column("Long uPNL")
        self.table.add_column("Short uPNL")
        self.table.add_column("Long cum. uPNL")
        self.table.add_column("Short cum. uPNL")
        self.table.add_column("Long Pos. Price")
        self.table.add_column("Short Pos. Price")


    def generate_main_table(self, symbol_data):
        try:
            symbol = symbol_data['symbol']

            # Create row data
            row_data = [
                symbol,
                str(symbol_data['min_qty']),
                str(symbol_data['min_qty']),
                str(symbol_data['current_price']),
                str(symbol_data['balance']),
                str(symbol_data['available_bal']),
                str(symbol_data['volume']),
                str(symbol_data['spread']),
                str(symbol_data['trend']),
                str(symbol_data['long_pos_qty']),
                str(symbol_data['short_pos_qty']),
                str(symbol_data['long_upnl']),
                str(symbol_data['short_upnl']),
                str(symbol_data['long_cum_pnl']),
                str(symbol_data['short_cum_pnl']),
                str(symbol_data['long_pos_price']),
                str(symbol_data['short_pos_price'])
                # ... convert all symbol_data values to string and add them here ...
            ]

            # If symbol not in table yet, add a new row for it
            if symbol not in self.all_symbol_data:
                new_row = self.table.add_row(*row_data)
                self.all_symbol_data[symbol] = new_row
            else:
                # Update the row for this symbol in the table
                row = self.all_symbol_data[symbol]
                for i, cell in enumerate(row.cells):
                    cell.text = row_data[i]

            return self.table
        except Exception as e:
            logging.info(f"Exception caught {e}")
            return Table()
        
    # def generate_main_table(self, symbol_data):
    #     try:
    #         symbol = symbol_data['symbol']

    #         # Update the rows dictionary
    #         self.rows[symbol] = [
    #             symbol,
    #             str(symbol_data['min_qty']),
    #             str(symbol_data['current_price']),
    #             str(symbol_data['balance']),
    #             str(symbol_data['available_bal']),
    #             str(symbol_data['volume']),
    #             str(symbol_data['spread']),
    #             str(symbol_data['trend']),
    #             str(symbol_data['long_pos_qty']),
    #             str(symbol_data['short_pos_qty']),
    #             str(symbol_data['long_upnl']),
    #             str(symbol_data['short_upnl']),
    #             str(symbol_data['long_cum_pnl']),
    #             str(symbol_data['short_cum_pnl']),
    #             str(symbol_data['long_pos_price']),
    #             str(symbol_data['short_pos_price'])
    #         ]

    #         # Recreate the table
    #         self.table = Table(header_style="bold magenta", title=f"Directional Scalper MFIRSI {self.version}")
    #         self.table.add_column("Symbol")
    #         self.table.add_column("Min. Qty")
    #         self.table.add_column("Price")
    #         self.table.add_column("Balance")
    #         self.table.add_column("Available Bal.")
    #         self.table.add_column("1m Vol")
    #         self.table.add_column("5m Spread")
    #         self.table.add_column("Trend")
    #         self.table.add_column("Long Pos. Qty")
    #         self.table.add_column("Short Pos. Qty")
    #         self.table.add_column("Long uPNL")
    #         self.table.add_column("Short uPNL")
    #         self.table.add_column("Long cum. uPNL")
    #         self.table.add_column("Short cum. uPNL")
    #         self.table.add_column("Long Pos. Price")
    #         self.table.add_column("Short Pos. Price")

    #         # Add all the rows from the dictionary
    #         for row_data in self.rows.values():
    #             self.table.add_row(*row_data)

    #         return self.table
    #     except Exception as e:
    #         logging.info(f"Exception caught {e}")
    #         return Table()

# Works well but does not have rows per symbol    
    # def generate_main_table(self, symbol_data):
    #     try:
    #         table = Table(header_style="bold magenta", title=f"Directional Scalper MFIRSI {self.version}")
    #         # Define the columns
    #         table.add_column("Symbol")
    #         table.add_column("Min. Qty")
    #         table.add_column("Price")
    #         table.add_column("Balance")
    #         table.add_column("Available Bal.")
    #         table.add_column("1m Vol")
    #         table.add_column("5m Spread")
    #         table.add_column("Trend")
    #         table.add_column("Long Pos. Qty")
    #         table.add_column("Short Pos. Qty")
    #         table.add_column("Long uPNL")
    #         table.add_column("Short uPNL")
    #         table.add_column("Long cum. uPNL")
    #         table.add_column("Short cum. uPNL")
    #         table.add_column("Long Pos. Price")
    #         table.add_column("Short Pos. Price")
    #         # ... continue adding columns ...


    #         # Add a row for the symbol
    #         table.add_row(
    #             symbol_data['symbol'],
    #             str(symbol_data['min_qty']),
    #             str(symbol_data['current_price']),
    #             str(symbol_data['balance']),
    #             str(symbol_data['available_bal']),
    #             str(symbol_data['volume']),
    #             str(symbol_data['spread']),
    #             str(symbol_data['trend']),
    #             str(symbol_data['long_pos_qty']),
    #             str(symbol_data['short_pos_qty']),
    #             str(symbol_data['long_upnl']),
    #             str(symbol_data['short_upnl']),
    #             str(symbol_data['long_cum_pnl']),
    #             str(symbol_data['short_cum_pnl']),
    #             str(symbol_data['long_pos_price']),
    #             str(symbol_data['short_pos_price'])
    #             # ... continue converting all values to strings ...
    #         )

            
    #         return table

    #     except Exception as e:
    #         logging.info(f"Exception caught {e}")
    #         return Table()

    def run(self, symbols):
        threads = [Thread(target=self.run_single_symbol, args=(symbol, self.lock)) for symbol in symbols]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    # def run(self, symbol):
    #     threads = [Thread(target=self.run_single_symbol, args=(symbol,))]

    #     for thread in threads:
    #         thread.start()

    #     for thread in threads:
    #         thread.join()
            
    #def run_single_symbol(self, symbol):
    def run_single_symbol(self, symbol, lock):
        print(f"Running for symbol (inside run_single_symbol method): {symbol}")
        console = Console()
        live = Live(console=console, refresh_per_second=10)

        quote_currency = "USDT"
        max_retries = 5
        retry_delay = 5

        # Initialize exchange-related variables outside the live context
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        max_leverage = self.exchange.get_max_leverage_bybit(symbol)

        logging.info("Setting up exchange")
        self.exchange.setup_exchange_bybit(symbol)

        logging.info("Setting leverage")
        if current_leverage != max_leverage:
            logging.info(f"Current leverage is not at maximum. Setting leverage to maximum. Maximum is {max_leverage}")
            self.exchange.set_leverage_bybit(max_leverage, symbol)

        previous_five_minute_distance = None
        previous_thirty_minute_distance = None
        previous_one_hour_distance = None
        previous_four_hour_distance = None

        with live:
            while True:
                # Get API data
                data = self.manager.get_data()
                one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
                one_hour_volume = self.manager.get_asset_value(symbol, data, "1hVol")
                one_minute_distance = self.manager.get_asset_value(symbol, data, "1mSpread")
                five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
                thirty_minute_distance = self.manager.get_asset_value(symbol, data, "30mSpread")
                one_hour_distance = self.manager.get_asset_value(symbol, data, "1hSpread")
                four_hour_distance = self.manager.get_asset_value(symbol, data, "4hSpread")
                trend = self.manager.get_asset_value(symbol, data, "Trend")
                mfirsi_signal = self.manager.get_asset_value(symbol, data, "MFI")
                eri_trend = self.manager.get_asset_value(symbol, data, "ERI Trend")
                rotatorsymbols = self.manager.get_symbols()
                #rotator_symbols = self.manager.get_auto_rotate_symbols(self.config.min_qty_threshold)
                rotator_symbols = self.manager.get_auto_rotate_symbols()

                print(f"{rotator_symbols}")

                quote_currency = "USDT"

                for i in range(max_retries):
                    try:
                        total_equity = self.exchange.get_balance_bybit(quote_currency)
                        break
                    except Exception as e:
                        if i < max_retries - 1:
                            logging.info(f"Error occurred while fetching balance: {e}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            raise e
                        
                #logging.info(f"Total equity: {total_equity}")

                for i in range(max_retries):
                    try:
                        available_equity = self.exchange.get_available_balance_bybit(quote_currency)
                        break
                    except Exception as e:
                        if i < max_retries - 1:
                            logging.info(f"Error occurred while fetching available balance: {e}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            raise e

                #logging.info(f"Available equity: {available_equity}")

                current_price = self.exchange.get_current_price(symbol)
                market_data = self.get_market_data_with_retry(symbol, max_retries = 5, retry_delay = 5)
                #contract_size = self.exchange.get_contract_size_bybit(symbol)
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]

                if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
                    self.max_long_trade_qty = self.max_short_trade_qty = self.calc_max_trade_qty(total_equity,
                                                                                                best_ask_price,
                                                                                                max_leverage)

                    # Set initial quantities if they're None
                    if self.initial_max_long_trade_qty is None:
                        self.initial_max_long_trade_qty = self.max_long_trade_qty
                        logging.info(f"Initial max trade qty set to {self.initial_max_long_trade_qty}")
                    if self.initial_max_short_trade_qty is None:
                        self.initial_max_short_trade_qty = self.max_short_trade_qty  
                        logging.info(f"Initial trade qty set to {self.initial_max_short_trade_qty}")                                                            
                            
                # Calculate the dynamic amount
                long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty
                short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty

                min_qty = float(market_data["min_qty"])
                min_qty_str = str(min_qty)

                # Get the precision level of the minimum quantity
                if ".0" in min_qty_str:
                    # The minimum quantity does not have a fractional part, precision is 0
                    precision_level = 0
                else:
                    # The minimum quantity has a fractional part, get its precision level
                    precision_level = len(min_qty_str.split(".")[1])

                # # Get the precision level of the minimum quantity
                # if ".0" in min_qty_str:
                #     # The minimum quantity has a fractional part, get its precision level
                #     precision_level = len(min_qty_str.split(".")[1])
                # else:
                #     # The minimum quantity does not have a fractional part, precision is 0
                #     precision_level = 0

                # Round the amount to the precision level of the minimum quantity
                long_dynamic_amount = round(long_dynamic_amount, precision_level)
                short_dynamic_amount = round(short_dynamic_amount, precision_level)

                self.check_amount_validity_once_bybit(long_dynamic_amount, symbol)
                self.check_amount_validity_once_bybit(short_dynamic_amount, symbol)

                # Check if the amount is less than the minimum quantity allowed by the exchange
                if long_dynamic_amount < min_qty:
                    logging.info(f"Dynamic amount too small for 0.001x, using min_qty")
                    long_dynamic_amount = min_qty
                
                if short_dynamic_amount < min_qty:
                    logging.info(f"Dynamic amount too small for 0.001x, using min_qty")
                    short_dynamic_amount = min_qty

                self.print_trade_quantities_once_bybit(self.max_long_trade_qty)
                self.print_trade_quantities_once_bybit(self.max_short_trade_qty)

                # Get the 1-minute moving averages
                logging.info(f"Fetching MA data")
                m_moving_averages = self.manager.get_1m_moving_averages(symbol)
                m5_moving_averages = self.manager.get_5m_moving_averages(symbol)
                ma_6_high = m_moving_averages["MA_6_H"]
                ma_6_low = m_moving_averages["MA_6_L"]
                ma_3_low = m_moving_averages["MA_3_L"]
                ma_3_high = m_moving_averages["MA_3_H"]
                ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
                ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

                position_data = self.exchange.get_positions_bybit(symbol)

                short_pos_qty = position_data["short"]["qty"]
                long_pos_qty = position_data["long"]["qty"]

                # get liquidation prices
                short_liq_price = position_data["short"]["liq_price"]
                long_liq_price = position_data["long"]["liq_price"]

                self.bybit_reset_position_leverage_long(long_pos_qty, total_equity, best_ask_price, max_leverage)
                self.bybit_reset_position_leverage_short(short_pos_qty, total_equity, best_ask_price, max_leverage)

                short_upnl = position_data["short"]["upnl"]
                long_upnl = position_data["long"]["upnl"]

                cum_realised_pnl_long = position_data["long"]["cum_realised"]
                cum_realised_pnl_short = position_data["short"]["cum_realised"]

                short_pos_price = position_data["short"]["price"] if short_pos_qty > 0 else None
                long_pos_price = position_data["long"]["price"] if long_pos_qty > 0 else None

                short_take_profit = None
                long_take_profit = None

                if five_minute_distance != previous_five_minute_distance:
                    short_take_profit = self.calculate_short_take_profit_spread_bybit(short_pos_price, symbol, five_minute_distance)
                    long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, five_minute_distance)
                else:
                    if short_take_profit is None or long_take_profit is None:
                        short_take_profit = self.calculate_short_take_profit_spread_bybit(short_pos_price, symbol, five_minute_distance)
                        long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, five_minute_distance)
                        
                previous_five_minute_distance = five_minute_distance

                should_short = self.short_trade_condition(best_ask_price, ma_3_high)
                should_long = self.long_trade_condition(best_bid_price, ma_3_low)

                should_add_to_short = False
                should_add_to_long = False
            
                if short_pos_price is not None:
                    should_add_to_short = short_pos_price < ma_6_low and self.short_trade_condition(best_ask_price, ma_6_high)
                    self.short_tp_distance_percent = ((short_take_profit - short_pos_price) / short_pos_price) * 100
                    self.short_expected_profit_usdt = abs(self.short_tp_distance_percent / 100 * short_pos_price * short_pos_qty)
                    logging.info(f"Short TP price: {short_take_profit}, TP distance in percent: {-self.short_tp_distance_percent:.2f}%, Expected profit: {self.short_expected_profit_usdt:.2f} USDT")

                if long_pos_price is not None:
                    should_add_to_long = long_pos_price > ma_6_high and self.long_trade_condition(best_bid_price, ma_6_low)
                    self.long_tp_distance_percent = ((long_take_profit - long_pos_price) / long_pos_price) * 100
                    self.long_expected_profit_usdt = self.long_tp_distance_percent / 100 * long_pos_price * long_pos_qty
                    logging.info(f"Long TP price: {long_take_profit}, TP distance in percent: {self.long_tp_distance_percent:.2f}%, Expected profit: {self.long_expected_profit_usdt:.2f} USDT")
                    
                logging.info(f"Short condition: {should_short}")
                logging.info(f"Long condition: {should_long}")
                logging.info(f"Add short condition: {should_add_to_short}")
                logging.info(f"Add long condition: {should_add_to_long}")

                # symbol_data = {
                #     'symbol': symbol,
                #     'min_qty': min_qty,
                #     'current_price': current_price,
                #     'balance': total_equity,
                #     'available_bal': available_equity,
                #     'volume': one_minute_volume,
                #     'spread': five_minute_distance,
                #     'trend': trend,
                #     'long_pos_qty': long_pos_qty,
                #     'short_pos_qty': short_pos_qty,
                #     'long_upnl': long_upnl,
                #     'short_upnl': short_upnl,
                #     'long_cum_pnl': cum_realised_pnl_long,
                #     'short_cum_pnl': cum_realised_pnl_short,
                #     'long_pos_price': long_pos_price,
                #     'short_pos_price': short_pos_price
                #     # ... continue adding all parameters ...
                # }

                # live.update(self.generate_main_table(symbol_data))

                symbol_data = {
                    'symbol': symbol,
                    'min_qty': min_qty,
                    'current_price': current_price,
                    'balance': total_equity,
                    'available_bal': available_equity,
                    'volume': one_minute_volume,
                    'spread': five_minute_distance,
                    'trend': trend,
                    'long_pos_qty': long_pos_qty,
                    'short_pos_qty': short_pos_qty,
                    'long_upnl': long_upnl,
                    'short_upnl': short_upnl,
                    'long_cum_pnl': cum_realised_pnl_long,
                    'short_cum_pnl': cum_realised_pnl_short,
                    'long_pos_price': long_pos_price,
                    'short_pos_price': short_pos_price
                    # ... continue adding all parameters ...
                }

                # # Generate and update the table
                # live.update(self.generate_main_table(symbol_data))

                # Acquire the lock before updating the table
                lock.acquire()
                try:
                    # Update the table data for this symbol
                    self.generate_main_table(symbol_data)

                    # Update the live table
                    live.update(self.table)
                finally:
                    # Release the lock
                    lock.release()
                
                # symbol_data = {
                #     'symbol': symbol,
                #     'min_qty': min_qty,
                #     'current_price': current_price,
                #     'balance': total_equity,
                #     'available_bal': available_equity,
                #     'volume': one_minute_volume,
                #     'spread': five_minute_distance,
                #     'trend': trend,
                #     'long_pos_qty': long_pos_qty,
                #     'short_pos_qty': short_pos_qty,
                #     'long_upnl': long_upnl,
                #     'short_upnl': short_upnl,
                #     'long_cum_pnl': cum_realised_pnl_long,
                #     'short_cum_pnl': cum_realised_pnl_short,
                #     'long_pos_price': long_pos_price,
                #     'short_pos_price': short_pos_price
                #     # ... continue adding all parameters ...
                # }

                # # Update the data for this symbol
                # self.all_symbol_data[symbol] = symbol_data

                # # Generate and update the table
                # live.update(self.generate_main_table())


                open_orders = self.exchange.get_open_orders(symbol)

                # Entry logic
                # Long and short entry placement
                self.bybit_hedge_entry_maker(symbol, trend, one_minute_volume, five_minute_distance, min_vol, min_dist, long_dynamic_amount, short_dynamic_amount, long_pos_qty, short_pos_qty, long_pos_price, short_pos_price, should_long, should_short, should_add_to_long, should_add_to_short)

                # Take profit placement 

                # Call the function to update long take profit spread
                if long_pos_qty > 0 and long_take_profit is not None:
                    self.bybit_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders)

                # Call the function to update short take profit spread
                if short_pos_qty > 0 and short_take_profit is not None:
                    self.bybit_hedge_placetp_maker(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders)

                # Take profit spread replacement
                if long_pos_qty > 0 and long_take_profit is not None:
                    self.next_long_tp_update = self.update_take_profit_spread_bybit(symbol, long_pos_qty, long_take_profit, positionIdx=1, order_side="sell", open_orders=open_orders, next_tp_update=self.next_long_tp_update)

                if short_pos_qty > 0 and short_take_profit is not None:
                    self.next_short_tp_update = self.update_take_profit_spread_bybit(symbol, short_pos_qty, short_take_profit, positionIdx=2, order_side="buy", open_orders=open_orders, next_tp_update=self.next_short_tp_update)

                # Cancel entries
                self.cancel_entries_bybit(symbol, best_ask_price, ma_1m_3_high, ma_5m_3_high)

                time.sleep(30)
