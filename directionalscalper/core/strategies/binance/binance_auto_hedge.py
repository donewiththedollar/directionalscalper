import time
import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, ROUND_DOWN
from typing import Tuple
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich import box
import threading
import os
import logging
from ..logger import Logger

from directionalscalper.core.strategies.binance.binance_strategy import BinanceStrategy

logging = Logger(logger_name="BinanceAutoHedge", filename="BinanceAutoHedge.log", stream=True)

class BinanceAutoHedgeStrategy(BinanceStrategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
        self.last_cancel_time = 0
        self.current_wallet_exposure = 1.0
        self.printed_trade_quantities = False
        self.max_long_trade_qty = None
        self.max_short_trade_qty = None
        self.initial_max_long_trade_qty = None
        self.initial_max_short_trade_qty = None
        self.long_leverage_increased = False
        self.short_leverage_increased = False
        self.checked_amount_validity_binance = False
        self.long_leverage_increased = False
        self.short_leverage_increased = False
        self.long_pos_leverage = 1.0
        self.short_pos_leverage = 1.0
        self.short_tp_distance_percent = 0.0
        self.short_expected_profit_usdt = 0.0
        self.long_tp_distance_percent = 0.0
        self.long_expected_profit_usdt = 0.0
        self.version = "2.0.6"

    def generate_main_table(self, symbol, min_qty, current_price, balance, volume, spread, trend, long_pos_qty, short_pos_qty, long_upnl, short_upnl, long_cum_pnl, short_cum_pnl, long_pos_price, short_pos_price, long_dynamic_amount, short_dynamic_amount, long_take_profit, short_take_profit, long_pos_lev, short_pos_lev, long_max_trade_qty, short_max_trade_qty, long_expected_profit, short_expected_profit, should_long, should_add_to_long, should_short, should_add_to_short, eri_trend):
        try:
            table = Table(show_header=False, header_style="bold magenta", title=f"Directional Scalper {self.version}")
            table.add_column("Key")
            table.add_column("Value")
            #min_vol_dist_data = self.manager.get_min_vol_dist_data(self.symbol)
            #mode = self.find_mode()
            #trend = self.find_trend()
            #market_data = self.get_market_data()

            table_data = {
                "Symbol": symbol,
                "Price": current_price,
                "Balance": balance,
                #"Available bal.": available_bal,
                "Long MAX QTY": long_max_trade_qty,
                "Short MAX QTY": short_max_trade_qty,
                "Long entry QTY": long_dynamic_amount,
                "Short entry QTY": short_dynamic_amount,
                "Long pos. QTY": long_pos_qty,
                "Short pos. QTY": short_pos_qty,
                "Long uPNL": long_upnl,
                "Short uPNL": short_upnl,
                "Long cum. uPNL": long_cum_pnl,
                "Short cum. uPNL": short_cum_pnl,
                "Long pos. price": long_pos_price,
                "Long take profit": long_take_profit,
                "Long expected profit": "{:.2f} USDT".format(long_expected_profit),
                "Short pos. price": short_pos_price,
                "Short take profit": short_take_profit,
                "Short expected profit": "{:.2f} USDT".format(short_expected_profit),
                "Long pos. lev.": long_pos_lev,
                "Short pos. lev.": short_pos_lev,
               # "Long liq price": long_liq_price,
               # "Short liq price": short_liq_price,
                "1m Vol": volume,
                "5m Spread:": spread,
                "Trend": trend,
                "ERI Trend": eri_trend,
                "Long condition": should_long,
                "Add long cond.": should_add_to_long,
                "Short condition": should_short,
                "Add short cond.": should_add_to_short,
                "Min. volume": self.config.min_volume,
                "Min. spread": self.config.min_distance,
                "Min. qty": min_qty,
            }

            # for key, value in table_data.items():
            #     try:
            #         if float(value) < 0:
            #             table.add_row(Text(key, style="bold blue"), Text(str(value), style="bold red"))
            #         else:
            #             table.add_row(Text(key, style="bold blue"), Text(str(value), style="bold cyan"))
            #     except ValueError:
            #         # Value could not be converted to a float, so it's not a number
            #         table.add_row(Text(key, style="bold blue"), Text(str(value), style="bold cyan"))

            # for key, value in table_data.items():
            #     table.add_row(Text(key, style="bold blue"), Text(str(value), style="bold cyan"))

            for key, value in table_data.items():
                table.add_row(key, str(value))
            
            return table

        except Exception as e:
            logging.info(f"Exception caught {e}")
            return Table()

    def run(self, symbol):
        console = Console()

        live = Live(console=console, refresh_per_second=2)

        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        max_leverage = self.exchange.get_max_leverage_binance(symbol)
        quote_currency = "USDT"
        max_retries = 5
        retry_delay = 5
        #print(f"Max leverage: {max_leverage}")
        #current_leverage = self.exchange.get_current_leverage_bybit(symbol)
        min_notional = 5.10

        # Set to hedge mode
        self.exchange.set_hedge_mode_binance()

        with live:
            while True:
            # Get API data
                data = self.manager.get_data()
                one_minute_volume = self.manager.get_asset_value(symbol, data, "1mVol")
                five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
                trend = self.manager.get_asset_value(symbol, data, "Trend")
                eri_trend = self.manager.get_asset_value(symbol, data, "ERI Trend")

                print(f"Binance auto hedge strategy running")
                print(f"Min volume: {min_vol}")
                print(f"Min distance: {min_dist}")
                print(f"1m Volume: {one_minute_volume}")
                print(f"5m Spread: {five_minute_distance}")
                print(f"Trend: {trend}")

                for i in range(max_retries):
                    try:
                        total_equity = self.exchange.get_balance_binance(quote_currency)
                        print(f"Total equity: {total_equity}")
                        break
                    except Exception as e:
                        if i < max_retries - 1:
                            logging.info(f"Error occurred while fetching available balance: {e}. Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            raise e

                current_price = self.exchange.get_current_price_binance(symbol)
                print(f"Current price: {current_price}")

                # market_info = self.exchange.get_symbol_info_binance(symbol)

                # print(market_info)

                market_data = self.get_market_data_with_retry_binance(symbol, max_retries = 5, retry_delay = 5)
                #print(f"Market data: {market_data}")
                best_ask_price = self.exchange.get_orderbook(symbol)['asks'][0][0]
                best_bid_price = self.exchange.get_orderbook(symbol)['bids'][0][0]
                print(f"Best bid: {best_bid_price}")
                print(f"Best ask: {best_ask_price}")

                #self.exchange.test_func()

                step_size = market_data['step_size']
                print(f"Step size: {step_size}")

                min_qty = market_data['min_qty']
                min_qty_str = str(min_qty)
                print(f"Min qty: {min_qty}")

                min_qty_notional = min_notional / current_price  # Compute minimum quantity
                print(f"Min qty based on notional: {min_qty_notional}")

                precision = int(-math.log10(float(step_size)))

                # Use floor division to make sure it's a multiple of step_size
                precise_min_qty = math.floor(min_qty_notional * 10**precision) / 10**precision

                print(f"Min qty rounded precision: {precise_min_qty}")

                # ensure the min_qty is respected
                min_qty = max(min_qty, precise_min_qty)

                print(f"Min qty value: {min_qty}")

                if self.max_long_trade_qty is None or self.max_short_trade_qty is None:
                    self.max_long_trade_qty = self.max_short_trade_qty = self.calc_max_trade_qty_binance(total_equity,
                                                                                                best_ask_price,
                                                                                                max_leverage, step_size)

                    # Set initial quantities if they're None
                    if self.initial_max_long_trade_qty is None:
                        self.initial_max_long_trade_qty = self.max_long_trade_qty
                        print(f"Initial max trade qty: {self.initial_max_long_trade_qty}")
                        logging.info(f"Initial max trade qty set to {self.initial_max_long_trade_qty}")
                    if self.initial_max_short_trade_qty is None:
                        self.initial_max_short_trade_qty = self.max_short_trade_qty  
                        print(f"Initial max trade qty short set to {self.initial_max_short_trade_qty}")
                        logging.info(f"Initial trade qty set to {self.initial_max_short_trade_qty}")                                                            
                            
                print(f"Max long trade qty: {self.max_long_trade_qty}")
                print(f"Max short trade qty: {self.max_short_trade_qty}")

                self.print_trade_quantities_once_bybit(self.max_long_trade_qty)
                self.print_trade_quantities_once_bybit(self.max_short_trade_qty)

                # Calculate the dynamic amount
                # Calculate the dynamic amounts based on the initial max trade quantities
                long_dynamic_amount = 0.001 * self.initial_max_long_trade_qty
                short_dynamic_amount = 0.001 * self.initial_max_short_trade_qty

                print(f"Long dynamic amount before adjustment: {long_dynamic_amount}")
                print(f"Short dynamic amount before adjustment: {short_dynamic_amount}")

                # Ensure dynamic amounts meet the minimum requirements
                long_dynamic_amount = max(long_dynamic_amount, precise_min_qty)
                short_dynamic_amount = max(short_dynamic_amount, precise_min_qty)

                print(f"Long dynamic amount after adjustment: {long_dynamic_amount}")
                print(f"Short dynamic amount after adjustment: {short_dynamic_amount}")

                # Get the precision of the step size
                precision = int(-math.log10(float(step_size)))

                # Round the amounts to the precision level of the step size
                long_dynamic_amount = round(long_dynamic_amount, precision)
                short_dynamic_amount = round(short_dynamic_amount, precision)

                print(f"Long dynamic amount after rounding: {long_dynamic_amount}")
                print(f"Short dynamic amount after rounding: {short_dynamic_amount}")

                # max_trade_qty = round(
                #     (float(total_equity) * wallet_exposure / float(best_ask_price))
                #     / (100 / max_leverage),
                #     int(float(market_data["min_qty"])),
                # )            
                
                #print(f"Max trade quantity for {symbol}: {max_trade_qty}")

                # #tiers = self.exchange.get_leverage_tiers_binance_binance([symbol])
                # tiers = self.exchange.get_leverage_tiers_binance_binance()

                # print(tiers)

                self.check_amount_validity_once_binance(long_dynamic_amount, symbol)
                self.check_amount_validity_once_binance(short_dynamic_amount, symbol)

                # Get the 1-minute moving averages
                print(f"Fetching MA data")
                m_moving_averages = self.manager.get_1m_moving_averages(symbol)
                m5_moving_averages = self.manager.get_5m_moving_averages(symbol)
                ma_6_high = m_moving_averages["MA_6_H"]
                ma_6_low = m_moving_averages["MA_6_L"]
                ma_3_low = m_moving_averages["MA_3_L"]
                ma_3_high = m_moving_averages["MA_3_H"]
                ma_1m_3_high = self.manager.get_1m_moving_averages(symbol)["MA_3_H"]
                ma_5m_3_high = self.manager.get_5m_moving_averages(symbol)["MA_3_H"]

                position_data = self.exchange.get_positions_binance(symbol)

                #self.exchange.print_positions_structure_binance()

                #print(f"Binance pos data: {position_data}")

                short_pos_qty = position_data["short"]["qty"]
                long_pos_qty = position_data["long"]["qty"]

                print(f"Short pos qty: {short_pos_qty}")
                print(f"Long pos qty: {long_pos_qty}")

                short_upnl = position_data["short"]["upnl"]
                long_upnl = position_data["long"]["upnl"]

                print(f"Short uPNL: {short_upnl}")
                print(f"Long uPNL: {long_upnl}")

                cum_realised_pnl_long = position_data["long"]["cum_realised"]
                cum_realised_pnl_short = position_data["short"]["cum_realised"]

                print(f"Short cum. PNL: {cum_realised_pnl_short}")
                print(f"Long cum. PNL: {cum_realised_pnl_long}")

                short_pos_price = position_data["short"]["price"] if abs(short_pos_qty) > 0 else None
                long_pos_price = position_data["long"]["price"] if abs(long_pos_qty) > 0 else None

                print(f"Long pos price {long_pos_price}")
                print(f"Short pos price {short_pos_price}")

                # Leverage increase / reset
                self.bybit_reset_position_leverage_long(long_pos_qty, total_equity, best_ask_price, max_leverage)
                self.bybit_reset_position_leverage_short(short_pos_qty, total_equity, best_ask_price, max_leverage)

                short_take_profit = None
                long_take_profit = None

                # Take profit calc
                if short_pos_qty != 0:
                    short_take_profit = self.calculate_short_take_profit_binance(short_pos_price, symbol)
                    print(f"Short take profit: {short_take_profit}")

                if long_pos_qty != 0:
                    long_take_profit = self.calculate_long_take_profit_binance(long_pos_price, symbol)
                    print(f"Long take profit: {long_take_profit}")


                should_short = best_bid_price > ma_3_high
                should_long = best_bid_price < ma_3_high

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
                    

                print(f"Short condition: {should_short}")
                print(f"Long condition: {should_long}")
                print(f"Add short condition: {should_add_to_short}")
                print(f"Add long condition: {should_add_to_long}")

                live.update(self.generate_main_table(
                    symbol,
                    min_qty,
                    current_price,
                    total_equity,
                    #available_equity,
                    one_minute_volume,
                    five_minute_distance,
                    trend,
                    long_pos_qty,
                    short_pos_qty,
                    long_upnl,
                    short_upnl,
                    cum_realised_pnl_long,
                    cum_realised_pnl_short,
                    long_pos_price,
                    short_pos_price,
                    long_dynamic_amount,
                    short_dynamic_amount,
                    long_take_profit,
                    short_take_profit,
                    self.long_pos_leverage,
                    self.short_pos_leverage,
                    self.max_long_trade_qty,
                    self.max_short_trade_qty,
                    self.long_expected_profit_usdt,
                    self.short_expected_profit_usdt,
                    #long_liq_price,
                    #short_liq_price,
                    should_long,
                    should_add_to_long,
                    should_short,
                    should_add_to_short,
                    eri_trend,
                ))

                open_orders = self.exchange.get_open_orders_binance(symbol)

                print(f"Open orders: {open_orders}")

                self.binance_auto_hedge_entry(trend, one_minute_volume, five_minute_distance, min_vol, min_dist, should_long, 
                                            long_pos_qty, long_dynamic_amount, best_bid_price, long_pos_price,
                                            should_add_to_long, self.max_long_trade_qty, should_short, short_pos_qty,
                                                short_dynamic_amount, best_ask_price, short_pos_price, should_add_to_short,
                                                self.max_short_trade_qty, symbol)

                # Long and short take profit placement, order cancellation ID processing
                if long_pos_qty != 0 and long_take_profit is not None:     
                    self.binance_hedge_placetp_maker(symbol, long_pos_qty, long_take_profit, "LONG", open_orders)

                if short_pos_qty != 0 and short_take_profit is not None:
                    self.binance_hedge_placetp_maker(symbol, abs(short_pos_qty), short_take_profit, "SHORT", open_orders)

                # Cancel entries
                current_time = time.time()
                if current_time - self.last_cancel_time >= 60:  # Execute this block every 1 minute
                    try:
                        if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                            self.exchange.cancel_all_entries_binance(symbol)
                            print(f"Canceled entry orders for {symbol}")
                            time.sleep(0.05)
                    except Exception as e:
                        print(f"An error occurred while canceling entry orders: {e}")

                    self.last_cancel_time = current_time  # Update the last cancel time

                time.sleep(30)