import time
import math
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

logging = Logger(filename="bybithedgemfirsitriggerpostfees.log", stream=True)

class BybitHedgeMFIRSITriggerPostOnlyAvoidFees(Strategy):
    def __init__(self, exchange, manager, config):
        super().__init__(exchange, config, manager)
        self.manager = manager
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



    def generate_main_table(self, symbol, min_qty, current_price, balance, available_bal, volume, spread, trend, long_pos_qty, short_pos_qty, long_upnl, short_upnl, long_cum_pnl, short_cum_pnl, long_pos_price, short_pos_price, long_dynamic_amount, short_dynamic_amount, long_take_profit, short_take_profit, long_pos_lev, short_pos_lev, long_max_trade_qty, short_max_trade_qty, long_expected_profit, short_expected_profit, long_liq_price, short_liq_price, should_long, should_add_to_long, should_short, should_add_to_short, mfirsi_signal, eri_trend):
        try:
            table = Table(show_header=False, header_style="bold magenta", title=f"Directional Scalper MFIRSI ONLY {self.version}")
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
                "Available bal.": available_bal,
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
                "Long liq price": long_liq_price,
                "Short liq price": short_liq_price,
                "1m Vol": volume,
                "5m Spread:": spread,
                "Trend": trend,
                "ERI Trend": eri_trend,
                "MFIRSI Signal": mfirsi_signal,
                "Long condition": should_long,
                "Add long cond.": should_add_to_long,
                "Short condition": should_short,
                "Add short cond.": should_add_to_short,
                "Min. volume": self.config.min_volume,
                "Min. spread": self.config.min_distance,
                "Min. qty": min_qty,
            }

            for key, value in table_data.items():
                table.add_row(key, str(value))
            
            return table

        except Exception as e:
            logging.info(f"Exception caught {e}")
            return Table()

    def run(self, symbol):
        console = Console()

        live = Live(console=console, refresh_per_second=2)

        quote_currency = "USDT"
        max_retries = 5
        retry_delay = 5

        # Initialize exchange-related variables outside the live context
        wallet_exposure = self.config.wallet_exposure
        min_dist = self.config.min_distance
        min_vol = self.config.min_volume
        min_dist_btc = self.config.min_distance_btc
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
                one_minute_distance = self.manager.get_asset_value(symbol, data, "1mSpread")
                five_minute_distance = self.manager.get_asset_value(symbol, data, "5mSpread")
                thirty_minute_distance = self.manager.get_asset_value(symbol, data, "30mSpread")
                one_hour_distance = self.manager.get_asset_value(symbol, data, "1hSpread")
                four_hour_distance = self.manager.get_asset_value(symbol, data, "4hSpread")
                trend = self.manager.get_asset_value(symbol, data, "Trend")
                mfirsi_signal = self.manager.get_asset_value(symbol, data, "MFI")
                eri_trend = self.manager.get_asset_value(symbol, data, "ERI Trend")

                self.initialize_MFIRSI(symbol)
                should_long_mfirsi = self.should_long_MFI(symbol)
                should_short_mfirsi = self.should_short_MFI(symbol)

                logging.info(f"Should long MFIRSI: {should_long_mfirsi}")
                logging.info(f"Should short MFRSI: {should_short_mfirsi}")

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
                        
                #print(f"Total equity: {total_equity}")

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

                # print(f"Available equity: {available_equity}")

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

                # Leverage increase logic for long positions
                if long_pos_qty >= self.initial_max_long_trade_qty and self.long_pos_leverage <= 1.0:
                    self.max_long_trade_qty = 2 * self.initial_max_long_trade_qty  # double the maximum long trade quantity
                    self.long_leverage_increased = True
                    self.long_pos_leverage = 2.0
                    logging.info(f"Long leverage temporarily increased to {self.long_pos_leverage}x")
                elif long_pos_qty >= 2 * self.initial_max_long_trade_qty and self.long_pos_leverage <= 2.0:
                    self.max_long_trade_qty = 3 * self.initial_max_long_trade_qty  # triple the maximum long trade quantity
                    self.long_pos_leverage = 3.0
                    logging.info(f"Long leverage temporarily increased to {self.long_pos_leverage}x")
                elif long_pos_qty < (self.max_long_trade_qty / 2) and self.long_pos_leverage > 1.0:
                    self.max_long_trade_qty = self.calc_max_trade_qty(total_equity,
                                                                    best_ask_price,
                                                                    max_leverage)
                    self.long_leverage_increased = False
                    self.long_pos_leverage = 1.0
                    logging.info(f"Long leverage returned to normal {self.long_pos_leverage}x")

                # Leverage increase logic for short positions
                if short_pos_qty >= self.initial_max_short_trade_qty and self.short_pos_leverage <= 1.0:
                    self.max_short_trade_qty = 2 * self.initial_max_short_trade_qty  # double the maximum short trade quantity
                    self.short_leverage_increased = True
                    self.short_pos_leverage = 2.0
                    logging.info(f"Short leverage temporarily increased to {self.short_pos_leverage}x")
                elif short_pos_qty >= 2 * self.initial_max_short_trade_qty and self.short_pos_leverage <= 2.0:
                    self.max_short_trade_qty = 3 * self.initial_max_short_trade_qty  # triple the maximum short trade quantity
                    self.short_pos_leverage = 3.0
                    logging.info(f"Short leverage temporarily increased to {self.short_pos_leverage}x")
                elif short_pos_qty < (self.max_short_trade_qty / 2) and self.short_pos_leverage > 1.0:
                    self.max_short_trade_qty = self.calc_max_trade_qty(total_equity,
                                                                    best_ask_price,
                                                                    max_leverage)
                    self.short_leverage_increased = False
                    self.short_pos_leverage = 1.0
                    logging.info(f"Short leverage returned to normal {self.short_pos_leverage}x")
                    
                # logging.info(f"Long position currently at {self.long_pos_leverage}x leverage")
                # logging.info(f"Short position currently at {self.short_pos_leverage}x leverage")

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

                # if thirty_minute_distance != previous_thirty_minute_distance:
                #     short_take_profit = self.calculate_short_take_profit_spread_bybit(short_pos_price, symbol, thirty_minute_distance)
                #     long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, thirty_minute_distance)
                # else:
                #     if short_take_profit is None or long_take_profit is None:
                #         short_take_profit = self.calculate_short_take_profit_spread_bybit(short_pos_price, symbol, thirty_minute_distance)
                #         long_take_profit = self.calculate_long_take_profit_spread_bybit(long_pos_price, symbol, thirty_minute_distance)
                #self, short_pos_price, quantity, symbol, decrease_percentage=0
                # 
                    
                # if thirty_minute_distance != previous_thirty_minute_distance:
                #     short_take_profit = self.calculate_short_take_profit_spread_bybit_fees(short_pos_price, short_pos_qty, symbol, thirty_minute_distance)
                #     long_take_profit = self.calculate_long_take_profit_spread_bybit_fees(long_pos_price, long_pos_qty, symbol, thirty_minute_distance)
                # else:
                #     if short_take_profit is None or long_take_profit is None:
                #         short_take_profit = self.calculate_short_take_profit_spread_bybit_fees(short_pos_price, short_pos_qty, symbol, thirty_minute_distance)
                #         long_take_profit = self.calculate_long_take_profit_spread_bybit_fees(long_pos_price, long_pos_qty, symbol, thirty_minute_distance)
                        
                #previous_thirty_minute_distance = thirty_minute_distance

                should_short = self.short_trade_condition(best_bid_price, ma_3_high)
                should_long = self.long_trade_condition(best_ask_price, ma_3_low)

                # should_short = self.short_trade_condition(best_ask_price, ma_3_high)
                # should_long = self.long_trade_condition(best_bid_price, ma_3_low)

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

                live.update(self.generate_main_table(
                    symbol,
                    min_qty,
                    current_price,
                    total_equity,
                    available_equity,
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
                    long_liq_price,
                    short_liq_price,
                    should_long,
                    should_add_to_long,
                    should_short,
                    should_add_to_short,
                    mfirsi_signal,
                    eri_trend,
                ))

                if one_minute_volume is not None and five_minute_distance is not None:
                    if one_minute_volume > min_vol and five_minute_distance > min_dist_btc:

                        mfi = self.manager.get_asset_value(symbol, data, "MFI")

                        if mfi is not None and isinstance(mfi, str):
                            if mfi.lower() == "long" and long_pos_qty == 0:
                                logging.info(f"Placing initial long entry with post-only order")
                                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1)
                                logging.info(f"Placed initial long entry with post-only order")
                            elif mfi.lower() == "long" and long_pos_qty < self.max_long_trade_qty and best_bid_price < long_pos_price:
                                logging.info(f"Placing additional long entry with post-only order")
                                self.postonly_limit_order_bybit(symbol, "buy", long_dynamic_amount, best_bid_price, positionIdx=1)
                            elif mfi.lower() == "short" and short_pos_qty == 0:
                                logging.info(f"Placing initial short entry with post-only order")
                                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2)
                                logging.info(f"Placed initial short entry with post-only order")
                            elif mfi.lower() == "short" and short_pos_qty < self.max_short_trade_qty and best_ask_price > short_pos_price:
                                logging.info(f"Placing additional short entry with post-only order")
                                self.postonly_limit_order_bybit(symbol, "sell", short_dynamic_amount, best_ask_price, positionIdx=2)

                open_orders = self.exchange.get_open_orders(symbol)

                if long_pos_qty > 0 and long_take_profit is not None:
                    existing_long_tps = self.get_open_take_profit_order_quantities(open_orders, "sell")
                    total_existing_long_tp_qty = sum(qty for qty, _ in existing_long_tps)
                    logging.info(f"Existing long TPs: {existing_long_tps}")
                    if not math.isclose(total_existing_long_tp_qty, long_pos_qty):
                        try:
                            for qty, existing_long_tp_id in existing_long_tps:
                                if not math.isclose(qty, long_pos_qty):
                                    self.exchange.cancel_order_by_id(existing_long_tp_id, symbol)
                                    logging.info(f"Long take profit {existing_long_tp_id} canceled")
                                    time.sleep(0.05)
                        except Exception as e:
                            logging.info(f"Error in cancelling long TP orders {e}")

                    if len(existing_long_tps) < 1:
                        try:
                            self.exchange.postonly_create_take_profit_order_bybit(symbol, "limit", "sell", long_pos_qty, long_take_profit, positionIdx=1, reduce_only=True)
                            logging.info(f"Long take profit set at {long_take_profit}")
                            time.sleep(0.05)
                        except Exception as e:
                            logging.info(f"Error in placing long TP: {e}")

                if short_pos_qty > 0 and short_take_profit is not None:
                    existing_short_tps = self.get_open_take_profit_order_quantities(open_orders, "buy")
                    total_existing_short_tp_qty = sum(qty for qty, _ in existing_short_tps)
                    logging.info(f"Existing short TPs: {existing_short_tps}")
                    if not math.isclose(total_existing_short_tp_qty, short_pos_qty):
                        try:
                            for qty, existing_short_tp_id in existing_short_tps:
                                if not math.isclose(qty, short_pos_qty):
                                    self.exchange.cancel_order_by_id(existing_short_tp_id, symbol)
                                    logging.info(f"Short take profit {existing_short_tp_id} canceled")
                                    time.sleep(0.05)
                        except Exception as e:
                            logging.info(f"Error in cancelling short TP orders: {e}")

                    if len(existing_short_tps) < 1:
                        try:
                            self.exchange.postonly_create_take_profit_order_bybit(symbol, "limit", "buy", short_pos_qty, short_take_profit, positionIdx=2, reduce_only=True)
                            logging.info(f"Short take profit set at {short_take_profit}")
                            time.sleep(0.05)
                        except Exception as e:
                            logging.info(f"Error in placing short TP: {e}")

                if long_pos_qty > 0 and long_take_profit is not None:
                    existing_long_tps = self.get_open_take_profit_order_quantities(open_orders, "sell")
                    total_existing_long_tp_qty = sum(qty for qty, _ in existing_long_tps)
                    logging.info(f"Existing long TPs: {existing_long_tps}")
                    now = datetime.now()
                    if now >= self.next_long_tp_update or not math.isclose(total_existing_long_tp_qty, long_pos_qty):
                        try:
                            for qty, existing_long_tp_id in existing_long_tps:
                                self.exchange.cancel_order_by_id(existing_long_tp_id, symbol)
                                logging.info(f"Long take profit {existing_long_tp_id} canceled")
                                time.sleep(0.05)
                            self.exchange.postonly_create_take_profit_order_bybit(symbol, "limit", "sell", long_pos_qty, long_take_profit, positionIdx=1, reduce_only=True)
                            logging.info(f"Long take profit set at {long_take_profit}")
                            self.next_long_tp_update = self.calculate_next_update_time()  # Calculate the next update time after placing the order
                        except Exception as e:
                            logging.info(f"Error in updating long TP: {e}")

                if short_pos_qty > 0 and short_take_profit is not None:
                    existing_short_tps = self.get_open_take_profit_order_quantities(open_orders, "buy")
                    total_existing_short_tp_qty = sum(qty for qty, _ in existing_short_tps)
                    logging.info(f"Existing short TPs: {existing_short_tps}")
                    now = datetime.now()
                    if now >= self.next_short_tp_update or not math.isclose(total_existing_short_tp_qty, short_pos_qty):
                        try:
                            for qty, existing_short_tp_id in existing_short_tps:
                                self.exchange.cancel_order_by_id(existing_short_tp_id, symbol)
                                logging.info(f"Short take profit {existing_short_tp_id} canceled")
                                time.sleep(0.05)
                            self.exchange.postonly_create_take_profit_order_bybit(symbol, "limit", "buy", short_pos_qty, short_take_profit, positionIdx=2, reduce_only=True)
                            logging.info(f"Short take profit set at {short_take_profit}")
                            self.next_short_tp_update = self.calculate_next_update_time()  # Calculate the next update time after placing the order
                        except Exception as e:
                            logging.info(f"Error in updating short TP: {e}")

                # Cancel entries
                current_time = time.time()
                if current_time - self.last_cancel_time >= 60:  # Execute this block every 1 minute
                    try:
                        if best_ask_price < ma_1m_3_high or best_ask_price < ma_5m_3_high:
                            self.exchange.cancel_all_entries_bybit(symbol)
                            logging.info(f"Canceled entry orders for {symbol}")
                            time.sleep(0.05)
                    except Exception as e:
                        logging.info(f"An error occurred while canceling entry orders: {e}")

                    self.last_cancel_time = current_time  # Update the last cancel time

                time.sleep(30)
