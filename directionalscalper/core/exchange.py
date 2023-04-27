import logging
from typing import Optional
import ccxt
import pandas as pd

log = logging.getLogger(__name__)

class Exchange:
    def __init__(self, exchange_id, api_key, secret_key, passphrase=None):
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.name = exchange_id
        self.initialise()
        self.symbols = self._get_symbols()

    def initialise(self):
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange_params = {
            "apiKey": self.api_key,
            "secret": self.secret_key,
        }
        if self.passphrase:
            exchange_params["password"] = self.passphrase

        self.exchange = exchange_class(exchange_params)

    def _get_symbols(self):
        markets = self.exchange.load_markets()
        symbols = [market['symbol'] for market in markets.values()]
        return symbols

    def setup_exchange(self, symbol) -> None:
        values = {"position": False, "margin": False, "leverage": False}
        try:
            self.exchange.set_position_mode(hedged="BothSide", symbol=symbol)
            values["position"] = True
        except Exception as e:
            log.warning(f"An unknown error occurred in with set_position_mode: {e}")
        try:
            self.exchange.set_margin_mode(marginMode="cross", symbol=symbol)
            values["margin"] = True
        except Exception as e:
            log.warning(f"An unknown error occurred in with set_margin_mode: {e}")
        market_data = self.get_market_data(symbol=symbol)
        try:
            self.exchange.set_leverage(leverage=market_data["leverage"], symbol=symbol)
            values["leverage"] = True
        except Exception as e:
            log.warning(f"An unknown error occurred in with set_leverage: {e}")
        log.info(values)

    def get_market_data_bitget(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)

            if self.exchange.id == 'bybit':
                if "info" in symbol_data:
                    values["precision"] = symbol_data["info"]["price_scale"]
                    values["leverage"] = symbol_data["info"]["leverage_filter"][
                        "max_leverage"
                    ]
                    values["min_qty"] = symbol_data["info"]["lot_size_filter"][
                        "min_trading_qty"
                    ]
            elif self.exchange.id == 'bitget':
                if "precision" in symbol_data:
                    values["precision"] = symbol_data["precision"]["price"]
                if "limits" in symbol_data:
                    values["min_qty"] = symbol_data["limits"]["amount"]["min"]
            else:
                log.warning("Exchange not recognized for fetching market data.")

        except Exception as e:
            log.warning(f"An unknown error occurred in get_market_data(): {e}")
        return values


    def get_market_data(self, symbol: str) -> dict:
        values = {"precision": 0.0, "leverage": 0.0, "min_qty": 0.0}
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)
            if "info" in symbol_data:
                values["precision"] = symbol_data["info"]["price_scale"]
                values["leverage"] = symbol_data["info"]["leverage_filter"][
                    "max_leverage"
                ]
                values["min_qty"] = symbol_data["info"]["lot_size_filter"][
                    "min_trading_qty"
                ]

        except Exception as e:
            log.warning(f"An unknown error occurred in get_market_data(): {e}")
        return values

    def get_balance_bitget(self, quote, account_type='futures'):
        if account_type == 'futures':
            if self.exchange.has['fetchBalance']:
                # Fetch the balance
                balance = self.exchange.fetch_balance(params={'type': 'swap'})

                for currency_balance in balance['info']:
                    if currency_balance['marginCoin'] == quote:
                        return float(currency_balance['equity'])
        else:
            # Handle other account types or fallback to default behavior
            pass

    def get_price_precision(self, symbol):
        market = self.exchange.market(symbol)
        smallest_increment = market['precision']['price']
        price_precision = len(str(smallest_increment).split('.')[-1])
        return price_precision

    def get_balance(self, quote: str) -> dict:
        values = {
            "available_balance": 0.0,
            "pnl": 0.0,
            "upnl": 0.0,
            "wallet_balance": 0.0,
            "equity": 0.0,
        }
        try:
            data = self.exchange.fetch_balance()
            if "info" in data:
                if "result" in data["info"]:
                    if quote in data["info"]["result"]:
                        values["available_balance"] = float(
                            data["info"]["result"][quote]["available_balance"]
                        )
                        values["pnl"] = float(
                            data["info"]["result"][quote]["realised_pnl"]
                        )
                        values["upnl"] = float(
                            data["info"]["result"][quote]["unrealised_pnl"]
                        )
                        values["wallet_balance"] = round(
                            float(data["info"]["result"][quote]["wallet_balance"]), 2
                        )
                        values["equity"] = round(
                            float(data["info"]["result"][quote]["equity"]), 2
                        )
        except Exception as e:
            log.warning(f"An unknown error occurred in get_balance(): {e}")
        return values

    def get_orderbook(self, symbol) -> dict:
        values = {"bids": [], "asks": []}
        try:
            data = self.exchange.fetch_order_book(symbol)
            if "bids" in data and "asks" in data:
                if len(data["bids"]) > 0 and len(data["asks"]) > 0:
                    if len(data["bids"][0]) > 0 and len(data["asks"][0]) > 0:
                        values["bids"] = data["bids"]
                        values["asks"] = data["asks"]
        except Exception as e:
            log.warning(f"An unknown error occurred in get_orderbook(): {e}")
        return values

    
    # def get_orderbook(self, symbol) -> dict:
    #     values = {"bids": [], "asks": []}
    #     try:
    #         data = self.exchange.fetch_order_book(symbol)
    #         if "bids" in data and "asks" in data:
    #             values["bids"] = data["bids"]
    #             values["asks"] = data["asks"]
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in get_orderbook(): {e}")
    #     return values


    # def get_orderbook(self, symbol) -> dict:
    #     values = {"bids": 0.0, "asks": 0.0}
    #     try:
    #         data = self.exchange.fetch_order_book(symbol)
    #         if "bids" in data and "asks" in data:
    #             if len(data["bids"]) > 0 and len(data["asks"]) > 0:
    #                 if len(data["bids"][0]) > 0 and len(data["asks"][0]) > 0:
    #                     values["bids"] = float(data["bids"][0][0])
    #                     values["asks"] = float(data["asks"][0][0])
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in get_orderbook(): {e}")
    #     return values

    def get_positions_bitget_debug(self, symbol) -> dict:
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
            data = self.exchange.fetch_positions([symbol])
            if len(data) == 2:
                for position in data:
                    side = position["side"]
                    print(f"Full position data for {side} position: {position}")  # Add this line

                    print("contractSize:", position["contractSize"])
                    values[side]["qty"] = float(position["contractSize"])

                    print("entryPrice:", position["entryPrice"])
                    values[side]["price"] = float(position["entryPrice"])

                    print("achievedProfits:", position["info"]["achievedProfits"])
                    values[side]["realised"] = round(float(position["info"]["achievedProfits"]), 4)

                    print("unrealizedPnl:", position["unrealizedPnl"])
                    values[side]["upnl"] = round(float(position["unrealizedPnl"]), 4)

                    print("liquidationPrice:", position["liquidationPrice"])
                    values[side]["liq_price"] = float(position["liquidationPrice"])

                    print("entryPrice:", position["entryPrice"])
                    values[side]["entry_price"] = float(position["entryPrice"])

                print("Raw Bitget positions response:", data)
                print("Parsed Bitget positions data:", values)
        except Exception as e:
            log.warning(f"An unknown error occurred in get_positions_bitget_debug(): {e}")
        return values

    def get_positions_bitget(self, symbol) -> dict:
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
            data = self.exchange.fetch_positions([symbol])
            if len(data) == 2:
                for position in data:
                    side = position["side"]
                    values[side]["qty"] = float(position["contracts"])  # Use "contracts" instead of "contractSize"
                    values[side]["price"] = float(position["entryPrice"])
                    values[side]["realised"] = round(float(position["info"]["achievedProfits"]), 4)
                    values[side]["upnl"] = round(float(position["unrealizedPnl"]), 4)
                    if position["liquidationPrice"] is not None:
                        values[side]["liq_price"] = float(position["liquidationPrice"])
                    else:
                        print(f"Warning: liquidationPrice is None for {side} position")
                        values[side]["liq_price"] = None
                    values[side]["entry_price"] = float(position["entryPrice"])
        except Exception as e:
            log.warning(f"An unknown error occurred in get_positions_bitget(): {e}")
        return values


    # def get_positions_bitget(self, symbol) -> dict:
    #     values = {
    #         "long": {
    #             "qty": 0.0,
    #             "price": 0.0,
    #             "realised": 0,
    #             "cum_realised": 0,
    #             "upnl": 0,
    #             "upnl_pct": 0,
    #             "liq_price": 0,
    #             "entry_price": 0,
    #         },
    #         "short": {
    #             "qty": 0.0,
    #             "price": 0.0,
    #             "realised": 0,
    #             "cum_realised": 0,
    #             "upnl": 0,
    #             "upnl_pct": 0,
    #             "liq_price": 0,
    #             "entry_price": 0,
    #         },
    #     }
    #     try:
    #         data = self.exchange.fetch_positions([symbol])
    #         if len(data) == 2:
    #             for position in data:
    #                 side = position["side"]
    #                 values[side]["qty"] = float(position["contractSize"])
    #                 values[side]["price"] = float(position["entryPrice"])
    #                 values[side]["realised"] = round(float(position["info"]["achievedProfits"]), 4)
    #                 values[side]["upnl"] = round(float(position["unrealizedPnl"]), 4)
    #                 if position["liquidationPrice"] is not None:
    #                     values[side]["liq_price"] = float(position["liquidationPrice"])
    #                 else:
    #                     print(f"Warning: liquidationPrice is None for {side} position")
    #                     values[side]["liq_price"] = None
    #                 values[side]["entry_price"] = float(position["entryPrice"])
    #             # print("Raw Bitget positions response:", data)
    #             # print("Parsed Bitget positions data:", values)
    #                 # You can add any other necessary values here as well.
    #                 data = self.exchange.fetch_positions([symbol])
    #                 print(f"Raw fetch_positions response: {data}")
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in get_positions_bitget(): {e}")
    #     return values

    def get_positions(self, symbol) -> dict:
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
            data = self.exchange.fetch_positions([symbol])
            if len(data) == 2:
                sides = ["long", "short"]
                for side in [0, 1]:
                    values[sides[side]]["qty"] = float(data[side]["contracts"])
                    values[sides[side]]["price"] = float(data[side]["entryPrice"])
                    values[sides[side]]["realised"] = round(
                        float(data[side]["info"]["realised_pnl"]), 4
                    )
                    values[sides[side]]["cum_realised"] = round(
                        float(data[side]["info"]["cum_realised_pnl"]), 4
                    )
                    if data[side]["info"]["unrealised_pnl"] is not None:
                        values[sides[side]]["upnl"] = round(
                            float(data[side]["info"]["unrealised_pnl"]), 4
                        )
                    if data[side]["precentage"] is not None:
                        values[sides[side]]["upnl_pct"] = round(
                            float(data[side]["precentage"]), 4
                        )
                    if data[side]["liquidationPrice"] is not None:
                        values[sides[side]]["liq_price"] = float(
                            data[side]["liquidationPrice"]
                        )
                    if data[side]["entryPrice"] is not None:
                        values[sides[side]]["entry_price"] = float(
                            data[side]["entryPrice"]
                        )
        except Exception as e:
            log.warning(f"An unknown error occurred in get_positions(): {e}")
        return values

    def get_current_price(self, symbol: str) -> float:
        current_price = 0.0
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            if "bid" in ticker and "ask" in ticker:
                current_price = (ticker["bid"] + ticker["ask"]) / 2
        except Exception as e:
            log.warning(f"An unknown error occurred in get_positions(): {e}")
        return current_price

    def get_moving_averages(
        self, symbol: str, timeframe: str = "1m", num_bars: int = 20
    ) -> dict:
        values = {"MA_3_H": 0.0, "MA_3_L": 0.0, "MA_6_H": 0.0, "MA_6_L": 0.0}
        try:
            bars = self.exchange.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, limit=num_bars
            )
            df = pd.DataFrame(
                bars, columns=["Time", "Open", "High", "Low", "Close", "Volume"]
            )
            df["Time"] = pd.to_datetime(df["Time"], unit="ms")
            df["MA_3_High"] = df.High.rolling(3).mean()
            df["MA_3_Low"] = df.Low.rolling(3).mean()
            df["MA_6_High"] = df.High.rolling(6).mean()
            df["MA_6_Low"] = df.Low.rolling(6).mean()
            values["MA_3_H"] = df["MA_3_High"].iat[-1]
            values["MA_3_L"] = df["MA_3_Low"].iat[-1]
            values["MA_6_H"] = df["MA_6_High"].iat[-1]
            values["MA_6_L"] = df["MA_6_Low"].iat[-1]
        except Exception as e:
            log.warning(f"An unknown error occurred in get_moving_averages(): {e}")
        return values

    def get_open_orders(self, symbol: str) -> dict:
        values = {"id": "", "price": 0.0, "qty": 0.0}
        try:
            order = self.exchange.fetch_open_orders(symbol)
            if len(order) > 0:
                if "info" in order[0]:
                    order_status = order[0]["info"]["order_status"]
                    order_side = order[0]["info"]["side"]
                    reduce_only = order[0]["info"]["reduce_only"]
                    if order_status == "New" and order_side == "Buy" and reduce_only:
                        values["id"] = order[0]["info"]["order_id"]
                        values["price"] = order[0]["info"]["price"]
                        values["qty"] = order[0]["info"]["qty"]
        except Exception as e:
            log.warning(f"An unknown error occurred in get_open_orders(): {e}")
        return values

    # def cancel_entry(self, symbol: str) -> None:
    #     try:
    #         orders = self.exchange.fetch_open_orders(symbol)
    #         for order in orders:
    #             if "info" in order:
    #                 order_id = order["info"]["orderId"]
    #                 order_status = order["info"]["status"]
    #                 order_side = order["info"]["side"]
    #                 reduce_only = order["info"]["reduceOnly"]
    #                 if (
    #                     order_status != "filled"
    #                     and order_side == "buy"
    #                     and order_status != "cancelled"
    #                     and not reduce_only
    #                 ):
    #                     self.exchange.cancel_order(symbol=symbol, id=order_id)
    #                     log.info(f"Cancelling order: {order_id}")
    #                 elif (
    #                     order_status != "filled"
    #                     and order_side == "sell"
    #                     and order_status != "cancelled"
    #                     and not reduce_only
    #                 ):
    #                     self.exchange.cancel_order(symbol=symbol, id=order_id)
    #                     log.info(f"Cancelling order: {order_id}")
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in cancel_entry(): {e}")

    def debug_open_orders(self, symbol: str) -> None:
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            print(open_orders)
        except:
            print(f"Fuck")

    def cancel_long_entry(self, symbol: str) -> None:
        self._cancel_entry(symbol, order_side="Buy")

    def cancel_short_entry(self, symbol: str) -> None:
        self._cancel_entry(symbol, order_side="Sell")

    def _cancel_entry(self, symbol: str, order_side: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            for order in orders:
                if "info" in order:
                    order_id = order["info"]["order_id"]
                    order_status = order["info"]["order_status"]
                    side = order["info"]["side"]
                    reduce_only = order["info"]["reduce_only"]
                    if (
                        order_status != "Filled"
                        and side == order_side
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        log.info(f"Cancelling {order_side} order: {order_id}")
        except Exception as e:
            log.warning(f"An unknown error occurred in _cancel_entry(): {e}")

    def cancel_all_entries(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            long_orders = 0
            short_orders = 0

            # Count the number of open long and short orders
            for order in orders:
                order_info = order["info"]
                order_status = order_info["state"]
                order_side = order_info["side"]
                reduce_only = order_info["reduceOnly"]
                
                if order_status != "Filled" and order_status != "Cancelled" and not reduce_only:
                    if order_side == "open_long":
                        long_orders += 1
                    elif order_side == "open_short":
                        short_orders += 1

            # Cancel extra long or short orders if more than one open order per side
            if long_orders > 1 or short_orders > 1:
                for order in orders:
                    order_info = order["info"]
                    order_id = order_info["orderId"]
                    order_status = order_info["state"]
                    order_side = order_info["side"]
                    reduce_only = order_info["reduceOnly"]

                    if (
                        order_status != "Filled"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        print("Cancelling order: {order_id}")
                        # log.info(f"Cancelling order: {order_id}")
        except Exception as e:
            log.warning(f"An unknown error occurred in cancel_entry(): {e}")


    def cancel_entry_bitget(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            
            for order in orders:
                order_info = order["info"]
                order_id = order_info["orderId"]
                order_status = order_info["state"]
                order_side = order_info["side"]
                reduce_only = order_info["reduceOnly"]
                
                if (
                    order_status != "Filled"
                    and order_status != "Cancelled"
                    and not reduce_only
                ):
                    self.exchange.cancel_order(symbol=symbol, id=order_id)
                    log.info(f"Cancelling order: {order_id}")
        except Exception as e:
            log.warning(f"An unknown error occurred in cancel_entry(): {e}")


    def cancel_entry(self, symbol: str) -> None:
        try:
            order = self.exchange.fetch_open_orders(symbol)
            #print(f"Open orders for {symbol}: {order}")
            if len(order) > 0:
                if "info" in order[0]:
                    order_id = order[0]["info"]["order_id"]
                    order_status = order[0]["info"]["order_status"]
                    order_side = order[0]["info"]["side"]
                    reduce_only = order[0]["info"]["reduce_only"]
                    if (
                        order_status != "Filled"
                        and order_side == "Buy"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        log.info(f"Cancelling order: {order_id}")
                    elif (
                        order_status != "Filled"
                        and order_side == "Sell"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        log.info(f"Cancelling order: {order_id}")
        except Exception as e:
            log.warning(f"An unknown error occurred in cancel_entry(): {e}")

    def cancel_close_bitget(self, symbol: str, side: str) -> None:
        side_map = {"long": "open_long", "short": "open_short"}
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        order_id = order["info"]["orderId"]
                        order_status = order["info"]["state"]
                        order_side = order["info"]["side"]
                        reduce_only = order["info"]["reduceOnly"]

                        if (
                            order_status != "filled"
                            and order_side == side_map[side]
                            and order_status != "canceled"
                            and not reduce_only
                        ):
                            self.exchange.cancel_order(symbol=symbol, id=order_id)
                            log.info(f"Cancelling order: {order_id}")
        except Exception as e:
            log.warning(f"An unknown error occurred in cancel_close_bitget(): {e}")

    # def cancel_close_bitget(self, symbol: str) -> None:
    #     try:
    #         orders = self.exchange.fetch_open_orders(symbol)
    #         if len(orders) > 0:
    #             for order in orders:
    #                 if "info" in order:
    #                     order_id = order["info"]["orderId"]
    #                     order_status = order["info"]["state"]
    #                     order_side = order["info"]["side"]
    #                     reduce_only = order["info"]["reduceOnly"]

    #                     if (
    #                         order_status != "filled"
    #                         and order_side == "open_short"
    #                         and order_status != "canceled"
    #                         and not reduce_only
    #                     ):
    #                         self.exchange.cancel_order(symbol=symbol, id=order_id)
    #                         log.info(f"Cancelling order: {order_id}")
    #                     elif (
    #                         order_status != "filled"
    #                         and order_side == "open_long"
    #                         and order_status != "canceled"
    #                         and not reduce_only
    #                     ):
    #                         self.exchange.cancel_order(symbol=symbol, id=order_id)
    #                         log.info(f"Cancelling order: {order_id}")
    #     except Exception as e:
    #         log.warning(f"An unknown error occurred in cancel_close_bitget(): {e}")

    def cancel_close(self, symbol: str, side: str) -> None:
        try:
            order = self.exchange.fetch_open_orders(symbol)
            if len(order) > 0:
                if "info" in order[0]:
                    order_id = order[0]["info"]["order_id"]
                    order_status = order[0]["info"]["order_status"]
                    order_side = order[0]["info"]["side"]
                    reduce_only = order[0]["info"]["reduce_only"]
                    if (
                        order_status != "Filled"
                        and order_side == "Buy"
                        and side == "long"
                        and order_status != "Cancelled"
                        and reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        log.info(f"Cancelling order: {order_id}")
                    elif (
                        order_status != "Filled"
                        and order_side == "Sell"
                        and side == "short"
                        and order_status != "Cancelled"
                        and reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        log.info(f"Cancelling order: {order_id}")
        except Exception as e:
            log.warning(f"{e}")

    def fetch_open_orders(self, symbol: Optional[str] = None, since: Optional[int] = None, limit: Optional[int] = None, params={}):
        if symbol is None:
            raise ValueError("Symbol is required for fetch_open_orders.")
            
        self.exchange.load_markets()
        market = self.market(symbol)
        marketType = None
        query = None
        marketType, query = self.handle_market_type_and_params('fetchOpenOrders', market, params)
        request = {
            'symbol': market['id'],
        }
        method = self.get_supported_mapping(marketType, {
            'spot': 'privateSpotPostTradeOpenOrders',
            'swap': 'privateMixGetOrderCurrent',
            'future': 'privateMixGetOrderCurrent',
        })
        stop = self.safe_value(query, 'stop')
        if stop:
            if marketType == 'spot':
                method = 'privateSpotPostPlanCurrentPlan'
                if limit is not None:
                    request['pageSize'] = limit
            else:
                method = 'privateMixGetPlanCurrentPlan'
            query = self.omit(query, 'stop')
        response = getattr(self, method)(self.extend(request, query))
        data = self.safe_value(response, 'data', [])
        if not isinstance(data, list):
            data = self.safe_value(data, 'orderList', [])
        return self.parse_orders(data, market, since, limit)
    
    def create_take_profit_order(self, symbol, order_type, side, amount, price=None, reduce_only=False):
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")
            if side == "buy":
                side = "close_short" if reduce_only else "open_long"
            elif side == "sell":
                side = "close_long" if reduce_only else "open_short"
            else:
                raise ValueError(f"Invalid side: {side}")

            params = {"reduceOnly": reduce_only}
            return self.exchange.create_order(symbol, order_type, side, amount, price, params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")    

    # def create_take_profit_order_bitget(self, symbol, type, side, amount, price, reduce_only=False):
    #     params = {
    #         'takeProfitPrice': price,
    #         'reduceOnly': reduce_only
    #     }
    #     return self.create_order(symbol, type, side, amount, None, params)

    # def create_take_profit_order(self, symbol, type, side, amount, price, reduce_only=False):
    #     params = {
    #         'takeProfitPrice': price,
    #         'reduceOnly': reduce_only
    #     }
    #     return self.create_order(symbol, type, side, amount, None, params)

    def create_market_order(self, symbol: str, side: str, amount: float, params={}, close_position: bool = False) -> None:
        try:
            if side not in ["buy", "sell"]:
                log.warning(f"side {side} does not exist")
                return

            order_type = "market"

            # Determine the correct order side for closing positions
            if close_position:
                market = self.exchange.market(symbol)
                if market['type'] in ['swap', 'future']:
                    if side == "buy":
                        side = "close_short"
                    elif side == "sell":
                        side = "close_long"

            response = self.exchange.create_order(symbol, order_type, side, amount, params=params)
            return response
        except Exception as e:
            log.warning(f"An unknown error occurred in create_market_order(): {e}")

    def create_limit_order(self, symbol, side, amount, price, reduce_only=False, **params):
        if side == "buy":
            return self.create_limit_buy_order(symbol, amount, price, reduce_only=reduce_only, **params)
        elif side == "sell":
            return self.create_limit_sell_order(symbol, amount, price, reduce_only=reduce_only, **params)
        else:
            raise ValueError(f"Invalid side: {side}")


    def create_limit_buy_order(self, symbol: str, qty: float, price: float, **params) -> None:
        self.exchange.create_order(
            symbol=symbol,
            type='limit',
            side='buy',
            amount=qty,
            price=price,
            **params
        )

    def create_limit_sell_order(self, symbol: str, qty: float, price: float, **params) -> None:
        self.exchange.create_order(
            symbol=symbol,
            type='limit',
            side='sell',
            amount=qty,
            price=price,
            **params
        )

    def create_order(self, symbol, order_type, side, amount, price=None, reduce_only=False, **params):
        if reduce_only:
            params.update({'reduceOnly': 'true'})

        if order_type == 'limit':
            if side == "buy":
                return self.create_limit_buy_order(symbol, amount, price, **params)
            elif side == "sell":
                return self.create_limit_sell_order(symbol, amount, price, **params)
            else:
                raise ValueError(f"Invalid side: {side}")
        elif order_type == 'market':
            #... handle market orders if necessary
            return self.create_market_order(symbol, side, amount, params)
        else:
            raise ValueError("Invalid order type. Use 'limit' or 'market'.")
        