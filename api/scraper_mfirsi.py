from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from datetime import datetime

import pandas as pd
import pidfile
import ta


sys.path.append(".")
from directionalscalper.api.exchanges.binance import Binance
from directionalscalper.api.exchanges.bybit import Bybit
from directionalscalper.core.logger import Logger

log = Logger(filename="scraper.log", stream=True)

class Scraper:
    def __init__(self, exchange, filters: dict):
        log.info("Scraper initalising")
        self.exchange = exchange
        self.filters = filters
        self.symbols = self.exchange.get_futures_symbols()
        self.prices = self.exchange.get_futures_prices()
        log.info(f"{len(self.symbols)} symbols found")
        if "quote_symbols" in self.filters:
            self.symbols = self.filter_quote(
                symbols=self.symbols, quotes=self.filters["quote_symbols"]
            )
        if "top_volume" in self.filters:
            self.volumes = self.exchange.get_futures_volumes()
            self.symbols = self.filter_volume(
                symbols=self.symbols,
                volumes=self.volumes,
                limit=self.filters["top_volume"],
            )

    def filter_quote(self, symbols, quotes):
        log.info(f"Filtering on {len(quotes)} quote symbols")
        filtered = []
        for symbol in symbols:
            if symbol.endswith(tuple(quotes)):
                filtered.append(symbol)
        log.info(f"Filtered to {len(filtered)} symbols")
        return filtered

    def filter_volume(self, symbols, volumes, limit):
        log.info(f"Filtering top {limit} symbols by 24h volume")
        volumes = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
        volumes = volumes[:limit]
        volumes = dict(volumes)
        volume_keys = [*volumes]
        filtered = []
        for symbol in symbols:
            if symbol in volume_keys:
                filtered.append(symbol)
        log.info(f"Filtered to {len(filtered)} symbols")
        return filtered

    def get_spread(
        self, symbol: str, limit: int, timeframe: str = "1m", data: list | None = None
    ):
        if data is None:
            data = self.exchange.get_futures_kline(
                symbol=symbol, interval=timeframe, limit=limit
            )
        spread = 0.0
        lowest_low = 999999
        highest_high = 0
        for d in data:
            if d["high"] > highest_high:
                highest_high = d["high"]
            if d["low"] < lowest_low:
                lowest_low = d["low"]
        if highest_high > 0:
            spread = round((highest_high - lowest_low) / highest_high * 100, 4)
        return spread

    def spread_calc(self, data):
        data["high-low"] = abs(data["high"] - abs(data["low"]))
        spread = data[["high-low"]].max(axis=1)
        return spread

    def volume_calc(self, data):
        return data[["volume"]].max(axis=1)

    def get_candle_info(self, symbol: str, timeframe: str, limit: int):
        bars = self.exchange.get_futures_kline(
            symbol=symbol, interval=timeframe, limit=limit
        )
        df = pd.DataFrame(
            bars[:-1], columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["time"] / 1000, unit="s")

        df["ema_6"] = df["close"].ewm(span=6).mean()
        df["ema_6_high"] = df["high"].ewm(span=6).mean()
        df["ema_6_low"] = df["low"].ewm(span=6).mean()

        df["spread"] = (df["ema_6_high"] - df["ema_6_low"]) / df["ema_6_low"] * 100

        return df

    def get_candle_data(self, symbol: str, interval: str, limit: int):
        bars = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["MA_3_High"] = df.high.rolling(3).mean()
        df["MA_3_Low"] = df.low.rolling(3).mean()
        df["MA_6_High"] = df.high.rolling(6).mean()
        df["MA_6_Low"] = df.low.rolling(6).mean()

        return {
            "high_3": df["MA_3_High"].iat[-1],
            "low_3": df["MA_3_Low"].iat[-1],
            "high_6": df["MA_6_High"].iat[-1],
            "low_6": df["MA_6_Low"].iat[-1],
        }

    # def get_candle_data(self, symbol: str, interval: str, limit: int):
    #     bars = self.exchange.get_futures_kline(
    #         symbol=symbol, interval=interval, limit=limit
    #     )
    #     df = pd.DataFrame(
    #         bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
    #     )
    #     df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    #     df["MA_3_High"] = df.high.rolling(3).mean()
    #     df["MA_3_Low"] = df.low.rolling(3).mean()
    #     df["MA_6_High"] = df.high.rolling(6).mean()
    #     df["MA_6_Low"] = df.low.rolling(6).mean()

    #     # Log the DataFrame
    #     log.info(f"DataFrame for {symbol} with {limit} {interval} candles:\n{df}")

    #     ma_values = {
    #         "high_3": df["MA_3_High"].iat[-1],
    #         "low_3": df["MA_3_Low"].iat[-1],
    #         "high_6": df["MA_6_High"].iat[-1],
    #         "low_6": df["MA_6_Low"].iat[-1],
    #     }

    #     # Log the calculated moving averages
    #     log.info(f"Moving averages for {symbol} with {limit} {interval} candles:\n{ma_values}")

    #     return ma_values

    def get_ema(self, symbol: str, interval: str, limit: int, column: str, window: int):
        bars = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )  # 1m, 18, 6
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df[f"EMA{window} {column}"] = ta.trend.EMAIndicator(
            df[column], window=6
        ).ema_indicator()
        return round(
            (df[f"EMA{window} {column}"][limit - 1]).astype(float),
            self.symbols["price_scale"],
        )

    def get_sma(self, symbol: str, interval: str, limit: int, column: str, window: int):
        bars = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        sma = ta.trend.SMAIndicator(df[column], window=window).sma_indicator()

        current_sma = float(sma[limit - 1])

        last_close_price = bars[limit - 1]["close"]

        return round((last_close_price - current_sma) / last_close_price * 100, 4)

    def get_average_true_range(self, symbol: str, period, interval: str, limit: int):
        data = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )
        data["tr"] = self.get_true_range(data=data)
        atr = data["tr"].rolling(period).mean()
        return atr

    def get_true_range(self, data):
        data["previous_close"] = data["close"].shift(1)
        data["high-low"] = abs(data["high"] - data["low"])
        data["high-pc"] = abs(data["high"] - data["previous_close"])
        data["low-pc"] = abs(data["low"] - data["previous_close"])
        tr = data[["high-low", "high-pc", "low-pc"]].max(axis=1)
        return tr

    def get_mfi(self, symbol: str, interval: str, limit: int, lookback: int = 100) -> str:
        log.info(f"Getting MFI for symbol: {symbol}, interval: {interval}, limit: {limit}")
        bars = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # Calculate MFI, RSI, MA and whether open < close
        log.info(f"Input data for MFI calculation: {df[['high', 'low', 'close', 'volume']].tail(15)}")
        df['mfi'] = ta.volume.MFIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=14,
            fillna=False
        ).money_flow_index()
        log.info(f"Calculated MFI: {df['mfi'].tail(15)}")
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['ma'] = ta.trend.sma_indicator(df['close'], window=14)
        df['open_less_close'] = (df['open'] < df['close']).astype(int)

        last_mfi = df.iloc[-1]['mfi']
        last_rsi = df.iloc[-1]['rsi']
        last_ma = df.iloc[-1]['ma']
        last_open = df.iloc[-1]['open']
        last_close = df.iloc[-1]['close']
        log.info(f"Calculated MFI: {last_mfi}, RSI: {last_rsi}, MA: {last_ma}, Open: {last_open}, Close: {last_close} for symbol: {symbol}")

        df['buy_condition'] = ((df['mfi'] < 20) & (df['rsi'] < 35) & (df['open_less_close'] == 1)).astype(int)
        df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 65) & (df['open_less_close'] == 0)).astype(int)
        #df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 35) & (df['open_less_close'] == 0)).astype(int)

        # Check the last row for whether it's a buy or sell condition
        if df.iloc[-1]['buy_condition'] == 1:
            log.info(f"Buy condition met for symbol: {symbol}")
            return 'long'
        elif df.iloc[-1]['sell_condition'] == 1:
            log.info(f"Sell condition met for symbol: {symbol}")
            return 'short'
        else:
            # If neither condition is met on the last bar, look back at previous bars
            for i in range(2, min(len(df), lookback) + 1):  # look back up to 'lookback' bars
                if df.iloc[-i]['buy_condition'] == 1:
                    log.info(f"Buy condition met on a lookback for symbol: {symbol}")
                    return 'long'
                elif df.iloc[-i]['sell_condition'] == 1:
                    log.info(f"Sell condition met on a lookback for symbol: {symbol}")
                    return 'short'
            # In case no buy or sell condition was ever met, return 'neutral'
            log.info(f"No buy or sell condition met for symbol: {symbol}")
            return 'neutral'

    # def get_mfi(self, symbol: str, interval: str, limit: int) -> str:
    #     bars = self.exchange.get_futures_kline(
    #         symbol=symbol, interval=interval, limit=limit
    #     )
    #     df = pd.DataFrame(
    #         bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
    #     )

    #     # Calculate MFI, RSI, MA and whether open < close
    #     df['mfi'] = ta.volume.MFIIndicator(
    #         high=df['high'],
    #         low=df['low'],
    #         close=df['close'],
    #         volume=df['volume'],
    #         window=14,
    #         fillna=False
    #     ).money_flow_index()
    #     df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    #     df['ma'] = ta.trend.sma_indicator(df['close'], window=14)
    #     df['open_less_close'] = (df['open'] < df['close']).astype(int)


    #     df['buy_condition'] = ((df['mfi'] < 20) & (df['rsi'] < 35) & (df['open_less_close'] == 1)).astype(int)
    #     df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 35) & (df['open_less_close'] == 0)).astype(int)

    #     # Check the last row for whether it's a buy or sell condition
    #     if df.iloc[-1]['buy_condition'] == 1:
    #         return 'long'
    #     elif df.iloc[-1]['sell_condition'] == 1:
    #         return 'short'
    #     else:
    #         return 'neutral'


### Best lookback with logging
    # def get_mfi(self, symbol: str, interval: str, limit: int, lookback: int = 100) -> str:
    #     log.info(f"Getting MFI for symbol {symbol} with interval {interval}, limit {limit} and lookback {lookback}")

    #     bars = self.exchange.get_futures_kline(symbol=symbol, interval=interval, limit=limit)
    #     df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])

    #     df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
    #     df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    #     df['ma'] = ta.trend.sma_indicator(df['close'], window=14)
    #     df['open_less_close'] = (df['open'] < df['close']).astype(int)

    #     df['buy_condition'] = ((df['mfi'] < 20) & (df['rsi'] < 35) & (df['open_less_close'] == 1)).astype(int)
    #     df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 35) & (df['open_less_close'] == 0)).astype(int)

    #     log.info(f"Last MFI: {df.iloc[-1]['mfi']}, Last RSI: {df.iloc[-1]['rsi']}")

    #     if df.iloc[-1]['buy_condition'] == 1:
    #         log.info("Last condition was a buy.")
    #         return 'long'
    #     elif df.iloc[-1]['sell_condition'] == 1:
    #         log.info("Last condition was a sell.")
    #         return 'short'
    #     else:
    #         for i in range(2, min(len(df), lookback) + 1):  # look back up to 'lookback' bars
    #             if df.iloc[-i]['buy_condition'] == 1:
    #                 log.info(f"Found buy condition {i} bars ago.")
    #                 return 'long'
    #             elif df.iloc[-i]['sell_condition'] == 1:
    #                 log.info(f"Found sell condition {i} bars ago.")
    #                 return 'short'

    #     log.info("No buy or sell conditions met.")
    #     return 'neutral'
        
## MFI Return neutral
    # def get_mfi(self, symbol: str, interval: str, limit: int, lookback: int = 100) -> str:
    #     bars = self.exchange.get_futures_kline(
    #         symbol=symbol, interval=interval, limit=limit
    #     )
    #     df = pd.DataFrame(
    #         bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
    #     )

    #     # Calculate MFI, RSI, MA and whether open < close
    #     df['mfi'] = ta.volume.MFIIndicator(
    #         high=df['high'],
    #         low=df['low'],
    #         close=df['close'],
    #         volume=df['volume'],
    #         window=14,
    #         fillna=False
    #     ).money_flow_index()
    #     df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    #     df['ma'] = ta.trend.sma_indicator(df['close'], window=14)
    #     df['open_less_close'] = (df['open'] < df['close']).astype(int)

    #     df['buy_condition'] = ((df['mfi'] < 20) & (df['rsi'] < 35) & (df['open_less_close'] == 1)).astype(int)
    #     df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 35) & (df['open_less_close'] == 0)).astype(int)

    #     # Check the last row for whether it's a buy or sell condition
    #     if df.iloc[-1]['buy_condition'] == 1:
    #         return 'long'
    #     elif df.iloc[-1]['sell_condition'] == 1:
    #         return 'short'
    #     else:
    #         # If neither condition is met on the last bar, look back at previous bars
    #         for i in range(2, min(len(df), lookback) + 1):  # look back up to 'lookback' bars
    #             if df.iloc[-i]['buy_condition'] == 1:
    #                 return 'long'
    #             elif df.iloc[-i]['sell_condition'] == 1:
    #                 return 'short'
    #         # In case no buy or sell condition was ever met, return 'neutral'
    #         return 'neutral'

## Old mfi function
    # def get_mfi(self, symbol: str, interval: str, limit: int) -> str:
    #     log.info(f"Getting MFI for symbol {symbol} with interval {interval}, limit {limit}")
    #     bars = self.exchange.get_futures_kline(symbol=symbol, interval=interval, limit=limit)
    #     df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])

    #     df['mfi'] = ta.volume.MFIIndicator(
    #         high=df['high'],
    #         low=df['low'],
    #         close=df['close'],
    #         volume=df['volume'],
    #         window=14,
    #         fillna=False
    #     ).money_flow_index()
    #     df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    #     df['ma'] = ta.trend.sma_indicator(df['close'], window=14)
    #     df['open_less_close'] = (df['open'] < df['close']).astype(int)

    #     log.info(f"Last MFI: {df.iloc[-1]['mfi']}, Last RSI: {df.iloc[-1]['rsi']}")

    #     df['buy_condition'] = ((df['mfi'] < 20) & (df['rsi'] < 35) & (df['open_less_close'] == 1)).astype(int)
    #     df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 35) & (df['open_less_close'] == 0)).astype(int)

    #     if df.iloc[-1]['buy_condition'] == 1:
    #         log.info("Last condition was a buy.")
    #         return 'long'
    #     elif df.iloc[-1]['sell_condition'] == 1:
    #         log.info("Last condition was a sell.")
    #         return 'short'
    #     else:
    #         log.info("No buy or sell conditions met.")
    #         return 'neutral'

    ### Original analyse_symbol
    # def analyse_symbol(self, symbol: str) -> dict:
    #     len_slow_ma = 64
    #     len_power_ema = 13
    #     log.info(f"Analysing: {symbol}")
    #     values = {"Asset": symbol}

    #     values["Min qty"] = exchange.get_symbol_info(
    #         symbol=symbol, info="min_order_qty"
    #     )

    #     values["Price"] = self.prices[symbol]

    #     candles_30m = self.exchange.get_futures_kline(
    #         symbol=symbol, interval="30m", limit=5
    #     )
    #     candles_5m = self.exchange.get_futures_kline(
    #         symbol=symbol, interval="5m", limit=5
    #     )

    #     # Log the 3 most recent 5-minute candles
    #     log.info(f"3 most recent 5-minute candles for {symbol}: {candles_5m[-3:]}")

    #     candles_1m = self.exchange.get_futures_kline(
    #         symbol=symbol, interval="1m", limit=5
    #     )

    #     # Get data for the last 4 hours (240 minutes)
    #     data = self.exchange.get_futures_kline(symbol=symbol, interval="1m", limit=240)
    #     values["1m Spread"] = self.get_spread(symbol=symbol, limit=1, data=data[-1:])
    #     values["5m Spread"] = self.get_spread(symbol=symbol, limit=5, data=data[-5:])
    #     values["30m Spread"] = self.get_spread(symbol=symbol, limit=30, data=data[-30:])
    #     values["1h Spread"] = self.get_spread(symbol=symbol, limit=60, data=data[-60:])
    #     values["4h Spread"] = self.get_spread(symbol=symbol, limit=240, data=data)

    #     # Define 1x 5m candle volume
    #     onexcandlevol = candles_5m[-1]["volume"]
    #     volume_1x_5m = values["Price"] * onexcandlevol
    #     values["5m 1x Volume (USDT)"] = round(volume_1x_5m)

    #     # Define 1x 1m candle volume
    #     onex1mcandlevol = candles_1m[-1]["volume"]
    #     volume_1x = values["Price"] * onex1mcandlevol
    #     values["1m 1x Volume (USDT)"] = round(volume_1x)

    #     # Define 1x 30m candle volume
    #     onex30mcandlevol = candles_30m[-1]["volume"]
    #     volume_1x_30m = values["Price"] * onex30mcandlevol
    #     values["30m 1x Volume (USDT)"] = round(volume_1x_30m)

    #     # Define MA data
    #     values["5m MA6 high"] = self.get_candle_data(
    #         symbol=symbol, interval="5m", limit=20
    #     )["high_6"]
    #     values["5m MA6 low"] = self.get_candle_data(
    #         symbol=symbol, interval="5m", limit=20
    #     )["low_6"]

    #     ma_order_pct = self.get_sma(
    #         symbol=symbol, interval="1m", limit=30, column="close", window=14
    #     )
    #     values["trend%"] = ma_order_pct

    #     if ma_order_pct > 0:
    #         values["Trend"] = "short"
    #     else:
    #         values["Trend"] = "long"

    #     # Define funding rates
    #     values["Funding"] = self.exchange.get_funding_rate(symbol=symbol) * 100

    #     values["Timestamp"] = str(int(datetime.now().timestamp()))

    #     # Get MFI
    #     #mfi = self.get_mfi(symbol=symbol, interval="5m", limit=200, lookback=100)
    #     mfi = self.get_mfi(symbol=symbol, interval="1m", limit=200, lookback=100)
    #     #mfi = self.get_mfi(symbol=symbol, interval="5m", limit=25)
    #     values["MFI"] = mfi

    #     df = pd.DataFrame(data)
    #     df["close"] = pd.to_numeric(df["close"])
    #     df["high"] = pd.to_numeric(df["high"])
    #     df["low"] = pd.to_numeric(df["low"])

    #     # Calculate slow EMA of closing prices
    #     slow_ma = df['close'].ewm(span=len_slow_ma, adjust=False).mean()

    #     # Determine the trend
    #     last_price = df['close'].values[-1]
    #     eri_trend = "bullish" if last_price > slow_ma.values[-1] else "bearish"

    #     # Calculate bull power and bear power
    #     bull_power = df['high'] - slow_ma
    #     bear_power = df['low'] - slow_ma

    #     # Smooth the power values using EMA
    #     bull_power_smoothed = bull_power.ewm(span=len_power_ema, adjust=False).mean()
    #     bear_power_smoothed = bear_power.ewm(span=len_power_ema, adjust=False).mean()

    #     # Add to the values dict
    #     values["ERI Bull Power"] = bull_power_smoothed.values[-1]
    #     values["ERI Bear Power"] = bear_power_smoothed.values[-1]
    #     values["ERI Trend"] = eri_trend

    #     return values


    def analyse_symbol(self, symbol: str) -> dict:
        len_slow_ma = 64
        len_power_ema = 13
        log.info(f"Analysing: {symbol}")
        values = {"Asset": symbol}

        values["Min qty"] = exchange.get_symbol_info(
            symbol=symbol, info="min_order_qty"
        )

        values["Price"] = self.prices[symbol]

        # Define 1x 30m candle volume
        candles_30m = self.exchange.get_futures_kline(
            symbol=symbol, interval="30m", limit=5
        )
        df_30m = pd.DataFrame(
            candles_30m,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_30m[['open', 'high', 'low', 'close', 'volume']] = df_30m[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        df_30m['typical_price'] = (df_30m['high'] + df_30m['low'] + df_30m['close']) / 3
        onex30mcandlevol = df_30m['typical_price'].iloc[-1] * df_30m['volume'].iloc[-1]
        volume_1x_30m = values["Price"] * onex30mcandlevol
        values["30m 1x Volume (USDT)"] = round(volume_1x_30m)

        # Define 1x 5m candle volume
        candles_5m = self.exchange.get_futures_kline(
            symbol=symbol, interval="5m", limit=5
        )
        df_5m = pd.DataFrame(
            candles_5m,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_5m[['open', 'high', 'low', 'close', 'volume']] = df_5m[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        df_5m['typical_price'] = (df_5m['high'] + df_5m['low'] + df_5m['close']) / 3
        onex5mcandlevol = df_5m['typical_price'].iloc[-1] * df_5m['volume'].iloc[-1]
        volume_1x_5m = values["Price"] * onex5mcandlevol
        values["5m 1x Volume (USDT)"] = round(volume_1x_5m)

        # Define 1x 1m candle volume
        candles_1m = self.exchange.get_futures_kline(
            symbol=symbol, interval="1m", limit=5
        )
        df_1m = pd.DataFrame(
            candles_1m,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_1m[['open', 'high', 'low', 'close', 'volume']] = df_1m[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        df_1m['typical_price'] = (df_1m['high'] + df_1m['low'] + df_1m['close']) / 3
        onex1mcandlevol = df_1m['typical_price'].iloc[-1] * df_1m['volume'].iloc[-1]
        volume_1x_1m = values["Price"] * onex1mcandlevol
        values["1m 1x Volume (USDT)"] = round(volume_1x_1m)

        # Get data for the last 4 hours (240 minutes)
        data = self.exchange.get_futures_kline(symbol=symbol, interval="1m", limit=240)
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        values["1m Spread"] = self.get_spread(symbol=symbol, limit=1, data=data[-1:])
        values["5m Spread"] = self.get_spread(symbol=symbol, limit=5, data=data[-5:])
        values["30m Spread"] = self.get_spread(symbol=symbol, limit=30, data=data[-30:])
        values["1h Spread"] = self.get_spread(symbol=symbol, limit=60, data=data[-60:])
        values["4h Spread"] = self.get_spread(symbol=symbol, limit=240, data=data)

        values["5m MA6 high"] = self.get_candle_data(
            symbol=symbol, interval="5m", limit=20
        )["high_6"]
        values["5m MA6 low"] = self.get_candle_data(
            symbol=symbol, interval="5m", limit=20
        )["low_6"]

        ma_order_pct = self.get_sma(
            symbol=symbol, interval="1m", limit=30, column="close", window=14
        )
        values["trend%"] = ma_order_pct

        if ma_order_pct > 0:
            values["Trend"] = "short"
        else:
            values["Trend"] = "long"

        # Define funding rates
        values["Funding"] = self.exchange.get_funding_rate(symbol=symbol) * 100

        values["Timestamp"] = str(int(datetime.now().timestamp()))

        # Get MFI
        mfi = self.get_mfi(symbol=symbol, interval="1m", limit=100, lookback=100)
        values["MFI"] = mfi

        # Get ERI
        df["close"] = pd.to_numeric(df["close"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])

        # Calculate slow EMA of closing prices
        slow_ma = df['close'].ewm(span=len_slow_ma, adjust=False).mean()

        # Determine the trend
        last_price = df['close'].values[-1]
        eri_trend = "bullish" if last_price > slow_ma.values[-1] else "bearish"

        # Calculate bull power and bear power
        bull_power = df['high'] - slow_ma
        bear_power = df['low'] - slow_ma

        # Smooth the power values using EMA
        bull_power_smoothed = bull_power.ewm(span=len_power_ema, adjust=False).mean()
        bear_power_smoothed = bear_power.ewm(span=len_power_ema, adjust=False).mean()

        # Add to the values dict
        values["ERI Bull Power"] = bull_power_smoothed.values[-1]
        values["ERI Bear Power"] = bear_power_smoothed.values[-1]
        values["ERI Trend"] = eri_trend

        return values


    # def analyse_symbol(self, symbol: str) -> dict:
    #     len_slow_ma = 64
    #     len_power_ema = 13
    #     log.info(f"Analysing: {symbol}")
    #     values = {"Asset": symbol}

    #     values["Min qty"] = exchange.get_symbol_info(
    #         symbol=symbol, info="min_order_qty"
    #     )

    #     values["Price"] = self.prices[symbol]

    #     candles_30m = self.exchange.get_futures_kline(
    #         symbol=symbol, interval="30m", limit=5
    #     )
    #     candles_5m = self.exchange.get_futures_kline(
    #         symbol=symbol, interval="5m", limit=5
    #     )

    #     # Log the 3 most recent 5-minute candles
    #     log.info(f"3 most recent 5-minute candles for {symbol}: {candles_5m[-3:]}")

    #     candles_1m = self.exchange.get_futures_kline(
    #         symbol=symbol, interval="1m", limit=5
    #     )

    #     # Get data for the last 4 hours (240 minutes)
    #     data = self.exchange.get_futures_kline(symbol=symbol, interval="1m", limit=240)
    #     values["1m Spread"] = self.get_spread(symbol=symbol, limit=1, data=data[-1:])
    #     values["5m Spread"] = self.get_spread(symbol=symbol, limit=5, data=data[-5:])
    #     values["30m Spread"] = self.get_spread(symbol=symbol, limit=30, data=data[-30:])
    #     values["1h Spread"] = self.get_spread(symbol=symbol, limit=60, data=data[-60:])
    #     values["4h Spread"] = self.get_spread(symbol=symbol, limit=240, data=data)

    #     # Define 1x 5m candle volume
    #     typical_price_5m = candles_5m[-1]["typical_price"]
    #     onexcandlevol = candles_5m[-1]["volume"]
    #     volume_1x_5m = typical_price_5m * onexcandlevol
    #     values["5m 1x Volume (USDT)"] = round(volume_1x_5m)

    #     # Define 1x 1m candle volume
    #     typical_price_1m = candles_1m[-1]["typical_price"]
    #     onex1mcandlevol = candles_1m[-1]["volume"]
    #     volume_1x = typical_price_1m * onex1mcandlevol
    #     values["1m 1x Volume (USDT)"] = round(volume_1x)

    #     # Define 1x 30m candle volume
    #     typical_price_30m = candles_30m[-1]["typical_price"]
    #     onex30mcandlevol = candles_30m[-1]["volume"]
    #     volume_1x_30m = typical_price_30m * onex30mcandlevol
    #     values["30m 1x Volume (USDT)"] = round(volume_1x_30m)

    #     # Define MA data
    #     values["5m MA6 high"] = self.get_candle_data(
    #         symbol=symbol, interval="5m", limit=20
    #     )["high_6"]
    #     values["5m MA6 low"] = self.get_candle_data(
    #         symbol=symbol, interval="5m", limit=20
    #     )["low_6"]

    #     ma_order_pct = self.get_sma(
    #         symbol=symbol, interval="1m", limit=30, column="close", window=14
    #     )
    #     values["trend%"] = ma_order_pct

    #     if ma_order_pct > 0:
    #         values["Trend"] = "short"
    #     else:
    #         values["Trend"] = "long"

    #     # Define funding rates
    #     values["Funding"] = self.exchange.get_funding_rate(symbol=symbol) * 100

    #     values["Timestamp"] = str(int(datetime.now().timestamp()))

    #     # Get MFI
    #     mfi = self.get_mfi(symbol=symbol, interval="1m", limit=200, lookback=100)
    #     values["MFI"] = mfi

    #     df = pd.DataFrame(data)
    #     df["close"] = pd.to_numeric(df["close"])
    #     df["high"] = pd.to_numeric(df["high"])
    #     df["low"] = pd.to_numeric(df["low"])

    #     # Calculate slow EMA of closing prices
    #     slow_ma = df['close'].ewm(span=len_slow_ma, adjust=False).mean()

    #     # Determine the trend
    #     last_price = df['close'].values[-1]
    #     eri_trend = "bullish" if last_price > slow_ma.values[-1] else "bearish"

    #     # Calculate bull power and bear power
    #     bull_power = df['high'] - slow_ma
    #     bear_power = df['low'] - slow_ma

    #     # Smooth the power values using EMA
    #     bull_power_smoothed = bull_power.ewm(span=len_power_ema, adjust=False).mean()
    #     bear_power_smoothed = bear_power.ewm(span=len_power_ema, adjust=False).mean()

    #     # Add to the values dict
    #     values["ERI Bull Power"] = bull_power_smoothed.values[-1]
    #     values["ERI Bear Power"] = bear_power_smoothed.values[-1]
    #     values["ERI Trend"] = eri_trend

    #     return values


    def analyse_all_symbols(self, max_workers: int = 20, retry_limit: int = 5):
        data = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_data = {
                executor.submit(self.retry_analyse_symbol, symbol, retry_limit): symbol
                for symbol in self.symbols
            }
            for future in concurrent.futures.as_completed(future_data):
                symbol_data = future_data[future]
                try:
                    symbol_data_result = future.result()
                    data.append(symbol_data_result)
                except Exception as e:
                    log.error(f"{symbol_data} generated an exception: {e}")

        df = pd.DataFrame(
            data,
            columns=[
                "Asset",
                "Min qty",
                "Price",
                "1m 1x Volume (USDT)",
                "5m 1x Volume (USDT)",
                "30m 1x Volume (USDT)",
                "1m Spread",
                "5m Spread",
                "30m Spread",
                "1h Spread",
                "4h Spread",
                "trend%",
                "Trend",
                "5m MA6 high",
                "5m MA6 low",
                "Funding",
                "Timestamp",
                "MFI",
                "ERI Bull Power",
                "ERI Bear Power",
                "ERI Trend",
            ],
        )

        df.sort_values(
            by=["1m 1x Volume (USDT)", "5m Spread"],
            inplace=True,
            ascending=[False, False],
        )
        return df

    def retry_analyse_symbol(self, symbol: str, retry_limit: int):
        retry_count = 0
        while retry_count < retry_limit:
            try:
                return self.analyse_symbol(symbol)
            except Exception as e:
                retry_count += 1
                log.error(f"Exception while analysing {symbol}. Retry attempt {retry_count}. Exception: {e}")
                time.sleep(1)  # Optional: delay before retrying

        raise Exception(f"Failed to analyse {symbol} after {retry_limit} attempts.")

    def get_historical_volume(self, symbol: str, interval: str, limit: int):
        data = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )
        return [candle["volume"] for candle in data]

    def get_all_historical_volume(self, interval: str, limit: int) -> dict:
        all_volume = {}
        for symbol in scraper.symbols:
            data = scraper.get_historical_volume(
                symbol=symbol, interval=interval, limit=limit
            )
            all_volume[symbol] = data
        return all_volume

    def output_df(self, dataframe, path: str, to: str = "json"):
        if to == "json":
            dataframe.to_json(path, orient="records")
        elif to == "csv":
            dataframe.to_csv(path)
        elif to == "parquet":
            dataframe.to_parquet(path)
        elif to == "dict":
            dataframe.to_dict(path, orient="records")
        else:
            log.error(f"Output to {to} not implemented")

    def filter_df(self, dataframe, filter_col: str, operator: str, value: int):
        if operator == ">":
            return dataframe[dataframe[filter_col] > value]
        elif operator == "<":
            return dataframe[dataframe[filter_col] < value]
        elif operator == "==":
            return dataframe[dataframe[filter_col] == value]
        else:
            log.error(f"Operator {operator} not implemented")

    def reduce_df(self, dataframe, columns: list):
        return dataframe[columns]


if __name__ == "__main__":
    log.info("Starting process")
    quote_symbols = ["USDT"]
    top_volume = 400
    filters = {"quote_symbols": quote_symbols, "top_volume": top_volume}
    while True:
        try:
            with pidfile.PIDFile("scraper.pid"):
                exchange = Bybit()
                scraper = Scraper(exchange=exchange, filters=filters)

                start_time = time.time()
                data = scraper.analyse_all_symbols()
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time taken to analyse all symbols: {elapsed_time:.2f} seconds")
                print(data)
                scraper.output_df(dataframe=data, path="/opt/bitnami/nginx/html/data/quantdatav2.json", to="json")
                scraper.output_df(dataframe=data, path="data/quantdata.csv", to="csv")

                to_trade = scraper.filter_df(
                    dataframe=data,
                    filter_col="1m 1x Volume (USDT)",
                    operator=">",
                    value=15000,
                )
                scraper.output_df(
                    dataframe=to_trade, path="data/whattotrade.csv", to="csv"
                )
                scraper.output_df(
                    dataframe=to_trade, path="data/whattotrade.json", to="json"
                )

                negative = scraper.filter_df(
                    dataframe=data, filter_col="Funding", operator="<", value=0
                )
                negative = scraper.reduce_df(
                    dataframe=negative,
                    columns=["Asset", "1m 1x Volume (USDT)", "Funding"],
                )
                scraper.output_df(
                    dataframe=negative, path="data/negativefunding.csv", to="csv"
                )
                scraper.output_df(
                    dataframe=negative, path="data/negativefunding.json", to="json"
                )

                positive = scraper.filter_df(
                    dataframe=data, filter_col="Funding", operator=">", value=0
                )
                positive = scraper.reduce_df(
                    dataframe=positive,
                    columns=["Asset", "1m 1x Volume (USDT)", "Funding"],
                )
                scraper.output_df(
                    dataframe=positive, path="data/positivefunding.csv", to="csv"
                )
                scraper.output_df(
                    dataframe=positive, path="data/positivefunding.json", to="json"
                )

                total_historical_volume = scraper.get_all_historical_volume(
                    interval="1h", limit=24
                )
                with open("data/total_historical_volume.json", "w") as outfile:
                    json.dump(total_historical_volume, outfile)

        except pidfile.AlreadyRunningError:
            log.warning("Already running.")
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}")
        finally:
            log.info("Iteration completed. Waiting for the next run.")
            time.sleep(30)

        time.sleep(30)
