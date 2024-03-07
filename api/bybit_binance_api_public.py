from __future__ import annotations

import json
import sys
import time
import os
from datetime import timedelta
from multiprocessing import Process, Pool, cpu_count

import pandas as pd
import pidfile
import ta
import numpy as np

from decimal import Decimal, ROUND_HALF_UP

from datetime import datetime

sys.path.append(".")
from directionalscalper.api.exchanges.binance import Binance
from directionalscalper.api.exchanges.bybit import Bybit
from directionalscalper.core.utils import send_public_request
from directionalscalper.core.logger import Logger
log = Logger(filename="combined_scraper.log", stream=True)

funding_cache = {}  # We will handle cache differently in multiprocessing if needed

class CombinedScraper:
    def __init__(self, exchange_name, filters: dict):
        # Note: In multiprocessing, each process will have its own instance of CombinedScraper
        # and therefore its own instance of funding_cache. If the cache needs to be shared,
        # a different approach will be required.
        self.funding_cache = {}  # Local cache for each process
        self.exchange_name = exchange_name
        self.FUNDING_CACHE_DURATION = timedelta(hours=4)  # Set cache duration

        if exchange_name == "binance":
            self.exchange = Binance()
        elif exchange_name == "bybit":
            self.exchange = Bybit()
        else:
            raise ValueError("Invalid exchange name provided. Use 'binance' or 'bybit'.")
        
        log.info("Scraper initializing for " + exchange_name)
        self.filters = filters
        self.symbols = self.exchange.get_futures_symbols()
        self.prices = self.exchange.get_futures_prices()
        log.info(f"{len(self.symbols)} symbols found for " + exchange_name)
        
        if "quote_symbols" in self.filters:
            self.symbols = self.filter_quote(symbols=self.symbols, quotes=self.filters["quote_symbols"])
        
        if "top_volume" in self.filters:
            self.volumes = self.exchange.get_futures_volumes()
            self.symbols = self.filter_volume(symbols=self.symbols, volumes=self.volumes, limit=self.filters["top_volume"])

    def get_all_historical_volume(self, exchange_name: str, interval: str, limit: int) -> dict:
        all_volume = {}

        if exchange_name == "bybit":
            for symbol in self.symbols:
                try:
                    volume = self.get_historical_volume_bybit(symbol, interval, limit)
                    all_volume[symbol] = volume
                except Exception as e:
                    log.error(f"Error getting historical volume for {symbol} on Bybit: {e}")
        elif exchange_name == "binance":
            for symbol in self.symbols:
                try:
                    volume = self.get_historical_volume_binance(symbol, interval, limit)
                    all_volume[symbol] = volume
                except Exception as e:
                    log.error(f"Error getting historical volume for {symbol} on Binance: {e}")
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        return all_volume

    def get_historical_volume_bybit(self, symbol: str, interval: str, limit: int):
        data = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )
        return symbol, [candle["volume"] for candle in data]

    def get_historical_volume_binance(self, symbol: str, interval: str, limit: int) -> tuple:
        endpoint = "/fapi/v1/klines"
        payload = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        _, raw_json = send_public_request(
            url=self.exchange.futures_api_url, 
            url_path=endpoint, 
            payload=payload
        )
        volumes = [entry[5] for entry in raw_json]  # Assuming volume is at index 5 in the klines data
        return symbol, volumes

    def get_cached_funding(self, symbol):
        now = datetime.now()

        # Check if data is in cache and still valid
        if symbol in self.funding_cache:
            last_fetched, rate = self.funding_cache[symbol]
            if now - last_fetched < self.FUNDING_CACHE_DURATION:
                return rate

        # If not in cache or stale, fetch data
        rate = self.exchange.get_funding_rate(symbol=symbol) * 100

        # Update the cache
        self.funding_cache[symbol] = (now, rate)

        return rate


    def output_df(self, dataframe, path: str, to: str = "json"):
        if to == "json":
            dataframe.to_json(path, orient="records", date_format='iso')
        elif to == "csv":
            dataframe.to_csv(path, index=False)
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

    def get_spread(self, symbol: str, limit: int, timeframe: str = "1m", data: list | None = None) -> float:
        if data is None:
            data = self.exchange.get_futures_kline(symbol=symbol, interval=timeframe, limit=limit)
        data_df = pd.DataFrame(data)
        highest_high = data_df['high'].max()
        lowest_low = data_df['low'].min()
        if highest_high > 0:
            return round((highest_high - lowest_low) / highest_high * 100, 4)
        return 0.0

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

    def get_hma(self, symbol: str, interval: str, limit: int, column: str, window: int):
        bars = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df['HMA'] = self.compute_hma(df, column, window)
        hma_order_pct = round((df[column].iloc[-1] - df['HMA'].iloc[-1]) / df[column].iloc[-1] * 100, 4)

        return hma_order_pct

    def compute_hma(self, df, column: str, window: int):
        # Step 1
        wma_half_period = df[column].rolling(window=int(window/2)).mean()

        # Step 2
        wma_full_period = df[column].rolling(window=window).mean()

        # Step 3
        series2 = 2 * wma_half_period - wma_full_period

        # Step 4 & 5
        hma = series2.rolling(window=int(np.sqrt(window))).mean()
        return hma

    def get_ema(self, symbol: str, interval: str, limit: int, column: str, window: int):
        bars = self.exchange.get_futures_kline(
            symbol=symbol, interval=interval, limit=limit
        )  # 1m, 18, 6
        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df[f"EMA{window} {column}"] = ta.trend.EMAIndicator(
            df[column], window=window
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

    def calculate_advanced_eri(self, symbol, timeframe, len_slow_ma=64, len_power_ema=13, limit=128):
        """
        Calculate an Elder-ray Index (ERI) similar to RustyC's approach, using VWMA followed by EMA.

        :param symbol: Trading pair symbol.
        :param timeframe: Timeframe for candlestick data.
        :param len_slow_ma: Length for slow moving average (VWMA followed by EMA).
        :param len_power_ema: Length for EMA of bull and bear power.
        :param limit: Number of candlesticks to fetch.
        :return: A dictionary containing ERI trend, bull power, and bear power.
        """
        # Fetching data
        data = self.exchange.get_futures_kline(symbol=symbol, interval=timeframe, limit=limit)

        # Create a DataFrame from the data
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

        # Calculate VWMA (Volume Weighted Moving Average)
        vwma = ((df['close'] * df['volume']).rolling(window=len_slow_ma).sum() / df['volume'].rolling(window=len_slow_ma).sum())

        # Calculate EMA of VWMA
        slow_vwma_ema = vwma.ewm(span=len_slow_ma, adjust=False).mean()

        # Determine the trend
        last_price = df['close'].values[-1]
        eri_trend = "bullish" if last_price > slow_vwma_ema.values[-1] else "bearish"

        # Calculate bull power and bear power
        bull_power = df['high'] - slow_vwma_ema
        bear_power = df['low'] - slow_vwma_ema

        # Smooth the power values using EMA
        bull_power_smoothed = bull_power.ewm(span=len_power_ema, adjust=False).mean()
        bear_power_smoothed = bear_power.ewm(span=len_power_ema, adjust=False).mean()

        # Prepare results
        result = {
            "ERI Trend": eri_trend,
            "ERI Bull Power": bull_power_smoothed.values[-1],
            "ERI Bear Power": bear_power_smoothed.values[-1]
        }

        return result
        
    def get_mfi(self, symbol: str, interval: str, limit: int, lookback: int = 1) -> str:
        bars = self.exchange.get_futures_kline(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])

        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['open_less_close'] = (df['open'] < df['close']).astype(int)
        
        df['buy_condition'] = ((df['mfi'] < 30) & (df['rsi'] < 40) & (df['open_less_close'] == 1)).astype(int)
        df['sell_condition'] = ((df['mfi'] > 80) & (df['rsi'] > 70) & (df['open_less_close'] == 0)).astype(int)

        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].any():
            return 'long'
        elif recent_conditions['sell_condition'].any():
            return 'short'
        else:
            return 'neutral'
        
    def analyse_symbol(self, symbol: str) -> dict:

        len_slow_ma = 64
        len_power_ema = 13
        log.info(f"Analysing: {symbol}")
        values = {"Asset": symbol}

        values["Min qty"] = self.exchange.get_symbol_info(
            symbol=symbol, info="min_order_qty"
        )

        values["Price"] = self.prices[symbol]

        candles_1h = self.exchange.get_futures_kline(
            symbol=symbol, interval="1h", limit=5
        )

        candles_30m = self.exchange.get_futures_kline(
            symbol=symbol, interval="30m", limit=5
        )
        candles_5m = self.exchange.get_futures_kline(
            symbol=symbol, interval="5m", limit=5
        )
        candles_1m = self.exchange.get_futures_kline(
            symbol=symbol, interval="1m", limit=5
        )

        data = self.exchange.get_futures_kline(symbol=symbol, interval="1m", limit=240)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)

        values["1m Spread"] = self.get_spread(symbol=symbol, limit=1, data=data[-1:])
        values["5m Spread"] = self.get_spread(symbol=symbol, limit=5, data=data[-5:])
        values["30m Spread"] = self.get_spread(symbol=symbol, limit=30, data=data[-30:])
        values["1h Spread"] = self.get_spread(symbol=symbol, limit=60, data=data[-60:])
        values["4h Spread"] = self.get_spread(symbol=symbol, limit=240, data=data)

        # Define 1x 5m candle volume
        onexcandlevol = candles_5m[-1]["volume"]
        volume_1x_5m = values["Price"] * onexcandlevol
        values["5m 1x Volume (USDT)"] = round(volume_1x_5m)

        # Define 1x 1m candle volume
        onex1mcandlevol = candles_1m[-1]["volume"]
        volume_1x = values["Price"] * onex1mcandlevol
        values["1m 1x Volume (USDT)"] = round(volume_1x)

        # Define 1x 30m candle volume
        onex30mcandlevol = candles_30m[-1]["volume"]
        volume_1x_30m = values["Price"] * onex30mcandlevol
        values["30m 1x Volume (USDT)"] = round(volume_1x_30m)

        onex1hcandlevol = candles_1h[-1]["volume"]
        volume_1x_1h = values["Price"] * onex1hcandlevol
        values["1h 1x Volume (USDT)"] = round(volume_1x_1h)

        # Define MA data
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
        #values["Funding"] = self.exchange.get_funding_rate(symbol=symbol) * 100
        values["Funding"] = self.get_cached_funding(symbol)

        values["Timestamp"] = str(int(datetime.now().timestamp()))

        mfi = self.get_mfi(symbol=symbol, interval="1m", limit=200, lookback=5)

        # mfi = self.get_mfi(symbol=symbol, interval="1m", limit=500, short_lookback=200, long_lookback=500)
        values["MFI"] = mfi


        # Convert the 'close' column to numeric values
        df['close'] = pd.to_numeric(df['close'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])

        # Calculate ATR and ATRP
        atr = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['ATR'] = atr
        df['ATRP'] = (df['ATR'] / df['close']) * 100

        # Get the last ATRP value
        values['ATRP'] = df['ATRP'].iloc[-1]

        # ERI

        # Use '60' instead of '1h' for the 1-hour interval
        eri_timeframe = "15m"  # 60 minutes for 1 hour

        # Calculating ERI
        eri_result = self.calculate_advanced_eri(symbol, eri_timeframe)
        # eri_result = self.calculate_original_eri(symbol, eri_timeframe)

        # Adding ERI values to the dictionary
        values.update(eri_result)

        return values

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

    def detect_peaks_and_troughs(self, prices):
        peaks = []
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i-1] < prices[i] > prices[i+1]:
                peaks.append(i)
            elif prices[i-1] > prices[i] < prices[i+1]:
                troughs.append(i)
        return peaks, troughs

    def calculate_atrp(self, dataframe: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        dataframe['ATR'] = ta.volatility.AverageTrueRange(
            dataframe['High'], dataframe['Low'], dataframe['Close'], n=period
        ).average_true_range()

        dataframe['ATRP'] = (dataframe['ATR'] / dataframe['Close']) * 100
        return dataframe

    def calculate_distance_from_line(self, price, slope, intercept, index):
        """Calculate the vertical distance of a price from the linear regression line."""
        line_price = (slope * index) + intercept
        return abs(line_price - price)

    def analyse_symbol_wrapper(self, args):
        # This wrapper will be used to pass multiple arguments to the function used with Pool
        return self.retry_analyse_symbol(*args)

    def analyse_all_symbols(self, retry_limit: int = 5):
        # Create a pool of 14 worker processes
        # with Pool(processes=14) as pool:
        pool_size = cpu_count()
        chunksize = max(1, len(self.symbols) // (pool_size * 4))  # Example heuristic for chunksize

        with Pool(processes=pool_size) as pool:
            # Map the analyse_symbol_wrapper function to all symbols with the retry_limit
            # and specify the chunksize
            data = pool.map(self.analyse_symbol_wrapper, [(symbol, retry_limit) for symbol in self.symbols], chunksize=chunksize)

        # Filter out None results if any failed analyses returned None
        data = [result for result in data if result is not None]

        # Create the DataFrame with the collected data
        df = pd.DataFrame(
            data,
            columns=[
                "Asset",
                "Min qty",
                "Price",
                "1m 1x Volume (USDT)",
                "5m 1x Volume (USDT)",
                "30m 1x Volume (USDT)",
                "1h 1x Volume (USDT)",
                "ATRP",
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
                "ERI Trend"
            ],
        )
        # Sort the DataFrame as required
        df.sort_values(by=["1m 1x Volume (USDT)", "5m Spread"], inplace=True, ascending=[False, False])
        return df

def run_scraper_for_exchange(exchange_name: str):
    log.info(f"Starting scraper for {exchange_name}")

    # User-defined parameters
    quote_symbols = ["USDT"]
    top_volume = 400
    filters = {"quote_symbols": quote_symbols, "top_volume": top_volume}

    while True:
        start_time = time.time()
        try:
            with pidfile.PIDFile(f"{exchange_name}_scraper.pid"):
                scraper = CombinedScraper(exchange_name=exchange_name, filters=filters)
                
                # Analyzing all symbols with multiprocessing
                df = scraper.analyse_all_symbols()
                
                # Define the main path and temporary path for the JSON file
                log.info(f"Setting file paths for exchange: {exchange_name}")
                main_path_quant = f"/var/www/api/data/quantdatav2_{exchange_name}.json"
                temp_path_quant = f"/var/www/api/data/quantdatav2_{exchange_name}_temp.json"
                log.info(f"Main path set to: {main_path_quant}")
                log.info(f"Temporary path set to: {temp_path_quant}")

                # Save analysis data to the temporary file
                log.info(f"Attempting to save analysis data to {temp_path_quant}.")
                scraper.output_df(dataframe=df, path=temp_path_quant, to="json")
                log.info(f"Saved analysis data to {temp_path_quant}.")

                # Rename the temporary file to the main file (atomic operation)
                os.rename(temp_path_quant, main_path_quant)

                # Filter and save 'to_trade' data
                to_trade = scraper.filter_df(
                    dataframe=df,
                    filter_col="5m 1x Volume (USDT)",
                    operator=">",
                    value=15000,
                )

                # Define the main path and temporary path for the JSON file
                main_path_trade = f"/var/www/api/data/whattotrade_{exchange_name}.json"
                temp_path_trade = f"/var/www/api/data/whattotrade_{exchange_name}_temp.json"

                # Save to_trade data to the temporary file
                scraper.output_df(dataframe=to_trade, path=temp_path_trade, to="json")

                # Rename the temporary file to the main file (atomic operation)
                os.rename(temp_path_trade, main_path_trade)

                # Filter and save 'rotator_symbols' data
                rotator_symbols = scraper.filter_df(
                    dataframe=df,
                    filter_col="1m 1x Volume (USDT)",
                    operator=">",
                    value=16000,
                )

                # Sorting rotator_symbols
                rotator_symbols = rotator_symbols.sort_values(by="1m 1x Volume (USDT)", ascending=False)

                #rotator_symbols = rotator_symbols.sort_values(by=["1m 1x Volume (USDT)", "5m Spread"], ascending=[False, False])

                # Define the main path and temporary path for the JSON file
                main_path = f"/var/www/api/data/rotatorsymbols_{exchange_name}.json"
                temp_path = f"/var/www/api/data/rotatorsymbols_{exchange_name}_temp.json"

                # Save rotator symbols to the temporary file
                scraper.output_df(
                    dataframe=rotator_symbols, 
                    path=temp_path, 
                    to="json"
                )

                # Rename the temporary file to the main file (atomic operation)
                os.rename(temp_path, main_path)

                # # If the exchange is bybit, save rotator symbols to the old path as well
                # if exchange_name == "bybit":
                #     scraper.output_df(dataframe=rotator_symbols, path="/var/www/api/data/rotatorsymbols.json", to="json")


                # Filter and save 'negative' funding data
                negative = scraper.filter_df(
                    dataframe=df, filter_col="Funding", operator="<", value=0
                )
                negative = scraper.reduce_df(
                    dataframe=negative,
                    columns=["Asset", "1m 1x Volume (USDT)", "Funding"],
                )

                scraper.output_df(
                    dataframe=negative, path=f"/var/www/api/data/negativefunding_{exchange_name}.json", to="json"
                )

                # Filter and save 'positive' funding data
                positive = scraper.filter_df(
                    dataframe=df, filter_col="Funding", operator=">", value=0
                )
                positive = scraper.reduce_df(
                    dataframe=positive,
                    columns=["Asset", "1m 1x Volume (USDT)", "Funding"],
                )

                scraper.output_df(
                    dataframe=positive, path=f"/var/www/api/data/positivefunding_{exchange_name}.json", to="json"
                )

                #total_historical_volume = scraper.get_all_historical_volume(interval="1h", limit=24, exchange_name=exchange_name)
                total_historical_volume = scraper.get_all_historical_volume(exchange_name=exchange_name, interval="1h", limit=24)

                # Then save the JSON
                with open(f"/var/www/api/data/total_historical_volume_{exchange_name}.json", "w") as outfile:
                    json.dump(total_historical_volume, outfile)

        except pidfile.AlreadyRunningError:
            log.warning(f"{exchange_name} scraper already running.")
        except Exception as e:
            log.error(f"An unexpected error occurred for {exchange_name} scraper: {e}")
        finally:
            elapsed_time = time.time() - start_time  # Compute the elapsed time
            log.info(f"{exchange_name} scraper iteration completed in {elapsed_time:.2f} seconds. Waiting for the next run.")
            time.sleep(10)  # Wait for 60 seconds before the next run

def scraper_and_notifier(exchange):
    while True:
        run_scraper_for_exchange(exchange)

if __name__ == "__main__":
    # Start both scrapers
    Process(target=scraper_and_notifier, args=("binance",)).start()
    Process(target=scraper_and_notifier, args=("bybit",)).start()

    while True:
        time.sleep(10)  # A short sleep to reduce CPU usage of this loop.
