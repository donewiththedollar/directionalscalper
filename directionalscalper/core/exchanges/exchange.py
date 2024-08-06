import os
import logging
import time
import random
import ta as ta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import CCIIndicator, ADXIndicator, EMAIndicator, SMAIndicator
import uuid
import ccxt
import pandas as pd
import numpy as np
import json
import requests, hmac, hashlib
import urllib.parse
import threading
import traceback
from typing import Optional, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from ccxt.base.errors import RateLimitExceeded
from ..strategies.logger import Logger
from requests.exceptions import HTTPError
from datetime import datetime, timedelta
from ccxt.base.errors import NetworkError

logging = Logger(logger_name="Exchange", filename="Exchange.log", stream=True)

from rate_limit import RateLimit

class Exchange:
    # Shared class-level cache variables
    symbols_cache = None
    symbols_cache_time = None
    symbols_cache_duration = 300  # Cache duration in seconds

    open_positions_shared_cache = None
    last_open_positions_time_shared = None
    open_positions_semaphore = threading.Semaphore()

    def __init__(self, exchange_id, api_key, secret_key, passphrase=None, market_type='swap'):
        self.order_timestamps = None
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.market_type = market_type  # Store the market type
        self.name = exchange_id
        self.initialise()
        self.symbols = self._get_symbols()
        self.market_precisions = {}
        self.open_positions_cache = None
        self.last_open_positions_time = None

        self.entry_order_ids = {}  # Initialize order history
        self.entry_order_ids_lock = threading.Lock()  # For thread safety
        self.rate_limiter = RateLimit(10, 1)

        self.last_signal = {}
        self.last_signal_time = {}
        self.signal_duration = 60  # Duration in seconds (1 minute)
        
    def initialise(self):
        exchange_class = getattr(ccxt, self.exchange_id)
        exchange_params = {
            "apiKey": self.api_key,
            "secret": self.secret_key,
            "enableRateLimit": True,
        }
        if os.environ.get('HTTP_PROXY') and os.environ.get('HTTPS_PROXY'):
            exchange_params["proxies"] = {
                'http': os.environ.get('HTTP_PROXY'),
                'https': os.environ.get('HTTPS_PROXY'),
            }
        if self.passphrase:
            exchange_params["password"] = self.passphrase

        if self.exchange_id.lower() == 'bybit_spot':
            exchange_params['options'] = {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
            self.exchange_id = 'bybit'  # Change the exchange ID to 'bybit' for CCXT
        elif self.exchange_id.lower() == 'bybit':
            exchange_params['options'] = {
                'defaultType': self.market_type,
                'adjustForTimeDifference': True,
            }
        else:
            exchange_params['options'] = {
                'defaultType': self.market_type,
                'adjustForTimeDifference': True,
            }

        if self.exchange_id.lower() == 'hyperliquid':
            exchange_params['options'] = {
                'sandboxMode': False,
                # Set Liquid-specific options here
            }

        if self.exchange_id.lower().startswith('bybit'):
            exchange_params['options']['brokerId'] = 'Nu000450'

        # Existing condition for Huobi
        if self.exchange_id.lower() == 'huobi' and self.market_type == 'swap':
            exchange_params['options']['defaultSubType'] = 'linear'

        # Additional condition for Blofin
        if self.exchange_id.lower() == 'blofin':
            exchange_params['options'] = {
                'defaultType': self.market_type,
                'adjustForTimeDifference': True,
            }
            
        # Initializing the exchange object
        self.exchange = exchange_class(exchange_params)
        # Checks if load_markets() have already been ran once.
        if not self.exchange.markets == None: return
        print(f"Loading exchange {self.exchange} for API data")
        self.exchange.load_markets()
        print(f'Loaded exchange {self.exchange} for API data')

    def get_mfirsi_ema_secondary_ema(self, symbol: str, limit: int = 100, lookback: int = 1, ema_period: int = 5, secondary_ema_period: int = 3) -> str:
        # Fetch OHLCV data
        ohlcv_data = self.exchange.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Calculate MFI and RSI
        df['mfi'] = ta.volume.MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14, fillna=False).money_flow_index()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # Calculate EMAs for MFI and RSI
        df['mfi_ema'] = df['mfi'].ewm(span=ema_period, adjust=False).mean()
        df['rsi_ema'] = df['rsi'].ewm(span=ema_period, adjust=False).mean()

        # Calculate secondary EMAs for MFI and RSI
        df['mfi_ema_secondary'] = df['mfi'].ewm(span=secondary_ema_period, adjust=False).mean()
        df['rsi_ema_secondary'] = df['rsi'].ewm(span=secondary_ema_period, adjust=False).mean()

        # Determine conditions using EMAs and secondary EMAs
        df['buy_condition'] = (
            (df['mfi_ema'] < 30) &
            (df['rsi_ema'] < 40) &
            (df['mfi_ema_secondary'] < df['mfi_ema']) &
            (df['rsi_ema_secondary'] < df['rsi_ema']) &
            (df['open'] < df['close'])
        ).astype(int)
        df['sell_condition'] = (
            (df['mfi_ema'] > 70) &
            (df['rsi_ema'] > 60) &
            (df['mfi_ema_secondary'] > df['mfi_ema']) &
            (df['rsi_ema_secondary'] > df['rsi_ema']) &
            (df['open'] > df['close'])
        ).astype(int)

        # Evaluate conditions over the lookback period
        recent_conditions = df.iloc[-lookback:]
        if recent_conditions['buy_condition'].sum() > 0:
            return 'long'
        elif recent_conditions['sell_condition'].sum() > 0:
            return 'short'
        else:
            return 'neutral'

    # Fetch OHLCV data for calculating ZigZag
    def fetch_ohlcv_data(self, symbol, timeframe='5m', limit=5000):
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    # Calculate ZigZag indicator
    def calculate_zigzag(self, ohlcv, length=4):
        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]

        def highest_high(index):
            return max(highs[max(0, index-length):index+length+1])

        def lowest_low(index):
            return min(lows[max(0, index-length):index+length+1])

        direction_up = False
        last_low = max(highs) * 100
        last_high = 0.0
        peaks_and_troughs = []

        for i in range(length, len(ohlcv) - length):
            h = highest_high(i)
            l = lowest_low(i)

            is_min = l == lows[i]
            is_max = h == highs[i]

            if direction_up:
                if is_min and lows[i] < last_low:
                    last_low = lows[i]
                    peaks_and_troughs.append(last_low)
                if is_max and highs[i] > last_low:
                    last_high = highs[i]
                    direction_up = False
                    peaks_and_troughs.append(last_high)
            else:
                if is_max and highs[i] > last_high:
                    last_high = highs[i]
                    peaks_and_troughs.append(last_high)
                if is_min and lows[i] < last_high:
                    last_low = lows[i]
                    direction_up = True
                    peaks_and_troughs.append(last_low)

        return peaks_and_troughs

    # Normalize prices for clustering
    def normalize_prices(self, prices):
        min_price = min(prices)
        max_price = max(prices)
        return [(price - min_price) / (max_price - min_price) for price in prices]

    # Mean deviation calculation
    def mean_deviation(self, prices):
        mean = sum(prices) / len(prices)
        return sum(abs(price - mean) for price in prices) / len(prices)

    # Function to identify significant support and resistance levels
    def get_significant_levels_dbscan(self, zigzag, ohlcv_data):
        normalized_zigzag = self.normalize_prices(zigzag)
        average_deviation = self.mean_deviation(normalized_zigzag)
        epsilon = average_deviation * 0.04
        minimum_points = 2

        data_points = np.array(normalized_zigzag).reshape(-1, 1)
        db = DBSCAN(eps=epsilon, min_samples=minimum_points).fit(data_points)
        labels = db.labels_

        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        def median(numbers):
            sorted_numbers = sorted(numbers)
            middle = len(sorted_numbers) // 2
            if len(sorted_numbers) % 2 == 0:
                return (sorted_numbers[middle - 1] + sorted_numbers[middle]) / 2
            else:
                return sorted_numbers[middle]

        support_resistance_levels = []
        for cluster in clusters.values():
            if cluster:
                cluster_prices = [zigzag[idx] for idx in cluster]
                cluster_volumes = [ohlcv_data[idx][5] for idx in cluster]
                level = median(cluster_prices)
                strength = len(cluster_prices)
                average_volume = sum(cluster_volumes) / len(cluster_volumes)
                support_resistance_levels.append({
                    'level': level,
                    'strength': strength,
                    'average_volume': average_volume,
                })

        support_resistance_levels.sort(key=lambda x: x['level'], reverse=True)
        return support_resistance_levels

    def normalize(self, series):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        scaler = MinMaxScaler()
        series_values = series.values.reshape(-1, 1)
        normalized_values = scaler.fit_transform(series_values).flatten()
        return pd.Series(normalized_values, index=series.index)

    def rescale(self, series, new_min=0, new_max=1):
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        old_min, old_max = series.min(), series.max()
        rescaled_values = new_min + (new_max - new_min) * (series - old_min) / (old_max - old_min)
        return pd.Series(rescaled_values, index=series.index)

    def n_rsi(self, series, n1, n2):
        rsi = RSIIndicator(series, window=n1).rsi()
        return self.rescale(rsi.ewm(span=n2, adjust=False).mean())

    def n_cci(self, high, low, close, n1, n2):
        cci = CCIIndicator(high, low, close, window=n1).cci()
        return self.normalize(cci.ewm(span=n2, adjust=False).mean())

    def n_wt(self, hlc3, n1=10, n2=11):
        ema1 = EMAIndicator(hlc3, window=n1).ema_indicator()
        ema2 = EMAIndicator(abs(hlc3 - ema1), window=n1).ema_indicator()
        ci = (hlc3 - ema1) / (0.015 * ema2)
        wt1 = EMAIndicator(ci, window=n2).ema_indicator()
        wt2 = SMAIndicator(wt1, window=4).sma_indicator()
        return self.normalize(wt1 - wt2)

    def n_adx(self, high, low, close, n1):
        adx = ADXIndicator(high, low, close, window=n1).adx()
        return self.rescale(adx)

    def regime_filter(self, series, high, low, use_regime_filter, threshold):
        if not use_regime_filter:
            return pd.Series([True] * len(series))

        def klmf(series, high, low):
            value1 = pd.Series(0, index=series.index)
            value2 = pd.Series(0, index=series.index)
            klmf = pd.Series(0, index=series.index)
            for i in range(1, len(series)):
                value1[i] = 0.2 * (series[i] - series[i - 1]) + 0.8 * value1[i - 1]
                value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1]
            omega = abs(value1 / value2)
            alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8
            for i in range(1, len(series)):
                klmf[i] = alpha[i] * series[i] + (1 - alpha[i]) * klmf[i - 1]
            return klmf

        klmf_values = klmf(series, high, low)
        abs_curve_slope = abs(klmf_values.diff())
        exponential_average_abs_curve_slope = EMAIndicator(abs_curve_slope, window=200).ema_indicator()
        normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope
        return normalized_slope_decline >= threshold

    def filter_adx(self, close, high, low, adx_threshold, use_adx_filter=False, length=14):
        if not use_adx_filter:
            return pd.Series([True] * len(close))
        adx = ADXIndicator(high, low, close, window=length).adx()
        return adx > adx_threshold

    def filter_volatility(self, high, low, close, use_volatility_filter, min_length=1, max_length=10):
        if not use_volatility_filter:
            return pd.Series([True] * len(close))
        recent_atr = AverageTrueRange(high, low, close, window=min_length).average_true_range()
        historical_atr = AverageTrueRange(high, low, close, window=max_length).average_true_range()
        return recent_atr > historical_atr

    def lorentzian_distance(self, feature_series, feature_arrays):
        distances = np.log(1 + np.abs(feature_series - feature_arrays))
        return distances.sum(axis=1)

    def generate_l_signals(self, symbol, limit=3000, neighbors_count=8, use_adx_filter=False, adx_threshold=20):
        try:
            # Fetch OHLCV data
            ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='3m', limit=limit)
            df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df.set_index('timestamp', inplace=True)

            # Calculate technical indicators
            df['rsi'] = self.n_rsi(df['close'], 14, 1)
            df['adx'] = self.n_adx(df['high'], df['low'], df['close'], 14)  # ADX is always calculated
            df['cci'] = self.n_cci(df['high'], df['low'], df['close'], 20, 1)
            df['wt'] = self.n_wt((df['high'] + df['low'] + df['close']) / 3, 10, 11)

            # Feature engineering
            features = df[['rsi', 'adx', 'cci', 'wt']].values  # ADX included in feature set
            feature_series = features[-1]
            feature_arrays = features[:-1]

            # Calculate Lorentzian distances and predictions
            y_train_series = np.where(df['close'].shift(-4) > df['close'], 1, -1)
            y_train_series = y_train_series[:-1]

            predictions = []
            distances = []
            lastDistance = -1

            for i in range(len(feature_arrays)):
                if i % 4 == 0:
                    d = np.log(1 + np.abs(feature_series - feature_arrays[i])).sum()
                    if d >= lastDistance:
                        lastDistance = d
                        distances.append(d)
                        predictions.append(y_train_series[i])
                        if len(predictions) > neighbors_count:
                            lastDistance = distances[int(neighbors_count * 3 / 4)]
                            distances.pop(0)
                            predictions.pop(0)

            prediction = np.sum(predictions)

            # Calculate EMA and SMA
            df['ema'] = EMAIndicator(df['close'], window=200).ema_indicator()
            df['sma'] = SMAIndicator(df['close'], window=200).sma_indicator()

            # Determine trends
            is_ema_uptrend = df['close'] > df['ema']
            is_ema_downtrend = df['close'] < df['ema']
            is_sma_uptrend = df['close'] > df['sma']
            is_sma_downtrend = df['close'] < df['sma']

            # Apply ADX filter if enabled
            adx_filter = self.filter_adx(df['close'], df['high'], df['low'], adx_threshold, use_adx_filter)

            # Generate signal based on prediction and trends
            new_signal = 'neutral'
            if prediction > 0 and is_ema_uptrend.iloc[-1] and is_sma_uptrend.iloc[-1] and adx_filter.iloc[-1]:
                new_signal = 'long'
            elif prediction < 0 and is_ema_downtrend.iloc[-1] and is_sma_downtrend.iloc[-1] and adx_filter.iloc[-1]:
                new_signal = 'short'

            current_time = time.time()
            logging.info(f"New signal for {symbol} : {new_signal}")
            logging.info(f"Last signal for {symbol} : {self.last_signal.get(symbol, 'None')}")

            # Avoid double entries and ensure signal change
            if symbol in self.last_signal:
                if self.last_signal[symbol] == new_signal and new_signal != 'neutral':
                    # Check if sufficient time has passed to act on the same signal
                    if current_time - self.last_signal_time[symbol] < 15:  # 15 second buffer
                        logging.info(f"Returning {new_signal} because {self.last_signal[symbol]} is same as new signal and time difference is within buffer")
                        return new_signal  # Return the same signal if within the buffer time
                    else:
                        self.last_signal_time[symbol] = current_time
                        return new_signal
                else:
                    self.last_signal[symbol] = new_signal
                    self.last_signal_time[symbol] = current_time
                    return new_signal
            else:
                self.last_signal[symbol] = new_signal
                self.last_signal_time[symbol] = current_time
                return new_signal
        except Exception as e:
            logging.info(f"Error in calculating signal: {e}")
            return 'neutral'
        
    # def generate_l_signals(self, symbol, limit=3000, neighbors_count=8, use_adx_filter=False, adx_threshold=20):
    #     try:
    #         # Fetch OHLCV data
    #         ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='3m', limit=limit)
    #         df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    #         df.set_index('timestamp', inplace=True)

    #         # Calculate technical indicators
    #         df['rsi'] = self.n_rsi(df['close'], 14, 1)
    #         df['adx'] = self.n_adx(df['high'], df['low'], df['close'], 14)  # ADX is always calculated
    #         df['cci'] = self.n_cci(df['high'], df['low'], df['close'], 20, 1)
    #         df['wt'] = self.n_wt((df['high'] + df['low'] + df['close']) / 3, 10, 11)

    #         # Feature engineering
    #         features = df[['rsi', 'adx', 'cci', 'wt']].values  # ADX included in feature set
    #         feature_series = features[-1]
    #         feature_arrays = features[:-1]

    #         # Calculate Lorentzian distances and predictions
    #         y_train_series = np.where(df['close'].shift(-4) > df['close'], 1, -1)
    #         y_train_series = y_train_series[:-1]

    #         predictions = []
    #         distances = []
    #         lastDistance = -1

    #         for i in range(len(feature_arrays)):
    #             if i % 4 == 0:
    #                 d = np.log(1 + np.abs(feature_series - feature_arrays[i])).sum()
    #                 if d >= lastDistance:
    #                     lastDistance = d
    #                     distances.append(d)
    #                     predictions.append(y_train_series[i])
    #                     if len(predictions) > neighbors_count:
    #                         lastDistance = distances[int(neighbors_count * 3 / 4)]
    #                         distances.pop(0)
    #                         predictions.pop(0)

    #         prediction = np.sum(predictions)

    #         # Calculate EMA and SMA
    #         df['ema'] = EMAIndicator(df['close'], window=200).ema_indicator()
    #         df['sma'] = SMAIndicator(df['close'], window=200).sma_indicator()

    #         # Determine trends
    #         is_ema_uptrend = df['close'] > df['ema']
    #         is_ema_downtrend = df['close'] < df['ema']
    #         is_sma_uptrend = df['close'] > df['sma']
    #         is_sma_downtrend = df['close'] < df['sma']

    #         # Apply ADX filter if enabled
    #         adx_filter = self.filter_adx(df['close'], df['high'], df['low'], adx_threshold, use_adx_filter)

    #         # Generate signal based on prediction and trends
    #         new_signal = 'neutral'
    #         if prediction > 0 and is_ema_uptrend.iloc[-1] and is_sma_uptrend.iloc[-1] and adx_filter.iloc[-1]:
    #             new_signal = 'long'
    #         elif prediction < 0 and is_ema_downtrend.iloc[-1] and is_sma_downtrend.iloc[-1] and adx_filter.iloc[-1]:
    #             new_signal = 'short'

    #         # Avoid double entries and ensure signal change
    #         if hasattr(self, 'last_signal'):
    #             if self.last_signal == new_signal:
    #                 return 'neutral'
    #             else:
    #                 self.last_signal = new_signal
    #                 return new_signal
    #         else:
    #             self.last_signal = new_signal
    #             return new_signal
    #     except Exception as e:
    #         logging.info(f"Error in calculating signal: {e}")
    #         return 'neutral'
        
    # def normalize(self, series):
    #     if not isinstance(series, pd.Series):
    #         series = pd.Series(series)
    #     scaler = MinMaxScaler()
    #     series_values = series.values.reshape(-1, 1)
    #     normalized_values = scaler.fit_transform(series_values).flatten()
    #     return pd.Series(normalized_values, index=series.index)

    # def rescale(self, series, new_min=0, new_max=1):
    #     if not isinstance(series, pd.Series):
    #         series = pd.Series(series)
    #     old_min, old_max = series.min(), series.max()
    #     rescaled_values = new_min + (new_max - new_min) * (series - old_min) / (old_max - old_min)
    #     return pd.Series(rescaled_values, index=series.index)

    # def n_rsi(self, series, n1, n2):
    #     rsi = RSIIndicator(series, window=n1).rsi()
    #     return self.rescale(rsi.ewm(span=n2, adjust=False).mean())

    # def n_cci(self, high, low, close, n1, n2):
    #     cci = CCIIndicator(high, low, close, window=n1).cci()
    #     return self.normalize(cci.ewm(span=n2, adjust=False).mean())

    # def n_wt(self, hlc3, n1=10, n2=11):
    #     ema1 = EMAIndicator(hlc3, window=n1).ema_indicator()
    #     ema2 = EMAIndicator(abs(hlc3 - ema1), window=n1).ema_indicator()
    #     ci = (hlc3 - ema1) / (0.015 * ema2)
    #     wt1 = EMAIndicator(ci, window=n2).ema_indicator()
    #     wt2 = SMAIndicator(wt1, window=4).sma_indicator()
    #     return self.normalize(wt1 - wt2)

    # def n_adx(self, high, low, close, n1):
    #     adx = ADXIndicator(high, low, close, window=n1).adx()
    #     return self.rescale(adx)

    # def regime_filter(self, series, high, low, use_regime_filter, threshold):
    #     if not use_regime_filter:
    #         return pd.Series([True] * len(series))

    #     def klmf(series, high, low):
    #         value1 = pd.Series(0, index=series.index)
    #         value2 = pd.Series(0, index=series.index)
    #         klmf = pd.Series(0, index=series.index)
    #         for i in range(1, len(series)):
    #             value1[i] = 0.2 * (series[i] - series[i - 1]) + 0.8 * value1[i - 1]
    #             value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1]
    #         omega = abs(value1 / value2)
    #         alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8
    #         for i in range(1, len(series)):
    #             klmf[i] = alpha[i] * series[i] + (1 - alpha[i]) * klmf[i - 1]
    #         return klmf

    #     klmf_values = klmf(series, high, low)
    #     abs_curve_slope = abs(klmf_values.diff())
    #     exponential_average_abs_curve_slope = EMAIndicator(abs_curve_slope, window=200).ema_indicator()
    #     normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope
    #     return normalized_slope_decline >= threshold

    # def filter_adx(self, close, high, low, adx_threshold, use_adx_filter, length=14):
    #     if not use_adx_filter:
    #         return pd.Series([True] * len(close))
    #     adx = ADXIndicator(high, low, close, window=length).adx()
    #     return adx > adx_threshold

    # def filter_volatility(self, high, low, close, use_volatility_filter, min_length=1, max_length=10):
    #     if not use_volatility_filter:
    #         return pd.Series([True] * len(close))
    #     recent_atr = AverageTrueRange(high, low, close, window=min_length).average_true_range()
    #     historical_atr = AverageTrueRange(high, low, close, window=max_length).average_true_range()
    #     return recent_atr > historical_atr

    # def lorentzian_distance(self, feature_series, feature_arrays):
    #     distances = np.log(1 + np.abs(feature_series - feature_arrays))
    #     return distances.sum(axis=1)

    # def generate_l_signals(self, symbol, limit=3000, neighbors_count=8):
    #     try:
    #         # Fetch OHLCV data
    #         ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='3m', limit=limit)
    #         df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    #         df.set_index('timestamp', inplace=True)

    #         # Calculate technical indicators
    #         df['rsi'] = self.n_rsi(df['close'], 14, 1)
    #         df['adx'] = self.n_adx(df['high'], df['low'], df['close'], 14)
    #         df['cci'] = self.n_cci(df['high'], df['low'], df['close'], 20, 1)
    #         df['wt'] = self.n_wt((df['high'] + df['low'] + df['close']) / 3, 10, 11)

    #         # Feature engineering
    #         features = df[['rsi', 'adx', 'cci', 'wt']].values
    #         feature_series = features[-1]
    #         feature_arrays = features[:-1]

    #         # Calculate Lorentzian distances and predictions
    #         y_train_series = np.where(df['close'].shift(-4) > df['close'], 1, -1)
    #         y_train_series = y_train_series[:-1]

    #         predictions = []
    #         distances = []
    #         lastDistance = -1

    #         for i in range(len(feature_arrays)):
    #             if i % 4 == 0:
    #                 d = np.log(1 + np.abs(feature_series - feature_arrays[i])).sum()
    #                 if d >= lastDistance:
    #                     lastDistance = d
    #                     distances.append(d)
    #                     predictions.append(y_train_series[i])
    #                     if len(predictions) > neighbors_count:
    #                         lastDistance = distances[int(neighbors_count * 3 / 4)]
    #                         distances.pop(0)
    #                         predictions.pop(0)

    #         prediction = np.sum(predictions)

    #         # Calculate EMA and SMA
    #         df['ema'] = EMAIndicator(df['close'], window=200).ema_indicator()
    #         df['sma'] = SMAIndicator(df['close'], window=200).sma_indicator()

    #         # Determine trends
    #         is_ema_uptrend = df['close'] > df['ema']
    #         is_ema_downtrend = df['close'] < df['ema']
    #         is_sma_uptrend = df['close'] > df['sma']
    #         is_sma_downtrend = df['close'] < df['sma']

    #         # Generate signal based on prediction and trends
    #         new_signal = 'neutral'
    #         if prediction > 0 and is_ema_uptrend.iloc[-1] and is_sma_uptrend.iloc[-1]:
    #             new_signal = 'long'
    #         elif prediction < 0 and is_ema_downtrend.iloc[-1] and is_sma_downtrend.iloc[-1]:
    #             new_signal = 'short'

    #         # Avoid double entries and ensure signal change
    #         if hasattr(self, 'last_signal'):
    #             if self.last_signal == new_signal:
    #                 return 'neutral'
    #             else:
    #                 self.last_signal = new_signal
    #                 return new_signal
    #         else:
    #             self.last_signal = new_signal
    #             return new_signal
    #     except Exception as e:
    #         logging.info(f"Error in calculating signal: {e}")
    #         return 'neutral'
        
    # def normalize(self, series):
    #     scaler = MinMaxScaler()
    #     series_values = series.values.reshape(-1, 1)  # Convert to 2D array for scaler
    #     normalized_values = scaler.fit_transform(series_values).flatten()
    #     return pd.Series(normalized_values, index=series.index)

    # def rescale(self, series, new_min=0, new_max=1):
    #     old_min, old_max = series.min(), series.max()
    #     rescaled_values = new_min + (new_max - new_min) * (series - old_min) / (old_max - old_min)
    #     return pd.Series(rescaled_values, index=series.index)

    # def n_rsi(self, series, n1, n2):
    #     rsi = RSIIndicator(series, window=n1).rsi()
    #     return self.rescale(rsi.ewm(span=n2, adjust=False).mean())

    # def n_cci(self, high, low, close, n1, n2):
    #     cci = CCIIndicator(high, low, close, window=n1).cci()
    #     return self.normalize(cci.ewm(span=n2, adjust=False).mean())

    # def n_wt(self, hlc3, n1=10, n2=11):
    #     ema1 = EMAIndicator(hlc3, window=n1).ema_indicator()
    #     ema2 = EMAIndicator(abs(hlc3 - ema1), window=n1).ema_indicator()
    #     ci = (hlc3 - ema1) / (0.015 * ema2)
    #     wt1 = EMAIndicator(ci, window=n2).ema_indicator()
    #     wt2 = SMAIndicator(wt1, window=4).sma_indicator()
    #     return self.normalize(wt1 - wt2)

    # def n_adx(self, high, low, close, n1):
    #     adx = ADXIndicator(high, low, close, window=n1).adx()
    #     return self.rescale(adx)

    # def regime_filter(self, series, high, low, use_regime_filter, threshold):
    #     if not use_regime_filter:
    #         return pd.Series([True] * len(series))

    #     def klmf(series, high, low):
    #         value1 = pd.Series(0, index=series.index)
    #         value2 = pd.Series(0, index=series.index)
    #         klmf = pd.Series(0, index=series.index)
    #         for i in range(1, len(series)):
    #             value1[i] = 0.2 * (series[i] - series[i - 1]) + 0.8 * value1[i - 1]
    #             value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1]
    #         omega = abs(value1 / value2)
    #         alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8
    #         for i in range(1, len(series)):
    #             klmf[i] = alpha[i] * series[i] + (1 - alpha[i]) * klmf[i - 1]
    #         return klmf

    #     klmf_values = klmf(series, high, low)
    #     abs_curve_slope = abs(klmf_values.diff())
    #     exponential_average_abs_curve_slope = EMAIndicator(abs_curve_slope, window=200).ema_indicator()
    #     normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope
    #     return normalized_slope_decline >= threshold

    # def filter_adx(self, close, high, low, adx_threshold, use_adx_filter, length=14):
    #     if not use_adx_filter:
    #         return pd.Series([True] * len(close))
    #     adx = ADXIndicator(high, low, close, window=length).adx()
    #     return adx > adx_threshold

    # def filter_volatility(self, high, low, close, use_volatility_filter, min_length=1, max_length=10):
    #     if not use_volatility_filter:
    #         return pd.Series([True] * len(close))
    #     recent_atr = AverageTrueRange(high, low, close, window=min_length).average_true_range()
    #     historical_atr = AverageTrueRange(high, low, close, window=max_length).average_true_range()
    #     return recent_atr > historical_atr

    # def lorentzian_distance(self, feature_series, feature_arrays):
    #     distances = np.log(1 + np.abs(feature_series - feature_arrays))
    #     return distances.sum(axis=1)

    # def generate_l_signals(self, symbol, limit=3000, neighbors_count=8):
    #     # Fetch OHLCV data
    #     ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='3m', limit=limit)
    #     df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    #     df.set_index('timestamp', inplace=True)

    #     # Calculate technical indicators
    #     df['rsi'] = self.n_rsi(df['close'], 14, 1)
    #     df['adx'] = self.n_adx(df['high'], df['low'], df['close'], 14)
    #     df['cci'] = self.n_cci(df['high'], df['low'], df['close'], 20, 1)
    #     df['wt'] = self.n_wt((df['high'] + df['low'] + df['close']) / 3, 10, 11)

    #     # Feature engineering
    #     features = df[['rsi', 'adx', 'cci', 'wt']].values

    #     # Training labels (future price movement)
    #     y_train_series = np.where(df['close'].shift(-4) > df['close'], 1, -1)
    #     y_train_series = y_train_series[:-1]

    #     # Initialize variables for prediction logic
    #     predictions = []
    #     distances = []
    #     lastDistance = -1

    #     for j in range(len(features)):
    #         if j >= limit - 1:  # maxBarsBack equivalent
    #             for i in range(min(j, limit - 1)):
    #                 d = 0
    #                 for feature_idx in range(features.shape[1]):
    #                     d += np.log(1 + np.abs(features[j][feature_idx] - features[i][feature_idx]))
    #                 if d >= lastDistance and i % 4 == 0:
    #                     lastDistance = d
    #                     distances.append(d)
    #                     predictions.append(y_train_series[i])
    #                     if len(predictions) > neighbors_count:
    #                         lastDistance = distances[int(neighbors_count * 3 / 4)]
    #                         distances.pop(0)
    #                         predictions.pop(0)

    #             prediction = sum(predictions)
    #         else:
    #             prediction = 0

    #     # Calculate EMA and SMA
    #     df['ema'] = EMAIndicator(df['close'], window=200).ema_indicator()
    #     df['sma'] = SMAIndicator(df['close'], window=200).sma_indicator()

    #     # Determine trends
    #     is_ema_uptrend = df['close'] > df['ema']
    #     is_ema_downtrend = df['close'] < df['ema']
    #     is_sma_uptrend = df['close'] > df['sma']
    #     is_sma_downtrend = df['close'] < df['sma']

    #     # Generate signal based on prediction and trends
    #     new_signal = 'neutral'
    #     if prediction > 0 and is_ema_uptrend.iloc[-1] and is_sma_uptrend.iloc[-1]:
    #         new_signal = 'long'
    #     elif prediction < 0 and is_ema_downtrend.iloc[-1] and is_sma_downtrend.iloc[-1]:
    #         new_signal = 'short'

    #     # Avoid double entries and ensure signal change
    #     if hasattr(self, 'last_signal'):
    #         if self.last_signal == new_signal:
    #             return 'neutral'
    #         else:
    #             self.last_signal = new_signal
    #             return new_signal
    #     else:
    #         self.last_signal = new_signal
    #         return new_signal

    # # Credit to 53RG0
    # def normalize(self, series):
    #     scaler = MinMaxScaler()
    #     series_values = series.values.reshape(-1, 1)  # Convert to 2D array for scaler
    #     normalized_values = scaler.fit_transform(series_values).flatten()
    #     return pd.Series(normalized_values, index=series.index)

    # def rescale(self, series, new_min=0, new_max=1):
    #     old_min, old_max = series.min(), series.max()
    #     rescaled_values = new_min + (new_max - new_min) * (series - old_min) / (old_max - old_min)
    #     return pd.Series(rescaled_values, index=series.index)

    # def n_rsi(self, series, n1, n2):
    #     rsi = RSIIndicator(series, window=n1).rsi()
    #     return self.rescale(rsi.ewm(span=n2, adjust=False).mean())

    # def n_cci(self, high, low, close, n1, n2):
    #     cci = CCIIndicator(high, low, close, window=n1).cci()
    #     return self.normalize(cci.ewm(span=n2, adjust=False).mean())

    # def n_wt(self, hlc3, n1=10, n2=11):
    #     ema1 = EMAIndicator(hlc3, window=n1).ema_indicator()
    #     ema2 = EMAIndicator(abs(hlc3 - ema1), window=n1).ema_indicator()
    #     ci = (hlc3 - ema1) / (0.015 * ema2)
    #     wt1 = EMAIndicator(ci, window=n2).ema_indicator()
    #     wt2 = SMAIndicator(wt1, window=4).sma_indicator()
    #     return self.normalize(wt1 - wt2)

    # def n_adx(self, high, low, close, n1):
    #     adx = ADXIndicator(high, low, close, window=n1).adx()
    #     return self.rescale(adx)

    # def regime_filter(self, series, high, low, use_regime_filter, threshold):
    #     if not use_regime_filter:
    #         return pd.Series([True] * len(series))

    #     def klmf(series, high, low):
    #         value1 = pd.Series(0, index=series.index)
    #         value2 = pd.Series(0, index=series.index)
    #         klmf = pd.Series(0, index=series.index)
    #         for i in range(1, len(series)):
    #             value1[i] = 0.2 * (series[i] - series[i - 1]) + 0.8 * value1[i - 1]
    #             value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1]
    #         omega = abs(value1 / value2)
    #         alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8
    #         for i in range(1, len(series)):
    #             klmf[i] = alpha[i] * series[i] + (1 - alpha[i]) * klmf[i - 1]
    #         return klmf

    #     klmf_values = klmf(series, high, low)
    #     abs_curve_slope = abs(klmf_values.diff())
    #     exponential_average_abs_curve_slope = EMAIndicator(abs_curve_slope, window=200).ema_indicator()
    #     normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope
    #     return normalized_slope_decline >= threshold

    # def filter_adx(self, close, high, low, adx_threshold, use_adx_filter, length=14):
    #     if not use_adx_filter:
    #         return pd.Series([True] * len(close))
    #     adx = ADXIndicator(high, low, close, window=length).adx()
    #     return adx > adx_threshold

    # def filter_volatility(self, high, low, close, use_volatility_filter, min_length=1, max_length=10):
    #     if not use_volatility_filter:
    #         return pd.Series([True] * len(close))
    #     recent_atr = AverageTrueRange(high, low, close, window=min_length).average_true_range()
    #     historical_atr = AverageTrueRange(high, low, close, window=max_length).average_true_range()
    #     return recent_atr > historical_atr

    # def lorentzian_distance(self, feature_series, feature_arrays):
    #     distances = np.log(1 + np.abs(feature_series - feature_arrays))
    #     return distances.sum(axis=1)

    # def generate_l_signals(self, symbol, limit=3000, neighbors_count=8):
    #     # Fetch OHLCV data
    #     ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='3m', limit=limit)
    #     df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    #     df.set_index('timestamp', inplace=True)

    #     # Calculate technical indicators
    #     df['rsi'] = self.n_rsi(df['close'], 14, 1)
    #     df['adx'] = self.n_adx(df['high'], df['low'], df['close'], 14)
    #     df['cci'] = self.n_cci(df['high'], df['low'], df['close'], 20, 1)
    #     df['wt'] = self.n_wt((df['high'] + df['low'] + df['close']) / 3, 10, 11)

    #     # Feature engineering
    #     features = df[['rsi', 'adx', 'cci', 'wt']].values
    #     feature_series = features[-1]
    #     feature_arrays = features[:-1]

    #     # Calculate Lorentzian distances
    #     distances = self.lorentzian_distance(feature_series, feature_arrays)
    #     nearest_indices = distances.argsort()[:neighbors_count]

    #     # Training labels (future price movement)
    #     y_train_series = np.where(df['close'].shift(-4) > df['close'], 1, -1)
    #     y_train_series = y_train_series[:-1]
    #     predictions = y_train_series[nearest_indices]
    #     prediction = np.sum(predictions)

    #     # Calculate EMA and SMA
    #     df['ema'] = EMAIndicator(df['close'], window=200).ema_indicator()
    #     df['sma'] = SMAIndicator(df['close'], window=200).sma_indicator()

    #     # Determine trends
    #     is_ema_uptrend = df['close'] > df['ema']
    #     is_ema_downtrend = df['close'] < df['ema']
    #     is_sma_uptrend = df['close'] > df['sma']
    #     is_sma_downtrend = df['close'] < df['sma']

    #     # Generate signal based on prediction and trends
    #     new_signal = 'neutral'
    #     if prediction > 0 and is_ema_uptrend.iloc[-1] and is_sma_uptrend.iloc[-1]:
    #         new_signal = 'long'
    #     elif prediction < 0 and is_ema_downtrend.iloc[-1] and is_sma_downtrend.iloc[-1]:
    #         new_signal = 'short'

    #     # Avoid double entries and ensure signal change
    #     if hasattr(self, 'last_signal'):
    #         if self.last_signal == new_signal:
    #             return 'neutral'
    #         else:
    #             self.last_signal = new_signal
    #             return new_signal
    #     else:
    #         self.last_signal = new_signal
    #         return new_signal
        

    # Works but maybe keeps old signal
    # def generate_l_signals(self, symbol, limit=1000, neighbors_count=8):
    #     ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
    #     df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    #     df.set_index('timestamp', inplace=True)

    #     df['rsi'] = self.n_rsi(df['close'], 14, 1)
    #     df['adx'] = self.n_adx(df['high'], df['low'], df['close'], 14)
    #     df['cci'] = self.n_cci(df['high'], df['low'], df['close'], 20, 1)
    #     df['wt'] = self.n_wt((df['high'] + df['low'] + df['close']) / 3, 10, 11)

    #     features = df[['rsi', 'adx', 'cci', 'wt']].values
    #     feature_series = features[-1]
    #     feature_arrays = features[:-1]

    #     distances = self.lorentzian_distance(feature_series, feature_arrays)
    #     nearest_indices = distances.argsort()[:neighbors_count]

    #     y_train_series = np.where(df['close'].shift(-4) > df['close'], 1, -1)
    #     y_train_series = y_train_series[:-1]
    #     predictions = y_train_series[nearest_indices]
    #     prediction = np.sum(predictions)

    #     df['ema'] = EMAIndicator(df['close'], window=200).ema_indicator()
    #     df['sma'] = SMAIndicator(df['close'], window=200).sma_indicator()

    #     is_ema_uptrend = df['close'] > df['ema']
    #     is_ema_downtrend = df['close'] < df['ema']
    #     is_sma_uptrend = df['close'] > df['sma']
    #     is_sma_downtrend = df['close'] < df['sma']

    #     if prediction > 0 and is_ema_uptrend.iloc[-1] and is_sma_uptrend.iloc[-1]:
    #         return 'long'
    #     elif prediction < 0 and is_ema_downtrend.iloc[-1] and is_sma_downtrend.iloc[-1]:
    #         return 'short'
    #     else:
    #         return 'neutral'

    # def normalize(self, series):
    #     scaler = MinMaxScaler()
    #     series = series.values.reshape(-1, 1)
    #     normalized_series = scaler.fit_transform(series).flatten()
    #     return pd.Series(normalized_series, index=series.index)

    # def rescale(self, series, new_min=0, new_max=1):
    #     old_min, old_max = series.min(), series.max()
    #     rescaled_series = new_min + (new_max - new_min) * (series - old_min) / (old_max - old_min)
    #     return pd.Series(rescaled_series, index=series.index)

    # def n_rsi(self, series, n1, n2):
    #     rsi = ta.momentum.RSIIndicator(series, window=n1).rsi()
    #     return self.rescale(rsi.ewm(span=n2, adjust=False).mean())

    # def n_cci(self, high, low, close, n1, n2):
    #     cci = ta.trend.CCIIndicator(high, low, close, window=n1).cci()
    #     return self.normalize(cci.ewm(span=n2, adjust=False).mean())

    # def n_wt(self, hlc3, n1=10, n2=11):
    #     ema1 = ta.trend.EMAIndicator(hlc3, window=n1).ema_indicator()
    #     ema2 = ta.trend.EMAIndicator(abs(hlc3 - ema1), window=n1).ema_indicator()
    #     ci = (hlc3 - ema1) / (0.015 * ema2)
    #     wt1 = ta.trend.EMAIndicator(ci, window=n2).ema_indicator()
    #     wt2 = ta.trend.SMAIndicator(wt1, window=4).sma_indicator()
    #     return self.normalize(wt1 - wt2)

    # def n_adx(self, high, low, close, n1):
    #     adx = ta.trend.ADXIndicator(high, low, close, window=n1).adx()
    #     return self.rescale(adx)

    # def regime_filter(self, series, high, low, use_regime_filter, threshold):
    #     if not use_regime_filter:
    #         return pd.Series([True] * len(series))

    #     def klmf(series, high, low):
    #         value1 = pd.Series(0, index=series.index)
    #         value2 = pd.Series(0, index=series.index)
    #         klmf = pd.Series(0, index=series.index)
    #         for i in range(1, len(series)):
    #             value1[i] = 0.2 * (series[i] - series[i - 1]) + 0.8 * value1[i - 1]
    #             value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1]
    #         omega = abs(value1 / value2)
    #         alpha = (-omega ** 2 + np.sqrt(omega ** 4 + 16 * omega ** 2)) / 8
    #         for i in range(1, len(series)):
    #             klmf[i] = alpha[i] * series[i] + (1 - alpha[i]) * klmf[i - 1]
    #         return klmf

    #     klmf_values = klmf(series, high, low)
    #     abs_curve_slope = abs(klmf_values.diff())
    #     exponential_average_abs_curve_slope = ta.trend.EMAIndicator(abs_curve_slope, window=200).ema_indicator()
    #     normalized_slope_decline = (abs_curve_slope - exponential_average_abs_curve_slope) / exponential_average_abs_curve_slope
    #     return normalized_slope_decline >= threshold

    # def filter_adx(self, close, high, low, adx_threshold, use_adx_filter, length=14):
    #     if not use_adx_filter:
    #         return pd.Series([True] * len(close))
    #     adx = ta.trend.ADXIndicator(high, low, close, window=length).adx()
    #     return adx > adx_threshold

    # def filter_volatility(self, high, low, close, use_volatility_filter, min_length=1, max_length=10):
    #     if not use_volatility_filter:
    #         return pd.Series([True] * len(close))
    #     recent_atr = ta.volatility.AverageTrueRange(high, low, close, window=min_length).average_true_range()
    #     historical_atr = ta.volatility.AverageTrueRange(high, low, close, window=max_length).average_true_range()
    #     return recent_atr > historical_atr

    # def lorentzian_distance(self, feature_series, feature_arrays):
    #     distances = np.log(1 + np.abs(feature_series - feature_arrays))
    #     return distances.sum(axis=1)

    # def generate_l_signals(self, symbol, limit=1000, neighbors_count=8):
    #     ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
    #     df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    #     df.set_index('timestamp', inplace=True)

    #     df['rsi'] = self.n_rsi(df['close'], 14, 1)
    #     df['adx'] = self.n_adx(df['high'], df['low'], df['close'], 14)
    #     df['cci'] = self.n_cci(df['high'], df['low'], df['close'], 20, 1)
    #     df['wt'] = self.n_wt((df['high'] + df['low'] + df['close']) / 3, 10, 11)

    #     features = df[['rsi', 'adx', 'cci', 'wt']].values
    #     feature_series = features[-1]
    #     feature_arrays = features[:-1]

    #     distances = self.lorentzian_distance(feature_series, feature_arrays)
    #     nearest_indices = distances.argsort()[:neighbors_count]

    #     y_train_series = np.where(df['close'].shift(-4) > df['close'], 1, -1)
    #     y_train_series = y_train_series.values[:-1]
    #     predictions = y_train_series[nearest_indices]
    #     prediction = np.sum(predictions)

    #     df['ema'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
    #     df['sma'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()

    #     is_ema_uptrend = df['close'] > df['ema']
    #     is_ema_downtrend = df['close'] < df['ema']
    #     is_sma_uptrend = df['close'] > df['sma']
    #     is_sma_downtrend = df['close'] < df['sma']

    #     if prediction > 0 and is_ema_uptrend.iloc[-1] and is_sma_uptrend.iloc[-1]:
    #         return 'long'
    #     elif prediction < 0 and is_ema_downtrend.iloc[-1] and is_sma_downtrend.iloc[-1]:
    #         return 'short'
    #     else:
    #         return 'neutral'

    def get_l_signal(self, symbol: str, limit: int = 1000, neighbors_count: int = 8) -> str:
        # Fetch OHLCV data
        ohlcv_data = self.fetch_ohlcv(symbol=symbol, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        # Calculate technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
        
        # Normalize indicators
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())
        
        df['rsi'] = normalize(df['rsi'])
        df['adx'] = normalize(df['adx'])
        df['cci'] = normalize(df['cci'])
        
        # Prepare feature matrix
        features = df[['rsi', 'adx', 'cci']].values
        distances = []
        
        # Calculate Lorentzian distances
        for i in range(len(features) - 1):
            dist = np.sum(np.log(1 + np.abs(features[-1] - features[i])))
            distances.append(dist)
        
        distances = np.array(distances)
        nearest_indices = distances.argsort()[:neighbors_count]
        
        # Determine prediction based on nearest neighbors
        close_prices = df['close'].values
        predictions = [1 if close_prices[idx + 4] > close_prices[idx] else -1 for idx in nearest_indices]
        prediction = np.sum(predictions)
        
        # Filters (simple EMA and SMA trend filters)
        ema_period = 200
        sma_period = 200
        df['ema'] = df['close'].ewm(span=ema_period, adjust=False).mean()
        df['sma'] = df['close'].rolling(window=sma_period).mean()
        
        is_ema_uptrend = df['close'].iloc[-1] > df['ema'].iloc[-1]
        is_ema_downtrend = df['close'].iloc[-1] < df['ema'].iloc[-1]
        is_sma_uptrend = df['close'].iloc[-1] > df['sma'].iloc[-1]
        is_sma_downtrend = df['close'].iloc[-1] < df['sma'].iloc[-1]
        
        # Generate signals
        if prediction > 0 and is_ema_uptrend and is_sma_uptrend:
            return 'long'
        elif prediction < 0 and is_ema_downtrend and is_sma_downtrend:
            return 'short'
        else:
            return 'neutral'

    def update_order_history(self, symbol, order_id, timestamp):
        with self.entry_order_ids_lock:
            # Check if the symbol is already in the order history
            if symbol not in self.entry_order_ids:
                self.entry_order_ids[symbol] = []
                logging.info(f"Creating new order history entry for symbol: {symbol}")

            # Append the new order data
            self.entry_order_ids[symbol].append({'id': order_id, 'timestamp': timestamp})
            logging.info(f"Updated order history for {symbol} with order ID {order_id} at timestamp {timestamp}")

            # Optionally, log the entire current order history for the symbol
            logging.debug(f"Current order history for {symbol}: {self.entry_order_ids[symbol]}")
            
    def set_order_timestamps(self, order_timestamps):
        self.order_timestamps = order_timestamps

    def populate_order_history(self, symbols: list, since: int = None, limit: int = 100):
        for symbol in symbols:
            try:
                logging.info(f"Fetching trades for {symbol}")
                recent_trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)

                # Check if recent_trades is None or empty
                if not recent_trades:
                    logging.info(f"No trade data returned for {symbol}. It might not be a valid symbol or no recent trades.")
                    continue

                last_trade = recent_trades[-1]
                last_trade_time = datetime.fromtimestamp(last_trade['timestamp'] / 1000)  # Convert ms to seconds

                if symbol not in self.order_timestamps:
                    self.order_timestamps[symbol] = []
                self.order_timestamps[symbol].append(last_trade_time)

                logging.info(f"Updated order timestamps for {symbol} with last trade at {last_trade_time}")

            except Exception as e:
                logging.error(f"Exception occurred while processing trades for {symbol}: {e}")

    def _get_symbols(self):
        current_time = time.time()
        if Exchange.symbols_cache and (current_time - Exchange.symbols_cache_time) < Exchange.symbols_cache_duration:
            logging.info("Returning cached symbols")
            return Exchange.symbols_cache

        while True:
            try:
                markets = self.exchange.load_markets()
                symbols = [market['symbol'] for market in markets.values()]
                Exchange.symbols_cache = symbols
                Exchange.symbols_cache_time = current_time
                return symbols
            except ccxt.errors.RateLimitExceeded as e:
                logging.info(f"Get symbols Rate limit exceeded: {e}, retrying in 10 seconds...")
                time.sleep(10)
            except Exception as e:
                logging.info(f"An error occurred while fetching symbols: {e}, retrying in 10 seconds...")
                time.sleep(10)

    # def _get_symbols(self):
    #     current_time = time.time()
    #     if self.symbols_cache and (current_time - self.symbols_cache_time) < self.cache_duration:
    #         logging.info("Returning cached symbols")
    #         return self.symbols_cache

    #     while True:
    #         try:
    #             #self.exchange.set_sandbox_mode(True)
    #             markets = self.exchange.load_markets()
    #             symbols = [market['symbol'] for market in markets.values()]
    #             self.symbols_cache = symbols
    #             self.symbols_cache_time = current_time
    #             logging.info(f"Get symbols accessed")
    #             return symbols
    #         except RateLimitExceeded as e:
    #             logging.info(f"Get symbols Rate limit exceeded: {e}, retrying in 10 seconds...")
    #             time.sleep(10)
    #         except Exception as e:
    #             logging.info(f"An error occurred while fetching symbols: {e}, retrying in 10 seconds...")
    #             time.sleep(10)

    # def _get_symbols(self):
    #     while True:
    #         try:
    #             #self.exchange.set_sandbox_mode(True)
    #             markets = self.exchange.load_markets()
    #             symbols = [market['symbol'] for market in markets.values()]
    #             return symbols
    #         except ccxt.errors.RateLimitExceeded as e:
    #             logging.info(f"Get symbols Rate limit exceeded: {e}, retrying in 10 seconds...")
    #             time.sleep(10)
    #         except Exception as e:
    #             logging.info(f"An error occurred while fetching symbols: {e}, retrying in 10 seconds...")
    #             time.sleep(10)

    def get_ohlc_data(self, symbol, timeframe='1H', since=None, limit=None):
        """
        Fetches historical OHLC data for the given symbol and timeframe using ccxt's fetch_ohlcv method.
        
        :param str symbol: Symbol of the market to fetch OHLCV data for.
        :param str timeframe: The length of time each candle represents.
        :param int since: Timestamp in ms of the earliest candle to fetch.
        :param int limit: The maximum amount of candles to fetch.
        
        :return: List of OHLCV data.
        """
        ohlc_data = self.fetch_ohlcv(symbol, timeframe, since, limit)
        
        # Parsing the data to a more friendly format (optional)
        parsed_data = []
        for entry in ohlc_data:
            timestamp, open_price, high, low, close_price, volume = entry
            parsed_data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })

        return parsed_data

    def calculate_max_trade_quantity(self, symbol, leverage, wallet_exposure, best_ask_price):
        # Fetch necessary data from the exchange
        market_data = self.get_market_data_bybit(symbol)
        dex_equity = self.get_balance_bybit('USDT')

        # Calculate the max trade quantity based on leverage and equity
        max_trade_qty = round(
            (float(dex_equity) * wallet_exposure / float(best_ask_price))
            / (100 / leverage),
            int(float(market_data['min_qty'])),
        )

        return max_trade_qty
    
    def debug_derivatives_positions(self, symbol):
        try:
            positions = self.exchange.fetch_derivatives_positions([symbol])
            logging.info(f"Debug positions: {positions}")
        except Exception as e:
            logging.info(f"Exception in debug derivs func: {e}")

    def debug_derivatives_markets_bybit(self):
        try:
            markets = self.exchange.fetch_derivatives_markets({'category': 'linear'})
            logging.info(f"Debug markets: {markets}")
        except Exception as e:
            logging.info(f"Exception in debug_derivatives_markets_bybit: {e}")

    def parse_trading_fee(self, fee_data):
        maker_fee = float(fee_data.get('makerFeeRate', '0'))
        taker_fee = float(fee_data.get('takerFeeRate', '0'))
        return {
            'maker_fee': maker_fee,
            'taker_fee': taker_fee
        }


    def debug_binance_market_data(self, symbol: str) -> dict:
        try:
            self.exchange.load_markets()
            symbol_data = self.exchange.market(symbol)
            print(symbol_data)
        except Exception as e:
            logging.info(f"Error occurred in debug_binance_market_data: {e}")
        
    def fetch_trades(self, symbol: str, since: int = None, limit: int = None, params={}):
        """
        Get the list of most recent trades for a particular symbol.
        :param str symbol: Unified symbol of the market to fetch trades for.
        :param int since: Timestamp in ms of the earliest trade to fetch.
        :param int limit: The maximum amount of trades to fetch.
        :param dict params: Extra parameters specific to the Bybit API endpoint.
        :returns: A list of trade structures.
        """
        try:
            return self.exchange.fetch_trades(symbol, since=since, limit=limit, params=params)
        except Exception as e:
            logging.error(f"Error fetching trades for {symbol}: {e}")
            return []

    def retry_api_call(self, function, *args, max_retries=100, delay=10, **kwargs):
        for i in range(max_retries):
            try:
                return function(*args, **kwargs)
            except Exception as e:
                logging.info(f"Error occurred during API call: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
        raise Exception(f"Failed to execute the API function after {max_retries} retries.")
    
    def get_price_precision(self, symbol):
        market = self.exchange.market(symbol)
        smallest_increment = market['precision']['price']
        price_precision = len(str(smallest_increment).split('.')[-1])
        return price_precision

    def get_precision_bybit(self, symbol):
        markets = self.exchange.fetch_derivatives_markets()
        for market in markets:
            if market['symbol'] == symbol:
                return market['precision']
        return None

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
            logging.info(f"An unknown error occurred in get_balance(): {e}")
        return values

    def is_valid_symbol(self, symbol: str) -> bool:
        try:
            markets = self.exchange.load_markets()
            return symbol in markets
        except Exception as e:
            logging.error(f"Error checking symbol validity: {e}")
            logging.error(traceback.format_exc())
            return False

    def fetch_ohlcv(self, symbol, timeframe='1d', limit=None, max_retries=100, base_delay=10, max_delay=60):
        """
        Fetch OHLCV data for the given symbol and timeframe.
        
        :param symbol: Trading symbol.
        :param timeframe: Timeframe string.
        :param limit: Limit the number of returned data points.
        :param max_retries: Maximum number of retries for API calls.
        :param base_delay: Base delay for exponential backoff.
        :param max_delay: Maximum delay for exponential backoff.
        :return: DataFrame with OHLCV data.
        """
        retries = 0

        while retries < max_retries:
            try:
                with self.rate_limiter:
                    # Fetch the OHLCV data from the exchange
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # Pass the limit parameter
                    
                    # Create a DataFrame from the OHLCV data
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Convert the timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Set the timestamp as the index
                    df.set_index('timestamp', inplace=True)
                    
                    return df

            except ccxt.RateLimitExceeded as e:
                retries += 1
                delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
                logging.info(f"Rate limit exceeded: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

            except ccxt.BaseError as e:
                # Log the error message
                logging.info(f"Failed to fetch OHLCV data: {self.exchange.id} {e}")
                # Log the traceback for further debugging
                logging.error(traceback.format_exc())
                return pd.DataFrame()

            except Exception as e:
                # Log the error message and traceback
                logging.info(f"Unexpected error occurred while fetching OHLCV data: {e}")
                logging.info(traceback.format_exc())
                
                # Attempt to handle the specific 'string indices must be integers' error
                if isinstance(e, TypeError) and 'string indices must be integers' in str(e):
                    logging.info(f"TypeError occurred: {e}")
                    
                    # Print the response for debugging
                    logging.info(f"Response content: {self.exchange.last_http_response}")
                    
                    try:
                        # Attempt to parse the response
                        response = json.loads(self.exchange.last_http_response)
                        logging.info(f"Parsed response into a dictionary: {response}")
                    except json.JSONDecodeError as json_error:
                        logging.info(f"Failed to parse response: {json_error}")
                
                return pd.DataFrame()

        logging.error(f"Failed to fetch OHLCV data after {max_retries} retries.")
        return pd.DataFrame()

    # def fetch_ohlcv(self, symbol, timeframe='1d', limit=None, max_retries=100, base_delay=10, max_delay=60):
    #     """
    #     Fetch OHLCV data for the given symbol and timeframe.
        
    #     :param symbol: Trading symbol.
    #     :param timeframe: Timeframe string.
    #     :param limit: Limit the number of returned data points.
    #     :param max_retries: Maximum number of retries for API calls.
    #     :param base_delay: Base delay for exponential backoff.
    #     :param max_delay: Maximum delay for exponential backoff.
    #     :return: DataFrame with OHLCV data.
    #     """
    #     retries = 0

    #     while retries < max_retries:
    #         try:
    #             with self.rate_limiter:
    #                 # Fetch the OHLCV data from the exchange
    #                 ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # Pass the limit parameter
                    
    #                 # Create a DataFrame from the OHLCV data
    #                 df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
    #                 # Convert the timestamp to datetime
    #                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
    #                 # Set the timestamp as the index
    #                 df.set_index('timestamp', inplace=True)
                    
    #                 return df

    #         except ccxt.RateLimitExceeded as e:
    #             retries += 1
    #             delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
    #             logging.info(f"Rate limit exceeded: {e}. Retrying in {delay:.2f} seconds...")
    #             time.sleep(delay)

    #         except ccxt.BaseError as e:
    #             # Log the error message
    #             logging.info(f"Failed to fetch OHLCV data: {self.exchange.id} {e}")
    #             # Log the traceback for further debugging
    #             logging.error(traceback.format_exc())
    #             return pd.DataFrame()

    #         except Exception as e:
    #             # Log the error message and traceback
    #             logging.info(f"Unexpected error occurred while fetching OHLCV data: {e}")
    #             logging.info(traceback.format_exc())
                
    #             # Attempt to handle the specific 'string indices must be integers' error
    #             if isinstance(e, TypeError) and 'string indices must be integers' in str(e):
    #                 logging.info(f"TypeError occurred: {e}")
                    
    #                 # Print the response for debugging
    #                 logging.info(f"Response content: {self.exchange.last_http_response}")
                    
    #                 try:
    #                     # Attempt to parse the response
    #                     response = json.loads(self.exchange.last_http_response)
    #                     logging.info(f"Parsed response into a dictionary: {response}")
    #                 except json.JSONDecodeError as json_error:
    #                     logging.info(f"Failed to parse response: {json_error}")
                
    #             return pd.DataFrame()

    #     raise Exception(f"Failed to fetch OHLCV data after {max_retries} retries.")
              
    # def fetch_ohlcv(self, symbol, timeframe='1d', limit=None, max_retries=100, base_delay=10, max_delay=60):
    #     """
    #     Fetch OHLCV data for the given symbol and timeframe.
        
    #     :param symbol: Trading symbol.
    #     :param timeframe: Timeframe string.
    #     :param limit: Limit the number of returned data points.
    #     :param max_retries: Maximum number of retries for API calls.
    #     :param base_delay: Base delay for exponential backoff.
    #     :param max_delay: Maximum delay for exponential backoff.
    #     :return: DataFrame with OHLCV data.
    #     """
    #     retries = 0

    #     while retries < max_retries:
    #         try:
    #             with self.rate_limiter:
    #                 # Fetch the OHLCV data from the exchange
    #                 ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # Pass the limit parameter
                    
    #                 # Create a DataFrame from the OHLCV data
    #                 df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
    #                 # Convert the timestamp to datetime
    #                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
    #                 # Set the timestamp as the index
    #                 df.set_index('timestamp', inplace=True)
                    
    #                 return df

    #         except ccxt.RateLimitExceeded as e:
    #             retries += 1
    #             delay = min(base_delay * (2 ** retries) + random.uniform(0, 0.1 * (2 ** retries)), max_delay)
    #             logging.info(f"Rate limit exceeded: {e}. Retrying in {delay:.2f} seconds...")
    #             time.sleep(delay)

    #         except ccxt.BaseError as e:
    #             # Log the error message
    #             logging.info(f"Failed to fetch OHLCV data: {self.exchange.id} {e}")
    #             # Log the traceback for further debugging
    #             logging.error(traceback.format_exc())
    #             return pd.DataFrame()

    #         except Exception as e:
    #             # Check if the error is related to response parsing
    #             if 'response' in locals() and isinstance(response, str):
    #                 logging.info(f"Response is a string: {response}")
    #                 try:
    #                     # Attempt to parse the response
    #                     response = json.loads(response)
    #                     logging.info("Parsed response into a dictionary")
    #                 except json.JSONDecodeError as json_error:
    #                     logging.info(f"Failed to parse response: {json_error}")
                
    #             # Log any other unexpected errors
    #             logging.info(f"Unexpected error occurred while fetching OHLCV data: {e}")
    #             logging.info(traceback.format_exc())
    #             return pd.DataFrame()

    #     raise Exception(f"Failed to fetch OHLCV data after {max_retries} retries.")
    
    # def fetch_ohlcv(self, symbol, timeframe='1d', limit=None):
    #     """
    #     Fetch OHLCV data for the given symbol and timeframe.
        
    #     :param symbol: Trading symbol.
    #     :param timeframe: Timeframe string.
    #     :param limit: Limit the number of returned data points.
    #     :return: DataFrame with OHLCV data.
    #     """
    #     try:
    #         # Fetch the OHLCV data from the exchange
    #         ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # Pass the limit parameter
            
    #         # Create a DataFrame from the OHLCV data
    #         df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
    #         # Convert the timestamp to datetime
    #         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
    #         # Set the timestamp as the index
    #         df.set_index('timestamp', inplace=True)
            
    #         return df
        
    #     except ccxt.BaseError as e:
    #         # Log the error message
    #         logging.error(f"Failed to fetch OHLCV data: {self.exchange.id} {e}")
            
    #         # Log the traceback for further debugging
    #         logging.error(traceback.format_exc())
            
    #         # Return an empty DataFrame in case of an error
    #         return pd.DataFrame()
        
    #     except Exception as e:
    #         # Check if the error is related to response parsing
    #         if 'response' in locals() and isinstance(response, str):
    #             logging.info(f"Response is a string: {response}")
    #             try:
    #                 # Attempt to parse the response
    #                 response = json.loads(response)
    #                 logging.info("Parsed response into a dictionary")
    #             except json.JSONDecodeError as json_error:
    #                 logging.error(f"Failed to parse response: {json_error}")
            
    #         # Log any other unexpected errors
    #         logging.info(f"Unexpected error occurred while fetching OHLCV data: {e}")
    #         logging.info(traceback.format_exc())
            
    #         return pd.DataFrame()


    # def fetch_ohlcv(self, symbol, timeframe='1d', limit=None):
    #     """
    #     Fetch OHLCV data for the given symbol and timeframe.
        
    #     :param symbol: Trading symbol.
    #     :param timeframe: Timeframe string.
    #     :param limit: Limit the number of returned data points.
    #     :return: DataFrame with OHLCV data.
    #     """
    #     try:
    #         # Fetch the OHLCV data from the exchange
    #         ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)  # Pass the limit parameter
            
    #         # Create a DataFrame from the OHLCV data
    #         df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
    #         # Convert the timestamp to datetime
    #         df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
    #         # Set the timestamp as the index
    #         df.set_index('timestamp', inplace=True)
            
    #         return df
        
    #     except ccxt.BaseError as e:
    #         # Log the error message
    #         logging.error(f"Failed to fetch OHLCV data: {self.exchange.id} {e}")
            
    #         # Log the traceback for further debugging
    #         logging.error(traceback.format_exc())
            
    #         # Return an empty DataFrame in case of an error
    #         return pd.DataFrame()
        
    #     except Exception as e:
    #         # Log any other unexpected errors
    #         logging.error(f"Unexpected error occurred while fetching OHLCV data: {e}")
    #         logging.error(traceback.format_exc())
            
    #         return pd.DataFrame()

    def get_orderbook(self, symbol, max_retries=3, retry_delay=5) -> dict:
        values = {"bids": [], "asks": []}

        for attempt in range(max_retries):
            try:
                data = self.exchange.fetch_order_book(symbol)
                if "bids" in data and "asks" in data:
                    if len(data["bids"]) > 0 and len(data["asks"]) > 0:
                        if len(data["bids"][0]) > 0 and len(data["asks"][0]) > 0:
                            values["bids"] = data["bids"]
                            values["asks"] = data["asks"]
                break  # if the fetch was successful, break out of the loop

            except HTTPError as http_err:
                print(f"HTTP error occurred: {http_err} - {http_err.response.text}")

                if "Too many visits" in str(http_err) or (http_err.response.status_code == 429):
                    if attempt < max_retries - 1:
                        delay = retry_delay * (attempt + 1)  # Variable delay
                        logging.info(f"Rate limit error in get_orderbook(). Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                else:
                    logging.error(f"HTTP error in get_orderbook(): {http_err.response.text}")
                    raise http_err

            except Exception as e:
                if attempt < max_retries - 1:  # if not the last attempt
                    logging.info(f"An unknown error occurred in get_orderbook(): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to fetch order book after {max_retries} attempts: {e}")
                    raise e  # If it's still failing after max_retries, re-raise the exception.

        return values
    
    # Huobi debug
    def get_positions_debug(self):
        try:
            positions = self.exchange.fetch_positions()
            print(f"{positions}")
        except Exception as e:
            print(f"Error is {e}")

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
            logging.info(f"An unknown error occurred in get_positions(): {e}")
        return values

    def get_current_price(self, symbol: str) -> float:
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logging.info(f"Fetched ticker for {symbol}: {ticker}")

            if "bid" in ticker and "ask" in ticker:
                bid = ticker["bid"]
                ask = ticker["ask"]
                
                # Convert bid and ask to float if they are strings
                if isinstance(bid, str):
                    bid = float(bid)
                if isinstance(ask, str):
                    ask = float(ask)
                
                # Check if bid and ask are numeric
                if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
                    return (bid + ask) / 2
                else:
                    raise TypeError(f"Bid or ask price is not numeric: bid={bid}, ask={ask}")
            else:
                raise KeyError(f"Ticker does not contain 'bid' or 'ask': {ticker}")
        except Exception as e:
            logging.info(f"An error occurred in get_current_price() for {symbol}: {e}")
            logging.info(traceback.format_exc())
            return None

    # Universal
    # def get_current_price(self, symbol: str) -> float:
    #     try:
    #         ticker = self.exchange.fetch_ticker(symbol)
    #         if "bid" in ticker and "ask" in ticker:
    #             return (ticker["bid"] + ticker["ask"]) / 2
    #     except Exception as e:
    #         logging.error(f"An error occurred in get_current_price() for {symbol}: {e}")
    #         return None
        
    # Binance
    def get_current_price_binance(self, symbol: str) -> float:
        current_price = 0.0
        try:
            orderbook = self.exchange.fetch_order_book(symbol)
            highest_bid = orderbook['bids'][0][0] if len(orderbook['bids']) > 0 else None
            lowest_ask = orderbook['asks'][0][0] if len(orderbook['asks']) > 0 else None
            if highest_bid and lowest_ask:
                current_price = (highest_bid + lowest_ask) / 2
        except Exception as e:
            logging.info(f"An unknown error occurred in get_current_price_binance(): {e}")
        return current_price

    # def get_leverage_tiers_binance(self, symbols: Optional[List[str]] = None):
    #     try:
    #         tiers = self.exchange.fetch_leverage_tiers(symbols)
    #         for symbol, brackets in tiers.items():
    #             print(f"\nSymbol: {symbol}")
    #             for bracket in brackets:
    #                 print(f"Bracket ID: {bracket['bracket']}")
    #                 print(f"Initial Leverage: {bracket['initialLeverage']}")
    #                 print(f"Notional Cap: {bracket['notionalCap']}")
    #                 print(f"Notional Floor: {bracket['notionalFloor']}")
    #                 print(f"Maintenance Margin Ratio: {bracket['maintMarginRatio']}")
    #                 print(f"Cumulative: {bracket['cum']}")
    #     except Exception as e:
    #         logging.error(f"An error occurred while fetching leverage tiers: {e}")

    def get_symbol_info_binance(self, symbol):
        try:
            markets = self.exchange.fetch_markets()
            print(markets)
            for market in markets:
                if market['symbol'] == symbol:
                    filters = market['info']['filters']
                    min_notional = [f['minNotional'] for f in filters if f['filterType'] == 'MIN_NOTIONAL'][0]
                    min_qty = [f['minQty'] for f in filters if f['filterType'] == 'LOT_SIZE'][0]
                    return min_notional, min_qty
        except Exception as e:
            logging.error(f"An error occurred while fetching symbol info: {e}")

    # def get_market_data_binance(self, symbol):
    #     market_data = self.exchange.load_markets(reload=True)  # Force a reload to get fresh data
    #     return market_data[symbol]

    # def get_market_data_binance(self, symbol):
    #     market_data = self.exchange.load_markets(reload=True)  # Force a reload to get fresh data
    #     print("Symbols:", market_data.keys())  # Print out all available symbols
    #     return market_data[symbol]

    def get_min_lot_size_binance(self, symbol):
        market_data = self.get_market_data_binance(symbol)

        # Extract the filters from the market data
        filters = market_data['info']['filters']

        # Find the 'LOT_SIZE' filter and get its 'minQty' value
        for f in filters:
            if f['filterType'] == 'LOT_SIZE':
                return float(f['minQty'])

        # If no 'LOT_SIZE' filter was found, return None
        return None

    def get_moving_averages(self, symbol: str, timeframe: str = "1m", num_bars: int = 20, max_retries=100, retry_delay=5) -> dict:
        values = {"MA_3_H": 0.0, "MA_3_L": 0.0, "MA_6_H": 0.0, "MA_6_L": 0.0}
        for i in range(max_retries):
            try:
                bars = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=num_bars)
                if not bars:
                    logging.info(f"No data returned for {symbol} on {timeframe}. Retrying...")
                    time.sleep(retry_delay)
                    continue
                
                df = pd.DataFrame(bars, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
                df["Time"] = pd.to_datetime(df["Time"], unit="ms")
                df["MA_3_High"] = df["High"].rolling(3).mean()
                df["MA_3_Low"] = df["Low"].rolling(3).mean()
                df["MA_6_High"] = df["High"].rolling(6).mean()
                df["MA_6_Low"] = df["Low"].rolling(6).mean()
                
                values["MA_3_H"] = df["MA_3_High"].iat[-1] if len(df["MA_3_High"]) > 0 else None
                values["MA_3_L"] = df["MA_3_Low"].iat[-1] if len(df["MA_3_Low"]) > 0 else None
                values["MA_6_H"] = df["MA_6_High"].iat[-1] if len(df["MA_6_High"]) > 0 else None
                values["MA_6_L"] = df["MA_6_Low"].iat[-1] if len(df["MA_6_Low"]) > 0 else None

                if None not in values.values():
                    break
            except Exception as e:
                if i < max_retries - 1:
                    logging.info(f"An unknown error occurred in get_moving_averages(): {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.info(f"Failed to fetch moving averages after {max_retries} attempts: {e}")
                    return values  # Return whatever we have, even if incomplete

        return values

    def get_order_status(self, order_id, symbol):
        try:
            # Fetch the order details from the exchange using the order ID
            order_details = self.fetch_order(order_id, symbol)

            logging.info(f"Order details for {symbol}: {order_details}")

            # Extract and return the order status
            return order_details['status']
        except Exception as e:
            logging.error(f"An error occurred fetching order status for {order_id} on {symbol}: {e}")
            return None


    def get_open_orders(self, symbol: str) -> list:
        open_orders_list = []
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            #print(orders)
            if len(orders) > 0:
                for order in orders:
                    if "info" in order:
                        #print(f"Order info: {order['info']}")  # Debug print
                        order_info = {
                            "id": order["info"]["orderId"],
                            "price": float(order["info"]["price"]),
                            "qty": float(order["info"]["qty"]),
                            "order_status": order["info"]["orderStatus"],
                            "side": order["info"]["side"],
                            "reduce_only": order["info"]["reduceOnly"],  # Update this line
                            "position_idx": int(order["info"]["positionIdx"])  # Add this line
                        }
                        open_orders_list.append(order_info)
        except Exception as e:
            logging.info(f"Bybit An unknown error occurred in get_open_orders(): {e}")
        return open_orders_list

    def cancel_all_entries_binance(self, symbol: str):
        try:
            # Fetch all open orders
            open_orders = self.get_open_orders_binance(symbol)

            for order in open_orders:
                # If the order is a 'LIMIT' order (i.e., an 'entry' order), cancel it
                if order['type'].upper() == 'LIMIT':
                    self.exchange.cancel_order(order['id'], symbol)
        except Exception as e:
            print(f"An error occurred while canceling entry orders: {e}")

    def debug_open_orders(self, symbol: str) -> None:
        try:
            open_orders = self.exchange.fetch_open_orders(symbol)
            logging.info(open_orders)
        except:
            logging.info(f"Fuck")

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
                        logging.info(f"Cancelling {order_side} order: {order_id}")
        except Exception as e:
            logging.info(f"An unknown error occurred in _cancel_entry(): {e}")

    def cancel_all_open_orders_bybit(self, derivatives: bool = False, params={}):
        """
        Cancel all open orders for all symbols.

        :param bool derivatives: Whether to cancel derivative orders.
        :param dict params: Additional parameters for the API call.
        :return: A list of canceled orders.
        """
        max_retries = 10  # Maximum number of retries
        retry_delay = 5  # Delay (in seconds) between retries

        for retry in range(max_retries):
            try:
                if derivatives:
                    return self.exchange.cancel_all_derivatives_orders(None, params)
                else:
                    return self.exchange.cancel_all_orders(None, params)
            except ccxt.RateLimitExceeded as e:
                # If rate limit error and not the last retry, then wait and try again
                if retry < max_retries - 1:
                    logging.info(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:  # If it's the last retry, raise the error
                    logging.error(f"Rate limit exceeded after {max_retries} retries.")
                    raise e
            except Exception as ex:
                # If any other exception, log it and re-raise
                logging.error(f"Error occurred while canceling orders: {ex}")
                raise ex

    def health_check(self, interval_seconds=300):
        """
        Periodically checks the health of the exchange and cancels all open orders.

        :param interval_seconds: The time interval in seconds between each health check.
        """
        while True:
            try:
                logging.info("Performing health check...")  # Log start of health check
                # You can add more health check logic here
                
                # Cancel all open orders
                self.cancel_all_open_orders_bybit()
                
                logging.info("Health check complete.")  # Log end of health check
            except Exception as e:
                logging.error(f"An error occurred during the health check: {e}")  # Log any errors
                
            time.sleep(interval_seconds)

    # def cancel_all_auto_reduce_orders_bybit(self, symbol: str, auto_reduce_order_ids: List[str]):
    #     try:
    #         orders = self.fetch_open_orders(symbol)
    #         logging.info(f"[Thread ID: {threading.get_ident()}] cancel_all_auto_reduce_orders function in exchange class accessed")
    #         logging.info(f"Fetched orders: {orders}")

    #         for order in orders:
    #             if order['status'] in ['open', 'partially_filled']:
    #                 order_id = order['id']
    #                 # Check if the order ID is in the list of auto-reduce orders
    #                 if order_id in auto_reduce_order_ids:
    #                     self.cancel_order(order_id, symbol)
    #                     logging.info(f"Cancelling auto-reduce order: {order_id}")

    #     except Exception as e:
    #         logging.warning(f"An unknown error occurred in cancel_all_auto_reduce_orders_bybit(): {e}")

    #v5 
    def cancel_all_reduce_only_orders_bybit(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            logging.info(f"[Thread ID: {threading.get_ident()}] cancel_all_reduce_only_orders_bybit function accessed")
            logging.info(f"Fetched orders: {orders}")

            for order in orders:
                if order['status'] in ['open', 'partially_filled']:
                    # Check if the order is a reduce-only order
                    if order['reduceOnly']:
                        order_id = order['id']
                        self.exchange.cancel_order(order_id, symbol)
                        logging.info(f"Cancelling reduce-only order: {order_id}")

        except Exception as e:
            logging.info(f"An error occurred in cancel_all_reduce_only_orders_bybit(): {e}")

    # v5
    def cancel_all_entries_bybit(self, symbol: str) -> None:
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            logging.info(f"[Thread ID: {threading.get_ident()}] cancel_all_entries function in exchange class accessed")
            logging.info(f"Fetched orders: {orders}")

            for order in orders:
                if order['status'] in ['open', 'partially_filled']:
                    # Check if the order is not a reduce-only order
                    if not order['reduceOnly']:
                        order_id = order['id']
                        self.exchange.cancel_order(order_id, symbol)
                        logging.info(f"Cancelling order: {order_id}")

        except Exception as e:
            logging.info(f"An unknown error occurred in cancel_all_entries_bybit(): {e}")

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
                        logging.info(f"Cancelling order: {order_id}")
                    elif (
                        order_status != "Filled"
                        and order_side == "Sell"
                        and order_status != "Cancelled"
                        and not reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.info(f"An unknown error occurred in cancel_entry(): {e}")
    
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
                        logging.info(f"Cancelling order: {order_id}")
                    elif (
                        order_status != "Filled"
                        and order_side == "Sell"
                        and side == "short"
                        and order_status != "Cancelled"
                        and reduce_only
                    ):
                        self.exchange.cancel_order(symbol=symbol, id=order_id)
                        logging.info(f"Cancelling order: {order_id}")
        except Exception as e:
            logging.info(f"{e}")

    def create_take_profit_order(self, symbol, order_type, side, amount, price=None, reduce_only=False):
        if order_type == 'limit':
            if price is None:
                raise ValueError("A price must be specified for a limit order")

            if side not in ["buy", "sell"]:
                raise ValueError(f"Invalid side: {side}")

            params = {"reduceOnly": reduce_only}
            return self.exchange.create_order(symbol, order_type, side, amount, price, params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def create_market_order(self, symbol: str, side: str, amount: float, params={}, close_position: bool = False) -> None:
        try:
            if side not in ["buy", "sell"]:
                logging.info(f"side {side} does not exist")
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
            logging.info(f"An unknown error occurred in create_market_order(): {e}")

    def test_func(self):
        try:
            market = self.exchange.market('DOGEUSDT')
            print(market['info'])
        except Exception as e:
            print(f"Exception caught in test func {e}")

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

        if self.exchange_id == 'bybit':
            order = self.exchange.create_order(symbol, order_type, side, amount, price, params=params)
        else:
            if order_type == 'limit':
                if side == "buy":
                    order = self.create_limit_buy_order(symbol, amount, price, **params)
                elif side == "sell":
                    order = self.create_limit_sell_order(symbol, amount, price, **params)
                else:
                    raise ValueError(f"Invalid side: {side}")
            elif order_type == 'market':
                # Special handling for market orders
                order = self.exchange.create_order(symbol, 'market', side, amount, None, params=params)
            else:
                raise ValueError("Invalid order type. Use 'limit' or 'market'.")

        return order

    # def create_order(self, symbol, order_type, side, amount, price=None, reduce_only=False, **params):
    #     if reduce_only:
    #         params.update({'reduceOnly': 'true'})

    #     if self.exchange_id == 'bybit':
    #         order = self.exchange.create_order(symbol, order_type, side, amount, price, params=params)
    #     else:
    #         if order_type == 'limit':
    #             if side == "buy":
    #                 order = self.create_limit_buy_order(symbol, amount, price, **params)
    #             elif side == "sell":
    #                 order = self.create_limit_sell_order(symbol, amount, price, **params)
    #             else:
    #                 raise ValueError(f"Invalid side: {side}")
    #         elif order_type == 'market':
    #             #... handle market orders if necessary
    #             order = self.create_market_order(symbol, side, amount, params)
    #         else:
    #             raise ValueError("Invalid order type. Use 'limit' or 'market'.")

    #     return order


    # def get_symbol_precision_bybit(self, symbol: str) -> Tuple[int, int]:
    #     try:
    #         market = self.exchange.market(symbol)
    #         price_precision = int(market['precision']['price'])
    #         quantity_precision = int(market['precision']['amount'])
    #         return price_precision, quantity_precision
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return None, None

    # def get_symbol_precision_bybit(self, symbol: str) -> Tuple[int, int]:
    #     market = self.exchange.market(symbol)
    #     price_precision = int(market['precision']['price'])
    #     quantity_precision = int(market['precision']['amount'])
    #     return price_precision, quantity_precision

    # def _get_symbols(self):
    #     markets = self.exchange.load_markets()
    #     symbols = [market['symbol'] for market in markets.values()]
    #     return symbols