from __future__ import annotations

import concurrent.futures
import logging
from decimal import Decimal
from logging import handlers

import pandas as pd
import ta
from exchanges.bybit import Bybit

log = logging.getLogger()
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
rotatingHandler = handlers.RotatingFileHandler(
    "scraper.log", maxBytes=5000000, backupCount=4
)
rotatingHandler.setFormatter(formatter)
log.setLevel(logging.INFO)
log.addHandler(rotatingHandler)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
log.addHandler(streamHandler)


class Scraper:
    def __init__(self, exchange):
        log.info("Scraper initalising")
        self.exchange = exchange
        self.symbols = self.exchange.get_futures_symbols()
        log.info(f"{len(self.symbols)} symbols found")

    def get_spread(self, symbol: str, limit: int, timeframe: str = "1m"):
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

        current_sma = Decimal(sma[limit - 1])

        last_close_price = bars[limit - 1]["close"]

        return round((last_close_price - current_sma) / last_close_price * 100, 4)

    def get_average_true_range(self, symbol: str, period, interval: str, limit: int):
        data = self.exchange.get_futures_kline(
            symbol=symbol, intveral=interval, limit=int
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

    def analyse_symbol(self, symbol: str) -> dict:
        log.info(f"Analysing: {symbol}")
        values = {"Asset": symbol}

        price = self.exchange.get_futures_price(symbol=symbol)
        values["Price"] = price

        candles_30m = self.exchange.get_futures_kline(
            symbol=symbol, interval="30m", limit=5
        )
        candles_5m = self.exchange.get_futures_kline(
            symbol=symbol, interval="5m", limit=5
        )
        candles_1m = self.exchange.get_futures_kline(
            symbol=symbol, interval="1m", limit=5
        )

        # Define candle spreads per timeframe
        values["1m Spread"] = self.get_spread(symbol=symbol, limit=1)
        values["5m Spread"] = self.get_spread(symbol=symbol, limit=5)
        values["30m Spread"] = self.get_spread(symbol=symbol, limit=30)

        # Define 1x 5m candle volume
        onexcandlevol = candles_5m[-1]["volume"]
        volume_1x_5m = price * onexcandlevol
        values["5m 1x Volume (USDT)"] = round(volume_1x_5m)

        # Define 1x 1m candle volume
        onex1mcandlevol = candles_1m[-1]["volume"]
        volume_1x = price * onex1mcandlevol
        values["1m 1x Volume (USDT)"] = round(volume_1x)

        # Define 1x 30m candle volume
        onex30mcandlevol = candles_30m[-1]["volume"]
        volume_1x_30m = price * onex30mcandlevol
        values["30m 1x Volume (USDT)"] = round(volume_1x_30m)

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

        if ma_order_pct.compare(0) == 1:
            values["Trend"] = "short"
        else:
            values["Trend"] = "long"

        # Define funding rates
        values["Funding"] = self.exchange.get_funding_rate(symbol=symbol) * 100

        return values

    def analyse_all_symbols(self):
        data = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            future_data = {
                executor.submit(self.analyse_symbol, symbol): symbol
                for symbol in self.symbols
            }
            for future in concurrent.futures.as_completed(future_data):
                symbol_data = future_data[future]
                try:
                    symbol_data_result = future.result()
                    data.append(symbol_data_result)
                except Exception as e:
                    log.error(f"{symbol_data} generated an exception: {e}")
                else:
                    log.error(f"{symbol_data} page is {symbol_data_result} bytes")

        df = pd.DataFrame(
            data,
            columns=[
                "Asset",
                "Price",
                "1m 1x Volume (USDT)",
                "5m 1x Volume (USDT)",
                "30m 1x Volume (USDT)",
                "1m Spread",
                "5m Spread",
                "30m Spread",
                "trend%",
                "Trend",
                "5m MA6 high",
                "5m MA6 low",
                "Funding",
            ],
        )

        df.sort_values(
            by=["1m 1x Volume (USDT)", "5m Spread"],
            inplace=True,
            ascending=[False, False],
        )
        return df

    def output_df(self, dataframe, path: str):
        dataframe.to_json(path, orient="records")


if __name__ == "__main__":
    exchange = Bybit()
    scraper = Scraper(exchange=exchange)
    data = scraper.analyse_all_symbols()
    print(data)
    # scraper.output_df(data, "/opt/bitnami/nginx/html/data/quantdata.json")
