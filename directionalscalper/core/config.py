import json
from enum import Enum

from pydantic import BaseModel, ValidationError, validator


class Exchanges(Enum):
    BYBIT = "bybit"


class API(BaseModel):
    filename: str = "quantdatav2.json"
    mode: str = "remote"
    url: str = "http://api.tradesimple.xyz/data/"


class Bot(BaseModel):
    avoid_fees: bool = False
    bot_name: str
    divider: int = 7
    inverse_direction: str = "short"
    linear_taker_fee: float = 0.17
    min_distance: float = 0.15
    min_fee: float = 0.17
    min_volume: int = 15000
    profit_multiplier_pct: float = 0.01
    symbol: str
    violent_multiplier: float = 2.00
    wallet_exposure: float = 1.00

    @validator("profit_multiplier_pct")
    def minimum_profit_multiplier_pct(cls, v):
        if v < 0.0:
            raise ValueError("profit_multiplier_pct must be greater than 0.0")
        return v

    @validator("min_volume")
    def minimum_min_volume(cls, v):
        if v < 0.0:
            raise ValueError("min_volume must be greater than 0")
        return v

    @validator("min_distance")
    def minimum_min_distance(cls, v):
        if v < 0.0:
            raise ValueError("min_distance must be greater than 0")
        return v

    @validator("min_fee")
    def minimum_min_fee(cls, v):
        if v < 0.0:
            raise ValueError("min_fee must be greater than 0")
        return v

    @validator("divider")
    def minimum_divider(cls, v):
        if v < 0:
            raise ValueError("divider must be greater than 0")
        return v

    @validator("linear_taker_fee")
    def minimum_linear_taker_fee(cls, v):
        if v < 0.0:
            raise ValueError("linear_taker_fee must be greater than 0")
        return v


class Exchange(BaseModel):
    name: str = Exchanges.BYBIT.value  # type: ignore
    api_key: str = ""
    api_secret: str = ""


class Logger(BaseModel):
    level: str = "info"

    @validator("level")
    def check_level(cls, v):
        levels = ["notset", "debug", "info", "warn", "error", "critical"]
        if v not in levels:
            raise ValueError(f"Log level must be in {levels}")
        return v


class Telegram(BaseModel):
    api_token: str = ""
    chat_id: str = ""


class Config(BaseModel):
    api = API
    exchange = Exchange
    logger = Logger
    bot = Bot


def load_config(path):
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    else:
        f = open(path)
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"ERROR: Invalid JSON: {exc.msg}, line {exc.lineno}, column {exc.colno}"
            )
        try:
            return Config(**data)
        except ValidationError as e:
            raise ValueError(f"{e}")
