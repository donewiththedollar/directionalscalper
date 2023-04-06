import json
from enum import Enum

from pydantic import BaseModel, ValidationError, validator


class Exchanges(Enum):
    BYBIT = "bybit"


class Config(BaseModel):
    exchange: str = Exchanges.BYBIT.value  # type: ignore
    exchange_api_key: str
    exchange_api_secret: str
    min_volume: int = 15000
    min_distance: float = 0.15
    bot_name: str
    symbol: str
    min_fee: float = 0.17
    divider: int = 7
    telegram_api_token: str = ""
    telegram_chat_id: str = ""
    avoid_fees: bool = False
    linear_taker_fee: float = 0.17
    wallet_exposure: float = 1.00
    violent_multiplier: float = 2.00
    profit_multiplier_pct: float = 0.01
    inverse_direction: str = "short"


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
            raise ValueError("divier must be greater than 0")
        return v

    @validator("linear_taker_fee")
    def minimum_linear_taker_fee(cls, v):
        if v < 0.0:
            raise ValueError("linear_taker_fee must be greater than 0")
        return v

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
