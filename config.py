from __future__ import annotations
from typing import List


import json
from enum import Enum
from typing import Union

from pydantic import BaseModel, HttpUrl, ValidationError, validator


class Exchanges(Enum):
    BYBIT = "bybit"


class Messengers(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"


class API(BaseModel):
    filename: str = "quantdatav2.json"
    mode: str = "remote"
    url: str = "http://api.tradesimple.xyz/data/"


class Bot(BaseModel):
    bot_name: str
    min_distance: float = 0.15
    min_distance_btc: float = 0.03
    min_fee: float = 0.17
    min_volume: int = 15000
    symbol: str
    violent_multiplier: float = 2.00
    wallet_exposure: float = 1.00

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

    # @validator("min_distance_btc")
    # def minimum_min_distance(cls, v):
    #     if v < 0.0:
    #         raise ValueError("min_distance must be greater than 0")
    #     return v

class Exchange(BaseModel):
    name: str
    api_key: str
    api_secret: str
    passphrase: str = None


# class Exchange(BaseModel):
#     name: str = Exchanges.BYBIT.value  # type: ignore
#     api_key: str = ""
#     api_secret: str = ""


class Logger(BaseModel):
    level: str = "info"

    @validator("level")
    def check_level(cls, v):
        levels = ["notset", "debug", "info", "warn", "error", "critical"]
        if v not in levels:
            raise ValueError(f"Log level must be in {levels}")
        return v

class Discord(BaseModel):
    active: bool = False
    embedded_messages: bool = True
    messenger_type: str = Messengers.DISCORD.value  # type: ignore
    webhook_url: HttpUrl

    @validator("webhook_url")
    def minimum_divider(cls, v):
        if not str(v).startswith("https://discord.com/api/webhooks/"):
            raise ValueError(
                "Discord webhook begins: https://discord.com/api/webhooks/"
            )
        return v

# class Discord(BaseModel):
#     active: bool = False
#     embedded_messages: bool = True
#     messenger_type: str = Messengers.DISCORD.value  # type: ignore
#     webhook_url: HttpUrl

#     @validator("webhook_url")
#     def minimum_divider(cls, v):
#         if not v.startswith("https://discord.com/api/webhooks/"):
#             raise ValueError(
#                 "Discord webhook begins: https://discord.com/api/webhooks/"
#             )
#         return v

class Telegram(BaseModel):
    active: bool = False
    embedded_messages: bool = True
    messenger_type: str = Messengers.TELEGRAM.value  # type: ignore
    bot_token: str
    chat_id: str


class Config(BaseModel):
    api: API
    bot: Bot
    exchanges: List[Exchange]
    logger: Logger
    messengers: dict[str, Union[Discord, Telegram]]

# class Config(BaseModel):
#     api: API
#     bot: Bot
#     exchange: Exchange
#     logger: Logger
#     messengers: dict[str, Union[Discord, Telegram]]


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


def get_exchange_name(cli_exchange_name):
    if cli_exchange_name:
        return cli_exchange_name
    else:
        with open('config.json') as file:
            data = json.load(file)
            return data['exchanges'][0]['name']

def get_exchange_credentials(exchange_name):
    with open('config.json') as file:
        data = json.load(file)
        exchange_data = None
        for exchange in data['exchanges']:
            if exchange['name'] == exchange_name:
                exchange_data = exchange
                break
        if exchange_data:
            api_key = exchange_data['api_key']
            secret_key = exchange_data['api_secret']
            passphrase = exchange_data.get('passphrase')  # Not all exchanges require a passphrase
            return api_key, secret_key, passphrase
        else:
            raise ValueError(f"Exchange {exchange_name} not found in the config file.")


# def get_exchange_name(cli_exchange_name):
#     if cli_exchange_name:
#         return cli_exchange_name
#     else:
#         with open('config.json') as file:
#             data = json.load(file)
#             return data['exchange']

# def get_exchange_credentials(exchange_name):
#     with open('config.json') as file:
#         data = json.load(file)
#         exchange_data = data['exchanges'][exchange_name]
#         api_key = exchange_data['api_key']
#         secret_key = exchange_data['secret_key']
#         passphrase = exchange_data.get('passphrase')  # Not all exchanges require a passphrase
#         return api_key, secret_key, passphrase
