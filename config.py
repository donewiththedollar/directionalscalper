from __future__ import annotations
from typing import List, Optional

import os
from pathlib import Path
import json
from enum import Enum
from typing import Union

from pydantic import BaseModel, HttpUrl, ValidationError, validator, DirectoryPath

from directionalscalper.core.strategies.logger import Logger
logging = Logger(logger_name="Configuration", filename="Configuration.log", stream=True)

VERSION = "v2.9.9"

class Exchanges(Enum):
    BYBIT = "bybit"


class Messengers(Enum):
    DISCORD = "discord"
    TELEGRAM = "telegram"


class API(BaseModel):
    filename: str = "quantdatav2.json"
    mode: str = "remote"
    url: str = "https://api.quantumvoid.org/volumedata/"
    data_source_exchange: str = "bybit"

class Hotkeys(BaseModel):
    hotkeys_enabled: bool = False
    enter_long: str = "1"
    take_profit_long: str = "2"
    enter_short: str = "3"
    take_profit_short: str = "4"


class Bot(BaseModel):
    bot_name: str
    volume_check: bool = True
    min_distance: float = 0.15
    min_volume: int = 15000
    wallet_exposure_limit: float = 0.0001
    user_defined_leverage_long: float = 0
    user_defined_leverage_short: float = 0
    upnl_profit_pct: float = 0.003
    max_upnl_profit_pct: float = 0.004
    stoploss_enabled: bool = False
    stoploss_upnl_pct: float = 0.070
    liq_stoploss_enabled: bool = False
    liq_price_stop_pct: float = 0.50
    percentile_auto_reduce_enabled: bool = False
    auto_reduce_enabled: bool = False
    auto_reduce_start_pct: float = 0.098
    upnl_threshold_pct: float = 0.10
    max_pos_balance_pct: float = 0.20
    auto_reduce_maxloss_pct: float = 0.50
    auto_reduce_marginbased_enabled: bool = False
    auto_reduce_wallet_exposure_pct: float = 0.10
    entry_during_autoreduce: bool = True
    hedge_ratio: float = 0.26
    hedge_price_difference_threshold: float = 0.15
    auto_leverage_upscale: bool = False
    min_qty_threshold: float = 0
    symbol: str
    long_liq_pct: float = 0.05
    short_liq_pct: float = 0.05
    MaxAbsFundingRate: float = 0.0002
    wallet_exposure: float = 1.00
    test_orders_enabled: bool = False
    max_usd_value: Optional[float] = None
    blacklist: List[str] = []
    whitelist: List[str] = []
    dashboard_enabled: bool = False
    shared_data_path: Optional[str] = None
    #risk_management: Optional[dict] = None
    linear_grid: Optional[dict] = None
    hotkeys: Hotkeys
    # shared_data_path: Optional[DirectoryPath] = None

    @validator('hotkeys')
    def validate_hotkeys(cls, value):
        if not value:
            raise ValueError("hotkeys must be provided and valid")
        return value

    @validator('linear_grid')
    def validate_linear_grid(cls, value):
        if value is None:
            raise ValueError("linear_grid must be a dictionary - check example config")
        return value

    @validator("upnl_profit_pct")
    def minimum_upnl_profit_pct(cls, v):
        if v < 0.0:
            raise ValueError("upnl_profit_pct must be greater than 0")
        return v
    
    @validator("max_upnl_profit_pct")
    def minimum_max_upnl_profit_pct(cls, v):
        if v < 0.0:
            raise ValueError("max_upnl_profit_pct must be greater than 0")
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

    @validator("long_liq_pct")
    def minimum_long_liq_pct(cls, v):
        if v < 0.0:
            raise ValueError("long_liq_pct must be greater than 0")
        return v

    @validator("short_liq_pct")
    def minimum_short_liq_pct(cls, v):
        if v < 0.0:
            raise ValueError("short_liq_pct must be greater than 0")
        return v

    @validator('test_orders_enabled')
    def check_test_orders_enabled_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("test_orders_enabled must be a boolean")
        return v
    
    @validator('auto_reduce_enabled')
    def check_auto_reduce_enabled_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("auto_reduce_enabled must be a boolean")
        return v

    @validator('liq_stoploss_enabled')
    def check_liq_stoploss_enabled_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("liq_stoploss_enabled must be a boolean")
        return v

    @validator('liq_price_stop_pct')
    def validate_liq_price_stop_pct(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("liq_price_stop_pct must be between 0.0 and 1.0")
        return v

    @validator('auto_reduce_start_pct')
    def validate_auto_reduce_start_pct(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("auto_reduce_start_pct must be between 0.0 and 1.0")
        return v
    
    @validator('upnl_threshold_pct')
    def validate_upnl_threshold_pct(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("upnl_threshold_pct must be between 0.0 and 1.0")
        return v
    
    @validator('auto_reduce_maxloss_pct')
    def validate_auto_reduce_maxloss_pct(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("auto_reduce_maxloss_pct must be between 0.0 and 1.0")
        return v

    @validator('entry_during_autoreduce')
    def check_entry_during_autoreduce_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("entry_during_autoreduce must be a boolean")
        return v

    @validator('auto_reduce_marginbased_enabled')
    def check_auto_reduce_marginbased_enabled_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("auto_reduce_marginbased_enabled must be a boolean")
        return v

    @validator('auto_reduce_wallet_exposure_pct')
    def validate_auto_reduce_wallet_exposure_pct(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError("auto_reduce_wallet_exposure_pct must be between 0.0 and 1.0")
        return v

    @validator('percentile_auto_reduce_enabled')
    def check_percentile_auto_reduce_enabled_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("percentile_auto_reduce_enabled must be a boolean")
        return v

    @validator('max_pos_balance_pct')
    def validate_max_pos_balance_pct(cls, v):
        if v < 0.0:
            raise ValueError("max_pos_balance_pct must be between 0.0 and 1.0")
        return v

    @validator('auto_leverage_upscale')
    def check_auto_leverage_upscale_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("auto_leverage_upscale must be a boolean")
        return v

    @validator('volume_check')
    def check_volume_check_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError("volume_check must be a boolean")
        return v
    
class Exchange(BaseModel):
    name: str
    account_name: str
    api_key: str
    api_secret: str
    passphrase: str = None
    symbols_allowed: int = 12

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

class Telegram(BaseModel):
    active: bool = False
    embedded_messages: bool = True
    messenger_type: str = Messengers.TELEGRAM.value  # type: ignore
    bot_token: str
    chat_id: str


class Config(BaseModel):
    api: API
    bot: Bot
    exchanges: List[Exchange]  # <-- Changed from List[Exchange]
    logger: Logger
    messengers: dict[str, Union[Discord, Telegram]]

def resolve_shared_data_path(relative_path: str) -> Path:
    # Get the directory of the config file
    base_path = Path(__file__).parent

    # Resolve the relative path
    absolute_path = (base_path / relative_path).resolve()

    if not absolute_path.exists():
        raise ValueError(f"Shared data path does not exist: {absolute_path}")

    return absolute_path

def load_config(path):
    if not path.is_file():
        raise ValueError(f"{path} does not exist")
    else:
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"ERROR: Invalid JSON: {exc.msg}, line {exc.lineno}, column {exc.colno}"
                )
        try:
            config_data = Config(**data)

            # Resolve shared_data_path if it's provided
            if config_data.bot.shared_data_path:
                config_data.bot.shared_data_path = str(resolve_shared_data_path(config_data.bot.shared_data_path))

            return config_data
        except ValidationError as e:
            # Enhanced error output
            error_details = "\n".join([f"{err['loc']} - {err['msg']}" for err in e.errors()])
            raise ValueError(f"Configuration Error(s):\n{error_details}")
        
# def load_config(path):
#     if not path.is_file():
#         raise ValueError(f"{path} does not exist")
#     else:
#         f = open(path)
#         try:
#             data = json.load(f)
#         except json.JSONDecodeError as exc:
#             raise ValueError(
#                 f"ERROR: Invalid JSON: {exc.msg}, line {exc.lineno}, column {exc.colno}"
#             )
#         try:
#             return Config(**data)
#         except ValidationError as e:
#             # Enhancing the error output for better clarity
#             error_details = "\n".join([f"{err['loc']} - {err['msg']}" for err in e.errors()])
#             raise ValueError(f"Configuration Error(s):\n{error_details}")

def get_exchange_name(cli_exchange_name):
    if cli_exchange_name:
        return cli_exchange_name
    else:
        with open('config.json') as file:
            data = json.load(file)
            return data['exchanges'][0]['name']

def get_exchange_credentials(exchange_name, account_name):
    with open('config.json') as file:
        data = json.load(file)
        exchange_data = None
        for exchange in data['exchanges']:
            if exchange['name'] == exchange_name and exchange['account_name'] == account_name:
                exchange_data = exchange
                break
        if exchange_data:
            api_key = exchange_data['api_key']
            secret_key = exchange_data['api_secret']
            passphrase = exchange_data.get('passphrase')
            symbols_allowed = exchange_data.get('symbols_allowed', 12)  # Default to 12 if not specified

            # Logging the symbols_allowed value
            if 'symbols_allowed' in exchange_data:
                logging.info(f"Retrieved symbols_allowed for {exchange_name}: {symbols_allowed}")
            else:
                logging.warning(f"symbols_allowed not found for {exchange_name}. Defaulting to 12.")
            
            return api_key, secret_key, passphrase, symbols_allowed
        else:
            raise ValueError(f"Account {account_name} for exchange {exchange_name} not found in the config file.")
        
# def get_exchange_credentials(exchange_name, account_name):
#     with open('config.json') as file:
#         data = json.load(file)
#         exchange_data = None
#         for exchange in data['exchanges']:
#             if exchange['name'] == exchange_name and exchange['account_name'] == account_name:
#                 exchange_data = exchange
#                 break
#         if exchange_data:
#             api_key = exchange_data['api_key']
#             secret_key = exchange_data['api_secret']
#             passphrase = exchange_data.get('passphrase')
#             symbols_allowed = exchange_data.get('symbols_allowed', 12)  # Default to 12 if not specified
#             return api_key, secret_key, passphrase, symbols_allowed
#         else:
#             raise ValueError(f"Account {account_name} for exchange {exchange_name} not found in the config file.")

# def get_exchange_credentials(exchange_name, account_name):
#     with open('config.json') as file:
#         data = json.load(file)
#         exchange_data = None
#         for exchange in data['exchanges']:
#             if exchange['name'] == exchange_name and exchange['account_name'] == account_name:
#                 exchange_data = exchange
#                 break
#         if exchange_data:
#             api_key = exchange_data['api_key']
#             secret_key = exchange_data['api_secret']
#             passphrase = exchange_data.get('passphrase')  # Not all exchanges require a passphrase
#             return api_key, secret_key, passphrase
#         else:
#             raise ValueError(f"Account {account_name} for exchange {exchange_name} not found in the config file.")
        
# def get_exchange_credentials(exchange_name):
#     with open('config.json') as file:
#         data = json.load(file)
#         exchange_data = None
#         for exchange in data['exchanges']:
#             if exchange['name'] == exchange_name:
#                 exchange_data = exchange
#                 break
#         if exchange_data:
#             api_key = exchange_data['api_key']
#             secret_key = exchange_data['api_secret']
#             passphrase = exchange_data.get('passphrase')  # Not all exchanges require a passphrase
#             return api_key, secret_key, passphrase
#         else:
#             raise ValueError(f"Exchange {exchange_name} not found in the config file.")


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
