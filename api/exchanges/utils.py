import json
import logging
from enum import Enum

import requests  # type: ignore

log = logging.getLogger(__name__)


class Exchanges(str, Enum):
    BYBIT = "bybit"


class Intervals(str, Enum):
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"


def get_api_data(url: str, endpoint: str):
    response_json = {}
    try:
        response = requests.get(f"{url}{endpoint}")
        response.raise_for_status()  # Raise an exception if an HTTP error occurs
        response_json = response.json()
    except (requests.exceptions.HTTPError, json.JSONDecodeError) as e:
        log.warning(f"{e}")
    return response_json
