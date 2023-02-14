import os
import ccxt
import traceback
import logging
import logging.config
import pybit
from pybit import usdt_perpetual
#from pybit.usdt_perpetual import HTTP
from pybit import HTTP
from uuid import uuid4
import config
from config import *
import time


def notify_message(message):
    import telegram
    telegram_http_api = ""
    telegram_user_id = ""

    tgbot = telegram.Bot(token=telegram_http_api)
    tgbot.send_message(chat_id=telegram_user_id, text=message, parse_mode='HTML')

exchange = ccxt.bybit(
    {
        'apiKey':config.bybit_api_key,
        'secret':config.bybit_api_secret
    }
)

ws_perp = usdt_perpetual.WebSocket(
    test=False,
    api_key=config.bybit_api_key,
    api_secret=config.bybit_api_secret,
    domain=domain
)

def main():

    try:
        session = HTTP(
            endpoint="https://api.bybit.com",
            api_key=config.bybit_api_key,
            api_secret=config.bybit_api_secret,
        )
        notify_message(
                f"Fleet 2 is running")
    except:
        logging.error(f"failed to connect with Bybit")
        logging.error(f"{traceback.format_exc()}")
        return

if __name__ == "__main__":
    while True:
        print("Checking..")
        main()
        time.sleep(3600)
