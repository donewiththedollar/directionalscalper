<h1 align="center">Directional Scalper Multi Exchange</h1>
<p align="center">
A hedge scalping strategy based on directional analysis using a quantitative approach<br>
</p>
<p align="center">
<img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/donewiththedollar/directionalscalper"> 
<a href="https://github.com/donewiththedollar/directionalscalper/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

>  Exchanges to support: Bybit, Bitget, OKX, Binance, Phemex

### Links
* Dashboard: https://tradesimple.xyz
* API: http://api.tradesimple.xyz/data/quantdata.json
* APIv2: http://api.tradesimple.xyz/data/quantdatav2.json

Directional Scalper        |  API Scraper
:-------------------------:|:-------------------------:
![](https://github.com/donewiththedollar/directional-scalper/blob/main/directional-scalper.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/scraper.gif)

## Quickstart
- Clone the repo `git clone https://github.com/donewiththedollar/directionalscalper.git`
- Install requirements `pip3 install -r requirements.txt`

### Setting up the bot
 1. Create `config.json` from `config.example.json` in /config directory
 2. Enter exchange_api_key and exchange_api_secret
 3. Check/fill all other options. For telegram see below

 1. Get token from botfather after creating new bot, send a message to your new bot
 2. Go to https://api.telegram.org/bot<bot_token>/getUpdates
 3. Replacing <bot_token> with your token from the botfather after creating new bot
 4. Look for chat id and copy the chat id into config.json

### Starting the bot
* Hedge mode is recommended, but you can of course use the other modes as well. Low lot size is recommended.
> python3.11 bot.py --exchange bitget --symbol XRPUSDT_UMCBL --amount 15 --strategy bitget_hedge --config config_sub1.json

> python3.11 bot.py --exchange bybit --symbol XRPUSDT --amount 1 --strategy bybit_hedge --config config_sub1.json

### Modes
* --mode [hedge], WIP
> Some (most) modes are in development, hedge mode is the recommended mode that has proven to be profitable and allows you to control your risk accordingly.

### Parameters
> --config', type=str, default='config.json', help='Path to the configuration file')

> --exchange', type=str, help='The name of the exchange to use')

> --strategy', type=str, help='The name of the strategy to use')

> --symbol', type=str, help='The trading symbol to use')

> --amount', type=str, help='The size to use')

### Docker
To run the bot inside docker container use the following command:
> docker-compose run directional-scalper python3 bot.py --mode hedge --symbol GALAUSDT --iqty 1 --tg off

* There are six working modes:
> long, short, hedge, aggressive, inverse, violent


### Developer instructions
- Install developer requirements from pipenv `pipenv install --dev` (to keep requirements in a virtual environment)
- Install pre-commit hooks `pre-commit install` (if you intend to commit code to the repo)
- Run pytest `pytest -vv` (if you have written any tests to make sure the code works as expected)


### To do:
* Instance manager
* Auto calculation for violent parameters (violent_multiplier and wallet_exposure are key)
* Auto calculation for lot size so the user does not have to determine size


### Donations
If you would like to show your appreciation for this project through donations, there are a few addresses here to choose from
* **BTC**: bc1q9hyvvtcsm0k39svz59hjgz4f6dr6c2k4wlrxmc
