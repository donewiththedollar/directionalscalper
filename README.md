<h1 align="center">Directional Scalper Multi Exchange</h1>
<p align="center">
An algorithmic trading framework built using CCXT for multiple exchanges<br>
</p>
<p align="center">
<img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/donewiththedollar/directionalscalper"> 
<a href="https://github.com/donewiththedollar/directionalscalper/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

### Links
* Website: https://quantumvoid.org
* API (BYBIT): https://api.quantumvoid.org/data/quantdatav2_bybit.json
* Discord: https://discord.gg/4GvHqPxfud

Directional Scalper        |  API Scraper               |  Dashboard                | Directional Scalper Multi | Menu GUI
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/donewiththedollar/directional-scalper/blob/main/directional-scalper.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/scraper.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/dashboardimg.gif)  |  ![](https://github.com/donewiththedollar/directionalscalper/blob/main/directionalscalpermulti.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/menugui.gif)

## Quickstart
- Clone the repo `git clone https://github.com/donewiththedollar/directionalscalper.git`
- Install requirements `pip3.11 install -r requirements.txt`
- Add API key(s) to config.json in /configs folder
- Run the bot `python3.11 multi_bot_signalscreener.py --config config.json or the old multi_bot.py --config config.json` to display the menu to select a strategy

  OR via command line parameters
  
- Multi bot auto symbol rotator strategy example: `python3.11 multi_bot_signalscreener.py --exchange bybit --account_name account_1 --strategy qsgridob --config config.json`
- Old single coin strategy example: `python3.11 bot.py --exchange bybit --symbol DOGEUSDT --strategy qstrendob --config config.json`

## Working Exchanges
> Bybit, Bybit Unified (per strategy)

> Easily compatible with any exchange that cooperates with CCXT, some functions are slightly different per exchange.
 
> Exchanges that are WIP: Huobi, Okx, Binance, Bitget, MEXC (There is still no futures API), Phemex

## Dashboard setup
- Run multi_bot `python3.11 multi_bot_signalscreener.py` or with arguments `python3.11 multi_bot_signalscreener.py --exchange bybit --account_name account_1 --strategy qsgridob --config config.json`
- Start a tmux session `tmux new -s dash`
- Inside the tmux session, ensure you are in project directory and `streamlit run dashboard.py`
- If you are having issues, you may have not ran `pip3.11 install -r requirements.txt` again as requirements have changed in recent revisions.


### Contributions / Donations

* USDT (TRC20): TMnCyGsd6BtFRi9yHJ5avtpVGuGzRu3cRb
  
* USDT (ERC20): 0xb40b2842d4ce93e31CFC8DC2629E2Bd426e4b87E

* DOGE: DAZid4pETjmrgGkYvgN5rZZtCaBpYtiK8E

* BTC: bc1q9hyvvtcsm0k39svz59hjgz4f6dr6c2k4wlrxmc

### Affiliate links
## Bybit https://partner.bybit.com/b/quantumvoid

### Full installation instructions
Steps to set up Directionalscalper v2

- `sudo apt-get update`
- `sudo apt-get upgrade -y`
- `git clone https://github.com/donewiththedollar/directionalscalper.git`

Now you have the bot but you must install Python 3.11

Here is how you can install Python 3.11 from source: 

All of these commands one by one copy and pasted into terminal: 
`wget https://www.python.org/ftp/python/3.11.0/Python-3.11.0.tgz`
`tar -xvf Python-3.11.0.tgz`

`cd Python-3.11.0`

- `sudo apt-get update`
- `sudo apt-get install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev libbz2-dev`

- `./configure --enable-optimizations`
- `make`
- `sudo make altinstall`

- `python3.11 --version`

> After typing python3.11 —version, it should display that you have python3.11 installed 

> Now install PIP

- `curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11`

- `pip3.11 -V`

This pip3.11 -V should show you have pip3.11 installed, no error.


Starting the bot for the first time making sure it works

Make sure you are in the directory directionalscalper

- `so cd ~/directionalscalper should bring you there if you are not already there.`

> Run these:

- `pip3.11 install -r requirements.txt`

> Modify your config_example.json file with your bitget API keys and then run this:
- `python3.11 bot.py --exchange bybit --symbol SUIUSDT --strategy bybit_hedge_mfirsi_maker --config config_bybit_sub1.json`
> or a different exchange:
- `python3.11 bot.py --exchange bitget --symbol OPUSDT_UMCBL --strategy bitget_hedge_dynamic --config config_example.json`
 
### Setting up the bot
 1. Create `config.json` from `config.example.json` in /configs directory
 2. Enter exchange_api_key and exchange_api_secret
 3. Check/fill all other options. For telegram/discord see below
 
### Starting the bot menu

> python3.11 multi_bot.py --config config_name.json

### Starting the bot using arguments

> python3.11 multi_bot.py (no params opens menu)
> python3.11 multi_bot.py --exchange bybit --account_name account_1 --strategy qstrend --config config.json
> python3.11 multi_bot.py --exchange bybit --account_name account_2 --strategy qstrendob --config config.json


### Old bot params (likely outdated)

> python3.11 bot.py --exchange bybit --symbol SUIUSDT --strategy bybit_hedge_mfirsi_maker --config config_bybit_sub1.json

> python3.11 bot.py --exchange bitget --symbol XRPUSDT_UMCBL --amount 15 --strategy bitget_hedge --config config_main.json
 
> python3.11 bot.py --exchange huobi --strategy huobi_auto_hedge --account_name account_4 --amount 1 --symbol DOGEUSDT --config config.json
 
* Example of starting a different strategy. In this example amount is not needed, as it is dynamic based on minimum required by Bitget.
 
> python3.11 bot.py --exchange bitget --symbol XRPUSDT_UMCBL --strategy bitget_hedge_dynamic --config config_main.json

### Current strategies
## Binance - strategies need to be updated
* binance_auto_hedge

## Bybit Signalscreener
* qsgridob
  
## Bybit multi
* bybit_1m_qfl_mfi_eri_walls
* bybit_1m_qfl_mfi_eri_autohedge_walls_atr
* bybit_mfirsi_imbalance
* bybit_mfirsi_quickscalp
* qstrend
* qstrendob
* qstrendlongonly

## Bybit single coin (Outdated)
* bybit_hedge
* bybit_auto_hedge - Dynamic entry, take profit distance, position leverage per side. Table included.
* bybit_auto_hedge_maker
* bybit_hedge_mfirsi_maker - MFI, RSI, ERI, MA for entry. Pure maker, dynamic entry size, dynamic take profit based on 5m spread
* bybit_hedge_mfirsionly_maker - MFIRSI only as entry. Pure maker, dynamic entry size, dynamic take profit based on 5m spread
* bybit_longonly
* bybit_shortonly
* bybit_longonly_dynamic_leverage
* bybit_shortonly_dynamic_leverage
 
## Bitget - strategies need to be updated
* bitget_hedge
* bitget_hedge_dynamic
* bitget_longonly_dynamic
* bitget_shortonly_dynamic
 
## Huobi - strategies need to be updated
* huobi_hedge
* huobi_auto_hedge
 
 
### Parameters
> --config', type=str, default='config.json', help='Path to the configuration file')

> --exchange', type=str, help='The name of the exchange to use')

> --strategy', type=str, help='The name of the strategy to use')

> --symbol', type=str, help='The trading symbol to use')

> --amount', type=str, help='The size to use')

### Docker
To run the bot inside docker container use the following command:
> docker-compose run directional-scalper python3.11 bot.py --symbol SUIUSDT --strategy bybit_hedge_mfirsi_maker --config config_main.json

### Proxy
If you need to use a proxy to access the Exchange API, you can set the environment variables as shown in the following example:
```bash
$ export HTTP_PROXY="http://10.10.1.10:3128"  # these proxies won't work for you, they are here for example
$ export HTTPS_PROXY="http://10.10.1.10:1080"
```

### Setting up Telegram alerts (not used currently)
1. Get token from botfather after creating new bot, send a message to your new bot
2. Go to https://api.telegram.org/bot<bot_token>/getUpdates
3. Replacing <bot_token> with your token from the botfather after creating new bot
4. Look for chat id and copy the chat id into config.json

### Developer instructions
- Install developer requirements from pipenv `pipenv install --dev` (to keep requirements in a virtual environment)
- Install pre-commit hooks `pre-commit install` (if you intend to commit code to the repo)
- Run pytest `pytest -vv` (if you have written any tests to make sure the code works as expected)


### To do:
* A lot of top secret cutting edge stuff
* Huobi, Binance, Phemex, MEXC base. (MEXC Futs API down until Q4)
