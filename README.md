<h1 align="center">Directional Scalper Multi Exchange</h1>
<p align="center">
An algorithmic trading framework built using CCXT for multiple exchanges<br>
</p>
<p align="center">
<img alt="GitHub Pipenv locked Python version" src="https://img.shields.io/github/pipenv/locked/python-version/donewiththedollar/directionalscalper"> 
<a href="https://github.com/donewiththedollar/directionalscalper/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

![Visitor Count](https://komarev.com/ghpvc/?username=donewiththedollar)

![GitHub Stats](https://github-readme-stats.vercel.app/api?username=donewiththedollar&show_icons=true&theme=radical)

## Directional Scalper documentation
[Documentation](https://donewiththedollar.github.io/directionalscalper/)

### Links
* Website: https://quantumvoid.org
* API (BYBIT): https://api.quantumvoid.org/data/quantdatav2_bybit.json
* Discord: https://discord.gg/4GvHqPxfud

Directional Scalper        |  API Scraper               |  Dashboard                | Directional Scalper Multi | Menu GUI
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/donewiththedollar/directional-scalper/blob/main/directional-scalper.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/scraper.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/dashboardimg.gif)  |  ![](https://github.com/donewiththedollar/directionalscalper/blob/main/directionalscalpermulti.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/menugui.gif)


### Requirements
- Python 3.11+ (Docker image: `python:3.14`)
- Dependencies: `pip install -r requirements.txt`

### Quick start (multi-symbol rotator)
```bash
python3 multi_bot_aio.py --exchange bybit --account_name account_1 --strategy qsgridob --config configs/config.json
```
Copy `configs/config_example.json` + `configs/account_example.json` to `configs/config.json` / `configs/account.json` first.

### Strategies (multi-bot)
| Name | Description |
|---|---|
| `qsgridob` | Linear grid base futures (signal-driven) |
| `qsgridob_nosignal` | Same grid, immediate-entry (no signal wait) |
| `qstrendobdynamictp` | Trend scalp with dynamic TP |
| `breathinggrid` | **Volume-farming breathing grid** — volatility-derived 6-level maker ladder that re-prices (“breathes”) every cycle; sequential-add gating, fee-aware TPs, leverage/notional caps, session drawdown breaker. Runs on Bybit and BloFin (BloFin order flow is attributed to the configured broker code). **Untested on live funds — start with testnet keys or a dust `budget_usd`.** |

Start the breathing grid with the helper script:
```bash
./start_breathing_grid.sh                     # BloFin defaults
EXCHANGE=bybit ./start_breathing_grid.sh      # Bybit
ACCOUNT_NAME=myacct CONFIG=configs/config.json ./start_breathing_grid.sh
```
Configure sizing via the optional `bot.breathing_grid` dict in your config (see `configs/config_example.json`).

### DS Bridge — terminal UI
A Hummingbot-style terminal console: status strip, running-bots table, classified LOG/ALERTS pane, and a persistent `>>>` command line.
```bash
./start_tui.sh          # or: python3 -m ds_tui
```
The TUI is read-only over trading state. Commands:

| Command | Action |
|---|---|
| `bots` / `status` | refresh running-bot discovery |
| `log <name> [lines]` | tail a log from `logs/` |
| `errors [n]` | recent critical lines across logs |
| `grep <pattern>` | search newest logs |
| `config <file.json>` | view a config — **all secrets redacted** |
| `stop <pid\|name>` | stop a bot process (**y/N confirmed**) |
| `help`, `quit` | you guessed it |

### Docker
To run the bot inside docker container use the following command:
> docker-compose run directional-scalper python3 bot.py --symbol SUIUSDT --strategy qsgridob --config configs/config_example.json

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
- Run tests `pytest -vv` (smoke + breathing-grid policy unit tests live in `tests/`)
- Terminal UI lives in `ds_tui/`; strategy math is pure/testable in `directionalscalper/core/strategies/bybit/gridbased/breathing_policy.py`


### To do:
* A lot of top secret cutting edge stuff
* Huobi, Binance, Phemex, MEXC base. (MEXC Futs API down until Q4)
