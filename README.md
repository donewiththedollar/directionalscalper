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
