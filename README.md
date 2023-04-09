<h1 align="center">Directional Scalper</h1>
<p align="center">
A hedge scalping strategy based on directional analysis using a quantitative approach<br>
</p>


### Supports Bybit only, other exchanges coming soon

### Links
* Dashboard: https://tradesimple.xyz
* API: http://api.tradesimple.xyz/data/quantdata.json
* APIv2: http://api.tradesimple.xyz/data/quantdatav2.json

Solarized dark             |  Solarized Ocean
:-------------------------:|:-------------------------:
![](https://github.com/donewiththedollar/directional-scalper/blob/main/directional-scalper.gif)  |  ![](https://github.com/donewiththedollar/directional-scalper/blob/main/scraper.gif)

# Instructions
* Install requirements
> pip3 install -r requirements.txt

### Developer instructions
- Install developer requirements from pipenv `pipenv install --dev`
- Install pre-commit hooks `pre-commit install`
- Run pytest `pytest -vv`

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
> python3 bot.py --mode hedge --symbol GALAUSDT --iqty 1 --tg off --config config.json --avoidfees on
* Starting the bot in violent mode is not recommended, but ensures violent profit taking while putting you at risk of liquidation depending on your wallet_exposure and violent_multiplier
> python3 bot.py --mode violent --symbol OPUSDT --iqty 0.1 --tg off --config config.json --avoidfees on

* Starting the bot in debug mode for inverse perpetuals BTCUSD
* Inverse is currently short only, used as a hedge against your BTC balance, to accumulate BTC with no risk, no losses
> python3 bot_inverse_debugmode.py --mode inverse --symbol BTCUSD --iqty 1 --tg off

### Modes
* --mode [hedge, aggressive, violent, long, short, longbias, btclinear-long, btclinear-short]
> Some (most) modes are in development, hedge mode is the recommended mode that has proven to be profitable and allows you to control your risk accordingly.

### Parameters
> --avoidfees [on, off]
> --deleverage [on, off]
* only use one or the other [avoidfees, or deleverage], deleverage is incremental TP, while avoidfees is incremental TP including the taker fees.


### Docker
To run the bot inside docker container use the following command:
> docker-compose run directional-scalper python3 bot.py --mode hedge --symbol GALAUSDT --iqty 1 --tg off

* There are six working modes:
> long, short, hedge, aggressive, inverse, violent

### To do:
* Instance manager
* Auto calculation for violent parameters (violent_multiplier and wallet_exposure are key)
* Auto calculation for lot size so the user does not have to determine size
* Refactor so the main bot is not thousands of lines of code :)


### Donations
If you would like to show your appreciation for this project through donations, there are a few addresses here to choose from
* **BTC**: bc1qu5p292xs9jvu0vuanjcpsqszmg4hkmrrahdpj7
* **DOGE**: D9iNDsVpJaXqChmveCUsvnM87sQo5Tcia6
