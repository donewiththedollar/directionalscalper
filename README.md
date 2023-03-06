# Directional Scalper
## A hedge scalping strategy based on directional analysis using a quantitative approach
### Supports Bybit only, other exchanges coming soon

### Links
* Dashboard: https://tradesimple.xyz
* API: http://api.tradesimple.xyz/data/quantdata.json

![](https://github.com/donewiththedollar/directional-scalper/blob/main/directional-scalper.gif)
# Instructions
* Install requirements
> pip3 install -r requirements.txt
### Setting up the bot
 1. Create config.json from config.json.example
 2. Enter exchange_api_key and exchange_api_secret
 3. Check/fill all other options. For telegram see below

 1. Get token from botfather after creating new bot, send a message to your new bot
 2. Go to https://api.telegram.org/bot<bot_token>/getUpdates
 3. Replacing <bot_token> with your token from the botfather after creating new bot
 4. Look for chat id and copy the chat id into config.json

### Starting the bot
* Hedge mode is recommended, but you can of course use the other modes as well. Low lot size is recommended.
> python3 bot.py --mode hedge --symbol GALAUSDT --iqty 1 --tg off
* Starting the bot in debug mode for inverse perpetuals BTCUSD
* Inverse is currently short only, used as a hedge against your BTC balance, to accumulate BTC with no risk, no losses
> python3 bot_inverse_debugmode.py --mode inverse --symbol BTCUSD --iqty 1 --tg off


### Docker
To run the bot inside docker container use the following command:
> docker-compose run directional-scalper python3 bot.py --mode hedge --symbol GALAUSDT --iqty 1 --tg off

* There are five modes:
> long, short, hedge, persistent, inverse
* To do:
> Finish inverse perps mode for both long and short
> Instance manager
