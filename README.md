# Directional Scalper
## A hedge scalping strategy based on directional analysis using a quantitative approach
## Supports Bybit only, other exchanges coming soon

### Links
* Dashboard: https://tradesimple.xyz
* API: http://api.tradesimple.xyz/data/quantdata.json

![](https://github.com/donewiththedollar/directional-scalper/blob/main/directional-scalper.gif)
### Instructions
* Install requirements
> pip3 install -r requirements.txt
* Starting the bot
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
