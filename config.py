from pybit import inverse_perpetual

# Bybit API keys
bybit_api_key = ''
bybit_api_secret = ''

endpoint = 'https://api.bybit.com'
domain = 'bybit'

unauth = inverse_perpetual.HTTP(endpoint=endpoint)
invpcl = inverse_perpetual.HTTP(endpoint=endpoint, api_key=bybit_api_key, api_secret=bybit_api_secret)

config_min_volume = 15000
config_min_distance = 0.15

config_botname = 'botnameherefortg'

symbol = 'BTCUSD'
csize = 2
min_fee = 0.17
divider = 7
