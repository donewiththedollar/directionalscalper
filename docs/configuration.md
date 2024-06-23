# Configuration

## Account Section

Configure your account settings in the `account.json` file under the `account` section. This includes API keys, account names, and other related settings.

### Parameters:

- **name**: Exchange name
- **api_key**: Your exchange API key.
- **api_secret**: Your exchange API secret.
- **account_name**: A name to identify your account configuration.

## Accounts

Configure your accounts using 'account.json' in the configs folder.

```json
{
    "exchanges": [
        {
            "name": "bybit",
            "account_name": "account_1",
            "api_key": "your_api_key",
            "api_secret": "your_api_secret"
        },
        {
            "name": "bybit_spot",
            "account_name": "account_2",
            "api_key": "your_api_key",
            "api_secret": "your_api_secret"
        },
        {
            "name": "bybit_unified",
            "account_name": "account_3",
            "api_key": "your_api_key",
            "api_secret": "your_api_secret"
        }
    ]
}
```

## Bot Section

Configure your bot settings in the `config.json` file under the `bot` section. This includes strategy parameters, risk management settings, and other related settings.

### Risk management

- **user_defined_leverage_long**: This parameter defines the leverage to use for long positions. Leverage is a multiplier that increases the size of the position and the potential profit/loss. For example, a leverage of 10 means that for every $1 you have, you can place a trade worth $10. Be careful with this setting as higher leverage also increases the risk.
- **user_defined_leverage_short**: This parameter defines the leverage to use for short positions. Similar to `user_defined_leverage_long`, it's a multiplier that increases the size of the position and the potential profit/loss.

### Linear Grid Strategy Parameters

Here are the parameters for the `linear_grid` strategy in your `config.json` file:

- **max_qty_percent_long**: Defines the maximum percentage of your wallet balance that can be used for long positions.
- **max_qty_percent_short**: Defines the maximum percentage of your wallet balance that can be used for short positions.
- **auto_reduce_cooldown_enabled**: Boolean parameter that determines whether the auto-reduce cooldown feature is enabled. This feature automatically reduces the size of your positions after a certain period.
- **auto_reduce_cooldown_start_pct**: Defines the percentage at which the auto-reduce cooldown feature starts.
- **wallet_exposure_limit_long**: Defines the maximum exposure limit for long positions as a percentage of your wallet balance.
- **wallet_exposure_limit_short**: Defines the maximum exposure limit for short positions as a percentage of your wallet balance.
- **levels**: Defines the number of grid levels, representing the total number of buy and sell orders placed above and below the current price level.
- **strength**: Defines the strength of the grid. A higher value means the grid levels are spaced further apart.
- **outer_price_distance**: Defines the distance from the current price to the outermost grid levels.
- **min_outer_price_distance**: Defines the minimum distance from the current price to the outermost grid levels.
- **max_outer_price_distance**: Defines the maximum distance from the current price to the outermost grid levels.
- **long_mode**: Boolean parameter that determines whether the bot can open long positions.
- **short_mode**: Boolean parameter that determines whether the bot can open short positions.
- **reissue_threshold**: Defines the threshold at which the bot will reissue orders that have been partially filled.
- **buffer_percentage**: Defines the buffer percentage for the grid. This is the percentage of the price range kept empty between the outermost grid levels and the upper and lower price limits.
- **initial_entry_buffer_pct**: Defines the buffer percentage for the initial entry.
- **min_buffer_percentage**: Defines the minimum buffer percentage for the grid.
- **max_buffer_percentage**: Defines the maximum buffer percentage for the grid.
- **enforce_full_grid**: Boolean parameter that determines whether the bot should always maintain a full grid of orders.
- **min_buffer_percentage_ar**: Defines the minimum buffer percentage for auto-reduce.
- **max_buffer_percentage_ar**: Defines the maximum buffer percentage for auto-reduce.
- **upnl_auto_reduce_threshold_long**: Defines the unrealized profit and loss (UPNL) threshold for auto-reducing long positions.
- **upnl_auto_reduce_threshold_short**: Defines the UPNL threshold for auto-reducing short positions.
- **failsafe_enabled**: Boolean parameter that determines whether the failsafe feature is enabled. This feature automatically closes all positions if the UPNL reaches a certain threshold.
- **failsafe_start_pct**: Defines the percentage at which the failsafe feature starts.
- **long_failsafe_upnl_pct**: Defines the UPNL percentage for the long failsafe.
- **short_failsafe_upnl_pct**: Defines the UPNL percentage for the short failsafe.

### Example Configuration Snippet

```json
{
  "api": {
      "filename": "quantdatav2_bybit.json",
      "mode": "remote",
      "url": "https://api.quantumvoid.org/volumedata/",
      "data_source_exchange": "bybit"
  },
  "bot": {
      "bot_name": "your_bot_name",
      "symbol": "BTCUSDT",
      "volume_check": false,
      "min_distance": 0.15,
      "min_volume": 10000,
      "wallet_exposure_limit": 0.001,
      "user_defined_leverage_long": 8,
      "user_defined_leverage_short": 5,
      "upnl_profit_pct": 0.0022,
      "max_upnl_profit_pct": 0.0029,
      "auto_reduce_enabled": false,
      "auto_reduce_start_pct": 0.068,
      "entry_during_autoreduce": false,
      "stoploss_enabled": false,
      "stoploss_upnl_pct": 0.05,
      "liq_stoploss_enabled": false,
      "liq_price_stop_pct": 0.50,
      "percentile_auto_reduce_enabled": false,
      "upnl_threshold_pct": 0.50,
      "max_pos_balance_pct": 0.50,
      "auto_reduce_wallet_exposure_pct": 0.20,
      "auto_reduce_maxloss_pct": 0.30,
      "auto_reduce_marginbased_enabled": false,
      "hedge_ratio": 0.10,
      "hedge_price_difference_threshold": 0.10,
      "auto_leverage_upscale": false,
      "test_orders_enabled": false,
      "max_usd_value": 50,
      "min_qty_threshold": 0,
      "long_liq_pct": 0.05,
      "short_liq_pct": 0.05,
      "MaxAbsFundingRate": 0.0002,
      "blacklist": ["BTCUSDT", "ETHUSDT"],
      "whitelist": [],
      "dashboard_enabled": false,
      "shared_data_path": "data/",
      "linear_grid": {
          "max_qty_percent_long": 5,
          "max_qty_percent_short": 5,
          "auto_reduce_cooldown_enabled": false,
          "auto_reduce_cooldown_start_pct": 0.051,
          "wallet_exposure_limit_long": 0.001,
          "wallet_exposure_limit_short": 0.001,
          "levels": 3,
          "strength": 1.4,
          "outer_price_distance": 0.059,
          "min_outer_price_distance": 0.019,
          "max_outer_price_distance": 0.039,
          "long_mode": true,
          "short_mode": true,
          "reissue_threshold": 0.001,
          "buffer_percentage": 0.10,
          "initial_entry_buffer_pct": 0.0001,
          "min_buffer_percentage": 0.0035,
          "max_buffer_percentage": 0.010,
          "enforce_full_grid": true,
          "min_buffer_percentage_ar": 0.002,
          "max_buffer_percentage_ar": 0.004,
          "upnl_auto_reduce_threshold_long": 10.0,
          "upnl_auto_reduce_threshold_short": 10.0,
          "failsafe_enabled": false,
          "failsafe_start_pct": 0.05,
          "long_failsafe_upnl_pct": 10.0,
          "short_failsafe_upnl_pct": 10.0
      },
      "hotkeys": {
          "hotkeys_enabled": false,
          "enter_long": "1",
          "take_profit_long": "2",
          "enter_short": "3",
          "take_profit_short": "4"
      }
  },
  "exchanges": [
      {
          "name": "bybit",
          "account_name": "account_1",
          "symbols_allowed": 10 
      },
      {
          "name": "bybit_spot",
          "account_name": "account_2",
          "symbols_allowed": 5
      },
      {
          "name": "bybit_unified",
          "account_name": "account_3",
          "symbols_allowed": 5
      }
  ],
  "logger": {
      "level": "info"
  },
  "messengers": {
      "discord": {
          "active": false,
          "embedded_messages": true,
          "messenger_type": "discord",
          "webhook_url": "https://discord.com/api/webhooks/your_webhook_id/your_webhook_token"
      },
      "telegram": {
          "active": false,
          "embedded_messages": true,
          "messenger_type": "telegram",
          "bot_token": "your_bot_token",
          "chat_id": "your_chat_id"
      }
  }
}
```

