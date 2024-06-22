# Configuration

## Account Section

Configure your account settings in the `config.json` file under the `account` section. This includes API keys, account names, and other related settings.

### Parameters:

- **api_key**: Your exchange API key.
- **api_secret**: Your exchange API secret.
- **account_name**: A name to identify your account configuration.

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
  "account": {
    "api_key": "your_api_key",
    "api_secret": "your_api_secret",
    "account_name": "account_1"
  },
  "bot": {
    "linear_grid": {
      "max_qty_percent_long": 50,
      "max_qty_percent_short": 50,
      "auto_reduce_cooldown_enabled": true,
      "auto_reduce_cooldown_start_pct": 10,
      "wallet_exposure_limit_long": 30,
      "wallet_exposure_limit_short": 30,
      "levels": 10,
      "strength": 2,
      "outer_price_distance": 5,
      "min_outer_price_distance": 2,
      "max_outer_price_distance": 10,
      "long_mode": true,
      "short_mode": true,
      "reissue_threshold": 5,
      "buffer_percentage": 1,
      "initial_entry_buffer_pct": 0.5,
      "min_buffer_percentage": 0.2,
      "max_buffer_percentage": 2,
      "enforce_full_grid": true,
      "min_buffer_percentage_ar": 0.2,
      "max_buffer_percentage_ar": 2,
      "upnl_auto_reduce_threshold_long": 10,
      "upnl_auto_reduce_threshold_short": 10,
      "failsafe_enabled": true,
      "failsafe_start_pct": 50,
      "long_failsafe_upnl_pct": -20,
      "short_failsafe_upnl_pct": -20
    }
  }
}
```