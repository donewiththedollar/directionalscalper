# Usage

This guide will help you run the Directional Scalper bot using various command-line instructions. Ensure you have properly configured your `config.json` and `account.json` files before proceeding.

## Running the Bot

### Displaying the Menu and Selecting a Strategy

To display the menu and select a strategy, use the following command:

```
python3.11 multi_bot_signalscreener.py --config configs/config.json
```

or

```
python3.11 multi_bot.py --config configs/config.json
```

### Running the Bot with Command Line Parameters

You can also run the bot directly with specific command-line parameters. Below are examples of different strategies you can use:

#### Multi-Bot Auto Symbol Rotator Strategy

To run the multi-bot auto symbol rotator strategy, use the following command:

```
python3.11 multi_bot_signalscreener_multicore.py --exchange bybit --account_name account_1 --strategy qsgridob --config configs/config.json
```

#### Old Single Coin Strategy

To run the old single coin strategy, use the following command:

```
python3.11 bot.py --exchange bybit --symbol DOGEUSDT --strategy qstrendob --config configs/config.json
```

## Command Line Parameters

### General Parameters

- **--exchange**: Specifies the exchange to use (e.g., bybit, binance).
- **--account_name**: Specifies the account name as defined in your `account.json`.
- **--strategy**: Specifies the trading strategy to use (e.g., qsgridob, qstrendob).
- **--config**: Specifies the path to your `config.json` file.

### Example Commands

1. **Running the Multi-Bot Auto Symbol Rotator Strategy for Bybit:**

   ```
   python3.11 multi_bot_signalscreener_multicore.py --exchange bybit --account_name account_1 --strategy qsgridob --config configs/config.json
   ```

2. **Running the Old Single Coin Strategy for DOGEUSDT on Bybit:**

   ```
   python3.11 bot.py --exchange bybit --symbol DOGEUSDT --strategy qstrendob --config configs/config.json
   ```

## Additional Notes

- Make sure to adjust the leverage, risk management, and other parameters in your `config.json` to suit your trading preferences.
- For advanced usage and more strategies, refer to the [Configuration](configuration.md) and [Strategies](strategies.md) sections of the documentation.

This guide provides a straightforward approach to running the Directional Scalper bot with various strategies and configurations. If you encounter any issues, refer to the relevant sections of the documentation for more details.
