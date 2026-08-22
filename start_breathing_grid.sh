#!/usr/bin/env bash
# Launch the breathing-grid volume-farm strategy (multi-bot rotator).
#
# Usage:
#   ./start_breathing_grid.sh                      # BloFin (broker-attributed) defaults
#   EXCHANGE=bybit ./start_breathing_grid.sh       # Bybit
#   BUDGET_NOTE: set bot.breathing_grid.budget_usd in your config first.
#
# Env overrides:
#   EXCHANGE      (default blofin)
#   ACCOUNT_NAME  (default account_1)
#   CONFIG        (default configs/config.json)

set -euo pipefail
cd "$(dirname "$0")"

EXCHANGE="${EXCHANGE:-blofin}"
ACCOUNT_NAME="${ACCOUNT_NAME:-account_1}"
CONFIG="${CONFIG:-configs/config.json}"

echo "Starting breathinggrid on ${EXCHANGE} (account: ${ACCOUNT_NAME}, config: ${CONFIG})"
echo "NOTE: untested on live funds — start with testnet keys or a dust budget_usd."

exec python3 multi_bot_aio.py \
    --exchange "${EXCHANGE}" \
    --account_name "${ACCOUNT_NAME}" \
    --strategy breathinggrid \
    --config "${CONFIG}"
