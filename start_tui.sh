#!/usr/bin/env bash
# Launch the DS Bridge TUI (python3 -m ds_tui).
set -euo pipefail
cd "$(dirname "$0")"

if ! python3 -c "import rich" 2>/dev/null; then
    echo "rich is required (already in requirements.txt). Installing..."
    python3 -m pip install -q rich
fi

exec python3 -m ds_tui
