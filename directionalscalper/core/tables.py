from rich.live import Live
from rich.table import Table
from rich.layout import Layout


def create_strategy_table(symbol, total_equity, long_upnl, short_upnl, short_pos_qty, long_pos_qty, amount, cumulative_realized_pnl):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Directional Scalper", justify="center")

    # Format long_upnl and short_upnl with two decimal places
    long_upnl_formatted = f"{long_upnl:.2f}"
    short_upnl_formatted = f"{short_upnl:.2f}"
    cumulative_realized_pnl_formatted = f"{cumulative_realized_pnl:.2f}"

    # Add the strategy information as rows in the table
    table.add_row(f"Symbol: {symbol}")
    table.add_row(f"Total Equity: {total_equity}")
    table.add_row(f"Long Position Qty: {long_pos_qty}")
    table.add_row(f"Short Position Qty: {short_pos_qty}")
    table.add_row(f"Long Position uPNL: {long_upnl_formatted}")
    table.add_row(f"Short Position uPNL: {short_upnl_formatted}")
    table.add_row(f"Cumulative Realized PNL: {cumulative_realized_pnl_formatted}")
    table.add_row(f"Amount: {amount}")

    return table

def display_live_table(table):
    with Live(table, refresh_per_second=4):
        input("Press Enter to stop the live table...")


if __name__ == "__main__":
    # Example usage
    symbol = "BTC/USDT"
    total_equity = "10000 USDT"
    short_pos_qty = 2
    long_pos_qty = 1
    amount = "0.5 BTC"

    strategy_table = create_strategy_table(symbol, total_equity, short_pos_qty, long_pos_qty, amount)
    display_live_table(strategy_table)
