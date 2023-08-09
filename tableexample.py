import random
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table

class LiveTableManager:
    def __init__(self):
        self.table = self.generate_table()
        self.row_data = {}  # Dictionary to store row data
        self.lock = threading.Lock()

    def generate_table(self) -> Table:
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Symbol")
        table.add_column("Min. Qty")
        table.add_column("Price")
        table.add_column("Balance")
        table.add_column("Available Bal.")
        table.add_column("1m Vol")
        table.add_column("5m Spread")
        table.add_column("Trend")
        table.add_column("Long Pos. Qty")
        table.add_column("Short Pos. Qty")
        table.add_column("Long uPNL")
        table.add_column("Short uPNL")
        table.add_column("Long cum. uPNL")
        table.add_column("Short cum. uPNL")
        table.add_column("Long Pos. Price")
        table.add_column("Short Pos. Price")
        
        for symbol_data in shared_data.values():
            row = [
                symbol_data['symbol'],
                str(symbol_data.get('min_qty', 0)),
                str(symbol_data.get('current_price', 0)),
                str(symbol_data.get('balance', 0)),
                str(symbol_data.get('available_bal', 0)),
                str(symbol_data.get('volume', 0)),
                str(symbol_data.get('spread', 0)),
                str(symbol_data.get('trend', '')),
                str(symbol_data.get('long_pos_qty', 0)),
                str(symbol_data.get('short_pos_qty', 0)),
                str(symbol_data.get('long_upnl', 0)),
                str(symbol_data.get('short_upnl', 0)),
                str(symbol_data.get('long_cum_pnl', 0)),
                str(symbol_data.get('short_cum_pnl', 0)),
                str(symbol_data.get('long_pos_price', 0)),
                str(symbol_data.get('short_pos_price', 0))
            ]
            table.add_row(*row)
        return table
        
    def display_table(self):
        console = Console()
        with Live(self.table, refresh_per_second=3) as live:
            while True:
                time.sleep(1)
                with self.lock:
                    live.update(self.generate_table())

def simulate_data_update():
    while True:
        symbol = random.choice(list(shared_data.keys()))
        new_data = {
            'symbol': symbol,
            'min_qty': random.randint(1, 10),
            'current_price': round(random.uniform(100, 200), 2),
            'balance': round(random.uniform(10000, 20000), 2),
            'available_bal': round(random.uniform(5000, 10000), 2),
            'volume': round(random.uniform(100, 500), 2),
            'spread': round(random.uniform(0.1, 1), 2),
            'trend': "Up" if random.random() > 0.5 else "Down",
            'long_pos_qty': random.randint(0, 5),
            'short_pos_qty': random.randint(0, 5),
            'long_upnl': round(random.uniform(-100, 100), 2),
            'short_upnl': round(random.uniform(-100, 100), 2),
            'long_cum_pnl': round(random.uniform(-500, 500), 2),
            'short_cum_pnl': round(random.uniform(-500, 500), 2),
            'long_pos_price': round(random.uniform(100, 150), 2),
            'short_pos_price': round(random.uniform(150, 200), 2)
        }
        shared_data[symbol] = new_data
        time.sleep(1)

if __name__ == '__main__':
    shared_data = {
        'BTCUSD': {
            'symbol': 'BTCUSD',
            'min_qty': 0.001,
            'current_price': 52300
        },
        'ETHUSD': {
            'symbol': 'ETHUSD',
            'min_qty': 0.01,
            'current_price': 3000
        },
        'XRPUSD': {
            'symbol': 'XRPUSD'
        },
    }

    
    table_manager = LiveTableManager()
    display_thread = threading.Thread(target=table_manager.display_table)
    display_thread.daemon = True
    display_thread.start()
    
    data_simulator_thread = threading.Thread(target=simulate_data_update)
    data_simulator_thread.daemon = True
    data_simulator_thread.start()
    
    while True:
        time.sleep(1)
