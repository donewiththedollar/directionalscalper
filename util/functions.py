from colorama import Fore

def print_lot_sizes(max_trade_qty, market_data):
    print(f"Min Trade Qty: {market_data[2]}")
    print_lot_size(1, Fore.LIGHTRED_EX, max_trade_qty, market_data)
    print_lot_size(0.01, Fore.LIGHTCYAN_EX, max_trade_qty, market_data)
    print_lot_size(0.005, Fore.LIGHTCYAN_EX, max_trade_qty, market_data)
    print_lot_size(0.002, Fore.LIGHTGREEN_EX, max_trade_qty, market_data)
    print_lot_size(0.001, Fore.LIGHTGREEN_EX, max_trade_qty, market_data)
    
def calc_lot_size(lot_size, max_trade_qty, market_data):
    trade_qty_x = max_trade_qty / (1.0 / lot_size)
    trade_qty_x_round = round(trade_qty_x, int(float(market_data[2])))
    return trade_qty_x, trade_qty_x_round

def print_lot_size(lot_size, color, max_trade_qty, market_data):
    not_enough_equity = Fore.RED + "({:.5g}) You cannot trade this lot size, not enough equity"
    trade_qty_x, trade_qty_x_round = calc_lot_size(lot_size, max_trade_qty, market_data)
    if trade_qty_x_round == 0:
        trading_not_possible = not_enough_equity.format(trade_qty_x)
        color = Fore.RED
    else:
        trading_not_possible = ""
    print(color + "{:.4g}x : {:.4g} {}".format(lot_size, trade_qty_x_round, trading_not_possible))
