from rich.table import Table
import tylerapi

# TODO make bot instance object to prevent long param list
def generate_main_table(version, short_pos_unpl, long_pos_unpl, short_pos_unpl_pct, long_pos_unpl_pct, symbol, dex_wallet, 
                        dex_equity, short_symbol_cum_realised, long_symbol_realised, short_symbol_realised,
                        trade_qty, long_pos_qty, short_pos_qty, long_pos_price, long_liq_price, short_pos_price, 
                        short_liq_price, max_trade_qty, market_data, trend, min_vol_dist_data,
                        min_volume, min_distance, mode) -> Table:
    table = Table(show_header=False, box=None, title=version)
    table.add_row(generate_table_info(short_pos_unpl, long_pos_unpl, short_pos_unpl_pct, long_pos_unpl_pct, symbol, dex_wallet, 
                        dex_equity, short_symbol_cum_realised, long_symbol_realised, short_symbol_realised,
                        trade_qty, long_pos_qty, short_pos_qty, long_pos_price, long_liq_price, short_pos_price, 
                        short_liq_price, max_trade_qty, market_data, trend)),
    table.add_row(generate_table_vol(min_vol_dist_data, min_volume, min_distance, symbol, mode))
    return table

# Generate table
def generate_table_vol(min_vol_dist_data, min_volume, min_distance, symbol, mode) -> Table:
    table = Table(width=50)
    table.add_column("Condition", justify="center")
    table.add_column("Config", justify="center")
    table.add_column("Current", justify="center")
    table.add_column("Status")
    table.add_row(
        "Trading:",
        str(min_vol_dist_data),
        str(),
        "[green]:heavy_check_mark:" if min_vol_dist_data else "off",
    )
    table.add_row(
        "Min Vol.",
        str(min_volume),
        str(tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())).split(
            "."
        )[0],
        "[red]TOO LOW"
        if tylerapi.get_asset_volume_1m_1x(symbol, tylerapi.grab_api_data())
           < min_volume
        else "[green]VOL. OK",
    )
    table.add_row()
    table.add_row(
        "Min Dist.",
        str(min_distance),
        '{:.2f}'.format(find_5m_spread(symbol)),
        "[red]TOO SMALL" if find_5m_spread(symbol) < min_distance else "[green]DIST. OK",
    )
    table.add_row("Mode", str(mode))
    # table.add_row(f"Long mode:", str(long_mode), str(), "[green]:heavy_check_mark:" if long_mode else "off")
    # table.add_row(f"Short mode:", str(short_mode), str(), "[green]:heavy_check_mark:" if short_mode else "off")
    # table.add_row(f"Hedge mode:", str(hedge_mode), str(), "[green]:heavy_check_mark:" if hedge_mode else "off")
    #    table.add_row(f"Telegram:", str(tgnotif))
    return table



def generate_table_info(short_pos_unpl, long_pos_unpl, short_pos_unpl_pct, long_pos_unpl_pct, symbol, dex_wallet, 
                        dex_equity, short_symbol_cum_realised, long_symbol_realised, short_symbol_realised,
                        trade_qty, long_pos_qty, short_pos_qty, long_pos_price, long_liq_price, short_pos_price, 
                        short_liq_price, max_trade_qty, market_data, trend) -> Table:
    total_unpl = short_pos_unpl + long_pos_unpl
    total_unpl_pct = short_pos_unpl_pct + long_pos_unpl_pct
    table = Table(show_header=False, width=50)
    table.add_column(justify="right")
    table.add_column(justify="left")
    table.add_row("Symbol", str(symbol))
    table.add_row("Balance", '${:.2f}'.format(dex_wallet))
    table.add_row("Equity", '${:.2f}'.format(dex_equity))
    table.add_row(
        "Realised cum.",
        f"[red]{'${:.2f}'.format(short_symbol_cum_realised)}"
        if short_symbol_cum_realised < 0
        else f"[green]{'${:.2f}'.format(short_symbol_cum_realised)}",
    )
    table.add_row(
        "Long Realised recent",
        f"[red]{'${:.2f}'.format(long_symbol_realised)}"
        if long_symbol_realised < 0
        else f"[green]{'${:.2f}'.format(long_symbol_realised)}",
    )
    table.add_row(
        "Short Realised recent",
        f"[red]{'${:.2f}'.format(short_symbol_realised)}"
        if short_symbol_realised < 0
        else f"[green]{'${:.2f}'.format(short_symbol_realised)}",
    )
    table.add_row(
        "Unrealised USDT",
        f"[red]{'${:.2f}'.format(total_unpl)}"
        if total_unpl < 0
        else f"[green]{'${:.2f}'.format(total_unpl)}",
    )
    table.add_row(
        "Unrealised %",
        f"[red]{'{:.2f}%'.format(total_unpl_pct)}"
        if total_unpl_pct < 0
        else f"[green]{'{:.2f}%'.format(total_unpl_pct)}",
    )
    table.add_row("Entry size", str(trade_qty))
    table.add_row("Long size", '{:.2f}'.format(long_pos_qty))
    table.add_row("Short size", '{:.2f}'.format(short_pos_qty))
    table.add_row("Long position price", '${:.2f}'.format(long_pos_price))
    table.add_row("Long liquidation price", '${:.5f}'.format(long_liq_price))
    table.add_row("Short position price", '${:.2f}'.format(short_pos_price))
    table.add_row("Short liquidation price", '${:.2f}'.format(short_liq_price))
    table.add_row("Max", '{:.2f}'.format(max_trade_qty))
    table.add_row(
        "0.001x", str(round(max_trade_qty / 500, int(float(market_data[2]))))
    )
    # table.add_row("Trend:", str(tyler_trend))
    table.add_row("Trend", str(trend))

    return table



def find_1m_spread(symbol):
    tylerapi.grab_api_data()
    tyler_1m_spread = tylerapi.get_asset_1m_spread(symbol, tylerapi.grab_api_data())

    return tyler_1m_spread

def find_5m_spread(symbol):
    tylerapi.grab_api_data()
    tyler_spread = tylerapi.get_asset_5m_spread(symbol, tylerapi.grab_api_data())

    return tyler_spread




#  global dex_btc_balance, dex_btc_upnl, dex_btc_wallet, dex_btc_equity
#    global inv_perp_equity, inv_perp_available_balance, inv_perp_used_margin, inv_perp_order_margin,
# inv_perp_order_margin, inv_perp_position_margin, inv_perp_occ_closing_fee, inv_perp_occ_funding_fee,
#  inv_perp_wallet_balance, inv_perp_realised_pnl, inv_perp_unrealised_pnl, inv_perp_cum_realised_pnl
# global sell_position_size, sell_position_prce
# global dex_btc_balance, dex_btc_upnl, dex_btc_wallet, dex_btc_equity


def generate_inverse_table_info(symbol, dex_btc_balance, dex_btc_equity, inv_perp_cum_realised_pnl, dex_btc_upnl_pct,
                                trade_qty, sell_position_size, trend, sell_position_prce, tp_price) -> Table:
    inverse_table = Table(show_header=False, width=50)
    inverse_table.add_column(justify="right")
    inverse_table.add_column(justify="left")
    inverse_table.add_row("Symbol", str(symbol))
    inverse_table.add_row("Balance", '${:.8f}'.format(float(dex_btc_balance)))
    inverse_table.add_row("Equity", '${:.8f}'.format(dex_btc_equity))
    # inverse_table.add_row(f"Realised cum.", f"[red]{str(inv_perp_cum_realised_pnl)}" if inv_perp_cum_realised_pnl < 0 else f"[green]{str(short_symbol_cum_realised)}")
    inverse_table.add_row(
        "Realised cum.",
        f"[red]${format(inv_perp_cum_realised_pnl, '.8f')}"
        if inv_perp_cum_realised_pnl < 0
        else f"[green]${format(inv_perp_cum_realised_pnl, '.8f')}",
    )
    # inverse_table.add_row(f"Unrealized PNL.", f"[red]{dex_btc_upnl}" if dex_btc_upnl < 0 else f"[green]{dex_btc_upnl}")
    # inverse_table.add_row(f"Realised recent", f"[red]{str(inv_perp_realised_pnl)}" if inv_perp_realised_pnl < 0 else f"[green]{str(inv_perp_realised_pnl)}")
    # inverse_table.add_row(f"Unrealised BTC", f"[red]{str(inv_perp_unrealised_pnl)}" if inv_perp_unrealised_pnl < 0 else f"[green]{str(short_pos_unpl + short_pos_unpl_pct)}")
    # inverse_table.add_row(f"Unrealised BTC", f"[red]{str(dex_btc_upnl)}" if dex_btc_upnl < 0 else f"[green]{str(dex_btc_upnl + dex_btc_upnl_pct)}")
    inverse_table.add_row(
        "Unrealised %",
        f"[red]{'${:.2f}%'.format(dex_btc_upnl_pct)}"
        if dex_btc_upnl_pct < 0
        else f"[green]{'{:.2f}%'.format(dex_btc_upnl_pct)}",
    )
    inverse_table.add_row("Entry size", str(trade_qty))
    inverse_table.add_row("Short pos size", str(sell_position_size))
    inverse_table.add_row("Trend:", str(trend))
    inverse_table.add_row("Entry price", str(sell_position_prce))
    inverse_table.add_row("Take profit", '${:.8f}'.format(tp_price))
    # inverse_table.add_row(f"Bid:", str)
    return inverse_table
