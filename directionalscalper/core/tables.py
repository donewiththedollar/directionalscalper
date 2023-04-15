import logging

from rich.table import Table

from directionalscalper.core.functions import calc_lot_size

log = logging.getLogger(__name__)


# TODO make bot instance object to prevent long param list
def generate_main_table(data: dict, manager) -> Table:
    try:
        table = Table(show_header=False, box=None, title=data["version"])
        table.add_row(generate_table_info(data=data)),
        table.add_row(
            generate_table_vol(
                manager=manager,
                min_vol_dist_data=data["min_vol_dist_data"],
                min_volume=data["min_volume"],
                min_distance=data["min_distance"],
                symbol=data["symbol"],
                mode=data["mode"],
            )
        )
        return table
    except Exception as e:
        log.warning(f"{e}")


def generate_table_vol(
    manager, min_vol_dist_data, min_volume, min_distance, symbol, mode
) -> Table:
    try:
        current_volume = manager.get_asset_value(
            symbol=symbol, data=manager.get_data(), value="1mVol"
        )

        min_trade_qty = manager.get_asset_value(
            symbol=symbol, data=manager.get_data(), value="Min qty"
        )

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
            str(current_volume).split(".")[0],
            "[red]TOO LOW" if current_volume < min_volume else "[green]VOL. OK",
        )
        table.add_row(
            "Min Trade Qty",
            str(min_trade_qty),
            "",
            "",
        )
        table.add_row()
        table.add_row(
            "Min Dist.",
            str(min_distance),
            "{:.4f}".format(
                find_spread(symbol=symbol, timeframe="5m", manager=manager)
            ),
            "[red]TOO SMALL"
            if find_spread(symbol=symbol, timeframe="5m", manager=manager)
            < min_distance
            else "[green]DIST. OK",
        )
        table.add_row("Mode", str(mode))

        return table
    except Exception as e:
        log.warning(f"{e}")


# # Generate table
# def generate_table_vol(
#     manager, min_vol_dist_data, min_volume, min_distance, symbol, mode
# ) -> Table:
#     try:
#         table = Table(width=50)
#         table.add_column("Condition", justify="center")
#         table.add_column("Config", justify="center")
#         table.add_column("Current", justify="center")
#         table.add_column("Status")
#         table.add_row(
#             "Trading:",
#             str(min_vol_dist_data),
#             str(),
#             "[green]:heavy_check_mark:" if min_vol_dist_data else "off",
#         )
#         table.add_row(
#             "Min Vol.",
#             str(min_volume),
#             str(
#                 manager.get_asset_value(
#                     symbol=symbol, data=manager.get_data(), value="1mVol"
#                 )
#             ).split(".")[0],
#             "[red]TOO LOW"
#             if manager.get_asset_value(
#                 symbol=symbol, data=manager.get_data(), value="1mVol"
#             )
#             < min_volume
#             else "[green]VOL. OK",
#         )
#         table.add_row()
#         table.add_row(
#             "Min Dist.",
#             str(min_distance),
#             "{:.4f}".format(
#                 find_spread(symbol=symbol, timeframe="5m", manager=manager)
#             ),
#             "[red]TOO SMALL"
#             if find_spread(symbol=symbol, timeframe="5m", manager=manager)
#             < min_distance
#             else "[green]DIST. OK",
#         )
#         table.add_row("Mode", str(mode))
#         # table.add_row(f"Long mode:", str(long_mode), str(), "[green]:heavy_check_mark:" if long_mode else "off")
#         # table.add_row(f"Short mode:", str(short_mode), str(), "[green]:heavy_check_mark:" if short_mode else "off")
#         # table.add_row(f"Hedge mode:", str(hedge_mode), str(), "[green]:heavy_check_mark:" if hedge_mode else "off")
#         #    table.add_row(f"Telegram:", str(tgnotif))
#         return table
#     except Exception as e:
#         log.warning(f"{e}")


def generate_table_info(data: dict) -> Table:
    try:
        if (
            data["short_symbol_cum_realised"] > -0.0001
            and data["short_symbol_cum_realised"] <= 0
        ):
            short_symbol_cum_realised = 0
        else:
            short_symbol_cum_realised = data["short_symbol_cum_realised"]
        if (
            data["long_symbol_cum_realised"] > -0.0001
            and data["long_symbol_cum_realised"] <= 0
        ):
            long_symbol_cum_realised = 0
        else:
            long_symbol_cum_realised = data["long_symbol_cum_realised"]
        if data["long_symbol_realised"] > -0.0001 and data["long_symbol_realised"] <= 0:
            long_symbol_realised = 0
        else:
            long_symbol_realised = data["long_symbol_realised"]
        if (
            data["short_symbol_realised"] > -0.0001
            and data["short_symbol_realised"] <= 0
        ):
            short_symbol_realised = 0
        else:
            short_symbol_realised = data["short_symbol_realised"]

        total_cum_realised = short_symbol_cum_realised + long_symbol_cum_realised

        total_unpl = data["short_pos_unpl"] + data["long_pos_unpl"]
        short_pos_unpl_pct, long_pos_unpl_pct = (
            data["short_pos_unpl_pct"],
            data["long_pos_unpl_pct"],
        )
        trade_qty_001x, trade_qty_001x_round = calc_lot_size(
            0.001, data["max_trade_qty"], data["market_data"]
        )

        table = Table(show_header=False, width=50)
        table.add_column(justify="right")
        table.add_column(justify="left")
        table.add_row("Symbol", f"{data['symbol']}")
        table.add_row("Balance", "${:.2f}".format(data["dex_wallet"]))
        table.add_row("Equity", "${:.2f}".format(data["dex_equity"]))
        table.add_row(
            "Realised cum.",
            f"[red]{'${:.4f}'.format(total_cum_realised)}"
            if total_cum_realised < 0
            else f"[green]{'${:.4f}'.format(total_cum_realised)}",
        )
        table.add_row(
            "Long Realised recent",
            f"[red]{'${:.4f}'.format(long_symbol_realised)}"
            if long_symbol_realised < 0
            else f"[green]{'${:.4f}'.format(long_symbol_realised)}",
        )
        table.add_row(
            "Short Realised recent",
            f"[red]{'${:.4f}'.format(short_symbol_realised)}"
            if short_symbol_realised < 0
            else f"[green]{'${:.4f}'.format(short_symbol_realised)}",
        )
        table.add_row(
            "Unrealised P&L USDT",
            f"[red]{'${:.4f}'.format(total_unpl)}"
            if total_unpl < 0
            else f"[green]{'${:.4f}'.format(total_unpl)}",
        )
        table.add_row(
            "Long Unrealized P&L %",
            f"[red]{long_pos_unpl_pct:.2f}%"
            if long_pos_unpl_pct < 0
            else f"[green]{'{:.2f}%'.format(long_pos_unpl_pct)}",
        )
        table.add_row(
            "Short Unrealized P&L %",
            f"[red]{'{:.2f}%'.format(short_pos_unpl_pct)}"
            if short_pos_unpl_pct < 0
            else f"[green]{'{:.2f}%'.format(short_pos_unpl_pct)}",
        )
        table.add_row(
            "Entry size",
            f"{data['trade_qty']}"
            if float(data["trade_qty"]) > trade_qty_001x
            else f"[red]{data['trade_qty']}",
        )
        table.add_row("Long size", "{:.4g}".format(data["long_pos_qty"]))
        table.add_row("Short size", "{:.4g}".format(data["short_pos_qty"]))
        table.add_row("Long position price", "${:.4f}".format(data["long_pos_price"]))
        table.add_row(
            "Long liquidation price", "${:.4f}".format(data["long_liq_price"])
        )
        table.add_row("Short position price", "${:.4f}".format(data["short_pos_price"]))
        table.add_row(
            "Short liquidation price", "${:.4f}".format(data["short_liq_price"])
        )
        table.add_row("Max", "{:.4g}".format(data["max_trade_qty"]))

        table.add_row(
            "0.001x",
            "[red]{:.4g} ({:.4g})".format(trade_qty_001x_round, trade_qty_001x)
            if trade_qty_001x_round == 0
            else "{:.4g}".format(trade_qty_001x_round),
        )
        # table.add_row("Trend:", str(tyler_trend))
        table.add_row("Trend", f"{data['trend']}")

        return table
    except Exception as e:
        log.warning(f"{e}")


def find_spread(manager, symbol: str, timeframe: str = "5m"):
    spread = timeframe + "Spread"
    try:
        return manager.get_asset_value(
            symbol=symbol, data=manager.get_data(), value=spread
        )
    except Exception as e:
        log.warning(f"{e}")
    return None


#  global dex_btc_balance, dex_btc_upnl, dex_btc_wallet, dex_btc_equity
#    global inv_perp_equity, inv_perp_available_balance, inv_perp_used_margin, inv_perp_order_margin,
# inv_perp_order_margin, inv_perp_position_margin, inv_perp_occ_closing_fee, inv_perp_occ_funding_fee,
#  inv_perp_wallet_balance, inv_perp_realised_pnl, inv_perp_unrealised_pnl, inv_perp_cum_realised_pnl
# global sell_position_size, sell_position_prce
# global dex_btc_balance, dex_btc_upnl, dex_btc_wallet, dex_btc_equity


def generate_inverse_table_info(
    symbol,
    dex_btc_balance,
    dex_btc_equity,
    inv_perp_cum_realised_pnl,
    dex_btc_upnl_pct,
    trade_qty,
    position_size,
    trend,
    position_prce,
    tp_price,
    is_long,
) -> Table:
    try:
        inverse_table = Table(show_header=False, width=50)
        inverse_table.add_column(justify="right")
        inverse_table.add_column(justify="left")
        inverse_table.add_row("Symbol", str(symbol))
        inverse_table.add_row("Balance", "${:.8f}".format(float(dex_btc_balance)))
        inverse_table.add_row("Equity", "${:.8f}".format(dex_btc_equity))
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
            f"[red]{'${:.4f}%'.format(dex_btc_upnl_pct)}"
            if dex_btc_upnl_pct < 0
            else f"[green]{'{:.4f}%'.format(dex_btc_upnl_pct)}",
        )
        inverse_table.add_row("Entry size", str(trade_qty))
        inverse_table.add_row("Trend:", str(trend))
        inverse_table.add_row("Entry price", str(position_prce))
        inverse_table.add_row("Take profit", "${:.8f}".format(tp_price))
        if is_long:
            inverse_table.add_row("Long pos size", str(position_size))
        else:
            inverse_table.add_row("Short pos size", str(position_size))
        # inverse_table.add_row(f"Bid:", str)
        return inverse_table
    except Exception as e:
        log.warning(f"{e}")
