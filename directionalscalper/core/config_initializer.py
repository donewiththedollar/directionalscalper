class ConfigInitializer:
    @staticmethod
    def initialize_config_attributes(strategy_instance, config):
        try:
            strategy_instance.levels = config.linear_grid['levels']
            strategy_instance.strength = config.linear_grid['strength']
            strategy_instance.long_mode = config.linear_grid['long_mode']
            strategy_instance.short_mode = config.linear_grid['short_mode']
            strategy_instance.reissue_threshold = config.linear_grid['reissue_threshold']
            strategy_instance.buffer_percentage = config.linear_grid['buffer_percentage']
            strategy_instance.enforce_full_grid = config.linear_grid['enforce_full_grid']
            strategy_instance.initial_entry_buffer_pct = config.linear_grid['initial_entry_buffer_pct']
            strategy_instance.min_buffer_percentage = config.linear_grid['min_buffer_percentage']
            strategy_instance.max_buffer_percentage = config.linear_grid['max_buffer_percentage']
            strategy_instance.wallet_exposure_limit_long = config.linear_grid['wallet_exposure_limit_long']
            strategy_instance.wallet_exposure_limit_short = config.linear_grid['wallet_exposure_limit_short']
            strategy_instance.max_qty_percent_long = config.linear_grid['max_qty_percent_long']
            strategy_instance.max_qty_percent_short = config.linear_grid['max_qty_percent_short']
            strategy_instance.max_usd_position_value_long = config.linear_grid.get('max_usd_position_value_long', None)
            strategy_instance.max_usd_position_value_short = config.linear_grid.get('max_usd_position_value_short', None)
            strategy_instance.one_symbol_optimization = config.linear_grid.get('one_symbol_optimization', False)
            strategy_instance.min_outer_price_distance = config.linear_grid['min_outer_price_distance']
            strategy_instance.min_outer_price_distance_long = config.linear_grid['min_outer_price_distance_long']
            strategy_instance.min_outer_price_distance_short = config.linear_grid['min_outer_price_distance_short']
            strategy_instance.max_outer_price_distance_long = config.linear_grid['max_outer_price_distance_long']
            strategy_instance.max_outer_price_distance_short = config.linear_grid['max_outer_price_distance_short']
            strategy_instance.graceful_stop_long = config.linear_grid['graceful_stop_long']
            strategy_instance.graceful_stop_short = config.linear_grid['graceful_stop_short']
            strategy_instance.entry_signal_type = config.linear_grid['entry_signal_type']
            strategy_instance.additional_entries_from_signal = config.linear_grid['additional_entries_from_signal']
            strategy_instance.auto_graceful_stop = config.linear_grid['auto_graceful_stop']
            strategy_instance.target_coins_mode = config.linear_grid['target_coins_mode']
            strategy_instance.buffer_percentage = config.linear_grid['buffer_percentage']
            strategy_instance.grid_behavior = config.linear_grid.get('grid_behavior', 'infinite')
            strategy_instance.stop_loss_enabled = config.linear_grid['stop_loss_enabled']
            strategy_instance.stop_loss_long = config.linear_grid['stop_loss_long']
            strategy_instance.stop_loss_short = config.linear_grid['stop_loss_short']
            strategy_instance.hedge_stop_loss = config.linear_grid['hedge_stop_loss']
            strategy_instance.enable_max_qty_stop_loss = config.linear_grid['enable_max_qty_stop_loss']
            strategy_instance.max_qty_percent_long_stop_loss = config.linear_grid['max_qty_percent_long_stop_loss']
            strategy_instance.max_qty_percent_short_stop_loss = config.linear_grid['max_qty_percent_short_stop_loss']
            strategy_instance.drawdown_behavior = config.linear_grid.get('drawdown_behavior', 'maxqtypercent')

            # >>> New auto-hedge fields <<<
            strategy_instance.auto_hedge_enabled = config.linear_grid['auto_hedge_enabled']
            strategy_instance.auto_hedge_ratio = config.linear_grid['auto_hedge_ratio']
            strategy_instance.auto_hedge_min_position_size = config.linear_grid['auto_hedge_min_position_size']
            strategy_instance.auto_hedge_price_diff_threshold = config.linear_grid['auto_hedge_price_diff_threshold']

            strategy_instance.DRAWDOWN_CLOSE_THRESHOLD = config.linear_grid['drawdown_close_threshold']
            # >>> Additional new toggles for hedging and grid logic <<<
            strategy_instance.disable_grid_on_hedge_side = config.linear_grid.get('disable_grid_on_hedge_side', False)
            strategy_instance.hedge_with_grid = config.linear_grid.get('hedge_with_grid', False)
            strategy_instance.hedge_levels = config.linear_grid.get('hedge_levels', 3)

            strategy_instance.forcibly_close_hedge = config.linear_grid.get('forcibly_close_hedge', False)

            strategy_instance.auto_shift_hedge = config.linear_grid.get('auto_shift_hedge', False)

            strategy_instance.side_with_grid = config.linear_grid.get('side_with_grid', False)

            strategy_instance.dynamic_grid = config.linear_grid.get('dynamic_grid', False)

            strategy_instance.reduce_opposing_side_by_20 = config.linear_grid.get('reduce_opposing_side_by_20', False)

            strategy_instance.signal_flip_close_at_loss = config.linear_grid.get('signal_flip_close_at_loss', False)

            # Sticky size configuration
            strategy_instance.sticky_size_enabled = config.linear_grid.get('sticky_size_enabled', False)
            strategy_instance.sticky_size_aggressiveness = config.linear_grid.get('sticky_size_aggressiveness', 1.0)
            strategy_instance.sticky_size_max_multiplier = config.linear_grid.get('sticky_size_max_multiplier', 5.0)
            strategy_instance.sticky_size_target_profit = config.linear_grid.get('sticky_size_target_profit', 0.001)
            strategy_instance.sticky_size_use_orderbook = config.linear_grid.get('sticky_size_use_orderbook', True)
            strategy_instance.sticky_size_min_volume_ratio = config.linear_grid.get('sticky_size_min_volume_ratio', 0.2)

            
            # Non-linear-grid config
            strategy_instance.upnl_threshold_pct = config.upnl_threshold_pct
            strategy_instance.volume_check = config.volume_check
            strategy_instance.max_usd_value = config.max_usd_value
            strategy_instance.blacklist = config.blacklist
            strategy_instance.test_orders_enabled = getattr(config, 'test_orders_enabled', False)
            strategy_instance.upnl_profit_pct = config.upnl_profit_pct
            strategy_instance.max_upnl_profit_pct = config.max_upnl_profit_pct
            strategy_instance.max_pos_balance_pct = config.max_pos_balance_pct

        except AttributeError as e:
            strategy_instance.logger.error(f"Failed to initialize attributes from config: {e}")
