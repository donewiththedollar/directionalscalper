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
            strategy_instance.min_buffer_percentage_ar = config.linear_grid['min_buffer_percentage_ar']
            strategy_instance.max_buffer_percentage_ar = config.linear_grid['max_buffer_percentage_ar']
            strategy_instance.upnl_auto_reduce_threshold_long = config.linear_grid['upnl_auto_reduce_threshold_long']
            strategy_instance.upnl_auto_reduce_threshold_short = config.linear_grid['upnl_auto_reduce_threshold_short']
            strategy_instance.failsafe_enabled = config.linear_grid['failsafe_enabled']
            strategy_instance.long_failsafe_upnl_pct = config.linear_grid['long_failsafe_upnl_pct']
            strategy_instance.short_failsafe_upnl_pct = config.linear_grid['short_failsafe_upnl_pct']
            strategy_instance.failsafe_start_pct = config.linear_grid['failsafe_start_pct']
            strategy_instance.auto_reduce_cooldown_enabled = config.linear_grid['auto_reduce_cooldown_enabled']
            strategy_instance.auto_reduce_cooldown_start_pct = config.linear_grid['auto_reduce_cooldown_start_pct']
            strategy_instance.max_qty_percent_long = config.linear_grid['max_qty_percent_long']
            strategy_instance.max_qty_percent_short = config.linear_grid['max_qty_percent_short']
            strategy_instance.min_outer_price_distance = config.linear_grid['min_outer_price_distance']
            strategy_instance.max_outer_price_distance_long = config.linear_grid['max_outer_price_distance_long']
            strategy_instance.max_outer_price_distance_short = config.linear_grid['max_outer_price_distance_short']
            strategy_instance.graceful_stop_long = config.linear_grid['graceful_stop_long']
            strategy_instance.graceful_stop_short = config.linear_grid['graceful_stop_short']
            strategy_instance.entry_signal_type = config.linear_grid['entry_signal_type']
            strategy_instance.additional_entries_from_signal = config.linear_grid['additional_entries_from_signal']
            strategy_instance.auto_graceful_stop = config.linear_grid['auto_graceful_stop']
            strategy_instance.target_coins_mode = config.linear_grid['target_coins_mode']
            strategy_instance.grid_behavior = config.linear_grid.get('grid_behavior', 'infinite')
            strategy_instance.stop_loss_enabled = config.linear_grid['stop_loss_enabled']
            strategy_instance.stop_loss_long = config.linear_grid['stop_loss_long']
            strategy_instance.stop_loss_short = config.linear_grid['stop_loss_short']
            strategy_instance.drawdown_behavior = config.linear_grid.get('drawdown_behavior', 'maxqtypercent')
            strategy_instance.upnl_threshold_pct = config.upnl_threshold_pct
            strategy_instance.volume_check = config.volume_check
            strategy_instance.max_usd_value = config.max_usd_value
            strategy_instance.blacklist = config.blacklist
            strategy_instance.test_orders_enabled = getattr(config, 'test_orders_enabled', False)
            strategy_instance.upnl_profit_pct = config.upnl_profit_pct
            strategy_instance.max_upnl_profit_pct = config.max_upnl_profit_pct
            strategy_instance.stoploss_enabled = config.stoploss_enabled
            strategy_instance.stoploss_upnl_pct = config.stoploss_upnl_pct
            strategy_instance.liq_stoploss_enabled = config.liq_stoploss_enabled
            strategy_instance.liq_price_stop_pct = config.liq_price_stop_pct
            strategy_instance.auto_reduce_enabled = config.auto_reduce_enabled
            strategy_instance.auto_reduce_start_pct = config.auto_reduce_start_pct
            strategy_instance.auto_reduce_maxloss_pct = config.auto_reduce_maxloss_pct
            strategy_instance.entry_during_autoreduce = config.entry_during_autoreduce
            strategy_instance.auto_reduce_marginbased_enabled = config.auto_reduce_marginbased_enabled
            strategy_instance.auto_reduce_wallet_exposure_pct = config.auto_reduce_wallet_exposure_pct
            strategy_instance.percentile_auto_reduce_enabled = config.percentile_auto_reduce_enabled
            strategy_instance.max_pos_balance_pct = config.max_pos_balance_pct
        except AttributeError as e:
            strategy_instance.logger.error(f"Failed to initialize attributes from config: {e}")
