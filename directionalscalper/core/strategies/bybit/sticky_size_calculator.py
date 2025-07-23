"""
Sticky Size Calculator - Dynamic position sizing to optimize average entry price
Pulls average position price toward current market price for faster profit realization
"""

import logging
from typing import Dict, Tuple, Optional, List
import numpy as np


class StickySizeCalculator:
    """
    Calculates optimal order sizes to "stick" the average position price
    close to the current market price for rapid profit capture
    """
    
    def __init__(self, config: Dict):
        self.enabled = config.get('sticky_size_enabled', False)
        self.aggressiveness = config.get('sticky_size_aggressiveness', 1.0)  # 0.5 = conservative, 2.0 = aggressive
        self.max_multiplier = config.get('sticky_size_max_multiplier', 5.0)  # Max size increase
        self.target_profit_pct = config.get('sticky_size_target_profit', 0.001)  # 0.1% default
        self.use_orderbook = config.get('sticky_size_use_orderbook', True)
        self.min_volume_ratio = config.get('sticky_size_min_volume_ratio', 0.2)  # Min 20% of orderbook volume
        
    def calculate_sticky_size(
        self,
        symbol: str,
        side: str,
        current_price: float,
        grid_level_price: float,
        base_order_size: float,
        existing_position_qty: float,
        existing_position_avg_price: float,
        orderbook: Optional[Dict] = None,
        recent_volatility: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Calculate the optimal order size to pull average price toward profit
        
        Args:
            symbol: Trading symbol
            side: 'long' or 'short'
            current_price: Current market price
            grid_level_price: Price where we want to place the order
            base_order_size: Original calculated order size
            existing_position_qty: Current position quantity
            existing_position_avg_price: Current average entry price
            orderbook: Optional orderbook data for volume analysis
            recent_volatility: Optional recent price volatility
            
        Returns:
            Tuple of (adjusted_size, reason_string)
        """
        
        if not self.enabled:
            return base_order_size, "Sticky size disabled"
            
        if existing_position_qty == 0:
            return base_order_size, "No existing position"
            
        # Calculate current position value
        position_value = existing_position_qty * existing_position_avg_price
        
        # Calculate target average price for immediate profit
        if side == 'long':
            # For long positions, we want avg price below current price
            target_avg_price = current_price * (1 - self.target_profit_pct)
            
            # Only apply sticky size if new entry improves average
            if grid_level_price >= existing_position_avg_price:
                return base_order_size, "Grid price not improving average"
                
        else:  # short
            # For short positions, we want avg price above current price
            target_avg_price = current_price * (1 + self.target_profit_pct)
            
            # Only apply sticky size if new entry improves average
            if grid_level_price <= existing_position_avg_price:
                return base_order_size, "Grid price not improving average"
        
        # Calculate required size to achieve target average
        required_size = self._calculate_required_size(
            existing_position_qty,
            existing_position_avg_price,
            grid_level_price,
            target_avg_price
        )
        
        # Apply aggressiveness factor
        adjusted_size = base_order_size + (required_size - base_order_size) * self.aggressiveness
        
        # Apply maximum multiplier constraint
        max_allowed_size = base_order_size * self.max_multiplier
        adjusted_size = min(adjusted_size, max_allowed_size)
        
        # Apply orderbook-based constraints if available
        if self.use_orderbook and orderbook:
            orderbook_limit = self._calculate_orderbook_limit(
                side, grid_level_price, orderbook
            )
            if orderbook_limit:
                adjusted_size = min(adjusted_size, orderbook_limit)
                
        # Apply volatility-based adjustments
        if recent_volatility:
            volatility_multiplier = 1 + (recent_volatility * 2)  # Higher volatility = larger sizes
            adjusted_size *= volatility_multiplier
            
        # Ensure minimum size
        adjusted_size = max(adjusted_size, base_order_size)
        
        # Calculate expected new average
        new_position_qty = existing_position_qty + adjusted_size
        new_avg_price = (
            (existing_position_qty * existing_position_avg_price + adjusted_size * grid_level_price) 
            / new_position_qty
        )
        
        # Calculate improvement percentage
        if side == 'long':
            improvement_pct = ((existing_position_avg_price - new_avg_price) / existing_position_avg_price) * 100
        else:
            improvement_pct = ((new_avg_price - existing_position_avg_price) / existing_position_avg_price) * 100
            
        reason = (
            f"Sticky size: {base_order_size:.4f} → {adjusted_size:.4f} "
            f"({adjusted_size/base_order_size:.1f}x), "
            f"Avg improvement: {improvement_pct:.2f}%"
        )
        
        logging.info(f"[{symbol}] {reason}")
        
        return adjusted_size, reason
    
    def _calculate_required_size(
        self,
        existing_qty: float,
        existing_avg: float,
        new_price: float,
        target_avg: float
    ) -> float:
        """
        Calculate the exact size needed to achieve target average price
        
        Formula: 
        (existing_qty * existing_avg + new_qty * new_price) / (existing_qty + new_qty) = target_avg
        
        Solving for new_qty:
        new_qty = existing_qty * (existing_avg - target_avg) / (target_avg - new_price)
        """
        
        denominator = target_avg - new_price
        if abs(denominator) < 0.0000001:  # Avoid division by zero
            return 0
            
        required_qty = existing_qty * (existing_avg - target_avg) / denominator
        
        # Ensure positive quantity
        return max(0, required_qty)
    
    def _calculate_orderbook_limit(
        self,
        side: str,
        price: float,
        orderbook: Dict
    ) -> Optional[float]:
        """
        Calculate maximum size based on orderbook liquidity
        """
        
        try:
            book_side = 'asks' if side == 'long' else 'bids'
            
            if book_side not in orderbook:
                return None
                
            # Find volume available near our price level
            total_volume = 0
            price_tolerance = 0.001  # 0.1% price range
            
            for level_price, level_volume in orderbook[book_side]:
                level_price = float(level_price)
                level_volume = float(level_volume)
                
                # Check if this level is within our price range
                price_diff_pct = abs(level_price - price) / price
                
                if price_diff_pct <= price_tolerance:
                    total_volume += level_volume
                    
            # Limit our order to a percentage of available volume
            if total_volume > 0:
                return total_volume * self.min_volume_ratio
                
        except Exception as e:
            logging.error(f"Error calculating orderbook limit: {e}")
            
        return None
    
    def calculate_grid_with_sticky_sizes(
        self,
        symbol: str,
        side: str,
        grid_levels: List[float],
        base_sizes: List[float],
        current_price: float,
        existing_position_qty: float,
        existing_position_avg_price: float,
        orderbook: Optional[Dict] = None
    ) -> List[Tuple[float, float]]:
        """
        Calculate sticky sizes for entire grid at once
        
        Returns:
            List of (price, adjusted_size) tuples
        """
        
        if not self.enabled:
            return list(zip(grid_levels, base_sizes))
            
        adjusted_grid = []
        cumulative_qty = existing_position_qty
        cumulative_avg = existing_position_avg_price
        
        for level_price, base_size in zip(grid_levels, base_sizes):
            # Calculate sticky size for this level
            adjusted_size, reason = self.calculate_sticky_size(
                symbol=symbol,
                side=side,
                current_price=current_price,
                grid_level_price=level_price,
                base_order_size=base_size,
                existing_position_qty=cumulative_qty,
                existing_position_avg_price=cumulative_avg,
                orderbook=orderbook
            )
            
            adjusted_grid.append((level_price, adjusted_size))
            
            # Update cumulative position for next level
            if cumulative_qty > 0:
                cumulative_avg = (
                    (cumulative_qty * cumulative_avg + adjusted_size * level_price) 
                    / (cumulative_qty + adjusted_size)
                )
            else:
                cumulative_avg = level_price
                
            cumulative_qty += adjusted_size
            
        return adjusted_grid
    
    def get_sticky_stats(
        self,
        original_sizes: List[float],
        adjusted_sizes: List[float]
    ) -> Dict:
        """
        Calculate statistics about sticky size adjustments
        """
        
        original_total = sum(original_sizes)
        adjusted_total = sum(adjusted_sizes)
        
        multipliers = [adj/orig if orig > 0 else 1 for orig, adj in zip(original_sizes, adjusted_sizes)]
        
        return {
            'total_multiplier': adjusted_total / original_total if original_total > 0 else 1,
            'avg_multiplier': np.mean(multipliers),
            'max_multiplier': max(multipliers),
            'min_multiplier': min(multipliers),
            'original_total': original_total,
            'adjusted_total': adjusted_total,
            'size_increase': adjusted_total - original_total,
            'size_increase_pct': ((adjusted_total - original_total) / original_total * 100) if original_total > 0 else 0
        }