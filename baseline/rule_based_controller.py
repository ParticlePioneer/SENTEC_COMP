"""
Rule-Based Controller for Smart Grid
Baseline controller that uses heuristic rules to dispatch power
"""
import numpy as np


class RuleBasedController:
    """
    Rule-based controller that prioritizes renewable energy and minimizes cost
    """
    
    def __init__(self, gen_max=800.0, grid_max=500.0, gen_cost=35.0, 
                 gen_emission=0.8, grid_emission=0.6):
        """
        Initialize rule-based controller
        
        Args:
            gen_max: Maximum generator capacity (kW)
            grid_max: Maximum grid import capacity (kW)
            gen_cost: Generator cost (PKR/kWh)
            gen_emission: Generator CO2 emission factor (kg/kWh)
            grid_emission: Grid CO2 emission factor (kg/kWh)
        """
        self.P_GEN_MAX = gen_max
        self.P_GRID_MAX = grid_max
        self.GEN_COST = gen_cost
        self.GEN_EMISSION = gen_emission
        self.GRID_EMISSION = grid_emission
    
    def get_action(self, demand, solar, wind, price):
        """
        Compute action based on current state using heuristic rules
        
        Strategy:
        1. Use all available renewable energy (solar + wind)
        2. If demand > renewables, prioritize grid import if price < gen_cost
        3. Use generator to fill remaining deficit
        
        Args:
            demand: Current demand (kW)
            solar: Available solar power (kW)
            wind: Available wind power (kW)
            price: Current grid price (PKR/kWh)
            
        Returns:
            Action array [P_gen, P_grid] in physical units (kW)
        """
        # Use all available renewable energy
        renewable_power = solar + wind
        remaining_demand = max(0.0, demand - renewable_power)
        
        # If no remaining demand, don't use generator or grid
        if remaining_demand <= 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Decide between grid and generator based on cost and emissions
        # Compare weighted costs (cost + emission factor)
        grid_weighted_cost = price + (self.GRID_EMISSION * 5.0)
        gen_weighted_cost = self.GEN_COST + (self.GEN_EMISSION * 5.0)
        
        P_gen = 0.0
        P_grid = 0.0
        
        # Optimal strategy: Use cheaper source first, then use both if needed
        # But optimally mix them to minimize total cost
        if remaining_demand <= self.P_GRID_MAX and grid_weighted_cost < gen_weighted_cost:
            # If demand can be met by grid alone and it's cheaper, use only grid
            P_grid = remaining_demand
        elif remaining_demand <= self.P_GEN_MAX and gen_weighted_cost < grid_weighted_cost:
            # If demand can be met by generator alone and it's cheaper, use only generator
            P_gen = remaining_demand
        else:
            # Need both sources - use optimal mix
            # Use cheaper source up to its capacity, then use the other
            if grid_weighted_cost < gen_weighted_cost:
                # Grid is cheaper - use grid first, then generator
                P_grid = min(remaining_demand, self.P_GRID_MAX)
                remaining_after_grid = remaining_demand - P_grid
                if remaining_after_grid > 0:
                    P_gen = min(remaining_after_grid, self.P_GEN_MAX)
            else:
                # Generator is cheaper - use generator first, then grid
                P_gen = min(remaining_demand, self.P_GEN_MAX)
                remaining_after_gen = remaining_demand - P_gen
                if remaining_after_gen > 0:
                    P_grid = min(remaining_after_gen, self.P_GRID_MAX)
        
        return np.array([P_gen, P_grid], dtype=np.float32)
    
    def get_normalized_action(self, demand, solar, wind, price):
        """
        Get normalized action [0, 1] for compatibility with RL environment
        
        Args:
            demand: Current demand (kW)
            solar: Available solar power (kW)
            wind: Available wind power (kW)
            price: Current grid price (PKR/kWh)
            
        Returns:
            Normalized action array [P_gen_norm, P_grid_norm]
        """
        action = self.get_action(demand, solar, wind, price)
        
        # Normalize to [0, 1]
        normalized_action = np.array([
            action[0] / self.P_GEN_MAX,
            action[1] / self.P_GRID_MAX
        ], dtype=np.float32)
        
        return np.clip(normalized_action, 0.0, 1.0)

