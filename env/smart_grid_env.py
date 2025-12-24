import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SmartGridEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # Physical Limits
        self.P_GEN_MAX = 800.0      # kW, generator max capacity
        self.P_GRID_MAX = 500.0     # kW, grid max import
        self.max_steps = 24         # one day simulation (hourly)

        # State variables
        self.time_step = 0
        self.demand = 0.0
        self.solar = 0.0
        self.wind = 0.0
        self.price = 0.0  # PKR/kWh, updated in reset()

        # Costs (PKR/kWh)
        self.GEN_COST = 35.0
        # self.price will be used dynamically as GRID_COST

        # Emissions (kg CO2/kWh)
        self.GEN_EMISSION = 0.8
        self.GRID_EMISSION = 0.6

        # Penalty for unmet demand
        self.DEFICIT_PENALTY = 10.0

        # Action space: normalized [0,1] for P_gen and P_grid
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation space: normalized [0,1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        print("SmartGridEnv initialized with observation space")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time_step = 0

        # Initialize microgrid state
        self.demand = np.random.uniform(300, 700)   # kW
        self.solar = np.random.uniform(0, 300)      # kW
        self.wind = np.random.uniform(0, 200)       # kW
        self.price = np.random.uniform(40, 70)      # PKR/kWh

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        self.time_step += 1

        # Scale normalized actions to physical units
        P_gen = action[0] * self.P_GEN_MAX
        P_grid = action[1] * self.P_GRID_MAX

        # Total supply
        total_supply = P_gen + P_grid + self.solar + self.wind
        deficit = max(0.0, self.demand - total_supply)

        # Cost computation
        gen_cost = P_gen * self.GEN_COST
        grid_cost = P_grid * self.price
        total_cost = gen_cost + grid_cost

        # CO2 emissions
        emissions = P_gen * self.GEN_EMISSION + P_grid * self.GRID_EMISSION

        # Deficit penalty
        deficit_penalty = self.DEFICIT_PENALTY * deficit

        # Final reward (negative because we minimize cost & emissions)
        reward = -(
            0.001 * total_cost +       # scaled for numerical stability
            0.01 * emissions +
            deficit_penalty
        )

        # Update environment state for next step
        self.demand = np.random.uniform(300, 700)
        self.solar = np.random.uniform(0, 300)
        self.wind = np.random.uniform(0, 200)
        self.price = np.random.uniform(40, 70)

        obs = self._get_observation()

        terminated = False
        truncated = self.time_step >= self.max_steps

        info = {
            "P_gen": P_gen,
            "P_grid": P_grid,
            "cost": total_cost,
            "emissions": emissions,
            "deficit": deficit
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        # Normalized observation: [demand, solar, wind, price, time_step, battery_placeholder]
        obs = np.array([
            self.demand / 1000.0,        # normalize to 0-1
            self.solar / 1000.0,
            self.wind / 1000.0,
            self.price / 100.0,           # normalize ~0.4-0.7
            self.time_step / self.max_steps,
            1.0                           # placeholder for battery SOC
        ], dtype=np.float32)
        return np.clip(obs,0.0,1.0)
