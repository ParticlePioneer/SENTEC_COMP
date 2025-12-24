import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SmartGridEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Observation space: normalized [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        print("SmartGridEnv initialized with observation space")
