import gymnasium as gym

class SmartGridEnv(gym.Env):
    def __init__(self):
        super().__init__()
        print("SmartGridEnv initialized")
