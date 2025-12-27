import pandas as pd
import sys
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to sys.path to support direct execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.smart_grid_env import SmartGridEnv

# Ensure absolute paths for results
results_dir = os.path.join(project_root, "results")
models_dir = os.path.join(results_dir, "models")
eval_dir = os.path.join(results_dir, "evaluation")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# =========================
# Initialize environment
# =========================
def make_env():
    return SmartGridEnv()

vec_env = DummyVecEnv([make_env])
# VecNormalize: norm_reward=False is CRITICAL for stable learning with large penalties
env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

# =========================
# Initialize PPO agent
# =========================
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    batch_size=128, 
    learning_rate=3e-4, 
    n_steps=2048
)

# =========================
# Train the agent
# =========================
total_timesteps = 600000
model.learn(total_timesteps=total_timesteps)

# =========================
# Save the trained model and stats
# =========================
model_path = os.path.join(models_dir, "ppo_smartgrid_diff_model")
stats_path = os.path.join(models_dir, "ppo_smartgrid_diff_vecnormalize.pkl")
model.save(model_path)
env.save(stats_path)
print(f"Model saved to {model_path}.zip")

# =========================
# Run one episode for evaluation
# =========================
obs = env.reset()
done = [False] # DummyVecEnv returns array
info = {}

while not done[0]:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

# =========================
# Convert log to DataFrame
# =========================
original_env = env.unwrapped.envs[0]
columns = ["t", "f_base", "f_ppo", "SOC_base", "SOC_ppo", "cost_base", "cost_ppo", "emis_base", "emis_ppo", "deficit_base", "deficit_ppo"]
df = pd.DataFrame(original_env.log, columns=columns)

log_path = os.path.join(eval_dir, "ppo_vs_baseline_log_diff.csv")
df.to_csv(log_path, index=False)
print(f"Episode log saved as {log_path}")

# Final totals
last_info = info[0]
print("\nFINAL RESULTS (24h Episode):")
print(f"Total Cost - Baseline: {last_info['total_cost_base']:.2f}")
print(f"Total Cost - PPO     : {last_info['total_cost_ppo']:.2f}")
print(f"Total Emissions - Baseline: {last_info['total_emis_base']:.2f}")
print(f"Total Emissions - PPO     : {last_info['total_emis_ppo']:.2f}")
print(f"Total Deficit - Baseline  : {last_info['total_def_base']:.2f}")
print(f"Total Deficit - PPO       : {last_info['total_def_ppo']:.2f}")