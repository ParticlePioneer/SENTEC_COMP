import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.smart_grid_env import SmartGridEnv

def run_evaluation(model_path, vecnorm_path, output_dir):
    print(f"Loading model: {model_path}")
    
    # Setup Env
    def make_env():
        return SmartGridEnv()
    
    vec_env = DummyVecEnv([make_env])
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, vec_env)
        env.training = False
        env.norm_reward = False
    else:
        env = vec_env
        print("Warning: No VecNormalize statistics found!")

    # Load Model
    model = PPO.load(model_path, env=env)
    
    # Run Episode
    obs = env.reset()
    done = False
    last_info = {}
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        last_info = info[0]

    # Process Logs
    original_env = env.unwrapped.envs[0]
    cols = ["t", "f_base", "f_ppo", "SOC_base", "SOC_ppo", "cost_base", "cost_ppo", "emis_base", "emis_ppo", "deficit_base", "deficit_ppo"]
    df = pd.DataFrame(original_env.log, columns=cols)
    
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "evaluation_log.csv"), index=False)
    
    # Save Summary Metrics for the user
    summary_metrics = pd.DataFrame([
        {
            "Agent": "Baseline",
            "Total_Cost": last_info["total_cost_base"],
            "Total_Emissions": last_info["total_emis_base"],
            "Total_Deficit": last_info["total_def_base"]
        },
        {
            "Agent": "RL_PPO",
            "Total_Cost": last_info["total_cost_ppo"],
            "Total_Emissions": last_info["total_emis_ppo"],
            "Total_Deficit": last_info["total_def_ppo"]
        }
    ])
    summary_metrics.to_csv(os.path.join(output_dir, "comparison_metrics.csv"), index=False)
    print(f"Summary metrics saved to {os.path.join(output_dir, 'comparison_metrics.csv')}")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # 1. Frequency
    axes[0].plot(df['t'], df['f_base'], label='Baseline', linestyle='--')
    axes[0].plot(df['t'], df['f_ppo'], label='PPO (RL)', linewidth=2)
    axes[0].axhline(y=50.0, color='r', linestyle=':', alpha=0.5)
    axes[0].set_title('Grid Frequency Dynamics' if 't' in df else 'Frequency')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].legend()
    axes[0].grid(True)

    # 2. SOC
    axes[1].plot(df['t'], df['SOC_base'], label='Baseline', linestyle='--')
    axes[1].plot(df['t'], df['SOC_ppo'], label='PPO (RL)', linewidth=2)
    axes[1].set_title('Battery SOC')
    axes[1].set_ylabel('SOC')
    axes[1].legend()
    axes[1].grid(True)

    # 3. Deficit
    axes[2].bar(df['t'] - 0.2, df['deficit_base'], width=0.4, label='Baseline')
    axes[2].bar(df['t'] + 0.2, df['deficit_ppo'], width=0.4, label='PPO (RL)')
    axes[2].set_title('Power Deficit (kW)')
    axes[2].set_ylabel('Deficit')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dynamics_comparison.png"))
    
    # Print Summary Table
    print("\n" + "="*40)
    print("      EVALUATION SUMMARY")
    print("="*40)
    print(f"{'Metric':<20} | {'Baseline':<12} | {'PPO':<12} | {'Diff (%)':<10}")
    print("-" * 60)
    
    metrics = [
        ("Total Cost", "total_cost_base", "total_cost_ppo"),
        ("Total Emissions", "total_emis_base", "total_emis_ppo"),
        ("Total Deficit", "total_def_base", "total_def_ppo")
    ]
    
    for label, base_key, ppo_key in metrics:
        b = last_info[base_key]
        p = last_info[ppo_key]
        diff = ((p - b) / b) * 100 if b != 0 else 0
        print(f"{label:<20} | {b:>12.2f} | {p:>12.2f} | {diff:>+8.1f}%")
    print("="*40)

if __name__ == "__main__":
    # Check results/models and fallback to root
    model_path = "./results/models/ppo_smartgrid_diff_model.zip"
    if not os.path.exists(model_path):
        model_path = "./ppo_smartgrid_diff_model.zip"
        
    stats_path = "./results/models/ppo_smartgrid_diff_vecnormalize.pkl"
    if not os.path.exists(stats_path):
        stats_path = "./ppo_smartgrid_diff_vecnormalize.pkl"

    run_evaluation(
        model_path=model_path,
        vecnorm_path=stats_path,
        output_dir="./results/evaluation"
    )
