"""
PPO Agent for Smart Grid Optimization
Uses Stable-Baselines3 PPO with custom policy network
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn


class CustomPPOPolicy(nn.Module):
    """
    Custom neural network policy for PPO
    Architecture: Input -> FC(64) -> FC(64) -> Output
    """
    def __init__(self, features_extractor_class, features_extractor_kwargs, *args, **kwargs):
        super().__init__()
        # This is handled by Stable-Baselines3's MlpPolicy
        pass


def create_ppo_agent(env, learning_rate=3e-4, n_steps=2048, batch_size=64, 
                     n_epochs=10, gamma=0.99, gae_lambda=0.95, 
                     clip_range=0.2, ent_coef=0.05, vf_coef=0.5,
                     policy_kwargs=None, verbose=1):
    """
    Create and configure a PPO agent
    
    Args:
        env: Gymnasium environment
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to collect per update
        batch_size: Minibatch size
        n_epochs: Number of epochs when optimizing the surrogate loss
        gamma: Discount factor
        gae_lambda: Factor for trade-off of bias vs variance for GAE
        clip_range: Clipping parameter for PPO
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function coefficient
        policy_kwargs: Additional arguments for the policy
        verbose: Verbosity level
        
    Returns:
        PPO agent instance
    """
    if policy_kwargs is None:
        # Custom policy network: [128, 128] hidden layers for better representation
        # Larger value network to better approximate the value function
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128, 64])],
            activation_fn=nn.Tanh,
        )
    
    # Wrap environment in Monitor for logging
    monitored_env = Monitor(env)
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        monitored_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log="./logs/tensorboard/",
    )
    
    return model


class PPOAgent:
    """
    Wrapper class for PPO agent with convenience methods
    """
    def __init__(self, env, **kwargs):
        """
        Initialize PPO agent
        
        Args:
            env: Gymnasium environment
            **kwargs: Additional arguments for PPO configuration
        """
        self.env = env
        self.model = create_ppo_agent(env, **kwargs)
    
    def train(self, total_timesteps=100000, log_interval=10, save_path=None):
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total number of timesteps to train
            log_interval: Log every N updates
            save_path: Path to save the trained model
        """
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:  # Only create directory if path contains a directory
                os.makedirs(save_dir, exist_ok=True)
        
        # Create evaluation callback
        eval_env = Monitor(self.env)
        os.makedirs('./logs/eval/', exist_ok=True)
        best_model_path = None
        if save_path:
            best_model_path = save_path.replace('.zip', '_best') if save_path.endswith('.zip') else save_path + '_best'
            best_model_dir = os.path.dirname(best_model_path)
            if best_model_dir:
                os.makedirs(best_model_dir, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path='./logs/eval/',
            eval_freq=5000,
            deterministic=True,
            render=False
        )
        
        # Create checkpoint callback
        os.makedirs('./logs/checkpoints/', exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./logs/checkpoints/',
            name_prefix='ppo_smartgrid'
        )
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            log_interval=log_interval
        )
        
        # Save final model
        if save_path:
            self.model.save(save_path)
            print(f"Model saved to {save_path}")
    
    def predict(self, observation, deterministic=True):
        """
        Predict action given observation
        
        Args:
            observation: Current state observation
            deterministic: Use deterministic policy
            
        Returns:
            Action array
        """
        return self.model.predict(observation, deterministic=deterministic)
    
    def load(self, path):
        """
        Load a trained model
        
        Args:
            path: Path to the saved model
        """
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def save(self, path):
        """
        Save the current model
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
        print(f"Model saved to {path}")

