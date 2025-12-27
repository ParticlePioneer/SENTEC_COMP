"""
Streamlit Dashboard for Smart Grid RL Optimizer
Interactive visualization of agent performance and smart grid simulation
"""
import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.smart_grid_env import SmartGridEnv
from agents.ppo_agent import PPOAgent
from baseline.rule_based_controller import RuleBasedController
from utils.evaluation_utils import run_episode, evaluate_agent, calculate_improvement


# Page configuration
st.set_page_config(
    page_title="AI Smart Grid RL Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


def load_model_if_exists(model_path):
    """Load RL model if it exists"""
    if os.path.exists(model_path):
        try:
            env = SmartGridEnv()
            agent = PPOAgent(env)
            agent.load(model_path)
            return agent, env, True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, False
    return None, None, False


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">⚡ AI Smart Grid RL Optimizer Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="./results/models/ppo_smartgrid.zip",
        help="Path to trained PPO model"
    )
    
    num_episodes = st.sidebar.slider(
        "Evaluation Episodes",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
        help="Number of episodes for evaluation"
    )
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Performance Comparison", "🎮 Interactive Simulation", "📈 Training Metrics"]
    )
    
    # Home page
    if page == "🏠 Home":
        st.header("Welcome to Smart Grid RL Optimizer")
        st.markdown("""
        This dashboard provides interactive visualization and analysis of the Reinforcement Learning 
        agent for smart grid optimization.
        
        **Features:**
        - Performance comparison between RL agent and baseline controller
        - Interactive simulation of smart grid operation
        - Real-time metrics and visualization
        - Training progress analysis
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Objective", "Cost & Emission Minimization")
        
        with col2:
            st.metric("RL Algorithm", "PPO (Proximal Policy Optimization)")
        
        with col3:
            st.metric("Environment", "Custom Smart Grid Simulator")
    
    # Performance Comparison page
    elif page == "📊 Performance Comparison":
        st.header("Performance Comparison: RL Agent vs Baseline")
        
        if st.button("Run Evaluation", type="primary"):
            with st.spinner("Running evaluation... This may take a few minutes."):
                # Create environment
                env = SmartGridEnv()
                
                # Load RL agent
                rl_agent, _, loaded = load_model_if_exists(model_path)
                if not loaded:
                    st.error(f"Could not load model from {model_path}. Please train the agent first.")
                    st.info("Run: `python training/train_agent.py` to train the agent.")
                    return
                
                # Create baseline controller
                baseline_controller = RuleBasedController(
                    gen_max=env.P_GEN_MAX,
                    grid_max=env.P_GRID_MAX,
                    gen_cost=env.GEN_COST,
                    gen_emission=env.GEN_EMISSION,
                    grid_emission=env.GRID_EMISSION
                )
                
                # Evaluate both agents
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Baseline Controller")
                    baseline_metrics = evaluate_agent(env, baseline_controller, num_episodes=num_episodes)
                    
                    st.metric("Mean Total Cost", f"{baseline_metrics['mean_total_cost']:.2f} PKR",
                             delta=f"±{baseline_metrics['std_total_cost']:.2f}")
                    st.metric("Mean Total Emissions", f"{baseline_metrics['mean_total_emissions']:.2f} kg CO₂",
                             delta=f"±{baseline_metrics['std_total_emissions']:.2f}")
                    st.metric("Mean Total Deficit", f"{baseline_metrics['mean_total_deficit']:.2f} kW",
                             delta=f"±{baseline_metrics['std_total_deficit']:.2f}")
                
                with col2:
                    st.subheader("RL Agent (PPO)")
                    rl_metrics = evaluate_agent(env, rl_agent, num_episodes=num_episodes, deterministic=True)
                    
                    st.metric("Mean Total Cost", f"{rl_metrics['mean_total_cost']:.2f} PKR",
                             delta=f"±{rl_metrics['std_total_cost']:.2f}")
                    st.metric("Mean Total Emissions", f"{rl_metrics['mean_total_emissions']:.2f} kg CO₂",
                             delta=f"±{rl_metrics['std_total_emissions']:.2f}")
                    st.metric("Mean Total Deficit", f"{rl_metrics['mean_total_deficit']:.2f} kW",
                             delta=f"±{rl_metrics['std_total_deficit']:.2f}")
                
                # Calculate improvements
                improvements = calculate_improvement(baseline_metrics, rl_metrics)
                
                st.subheader("Improvements")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    improvement_value = improvements['cost_reduction_percent']
                    st.metric("Cost Reduction", f"{improvement_value:.2f}%",
                             delta=f"{improvement_value:.2f}% improvement" if improvement_value > 0 else "No improvement",
                             delta_color="normal" if improvement_value > 0 else "inverse")
                
                with col2:
                    improvement_value = improvements['emission_reduction_percent']
                    st.metric("Emission Reduction", f"{improvement_value:.2f}%",
                             delta=f"{improvement_value:.2f}% improvement" if improvement_value > 0 else "No improvement",
                             delta_color="normal" if improvement_value > 0 else "inverse")
                
                with col3:
                    improvement_value = improvements['deficit_reduction_percent']
                    st.metric("Deficit Reduction", f"{improvement_value:.2f}%",
                             delta=f"{improvement_value:.2f}% improvement" if improvement_value > 0 else "No improvement",
                             delta_color="normal" if improvement_value > 0 else "inverse")
                
                # Visualization
                st.subheader("Comparison Charts")
                
                # Bar chart comparison
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Cost Comparison', 'Emissions Comparison'),
                    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
                )
                
                fig.add_trace(
                    go.Bar(name='Baseline', x=['Baseline', 'RL Agent'],
                          y=[baseline_metrics['mean_total_cost'], rl_metrics['mean_total_cost']],
                          error_y=dict(type='data', 
                                      array=[baseline_metrics['std_total_cost'], rl_metrics['std_total_cost']]),
                          marker_color='#ff7f0e'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(name='RL Agent', x=['Baseline', 'RL Agent'],
                          y=[baseline_metrics['mean_total_emissions'], rl_metrics['mean_total_emissions']],
                          error_y=dict(type='data',
                                      array=[baseline_metrics['std_total_emissions'], rl_metrics['std_total_emissions']]),
                          marker_color='#2ca02c'),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Agent", row=1, col=1)
                fig.update_xaxes(title_text="Agent", row=1, col=2)
                fig.update_yaxes(title_text="Cost (PKR)", row=1, col=1)
                fig.update_yaxes(title_text="Emissions (kg CO₂)", row=1, col=2)
                fig.update_layout(height=400, showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Interactive Simulation page
    elif page == "🎮 Interactive Simulation":
        st.header("Interactive Smart Grid Simulation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environment Parameters")
            demand = st.slider("Demand (kW)", 300, 700, 500)
            solar = st.slider("Solar Power (kW)", 0, 300, 150)
            wind = st.slider("Wind Power (kW)", 0, 200, 100)
            price = st.slider("Grid Price (PKR/kWh)", 40, 70, 55)
        
        with col2:
            st.subheader("Agent Selection")
            agent_type = st.radio("Select Agent", ["RL Agent (PPO)", "Baseline Controller"])
            
            if st.button("Run Single Step Simulation", type="primary"):
                env = SmartGridEnv()
                
                # Manually set environment state (hack for demo)
                env.demand = demand
                env.solar = solar
                env.wind = wind
                env.price = price
                env.time_step = 0
                
                obs = env._get_observation()
                
                if agent_type == "RL Agent (PPO)":
                    rl_agent, _, loaded = load_model_if_exists(model_path)
                    if loaded:
                        action, _ = rl_agent.predict(obs, deterministic=True)
                        P_gen = action[0] * env.P_GEN_MAX
                        P_grid = action[1] * env.P_GRID_MAX
                    else:
                        st.error("Could not load RL model. Using baseline instead.")
                        agent_type = "Baseline Controller"
                
                if agent_type == "Baseline Controller":
                    baseline_controller = RuleBasedController(
                        gen_max=env.P_GEN_MAX,
                        grid_max=env.P_GRID_MAX,
                        gen_cost=env.GEN_COST,
                        gen_emission=env.GEN_EMISSION,
                        grid_emission=env.GRID_EMISSION
                    )
                    action = baseline_controller.get_normalized_action(demand, solar, wind, price)
                    P_gen = action[0] * env.P_GEN_MAX
                    P_grid = action[1] * env.P_GRID_MAX
                
                # Calculate metrics
                total_supply = P_gen + P_grid + solar + wind
                deficit = max(0.0, demand - total_supply)
                gen_cost = P_gen * env.GEN_COST
                grid_cost = P_grid * price
                total_cost = gen_cost + grid_cost
                emissions = P_gen * env.GEN_EMISSION + P_grid * env.GRID_EMISSION
                
                # Display results
                st.subheader("Simulation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Generator Power", f"{P_gen:.2f} kW")
                with col2:
                    st.metric("Grid Import", f"{P_grid:.2f} kW")
                with col3:
                    st.metric("Total Cost", f"{total_cost:.2f} PKR")
                with col4:
                    st.metric("Emissions", f"{emissions:.2f} kg CO₂")
                
                # Power flow chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Solar', 'Wind', 'Generator', 'Grid'],
                    y=[solar, wind, P_gen, P_grid],
                    marker_color=['#ffd700', '#87ceeb', '#ff7f0e', '#2ca02c'],
                    name='Power Sources'
                ))
                
                fig.add_trace(go.Scatter(
                    x=['Demand'],
                    y=[demand],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='x'),
                    name='Demand'
                ))
                
                fig.update_layout(
                    title='Power Flow Visualization',
                    yaxis_title='Power (kW)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Training Metrics page
    elif page == "📈 Training Metrics":
        st.header("Training Metrics")
        st.info("Training metrics are logged to TensorBoard. Check the `./logs/tensorboard/` directory.")
        st.markdown("""
        To view training metrics in TensorBoard, run:
        ```bash
        tensorboard --logdir=./logs/tensorboard/
        ```
        Then open the URL shown in the terminal (usually http://localhost:6006)
        """)


if __name__ == '__main__':
    main()

