# AI Smart Grid Reinforcement Learning (RL) Optimizer

## Project Overview
The AI Smart Grid RL Optimizer is an advanced reinforcement learning–based control system designed to optimize power dispatch in a simulated micro-grid environment. The goal is to minimize a combined cost function of CO₂ emissions and operational cost while maintaining grid stability.

## Problem Statement
Modern smart grids must efficiently balance energy demand, operational cost, and environmental sustainability. Traditional controllers lack adaptability in dynamic conditions. This project applies Reinforcement Learning (RL) to autonomously learn optimal control strategies.

## Task Description
- Design a custom smart grid simulation environment
- Implement an RL agent for active control
- Minimize combined CO₂ emissions and operational cost
- Compare RL agent performance with a baseline controller

## Key Features
- **Custom Gymnasium Environment**: Realistic smart grid simulation with renewable energy sources
- **PPO Agent**: Proximal Policy Optimization agent with custom neural network architecture
- **Baseline Controller**: Rule-based heuristic controller for performance comparison
- **Comprehensive Evaluation**: Metrics calculation and visualization tools
- **Interactive Dashboard**: Streamlit-based web dashboard for real-time visualization
- **Training Infrastructure**: TensorBoard logging and model checkpointing

## Key Metrics
- Cost reduction (%)
- CO₂ emission reduction (%)
- Learning convergence (reward vs episodes)
- Grid stability (deficit minimization)

## Technology Stack
- **Reinforcement Learning**: Stable-Baselines3 (PPO algorithm)
- **Deep Learning**: PyTorch
- **Environment**: Gymnasium (custom RL environment)
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly, Matplotlib
- **Dashboard**: Streamlit
- **Logging**: TensorBoard

## Project Structure
```
AI-Smart-Grid-RL-Optimizer/
├── env/
│   ├── __init__.py
│   └── smart_grid_env.py          # Custom Gymnasium environment
├── agents/
│   ├── __init__.py
│   └── ppo_agent.py               # PPO agent implementation
├── baseline/
│   ├── __init__.py
│   └── rule_based_controller.py   # Rule-based baseline controller
├── training/
│   ├── __init__.py
│   └── train_agent.py             # Training script
├── evaluation/
│   ├── __init__.py
│   └── compare_results.py         # Evaluation and comparison script
├── dashboard/
│   ├── __init__.py
│   └── app.py                     # Streamlit dashboard
├── utils/
│   ├── __init__.py
│   └── evaluation_utils.py        # Evaluation utility functions
├── results/
│   ├── models/                    # Saved trained models
│   └── evaluation/                # Evaluation results and plots
├── logs/
│   ├── tensorboard/               # TensorBoard logs
│   ├── eval/                      # Evaluation logs
│   └── checkpoints/               # Model checkpoints
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Train the RL Agent

Train the PPO agent on the smart grid environment:

```bash
python training/train_agent.py
```

**Advanced training options**:
```bash
# Custom training parameters
python training/train_agent.py --timesteps 500000 --lr 1e-4 --save-path ./results/models/my_model
```

**Parameters**:
- `--timesteps`: Total number of training timesteps (default: 200000)
- `--lr`: Learning rate (default: 3e-4)
- `--save-path`: Path to save the trained model (default: ./results/models/ppo_smartgrid)

The trained model will be saved to `./results/models/ppo_smartgrid.zip` (or specified path).

**Monitor training progress**:
```bash
tensorboard --logdir=./logs/tensorboard/
```
Then open http://localhost:6006 in your browser.

### 2. Evaluate and Compare Results

Compare the trained RL agent with the baseline controller:

```bash
python evaluation/compare_results.py
```

**Advanced evaluation options**:
```bash
# Custom evaluation parameters
python evaluation/compare_results.py --model-path ./results/models/ppo_smartgrid.zip --episodes 200 --output-dir ./results/evaluation
```

**Parameters**:
- `--model-path`: Path to trained RL model (default: ./results/models/ppo_smartgrid.zip)
- `--episodes`: Number of evaluation episodes (default: 100)
- `--output-dir`: Output directory for results (default: ./results/evaluation)

This will generate:
- `comparison_metrics.csv`: Detailed metrics in CSV format
- `cost_comparison.png`: Cost comparison visualization
- `emissions_comparison.png`: Emissions comparison visualization
- `improvement_summary.html`: Interactive improvement summary

### 3. Launch Interactive Dashboard

Run the Streamlit dashboard for interactive visualization:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:
- **Performance Comparison**: Compare RL agent vs baseline controller
- **Interactive Simulation**: Run single-step simulations with custom parameters
- **Training Metrics**: Links to TensorBoard visualization

## Environment Details

### State Space (Observation)
The agent observes:
- `demand`: Current power demand (normalized 0-1)
- `solar`: Available solar power (normalized 0-1)
- `wind`: Available wind power (normalized 0-1)
- `price`: Current grid electricity price (normalized 0-1)
- `time_step`: Current time step (normalized 0-1)
- `battery_soc`: Battery state of charge (placeholder, normalized 0-1)

### Action Space
The agent controls:
- `P_gen`: Generator power dispatch (normalized 0-1, maps to 0-800 kW)
- `P_grid`: Grid import power (normalized 0-1, maps to 0-500 kW)

### Reward Function
The reward combines:
- **Cost**: Operational cost (generator + grid import)
- **Emissions**: CO₂ emissions from generator and grid
- **Penalty**: Large penalty for unmet demand (deficit)

### Environment Parameters
- Generator max capacity: 800 kW
- Grid max import: 500 kW
- Generator cost: 35 PKR/kWh
- Generator emission: 0.8 kg CO₂/kWh
- Grid emission: 0.6 kg CO₂/kWh
- Episode length: 24 steps (one day, hourly resolution)

## RL Agent Architecture

### PPO (Proximal Policy Optimization)
- **Algorithm**: PPO from Stable-Baselines3
- **Policy Network**: Multi-layer perceptron (MLP)
  - Hidden layers: [64, 64] (both policy and value networks)
  - Activation: Tanh
  - Architecture: Input (6) → FC(64) → FC(64) → Output (2)

### Hyperparameters (Default)
- Learning rate: 3e-4
- Steps per update: 2048
- Batch size: 64
- Optimization epochs: 10
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- PPO clip range: 0.2
- Entropy coefficient: 0.01
- Value function coefficient: 0.5

## Baseline Controller

The rule-based baseline controller uses the following strategy:
1. Use all available renewable energy (solar + wind)
2. Prioritize grid import if total cost (price + weighted emissions) < generator cost
3. Use generator to fill remaining deficit
4. Minimize unmet demand

This provides a competitive baseline for comparison with the RL agent.

## Results Interpretation

### Key Performance Indicators
- **Cost Reduction**: Percentage reduction in total operational cost
- **Emission Reduction**: Percentage reduction in CO₂ emissions
- **Deficit Reduction**: Percentage reduction in unmet demand

Positive values indicate the RL agent outperforms the baseline controller.

## Future Enhancements
- Multi-agent reinforcement learning for distributed control
- Battery storage integration for energy arbitrage
- Real-time pricing signals and demand response
- Cloud deployment and scalable architecture
- Integration with real smart grid data
- Advanced reward shaping for stability constraints

## Troubleshooting

### Model Not Found Error
If you get an error about model not found, make sure to train the agent first:
```bash
python training/train_agent.py
```

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues
By default, PyTorch will use CPU. For GPU acceleration, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## License
This project is open source and available for educational and research purposes.

## Conclusion
This project demonstrates how reinforcement learning can significantly improve smart grid efficiency, reduce emissions, and deliver business value in modern energy systems. The PPO agent learns to optimize power dispatch by balancing cost and environmental impact, outperforming traditional rule-based controllers.

## Contact & Contributions
For questions, issues, or contributions, please open an issue or submit a pull request.

---

**Built with using Stable-Baselines3, Gymnasium, and Streamlit**
