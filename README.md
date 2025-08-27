# Lux AI Season 3 - Multi-Agent Reinforcement Learning Solution

A comprehensive implementation of intelligent agents for the Lux AI Challenge Season 3, featuring multiple reinforcement learning approaches including Deep Q-Networks (DQN), Proximal Policy Optimization (PPO), and Behavior Tree-based decision making.

*I cloned the PPO training pipeline from this great notebook during the competition [Simple_ppo_training](https://www.kaggle.com/code/junhanzangai/simple-ppo-training).*

## 🌟 Project Overview

This project implements advanced AI agents for the Lux AI Season 3 competition, where teams compete in deep space exploration to discover ancient relics while managing resources and engaging in strategic combat. The challenge involves navigating randomly generated 2D maps with dynamic features like asteroids, nebula fields, and energy nodes.

### 🎯 Key Features

- **Multi-Agent Architecture**: Support for different agent implementations including heuristic, DQN, and PPO-based approaches
- **Behavior Tree Integration**: Hierarchical decision-making system combining rule-based logic with neural network guidance
- **Advanced State Representation**: Sophisticated encoding of game state including fog of war, energy management, and spatial relationships
- **Dynamic Strategy Adaptation**: Agents that adapt their strategies based on match progression and opponent behavior
- **Training Infrastructure**: Complete training pipeline with experience replay, GAE (Generalized Advantage Estimation), and model checkpointing

## 🏗️ Architecture

### Agent Implementations

1. **PyTree Agent** (`agent.py`): Hybrid approach combining behavior trees with DQN for strategic decision making
2. **PPO Agent** (`PPO_agent.py`): Advanced reinforcement learning agent using Proximal Policy Optimization
3. **Borna Agent** (`Borna_agent.py`): Heuristic-based agent with sophisticated exploration strategies
4. **Training System** (`train.py`): Comprehensive training infrastructure with multi-agent support

### Core Components

```
agent/
├── main.py              # Kaggle submission entry point
├── agent.py             # PyTree Agent (Behavior Tree + DQN)
├── PPO_agent.py         # PPO-based agent implementation
├── Borna_agent.py       # Heuristic agent with exploration logic
├── train.py             # Training infrastructure
└── checkpoints/         # Model checkpoints and training states
```

## 🚀 Getting Started

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision numpy
pip install py-trees  # For behavior tree implementation
pip install luxai-s3  # Lux AI Season 3 environment
pip install matplotlib pandas  # For training visualization
```

### Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd kaggle_lux_ai_neurips
   ```

2. **Train an Agent**:
   ```bash
   cd agent
   python train.py  # Train PPO agent
   python agent.py  # Train PyTree agent
   ```

3. **Run Evaluation**:
   ```bash
   python main.py  # Run trained agent
   ```

## 🧠 Agent Strategies

### PyTree Agent (Hybrid AI)
- **Behavior Tree**: Hierarchical decision-making with priority-based action selection
- **DQN Integration**: Neural network guidance for exploration and tactical decisions
- **Adaptive Exploration**: Smart exploration using visited maps and Q-value guidance
- **Multi-objective Optimization**: Balances exploration, relic collection, and combat

### PPO Agent (Deep RL)
- **Actor-Critic Architecture**: Separate networks for policy and value estimation
- **Advanced State Encoding**: Multi-layered representation of game state
- **Experience Replay**: Efficient learning from collected experiences
- **GAE**: Generalized Advantage Estimation for stable training

### Borna Agent (Heuristic Expert)
- **Rule-based Logic**: Hand-crafted strategies based on game mechanics understanding
- **Dynamic Adaptation**: Strategies that change based on match progression
- **Energy Management**: Sophisticated energy conservation and utilization
- **Relic Discovery**: Systematic exploration and exploitation patterns

## 🎮 Game Mechanics Handled

### Core Features
- **Fog of War**: Limited visibility with strategic information gathering
- **Energy Management**: Resource optimization for movement and combat
- **Relic Discovery**: Hidden scoring tiles around relic nodes
- **Combat System**: Energy-based sapping with area-of-effect damage
- **Dynamic Environment**: Moving asteroids and nebula fields

### Strategic Elements
- **Multi-match Games**: Learning and adaptation across 5-match sequences
- **Randomized Parameters**: Handling variable game mechanics between matches
- **Team Coordination**: Multi-unit strategy and positioning
- **Risk Assessment**: Balancing exploration vs exploitation

## 🔧 Technical Implementation

### State Representation
```python
# Multi-dimensional state encoding
state = {
    'unit_positions': unit_coords,
    'energy_levels': energy_values,
    'visibility_map': fog_of_war_mask,
    'relic_positions': discovered_relics,
    'enemy_positions': visible_enemies,
    'map_features': terrain_data
}
```

### Action Space
- **Movement**: 5 directional actions (stay, up, right, down, left)
- **Combat**: Targeted energy sapping with range and area effects
- **Resource Management**: Energy conservation and strategic positioning

### Reward Engineering
- **Relic Points**: Primary scoring mechanism
- **Energy Efficiency**: Bonus for optimal resource usage
- **Exploration Bonus**: Rewards for discovering new areas
- **Combat Effectiveness**: Rewards for successful enemy engagement

## 📚 References

- [Lux AI Season 3 Official Documentation](https://github.com/Lux-AI-Challenge/Lux-Design-S3)
- [Proximal Policy Optimization Paper](https://arxiv.org/abs/1707.06347)
- [Deep Q-Networks](https://arxiv.org/abs/1312.5602)
- [Behavior Trees in AI](https://en.wikipedia.org/wiki/Behavior_tree_(artificial_intelligence,_robotics_and_control))
- [LUX AI Winning Solution] ()

## 🤝 Contributing

This project represents a comprehensive exploration of multi-agent reinforcement learning in competitive environments. The codebase is structured to support further research and development in:

- Advanced RL algorithms
- Multi-agent coordination
- Strategic game AI
- Competition optimization

## 📄 License

This project is developed for educational and research purposes as part of the Lux AI Challenge Season 3 competition.

---

*Built with ❤️ for the Lux AI Challenge Season 3 - Where ancient relics await discovery in the depths of space.*
