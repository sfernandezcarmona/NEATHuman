# NEAT Reinforcement Learning for Humanoid Standup Task

This repository contains Python scripts implementing NEAT (NeuroEvolution of Augmenting Topologies) for training an agent to perform the humanoid standup task in the OpenAI Gym environment.

## 📝 Overview

The NEAT algorithm is utilized to evolve neural network architectures and weights to control an agent in the "HumanoidStandup-v4" environment. The training process involves multiple generations of evolving neural networks, with rewards assigned based on the agent's performance in the environment.

## 📁 Files

### 1. `train_neat.py`

- **Purpose**: Implements the NEAT algorithm for training the agent.
- **Key Components**:
  - NEAT configuration setup.
  - Environment initialization.
  - Fitness evaluation of genomes.
  - Checkpointing and data logging.
- **Execution**: Run this script to train the agent using NEAT.

### 2. `test_neat.py`

- **Purpose**: Utilizes a trained NEAT model to control the agent in the environment and evaluate its performance.
- **Key Components**:
  - Loading the trained model from a file.
  - Running the agent in the environment.
  - Collecting and printing episode rewards.
- **Execution**: Run this script to evaluate the trained agent.

### 3. `plots.py`

- **Purpose**: Contains functions to generate plots for visualizing training progress.

## 🚀 Usage

1. **Training**:
   - Run `train_neat.py` to train the agent using NEAT. (Best models will be saved in a folder called `models`)
   - Adjust configuration parameters and episode reward function as needed.

2. **Testing**:
   - After training, run `test_neat.py` with the desired model to evaluate the trained agent's performance.
   - Performance of this model will be saved in a video in a folder called `videos`

## 🔧 Dependencies

- Python 3.x
- OpenAI Gym
- NEAT Python
- Pandas
- Matplotlib
- NumPy

## 🙏 Acknowledgments

- This project builds upon the NEAT Python library and OpenAI Gym environment.
- Credit to the developers and contributors of NEAT and OpenAI Gym.

