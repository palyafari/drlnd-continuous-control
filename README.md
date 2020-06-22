# About this project

This projects contains my solution of the second project in the **[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)** of [Udacity](https://www.udacity.com/).
The goal of this project is to train an agent to move its double-jointed arm to the target location and keep it there for as many time steps as possible.




# The Environment

This project uses the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) Unity ML-Agents environment.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

![Trained Agent](./img/reacher.gif)

Udacity provides 2 versions of the environments to allow to experiment with malgorithms of different types:

- The first version contains a single agent. The environment is considered solved, when the agent gets an average score of +30 over 100 consecutive episodes.
- The second version contains 20 identical agents, each with its own copy of the environment. The barrier to solve this environment takes into account the presence of multiple agents as follows:
    - After each episode, the rewards that each agent received are added up (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. Then the average of these 20 scores are taken. 

*As for now, this project solves the first version only.*

To run the code in this project, the specified environment of Udacity is needed. To set it up, follow the instructions below.

## Step 1 - Getting started
Install PyTorch, the ML-Agents toolkit, and a few more Python packages according to the [instructions in the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).

Furthermore, install the [dataclasses](https://docs.python.org/3/library/dataclasses.html) python module.

## Step 2 - Download the Unity Environment
For this project, you **don't** need to install Unity. Instead, choose the pre-built environmen provided by Udacity matching your operating system:

### First version: 1 Agent
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

### Second version: 20 Agents
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

# Instructions

To explore the environment and train the agent, start a jupyter notebook, open Continous_Control.ipynb and execute the steps. For more information, and an exmaple on how to use the agent, please check instructions inside the notebook.

## Project structure

* `Continouos_Control.ipynb`: the jupyter notebook for executing the training
* `src\agent.py` : the implementation of the Agent
* `src\model.py` : the PyTorch models of the neural networks used by the Agent
* `src\replay_buffer.py` : The replay buffer implementation for memory
* `src\config.py` : the default configuration/hyperparameters of the models
* `src\noise.py`  : the implementation of the Ornstein-Uhlenbeck process


# Results

The trained agent solved the environment in 86 episodes.
For a detailed explanation, please read the [project report](./Report.md)


# Notes
The project uses the code and task description provided in the **[Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)**  class as a basis.
