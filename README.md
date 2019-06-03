# RL-Halite
A reinforcement learning agent framework for the game halite.io.

## Installation
Install Python. Both Python2 and Python3 are supported, although Python3 is preferred.

Install setuptools:
```
pip install setuptools
```
Install the latest version of [TensorFlow](https://tensorflow.org). Follow the instructions at their website. You will likely also want to use cuDNN and CUDA if you are using GPU acceleration.

Then, simply clone this repo and run the setup script:
 ```
 git clone https://github.com/CornellDataScience/RL-Halite
 cd RL-Halite
 pip install .
 ```
 If you wish to contribute to the project, we suggest running `pip install -e .` instead, which will allow you to have any changes to source files reflected in the installation without having to reinstall.

## General Architecture of the Project
- **Experiment**: This is the script that is the entry point of the entire program. You can make this whatever you want, but typically it calls the functions exposed by the agent(s) and environment(s), controlling the entire flow of the training or inference process. It will be in charge of distributing training, passing the data between the environment and agent, and looping the agent's training until convergence.

- **Environment**: The environment is represented by the `BatchedEnv` class, which manages multiple environments, runs them in parallel on each cpu core, and then batches their experiences together in a synchronous fashion. Internally, it can use any gym-like environment. By wrapping individual gym envionments in this way, we make it easy to experiment with synchronous reinforcement learning methods such as A2C and Synchronous Q-Learning.

- **Agent**: This class controls the interaction between the observations and the actual policy function that is used. It is where the various Reinforcement Learning algorithms are implemented (example: Q-Learning, PPO).

- **Model**: This class is where the models of the various functions approximators needed in the different Agents are implemented. These could be neural networks, linear models, or tabular methods of modeling the functions (example: action-value model, policy model). By separating the Model  from the Agent, it is easy to exchange different agents with different models and isolate bugs in the codebase. It also allows developers unfamiliar with reinforcement learning to work with an api that resembles supervised learning, allowing them to craft an appropriate model for a given environment and simplifying the research process.
