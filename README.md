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
 python setup.py install
 ```
 If you wish to contribute to the project, we suggest running `python setup.py develop` instead, which will allow you to have any changes to source files reflected in the installation without having to reinstall.

## General Architecture of the Project
- **Environment**: This class controls the stepping of the environment, access to the historical observations/trajectory (statec, actions, rewards), without restriction on the number of agents. Also supports any additional features specific to the environment, such as rendering and more (example: CartPole, Pacman).

- **Agent**: This class controls the interaction between the observations and the actual policy function that is used. It is where the various Reinforcement Learning algorithms are implemented (example: Q-Learning, PPO).

- **Model**: This class is where the models of the various functions needed in the different Agents are implemented. These could be neural networks, linear models, or tabular methods of modeling the functions (example: action-value model, policy model).

- **Trainer**: This is the script that calls the functions exposed by the agent(s) and environment(s), controlling the entire flow of the program. It will be in charge of distributing training, pipelining the function calls between the environment and agent, and looping the agent's training until convergence.
