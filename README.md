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
**Agent**: This class controls the interaction between the observations and the actual policy function that is used. It takes a policy function as input along with other hyperparamters of the agent

**Environment**: This class controls the stepping of the environment,the trajectory, observations and rewards with no restriction on the number of agents along with potentially rendering.

**Model**: This class is where various Reinforcement Algorithms are implemented.This class simply learns a function to map the input space to the action space, recieving input from the Environment through the Agent.

**Trainer**: This is the manager that calls the functions exposed by Agent and Environment, controlling the entire flow of the program.
