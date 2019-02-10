# Makrl
Makrl (Modularized Algorithm Kit for Reinforcement Learning) is a reinforcement learning library that makes implementing state of the art RL algorithms easier for both experienced researchers, newcomers to the field, and software engineers. 

Most implementations of RL algorithms are complicated and hard to understand. There is no clear division between how the neural networks are trained, how the agent-environment interface is run, and how environments are batched together and parallelized on the CPU. Instead of writing code like well engineered software, everything is lumped together resulting in indecipherable code.

This has a real impact on researchers - if your agent isn't performing well on some task, there are so many moving parts that must be investigated - is it a bug in the code? Is it a bad choice of RL algorithm? Is the neural network architecture a bad choice for the task at hand? Without a modular code design, not only is it extremely difficult to make sense of these different moving parts, but its difficult to diagnose what is the issue in the approach and easy to introduce buggy code.

Using a modularized design enables the following:
- Synchronous, parallel environment batching across CPU cores for efficient training, enabling hyperparameters like batch size to be independent of the number of CPU cores.
- Policy and value functions that frame the training process as a regression problem, making it easy for people with supervised learning experience to apply that experience to reinforcement learning
- Separation of the RL algorithms from the neural network architectures.
- Mixing and matching different neural network architectures with different RL algorithms, making it easy to experiment and diagnose problems
- Separation of concerns between between more research focused parts of the codebase and more sotware engineering parts of the codebase.

## Installation
Install Python. Both Python2 and Python3 are supported, although Python3 is preferred.

Install setuptools:
```
pip install setuptools
```
Install the latest version of [TensorFlow](https://tensorflow.org). Follow the instructions at their website. You will likely also want to use cuDNN and CUDA if you are using GPU acceleration.


Install the [Layers](https://github.com/TheButlah/Layers) library.

Then, simply clone this repo and run the setup script:
 ```
 git clone https://github.com/TheButlah/makrl
 cd makrl
 python setup.py install
 ```
 If you wish to contribute to the project, we suggest running `python setup.py develop` instead, which will allow you to have any changes to source files reflected in the installation without having to reinstall.

## General Architecture of the Project
- **Environment**: This class controls the stepping of the environment, access to the historical observations/trajectory (states, actions, rewards), without restriction on the number of agents. Also supports any additional features specific to the environment, such as rendering and more (example: CartPole, Pacman).

- **Agent**: This class controls the interaction between the observations and the actual policy function that is used. It is where the various Reinforcement Learning algorithms are implemented (example: Q-Learning, PPO).

- **Model**: This class is where the models of the various functions needed in the different Agents are implemented. These could be neural networks, linear models, or tabular methods of modeling the functions (example: action-value model, policy model).

- **Trainer**: This is the script that calls the functions exposed by the agent(s) and environment(s), controlling the entire flow of the program. It will be in charge of distributing training, pipelining the function calls between the environment and agent, and looping the agent's training until convergence.
