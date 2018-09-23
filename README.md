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