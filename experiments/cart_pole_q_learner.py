# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip
from six import viewitems

import argparse
import os
import time
import logzero
import logging

from logzero import logger

import gym
import numpy as np

import utils
from utils import BatchedEnv
# from models import FCQNet


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():

  # Parse args and set up logger #
  ################################
  parser = argparse.ArgumentParser(
    description="Trains a q-learning agent on the Cart-Pole-V1 task.")

  def pos_int(value):
    ival = int(value)
    if ival <= 0:
      raise argparse.ArgumentTypeError("%s is not a positive integer" % value)
    return ival

  parser.add_argument("-n", "--name",
    type=str,
    default="%s" % time.strftime("cart-pole-q-learner_%Y%m%d_%H%M%S"),
    help="The name of the experiment being run. Used to identify different "
         "experiments of the same type. Will by default be named based on the "
         "current time.")
  parser.add_argument("-s", "--save-to", metavar="PATH",
    type=str,
    help="A path to the directory to save the results of the experiment. "
         "If not specified, the results will be saved in a new directory named "
         "the same as the --name argument in the directory where this script "
         "is located.")
  parser.add_argument("-l", "--load-from", metavar="PATH",
    type=str,
    help="This flag will cause the model to be loaded from a prior experiment "
         "located at the given path, rather than starting fresh. This is "
         "useful when loading a (partially) trained agent.")
  parser.add_argument("-v", "--verbosity", metavar="LEVEL",
    type=str.upper,
    nargs='?',
    default="INFO", const="DEBUG",
    choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
    help="Sets the console logger verbosity. Note that all logger messages "
         "will still be written to the logfile. Specifying this flag without "
         "a level will set the verbosity to DEBUG. (default: %(default)s) "
         "(choices: %(choices)s)")
  parser.add_argument("--num-steps", metavar="N",
    type=pos_int,
    default=100,
    help="Sets the number of steps to run for. A step is completed when every "
         "environment in the batch has finished a step of its environment. "
         "(default: %(default)d)")
  parser.add_argument("--batch-size", metavar="N",
    type=pos_int,
    default=utils.num_cores(),
    help="Sets the number of environments, which is the batch size used for "
         "training. If an environment's episode terminates, it will be reset, "
         "ensuring there will always be a full set of running environments in "
         "the batch. By default, the batch size is the number of available CPU "
         "cores. (default: %(default)d)")
  parser.add_argument("--max-episode-len", metavar="N",
    type=pos_int,
    default=128,
    help="Sets the maximum number of steps that an environment can take before "
         "we terminate the episode. (default: %(default)d)")
  parser.add_argument("--seed",
    type=pos_int,
    help="Sets the random seed for this experiment.")
  args = parser.parse_args()

  args.save_to = os.path.join(BASE_DIR, args.name)
  if not os.path.exists(args.save_to):
    os.mkdir(args.save_to)

  # Set up the logfile
  logzero.logfile(
    filename=os.path.join(args.save_to, 'log.txt'),
    mode='w',
    loglevel=logging.DEBUG)  # Logfile always uses most verbose option.
  # Set the console's stderr to use the defined verbosity
  logger.handlers[0].setLevel(args.verbosity)
  logger.debug("Logger created and args parsed!")

  # Print the command line arguments the program ran with
  logger.info("Script location: %s" % __file__)
  logger.info("Parsed args: %s" % vars(args))

  # Initialize framework #
  ########################
  logger.info("Initializing framework...")
  if args.seed:
    np.random.seed(args.seed)

  q_model = None  # FCQNet(...)
  q_learner = None  # QLearner(...)

  logger.debug("Constructing environment batch")
  envs = BatchedEnv([lambda: gym.make('CartPole-v1')]*args.batch_size)

  # Run the agent-environment interface #
  #######################################
  logger.info("Running agent-environment interface...")
  logger.debug("Resetting envs for first time")
  start_states = states = envs.reset()
  for global_step in range(args.num_steps):
    print(envs.step_counters)
    actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
    obs, rewards, dones, infos = envs.step(actions)
    print(obs, rewards)

  envs.close()


if __name__ == "__main__":
  main()
