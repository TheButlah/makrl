# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

import argparse
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved', '')


def main():
  parser = argparse.ArgumentParser(
    description="Trains a q-learning agent on the Cart-Pole-V1 task.")
  parser.add_argument('-n', '--name',
    type=str,
    default='%s' % time.strftime('cart-pole-q-learner_%Y%m%d_%H%M%S'),
    help="The name of the experiment being run. Used to identify different "
         "experiments of the same type. Will by default be named based on the "
          "current time.")
  parser.add_argument('-s', '--save-to', metavar='PATH',
    type=str,
    help="A path to the directory to save the results of the experiment. "
         "Otherwise, the results will be saved in a new directory named the "
         "same as the --name argument in the current working directory.")
  parser.add_argument("-l", "--load-from", metavar='PATH',
    type=str, nargs='?',
    help="This flag will cause the model to be loaded from a prior experiment "
         "rather than starting fresh. If the "
         "this experiment. This is useful when loading a (partially) trained "
         "agent.")
  args = parser.parse_args()
  print(args)


if __name__ == "__main__":
  main()
