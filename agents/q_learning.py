# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from . import Agent

from collections import defaultdict


class QLearning(Agent):

  def __init__(self, model):
    super(QLearning, self).__init__()
    self._model = model
    self._state_counts = defaultdict(lambda: 0)

  def train_step(self, observations):
    raise NotImplementedError

  def pick_action(self, observations):
    raise NotImplementedError

  def pick_exploratory_action(self, observations):
    pass

  # TODO: Finish QLearning implementation
