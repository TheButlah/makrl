# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

import sys
sys.path.append("../")

from agents import Agent

from collections import defaultdict


class A2CAgent(Agent):

  def __init__(self, model):
    super(QLearning, self).__init__()
    self._model = model
    self._state_counts = defaultdict(lambda: 0)
    self.eval_step = self._model.eval_step

  def train_step(self, observations,lrnow,cliprangenow, *_args, **_kwargs):
    self._model.learn(*observations,lrnow,cliprangenow)

  def pick_action(self, observations):
    self._model.eval_step(observations)

  def pick_exploratory_action(self, observations):
    pick_action(observations)

  # TODO: Finish A2C Agent implementation