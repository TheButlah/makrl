# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from . import Agent


class QLearning(Agent):
    def __init__(self):
        super(QLearning, self).__init__()
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def pick_action(self, trajectory):
        raise NotImplementedError

    # TODO: Finish QLearning implementation
