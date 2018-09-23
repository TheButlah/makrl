# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division


class Agent(object):
    """An Abstract class that represents an agent that interacts with an
    environment."""

    def __init__(self):
        """Constructs an agent and initializes it."""
        raise NotImplementedError

    def train_step(self):
        """Trains the agent for a single step of the simulation."""
        raise NotImplementedError

    def pick_action(self, trajectory):
        raise NotImplementedError

    # TODO: Finish agent API
