# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass


class Agent(with_metaclass(ABCMeta, object)):
    """An Abstract class that represents an agent that interacts with an
    environment."""

    @abstractmethod
    def __init__(self):
        """Constructs an agent and initializes it."""
        pass

    @abstractmethod
    def train_step(self):
        """Trains the agent for a single step of the simulation."""
        pass

    @abstractmethod
    def pick_action(self, trajectory):
        pass

    # TODO: Finish agent API
