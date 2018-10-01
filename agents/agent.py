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
  def __init__(self, model):
    """Constructs an agent and initializes it.

    Args:
      model:  The model that the agent will use to model its required functions.
    """
    pass

  @abstractmethod
  def train_step(self, observations):
    """Trains the agent for a single step of the simulation.

    Args:
      observations:  A batch of lists of observations. Each observation list is
        a list of past (state, action, reward) tuples, where the tail of the
        list is the latest tuple.
    """
    pass

  @abstractmethod
  def pick_action(self, observations):
    """Picks an action using the target policy. This is the agent's best current
    approximation to the optimal policy. This may be the same policy as the
    exploratory/behavior policy in the case of on-policy methods.

    Args:
      observations:  A batch of lists of observations. Each observation list is
        a list of past (state, action, reward) tuples, where the tail of the
        list is the latest tuple.

    Returns:
      A ndarray of probabilities over the action space.
    """
    pass

  @abstractmethod
  def pick_exploratory_action(self, observations):
    """Picks an action using the exploratory/behavior policy. This may be the
    same policy as the target policy in the case of on-policy methods.

    Args:
      observations:  A batch of lists of observations. Each observation list is
        a list of past (state, action, reward) tuples, where the tail of the
        list is the latest tuple.

    Returns:
      A ndarray of probabilities over the action space.
    """
    pass
