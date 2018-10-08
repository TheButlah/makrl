# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from . import Agent

from collections import defaultdict

# TODO:  Not sure why this is separate from Agent atm. TBD.
class PPO(Agent):

  def __init__(self, model):
    """Constructs an agent and initializes it.

    Args:
      model:  The model that the agent will use to model its required functions.
    """

    super(PPO, self).__init__()
    self._model = model
    self._state_counts = defaultdict(lambda: 0)


  def train_step(self, observations):
    """Trains the agent for a single step of the simulation.

    Args:
      observations:  A batch of lists of observations. Each observation list is
        a list of past (state, action, reward) tuples, where the tail of the
        list is the latest tuple.
    """


    raise NotImplementedError

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
    raise NotImplementedError

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
