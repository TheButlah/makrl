# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass

from collections import deque


class Environment(with_metaclass(ABCMeta, object)):
  """An abstract class that represents the environment that agents will interact
  with."""

  @abstractmethod
  def __init__(self, max_trajectory_length=None, seed=None):
    """Constructs an Environment object.

    Args:
      max_trajectory_length:  An int for the maximum length of the trajectory to
        keep. If `None`, the trajectory will have an unbounded length.
      seed:  An integer used to seed the environment. If `None`, the seed will
        be random.
    """
    self._seed = seed
    self._env = None
    """This is the underlying game environment that can be played."""

    self._action_dict = None
    """This is a dict of strings that maps to the internal action type"""

    self._trajectory = deque(maxlen=max_trajectory_length)
    """This is a queue of (s_t, a_t, r_t) transitions."""
    pass

  @abstractmethod
  def step(self, action):
    """Steps the environment based on a provided action. Also updates the stored
    trajectory.

    Args:
      action: The action to take when stepping the environment. The action
        should be one of the values of the action dictionary.

    Returns: A scalar reward, or None if the episode terminates.
    """
    pass

  @abstractmethod
  def get_state(self):
    """Gets the current state of the environment."""
    pass

  @abstractmethod
  def reset(self, seed=None):
    """Resets the environment to an initial state.

    Args:
      seed:  An integer used to seed the environment. If `None`, the seed will
        be random.
    """
    pass

  def get_action_dict(self):
    """This is a dict of strings that map to the internal action type."""
    return self._action_dict

  def get_trajectory(self):
    """Returns the latest trajectory of the environment, up to
    `max_trajectory_length`. The trajectory is a collection of (S,A,R)
    tuples in a deque (queue as linked list). The last element of this deque
    is the transition the environment experienced from the latest call to
    `step()`. Transitions older than `max_trajectory_length` will be
    discarded."""
    return self._trajectory
