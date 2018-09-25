# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass

from . import Model


class PolicyModel(with_metaclass(ABCMeta, Model)):
  """An abstract class that represents a policy model that can be used by an
  agent."""

  @abstractmethod
  def __init__(self, state_shape, action_shape, *, load_model=None):
    """Constructs the model and initializes it.

    Args:
      state_shape:  The shape tuple of a single state tensor.
      action_shape:  The shape tuple of a single action tensor.
      load_model:  A string giving a path to the model to load.
    """
    super(PolicyModel, self).__init__(load_model=load_model)
    self._s_shape = (None,) + tuple(state_shape)
    self._a_shape = (None,) + tuple(action_shape)
    pass

  @abstractmethod
  def save(self, save_path):
    super(PolicyModel, self).save(save_path)
    pass

  @abstractmethod
  def predict_pi(self, state):
    """Outputs the policy for a given batch of states.
    Args:
      state:  A batched representation of the state of the environment as a
        numpy array.

    Returns:  A batch of action-probabilities as a numpy array.
    """
    pass

  @abstractmethod
  def update_pi(self, advantage, state):
    """Informs the model of the advantage values for a given batch of states.
    Note that `advantage` is capable of being an n-step advantage.

    Args:
      advantage:  A batch of the advantage values as a numpy array. The model
        will use this to improve.
      state:  A batched representation of the state of the environment for
        which the value function will be updated, as a numpy array.
    """
    pass
