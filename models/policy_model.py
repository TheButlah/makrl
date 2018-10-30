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
  def __init__(self, state_shape, action_shape, step_major=False,
    load_model=None):
    """Constructs the model and initializes it.

    Args:
      state_shape:  A tuple representing the shape of the environment's state
        space.
      action_shape:  A tuple representing the shape of the environment's action
        space.
      step_major:  Whether or not the first dimension of any arguments with a
        step dimension will be the first axis or the second. If `True`, the
        shapes described for any such arguments should have their first two
        dimensions transposed. There may be performance benefits to having the
        step dimension first instead of the batch dimension, but because batched
        computation typically has the batch dimension first, this parameter is
        `False` by default.
      load_model:  A string giving a path to the model to load.
    """
    super(PolicyModel, self).__init__(
      state_shape, step_major=step_major, load_model=load_model)
    self._a_shape = tuple(action_shape)
    pass

  @abstractmethod
  def save(self, save_path):
    super(PolicyModel, self).save(save_path)
    pass

  @abstractmethod
  def predict_pi(self, states):
    """Predicts the desired action probabilities according to this policy for a
    given batch of states.

    Args:
      states:  A batched representation of the states of the environments.
        Shaped `(batch_size,) + state_shape`, where `state_shape` is a tuple
        representing the shape of the environment's state space.

    Returns:  A batch of action-probabilities as a numpy array, shaped
      `(batch_size,) + action_shape`, where `action_shape` is
        a tuple representing the shape of the environment's action space.
    """
    pass

  @abstractmethod
  def update_pi(self, advantages, states):
    """Informs the model of the advantage values for a given batch of states.

    Advantages are used instead of returns or value-estimates, because as long
    as the baseline in an advantage function is not conditioned on taking a
    particular action for the timestep being evaluated, the policy gradient is
    unbiased. Therefore, advantages are more general than returns/values.

    Args:
      advantages:  A batch of the advantage values as a numpy array, shaped
        `(batch_size,)`
      states:  A batched representation of the states of the environments.
        Shaped `(batch_size,) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.
    """
    pass
