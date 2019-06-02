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
  def update_pi(self, states, actions, advantages, num_steps=None):
    """Updates the model based on experience gained from a given batch of
    observed environment transitions. `advantages` is provided by an `Agent`,
    usually via another model acting as the critic/advantage function, trained
    separately. Note that using advantages instead of returns is explicitly more
    general, as it reduces the variance of the model and the advantage is equal
    to the return when the baseline is 0. This function then uses the provided
    data to maximize the policy gradient at each step in the observation
    sequence.

    Args:
      states:  A batch of observed states from transitions of the environment.
        Shaped `(batch_size, max_steps) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.
      actions:  A batch of observed actions from transitions of the environment.
        Shaped `(batch_size, max_steps, len(action_shape)`, where `action_shape`
        is a tuple representing the shape of the environment's action space. The
        last axis of `actions` should contain tuples that identifies the actions
        taken via their index in the action space.
      advantages:  A batch of advantages at each step in the observation
        sequence. Shaped `(batch_size, max_steps)`.
      num_steps:  Either a scalar or a list of integers shaped `(batch_size,)`
        representing the number of steps for each observation in the batch. This
        argument is needed as some observations' episodes may terminate,
        resulting in a non-uniform number of steps for each observation. No
        element of `num_steps` may be greater than `max_steps` or less than 1.
        If `None`, it is assumed that all observations in the batch are
        `max_steps` long.
    """
    """Developer's note: because `returns` can be computed based entirely on
    `rewards`, it would have been easy to have the `Agent` pass in the computed
    `returns` ndarray. However, this has not been done to more easily allow the
    possibility of models conditioned on past rewards. Hence, we simply delegate
    all responsibility for computing `returns` to the `Model`."""
    # TODO: only calculate loss on all but last bootstrapping steps
    # TODO: document the return of this function
    pass
