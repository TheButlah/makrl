# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass

from . import Model


class ActionModel(with_metaclass(ABCMeta, Model)):
  """An abstract class that represents an Action-Value model that can be used by
  an agent."""

  @abstractmethod
  def __init__(self, state_shape, action_shape, step_major=False,
    load_model=None):
    """Constructs the model and initializes it.

    Args:
      state_shape:  A tuple representing the shape of the environment's state
        space.
      action_shape:  A tuple representing the shape of the environment's action
        space.
      step_major:  Whether or not the first dimension of the any arguments with
        a step dimension will be the first axis or the second. If `True`, the
        shapes described for any such arguments should have their first two
        dimensions transposed. There may be performance benefits to having the
        step dimension first instead of the batch dimension, but because batched
        computation typically has the batch dimension first, this parameter is
        `False` by default.
      load_model:  A string giving a path to the model to load.
    """
    super(ActionModel, self).__init__(
      state_shape, step_major=step_major, load_model=load_model)
    self._a_shape = (None,) + tuple(action_shape)

  @abstractmethod
  def predict_q(self, states, actions=None):
    """Predicts the action-value q for a batch of state-action pairs. If
    `actions` is not provided, the optimal action for each element in the batch
    will be selected from all possible actions and returned in addition to the
    action-value.

    Args:
      states:  A batched representation of the states of the environments.
        Shaped `(batch_size,) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.
      actions:  A batched representation of the actions to take. If `None`, the
        action will be automatically selected greedily, such each action will
        give the maximum value for that state. In that case, the action selected
        will be returned along with the value. Shaped
        `(batch_size,) + action_shape`, where `action_shape` is a tuple
        representing the shape of the environment's action space.

    Returns:
      A batch of action-values as a numpy array. If `actions` was `None`, the
      result will instead be a tuple of the action-values and the selected
      actions.
    """
    pass

  @abstractmethod
  def update_q(self, states, actions, rewards, mu=None, num_steps=None):
    """Updates the model based on experience gained from a given batch of
    observed state-action-reward transitions, each represented in separate numpy
    arrays. This function looks at the TD error for each step in the batch of
    observations and uses that error to update the model. The update will
    base its estimate of the return using the last value of the sequence, which
    will be a return value instead of a reward value. This final return will be
    based on some value determined by the `Agent`. It will be the best estimate
    for the return experienced at the latest action in `observations`, taking
    into account the actual experienced reward when the latest action was taken,
    and hence will be more accurate than the current predicted return at that
    action. By using this return, this function can compute the TD errors and
    update the model.

    Args:
      states:  A batch of observed states from transitions of the environment.
        Shaped `(batch_size, max_steps) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.
      actions:  A batch of observed states from transitions of the environment.
        Shaped `(batch_size, max_steps) + action_shape`, where `action_shape` is
        a tuple representing the shape of the environment's action space.
      rewards:  A batch of observed rewards from transitions of the environment.
        Note that the last element of this ndarray will actually be a return and
        not a reward. Shaped `(batch_size, max_steps)`.
      mu:  Weighting factors of the states. For example, these may be the ratios
        computed from importance sampling, or the percentage of time spent in a
        particular state. Shaped `(batch_size, max_steps)`.
      num_steps:  A list of integers representing the number of steps for each
        observation in the batch as some observations' episodes may terminate,
        resulting in a non-uniform number of steps for each observation. No
        element of `num_steps` may be greater than `max_steps`. If `None`, it is
        assumed that all observations in the batch are `max_steps` long. Shaped
        `(batch_size,)`.

    Returns:
      The net loss/TD error for all steps averaged over the batch.
    """
    pass

