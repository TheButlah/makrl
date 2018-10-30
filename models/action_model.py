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
      step_major:  Whether or not the first dimension of any arguments with a
        step dimension will be the first axis or the second. If `True`, the
        shapes described for any such arguments should have their first two
        dimensions transposed. There may be performance benefits to having the
        step dimension first instead of the batch dimension, but because batched
        computation typically has the batch dimension first, this parameter is
        `False` by default.
      load_model:  A string giving a path to the model to load.
    """
    super(ActionModel, self).__init__(
      state_shape, step_major=step_major, load_model=load_model)
    self._a_shape = tuple(action_shape)

  @abstractmethod
  def predict_q(self, states, actions=None):
    """Predicts the action-value q for a batch of state-action pairs. If
    `actions` is not an ndarray, a variety of computations on the action space
    can be performed. For example, if actions is `None` or "argmax", this
    greedily selects an action, which is used in many algorithms such as Q
    Learning.

    Args:
      states:  A batched representation of the states of the environments.
        Shaped `(batch_size,) + state_shape`, where `state_shape` is a tuple
        representing the shape of the environment's state space.
      actions:  Can be `None`, "argmax", "all", or a batch of actions to
        evaluate the function on.

        If `None` or "argmax", the actions will be selected greedily, where each
        action will be selected so that the maximum value at that state-action
        pair will be achieved.

        If "all", all actions will be evaluated for each state, i.e. no argmax
        or random subset will be used.

        If a ndarray, it will be treated as a batch of actions to match with the
        batch of states, such that it will produce a batch of state-action pairs
        shaped `(batch_size, len(action_shape))`, where `action_shape` is a
        tuple representing the shape of the environment's action space. In other
        words `actions` will be a matrix with `batch_size` rows, where each row
        is a tuple used as the index of the action in the action space.

    Returns:
      If `actions` was `None` or "argmax", the output will be a
      `(values, actions)` tuple, where `values` is a ndarray shaped
      `(batch_size,)` and `actions` is a list of the index of the selected
      actions in the action space, shaped `(batch_size, len(action_shape))`.

      If `actions` was "all", returns a ndarray shaped
      `(batch_size,) + action_shape` for the action-values at every possible
      action in the action space.

      If `actions` was a ndarray, returns a ndarray shaped `(batch_size,)` for
      the action-values associated with the provided list of actions.
    """
    pass

  @abstractmethod
  def update_q(self, states, actions, rewards, mu=None, num_steps=None):
    """Updates the model based on experience gained from a given batch of
    observed state-action-reward transitions, each represented in separate numpy
    arrays. This function looks at the TD error for each step in the batch of
    observations and uses that error to update the model.

    The update will base its estimate of the return using the last value in
    `rewards`, which will be a return value instead of a reward value. This
    final return will be based on some value determined by the `Agent`. It will
    be the best estimate for the return experienced at the latest action, taking
    into account the actual experienced reward at that action and possibly
    future actions, and hence will be more accurate than the current predicted
    return at that action. By using this return, this function can compute the
    TD errors and update the model.

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
      mu:  Weighting factors of the states. This will determine how much weight
        each state has when contributing to the TD error. For example, these may
        be the ratios computed from importance sampling, or the percentage of
        time spent in a particular state. Shaped `(batch_size, max_steps)`. If
        `None`, no weighting is applied.
      num_steps:  Either a scalar or a list of integers shaped `(batch_size,)`
        representing the number of steps for each observation in the batch. This
        argument is needed as some observations' episodes may terminate,
        resulting in a non-uniform number of steps for each observation. No
        element of `num_steps` may be greater than `max_steps`. If `None`, it is
        assumed that all observations in the batch are `max_steps` long.

    Returns:
      (loss, returns) where `loss` is the sum of loss/TD errors at each step
      averaged over the batch, and `returns` is the list of returns constructed
      from `rewards`, so that the `Agent` need not recompute those returns.
    """
    """Developer's note: because `returns` can be computed based entirely on
    `rewards`, it would have been easy to have the `Agent` pass in the computed
    `returns` ndarray. However, this has not been done to more easily allow the
    possibility of models conditioned on past rewards. Such models would have to
    convert back from `returns` to `rewards`, thereby wasting computation.
    Hence, we simply delegate all responsibility for computing `returns` to the
    `Model`."""
    pass
