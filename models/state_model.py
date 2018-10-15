# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass

from . import Model


class StateModel(with_metaclass(ABCMeta, Model)):
  """An abstract class that represents an State-Value model that can be used by
  an agent."""

  @abstractmethod
  def predict_v(self, states):
    """Predicts the state-value q for a batch of states.

    Args:
      states:  A batched representation of the states of the environments.
        Shaped `(batch_size,) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.

    Returns:
      A batch of state-values as a numpy array.
    """
    pass

  @abstractmethod
  def update_v(self, states, actions, rewards, mu=None, num_steps=None):
    """Updates the model based on experience gained from a given batch of
    observed state-action-reward transitions, each represented in separate numpy
    arrays. This function looks at the TD error for each step in the batch of
    observations and uses that error to update the model. The update will
    base its estimate of the return using the last value of the sequence, which
    will be a return value instead of a reward value. This final return will be
    based on some value determined by the `Agent`. It will be the best estimate
    for the return experienced after the latest state in `observations`, taking
    into account the actual experienced reward at least one step after the state
    was visited, and hence will be more accurate than the current predicted
    return at that state. By using this return, this function can compute the TD
    errors and update the model.

    Args:
      states:  A batch of observed states from transitions of the environment.
        Shaped `(batch_size, max_steps) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.
      actions:  A batch of observed states from transitions of the environment.
        Shaped `(batch_size, max_steps-1) + action_shape`, where `action_shape`
        is a tuple representing the shape of the environment's action space.
        `Actions` will have one less step than what is provided for `states`,
        because a state-value function should only depend on the state and not
        also the value. This ensures that only actions before the last state
        are provided. Note that the reason why `actions` is provided at all is
        because predicting the value of a state conditioned on past state-action
        pairs is valid, as long as the action following the state that we are
        predicting is not used.
      rewards:  A batch of observed rewards from transitions of the environment.
        Note that the last element of this ndarray will actually be a return and
        not a reward. Shaped `(batch_size, max_steps)`.
      mu:  Weighting factors of the states. For example, these may be the ratios
        computed from importance sampling, or the percentage of time spent in a
        particular state. Shaped `(batch_size, max_steps)`. If `None`, no
        weighting is applied.
      num_steps:  Either a scalar or a list of integers shaped `(batch_size,)`
        representing the number of steps for each observation in the batch. This
        argument is needed as some observations' episodes may terminate,
        resulting in a non-uniform number of steps for each observation. No
        element of `num_steps` may be greater than `max_steps`. If `None`, it is
        assumed that all observations in the batch are `max_steps` long.

    Returns:
      (loss, returns) where `loss` is the net loss/TD error for all steps
      averaged over the batch, and `returns` is the list of returns constructed
      from `rewards`, so that the `Agent` need not recompute those returns.
    """
    """Developer's note: because `returns` can be computed based entirely on
    `rewards`, it would have been easy to have the `Agent` pass in the computed
    `rewards` ndarray. However, the developers decided against this to allow the
    possibility of models conditioned on past rewards. Such models would have to
    convert back from `returns` to `rewards`, thereby wasting computation.
    Hence, we simply delegate all responsibility for computing `returns` to the
    `Model`, and have it return what it computed."""
    pass
