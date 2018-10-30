# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass


class Model(with_metaclass(ABCMeta, object)):
  """An abstract class that represents a model that can be used by an agent."""

  @abstractmethod
  def __init__(self, state_shape, step_major=False, load_model=None):
    """Constructs the model and initializes it.

    Args:
      state_shape:  A tuple representing the shape of the environment's state
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
    self._s_shape = tuple(state_shape)
    self._step_major = step_major

  @abstractmethod
  def save(self, save_path):
    """Saves the model to a given location.

    Args:
      save_path:  A string giving the path to save the model to.
    """
    pass

  @property
  def model_state(self):
    """Gets the current batched internal state of the model. Note that this is
    an entirely different concept from the environment state.

    `model_state` can be used later to reset the state of the model to allow for
    non-sequential calls to the model methods, such as to indicate when a new
    episode should begin or when performing multiple method calls per timestep,
    such as in Expected SARSA. It is the responsibility of the `Agent` to
    properly save and restore this internal state is it is fundamental to any
    sequence based model, regardless of whether the particular `Model` in
    question is actually non-markovian (stateful).


    In internally stateless/markovian models, that is, models conditioned only
    on the present environment state and no prior states, model_state will
    return `None`.

    For internally stateful/non-markovian models, that is, models conditioned on
    both present environment states/actions and some internal state,
    `model_state` will return a batch of internal states - one for each episode
    in the latest method invoked on this model.

    Here are a few examples of what `model_state` might be:
      Stacked LSTM:  Batch of a list of LSTM state tuples for each LSTM layer
        for each episode.

      Linear model on latest 4 environment states:  Batch of last 3 environment
        states seen for each episode. These will be concatenated with the target
        state and fed to a linear model.

      Time-series CNN:  Batch of last n-1 seen environment states, where n is
        the max number of past timesteps to examine.

      CNN only on current environment state:  `None`

    Returns:
      The current batched internal state for non-markovian models, or `None` for
      markovian models.
    """
    return None

  @model_state.setter
  def model_state(self, state):
    """Sets the current batched internal state of the model. Note that this is
    an entirely different concept from the environment state.

    The `Agent` should ensure that the new `model_state` will match along the
    batch dimension with the next method invocation made on this `Model`.

    Args:
      state:  The desired batched internal state of the model. If `None`, resets
        all states. Otherwise, setting any particular state in the batch to
        `None` will reset that state.
    """
    pass

