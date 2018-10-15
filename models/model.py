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
    self._s_shape = (None,) + tuple(state_shape)
    self.step_major = step_major
    pass

  @abstractmethod
  def save(self, save_path):
    """Saves the model to a given location.

    Args:
      save_path:  A string giving the path to save the model to.
    """
    pass

  @property
  def model_state(self):
    """Gets the current internal state of the model. For example, in RNN based
    models, this returns the hidden state vector of the model, but in linear
    models will return `None`. Can be used later to reset the state of the model
    to allow for non-sequential calls, such as when an episode terminates or for
    multiple evaluations per timestep, such as in Expected SARSA.

    NOTE: this is NOT the environment state.

    Returns:
      The current internal state.
    """
    return None

  @model_state.setter
  def model_state(self, state):
    """Sets the current internal state of the model. Should be a valid previous
    state of the model. If `state` is `None`, will set the internal state of the
    model to its initial default state.

    NOTE: this is NOT the environment state.

    Args:
      state:  The desired internal state of the model. If `None`, resets the
        model to its initial default state.
    """
    pass

