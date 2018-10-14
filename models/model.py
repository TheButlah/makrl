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
  def __init__(self, load_model=None):
    """Constructs the model and initializes it.

    Args:
      load_model:  A string giving a path to the model to load.
    """
    pass

  @abstractmethod
  def save(self, save_path):
    """Saves the model to a given location.

    Args:
      save_path:  A string giving the path to save the model to.
    """
    pass

  @property
  @abstractmethod
  def model_state(self):
    """Gets the current internal state of the model. For example, in RNN based
    models, this returns the hidden state vector of the model, but in linear
    models will return `None`. Can be used later to reset the state of the model
    to allow for non-sequential calls, such as when an episode terminates or for
    multiple evaluations per timestep, such as in Expected SARSA.
    """
    pass

  @model_state.setter
  def model_state(self, state):
    """Sets the current internal state of the model. Should be a valid prior
    state of the model. If `state` is `None`, will set the internal state of the
    model to its original default/initial state.
    """
    pass

