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
