# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division


class Model(object):
    """An abstract class that represents a model that can be used by an
    agent."""

    def __init__(self, load_model=None):
        """Constructs the model and initializes it.
        Args:
            load_model:  A string giving a path to the model to load.
        """
        raise NotImplementedError

    def save(self):
        """Saves the model to a given location."""
        raise NotImplementedError
