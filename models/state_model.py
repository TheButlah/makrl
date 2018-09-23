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
    """An abstract class that represents an State-Value model that can be used
    by an agent."""

    @abstractmethod
    def __init__(self, *, load_model=None):
        super(StateModel, self).__init__(load_model=load_model)
        pass

    @abstractmethod
    def save(self, save_path):
        super(StateModel, self).save(save_path)
        pass

    @abstractmethod
    def predict_v(self, state):
        """Predicts the state-value v for a batch of states.

        Args:
            state:  A batched representation of the state of the environment as
                a numpy array.

        Returns:
            A batch of state-values as a numpy array.
        """
        pass

    @abstractmethod
    def update_v(self, target_return, state):
        """Informs the model of updated state-values for a given batch of
        states.

        Args:
            target_return:  A batch of state-values as a numpy array. These
                values should be a new, ideally better estimate of the return at
                the given state. The model will use this to improve.
            state:  A batched representation of the state of the environment for
                which the value function will be updated, as a numpy array.
        """
        pass
