# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass

from . import Model


class ActionValue(with_metaclass(ABCMeta, Model)):
    """An abstract class that represents an Action-Value model that can be used
    by an agent."""

    @abstractmethod
    def __init__(self, *, load_model=None):
        super(ActionValue, self).__init__(load_model=load_model)
        pass

    @abstractmethod
    def save(self, save_path):
        super(ActionValue, self).save(save_path)
        pass

    @abstractmethod
    def predict_q(self, state, action):
        """Predicts the action-value q for a batch of state-action pairs.

        Args:
            state:  A batched representation of the state of the environment as
                a numpy array.
            action:  A batched representation of the action to take.

        Returns:
            A batch of action-values as a numpy array.
        """
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
