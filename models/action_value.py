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
        """Predicts the action-value q for a batch of state-action pairs, or
        greedily selects the best action for a state and then returns the value
        of that state-action pair.

        Args:
            state:  A batched representation of the state of the environment as
                a numpy array.
            action:  A batched representation of the action to take. If `None`,
                the action will be automatically selected greedily, such that
                the action will give the maximum value for that state. In that
                case, the action selected will be returned along with the value.

        Returns:
            A batch of action-values as a numpy array. If `action` was null,
            the result will instead be a tuple of the action-values and the
            selected actions.
        """
        pass

    @abstractmethod
    def update_q(self, target_return, state, action):
        """Informs the model of updated action-values for a given batch of
        state-action pairs. Note that `target_return` is capable of being an
        n-step return.

        Args:
            target_return:  A batch of action-values as a numpy array. These
                values should be a new, ideally better estimate of the return at
                the given state-action pair. The model will use this to improve.
            state:  A batched representation of the state of the environment for
                which the value function will be updated, as a numpy array.
            action:  A batched representation of the actions for which the value
                function will be updated.
        """
        pass
