# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division


class Environment(object):
    """Represents the environment that the agent(s) will interact with."""

    def __init__(self, max_trajectory_length=None):
        """Constructs an Environment object.

        Args:
            max_trajectory_length:  An int for the maximum length of the
                trajectory to keep. If `None`, the trajectory will have an
                unbounded length.
        """

        self._action_dict = None
        """This is a dict of strings that maps to the internal action type"""

        raise NotImplementedError

    def step(self, action):
        """Steps the environment based on a provided action.

        Args:
            action: The action to take when stepping the environment. The action
                should be one of the elements of the action dictionary.
        """
        raise NotImplementedError

    def get_action_dict(self):
        """This is a dict of strings that map to the internal action type."""
        return self._action_dict
