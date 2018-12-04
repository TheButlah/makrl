# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from time import strftime

from six.moves import range, zip

from . import ActionModel

import tensorflow as tf
import numpy as np
import os

import layers as l


class FCQNet(ActionModel):
  """A class that represents an Fully Connected Q Network that can be used by an
  agent."""

  def __init__(self, state_shape, action_shape,
    hidden_neuron_list=None, learning_rate=0.001, step_major=False,
    scope=None,
    load_model=None):
    """Constructs the model and initializes it.

    Args:
      state_shape:  A tuple representing the shape of the environment's state
        space.
      action_shape:  A tuple representing the shape of the environment's action
        space.
      hidden_neuron_list:  A list of integers larger than 0, for the number of
        neurons/features in each hidden layer of the model. The length of this
        list is how many hidden layers there will be. If `None`, uses default
        number of layers/features.
      learning_rate:  The learning rate for the model.
      step_major:  Whether or not the first dimension of the any arguments with
        a step dimension will be the first axis or the second. If `True`, the
        shapes described for any such arguments should have their first two
        dimensions transposed. There may be performance benefits to having the
        step dimension first instead of the batch dimension, but because batched
        computation typically has the batch dimension first, this parameter is
        `False` by default.
      scope:  A string or scope for the model. If `None`, a default scope will
        be used.
      load_model:  A string giving a path to the model to load.
    """
    super(FCQNet, self).__init__(
      state_shape, action_shape, step_major=step_major, load_model=load_model)

    if self._hidden_neuron_list is None:
      self._hidden_neuron_list = (128,)*2
    else:
      self._hidden_neuron_list = tuple(hidden_neuron_list)

    self._learning_rate = learning_rate

    with tf.variable_scope(scope, default_name='FCQNet') as the_scope:
      self._scope = the_scope
      self._build_graph()
      self._init_vars(load_model)

  def _init_vars(self, load_model_from=None):
    """Initializes the model variables, potentially from an already existing
    model file.

    Args:
      load_model_from:  If `None`, initializes the variables. Otherwise, loads
        the model variables from the given checkpoint.
    """
    self._sess = tf.Session()
    with self._sess.as_default():
      self._saver = tf.train.Saver()
      if load_model_from is not None:
        print("Restoring model...")
        load_model_from = os.path.abspath(load_model_from)
        self._saver.restore(self._sess, load_model_from)
        print("Model restored!")
      else:
        print("Initializing model...")
        self._sess.run(tf.global_variables_initializer())
        print("Model initialized!")

  def _build_graph(self):
    """Builds the model graph."""

    # TODO: Deal with the case when `predict_q` has `actions` set to `None`
    # TODO: Convert to n-step instead of 1 step
    # TODO: Deal with rest of framework api change
    # Data Inputs and Hyperparameters #
    ###################################
    with tf.variable_scope('Inputs'):
      # The state tensor
      self._state = tf.placeholder(
        tf.float32, shape=self._s_shape, name='State')

      # The action tensor
      self._action = tf.placeholder(
        tf.float32, shape=self._a_shape, name='Action')

      # The target action-value to update the model towards
      self._rewards = tf.placeholder(
        tf.float32, shape=(None,), name='QTarget')

      # The weighting factor of the state-action pair.
      self._mu = tf.placeholder_with_default(
        tf.ones(tf.shape(self._state)), shape=(None,), name='Mu')

      # Whether the model is in a training step
      self._is_training = tf.placeholder(
        tf.bool, shape=(), name='IsTraining')

    with tf.variable_scope('Hyperparams'):
      self._lr = tf.placeholder_with_default(
        self._learning_rate, shape=(), name='LearningRate')

    # Layers #
    ##########
    with tf.variable_scope('Hidden'):
      # Stack on index after batch. Think equivalent of python `zip` function
      zipped = latest_layer = tf.stack((self._state, self._action),
                                       axis=1, name='Zipped')

      # Build hidden layers
      for n_neurons in self._hidden_neuron_list:
        latest_layer, _ = l.fc(latest_layer, n_neurons,
                               phase_train=self._is_training)

    with tf.variable_scope('Output'):
      # The predicted action-value. Will be a scalar
      self._q_pred, _ = l.fc(latest_layer, 1, activation=None,
                             phase_train=self._is_training, scope='QPred')

      assert self._q_pred.shape.as_list() == [None, 1]

      # Squeeze to get rid of singleton feature dimension
      self._q_pred = tf.squeeze(self._q_pred)
      assert self._q_pred.shape.as_list() == [None]

    # Training #
    ############
    with tf.variable_scope('Training'):

      # TODO: q_target doesn't exist anymore
      self._loss = tf.losses.mean_squared_error(
        labels=self._q_target, predictions=self._q_pred, weights=self._mu)

      optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)

      # Clip gradients to avoid mega magnitudes
      grads = optimizer.compute_gradients(self._loss)
      clipped = [(tf.clip_by_value(grad, -10., 10.), var)
                 for grad, var in grads]

      # The training operation to call at each training step
      self._train_step = optimizer.apply_gradients(clipped)

  def predict_q(self, states, actions=None):
    """Predicts the action-value q for a batch of state-action pairs. If
    `actions` is not provided, the optimal action for each element in the batch
    will be selected from all possible actions and returned in addition to the
    action-value.

    Args:
      states:  A batched representation of the states of the environments.
        Shaped `(batch_size,) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.
      actions:  A batched representation of the actions to take. If `None`, the
        action will be automatically selected greedily, such each action will
        give the maximum value for that state. In that case, the action selected
        will be returned along with the value. Shaped
        `(batch_size,) + action_shape`, where `action_shape` is a tuple
        representing the shape of the environment's action space.

    Returns:
      A batch of action-values as a numpy array. If `actions` was `None`, the
      result will instead be a tuple of the action-values and the selected
      actions.
    """
    assert states.ndim == 4
    assert states.shape[0] == actions.shape[0]

    with self._sess.as_default():

      feed_dict = {
        self._state: states,
        self._action: actions,
        self._is_training: False}

      fetches = self._q_pred

      return self._sess.run(fetches, feed_dict=feed_dict)

  def update_q(self, states, actions, rewards, mu=None, num_steps=None,
    step_major=False, num_epochs=1, learning_rate=None):
    """Updates the model based on experience gained from a given batch of
    observed state-action-reward transitions, each represented in separate numpy
    arrays. This function looks at the TD error for each step in the batch of
    observations and uses that error to update the model. The update will
    base its estimate of the return using the last value of the sequence, which
    will be a return value instead of a reward value. This final return will be
    based on some value determined by the `Agent`. It will be the best estimate
    for the return experienced at the latest action in `observations`, taking
    into account the actual experienced reward when the latest action was taken,
    and hence will be more accurate than the current predicted return at that
    action. By using this return, this function can compute the TD errors and
    update the model.

    Args:
      states:  A batch of observed states from transitions of the environment.
        Shaped `(batch_size, max_steps) + state_shape`, where `state_shape` is a
        tuple representing the shape of the environment's state space.
      actions:  A batch of observed states from transitions of the environment.
        Shaped `(batch_size, max_steps) + action_shape`, where `action_shape` is
        a tuple representing the shape of the environment's action space.
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
      step_major:  Whether or not the first dimension of the various arguments
        with a step dimension will be the first axis or the second. If `True`,
        the shapes described in this docstring for `states`, `actions`,
        `rewards`, and `mu` should have their first two dimensions transposed.
        There may be performance benefits to having the step dimension first
        instead of the batch dimension, but because batched computation usually
        has the batch dimension first, this parameter is `False` by default.
      num_epochs:  The number of iterations over the provided batch to perform
        for this update step. It is suggested to keep this 1 so that the model
        doesn't become too biased due to the small size of the batch.
      learning_rate:  The learning rate for the model. If `None`, uses the rate
        defined in the constructor.

    Returns:
      The net loss/TD error for all steps averaged over the batch.
    """
    super(FCQNet, self).update_q(
      states, actions, rewards, mu=mu, num_steps=num_steps)
    assert (num_epochs >= 1)
    assert (states.ndim > 1)
    # TODO: add check to make sure mu num_steps batch dims are same as others
    # TODO: add check for num_steps being <= max_steps
    assert (states.shape[d] == actions.shape[d] == rewards.shape[d]
            for d in (0, 1))

    # TODO: compute returns (in tensorflow?)
    # returns = None

    with self._sess.as_default():

      feed_dict = {
        self._state: states,
        self._action: actions,
        self._rewards: rewards,
        # self._returns: returns, # TODO: Should we make this placeholder?
        self._mu: mu,
        self._is_training: True,
        self._lr: learning_rate}

      # Choose whether to compute the best action or use the provided ones
      fetches = [self._train_step, self._loss]  # TODO: Should we fetch returns?

      # Training step(s)
      for epoch in range(num_epochs):
        result = self._sess.run(fetches, feed_dict=feed_dict)

      # TODO: We should also be returning returns
      return np.squeeze(result[1:])

  def save(self, save_path):
    """Saves the model in the specified file.
    Args:
      save_path:  The relative path to the file.
    """
    with self._sess.as_default():
      print("Saving Model")
      if save_path is None:
        save_path = "saved/SimpleFCN-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
      dirname = os.path.dirname(save_path)
      if dirname is not '':
        os.makedirs(dirname, exist_ok=True)
      save_path = os.path.abspath(save_path)
      path = self._saver.save(self._sess, save_path)
      print("Model successfully saved in file: %s" % path)
