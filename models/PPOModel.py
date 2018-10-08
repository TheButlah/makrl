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
from policy_model import PolicyModel

class PPOModel(PolicyModel):
  """A class that represents the Proximal Policy Optimization that can be used by an
  agent."""

  def __init__(self, state_shape, action_shape,
               hidden_neuron_list=None,
               learning_rate=0.001,
               scope=None,
               load_model=None,
               epsilon=0.2
               ):


    """Constructs the PPOModel and initializes it.

    Args:
      state_shape:  The shape tuple of a single state tensor.
      action_shape:  The shape tuple of a single action tensor.
      hidden_neuron_list:  A list of integers larger than 0, for the number of
        neurons/features in each hidden layer of the model. The length of this
        list is how many hidden layers there will be. If `None`, uses default
        number of layers/features.
      learning_rate:  The learning rate for the model.
      scope:  A string or scope for the model. If `None`, a default scope will
        be used.
      load_model:  A string giving a path to the model to load.

      #TODO: Explain parameters
    """
    super(PPOModel, self).__init__(
      state_shape, action_shape, load_model=load_model)

    if self._hidden_neuron_list is None:
        self._hidden_neuron_list = None # sort of arbitrary default to have 2 layers?
    else:
        self._hidden_neuron_list = tuple(hidden_neuron_list)

    self._learning_rate = learning_rate
    self._epsilon = epsilon

    with tf.variable_scope(scope, default_name='PPO') as the_scope:
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
    # Data Inputs and Hyperparameters #
    ###################################
    with tf.variable_scope('Inputs'):
      # The state tensor
      self._state = tf.placeholder(
        tf.float32, shape=self._s_shape, name='State')

      # The action tensor
      self._action = tf.placeholder(
        tf.float32, shape=self._a_shape, name='Action')

      # Whether the model is in a training step
      self._is_training = tf.placeholder(
        tf.bool, shape=(), name='IsTraining')

    with tf.variable_scope('Hyperparams'):
        self._lr = tf.placeholder_with_default(
            self._learning_rate, shape=(), name='LearningRate')
        self._epsilon = tf.placeholder_with_default(self._epsilon)


    # Layers #
    ##########
 h tf.variable_scope('Hidden'):
        # Stack on index after batch. Think equivalent of python `zip` function
        zipped = latest_layer = tf.stack((self._state, self._action),
                                   axis=1, name='Zipped')

    # Build hidden layers
    for n_neurons in self._hidden_neuron_list:
        latest_layer, _ = l.fc(latest_layer, n_neurons,
                               phase_train=self._is_training)

    with tf.variable_scope('Output'):
        # The predicted action-value. Will be a scalar
        self._pred, _ = l.fc(latest_layer, 1, activation=None,
                             phase_train=self._is_training, scope='Pred') #why is scope defined in two ways?

        assert self._pred.shape.as_list() == [None, 1]

        # Squeeze to get rid of singleton feature dimension
        self._pred = tf.squeeze(self._pred)
        assert self._pred.shape.as_list() == [None]

    # Training #
    ############
    with tf.variable_scope('Training'):
        self._loss = tf.losses.
        self._loss = tf.losses.mean_squared_error(
            labels=self., predictions=self._pred, weights=self._mu)
        self._reward_ratio =
        self._clipped_loss = tf.clip_by_value(self._reward_ratio,1-self._epsilon,1+self._epsilon,name='clipped_loss_term')
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr) #TODO: Modular optimizer selection



        # Clip gradients to avoid mega magnitudes
        grads = optimizer.compute_gradients(self._loss)
        clipped = [(tf.clip_by_value(grad, -10., 10.), var)
                 for grad, var in grads]

        # The training operation to call at each training step
        self._train_step = optimizer.apply_gradients(clipped)

  def predict_q(self, states, actions):
    """Uses the model to predict the q function for the given state-action pair.

    Args:
      states:  A numpy ndarray that represents the states of the environment.
        Should should have a shape of [batch_size, height, width, channels].
      actions: A numpy ndarray indicating which actions will be taken.

    Returns:
      A numpy ndarray of the q values, shaped [batch_size].
    """
    assert states.ndim == 4
    assert states.shape[0] == actions.shape[0]

    with self._sess.as_default():

      feed_dict = {
        self._state: states,
        self._action: actions,
        self._is_training: False}

      fetches = self._pred

      return self._sess.run(fetches, feed_dict=feed_dict)



  def update_q(self, target_returns, states, actions,
    mu=None, num_epochs=1, learning_rate=None):
    """Updates the model parameters using the provided target batch.
    Args:
      target_returns: A batch of action-values as a numpy array. These values
        should be a new, ideally better estimate of the return at the given
        state-action pair. The model will use this to improve.
      states:  A batched representation of the state of the environment for
        which the value function will be updated, as a numpy array.
      actions:  A batched representation of the actions for which the value
        function will be updated.
      mu:  A numpy ndarray that contains a weighting factor in range [0,1] for
        each state in the batch. States we care more about approximating
        accurately should be given a higher weight. For example, mu could be
        the fraction of time spent in a given state, which would mean that
        states we pass through often should be more important to approximate
        correctly. If `None`, no weighting will be performed.
      num_epochs:  The number of iterations over the provided batch to perform
        for this update step. It is suggested to keep this 1 so that the model
        doesn't become too biased due to the small size of the batch.
      learning_rate:  The learning rate for the model. If `None`, uses the rate
        defined in the constructor.

    Returns:
      The loss value after the update.
    """
    super(FCQNet, self).update_q(target_returns, states, actions)
    assert (num_epochs >= 1)
    assert (states.ndim > 1)
    assert (states.shape[0] == target_returns.shape[0])
    assert (True if actions is None else states.shape[0] == actions.shape[0])

    with self._sess.as_default():

      feed_dict = {
        self._state: states,
        self._action: actions,
        self._q_target: target_returns,
        self._mu: mu,
        self._is_training: True,
        self._lr: learning_rate}

      # Choose whether to compute the best action or use the provided ones
      fetches = [self._train_step, self._loss]

      # Training step(s)
      for epoch in range(num_epochs):
        result = self._sess.run(fetches, feed_dict=feed_dict)

      return np.squeeze(result[1:])

  def save(self, save_path=None):
    """Saves the model in the specified file.
    Args:
      save_path:  The relative path to the file. By default, it is
        saved/SimpleFCN-Year-Month-Date_Hour-Minute-Second.ckpt
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



