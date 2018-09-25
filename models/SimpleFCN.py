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

  def __init__(self, state_shape, action_shape, *,
               hidden_neuron_list=None, learning_rate=0.001,
               scope=None,
               load_model=None):
    """Constructs the model and initializes it.

    Args:
      state_shape:  The shape tuple of a single state tensor.
      action_shape:  The shape tuple of a single action tensor.
      hidden_neuron_list:  A list of integers larger than 0, for the number of
        neurons/features in each hidden layer of the model. The length of this
        list is how many hidden layers there will be. If `None`, uses default
        number of layers/features.
      load_model:  A string giving a path to the model to load.
    """
    super(FCQNet, self).__init__(
      state_shape, action_shape, load_model=load_model)

    if self._hidden_neuron_list is None:
      self._hidden_neuron_list = tuple([128]*2)
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
      self._q_target = tf.placeholder(
        tf.float32, shape=(None,), name='QTarget')

      # The weighting factor of the state-action pair.
      self._mu = tf.placeholder(
        tf.float32, shape=(None,), name='Mu')

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

      self._loss = tf.losses.mean_squared_error(
        labels=self._q_target, predictions=self._q_pred, weights=self._mu)

      optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)

      # Clip gradients to avoid mega magnitudes
      grads = optimizer.compute_gradients(self._loss)
      clipped = [(tf.clip_by_value(grad, -10., 10.), var)
                 for grad, var in grads]

      # The training operation to call at each training step
      self._train_step = optimizer.apply_gradients(clipped)

  def predict_q(self, state, action):
    # TODO: Implement
    pass

  def update_q(self, target_return, state, action):
    # TODO: Implement
    pass

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