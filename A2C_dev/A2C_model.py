import numpy as np
import tensorflow as tf

import sys
import inspect

sys.path.append("../")

#from baselines.common.distributions import make_pdtype
from distributions import make_pdtype
from layers import conv, fc, pool
import gym
from . import PolicyModel, ValueModel

from six.moves import range, zip

from abc import ABCMeta, abstractmethod
from six import with_metaclass

class A2CNetwork(object):
    """Docstring for Network Object (Only Used for this A2C structure)"""
    def __init__(self,sess,obs_space,action_space,nbatch,nsteps,hidden_neuron_list=None,images=True,reuse=True):

        # Based on the action space, will select what probability distribution type
        # to distribute action in our stochastic policy (from OpenAI baselines)
        self.sess = sess
        self.pdtype = make_pdtype(action_space)
        self.action_space = action_space

        obs_shape = obs_space.shape
        if isinstance(obs_space,gym.spaces.Box) and images:
            height, width, channel = obs_space.shape
            obs_shape = (height,width,channel)

        self.inputs = tf.placeholder(tf.float32, [None, *obs_shape])

        with tf.variable_scope("PPO_A2C",reuse = reuse):
            if images:
                self.conv1, _ = conv(self.inputs,32,size=8,stride=4)
                self.conv2, _ = conv(self.conv1,32,size=4,stride=2)
                self.conv3, _ = conv(self.conv2,16,size=3,stride=1)
                self.flat = tf.reshape(self.conv3,[-1,np.prod(self.conv3.shape[1:])]) #tf.contrib.layers.flatten(self.conv3)
            else:
                self.fcc1, _  = fc(self.inputs,50)
                self.fcc2, _ = fc(self.fcc1,50)
                self.flat, _ = fc(self.fcc2,50)
            
            self.fc1, _ = fc(self.flat,512)

            self.pd, self.pi = self.pdtype.pdfromlatent(self.fc1, init_scale=0.01)

             # Take an action. This is a schochatic policy so we don't always take
             # the highest probability action.
            self.action = self.pd.sample()

            # Calculate the neg log of our probability
            self.neglogp = self.pd.neglogp(self.action)

            self.v, _ = fc(self.fc1,1,activation=None)

        self.initial_state = None


    def select_action(self,obs, *_args, **_kwargs):
        return self.sess.run(self.action, {self.inputs: obs})
        #return np.asarray([self.action_space.sample() for _ in range(obs.shape[0])])

    def value(self,obs, *_args, **_kwargs):
        return self.sess.run(self.v, {self.inputs: obs})
        #return np.zeros_like((obs.shape[0],))

    def eval_step(self,obs, *_args, **_kwargs):
        action, value, neglogp = self.sess.run([self.action, self.v ,self.neglogp], {self.inputs: obs})
        #action = np.asarray([self.action_space.sample() for _ in range(obs.shape[0])])
        #value = np.zeros((obs.shape[0],))
        #assert (value.shape == (obs.shape[0],))
        #neglogp = np.zeros((obs.shape[0],1))
        return action, value, neglogp


class A2CModel(PolicyModel,ValueModel):
    def __init__(self,
                sess,
                policy,
                ob_space,
                action_space,
                nenvs,
                nsteps,
                ent_coef,
                vf_coef,
                max_grad_norm,
                images=True):

        super(PolicyModel, self).__init__(ob_space, action_space, step_major=False,load_model=None)
        self.sess = sess

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        # CREATE THE PLACEHOLDERS
        self.actions = tf.placeholder(tf.int32, [None], name="actions")
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")

        # CREATE OUR TWO MODELS
        # Step_model that is used for sampling
        self.step_model = policy(self.sess, ob_space, action_space, nenvs, 1, reuse=False,images=images)

        # Test model for testing our agent
        #test_model = policy(sess, ob_space, action_space, 1, 1, reuse=False)

        # Train model for training
        self.train_model = policy(self.sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True,images=images)


        self.value_pred = self.train_model.v

        self.value_loss = tf.reduce_mean(tf.square(self.value_pred - self.rewards))

        # Output -log(pi) (new -log(pi))
        self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_model.pi, labels=self.actions)

        # Final PG loss
        self.pg_loss = tf.reduce_mean(self.advantages * self.neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        self.entropy = tf.reduce_mean(self.train_model.pd.entropy())

        self.loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

        # 1. Get the model parameters
        self.params = tf.trainable_variables()

        # 2. Calculate the gradients
        self.grads = tf.gradients(self.loss, self.params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            self.grads, self.grad_norm = tf.clip_by_global_norm(self.grads, max_grad_norm)
        self.grads = list(zip(self.grads, self.params))

        # 3. Build our trainer
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr, epsilon=1e-5)

        # 4. Backpropagation
        self._train = self.trainer.apply_gradients(self.grads)

        self.eval_step = self.step_model.eval_step

        self.predict_pi = self.step_model.select_action
        self.predict_v = self.step_model.value

        self.initial_state = self.step_model.initial_state
        self.sess.run(tf.global_variables_initializer())


    def predict_v(self, states, mu=None):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)


    def update_v(self, states, actions, rewards, mu=None, num_steps=None):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)

    def predict_pi(self, states):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)

    def update_pi(self, advantages, states):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)

    def save(self,save_path):
        """
        Save the model
        """
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self,load_path):
        """
        Load the model
        """
        saver = tf.train.Saver()
        print('Loading ' + load_path)
        saver.restore(self.sess, load_path)


    def learn(self, states_in, actions, returns, values, neglogpacs, lr, cliprange):

        values = np.squeeze(values)
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advantages = returns - values

        #print(advantages.shape)
        #print(returns.shape)
        #print(values.shape)
        
        # Normalize the advantages (taken from aborghi implementation)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # We create the feed dictionary
        feed_dict = {self.train_model.inputs: states_in,
                 self.actions: actions,
                 self.advantages: advantages, # Use to calculate our policy loss
                 self.rewards: returns, # Use as a bootstrap for real value
                 self.lr: lr,
                 self.cliprange: cliprange,
                 self.oldneglopac: neglogpacs,
                 self.oldvpred: values}

        policy_loss, value_loss, policy_entropy, _= self.sess.run([self.pg_loss, self.vf_loss, self.entropy, self._train], feed_dict)
        #policy_loss, value_loss, policy_entropy = 0,0,0

        return policy_loss, value_loss, policy_entropy


class A2C_PPOModel(PolicyModel,ValueModel):
    def __init__(self,
                sess,
                policy,
                ob_space,
                action_space,
                nenvs,
                nsteps,
                ent_coef,
                vf_coef,
                max_grad_norm,
                images=True):

        super(PolicyModel, self).__init__(ob_space, action_space, step_major=False,load_model=None)
        self.sess = sess

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        # CREATE THE PLACEHOLDERS
        self.actions = tf.placeholder(tf.int32, [None], name="actions")
        self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
        self.rewards = tf.placeholder(tf.float32, [None], name="rewards")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        # Keep track of old actor
        self.oldneglopac = tf.placeholder(tf.float32, [None], name="oldneglopac")
        # Keep track of old critic
        self.oldvpred = tf.placeholder(tf.float32, [None], name="oldvpred")
        # Cliprange
        self.cliprange = tf.placeholder(tf.float32, [])

        # CREATE OUR TWO MODELS
        # Step_model that is used for sampling
        self.step_model = policy(self.sess, ob_space, action_space, nenvs, 1, reuse=False,images=images)

        # Test model for testing our agent
        #test_model = policy(sess, ob_space, action_space, 1, 1, reuse=False)

        # Train model for training
        self.train_model = policy(self.sess, ob_space, action_space, nenvs*nsteps, nsteps, reuse=True,images=images)


        self.value_pred = self.train_model.v
        self.value_pred_clipped = self.oldvpred + tf.clip_by_value(self.train_model.v - self.oldvpred,  - self.cliprange, self.cliprange)

        # Unclipped and value loss
        self.value_loss_unclipped = tf.square(self.value_pred - self.rewards)
        self.value_loss_clipped = tf.square(self.value_pred_clipped - self.rewards)

        # Value loss 0.5 * SUM [max(unclipped, clipped)
        self.vf_loss = 0.5 * tf.reduce_mean(tf.maximum(self.value_loss_unclipped,self.value_loss_clipped))

        # Output -log(pi) (new -log(pi))
        self.neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_model.pi, labels=self.actions)

        # we want ratio (pi current policy / pi old policy)
        # But neglopac returns us -log(policy)
        # So we want to transform it into ratio
        # e^(-log old - (-log new)) == e^(log new - log old) == e^(log(new / old))
        # = new/old (since exponential function cancels log)
        # Wish we can use latex in comments

        self.ratio = tf.exp(self.oldneglopac - self.neglogpac)

        # doing Gradient ascent, so we can multiply the advantages by -1
        # when performing the (pi new / pi old) * - Advantages operation
        self.pg_loss_unclipped = -self.advantages * self.ratio

        # value, min [1 - e] , max [1 + e]
        self.pg_loss_clipped = -self.advantages * tf.clip_by_value(self.ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)

        # Final PG loss
        self.pg_loss = tf.reduce_mean(tf.maximum(self.pg_loss_unclipped, self.pg_loss_clipped))

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        self.entropy = tf.reduce_mean(self.train_model.pd.entropy())

        self.loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

        # 1. Get the model parameters
        self.params = tf.trainable_variables()

        # 2. Calculate the gradients
        self.grads = tf.gradients(self.loss, self.params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            self.grads, self.grad_norm = tf.clip_by_global_norm(self.grads, max_grad_norm)
        self.grads = list(zip(self.grads, self.params))

        # 3. Build our trainer
        self.trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr, epsilon=1e-5)

        # 4. Backpropagation
        self._train = self.trainer.apply_gradients(self.grads)

        self.eval_step = self.step_model.eval_step

        self.predict_pi = self.step_model.select_action
        self.predict_v = self.step_model.value

        self.initial_state = self.step_model.initial_state
        self.sess.run(tf.global_variables_initializer())


    def predict_v(self, states, mu=None):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)


    def update_v(self, states, actions, rewards, mu=None, num_steps=None):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)

    def predict_pi(self, states):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)

    def update_pi(self, advantages, states):
        fileName = inspect.stack()[1][1]
        line = inspect.stack()[1][2]
        method = inspect.stack()[1][3]

        print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
        sys.exit(1)

    def save(self,save_path):
        """
        Save the model
        """
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)

    def load(self,load_path):
        """
        Load the model
        """
        saver = tf.train.Saver()
        print('Loading ' + load_path)
        saver.restore(self.sess, load_path)


    def learn(self, states_in, actions, returns, values, neglogpacs, lr, cliprange):

        values = np.squeeze(values)
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advantages = returns - values

        #print(advantages.shape)
        #print(returns.shape)
        #print(values.shape)
        
        # Normalize the advantages (taken from aborghi implementation)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # We create the feed dictionary
        feed_dict = {self.train_model.inputs: states_in,
                 self.actions: actions,
                 self.advantages: advantages, # Use to calculate our policy loss
                 self.rewards: returns, # Use as a bootstrap for real value
                 self.lr: lr,
                 self.cliprange: cliprange,
                 self.oldneglopac: neglogpacs,
                 self.oldvpred: values}

        policy_loss, value_loss, policy_entropy, _= self.sess.run([self.pg_loss, self.vf_loss, self.entropy, self._train], feed_dict)
        #policy_loss, value_loss, policy_entropy = 0,0,0

        return policy_loss, value_loss, policy_entropy