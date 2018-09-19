"""
Dueling DQN with a Prioritized Experience Replay. This is based on this paper: https://arxiv.org/abs/1511.06581

This module stores the Dueling DQN along with the Prioritized Memory Module that the NN uses to learn. The Advantage of writing the Dueling DQN this way, is that we can, with boolean arguments, turn off certain features until it becomes a vanilla DQN.

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import os
import random

class PriorityTree(object):
    """
    """

    data_pointer = 0

    def __init__(self, capacity):
        """
        """
        self.capacity = 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity,dtype=object)


    def update(self, tree_index, p):
        """
        """
        change= p - self.tree[tree_index]
        self.tree[tree_index]= p

        # propogate change throughout
        while tree_index != 0:
            tree_index= (tree_index - 1)//2
            self.tree[tree_index] += change


    def add(self, p, data):
        """
        """
        tree_index= self.data_pointer + self.capacity - 1
        self.data[self.data_pointer]= data #update vector of data
        self.update(tree_index,p) #update the Priority Tree

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer= 0


    def get_node(self, v):
        """
        """
        parent_index = 0

        while True:
            cl_index= 2 * parent_index + 1 # Left Child Index
            cr_index= cl_index + 1 # Right Child Index

            if cl_index >= len(self.tree):
                leaf_index= parent_index
                break
            else:
                if v <= self.tree[cl_index]:
                    parent_index= cl_index
                else:
                    v -= self.tree[cl_index]
                    parent_index= cr_index

        data_index= leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]


    @property
    def total_p(self):
        return self.tree[0] # the root


class Memory(object):
    """
    """
    epsilon= 0.01 # avoid zero priority (make small)
    alpha= 0.6 # [0~1] convert importance of TD error to a priority
    beta= 0.4 # importance-sampling, increasing to 1
    beta_increment_per_sampling= 0.001
    abs_error_upper= 1. #clipped absolute error


    def __init__(self,capacity):
        """
        """
        self.tree= PriorityTree(capacity)


    def store(self, transition):
        """
        """
        max_priority= np.max(self.tree.tree)
        if max_priority == 0:
            max_priority= self.abs_error_upper

        self.tree.add(max_priority, transition)


    def sample(self, n):
        """
        """
        batch_index, batch_memory, weight= np.empty((n,)), np.empty((n,self.tree.data.size[0])), np.empty((n,1))
        priority_segment= self.tree.total_p / n
        self.beta= np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob= np.min(self.tree.tree) / self.tree.total_p
        for i in range(n):
            a, b= priority_segment*i, priority_segment*(i+1)
            v= np.random.uniform(a,b)
            index, p, data= self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            weights[i,0]= np.power(prob/min_prob, -self.beta)
            batch_index[i], batch_memory[i, :]= index, data

        return batch_index, batch_memory, weights


    def batch_update(self,tree_index,abs_error):
        abs_error += self.epsilon
        clipped_error= np.minimum(abs_error, abs_error_upper)
        priors= np.pow(clipped_error, self.alpha)

        for ti, p  in zip(tree_index,priors):
            self.tree.update(ti,p)


class DuelingDQNPrioritizedReplay(object):
    """
    """
    def __init__(
        self,
        n_actions,
        n_features,
        learning_rate=0.001,
        gamma=0.9,
        epsilon=0.9,
        replace_target_iter=500,
        memory_size=5000,
        batch_size=100,
        epsilon_increment=None,
        output_graph=False,
        dueling=True,
        prioritized=True,
        sess=None,
        image_data=False
    ):
        """
        """
        self.n_actions= n_actions
        self.n_features= n_features
        self.lr= learning_rate
        self.gamma= gamma
        self.epsilon= epsilon
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon

        self.prioritized= prioritized
        self.dueling= dueling

        self.learn_step_counter= 0
        self.memory_counter= 0

        self._build_network()

        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features*2+2))

        t_params= tf.get_collection('target_net_params')
        e_params= tf.get_collection('eval_net_params')
        self.replace_target_op= [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess= tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess= sess
        if output_graph:
            tf.summary.FileWriter("./logs/", self.sess.graph)

        self.cost_history= []

    def _build_layers(self,state,col_names,n_l1):
        """
        """
        with tf.variable_scope('l1'):
            self.w1= tf.get_variable('w1', [self.n_features, n_l1], collections=col_names)
            self.b1= tf.get_variable('b1', [1,n_l1], collections=col_names)
            self.l1= tf.nn.relu(tf.matmul(state, self.w1) + self.b1)

        if self.dueling:
            #Dueling DQN!
            with tf.variable_scope('value'):
                self.w2v= tf.get_variable('w2v', [n_l1, 1], collections=col_names)
                self.b2v = tf.get_variable('b2v', [1, 1], collections=col_names)
                self.V = tf.matmul(self.l1, self.w2v) + self.b2v

            with tf.variable_scope('Advantage'):
                self.w2a = tf.get_variable('w2a', [n_l1, self.n_actions],collections=col_names)
                self.b2a = tf.get_variable('b2a', [1, self.n_actions], collections=col_names)
                self.A = tf.matmul(self.l1, self.w2a) + self.b2a

            with tf.variable_scope('Q'):
                out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a) eq(9) from paper
        else:
            with tf.variable_scope('Q'):
                self.w2 = tf.get_variable('w2', [n_l1, self.n_actions], collections=col_names)
                self.b2 = tf.get_variable('b2', [1, self.n_actions], collections=col_names)
                out = tf.matmul(self.l1, self.w2) + self.b2

        return out


    def _build_network(self):
        """
        """
        self.state= tf.placeholder(tf.float32, [None,self.n_features],name='state')
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')

        #--- Eval Net ----
        with tf.variable_scope('eval_net'):
            col_names, n_l1 = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20 # configuration of layers

            self.q_eval= self._build_layers(self.state,col_names,n_l1)
            if self.prioritized:
                self.weights = tf.placeholder(tf.float32, [None, 1], name='weights')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)


        #--- Target Net ----
        self.state_ = tf.placeholder(tf.float32, [None, self.n_features],name='state_')
        with tf.variable_scope('target_net'):
            col_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = self._build_layers(self.state_, col_names, n_l1)


    def store_trans(self,state,action,reward,state_):
        """
        """
        if self.prioritized:    # prioritized replay
            transition = np.hstack((state, [action, reward], state_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            transition = np.hstack((state, [action, reward], state_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1


    def pick_action(self,obs):
        """
        """
        obs= obs[np.newaxis, :]
        if np.random.uniform() < self.epsilon:  # choosing action
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: obs})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action



    def learn(self):
        """
        """
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')

        if self.prioritized:
            tree_index, batch_memory, weights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]


        q_next = self.sess.run(self.q_next, feed_dict={self.state_: batch_memory[:, -self.n_features:]}) # next observation
        q_eval = self.sess.run(self.q_eval, {self.state: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.state: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.weights: weights})
            self.memory.batch_update(tree_idx, abs_errors)     # update priority
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.state: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})
        self.cost_history.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
