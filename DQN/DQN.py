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
        tree_index= self.data_pointer + capacity - 1
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
            max_priority= abs_error_upper

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
