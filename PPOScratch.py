'''
A preliminary PPO Implementation by Shubhom.
This will be integrated with the established
framework for Halite in the future.

See original paper: https://arxiv.org/pdf/1707.06347.pdf
'''

import tensorflow as tf

class Policy():
    '''
    A policy is created from a default Tensorflow graph
    '''

    def __init__(self,architecture):
        '''
        TODO: Make this make sense
        @ param: architecture
        '''


    def create(self,input_dim,output_dim,hid_size,num_hid_layers=2):
        with tf.get_default_graph() as graph:
            last_out = tf.placeholder(input_dim)
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(
                            tf.layers.dense(last_out,
                            hid_size[i],
                            name="fc%i" % (i + 1),
                            kernel_initializer=tf.initializers.random_normal)


class PPOModel():
    '''
    An instance of the PPOModel contains the old and new policy and
    model specific hyperparameters.
    '''

    def __init__(self,environment,policy,horizon,epsilon=0.2):
        '''
        TODO: Specs
        @param environment: Gym-like environment agent is trained on
        @param policy: Policy network definition. This should be a list of layers.
        @param horizon: Timesteps we look at to run old policy
        @param epsilon: Advantage clipping coefficient
        '''

        self.env = environment
        self.old_policy = self.policy_instance(policy,'old')
        self.new_policy = self.policy_instance(policy,'new')
        self.T = horizon
        self.epsilon = epsilon

    def policy_instance(self,policy,scope):
        '''
        TODO: Specs
        :param policy:
        :param scope:
        :return:
        '''

        if scope != 'old' and scope != 'new':
            raise ValueError ('Policy must be instantiated as "old" or "new"')
        with tf.get_default_graph() as graph:
            # Want to separate policy network architecture out of PPO Model implementation?
            with tf.variable_scope(scope):
                policy.create()

