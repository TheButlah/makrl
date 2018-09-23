import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from experience import ReplayBuffer,Experience  #used to implement action replay


# This is a Deep-Q-learning demo with the Atari Ms Pacman game from gym
# See 'Playing Atari with Deep Reinforcement Learning' by Mnih et. al. from DeepMind
# Observation output of env.step() is the screen image

# We need to preprocess the images to speed up training
def rgb2gray(frame):
    return np.dot(frame, [0.299, 0.587, 0.114])

def preprocess_observation(frame):
    """
    Image croping step used from tiny_dqn
    :param obs:frame to process
    :return: cropped and processed image
    """
    img = rgb2gray(frame)
    img = img[1:176:2, ::2] # crop and downsize
    # img = img.mean(axis=2) # to greyscale
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

# A simple object to manage game frames
class FrameCache(object):
    def __init__(self,size):
        self.num_frames = size
        self.stacks = [] #empty list of frame stacks


    def add_base_frame(self,frame):
        """
        Add a new sequences starting with frame
        :param frame: input game state
        :return: none
        """
        self.stacks.append(([frame]))
    def len_stacks(self):
        """
        Returns length of stacks list to update during training
        :return: length of stacks
        """
        return len(self.stacks)
    def len_full_stacks(self):
        """
        Number of "complete" stacks i.e. number of complete input states
        :return: number of full stacks of size num_frames
        """
        sum = 0
        for i in range (0,self.len_stacks()):
            if len(self.stacks[i]) == self.num_frames-1:
                sum+=1
        return sum

    def add_to_stack(self,frame,index):
        """
        Add frame to the sequence beginning at index
        :param frame: input game state
        :param index: Index where sequence begins
        :return: none
        """
        if len(self.stacks[index])<self.num_frames:
            self.stacks[index].append(frame)
        else:
            pass

    def clear(self):
        """
        Clear a sequence
        :return: None
        """
        self.stacks = []

    def clear_stack(self,index):
        """

        :param index: Index where sequence begins
        :return:
        """
        del (self.stacks[index])




    def get_last_fulls(self,test=False):

        i = (len(self.stacks))-1

        if test:
            if i==100:
                j = i
                while j>90:
                    print (len(self.stacks[j]))
                    j = j-1

        while i>1:
            if len(self.stacks[i])==self.num_frames:
                if len(self.stacks[i-1])==self.num_frames:
                    return self.stacks[i-1],self.stacks[i]
                else:
                    pass
            else:
                i = i - 1
        return None,None

        #
        # while i>0 and len(self.stacks)>1:
        #     print (i)
        #     print ('Len self.stacks [i-1]',len(self.stacks[i-1]))
        #     if len(self.stacks[i]) == self.num_frames:
        #         assert len(self.stacks[i-1]) == self.num_frames
        #         return self.stacks[i - 1], self.stacks[i]
        #     else:
        #         i = i-1




class PacWoman():

    def __init__(self,ckpt,name='Cornellia',lr=0.001,buffer_size=200000,num_games_trained=100000,):
        self.name = name
        self.ckptdir = ckpt
        self.env = gym.make('MsPacman-v0')
        self.num_games_train = num_games_trained
        # self.max_steps = 100
        self.graph = None
        self.replay = ReplayBuffer(buffer_size=buffer_size)
        # self.alpha = 0.618
        # self.qmatrix = np.zeros
        self.num_pixels = 210*160*3 #screen_dim* channels (i.e length*width*3)
        self.num_frames = 4 #number of game frames that're used for input
        self.learning_rate = lr
        self.epsilons = []
        self.n_actions = self.env.action_space.n  # 9 discrete actions are available
    # This convnet tries to learn the Q-function (i.e. Q-values over state/action space); function q_network initializes graph
    # Important: use name scope to differentiate online and target networks
    def q_network(self,name_scope):

        if name_scope.lower()!='online' and name_scope.lower()!='target':
            raise ValueError('Name scope must be \'online\' or \'target\'')
        if self.graph == None:
            self.graph = tf.Graph()
        # self.num_frames = np.random.randint(2,4) #randomly choose 2,3, or 4 frames
        with self.graph.as_default():


            with tf.variable_scope(name_scope,reuse=tf.AUTO_REUSE):
                input_state = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, 88, 80, 1],
                                                  name='Input')  # input image frame that has been preprocessed

                #change conv layers and pooling depending on image preprocessing
                print(name_scope)
                h1 = tf.layers.conv2d(inputs=input_state,
                        filters=16,
                        strides=[4,4],
                        kernel_size=[8, 8],
                        padding="same",
                        activation=tf.nn.relu,
                        use_bias=True)
                h2 = tf.layers.conv2d(inputs=h1,
                                      filters=32,
                                      strides=[4,4],
                                      kernel_size=[2,2],
                                      padding="same",
                                      activation=tf.nn.relu,use_bias=True)
                h2 = tf.contrib.layers.flatten(h2)
                h3 = tf.layers.dense(h2,units=256,activation=tf.nn.relu,use_bias=True)
                h4 = tf.layers.dense(h3,units=self.n_actions,activation=tf.nn.relu,use_bias=True)

            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope=name_scope)
            trainable_vars_by_name = {var.name[len(name_scope):]: var
                                      for var in trainable_vars}
            return h4,trainable_vars_by_name,input_state

    def print_train_vars(self):
        with self.graph.as_default():

            print ('Online variables: ')
            print (tf.trainable_variables(scope='online'))

            print ('Target variables: ')
            print (tf.trainable_variables(scope='target'))

            print ('Train variables (should be empty):')
            print (tf.global_variables(scope='train'))




    #Predict move after model has been trained
    def predict(self,inp):
        if tf.get_default_session() is None:
            self.load_model()
        fed = {input_state: inp}
        out = tf.get_default_session().run(h4,feed_dict = fed)
        tf.get_default_session().close()
        return out

    #TODO: saver load
    def load_model(self):
        # Load model from specified directory
        if tf.get_default_session() is  None:
            pred_sess = tf.InteractiveSession(graph = self.graph)
        assert tf.get_default_graph is self.graph
        self.saver.restore(pred_sess,self.ckptdir)
        pass

    #TODO: Epilson decay
    def epsilon_greedy(self,qs, epsilon=0.5):
        """
        Epsilon-greedy way to choose next action, where qs is at the current step

        :param qs:  a matrix of q-values across the action space
        :param epsilon: predetermined probability of choosing random action
        :param k: parameter for decay that normalizes step decay of epsilon
        :param show_eps: show epsilon decay graph over time
        :param show_interval: iteration interval for doing above
        :param rand_act: epsilon=1 s.t. a random action is always chosen (helps show baseline)
        :return: which action is chosen
        """
        if not rand_act:
            eps = np.maximum(epsilon*np.exp(-iteration/k),mineps)
        else:
            eps = 1

        rand = np.random.rand()
        if (len(self.epsilons)==0):
            self.epsilons.append(eps)
            has_shown = False
        elif (self.epsilons[len(self.epsilons)-1]!=eps):
            self.epsilons.append(eps)
            has_shown = False
        else:
            has_shown = True

        if show_eps and not has_shown and iteration%show_interval==0 and iteration!=0:
            plt.figure()
            xs = np.arange(0,iteration)
            plt.plot(self.epsilons,'r',label='epsilon')
            plt.title('Epsilon in Game %d'%iteration)
            plt.xlabel('Game Played')
            plt.grid()
            plt.ylabel('Epsilon')
            plt.savefig('Epsilon_over_time.png')
            plt.show(block=False)
            time.sleep(5)
            plt.close()


        if rand<epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(qs)




    def train(self,name_scope,action_verbose=False,update_interval=20,graph_interval = 50,monitor_interval=100,discount_rate=0.99,obs_verbose=False,reward_verbose=False):
        online_q_values, online_vars, online_input = self.q_network(name_scope="online")
        target_q_values, target_vars, target_input = self.q_network(name_scope="target")

        copy_ops = [target_var.assign(online_vars[var_name])
                    for var_name, target_var in target_vars.items()]
        copy_online_to_target = tf.group(*copy_ops)


        with self.graph.as_default() as graph:
            with tf.variable_scope('train',reuse=tf.AUTO_REUSE):
                X_action = tf.placeholder(dtype=tf.int32,shape=[None,1],name='X_action')
                max_reward = tf.placeholder(dtype=tf.float32,shape=[None,1],name='max_reward')
                # expected Bellman reward; self.q variable takes a one-hot encoding of output-action and multiplies
                # by policy net output to determine q-value for given state(s) over all n actions
                # by policy net output to determine q-value for given state(s) over all actions
                q = tf.reduce_sum(online_q_values* tf.one_hot(X_action, self.n_actions),axis=1, keep_dims=True)
                loss = tf.square((max_reward-q))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_opt = optimizer.minimize(loss,var_list=tf.trainable_variables(scope='online')) #important to specify var_list
                tf.add_to_collection(tf.GraphKeys.TRAIN_OP,train_opt)
                self.saver = tf.train.Saver()
                init = tf.global_variables_initializer()

        # Some initialization steps
        train_sess = tf.InteractiveSession(graph=self.graph)
        train_sess.run(init)
        self.replay.clear()
        assert tf.get_default_graph() is self.graph

        # Monitor game by game progress
        agg_rewards = []
        step_agg_rewards = []
        game_steps = []

        # Outer loop for number of games to play
        for i in range(self.num_games_train):
            print ('==========================\nGame #%d begun'%(i+1))
            game_step = 0
            init_state = self.env.reset()
            init_state = preprocess_observation(init_state)
            next_state = None
            done = False
            agg_reward = 0
            frame_count = 0 #Total number of frames encounted in game
            last_frame = None #variable used for pixelwise max operation
            maxop = True #take framewise max of num_frames neighbors
            self.frame_cache = FrameCache(size=self.num_frames)



            # This while loop is **like** each step in the game (not really because we're using multiple frames)
            # Frames go like x1,x2,x3,x4 then x2,x3,x4,x5 ... end of game but skip between
            while done == False:
                game_step += 1
                if i%(render_interval) == 0:
                    self.env.render()
                frame_skip = False #initialize skipping to False
                # q_values = online_q_values.eval(feed_dict={online_input: [next_state],target_input: None})
                # action = self.epsilon_greedy(q_values)
                init_state = np.reshape(init_state,[1,init_state.shape[0],init_state.shape[1],init_state.shape[2]])

                qs = q.eval(feed_dict={online_input: init_state,target_input: np.zeros_like(init_state),X_action:[[0]], max_reward: [[0]]})
                action = self.epsilon_greedy(qs,epsilon=0.6,iteration=i,k=1000,show_eps=True)
                action = self.epsilon_greedy(qs)
                next_state, reward, done, info = self.env.step(action)
                next_state = preprocess_observation(next_state)



                # metrics for monitoring improvement
                agg_reward+=reward
                step_agg_reward = agg_reward/game_step

                if game_step%monitor_interval == 0:
                    print ('Aggregate reward at step %d: %f'%(game_step,agg_reward))
                    print ('Step Aggregate Reward: %f'%step_agg_reward)

                # experience replay addition
                if maxop == True and last_frame is not None:
                    next_state = np.maximum(last_frame,next_state) #take element-wise max of two frames
                if not frame_skip:
                    self.frame_cache.add_base_frame(next_state) #this is going to be the last index of frame stack

                    #maybe not the most efficient way to do this
                    for k in range (0,(self.frame_cache.len_stacks()-1)):
                        self.frame_cache.add_to_stack(next_state,k) #adds frame to every preceding stack if it's not full yet
                frame_count += 1

                #Flip frame_skip every num_frames interval (it's possible to change this interval but let's keep it simple)
                if frame_count % self.num_frames == 0:
                    if frame_skip == True:
                        frame_skip = False
                    else:
                        frame_skip = True

                last_frame = next_state #now this frame is the last state
                init_state = next_state

                if self.frame_cache.len_stacks()>self.num_frames: #at least num_frames actions must be taken before one stack frame is full
                    first_state_frames,second_state_frames = self.frame_cache.get_last_fulls()
                    if first_state_frames is not None and second_state_frames is not None:
                        exp_input = Experience(first_state_frames, action, reward, second_state_frames,done)
                        self.replay.add(exp_input)
            if i%render_interval == 0:
                self.env.close()

            agg_rewards.append(agg_reward)
            step_agg_rewards.append(step_agg_reward)
            game_steps.append(game_step)


            samples = self.replay.sample_batch(batch_size=1) #try doing different batch sizes like Ryan said as parallelization, if batch_size>1 flatten this
            for exp in samples:
           # Experience Fields
            # self.first_state
            # self.action
            # self.reward
            # self.second_state
            # self.terminal

                fed = {target_input:exp.first_state}
                target_qs = target_q_values.eval(feed_dict=fed)

                max_next_q_values = np.max(target_qs, axis=1, keepdims=True)
                if exp.terminal: #done == True
                   y_val = exp.reward
                else:
                   y_val = exp.reward + discount_rate * max_next_q_values

                print('\n\nGame ', i)
                print('Step ', game_step)
                print('y_val')
                print (y_val)
                print ('\ntarget_qs')
                print (target_qs)
                print('\nmax_next_qs')
                print(max_next_q_values)
                print ('\n\n')

                outs,out_loss,_ = train_sess.run([q,loss,train_opt],feed_dict={max_reward:max_next_q_values, X_action: [[exp.action]],online_input:exp.first_state,target_input:exp.first_state})

                if action_verbose:
                    print ('Mean output loss at terminal step %d: %f'%(game_step,np.mean(out_loss)))

            if i%update_interval == 0:
                train_sess.run(copy_online_to_target)
                print ('Online to target copy complete after game %d'%i)

            if i % graph_interval == 0 and i>1:
                f, (ax1, ax2, ax3) = plt.subplots(3)
                xs = np.arange(0, len(agg_rewards))
                ax1.set_title('Aggregate Reward Update at Game %d' %i)
                ax2.set_title('Step Aggregate Reward Update at Game %d' % i)
                ax3.set_title('Game Steps Used Update at Game %d' % i)
                ax1.plot(xs, agg_rewards, 'r', label='Agg Rewards')
                ax2.plot(xs, step_agg_rewards, 'b', label='Step Agg Rewards')
                ax3.plot(xs, game_steps, 'y', label='Game Steps')
                ax1.set_xlabel('Game Played')
                ax1.set_ylabel('Agg Reward')
                ax2.set_xlabel('Game Played')
                ax2.set_ylabel('Step Agg Reward')
                ax3.set_xlabel('Game Played')
                ax3.set_ylabel('Game Steps Used')
                f.tight_layout()
                plt.savefig('Metrics Measurement.png')
                plt.show(block=False)
                time.sleep(15)
                plt.close()
        self.saver.save(train_sess,self.ckptdir)
        print ('Model saved at %s'%self.ckptdir)






    #sample frames to test other functionalities
    def fake_images(self,batch):
        return 255*np.random.random_sample([batch,210,160,3])-255

if __name__ == "__main__":
    actor = PacWoman(ckpt='rl_research/mspacman/',lr=0.001,buffer_size=20000) #try changing buffer size
    # Initialize target and online networks
    actor.q_network(name_scope='online')
    actor.q_network(name_scope='target')
    print ('Before train is called: ')
    actor.print_train_vars()
    print ('Now train :')
    actor.train(name_scope='online',action_verbose=True,update_interval=50)
