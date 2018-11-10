import sonic as env
import tensorflow as tf
import gym
from runners import AbstractEnvRunner
import time
import numpy as np
from baselines import logger
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common import explained_variance
from A2C_model import A2C_model, A2CNetwork
from A2C_agent import A2CAgent


class Runner(AbstractEnvRunner):
    """
    We use this object to make a mini batch of experiences
    __init__:
    - Initialize the runner
    run():
    - Make a mini batch
    """
    def __init__(self, env, model, nsteps, total_timesteps, gamma, lam):
        super().__init__(env = env, model = model, nsteps = nsteps)

        # Discount rate
        self.gamma = gamma

        # Lambda used in GAE (General Advantage Estimation)
        self.lam = lam

        # Total timesteps taken
        self.total_timesteps = total_timesteps

    def run(self):
        # Here, we init the lists that will contain the mb of experiences
        mb_obs, mb_actions, mb_rewards, mb_values, mb_neglopacs, mb_dones = [],[],[],[],[],[]
        
        # For n in range number of steps
        for n in range(self.nsteps):
            # Given observations, get action value and neglopacs
            # We already have self.obs because AbstractEnvRunner run self.obs[:] = env.reset()
            actions, values, neglopacs = self.model.eval_step(self.obs, self.dones)
            # print(self.obs.shape)
            # print(values.shape)
            assert (values.shape == (self.obs.shape[0],1))
            #print(actions.shape,values.shape, self.obs.shape)
            # Append the observations into the mb
            mb_obs.append(np.copy(self.obs)) #obs len nenvs (1 step per env)

            # Append the actions taken into the mb
            mb_actions.append(actions)

            # Append the values calculated into the mb
            mb_values.append(values)

            # Append the negative log probability into the mb
            mb_neglopacs.append(neglopacs)

            # Append the dones situations into the mb
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            # Infos contains a ton of useful informations
            # {'level_end_bonus': 0, 'rings': 0, 'score': 0, 'zone': 1, 'act': 0, 'screen_x_end': 6591, 'screen_y': 12, 'lives': 3, 'x': 96, 'y': 108, 'screen_x': 0}
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)

            mb_rewards.append(rewards)

        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglopacs = np.asarray(mb_neglopacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs)
        
        # print("obs",mb_obs.shape)
        # print("rewards",mb_rewards.shape)
        # print("actions",mb_actions.shape)
        # print("values",mb_values.shape)
        # print("neglopacs",mb_neglopacs.shape)
        # print("dones",mb_dones.shape)
        # print("last_values", last_values.shape)
        ### GENERALIZED ADVANTAGE ESTIMATION
        # discount/bootstrap off value fn
        # We create mb_returns and mb_advantages
        # mb_returns will contain Advantage + value
        mb_returns = np.zeros_like(mb_rewards)
        mb_advantages = np.zeros_like(mb_rewards)
        # print("advantage",mb_advantages.shape)
        lastgaelam = 0

        # From last step to first step
        for t in reversed(range(self.nsteps)):
            # If t == before last step
            if t == self.nsteps - 1:
                # If a state is done, nextnonterminal = 0
                # In fact nextnonterminal allows us to do that logic

                #if done (so nextnonterminal = 0):
                #    delta = R - V(s) (because self.gamma * nextvalues * nextnonterminal = 0)
                # else (not done)
                    #delta = R + gamma * V(st+1)
                nextnonterminal = 1.0 - self.dones

                # V(t+1)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - np.reshape(mb_dones[t+1],(-1,))

                nextvalues = mb_values[t+1]

            # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
            # print("Pre-Delta",mb_rewards[t].shape, mb_values[t].shape,nextvalues.shape)
            # print(mb_rewards[t],mb_values[t],nextvalues)
            # print("Pre-Delt2",self.gamma, nextnonterminal)
            assert (mb_rewards[t].shape == (self.obs.shape[0],))
            assert (mb_values[t].shape == (self.obs.shape[0],1))
            delta = mb_rewards[t] + (self.gamma * np.squeeze(nextvalues) * nextnonterminal) - np.squeeze(mb_values[t])
            
            # print("delta",delta.shape)
            # Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
            mb_advantages[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

        # Returns
        mb_returns = mb_advantages + np.squeeze(mb_values)

        return map(sf01, (mb_obs, mb_actions, mb_returns, mb_values, mb_neglopacs))


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def constfn(val):
    def f(_):
        return val
    return f


def learn(  sess,
            agent_init=A2CAgent,
            model_init=A2CModel,
            network_init=A2CNetwork,
            env,
            nsteps,
            total_timesteps,
            gamma,
            lam,
            vf_coef,
            ent_coef,
            lr,
            cliprange,
            max_grad_norm,
            log_interval,
            images):

    noptepochs = 4
    nminibatches = 8

    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)

    # Get the nb of env
    nenvs = env.num_envs

    # Get state_space and action_space
    ob_space = env.observation_space
    ac_space = env.action_space

    # Calculate the batch_size
    batch_size = nenvs * nsteps # For instance if we take 5 steps and we have 5 environments batch_size = 25

    batch_train_size = batch_size // nminibatches

    assert batch_size % nminibatches == 0

    # Instantiate the model object (that creates step_model and train_model)
    model = model(sess = sess,
                policy=network,
                ob_space=ob_space,
                action_space=ac_space,
                nenvs=nenvs,
                nsteps=nsteps,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                images = images)

    agent = agent(model)

    # Load the model
    # If you want to continue training
    # load_path = "./models/10/model.ckpt"
    # model.load(load_path)

    # Instantiate the runner object
    runner = Runner(env, agent, nsteps=nsteps, total_timesteps=total_timesteps, gamma=gamma, lam=lam)

    # Start total timer
    tfirststart = time.time()

    nupdates = total_timesteps//batch_size+1

    for update in range(1, nupdates+1):
        # Start timer
        tstart = time.time()

        frac = 1.0 - (update - 1.0) / nupdates

        # Calculate the learning rate
        lrnow = lr(frac)

        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        # Get minibatch
        obs, actions, returns, values, neglogpacs = runner.run()
        
        # print("returns",returns.shape,"\nvalues",values.shape)
        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mb_losses = []
        total_batches_train = 0

        # Index of each element of batch_size
        # Create the indices array
        indices = np.arange(batch_size)

        for _ in range(noptepochs):
            # Randomize the indexes
            np.random.shuffle(indices)

            # 0 to batch_size with batch_train_size step
            for start in range(0, batch_size, batch_train_size):
                end = start + batch_train_size
                mbinds = indices[start:end]
                slices = (arr[mbinds] for arr in (obs, actions, returns, values, neglogpacs))
                mb_losses.append(agent.train_step(*slices, lrnow, cliprangenow))


        # Feedforward --> get losses --> update
        lossvalues = np.mean(mb_losses, axis=0)

        # End timer
        tnow = time.time()

        # Calculate the fps (frame per second)
        fps = int(batch_size / (tnow - tstart))

        if update % log_interval == 0 or update == 1:
        #     """
        #     Computes fraction of variance that ypred explains about y.
        #     Returns 1 - Var[y-ypred] / Var[y]
        #     interpretation:
        #     ev=0  =>  might as well have predicted zero
        #     ev=1  =>  perfect prediction
        #     ev<0  =>  worse than just predicting zero
        #     """
            ev = explained_variance(np.squeeze(values), returns)
            logger.record_tabular("serial_timesteps", update*nsteps)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(lossvalues[0]))
            logger.record_tabular("policy_entropy", float(lossvalues[2]))
            logger.record_tabular("value_loss", float(lossvalues[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time elapsed", float(tnow - tfirststart))

            savepath = "./models/" + "Car"  + str(update) + "/model.ckpt"
            model.save(savepath)
            print('Saving to', savepath)

            # Test our agent with 3 trials and mean the score
            # This will be useful to see if our agent is improving
            # test_score = testing(model)

            # logger.record_tabular("Mean score test level", test_score)
            logger.dump_tabular()

    env.close()


def main():
    config = tf.ConfigProto()

    # Avoid warning message errors
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    # Allowing GPU memory growth
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True

    #envs = SubprocVecEnv([env.make_train_0,env.make_train_1,env.make_train_2,env.make_train_3,env.make_train_4,env.make_train_5,env.make_train_6,env.make_train_7,env.make_train_8,env.make_train_9,env.make_train_10,env.make_train_11,env.make_train_12,env.make_train_13,env.make_train_14,env.make_train_15])
    envs = SubprocVecEnv([env.make_train_16])

    with tf.Session(config=config) as sess:
       learn(agent_init=A2CAgent,
              model_init=A2CModel,
              network_init=A2CNetwork,
              env=envs,
              nsteps=2048, # Steps per environment
              total_timesteps=10000000,
              gamma=0.9,
              lam = 0.95,
              vf_coef=0.5,
              ent_coef=0.01,
              lr = lambda _: 1e-4,
              cliprange = lambda _: 0.1, # 0.1 * learning_rate
              max_grad_norm = 0.5,
              log_interval = 10,
              sess = sess,
              images= False)


if __name__ == '__main__':
    main()