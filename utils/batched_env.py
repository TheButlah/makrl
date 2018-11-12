"""This file is a heavily modified version of the code here:
https://github.com/MG2033/A2C/blob/master/envs/subproc_vec_env.py"""

# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

import sys
import collections
import multiprocessing
from multiprocessing import Process, Pipe

import numpy as np


def num_cores():
  """Returns the number of CPU cores detected."""
  return multiprocessing.cpu_count()


def worker(pipe, seed, env_fn_wrapper):
  """Entry point for the worker subprocess"""
  env = env_fn_wrapper.obj()
  env.seed(seed)
  while True:
    cmd, data = pipe.recv()
    if cmd == 'step':
      ob, reward, done, info = env.step(data)
      total_info = info.copy()  # Pass by value instead of reference
      if done:
        ob = env.reset()
      pipe.send((ob, reward, done, total_info))
    elif cmd == 'reset':
      ob = env.reset()
      pipe.send(ob)
    elif cmd == 'close':
      pipe.close()
      break
    elif cmd == 'get_spaces':
      pipe.send((env.action_space, env.observation_space))
    elif cmd == 'monitor':
      is_monitor, is_train, experiment_dir, record_video_every = data
      env.monitor(is_monitor, is_train, experiment_dir, record_video_every)
    elif cmd == 'render':
      env.render()
    elif cmd == 'seed':
      env.seed(data)
    else:
      raise NotImplementedError


class CloudpickleWrapper(object):
    """Uses cloudpickle to serialize contents (otherwise multiprocessing tries
    to use pickle)"""

    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.obj)

    def __setstate__(self, obj):
        import pickle
        self.obj = pickle.loads(obj)


class BatchedEnv(object):
  def __init__(self, env_fns):
    """env_fns: list of functions to construct the envs for the subprocs.
    NOTE: Action/observation spaces must be identical for all envs."""
    self._num_envs = len(env_fns)
    # The two ends of each pipe, one pipe per env. `p_to_parents` given to
    # the subprocesses to communicate with parent, `p_to_workers` given to
    # parent to communicate with workers
    self._p_to_workers, self._p_to_parents = zip(
      *[Pipe() for _ in range(self.num_envs)])

    # Build the subprocesses
    self._ps = [
      Process(target=worker, args=(
        p_to_parent,
        np.random.randint(2**31),
        CloudpickleWrapper(env_fn)))
      for (p_to_parent, env_fn) in zip(self._p_to_parents, env_fns)]

    # Start the subprocesses
    for p in self._ps:
      p.start()

    # Identify the action/obs spaces.
    self._p_to_workers[0].send(('get_spaces', None))
    self._action_space, self._observation_space = self._p_to_workers[0].recv()
    self._step_counters = np.zeros(self.num_envs, dtype=np.int16)

  def step(self, actions):
    """Steps the environments based on the given batch of actions.

    Returns:
      (obs, rewards, dones, infos)
    """
    for pipe, action in zip(self._p_to_workers, actions):
      pipe.send(('step', action))
    results = [pipe.recv() for pipe in self._p_to_workers]
    obs, rews, dones, infos = zip(*results)
    # TODO: what is infos?
    self._step_counters += 1
    return np.stack(obs), np.stack(rews), np.stack(dones), infos

  def reset(self, env_ids=None):
    """Resets some or all environments.

    Args:
      env_ids:  If `None`, reset all envs. Otherwise, `env_ids` should be a
        list of the indices of envs to reset in the batch.

    Returns:
      A ndarray of shape `(len(env_ids),) + observation_space`.
    """
    if env_ids:
      assert (len(env_ids) <= self.num_envs
              and {} == set(env_ids) - set(range(self.num_envs)))
      pipes = [self._p_to_workers[i] for i in env_ids]
      self._step_counters[env_ids] = 0
    else:
      pipes = self._p_to_workers
      self._step_counters.fill(0)

    for pipe in pipes:
      pipe.send(('reset', None))

    return np.stack([pipe.recv() for pipe in pipes])

  def close(self):
    """Destroy the batch of environments and their processes."""
    for pipe in self._p_to_workers:
      pipe.send(('close', None))
    for p in self._ps:
      p.join()

    self._p_to_workers = self._p_to_parents = self._ps = None
    self._action_space = self._observation_space = self._num_envs = None
    self._step_counters = None

  def monitor(self, is_monitor=True, is_train=True, experiment_dir="",
              record_video_every=10):
    # TODO: Figure out how this works
    for pipe in self._p_to_workers:
      pipe.send((
        'monitor',
        (is_monitor, is_train, experiment_dir, record_video_every)))

  def render(self, env_ids=None):
    """Render some or all environments.

    Args:
      env_ids:  If `None`, render all envs. Otherwise, `env_ids` should be a
        list of the indices of envs to render in the batch.
    """
    if env_ids:
      pipes = [self._p_to_workers[i] for i in env_ids]
    else:
      pipes = self._p_to_workers
    for pipe in pipes:
      pipe.send(('render', None))

  def seed(self, seed_map=None):
    """Seeds some or all environments on a given seed or a random seed.

    Args:
      seed_map:  If `None`, sets all seeds to a new random seed. Otherwise,
      should be a dict mapping from env ids to seeds. The seeds should be either
      `None` to indicate a new random seed, or an int.
    """
    if seed_map:
      [self._p_to_workers[id].send(('seed', s)) for (id, s) in seed_map.items()]
    else:
      [pipe.send(('seed', np.random.randint(2**31)))
       for pipe in self._p_to_workers]

  @property
  def num_envs(self):
    return self._num_envs

  @property
  def action_space(self):
      return self._action_space

  @property
  def observation_space(self):
    return self._observation_space

  @property
  def step_counters(self):
    """A list of ints representing the current step number for every env."""
    return self._step_counters
