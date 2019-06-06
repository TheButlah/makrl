"""This file is was originally inspired by the code here:
https://github.com/MG2033/A2C/blob/master/envs/subproc_vec_env.py. It has since
evolved heavily, but we provide the original link as an acknowledgement."""

# Python 2-3 Compatibility
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from six.moves import range, zip

import sys
import collections
import multiprocessing
import numpy as np

from multiprocessing import Process, Pipe
from logzero import logger


def num_cores():
  """Returns the number of CPU cores detected."""
  return multiprocessing.cpu_count()


def worker(parent_end, worker_end, seed, env_fn_wrapper):
  """Entry point for the worker subprocess.

  Args:
    parent_end:  The parent's end of the pipe.
    worker_end:  The worker's end of the pipe.
    seed:  A positive integer to use as the seed
    env_fn_wrapper:  A `CloudpickleWrapper` for a function to make the env.
  """
  pipe = worker_end
  del worker_end

  # Child processes inherit parent file descriptors, regardless of whether we
  # explicitly pass the parent end of the pipe or not. By passing the parent's
  # end, we can close our handle on the file descriptor, meaning that once the
  # parent does the same, we can detect that the pipe has been closed.
  parent_end.close()  # close in order to detect when parent closes pipe
  del parent_end

  env = env_fn_wrapper.obj()
  env.seed(seed)

  try:
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
  except Exception as e:
    # We need to avoid throwing exceptions as the printout is not atomic
    # TODO: Figure out how to make this atomic
    sys.stderr.write(str(e))
    sys.stderr.flush()
  finally:
    pipe.close()  # This is closed on gc, so it's only here for explicitness


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
  """Class to synchronously batch together multiple environments. Exposes an
  OpenAI Gym style API to synchronously interact with environments, taking
  advantage of multithreading where applicable."""

  # TODO: Figure out how to deal with errors in subprocesses
  def __init__(self, env_fns):
    """env_fns: list of functions that construct the environments to batch
    together. Each environment should have a gym style api, compatible via duck
    typing.

    NOTE: Action/observation spaces must be identical for all environments."""
    logger.debug("Constructing `BatchedEnv`")
    self._num_envs = len(env_fns)
    # The two ends of each pipe, one pipe per env. `worker_ends` given to
    # the subprocesses to communicate with parent, `parent_ends` given to
    # parent to communicate with workers
    self._parent_ends, self._worker_ends = zip(
      *[Pipe() for _ in range(self.num_envs)])

    # Build the subprocesses
    self._ps = [
      Process(target=worker, args=(
        parent_end,
        worker_end,
        np.random.randint(2**31),
        CloudpickleWrapper(env_fn)))
      for (parent_end, worker_end, env_fn)
      in zip(self._parent_ends, self._worker_ends, env_fns)]

    # Start the subprocesses
    logger.debug("Starting subprocesses...")
    for p, worker_end in zip(self._ps, self._worker_ends):
      # Causes parent to kill children on exit. NOT the same as unix daemon.
      # Note that in the case of SIGTERM
      p.daemon = True
      p.start()
      # Once worker end is given to worker process, close the parent's copy
      worker_end.close()

    # Identify the action/obs spaces.
    self._parent_ends[0].send(('get_spaces', None))
    self._action_space, self._observation_space = self._parent_ends[0].recv()
    self._step_counters = np.zeros(self.num_envs, dtype=np.int16)

  def step(self, actions):
    """Steps the environments based on the given batch of actions.

    Returns:
      (states, rewards, dones, infos)
    """
    for pipe, action in zip(self._parent_ends, actions):
      pipe.send(('step', action))
    results = [pipe.recv() for pipe in self._parent_ends]
    states, rewards, dones, infos = zip(*results)
    self._step_counters += 1
    return np.stack(states), np.stack(rewards), np.stack(dones), infos

  def reset(self, mask=None):
    """Resets some or all environments.

    Args:
      mask:  If `None`, reset all envs. Otherwise, should be a boolean mask
        where `True` indicates the environments to reset.

    Returns:
      A ndarray of shape `(np.count_nonzero(mask),) + observation_space` if
      any envs were marked for reset, otherwise `None`.
    """
    if mask is not None:
      num_reset = np.count_nonzero(mask)
      if num_reset == 0:
        return None

      logger.debug("Resetting %d envs" % num_reset)
      self._step_counters[mask] = 0

      [pipe.send(('reset', None))
       for pipe, done in zip(self._parent_ends, mask) if done]

      return np.stack([pipe.recv()
                       for pipe, done in zip(self._parent_ends, mask)
                       if done])
    else:
      logger.debug("Resetting all envs")
      self._step_counters.fill(0)
      [pipe.send(('reset', None)) for pipe in self._parent_ends]
      return np.stack([pipe.recv() for pipe in self._parent_ends])

  def close(self):
    """Destroy the batch of environments and their processes."""
    logger.debug("Killing environments...")
    for pipe in self._parent_ends:
      pipe.send(('close', None))
    for p, parent_end in zip(self._ps, self._parent_ends):
      p.join()
      parent_end.close()  # This is gc'ed on exit, but its here for explicitness

    self._parent_ends = self._worker_ends = self._ps = None
    self._action_space = self._observation_space = self._num_envs = None
    self._step_counters = None

  def monitor(self, is_monitor=True, is_train=True, experiment_dir="",
              record_video_every=10):
    # TODO: Figure out how this works
    for pipe in self._parent_ends:
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
      pipes = [self._parent_ends[i] for i in env_ids]
    elif env_ids is None:
      pipes = self._parent_ends
    else:
      raise ValueError("invalid argument for `env_ids`!")
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
      [self._parent_ends[id].send(('seed', s)) for (id, s) in seed_map.items()]
    else:
      [pipe.send(('seed', np.random.randint(2**31)))
       for pipe in self._parent_ends]

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
