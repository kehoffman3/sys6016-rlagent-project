import copy
from typing import Sequence, Tuple

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import numpy as np
import sonnet as snt
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
import dm_memorytasks
import numpy as np
from datetime import datetime
import pandas as pd
import copy
import pygame

FLAGS = flags.FLAGS

_FRAMES_PER_SECOND = 30

flags.DEFINE_list(
    'screen_size', [2048, 1024],
    'Screen width/height in pixels. Scales the environment RGB observations to '
    'fit the screen size.')

flags.DEFINE_string(
    'docker_image_name', None,
    'Name of the Docker image that contains the Memory Tasks. '
    'If None, uses the default dm_memorytask name')

flags.DEFINE_boolean(
    'silent', False,
    'Whether or not the graphical interface should be displayed.')

flags.DEFINE_integer('seed', 123, 'Environment seed.')
flags.DEFINE_string('level_name', 'visible_goal_with_buildings_train',
                    'Name of memory task to run.')
flags.DEFINE_integer('episode_length_seconds', '360', 'Length of episode')

# Tuning parameters
flags.DEFINE_float('learning_rate', .0001, 'Learning rate of the network')
flags.DEFINE_float('epsilon', .1, 'Epsilon of the network')
flags.DEFINE_float('discount', .99, 'Discount of the network')

from typing import Sequence, Tuple
import tree
import trfl
import tensorflow_probability as tfp

tfd = tfp.distributions

# Action structure {'MOVE_BACK_FORWARD': 0, 'STRAFE_LEFT_RIGHT': 0, 'LOOK_LEFT_RIGHT': 0, 'LOOK_DOWN_UP': 0} (ranges from -1, 1)
# Define the 8 possible actions
# Ignore look up and look down since they dont help accomplish the task
_ACTIONS = [
    {'MOVE_BACK_FORWARD': 1},
    {'MOVE_BACK_FORWARD': -1},
    {'STRAFE_LEFT_RIGHT': -1},
    {'STRAFE_LEFT_RIGHT': 1},
    {'LOOK_LEFT_RIGHT': -1},
    {'LOOK_LEFT_RIGHT': 1},
    #{'LOOK_DOWN_UP': -1},
    #{'LOOK_DOWN_UP': 1},
]

# Define default no action
_NO_ACTION = {
    'MOVE_BACK_FORWARD': 0,
    'STRAFE_LEFT_RIGHT': 0,
    'LOOK_LEFT_RIGHT': 0,
    'LOOK_DOWN_UP': 0,
}


class DQN(base.Agent):
  """A simple DQN agent using TF2."""

  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      network: snt.Module,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: snt.Optimizer,
      epsilon: float,
      seed: int = None,
  ):

    # Internalise hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon = epsilon
    self._min_replay_size = min_replay_size

    # Seed the RNG.
    tf.random.set_seed(seed)
    self._rng = np.random.RandomState(seed)

    # Internalise the components (networks, optimizer, replay buffer).
    self._optimizer = optimizer
    self._replay = replay.Replay(capacity=replay_capacity)
    self._online_network = network
    self._target_network = copy.deepcopy(network)
    self._forward = tf.function(network)
    self._total_steps = tf.Variable(0)

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:
    # Epsilon-greedy policy.
    if self._rng.rand() < self._epsilon:
      return self._rng.randint(self._num_actions)

    observation = tf.convert_to_tensor(timestep.observation['RGB_INTERLEAVED'][None, ...].astype('float32'))
    # Greedy policy, breaking ties uniformly at random.
    q_values = self._forward(observation).numpy()
    #print(np.flatnonzero(q_values == q_values.max()))
    action = self._rng.choice(np.flatnonzero(q_values == q_values.max()))
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    # Add this transition to replay.
    self._replay.add([
        timestep.observation['RGB_INTERLEAVED'].astype('float32'),
        action,
        new_timestep.reward,
        new_timestep.discount,
        new_timestep.observation['RGB_INTERLEAVED'].astype('float32'),
    ])

    self._total_steps.assign_add(1)
    if tf.math.mod(self._total_steps, self._sgd_period) != 0:
      return

    if self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD.
    transitions = self._replay.sample(self._batch_size)
    self._training_step(transitions)

  @tf.function
  def _training_step(self, transitions: Sequence[tf.Tensor]) -> tf.Tensor:
    """Does a step of SGD on a batch of transitions."""
    o_tm1, a_tm1, r_t, d_t, o_t = transitions
    r_t = tf.cast(r_t, tf.float32)  # [B]
    d_t = tf.cast(d_t, tf.float32)  # [B]
    o_tm1 = tf.convert_to_tensor(o_tm1)
    o_t = tf.convert_to_tensor(o_t)

    with tf.GradientTape() as tape:
      q_tm1 = self._online_network(o_tm1)  # [B, A]
      q_t = self._target_network(o_t)  # [B, A]

      onehot_actions = tf.one_hot(a_tm1, depth=self._num_actions)  # [B, A]
      qa_tm1 = tf.reduce_sum(q_tm1 * onehot_actions, axis=-1)  # [B]
      qa_t = tf.reduce_max(q_t, axis=-1)  # [B]

      # One-step Q-learning loss.
      target = r_t + d_t * self._discount * qa_t
      td_error = qa_tm1 - target
      loss = 0.5 * tf.reduce_mean(td_error**2)  # []

    # Update the online network via SGD.
    variables = self._online_network.trainable_variables
    gradients = tape.gradient(loss, variables)
    self._optimizer.apply(gradients, variables)

    # Periodically copy online -> target network variables.
    if tf.math.mod(self._total_steps, self._target_update_period) == 0:
      for target, param in zip(self._target_network.trainable_variables,
                               self._online_network.trainable_variables):
        target.assign(param)
    return loss


# Default parameters
# def default_agent(obs_spec: specs.Array,
#                   action_spec: specs.DiscreteArray):
#   """Initialize a DQN agent with default parameters."""
#   del obs_spec  # Unused.
#   network = snt.Sequential([
#       snt.Flatten(),
#       snt.nets.MLP([50, 50, action_spec.num_values]),
#   ])
#   optimizer = snt.optimizers.Adam(learning_rate=1e-3)
#   return DQN(
#       action_spec=action_spec,
#       network=network,
#       batch_size=32,
#       discount=0.99,
#       replay_capacity=10000,
#       min_replay_size=100,
#       sgd_period=1,
#       target_update_period=4,
#       optimizer=optimizer,
#       epsilon=0.05,
#       seed=42)

def main(argv):
  if not FLAGS.silent:
    # Init pygame so we can watch the agent  
    pygame.init()
    try:
      pygame.mixer.quit()
    except NotImplementedError:
      pass
    pygame.display.set_caption('Memory Tasks DQN Agent')

  env_settings = dm_memorytasks.EnvironmentSettings(
      seed=FLAGS.seed, level_name=FLAGS.level_name, episode_length_seconds=FLAGS.episode_length_seconds)

  with dm_memorytasks.load_from_docker(
      name=FLAGS.docker_image_name, settings=env_settings) as env:

    if not FLAGS.silent:
      # Pygame setup
      screen = pygame.display.set_mode(
        (int(FLAGS.screen_size[0]), int(FLAGS.screen_size[1])))
      rgb_spec = env.observation_spec()['RGB_INTERLEAVED']
      surface = pygame.Surface((rgb_spec.shape[1], rgb_spec.shape[0]))
      clock = pygame.time.Clock()

    # Manually define the action spec
    action_spec = specs.DiscreteArray(
        dtype=int, num_values=len(_ACTIONS), name="action")

    network = snt.Sequential([
      snt.Flatten(),
      snt.nets.MLP([50, 50, action_spec.num_values]),
    ])
  
    # Set params of our agent
    optimizer = snt.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    agent = DQN(
      action_spec=action_spec,
      network=network,
      batch_size=64,
      discount=FLAGS.discount,
      replay_capacity=10000,
      min_replay_size=100,
      sgd_period=2,
      target_update_period=6,
      optimizer=optimizer,
      epsilon=FLAGS.epsilon,
      seed=50)
    # Observation: OrderedDict([('Score', Array(shape=(), dtype=dtype('float64'), name='Score')), ('RGB_INTERLEAVED', Array(shape=(72, 96, 3), dtype=dtype('uint8'), name='RGB_INTERLEAVED')), ('AvatarPosition', Array(shape=(3,), dtype=dtype('float32'), name='AvatarPosition'))])

    timestep = env.reset()
    score = 0
    action_tracker = {}
    result_dict = {"timestamp": [], "score": [], "reward":[]}
    while not timestep.last():
      # Returns the index of the traction  
      selected_action_idx = agent.select_action(timestep)

      # Track the number of each action taken
      action_ref = selected_action_idx
      if action_ref not in action_tracker:
          action_tracker[action_ref] = 1
      else:
          action_tracker[action_ref] += 1

      # Get the key/value of the action by index
      selected_action = _ACTIONS[selected_action_idx]

      # Make a deep copy of the initial array
      action = copy.deepcopy(_NO_ACTION)

      # Get the name of the action and set the value in the array
      selected_action_name = list(selected_action.keys())[0]
      action[selected_action_name] = selected_action[selected_action_name]
      
      # Give the actor's action to the envrionment
      timestep = env.step(action)

      if not FLAGS.silent:
        # Update the view based on the action
        frame = np.swapaxes(timestep.observation['RGB_INTERLEAVED'], 0, 1)
        pygame.surfarray.blit_array(surface, frame)
        pygame.transform.smoothscale(surface, screen.get_size(), screen)
        pygame.display.update()
        clock.tick(_FRAMES_PER_SECOND)
        
      if timestep.reward:
        score += timestep.reward
        logging.info('Total score: %1.1f, reward: %1.1f', score,
                     timestep.reward)
        result_dict["timestamp"].append(datetime.now())
        result_dict["score"].append(score)
        result_dict["reward"].append(timestep.reward)
      
    logging.info('Final score: %1.1f', score)
    logging.info(f'Actions taken (key is the index of _ACTION variable): {action_tracker}')
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(f"dqn_agent_results_{datetime.now().timestamp()}.csv")

if __name__ == '__main__':
  app.run(main)