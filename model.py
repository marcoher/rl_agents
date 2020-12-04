import tensorflow as tf
import numpy as np
import random
import os
import pickle

from tensorflow.keras import layers
from collections import deque


class Rescaling(tf.keras.layers.Layer):
  """ A simple layer for re-scaling data inside the model"""
  def __init__(self, scale=None, center=None, **kwargs):
    super(Rescaling, self).__init__(**kwargs)
    self.scale = scale
    self.center = center
    if self.scale is not None:
      self.tf_scale = tf.Variable(initial_value=scale,
                                  trainable=False)
    if self.center is not None:
      self.tf_center = tf.Variable(initial_value=center,
                                   trainable=False)

  def build(self, input_shape):
    if self.scale is None:
      self.tf_scale = self.add_weight(
          shape=(1, input_shape[-1]), initializer="ones",
          trainable=False
      )
    if self.center is None:
      self.tf_center = self.add_weight(
          shape=(1, input_shape[-1]), initializer="zeros",
          trainable=False)

  def call(self, inputs):
    return (inputs - self.tf_center) * self.tf_scale

  def get_config(self):
    config = super(Rescaling, self).get_config()
    config.update({"scale": self.scale, "center": self.center})
    return config


class Agent:
  """ TD3 algorithm """
  def __init__(self, observation_space, action_space,
               policy_units=(50, 50),
               critic_units=(50, 50),
               optimizer="adam",
               replay_memory=2000,
               discount=0.99,
               policy_freq=5,
               policy_noise=0.2,
               noise_clip=0.05,
               expl_noise=0.1,
               tau=0.005,
               load_path=None):
    """
    Implements the TD3 algorithm introduced in
    'Addressing Function Approximation Error in Actor-Critic Methods'
    (Fujimoto et al, 2018)
    """

    # observation space
    self._observation_space_dim = observation_space.shape[0]
    self._observation_space_high = observation_space.high.astype(np.float32)
    self._observation_space_low = observation_space.low.astype(np.float32)
    # action space
    self._action_space_dim = action_space.shape[0]
    self._action_space_high = action_space.high.astype(np.float32)
    self._action_space_low = action_space.low.astype(np.float32)
    # architechture parameters
    self.policy_units = policy_units
    self.critic_units = critic_units
    # algorithm parameters
    self.discount = discount
    self.policy_noise = policy_noise
    self.policy_freq = policy_freq
    self.tau = tau
    self.noise_clip = noise_clip
    self.expl_noise = expl_noise
    self._optimizer = optimizer
    # main attributes
    self.model_dict = dict()
    self.replay_buffer = deque(maxlen=replay_memory)
    self.critic_1 = self._build_compiled_q_network()
    self.critic_1._name = "critic_1"
    self.model_dict["critic_1"] = self.critic_1

    self.critic_2 = self._build_compiled_q_network()
    self.critic_2._name = "critic_2"
    self.model_dict["critic_2"] = self.critic_2
    self.actor = self._build_compiled_policy()
    self.actor._name = "actor"
    self.model_dict["actor"] = self.actor
    # targrt networks
    self.target_critic_1 = self._build_compiled_q_network()
    self.target_critic_1._name = "target_critic_1"
    self.model_dict["target_critic_1"] = self.target_critic_1
    self.target_critic_2 = self._build_compiled_q_network()
    self.target_critic_2._name = "target_critic_2"
    self.model_dict["target_critic_2"] = self.target_critic_2
    self.target_actor = self._build_compiled_policy()
    self.target_actor._name = "target_actor"
    self.model_dict["target_actor"] = self.target_actor
    self.align_all_models()
    if load_path is not None:
      self.load(load_path)

    # self.policy_optimizer = tf.keras.optimizers.Adam()
  def save(self, save_path=None):
    # where to save
    if save_path is None:
      save_path = './checkpoints'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    # save one by one
    for model in self.model_dict.keys():
      pickle.dump(self.model_dict[model].get_weights(),
                  open(save_path + "/" + model + ".p", "wb"))

  def load(self, load_path):
    for model in self.model_dict.keys():
      self.model_dict[model].set_weights(
          pickle.load(open(load_path + "/" + model + ".p", "rb")))

  def store(self, state, action, reward, next_state, done):
    self.replay_buffer.append((state, action, reward, next_state, done))

  def _build_compiled_policy(self):
    """ builds a policy approximator: pi: State -> Action """
    # policy network acts on states, so one input needed only
    # it takes values in action space
    # model = Sequential()
    state = tf.keras.Input(shape=self._observation_space_dim, name="state")
    # rescale input from observation space box to [-1, 1]^space_dim
    center_in = (self._observation_space_high + self._observation_space_low) / 2.
    scale_in = 2. / (self._observation_space_high - self._observation_space_low)
    output = Rescaling(scale_in, center_in, name="state_rescaling")(state)
    # DNN
    for i, num_units in enumerate(self.policy_units):
      output = layers.Dense(num_units, activation='relu',
                            name="dense_layer" + str(i))(output)

    # output = layers.Dense(50, activation='relu',
    #                       name="dense_layer1")(output)
    # output = layers.Dense(50, activation='relu',
    #                       name="dense_layer_2")(output)
    output = layers.Dense(self._action_space_dim, activation='softsign',
                          name="dense_layer_3")(output)
    # re-scale output from [-1,1]^action_dim to the action space box
    center_out = (self._action_space_high + self._action_space_low) / 2.
    scale_out = (self._action_space_high - self._action_space_low) / 2.
    output = Rescaling(scale_out, center_out, name="action")(output)
    model = tf.keras.models.Model(inputs=state, outputs=output)
    model.compile(loss="mse", optimizer=self._optimizer)
    return model

  def _build_compiled_q_network(self):
    """ builds a q approximator: Q: State x Action -> R """
    # q_network has two inputs: state and action
    state = tf.keras.Input(shape=self._observation_space_dim,
                           name="state")
    action = tf.keras.Input(shape=self._action_space_dim,
                            name="action")
    # rescale state input to [-1, 1]^space_dim
    center_1 = (self._observation_space_high + self._observation_space_low) / 2.
    scale_1 = 2. / (self._observation_space_high - self._observation_space_low)
    state_scaled = Rescaling(scale_1, center_1, name="state_rescaling")(state)
    # rescale action input to [-1, 1]^action_dim
    center_2 = (self._action_space_high + self._action_space_low) / 2.
    scale_2 = 2. / (self._action_space_high - self._action_space_low)
    action_scaled = Rescaling(scale_2, center_2,
                              name="action_rescaling")(action)
    # DNN
    output = layers.Concatenate(name="combined_input")(
        [state_scaled, action_scaled])
    for i, num_units in enumerate(self.critic_units):
      output = layers.Dense(num_units, activation='relu',
                            name="dense_layer" + str(i))(output)
    # output = layers.Dense(50, activation='relu', name="dense_layer1")(output)
    # output = layers.Dense(50, activation='relu', name="dense_layer_2")(output)
    output = layers.Dense(1, activation='linear', name="q_value")(output)
    # set optimizer
    model = tf.keras.models.Model(inputs=[state, action], outputs=output)
    model.compile(loss='mse', optimizer=self._optimizer)
    return model

  def align_models(self, model_from, model_to):
    """ aligns model_to (e.g. a target/slowly updated model)
    with the parameters of model_from (e.g. a model updated every step) """
    model_to.set_weights(model_from.get_weights())

  def align_all_models(self):
    self.align_models(self.critic_1, self.target_critic_1)
    self.align_models(self.critic_2, self.target_critic_2)
    self.align_models(self.actor, self.target_actor)

  def clip_action(self, action):
    """ truncates to action space (box) """
    action = np.maximum(action, self._action_space_low)
    action = np.minimum(action, self._action_space_high)
    return action

  def replay(self, batch_size, update=False):
    minibatch = random.sample(self.replay_buffer, batch_size)
    # stack into arrays for vectorized computation
    states = np.concatenate([b[0] for b in minibatch])
    actions = np.concatenate([b[1] for b in minibatch])
    rewards = np.array([b[2] for b in minibatch])
    rewards = rewards.reshape((-1, 1)).astype(np.float32)
    next_states = np.concatenate([b[3] for b in minibatch])
    target_actions = self.target_actor(next_states)
    regularizing_noise = tf.random.truncated_normal(
        shape=target_actions.shape, stddev=self.policy_noise)
    regularizing_noise = tf.clip_by_value(
        regularizing_noise,
        clip_value_min=-self.noise_clip,
        clip_value_max=self.noise_clip)
    target_actions = tf.clip_by_value(
        target_actions + regularizing_noise,
        clip_value_min=self._action_space_low,
        clip_value_max=self._action_space_high).numpy()
    # target value:
    # y = r + gamma * min_i Q'_i(s_next, a_next)
    y = rewards + self.discount * np.minimum(
        self.target_critic_1([next_states, target_actions]).numpy(),
        self.target_critic_2([next_states, target_actions]).numpy())
    # fit (without further minibatching the minibatch)
    self.critic_1.fit(x=[states, actions], y=y,
                      batch_size=batch_size, epochs=1, verbose=0)
    self.critic_2.fit(x=[states, actions], y=y,
                      batch_size=batch_size, epochs=1, verbose=0)
    # if model needs to be updated
    if update:
      self.update(states)

  def update(self, states):
    """ updates the policy and target networks """
    # set optimizer
    def loss_fn():
      return -tf.reduce_mean(self.critic_1([states, self.actor(states)]))
    policy_optimizer = tf.keras.optimizers.Adam()
    policy_optimizer.minimize(loss_fn,
                              var_list=self.actor.trainable_variables)
    self.target_actor.set_weights(
        [(1. - self.tau) * w_cur + self.tau * w_new
         for (w_cur, w_new) in zip(self.target_actor.get_weights(),
                                   self.actor.get_weights())])
    self.target_critic_1.set_weights(
        [(1. - self.tau) * w_cur + self.tau * w_new
         for (w_cur, w_new) in zip(self.target_critic_1.get_weights(),
                                   self.critic_1.get_weights())])
    self.target_critic_2.set_weights(
        [(1. - self.tau) * w_cur + self.tau * w_new
         for (w_cur, w_new) in zip(self.target_critic_2.get_weights(),
                                   self.critic_2.get_weights())])

  def play_and_learn(self, environment, minibatch_size=100, max_steps=10000):
    # Reset the enviroment
    state = environment.reset()
    state = np.reshape(state, [1, -1]).astype(np.float32)
    actions_taken = 0
    terminated = False
    while actions_taken < max_steps and not terminated:
        # get action with exploration noise
      actions_taken += 1
      action = self.actor(state)
      exploration_noise = tf.random.normal(
          shape=action.shape,
          stddev=self.policy_noise)
      action = tf.clip_by_value(action + exploration_noise,
                                clip_value_min=self._action_space_low,
                                clip_value_max=self._action_space_high).numpy()
    # Take action
      next_state, reward, terminated, info = environment.step(action)
      next_state = np.reshape(next_state, [1, -1]).astype(np.float32)
      self.store(state, action, reward, next_state, terminated)
      state = next_state
      if actions_taken > minibatch_size:
        self.replay(minibatch_size,
                    update=(actions_taken % self.policy_freq) == 0)
      if terminated:
        self.align_all_models()

  def play(self, environment, max_steps=1000):
    # Reset the enviroment
    state = environment.reset()
    state = np.reshape(state, [1, -1]).astype(np.float32)
    actions_taken = 0
    terminated = False
    while actions_taken < max_steps and not terminated:
        # get action with exploration noise
      environment.render()
      actions_taken += 1
      action = self.actor(state)
      action = tf.clip_by_value(action,
                                clip_value_min=self._action_space_low,
                                clip_value_max=self._action_space_high).numpy()
      next_state, reward, terminated, info = environment.step(action)
      next_state = np.reshape(next_state, [1, -1]).astype(np.float32)
      self.store(state, action, reward, next_state, terminated)
      state = next_state
