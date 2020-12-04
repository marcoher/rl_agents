import gym
import numpy as np

from model import Agent


env = gym.make("MountainCarContinuous-v0")

agent = Agent(env.observation_space, env.action_space,
              policy_units=(10, 5),
              critic_units=(10, 5),
              load_path='./checkpoints/')

agent.play(env)
