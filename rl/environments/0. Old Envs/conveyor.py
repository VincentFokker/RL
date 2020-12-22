import gym
import numpy as np
import cv2, random
from stable_baselines.common.env_checker import check_env

class custom_env(gym.Env):
    def __init__(self, config=None, **kwargs):
        """
		Initialize variables such as state and actions space.
		"""

    self.config = config['environment']  ## can define the environment variables (size in the config file

    def make_observation(self):
        """
        Should return the current state of the environment.
        """

    def reset(self):
        """"
		Resets the state to the initial (start) state, in order to restart process.
		""""

    # return self.make_observation()

    def step(self, action):
        """"
		Set a step in the action space, should update the space.

		Run one timestep of the environment's dynamics. When end of
        episode is reached, call reset() to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
		""""

    # return next_state, reward, terminate, info

    def render():
        """"
		Should render the observation based on the current state. (Pure visualization)
		""""


env = custom_env()
check_env(env, warn=True)