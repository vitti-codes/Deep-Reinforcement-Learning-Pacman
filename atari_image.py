import collections
import cv2
import numpy as np
from config import Constants, Hyper
import gym

""" modified from:
    https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/blob/master/Chapter06/lib/wrappers.py
"""

class RepeatActionAndMaxFrame(gym.Wrapper):

    def __init__(self, env=None):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = Hyper.image_jump
        self.shape = env.observation_space.high.shape
        self.frame_buffer = np.zeros_like((2,self.shape))

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(Hyper.image_jump):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreprocessFrame, self).__init__(env)
        shape_=Hyper.image_shape
        self.shape=(shape_[2], shape_[0], shape_[1])
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=self.shape,dtype=np.float32)

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env):
        repeat = Hyper.image_jump
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(repeat, axis=0),
                             env.observation_space.high.repeat(repeat, axis=0),
                             dtype=np.float32)
        self.obs_shape = self.observation_space.high.shape
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.obs_shape)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.array(self.stack).reshape(self.obs_shape)

        return obs

def make_env(env_name):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env)
    env = PreprocessFrame(env)
    env = StackFrames(env)

    return env