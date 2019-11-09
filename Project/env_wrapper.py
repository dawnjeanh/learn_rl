import warnings
warnings.filterwarnings('ignore')
import sys
f = True
while f:
    for p in sys.path:
        if 'python2.7' in p:
            sys.path.remove(p)
            f = True
            break
        else:
            f = False
print(sys.path)
    
import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Box
# from scipy.misc import imresize
import random
import cv2
import time
import logging
import os
import sys

class EnvWrapper:
    def __init__(self, env_name, debug=False):
        self.env = gym.make(env_name)
        self.action_space = self.env.action_space
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 4))
        self.frame_num = 0
        # self.monitor = self.env.monitor
        self.frames = np.zeros((84, 84, 4), dtype=np.int8)
        self.debug = debug
        if self.debug:
            cv2.namedWindow('Game')
            cv2.startWindowThread()

    def step(self, a):
        ob, reward, done, info = self.env.step(a)
        return self.process_frame(ob), reward, done, info

    def reset(self):
        self.frame_num = 0
        return self.process_frame(self.env.reset())

    def render(self):
        return self.env.render()

    def process_frame(self, frame):
        state_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        state_resized = cv2.resize(state_gray, (84, 110))
        gray_final = state_resized[16:100, :]
        if self.frame_num == 0:
            self.frames[:, :, 0] = gray_final
            self.frames[:, :, 1] = gray_final
            self.frames[:, :, 2] = gray_final
            self.frames[:, :, 3] = gray_final
        else:
            self.frames[:, :, 3] = self.frames[:, :, 2]
            self.frames[:, :, 2] = self.frames[:, :, 1]
            self.frames[:, :, 1] = self.frames[:, :, 0]
            self.frames[:, :, 0] = gray_final
        self.frame_num += 1
        if self.debug:
            cv2.imshow('Game', gray_final)
            cv2.waitKey(1)
        return self.frames.copy()

if __name__ == "__main__":
    env = EnvWrapper('CarRacing-v0', True)
    env.reset()
    while True:
        # env.render()
        env.step(env.env.action_space.sample())
        time.sleep(0.1)