import numpy as np
import random
import tensorflow as tf
import math

class ReplayMemory:
    def __init__(self, memory_size, minibatch_size):
        self.memory_size = memory_size
        self.minibatch_size = minibatch_size
        self.experience = [None] * self.memory_size
        self.current_index = 0
        self.size = 0

    def store(self, observation, action, reward, newobservation, is_terminal):
        self.experience[self.current_index] = (observation, action, reward, newobservation)
        self.current_index += 1
        self.size = min(self.size + 1, self.memory_size)
        if self.current_index >= self.memory_size:
            self.current_index -= self.memory_size

    def sample(self):
        if self.size < self.minibatch_size:
            return []
        samples_index = np.floor(np.random.random((self.minibatch_size,)) * self.size)
        samples = [self.experience[int(i)] for i in samples_index]
        return samples