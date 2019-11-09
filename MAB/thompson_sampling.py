import gym_bandits
import gym
import numpy as np
import math

env = gym.make("BanditTenArmedGaussian-v0")

num_rounds = 20000
count = np.zeros(env.action_space.n)
sum_rewards = np.zeros(env.action_space.n)
Q = np.zeros(env.action_space.n)

alpha = np.zeros(env.action_space.n)
beta = np.zeros(env.action_space.n)

def thompson_sampling(a, b):
    samples = [np.random.beta(a[i] + 1, b[i] + 1) for i in range(env.action_space.n)]
    return np.argmax(samples)

for i in range(num_rounds):
    arm = thompson_sampling(alpha, beta)
    observation, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]
    if reward > 0:
        alpha[arm] += 1
    else:
        beta[arm] += 1

print('optimal arm: {}'.format(np.argmax(Q)))