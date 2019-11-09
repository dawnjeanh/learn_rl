import gym_bandits
import gym
import numpy as np
import math

env = gym.make("BanditTenArmedGaussian-v0")

num_rounds = 20000
count = np.zeros(env.action_space.n)
sum_rewards = np.zeros(env.action_space.n)
Q = np.zeros(env.action_space.n)

def ucb(iters):
    ucb_ = np.zeros(env.action_space.n)
    if iters < env.action_space.n:
        return iters
    for arm in range(env.action_space.n):
        upper_bound = math.sqrt((2 * math.log(sum(count))) / count[arm])
        ucb_[arm] = Q[arm] + upper_bound
    return np.argmax(ucb_)

for i in range(num_rounds):
    arm = ucb(i)
    observation, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print('optimal arm: {}'.format(np.argmax(Q)))