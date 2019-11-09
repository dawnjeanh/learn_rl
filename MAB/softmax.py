import gym_bandits
import gym
import numpy as np
import math

env = gym.make("BanditTenArmedGaussian-v0")

num_rounds = 20000
count = np.zeros(env.action_space.n)
sum_rewards = np.zeros(env.action_space.n)
Q = np.zeros(env.action_space.n)

def softmax(tau):
    total = sum([math.exp(val / tau) for val in Q])
    probs = [math.exp(val / tau) / total for val in Q]
    threshold = np.random.random()
    cumulative_prob = 0.0
    for i, p in enumerate(probs):
        cumulative_prob += p
        if cumulative_prob > threshold:
            return i
    return np.argmax(probs)

for i in range(num_rounds):
    arm = softmax(0.5)
    observation, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print('optimal arm: {}'.format(np.argmax(Q)))