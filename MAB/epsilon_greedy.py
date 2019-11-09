import gym_bandits
import gym
import numpy as np

env = gym.make("BanditTenArmedGaussian-v0")

num_rounds = 20000
count = np.zeros(env.action_space.n)
sum_rewards = np.zeros(env.action_space.n)
Q = np.zeros(env.action_space.n)

def epsilon_greedy(epsolon):
    rand = np.random.random()
    if rand < epsolon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    return action

for i in range(num_rounds):
    arm = epsilon_greedy(0.5)
    observation, reward, done, info = env.step(arm)
    count[arm] += 1
    sum_rewards[arm] += reward
    Q[arm] = sum_rewards[arm] / count[arm]

print('optimal arm: {}'.format(np.argmax(Q)))