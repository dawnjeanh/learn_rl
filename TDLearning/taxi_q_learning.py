import gym
import random

env = gym.make('Taxi-v3')
env.render()

alpha = 0.4
gamma = 0.999
epsilon = 0.017

q = {}
for s in range(env.nS):
    for a in range(env.nA):
        q[(s, a)] = 0.0

def update_q_table(pre_state, action, reward, next_state, alpha, gamma):
    qa = max([q[(next_state, a)] for a in range(env.nA)])
    q[(pre_state, action)] += alpha * (reward + gamma * qa - q[(pre_state, action)])

def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.nA)), key=lambda x: q[(state, x)])

for i in range(8000):
    r = 0
    pre_state = env.reset()
    while True:
        env.render()
        action = epsilon_greedy_policy(pre_state, epsilon)
        next_state, reward, done, _ = env.step(action)
        update_q_table(pre_state, action, reward, next_state, alpha, gamma)
        pre_state = next_state
        r += reward
        if done:
            break
    print('total reward: {}'.format(r))
env.close()