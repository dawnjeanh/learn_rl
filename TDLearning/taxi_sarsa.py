import gym
import random

env = gym.make('Taxi-v3')
env.render()

alpha = 0.85
gamma = 0.90
epsilon = 0.8

q = {}
for s in range(env.nS):
    for a in range(env.nA):
        q[(s, a)] = 0.0

def update_q_table(state, action, reward, next_state, next_action, alpha, gamma):
    q[(state, action)] += alpha * (reward + gamma * q[(next_state, next_action)] - q[(state, action)])

def epsilon_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.nA)), key=lambda x: q[(state, x)])

for i in range(4000):
    r = 0
    state = env.reset()
    action = epsilon_greedy_policy(state, epsilon)
    while True:
        env.render()
        next_state, reward, done, _ = env.step(action)
        next_action = epsilon_greedy_policy(state, epsilon)
        update_q_table(state, action, reward, next_state, next_action, alpha, gamma)
        state = next_state
        action = next_action
        r += reward
        if done:
            break
    print('total reward: {}'.format(r))
env.close()