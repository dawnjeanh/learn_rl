import gym
import numpy as np
import pprint

env = gym.make('FrozenLake-v0')

print(env.observation_space.n)

print(env.action_space.n)
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(env.P)

def value_iteration(env, gamma=1.0):
    value_table = np.zeros(env.observation_space.n)
    no_of_iteration = 100000
    threshold = 1e-20
    for i in range(no_of_iteration):
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_state_rewards = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, rewards_prob, _ = next_sr
                    next_state_rewards.append((trans_prob * (rewards_prob + gamma * updated_value_table[next_state])))
                Q_value.append(np.sum(next_state_rewards))
            value_table[state] = max(Q_value)
        if np.sum(np.fabs(updated_value_table - value_table)) <= threshold:
            print('Value-iteration converged at iteration {}'.format(i + 1))
            break
    return value_table

def extract_policy(value_table, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, rewards_prob, _ = next_sr
                Q_table[action] += (trans_prob * (rewards_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)
    return policy

optimal_value_function = value_iteration(env)
optimal_policy = extract_policy(optimal_value_function)
print(optimal_policy)