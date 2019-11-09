import gym
import time

if __name__ == "__main__":
    env = gym.make('Taxi-v3')
    observation = env.reset()
    for t in range(1000):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("{} timesteps taken for the episode".format(t+1))
            break
    env.close()