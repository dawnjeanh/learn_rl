import warnings
warnings.filterwarnings('ignore')
import gym
import time
import logging
import os
import sys
import tensorflow as tf
import numpy as np

from DQN import DQN
from env_wrapper import EnvWrapper

ENV_NAME = 'Seaquest-v0'
TOTAL_FRAMES = 20000000
MAX_TRAINING_STEPS = 20*60*60/3
TESTING_GAMES = 30
MAX_TESTING_STEPS = 5*60*60/3
TRAIN_AFTER_FRAMES = 50000
epoch_size = 50000
MAX_NOOP_START = 30
LOG_DIR = 'logs'
outdir = 'results'
test_mode = True
logger = tf.summary.FileWriter(LOG_DIR)
# Intialize tensorflow session
session = tf.InteractiveSession()

env = EnvWrapper(ENV_NAME, test_mode)
# print(dir(env.action_space))
agent = DQN(
    state_size=env.observation_space.shape,
    action_size=env.action_space.n,
    session=session,
    summary_writer=logger,
    exploration_period=1000000,
    minibatch_size=32,
    discount_factor=0.99,
    experience_replay_buffer=1000000,
    target_qnet_update_frequency=20000,
    initial_exploration_epsilon=1.0,
    final_exploration_epsilon=0.1,
    reward_clipping=1.0,
)
session.run(tf.initialize_all_variables())
logger.add_graph(session.graph)
saver = tf.train.Saver(tf.all_variables())

# env.monitor.start(outdir+'/'+ENV_NAME,force = True, video_callable=multiples_video_schedule)
num_frames = 0
num_games = 0
current_game_frames = 0
init_no_ops = np.random.randint(MAX_NOOP_START+1)
last_time = time.time()
last_frame_count = 0.0
state = env.reset()

while num_frames <= TOTAL_FRAMES + 1:
    # if test_mode:
    #     env.render()
    num_frames += 1
    current_game_frames += 1
    action = agent.action(state, training=True)
    next_state, reward, done, _ = env.step(action)
    if current_game_frames >= init_no_ops:
        agent.store(state, action, reward, next_state,done)
    state = next_state
    if num_frames >= TRAIN_AFTER_FRAMES:
        agent.train()
    if done or current_game_frames > MAX_TRAINING_STEPS:
        state = env.reset()
        current_game_frames = 0
        num_games += 1
        init_no_ops = np.random.randint(MAX_NOOP_START+1)
    if num_frames % epoch_size == 0 and num_frames > TRAIN_AFTER_FRAMES:
        saver.save(session, outdir+"/"+ENV_NAME+"/model_"+str(num_frames/1000)+"k.ckpt")
        print("epoch: frames={}, games={}".format(num_frames, num_games))
    if num_frames % (2*epoch_size) == 0 and num_frames > TRAIN_AFTER_FRAMES:
        total_reward = 0
        avg_steps = 0
        for i in range(TESTING_GAMES):
            state = env.reset()
            init_no_ops = np.random.randint(MAX_NOOP_START+1)
            frm = 0
            while frm < MAX_TESTING_STEPS:
                frm += 1
                env.render()
                action = agent.action(state, training=False)
                if current_game_frames < init_no_ops:
                    action = 0
                state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
            avg_steps += frm
        avg_reward = float(total_reward)/TESTING_GAMES
        str_ = session.run( tf.scalar_summary('test reward('+str(epoch_size/1000)+'k)', avg_reward) )
        logger.add_summary(str_, num_frames)
        state = env.reset()