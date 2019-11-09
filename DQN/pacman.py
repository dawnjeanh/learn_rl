# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import random
from datetime import datetime
import matplotlib.pyplot as plt

env = gym.make("MsPacman-v0")
n_outputs = env.action_space.n

color = np.array([210, 164, 74]).mean()
def preprocess_observation(obs):
    img = obs[1:176:2, ::2]
    img = img.mean(axis=2)
    img[img==color] = 0
    img = (img - 128) / 128 - 1
    return img.reshape(88,80,1)

tf.reset_default_graph()
def q_network(X, name_scope):
    initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name_scope) as scope:
        layer_1 = conv2d(X, num_outputs=32, kernel_size=(8, 8), stride=4,
                         padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_1', layer_1)

        layer_2 = conv2d(layer_1, num_outputs=64, kernel_size=(4, 4), stride=2,
                         padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_2', layer_2)

        layer_3 = conv2d(layer_2, num_outputs=64, kernel_size=(3, 3), stride=1,
                         padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer_3', layer_3)

        flat = flatten(layer_3)
        fc = fully_connected(flat, num_outputs=128, weights_initializer=initializer)
        tf.summary.histogram('fc', fc)

        output = fully_connected(fc, num_outputs=n_outputs, activation_fn=None, weights_initializer=initializer)
        tf.summary.histogram('output', output)

        vars_ = {v.name[len(scope.name):]: v for v in tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}

        return vars_, output


eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 500000
def epsilon_greedy(action, step):
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action

exp_buffer = []
def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    men = np.array(exp_buffer)[perm_batch]
    return men[:, 0], men[:, 1], men[:, 2], men[:, 3], men[:, 4]

# hyperparameters
num_episodes = 800
batch_size = 48
input_shape = (None, 88, 80, 1)
learning_rate = 0.001
X_shape = (None, 88, 80, 1)
discount_factor = 0.97

global_step = 0
copy_steps = 100
steps_train = 4
start_steps = 2000
logdir = 'logs'

X = tf.placeholder(tf.float32, shape=X_shape)
in_training_mode = tf.placeholder(tf.bool)

mainQ, mainQ_output = q_network(X, 'mainQ')

targetQ, targetQ_output = q_network(X, 'targetQ')

X_action = tf.placeholder(tf.int32, shape=(None,))
Q_action = tf.reduce_sum(targetQ_output * tf.one_hot(X_action, n_outputs), axis=-1, keep_dims=True)

copy_op = [tf.assign(main_name, targetQ[var_name]) for var_name, main_name in mainQ.items()]
copy_target_to_main = tf.group(*copy_op)

y = tf.placeholder(tf.float32, shape=(None, 1))

loss = tf.reduce_mean(tf.square(y - Q_action))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

loss_summary = tf.summary.scalar('LOSS', loss)
merge_summary = tf.summary.merge_all()

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(num_episodes):
        done = False
        obs = env.reset()
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter()
        episodic_loss = []
        while not done:
            env.render()
            obs = preprocess_observation(obs)
            actions = mainQ_output.eval(feed_dict={X:[obs], in_training_mode:False})
            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1
            action = epsilon_greedy(action, global_step)
            next_obs, reward, done, _ = env.step(action)
            exp_buffer.append([obs, action, preprocess_observation(next_obs), reward, done])
            if global_step % steps_train == 0 and global_step > start_steps:
                o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)
                o_obs = [x for x in o_obs]
                o_next_obs = [x for x in o_next_obs]
                next_act = mainQ_output.eval(feed_dict={X:o_next_obs, in_training_mode:False})
                y_batch = o_rew + discount_factor * np.max(next_act, axis=-1) * (1 - o_done)

                mrg_summary = merge_summary.eval(feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:False})
                file_writer.add_summary(mrg_summary, global_step)

                train_loss, _ = sess.run([loss, training_op], feed_dict={X:o_obs, y:np.expand_dims(y_batch, axis=-1), X_action:o_act, in_training_mode:True})
                episodic_loss.append(train_loss)
            if (global_step + 1) % copy_steps == 0 and global_step > start_steps:
                copy_target_to_main.run()
            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
        print('Epoch {} reward: {}'.format(epoch, episodic_reward))
