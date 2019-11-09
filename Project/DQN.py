import numpy as np
import random
import tensorflow as tf
import math

from q_network import QNetworkDueling
from replay_memory import ReplayMemory

class DQN:
    def __init__(self, state_size, action_size, session,
                 summary_writer=None, exploration_period=1000,
                 minibatch_size=32, discount_factor=0.99,
                 experience_replay_buffer=10000,
                 target_qnet_update_frequency=10000,
                 initial_exploration_epsilon=1.0,
                 final_exploration_epsilon=0.05,
                 reward_clipping=-1):
        self.state_size = state_size
        self.action_size = action_size
        self.session = session
        self.exploration_period = float(exploration_period)
        self.minibatch_size = minibatch_size
        self.discount_factor = tf.constant(discount_factor)
        self.experience_replay_buffer = experience_replay_buffer
        self.summary_writer = summary_writer
        self.reward_clipping = reward_clipping
        self.target_qnet_update_frequency = target_qnet_update_frequency
        self.initial_exploration_epsilon = initial_exploration_epsilon
        self.final_exploration_epsilon = final_exploration_epsilon
        self.num_training_steps = 0
        
        self.qnet = QNetworkDueling(self.state_size, self.action_size, 'qnet')
        self.target_qnet = QNetworkDueling(self.state_size, self.action_size, 'target_qnet')
        self.qnet_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.99, epsilon=0.01)
        self.experience_replay = ReplayMemory(self.experience_replay_buffer, self.minibatch_size)

        self.create_graph()

    def create_graph(self):
        with tf.name_scope('pick_action'):
            self.state = tf.placeholder(tf.float32, (None, ) + self.state_size, name='state')
            self.q_values = tf.identity(self.qnet(self.state), name='q_values')
            self.predicted_actions = tf.argmax(self.q_values, dimension=1, name='predicted_actions')
            tf.summary.histogram('Q_values', tf.reduce_mean(tf.reduce_max(self.q_values, 1)))
        with tf.name_scope('estimating_future_rewards'):
            self.next_state = tf.placeholder(tf.float32, (None, ) + self.state_size, name='next_state')
            self.next_state_mask = tf.placeholder(tf.float32, (None, ), name='next_state_mask')
            self.rewards = tf.placeholder(tf.float32, (None, ), name='rewards')
            self.next_q_values_targetqnet = tf.stop_gradient(self.target_qnet(self.next_state), name='next_q_values_targetqnet')
            self.next_q_values_qnet = tf.stop_gradient(self.qnet(self.next_state), name='next_q_values_qnet')
            self.next_selected_actions = tf.argmax(self.next_q_values_qnet, dimension=1)
            self.next_selected_actions_onehot = tf.one_hot(indices=self.next_selected_actions, depth=self.action_size)
            self.next_max_q_values = tf.stop_gradient(tf.reduce_sum(tf.multiply(self.next_q_values_targetqnet, self.next_selected_actions_onehot), reduction_indices=[1,]) * self.next_state_mask)
            self.target_q_values = self.rewards + self.discount_factor * self.next_max_q_values
        with tf.name_scope('optimization_step'):
            self.action_mask = tf.placeholder(tf.float32, (None, self.action_size), name='action_mask')
            self.y = tf.reduce_mean(self.q_values * self.action_mask, reduction_indices=[1, ])
            # error clipping
            self.error = tf.abs(self.y - self.target_q_values)
            quadratic_part = tf.clip_by_value(self.error, 0.0, 1.0)
            linear_part = self.error - quadratic_part
            self.loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)
            # optimize the gradients
            qnet_gradients = self.qnet_optimizer.compute_gradients(self.loss, self.qnet.variables())
            for i, (grad, var) in enumerate(qnet_gradients):
                if grad is not None:
                    qnet_gradients[i] = (tf.clip_by_norm(grad, 10), var)
            self.qnet_optimize = self.qnet_optimizer.apply_gradients(qnet_gradients)
            with tf.name_scope('target_network_update'):
                self.hard_copy_to_target = self.copy_to_target_network(self.qnet, self.target_qnet)

    def copy_to_target_network(self, source_network, target_network):
        target_network_update = [v_target.assign(v_source) for v_source, v_target in zip(source_network.variables(), target_network.variables())]
        return tf.group(*target_network_update)

    def store(self, state, action, reward, next_state, is_terminal):
        if self.reward_clipping > 0.0:
            reward = np.clip(reward, -self.reward_clipping, self.reward_clipping)
        self.experience_replay.store(state, action, reward, next_state, is_terminal)

    def action(self, state, training=False):
        if self.num_training_steps > self.exploration_period:
            epsilon = self.final_exploration_epsilon
        else:
            epsilon = self.initial_exploration_epsilon - float(self.num_training_steps) * (self.initial_exploration_epsilon - self.final_exploration_epsilon) / self.exploration_period
        if not training:
            epsilon = 0.05
        if random.random() <= epsilon:
            action = random.randint(0, self.action_size - 1)
        else:
            action = self.session.run(self.predicted_actions, {self.state: [state]})[0]
        return action

    def train(self):
        if self.num_training_steps == 0:
            print('Training start ...')
            self.qnet.copy_to(self.target_qnet)
        minibatch = self.experience_replay.sample()
        batch_states = np.asanyarray([d[0] for d in minibatch])
        actions = [d[1] for d in minibatch]
        batch_actions = np.zeros((self.minibatch_size, self.action_size))
        for i in range(self.minibatch_size):
            batch_actions[i, actions[i]] = 1
        batch_rewards = np.asanyarray([d[2] for d in minibatch])
        batch_newstates = np.asanyarray([d[3] for d in minibatch])
        batch_newstates_mask = np.asanyarray([d[4] for d in minibatch])
        scores, _ = self.session.run([self.q_values, self.qnet_optimize], {
            self.state: batch_states,
            self.next_state: batch_newstates,
            self.next_state_mask: batch_newstates_mask,
            self.rewards: batch_rewards,
            self.action_mask: batch_actions
        })
        if self.num_training_steps % self.target_qnet_update_frequency == 0:
            self.session.run(self.hard_copy_to_target)
            print('mean maxQ in minibatch: {}'.format(np.mean(np.max(scores, 1))))
            # str_ = self.session.run(self.summary_writer)
        self.num_training_steps += 1