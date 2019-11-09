import numpy as np
import random
import tensorflow as tf
import math

class QNetwork:
    def __init__(self, name):
        self.name = name

    def weight_variable(self, shape, fanin=0):
        if fanin == 0:
            initial = tf.truncated_normal(shape, stddev=0.01)
        else:
            mod_init = 1.0 / math.sqrt(fanin)
            initial = tf.random_uniform(shape, minval=-mod_init, maxval=mod_init)
        return tf.Variable(initial)

    def bias_variable(self, shape, fanin=0):
        if fanin == 0:
            initial = tf.constant(0.01, shape=shape)
        else:
            mod_init = 1.0 / math.sqrt(fanin)
            initial = tf.random_uniform(shape, minval=-mod_init, maxval=mod_init)
        return initial

    def variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    def copy_to(self, dst_net):
        v1 = self.variables()
        v2 = dst_net.variables()
        for i in range(len(v1)):
            v2[i].assign(v1[i]).eval()

    def print_num_of_parameters(self):
        list_vars = self.variables()
        total_parameters = 0
        for var in list_vars:
            shape = var.get_shape()
            var_param = 1
            for dim in shape:
                var_param *= dim.value
            total_parameters += var_param
        print('network {}: {} -> {}M'.format(self.name, total_parameters, np.round(float(total_parameters)/1000000.0, 2)))

class QNetworkDueling(QNetwork):
    def __init__(self, input_size, output_size, name):
        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        with tf.variable_scope(self.name):
            self.W_conv1 = self.weight_variable([8, 8, 4, 32])
            self.B_conv1 = self.bias_variable([32])
            self.stride1 = 4
            self.W_conv2 = self.weight_variable([4, 4, 32, 64])
            self.B_conv2 = self.bias_variable([64])
            self.stride2 = 2
            self.W_conv3 = self.weight_variable([3, 3, 64, 64])
            self.B_conv3 = self.bias_variable([64])
            self.stride3 = 1
            self.W_fc4a = self.weight_variable([7*7*64, 512])
            self.B_fc4a = self.bias_variable([512])
            self.W_fc4b = self.weight_variable([7*7*64, 512])
            self.B_fc4b = self.bias_variable([512])
            # Value stream
            self.W_fc5a = self.weight_variable([512, 1])
            self.B_fc5a = self.bias_variable([1])
            # Advantage stream
            self.W_fc5b = self.weight_variable([512, self.output_size])
            self.B_fc5b = self.bias_variable([self.output_size])

    def __call__(self, input_tensor):
        if type(input_tensor) == list:
            input_tensor = tf.concat(1, input_tensor)
        with tf.variable_scope(self.name):
            # Conv
            self.h_conv1 = tf.nn.relu(tf.nn.conv2d(input_tensor, self.W_conv1, strides=[1, self.stride1, self.stride1, 1], padding='VALID') + self.B_conv1)
            self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_conv1, self.W_conv2, strides=[1, self.stride2, self.stride2, 1], padding='VALID') + self.B_conv2)
            self.h_conv3 = tf.nn.relu(tf.nn.conv2d(self.h_conv2, self.W_conv3, strides=[1, self.stride3, self.stride3, 1], padding='VALID') + self.B_conv3)
            # flatten & fc
            self.h_conv3_flat = tf.reshape(self.h_conv3, [-1, 7*7*64])
            self.h_fc4a = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4a) + self.B_fc4a)
            self.h_fc4b = tf.nn.relu(tf.matmul(self.h_conv3_flat, self.W_fc4b) + self.B_fc4b)
            # value stream
            self.h_value = tf.identity(tf.matmul(self.h_fc4a, self.W_fc5a) + self.B_fc5a)
            # advantage stream
            self.h_advantage = tf.identity(tf.matmul(self.h_fc4b, self.W_fc5b) + self.B_fc5b)
            # club v & a
            self.h_output = self.h_value + (self.h_advantage - tf.reduce_mean(self.h_advantage, reduction_indices=[1,], keep_dims=True))
            return self.h_output