import numpy as np
import tensorflow as tf

# functions commonly used for tensorflow models


def weight_variable(shape, name='weights', seed=None):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial, name=name)


def bias_variable(shape, value=0.1, name='biases'):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)