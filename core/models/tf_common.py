import tensorflow as tf

# functions commonly used for tensorflow models


def weight_variable(shape, name='weights', seed=None):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial, name=name)


def bias_variable(shape, value=0.1, name='biases'):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial, name=name)
