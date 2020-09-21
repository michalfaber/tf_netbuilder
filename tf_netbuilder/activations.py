import tensorflow as tf
import numpy as np


def hard_swish(x):
  with tf.name_scope('hard_swish'):
    return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


ACTIVATION_FUNCS = dict(
    re=tf.nn.relu,
    r6=tf.nn.relu6,
    hs=hard_swish,
    sw=tf.nn.swish
)
