import tensorflow as tf
import numpy as np


def hard_swish(x):
  with tf.name_scope('hard_swish'):
    return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


class FuncAsLayer:
    def __init__(self, func):
        self.func = func

    def __call__(self):
        return tf.keras.layers.Activation(self.func)


ACTIVATION_FUNCS = dict(
    re=tf.keras.layers.ReLU,
    r6=FuncAsLayer(tf.nn.relu6),
    pre=tf.keras.layers.PReLU,
    hs=FuncAsLayer(hard_swish),
    sw=FuncAsLayer(tf.nn.swish)
)
