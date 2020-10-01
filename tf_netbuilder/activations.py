from typing import Callable
import tensorflow as tf
import numpy as np


def hard_swish(x: tf.Tensor):
  with tf.name_scope('hard_swish'):
    return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


class FuncAsLayer:
    """
    This wrapper is a common denominator for function activations and layer activations. In tensorflow we have
    the activation PReLU which doesn't have function counterpart
    """
    def __init__(self, func: Callable):
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
