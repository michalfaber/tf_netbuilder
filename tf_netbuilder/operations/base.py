import numpy as np
import tensorflow as tf


class MOperation:
    def __init__(self, input_tenors_names):
        self.input_tenors_names = input_tenors_names

    def get_input_tenors_names(self):
        return self.input_tenors_names

    def get_out_chs_number(self, input_chs_list):
        raise NotImplemented()

    def __call__(self, inputs):
        raise NotImplemented()


class ConcatOper(MOperation):

    def __init__(self, input_tenors_names):
        super(ConcatOper, self).__init__(input_tenors_names)

    def get_out_chs_number(self, input_chs_list):
        return np.sum(input_chs_list)

    def __call__(self, inputs):
        return tf.concat(inputs, 3)


class MulOper(MOperation):

    def __init__(self, input_tenors_names):
        super(MulOper, self).__init__(input_tenors_names)

    def get_out_chs_number(self, input_chs_list):
        return input_chs_list[0]

    def __call__(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        return tf.math.multiply(x1, x2)


class UpscaleX2(MOperation):

    def __init__(self, input_tenors_names):
        super(UpscaleX2, self).__init__(input_tenors_names)

    def get_out_chs_number(self, input_chs_list):
        return input_chs_list[0]

    def __call__(self, inputs):
        b, w, h, c = inputs.shape
        return tf.image.resize(inputs, (w*2, h*2))


class Normalization127Input:
    def __call__(self, x):
        return tf.cast(x, tf.float32) / 128. - 1


class NoOpInput:
    def __call__(self, x):
        return x
