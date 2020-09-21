import tensorflow as tf


class ConvBnAct(tf.keras.layers.Layer):
    def __init__(self, cn_args, name, bn_args=None, act_func=None):
        super(ConvBnAct, self).__init__(name=name)

        use_bias = cn_args.get("use_bias")
        if not use_bias:
            use_bias = not bn_args
            cn_args['use_bias'] = use_bias

        self.conv = tf.keras.layers.Conv2D(**cn_args)
        self.bn_args = bn_args
        if bn_args is not None:
            self.bn = tf.keras.layers.BatchNormalization(**bn_args)

        self.act_func = act_func
        if act_func is not None:
            self.act = tf.keras.layers.Activation(act_func)

    def call(self, inputs):
        x = self.conv(inputs)
        if self.bn_args is not None:
            x = self.bn(x)

        if self.act_func is not None:
            x = self.act(x)

        return x


class Reduce1x1(tf.keras.layers.Layer):

    def __init__(self, kernel_size, strides, name):
        super(Reduce1x1, self).__init__(name=name)

        self.avg = tf.keras.layers.AveragePooling2D(
            pool_size=(kernel_size, kernel_size), strides=strides, padding='valid', data_format=None
        )

    def call(self, x):
        return self.avg(x)
