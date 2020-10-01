import tensorflow as tf


class ConvBnAct(tf.keras.layers.Layer):
    def __init__(self, cn_args, name, bn_args=None, activation=None):
        super(ConvBnAct, self).__init__(name=name)

        use_bias = cn_args.get("use_bias")
        if not use_bias:
            use_bias = not bn_args
            cn_args['use_bias'] = use_bias

        self.conv = tf.keras.layers.Conv2D(**cn_args)
        self.bn_args = bn_args
        if bn_args is not None:
            self.bn = tf.keras.layers.BatchNormalization(**bn_args)

        self.act = activation() if activation is not None else None

    def call(self, inputs):
        x = self.conv(inputs)
        if self.bn_args is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x
