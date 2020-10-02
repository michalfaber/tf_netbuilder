import tensorflow as tf

from tf_netbuilder.layers.mobilenet_v3 import InvertedResidual


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


class SmallerConv7x7(tf.keras.layers.Layer):
    def __init__(self, in_chs, out_chs, exp_ratio, se_factor, activation, kernel_initializer, bias_initializer, name):
        super(SmallerConv7x7, self).__init__(name=name)

        block_out_chs = out_chs // 3

        self.conv1 = InvertedResidual(in_chs=in_chs, out_chs=block_out_chs,
                                      kernel_size=3, strides=1, exp_ratio=exp_ratio,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                      se_factor=se_factor, activation=activation, name=name, divisible_by=8, 
                                      padding='same')

        self.conv2 = InvertedResidual(in_chs=block_out_chs, out_chs=block_out_chs,
                                      kernel_size=3, strides=1, exp_ratio=exp_ratio,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                      se_factor=se_factor, activation=activation, name=name, divisible_by=8, 
                                      padding='same')

        self.conv3 = InvertedResidual(in_chs=block_out_chs, out_chs=block_out_chs,
                                      kernel_size=3, strides=1, exp_ratio=exp_ratio,
                                      kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                      se_factor=se_factor, activation=activation, name=name, divisible_by=8, 
                                      padding='same')

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return tf.concat([x1, x2, x3], 3)
