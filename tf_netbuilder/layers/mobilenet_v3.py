import tensorflow as tf

from tf_netbuilder.utils import make_divisible


class SqueezeExcite(tf.keras.layers.Layer):

    def __init__(self, in_chs, kernel_initializer='glorot_uniform', bias_initializer='zeros', se_factor=3,
                 act_squeeze_fn=tf.nn.relu, gate_fn=tf.sigmoid, divisible_by=8, padding='same', **kwargs):
        super(SqueezeExcite, self).__init__(**kwargs)

        squeeze_channels = make_divisible(
            in_chs / se_factor, divisor=divisible_by)

        self.squeeze = tf.keras.layers.Conv2D(
            squeeze_channels, kernel_size=1, strides=(1, 1), padding=padding, data_format=None,
            dilation_rate=(1, 1), groups=1, activation=act_squeeze_fn, use_bias=True,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Conv"
        )
        self.excite = tf.keras.layers.Conv2D(
            in_chs, kernel_size=1, strides=(1, 1), padding=padding, data_format=None,
            dilation_rate=(1, 1), groups=1, activation=gate_fn, use_bias=True,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="Conv_1"
        )

    def call(self, x):
        x_se = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x_se = self.squeeze(x_se)
        x_se = self.excite(x_se)
        x = x * x_se

        return x


class InvertedResidual(tf.keras.layers.Layer):

    def __init__(self, in_chs, out_chs, kernel_size, strides, exp_ratio, se_factor, activation, name, kernel_initializer,
                 bias_initializer, divisible_by=8, padding='same'):
        super(InvertedResidual, self).__init__(name=name)

        self.has_residual = (in_chs == out_chs and strides == 1)

        # Point-wise expansion

        mids_chs = make_divisible(in_chs * exp_ratio, divisible_by)

        if mids_chs > in_chs:
            self.expansion_conv = tf.keras.layers.Conv2D(
                mids_chs, kernel_size=1, strides=(1, 1), padding=padding, data_format=None,
                dilation_rate=(1, 1), groups=1, activation=None, use_bias=False,
                kernel_initializer=kernel_initializer, name="expand"
            )
            self.expansion_bn = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, name="expand/BatchNorm"
            )
            self.expansion_act = activation()
            self.has_expansion = True
        else:
            self.has_expansion = False

        # Depth-wise convolution

        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size, strides=(strides, strides), padding=padding, depth_multiplier=1,
            data_format=None, dilation_rate=(1, 1), activation=None, use_bias=False,
            depthwise_initializer=kernel_initializer, name="depthwise"
        )
        self.depthwise_bn = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, name="depthwise/BatchNorm"
        )
        self.depthwise_act = activation()

        # Projection convolution

        self.projection_conv = tf.keras.layers.Conv2D(
            out_chs, kernel_size=1, strides=(1, 1), padding=padding, data_format=None,
            dilation_rate=(1, 1), groups=1, activation=None, use_bias=False,
            kernel_initializer=kernel_initializer, name="project"
        )
        self.projection_bn = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, name="project/BatchNorm"
        )

        # Squeeze-and-excitation

        if se_factor is not None and se_factor > 0.:
            self.has_se = True
            self.se = SqueezeExcite(mids_chs, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    se_factor=se_factor, act_squeeze_fn=tf.nn.relu,
                                    gate_fn=lambda x: tf.nn.relu6(x + 3) * 0.16667, name="squeeze_excite")
        else:
            self.has_se = False

    def call(self, inputs):

        residual = inputs
        x = inputs

        # Point-wise expansion

        if self.has_expansion:
            x = self.expansion_conv(x)
            x = self.expansion_bn(x)
            x = self.expansion_act(x)

        # Depth-wise convolution

        x = self.depthwise_conv(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_act(x)

        if self.has_se:
            x = self.se(x)

        # Projection convolution

        x = self.projection_conv(x)
        x = self.projection_bn(x)

        if self.has_residual:
            x += residual

        return x
