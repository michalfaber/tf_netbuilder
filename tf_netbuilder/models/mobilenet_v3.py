import tensorflow as tf

from tf_netbuilder.builder import NetModule
from tf_netbuilder.files import download_checkpoint


class MobilenetV3(tf.keras.Model):
    _model_def = {
        'inputs#': [
            ['img#norm1']  # custom name ‘img#’ for input and normalization -1 to 1
        ],
        'backbone#': [
            ['select:img#'],  # select the input
            ['cn_bn_r1_k3_s2_c16_nhs'],  # conv2d with batch norm, hard_swish
            ['ir_r1_k3_s1_e1_c16_nre'],  # inverted residual with expansion 1, relu
            ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # inverted residual
            ['c3#ir_r3_k5_s2_e3_c40_se4_nre'],  # custom name ‘c3#’ for the last repeated layer, size 1/8 of input
            ['ir_r1_k3_s2_e6_c80_nhs'],  # inverted residual with expansion 6, hard_swish
            ['ir_r1_k3_s1_e2.5_c80_nhs'],  # inverted residual with expansion 2.5, hard_swish
            ['ir_r2_k3_s1_e2.3_c80_nhs'],  # inverted residual with expansion 2.3, hard_swish
            ['c4#ir_r2_k3_s1_e6_c112_se4_nhs'],  # custom name ‘c4#’ for the last repeated layer, size 1/16 of input
            ['c5#ir_r3_k5_s2_e6_c160_se4_nhs'],  # custom name ‘c5#’ for the last repeated layer, size 1/32 of input
            ['cn_bn_r1_k1_s1_c960_nhs'],  # conv2d with batch norm, hard_swish,...
            ['avgpool_k7_s1'],  # average pooling
            ['cn_r1_k1_s1_c1280_nhs'],  # conv2d with hard_swish
        ]
    }

    _model_ins = 'inputs#'

    _model_outs = ['backbone#']

    @staticmethod
    def _global_pool(input_tensor, use_reduce_mean_for_pooling=False, pool_op=tf.nn.avg_pool2d):
        """Applies avg pool to produce 1x1 output.

        Args:
          input_tensor: input tensor
          use_reduce_mean_for_pooling: if True use reduce_mean for pooling
          pool_op: pooling op (avg pool is default)
        Returns:
          a tensor batch_size x 1 x 1 x depth.
        """
        if use_reduce_mean_for_pooling:
            return tf.reduce_mean(
                input_tensor, [1, 2], keepdims=True, name='ReduceMean')
        else:
            shape = input_tensor.get_shape().as_list()
            if shape[1] is None or shape[2] is None:
                kernel_size = tf.convert_to_tensor(value=[
                    1,
                    tf.shape(input=input_tensor)[1],
                    tf.shape(input=input_tensor)[2], 1
                ])
            else:
                kernel_size = [1, shape[1], shape[2], 1]
            output = pool_op(
                input_tensor, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID')
            # Recover output shape, for unknown shape.
            output.set_shape([None, 1, 1, None])
            return output

    def __init__(self, in_chs, num_classes):
        super(MobilenetV3, self).__init__()
        name = "MobilenetV3"

        self.net = NetModule(self._model_def,
                             self._model_ins,
                             self._model_outs, in_chs=[in_chs], name=name)

        self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

        self.logits_op = tf.keras.layers.Conv2D(
            num_classes, kernel_size=1, strides=(1, 1), padding='valid',
            dilation_rate=(1, 1), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros', name=name + "/logits/conv2d_1c_1x1"
        )

        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs):
        x, = self.net(inputs)

        x = self._global_pool(x)

        x = self.dropout(x)

        logits = self.logits_op(x)

        logits = tf.squeeze(logits, [1, 2])

        preds = self.softmax(logits)

        return logits, preds


def create_mobilenet_v3_224_1x(pretrained=False):

    pretrained_url = "https://github.com/michalfaber/tf_netbuilder/releases/download/v1.0/mobilenet_v3_224_1_0.zip"

    model = MobilenetV3(in_chs=3, num_classes=1001)

    model.build([tf.TensorShape((None, 224, 224, 3))])

    if pretrained:
        path = download_checkpoint(pretrained_url)
        model.load_weights(path)

    return model
