import tensorflow as tf

from tf_netbuilder.builder import NetModule
from tf_netbuilder.files import download_checkpoint


class MobilenetV3(tf.keras.Model):

    _model_def = {
        'inputs#': [
            ['img#norm1']
        ],
        'backbone#': [
            ['select:img#'],  # select the input
            ['cn_bn_r1_k3_s2_c16_nhs'],
            # stage 0, 112x112 in
            ['ir_r1_k3_s1_e1_c16_nre'],  # relu
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
            # # stage 2, 56x56 in
            ['c3#ir_r3_k5_s2_e3_c40_se4_nre'],  # relu           -> C3 1/8
            # # stage 3, 28x28 in
            ['ir_r1_k3_s2_e6_c80_nhs'],    # hard_swish
            ['ir_r1_k3_s1_e2.5_c80_nhs'],  # hard_swish
            ['ir_r2_k3_s1_e2.3_c80_nhs'],  # hard_swish
            # stage 4, 14x14in
            ['c4#ir_r2_k3_s1_e6_c112_se4_nhs'],  # hard-swish    -> C4 1/16
            # stage 5, 14x14in / 7x7out
            ['c5#ir_r3_k5_s2_e6_c160_se4_nhs'],  # hard-swish    -> C5 1/32
            # stage 6, 7x7 in
            ['cn_bn_r1_k1_s1_c960_nhs'],
            ['avgpool_k7_s1'],
            ['cn_r1_k1_s1_c1280_nhs'],
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
    # TODO: replace it ! pretrained_url = "https://github.com/michalfaber/tf_netbuilder/releases/download/v1.0/mobilenet_v3_224_1_0.zip"

    pretrained_url = "https://www.dropbox.com/s/gjehgwltk2ab8ib/mobilenet_v3_224_1_0.zip?dl=1"

    model = MobilenetV3(in_chs=3, num_classes=1001)

    model.build([tf.TensorShape((None, 224, 224, 3))])

    if pretrained:
        path = download_checkpoint(pretrained_url)
        model.load_weights(path)

    return model
