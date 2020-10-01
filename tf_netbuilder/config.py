import tensorflow as tf

from tf_netbuilder import NetBuilderConfig
from tf_netbuilder.parser import KEY_REPEATS_NUM
from tf_netbuilder.activations import ACTIVATION_FUNCS

from tf_netbuilder.layers import InvertedResidual, ConvBnAct
from tf_netbuilder.layers.prep_funcs import prepare_ir, prepare_cn, prepare_hd, prepare_avgpool, prepare_maxpool
from tf_netbuilder.operations import ConcatOper, MulOper, UpscaleX2, \
    NormalizationMinus05Plus05Input, NormalizationMinus1Plus1Input

# registering parsers

NetBuilderConfig.add_parser("n", lambda arg: ACTIVATION_FUNCS[arg], "activation")
NetBuilderConfig.add_parser("s", lambda arg: int(arg), "strides")
NetBuilderConfig.add_parser("c", lambda arg: int(arg), "out_chs")
NetBuilderConfig.add_parser("e", lambda arg: float(eval(arg)), "exp_ratio")
NetBuilderConfig.add_parser("se", lambda arg: int(arg), "se_factor")
NetBuilderConfig.add_parser("k", lambda arg: int(arg), "kernel_size")
NetBuilderConfig.add_parser("bn", lambda arg: True, "batch_norm")
# below is the internal parser and variable determining how many times a given layer should be repeated
NetBuilderConfig.add_parser("r", lambda arg: int(arg), KEY_REPEATS_NUM)

# registering blocks types

NetBuilderConfig.add_block_type("ir", InvertedResidual, prepare_ir)
NetBuilderConfig.add_block_type("cn", ConvBnAct, prepare_cn)
NetBuilderConfig.add_block_type("hd", tf.keras.layers.Conv2D, prepare_hd)
NetBuilderConfig.add_block_type("avgpool", tf.keras.layers.AveragePooling2D, prepare_avgpool)
NetBuilderConfig.add_block_type("maxpool", tf.keras.layers.MaxPooling2D, prepare_maxpool)

# registering operations

NetBuilderConfig.add_operation('cnct', ConcatOper)
NetBuilderConfig.add_operation('mul', MulOper)
NetBuilderConfig.add_operation('up_x2', UpscaleX2)

# registering operations on inputs

NetBuilderConfig.add_input_operation('norm1', NormalizationMinus1Plus1Input)
NetBuilderConfig.add_input_operation('norm05', NormalizationMinus05Plus05Input)
