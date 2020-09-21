import tensorflow as tf

from tf_netbuilder.parser import NetParser, KEY_REPEATS_NUM
from tf_netbuilder.activations import ACTIVATION_FUNCS
from tf_netbuilder.layers import *
from tf_netbuilder.operations import *

parser = NetParser()

parser.add_parser("n", lambda arg: ACTIVATION_FUNCS[arg], "activation")
parser.add_parser("s", lambda arg: int(arg), "strides")
parser.add_parser("c", lambda arg: int(arg), "out_chs")
parser.add_parser("e", lambda arg: float(eval(arg)), "exp_ratio")
parser.add_parser("se", lambda arg: int(arg), "se_factor")
parser.add_parser("k", lambda arg: int(arg), "kernel_size")
parser.add_parser("bn", lambda arg: True, "batch_norm")
parser.add_parser("r", lambda arg: int(arg), KEY_REPEATS_NUM)  # helper parser


def prepare_ir(in_chs: int, args: map):
    new_args = dict(
        name=args["name"],
        in_chs=in_chs,
        out_chs=args["out_chs"],
        kernel_size=args["kernel_size"],
        strides=args["strides"],
        exp_ratio=args["exp_ratio"],
        se_factor=args.get("se_factor"),
        act_func=args["activation"],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )
    return new_args


def prepare_cn(in_chs: int, args: map):

    new_args = dict(
        name=args["name"],
        act_func=args.get("activation"),
        cn_args=dict(
            filters=args["out_chs"],
            kernel_size=args["kernel_size"],
            strides=args["strides"],
            padding='same',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ),
    )

    if args.get("batch_norm"):
        bn_args = dict(
            momentum=0.999,
            epsilon=0.001,
            center=True,
            scale=True)
        new_args["bn_args"] = bn_args

    return new_args


def prepare_r1x1(in_chs: int, args: map):
    new_args = dict(
        name=args["name"],
        kernel_size=args["kernel_size"],
        strides=args["strides"]
    )
    return new_args


def prepare_hd(in_chs: int, args: map):

    new_args = dict(
        use_bias=False,
        kernel_size=args["kernel_size"],
        strides=args["strides"],
        filters=args["out_chs"],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )
    return new_args


parser.add_block_type("ir", InvertedResidual, prepare_ir)
parser.add_block_type("cn", ConvBnAct, prepare_cn)
parser.add_block_type("hd", tf.keras.layers.Conv2D, prepare_hd)
parser.add_block_type("r1x1", Reduce1x1, prepare_r1x1)

parser.add_operation('cnct', ConcatOper)
parser.add_operation('mul', MulOper)
parser.add_operation('up_x2', UpscaleX2)

parser.add_input_operation('norm127', Normalization127Input)
