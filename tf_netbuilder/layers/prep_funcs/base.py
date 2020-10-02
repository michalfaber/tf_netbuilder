

def prepare_hd_args(in_chs: int, args: map):

    new_args = dict(
        use_bias=False,
        kernel_size=args["kernel_size"],
        strides=args["strides"],
        filters=args["out_chs"],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )
    return new_args


def prepare_cn_args(in_chs: int, args: map):

    new_args = dict(
        name=args["name"],
        activation=args.get("activation"),
        cn_args=dict(
            filters=args["out_chs"],
            kernel_size=args["kernel_size"],
            strides=args["strides"],
            padding='same',
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        ),
    )

    if args.get("use_batch_norm"):
        bn_args = dict(
            momentum=0.999,
            epsilon=0.001,
            center=True,
            scale=True)
        new_args["bn_args"] = bn_args
        
    return new_args


def prepare_cn2_args(in_chs: int, args: map):

    new_args = dict(
        name=args["name"],
        in_chs=in_chs,
        out_chs=args["out_chs"],
        exp_ratio=args["exp_ratio"],
        se_factor=args.get("se_factor"),
        activation=args["activation"],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )
    return new_args


def prepare_avgpool_args(in_chs: int, args: map):
    new_args = dict(
        name=args["name"],
        padding='valid',
        pool_size=args["kernel_size"],
        strides=args["strides"]
    )    

    return new_args


def prepare_maxpool_args(in_chs: int, args: map):
    new_args = dict(
        name=args["name"],
        pool_size=args["kernel_size"],
        strides=args["strides"]
    )

    return new_args
