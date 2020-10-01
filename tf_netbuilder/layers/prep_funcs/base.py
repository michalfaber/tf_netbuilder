
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


def prepare_cn(in_chs: int, args: map):

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

    if args.get("batch_norm"):
        bn_args = dict(
            momentum=0.999,
            epsilon=0.001,
            center=True,
            scale=True)
        new_args["bn_args"] = bn_args

    return new_args


def prepare_avgpool(in_chs: int, args: map):
    new_args = dict(
        name=args["name"],
        padding='valid',
        pool_size=args["kernel_size"],
        strides=args["strides"]
    )    

    return new_args


def prepare_maxpool(in_chs: int, args: map):
    new_args = dict(
        name=args["name"],
        pool_size=args["kernel_size"],
        strides=args["strides"]
    )

    return new_args
