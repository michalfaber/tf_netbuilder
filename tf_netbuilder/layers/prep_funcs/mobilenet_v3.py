
def prepare_ir(in_chs: int, args: map):
    new_args = dict(
        name=args["name"],
        in_chs=in_chs,
        out_chs=args["out_chs"],
        kernel_size=args["kernel_size"],
        strides=args["strides"],
        exp_ratio=args["exp_ratio"],
        se_factor=args.get("se_factor"),
        activation=args["activation"],
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros'
    )
    return new_args