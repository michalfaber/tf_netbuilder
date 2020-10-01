from copy import deepcopy

from tf_netbuilder import NetBuilderConfig
from tf_netbuilder.utils import clean_name

KEY_REPEATS_NUM = "_repeats"
BLOCK_TYPE_OPER = 'oper'
SELECTOR_BLOCK = "select"


def decode_block_args(args: list):
    """
    Parses block definition. For example the representation 'k3_s2' is decoded to {'kernel_size': 3, 'stride': 2}
    """
    args_defs = {}
    repeats = 1
    for arg in args:
        parser = NetBuilderConfig.find_parser(arg)
        arg_inner = arg[len(parser.code):]
        if parser.arg_name == KEY_REPEATS_NUM:
            repeats = parser.func(arg_inner)
        else:
            args_defs[parser.arg_name] = parser.func(arg_inner)

    return args_defs, repeats


def decode_block(block_def: str):
    """
    Decode block definition string
    Gets a list of block arg (dicts) through a string notation of arguments.
    E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip

    All args can exist in any order with the exception of the leading string which
    is assumed to indicate the block type.

    leading string - block type (
      ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
    r - number of repeat layers,
    k - kernel size,
    s - strides (1-9),
    e - expansion ratio,
    c - output channels,
    se - squeeze/excitation ratio
    n - activation fn ('re', 'r6', 'hs', or 'sw')
    Args:
        block_str: a string reprround_channelsesentation of block arguments.
    Returns:
        A list of block args (dicts)
    """
    assert isinstance(block_def, str)

    # get the block name

    label_marker_idx = block_def.find('#')
    label = None
    if label_marker_idx > -1:
        head = block_def[:label_marker_idx]
        if head.find(":") == -1:
            label = block_def[:label_marker_idx+1]
            block_def = block_def[label_marker_idx+1:]

    # determine if this is operation or layer

    op_marker_idx = block_def.find(':')

    if op_marker_idx > -1:
        ops = block_def.split(':')
        op_name = ops[0]
        op_args = [o for o in ops[1:] if len(o) > 0]

        block_args = dict(
            block_type=BLOCK_TYPE_OPER,
            oper_name=op_name,
            oper_args=op_args
        )
        num_repeat = 1
    else:
        ops = block_def.split('_')
        block_type = ops[0]  # take down the leading string which indicates the block type
        ops = ops[1:]        # all remaining stings will be decoded
        block_args = {
            "block_type": block_type
        }

        args, num_repeat = decode_block_args(ops)

        block_args.update(args)

    return block_args, num_repeat, label


def decode_arch_def(arch_def: list):
    """
    Creates a dictionary containing decoded blocks. Ex: 'k3' is resolved as 'kernel_size': 3
    """
    arch_args = []
    default_block_name = "layer_{}"
    cnt = 0
    for stack_idx, blocks_defs in enumerate(arch_def):
        assert isinstance(blocks_defs, list)
        for block_def in blocks_defs:
            assert isinstance(block_def, str)
            ba, rep, name = decode_block(block_def)

            repeated = [deepcopy(ba) for _ in range(rep)]
            for i, r in enumerate(repeated):
                # only the first block in any stack can have a stride > 1
                if i >= 1:
                    if 'strides' in r:
                        r['strides'] = 1
                # assigned name receives only the last layer in repeated blocks
                if name and i == len(repeated) - 1:
                    r['name'] = name
                # default name is simply the sequence number
                else:
                    r['name'] = default_block_name.format(cnt)
                cnt += 1

            arch_args.extend(repeated)
    return arch_args


def make_block(arch_args, in_chs):
    """
    Creates a block instance
    """
    block_type_code = arch_args['block_type']
    out_chs = arch_args.get('out_chs')
    if out_chs is None:
        out_chs = in_chs

    arch_args['name'] = clean_name(arch_args['name'])

    block_type_def = NetBuilderConfig.get_block_type(block_type_code)
    args = block_type_def.transform_args_fn(in_chs, arch_args)
    block = block_type_def.block_type_fn(**args)

    return block, out_chs
