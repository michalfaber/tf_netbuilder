from copy import deepcopy

from tf_netbuilder import NetBuilderConfig
from tf_netbuilder.utils import clean_name

KEY_REPEATS_NUM = "_repeats"
BLOCK_TYPE_OPER = 'oper'
SELECTOR_BLOCK = "pb"


def decode_block_args(args: list):
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
    """ Decode block definition string

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
    Raises:
        ValueError: if the string def not properly specified (TODO)
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

    # determine if this is operation or torch layer

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
        block_type = ops[0]  # take the block type off the front
        ops = ops[1:]
        block_args = {
            "block_type": block_type
        }

        args, num_repeat = decode_block_args(ops)

        block_args.update(args)

    return block_args, num_repeat, label


def decode_arch_def(arch_def: list):
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
    block_type_code = arch_args['block_type']
    out_chs = arch_args.get('out_chs')
    if out_chs is None:
        out_chs = in_chs

    arch_args['name'] = clean_name(arch_args['name'])

    block_type_def = NetBuilderConfig.get_block_type(block_type_code)
    args = block_type_def.transform_args_fn(in_chs, arch_args)
    block = block_type_def.block_type_fn(**args)

    return block, out_chs

#
# class NetParser:
#     __conf = dict(
#         blocks_parsers={},
#         blocks_decoders={},
#         blocks_types={},
#         operations={},
#         input_opers={},
#         sorted_parsers_codes=None
#     )
#     def __init__(self):
#         self.blocks_parsers = {}
#         self.blocks_decoders = {}
#         self.blocks_types = {}
#         self.operations = {}
#         self.input_opers = {}
#         self.sorted_parsers_codes = None
#
#     def add_parser(self, code: str, parser: Callable, arg_name: str):
#         self.blocks_parsers[code] = BlockParserDef(arg_name=arg_name,
#                                                    code=code,
#                                                    func=parser)
#         self.sorted_parsers_codes = sorted(self.blocks_parsers.keys(), key=len, reverse=True)
#
#     def add_block_type(self, block_type: str, block_type_fn: Callable, transform_args_fn: Callable):
#         self.blocks_types[block_type] = BlockType(block_type_fn, transform_args_fn)
#
#     def add_operation(self, code: str, oper: ClassVar):
#         self.operations[code] = oper
#
#     def add_input_operation(self, code: str, oper: ClassVar):
#         self.input_opers[code] = oper
#
#     def find_parser(self, arg_def: str):
#         for k in self.sorted_parsers_codes:
#             if arg_def.startswith(k):
#                 return self.blocks_parsers[k]
#         raise Exception("Parser not found.")
#
#     def decode_block_args(self, args: list):
#         args_defs = {}
#         repeats = 1
#         for arg in args:
#             parser = self.find_parser(arg)
#             arg_inner = arg[len(parser.code):]
#             if parser.arg_name == KEY_REPEATS_NUM:
#                 repeats = parser.func(arg_inner)
#             else:
#                 args_defs[parser.arg_name] = parser.func(arg_inner)
#
#         return args_defs, repeats
#
#     def decode_block(self, block_def: str):
#         """ Decode block definition string
#
#         Gets a list of block arg (dicts) through a string notation of arguments.
#         E.g. ir_r2_k3_s2_e1_i32_o16_se0.25_noskip
#
#         All args can exist in any order with the exception of the leading string which
#         is assumed to indicate the block type.
#
#         leading string - block type (
#           ir = InvertedResidual, ds = DepthwiseSep, dsa = DeptwhiseSep with pw act, cn = ConvBnAct)
#         r - number of repeat layers,
#         k - kernel size,
#         s - strides (1-9),
#         e - expansion ratio,
#         c - output channels,
#         se - squeeze/excitation ratio
#         n - activation fn ('re', 'r6', 'hs', or 'sw')
#         Args:
#             block_str: a string reprround_channelsesentation of block arguments.
#         Returns:
#             A list of block args (dicts)
#         Raises:
#             ValueError: if the string def not properly specified (TODO)
#         """
#         assert isinstance(block_def, str)
#
#         # get the block name
#
#         label_marker_idx = block_def.find('#')
#         label = None
#         if label_marker_idx > -1:
#             head = block_def[:label_marker_idx]
#             if head.find(":") == -1:
#                 label = block_def[:label_marker_idx+1]
#                 block_def = block_def[label_marker_idx+1:]
#
#         # determine if this is operation or torch layer
#
#         op_marker_idx = block_def.find(':')
#
#         if op_marker_idx > -1:
#             ops = block_def.split(':')
#             op_name = ops[0]
#             op_args = [o for o in ops[1:] if len(o) > 0]
#
#             block_args = dict(
#                 block_type=BLOCK_TYPE_OPER,
#                 oper_name=op_name,
#                 oper_args=op_args
#             )
#             num_repeat = 1
#         else:
#             ops = block_def.split('_')
#             block_type = ops[0]  # take the block type off the front
#             ops = ops[1:]
#             block_args = {
#                 "block_type": block_type
#             }
#
#             args, num_repeat = self.decode_block_args(ops)
#
#             block_args.update(args)
#
#         return block_args, num_repeat, label
#
#     def decode_arch_def(self, arch_def: list):
#         arch_args = []
#         default_block_name = "layer_{}"
#         cnt = 0
#         for stack_idx, blocks_defs in enumerate(arch_def):
#             assert isinstance(blocks_defs, list)
#             for block_def in blocks_defs:
#                 assert isinstance(block_def, str)
#                 ba, rep, name = self.decode_block(block_def)
#
#                 repeated = [deepcopy(ba) for _ in range(rep)]
#                 for i, r in enumerate(repeated):
#                     # only the first block in any stack can have a stride > 1
#                     if i >= 1:
#                         if 'strides' in r:
#                             r['strides'] = 1
#                     # assigned name receives only the last layer in repeated blocks
#                     if name and i == len(repeated) - 1:
#                         r['name'] = name
#                     # default name is simply the sequence number
#                     else:
#                         r['name'] = default_block_name.format(cnt)
#                     cnt += 1
#
#                 arch_args.extend(repeated)
#         return arch_args
#
#     def make_block(self, arch_args, in_chs):
#         block_type_code = arch_args['block_type']
#         out_chs = arch_args.get('out_chs')
#         if out_chs is None:
#             out_chs = in_chs
#
#         arch_args['name'] = clean_name(arch_args['name'])
#
#         block_type_def = self.blocks_types[block_type_code]
#         args = block_type_def.transform_args_fn(in_chs, arch_args)
#         block = block_type_def.block_type_fn(**args)
#
#         return block, out_chs