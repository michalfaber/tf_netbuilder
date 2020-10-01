from typing import Callable, ClassVar


class BlockParserDef:
    def __init__(self, arg_name: str, code: str, func: Callable):
        self.arg_name = arg_name
        self.code = code
        self.func = func


class BlockType:
    def __init__(self, block_type_fn: Callable, transform_args_fn: Callable):
        self.block_type_fn = block_type_fn
        self.transform_args_fn = transform_args_fn


class NetBuilderConfig:
    __conf = dict(
        blocks_parsers={},
        blocks_decoders={},
        blocks_types={},
        operations={},
        input_opers={},
        sorted_parsers_codes=None
    )

    @staticmethod
    def add_parser(code: str, parser: Callable, arg_name: str):
        # dont call it twice for a code
        assert NetBuilderConfig.__conf['blocks_parsers'].get(code) is None

        NetBuilderConfig.__conf['blocks_parsers'][code] = BlockParserDef(arg_name=arg_name,
                                                                         code=code, func=parser)
        NetBuilderConfig.__conf['sorted_parsers_codes'] = sorted(
            NetBuilderConfig.__conf['blocks_parsers'].keys(), key=len, reverse=True
        )

    @staticmethod
    def add_block_type(block_type: str, block_type_fn: Callable, transform_args_fn: Callable):
        NetBuilderConfig.__conf['blocks_types'][block_type] = BlockType(block_type_fn, transform_args_fn)

    @staticmethod
    def add_operation(code: str, oper: ClassVar):
        NetBuilderConfig.__conf['operations'][code] = oper

    @staticmethod
    def add_input_operation(code: str, oper: ClassVar):
        NetBuilderConfig.__conf['input_opers'][code] = oper

    @staticmethod
    def get_operation(code: str):
        return NetBuilderConfig.__conf['operations'][code]

    @staticmethod
    def get_parser(code: str):
        return NetBuilderConfig.__conf['blocks_parsers'][code]

    @staticmethod
    def get_input_operation(code: str):
        return NetBuilderConfig.__conf['input_opers'][code]

    @staticmethod
    def get_block_type(code: str):
        return NetBuilderConfig.__conf['blocks_types'][code]

    @staticmethod
    def find_parser(code: str):
        all_codes = NetBuilderConfig.__conf['sorted_parsers_codes']
        for c in all_codes:
            if code.startswith(c):
                return NetBuilderConfig.get_parser(c)
        raise Exception("Parser not found.")