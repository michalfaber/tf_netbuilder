import tensorflow as tf

from tf_netbuilder import NetBuilderConfig
from tf_netbuilder.parser import BLOCK_TYPE_OPER, SELECTOR_BLOCK, decode_arch_def, make_block
from tf_netbuilder.operations import NoOpInput
from tf_netbuilder.utils import clean_name


class StackInputDescriptor:
    def __init__(self, input_name: str, channels: int):
        self.input_name = input_name
        self.channels = channels


class ExecutionItem:
    def __init__(self, block_name, block, out_chs, is_oper):
        self.block_name = block_name
        self.block = block
        self.out_chs = out_chs
        self.is_oper = is_oper


class StackExecutionItem:
    def __init__(self, stack_name: str, inputs_refs: str, out_chs: int):
        self.stack_name = stack_name
        self.inputs_refs = inputs_refs
        self.out_chs = out_chs


class InputItem:
    def __init__(self, input_name, oper, channels):
        self.input_name = input_name
        self.oper = oper
        self.channels = channels


class StackModule(tf.keras.layers.Layer):
    def __init__(self, stack_def: list, inputs_chs: list, name: str):
        super(StackModule, self).__init__(name=name)

        self.stack_def = stack_def
        self.in_chs = inputs_chs

        self.execution_list = []

        self.parse_stack()

    def get_last_channels_num(self):
        assert len(self.in_chs) == 1

        return self.in_chs[0]

    def parse_stack(self):
        all_blocks_args = decode_arch_def(self.stack_def)  # list of lists - stages contains layers

        for block_args in all_blocks_args:
            ei = self.create_execution_item(block_args, self.in_chs)
            self.in_chs = [ei.out_chs]

            if not ei.is_oper:
                self.__setattr__(clean_name(ei.block_name), ei.block)

            self.execution_list.append(ei)

    def find_exec_item(self, name):
        for ei in self.execution_list:
            if ei.block_name == name:
                return ei
        raise Exception(f'Block {name} not found in execution list. Check if order of blocks is correct.')

    def create_execution_item(self, block_args, in_chs):
        name = block_args.get('name')

        block_type = block_args.get('block_type')

        # operation

        if block_type == BLOCK_TYPE_OPER:
            oper_name = block_args['oper_name']
            oper_args = block_args['oper_args']
            oper_class = NetBuilderConfig.get_operation(oper_name)
            oper = oper_class(oper_args)

            input_tensor_chs = []
            for tn in oper.get_input_tenors_names():
                ei = self.find_exec_item(tn)
                input_tensor_chs.append(ei.out_chs)

            if len(input_tensor_chs) > 0:
                out_chs = oper.get_out_chs_number(input_tensor_chs)
            else:
                out_chs = oper.get_out_chs_number(in_chs)

            ei = ExecutionItem(name, oper, out_chs, is_oper=True)

            return ei

        # module item

        else:
            assert len(in_chs) == 1

            block, out_chs = make_block(block_args, in_chs[0])
            ei = ExecutionItem(name, block, out_chs, is_oper=False)

            return ei

    def resolve_reference(self, ref_name: str, evaluated_tensors: dict):
        t = evaluated_tensors.get(ref_name)
        if t is None:
            raise Exception(F'Evaluated tensor {ref_name} not found. Check if order of blocks is correct.')
        return t

    def call(self, x):
        evaluated_tensors = {}
        for ei in self.execution_list:

            if ei.is_oper:
                input_tensor_names = ei.block.get_input_tenors_names()
                input_tensors = [self.resolve_reference(it, evaluated_tensors) for it in input_tensor_names]
                if len(input_tensors) > 0:
                    x = ei.block(input_tensors)
                else:
                    x = ei.block(x)
            else:
                module = getattr(self, clean_name(ei.block_name))
                x = module(x)

            if '#' in ei.block_name:
                evaluated_tensors[ei.block_name] = x

        evaluated_tensors.clear()
        return x


class NetModule(tf.keras.layers.Layer):

    def __init__(self, net_def: dict, inputs_stack_name: str, output_names: list, in_chs: list, name: str):
        super(NetModule, self).__init__(name=name)

        self.execution_list = []
        self.inputs_stack_name = inputs_stack_name
        self.output_names = output_names

        # get inputs definition

        inputs_def = net_def[self.inputs_stack_name]
        input_blocks_defs = []
        for inp_stage in inputs_def:
            for inp_block in inp_stage:
                input_blocks_defs.append(inp_block)

        self.inputs_ops = []
        for idx, inp_block_def in enumerate(input_blocks_defs):
            name, op = self.get_input_oper(inp_block_def)
            self.inputs_ops.append(InputItem(input_name=name,
                                             oper=op,
                                             channels=in_chs[idx]))
        # build stacks

        for stack_name, stack in net_def.items():
            if stack_name not in inputs_stack_name:
                stack_without_input_stage = stack[1:]   # first item on the list constitute an input stage
                                                        # and shouldn't be evaluated inside the StackModul
                inputs_refs = self.get_stack_inputs_refs(stack[0][0])

                inputs_chs = self.get_stack_input_channels(inputs_refs)

                sm = StackModule(stack_without_input_stage, inputs_chs, clean_name(stack_name))
                out_chs = sm.get_last_channels_num()

                self.__setattr__(clean_name(stack_name), sm)

                self.execution_list.append(
                    StackExecutionItem(stack_name=stack_name,
                                       inputs_refs=inputs_refs,
                                       out_chs=out_chs)
                )

    def get_stack_inputs_refs(self, input_stage_def):
        parts = input_stage_def.split(":")
        selector_name = parts[0]
        refs = parts[1:]

        if selector_name == SELECTOR_BLOCK:
            return refs
        else:
            raise Exception("Invalid input selector.")
        pass

    def get_stack_input_channels(self, selector):
        channels = []
        for s in selector:
            found = False
            for inp in self.inputs_ops:
                if inp.input_name == s:
                    channels.append(inp.channels)
                    found = True
                    break

            for ei in self.execution_list:
                if ei.stack_name == s:
                    channels.append(ei.out_chs)
                    found = True
                    break
            if not found:
                raise Exception("Invalid selector reference.")

        return channels

    def get_input_oper(self, inp_block_def: str):
        parts = inp_block_def.split('#')
        if len(parts) != 2:
            raise Exception('Invalid definition of stack with inputs.')
        input_name, oper_name = parts
        input_name = input_name + "#"

        if len(oper_name) > 0:
            input_oper_class = NetBuilderConfig.get_input_operation(oper_name)
        else:
            input_oper_class = NoOpInput
        oper = input_oper_class()

        return input_name, oper

    def call(self, x):

        evaluated_stacks = {}

        # prepare inputs

        for idx, op in enumerate(self.inputs_ops):
            inputs = x if isinstance(x, tf.Tensor) else x[idx]

            ii = op.oper(inputs)
            evaluated_stacks[op.input_name] = ii

        # run all the stacks

        for stack_exec_item in self.execution_list:
            stack_name = stack_exec_item.stack_name
            inputs_refs = stack_exec_item.inputs_refs

            module = getattr(self, clean_name(stack_name))

            stack_inputs = [evaluated_stacks[ref] for ref in inputs_refs]
            if len(stack_inputs) == 1:
                stack_inputs = stack_inputs[0]

            x = module(stack_inputs)

            evaluated_stacks[stack_name] = x

        # gather outputs

        out_tensors = []
        for on in self.output_names:
            t = evaluated_stacks.get(on)
            out_tensors.append(t)

        evaluated_stacks.clear()

        return tuple(out_tensors)
