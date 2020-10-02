## Introduction

This library allows building deep learning architectures using string notation where string descriptors are decoded to tensorflow blocks: layers (tf.keras.layers.Conv2D) or operations on tensors (tf.concat). Those strings are grouped in python lists.

For example the lists :

```python
    ['cn_r2_k3_s1_c64_nre',
     'cn_r1_k3_s2_c128_nre']
```
...will be decoded to :

```python
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), stride=1, activation='tf.nn.relu')
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), stride=1, activation='tf.nn.relu')
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), stride=2, activation='tf.nn.relu')
```
Why is the first convolution repeated ? There is an argument ‘r2’ in the notation which means that this block should be repeated 2 times.

I came across this notation in this impressive git repo: [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) where the author uses this notation to build stacks of layers in some of his models. This is definitely a very concise method. I find it easier to manage experiments by storing model definitions as text files instead of constantly tweaking Python scripts.

My implementation goes a step further and introduces operations like concatenation, multiplications and any custom function which operates on tensors and can be used in the method **def call(self, x)** of custom class derived from **tf.keras.Model**. New block types or operations can be easily registered.

This allows building complex architectures with branches, upscaling, concatenating. All with simple text descriptors.

The folder **./tf_netbuilder/models** contains 2 sample models:
* mobilenet_v3.py
* openpose_singlenet.py

Test code is in the folder **./examples**.

##  Mobilenet V3

Definition of the architecture :

```python
   model_def = {
        'inputs#': [
            ['img#norm1']      # custom name ‘img#’ for input and normalization -1 to 1  
        ],
        'backbone#': [
            ['select:img#'],          # select the input
            ['cn_bn_r1_k3_s2_c16_nhs'],   # conv2d with batch norm, hard_swish
            ['ir_r1_k3_s1_e1_c16_nre'],   # inverted residual with expansion 1, relu
            ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # inverted residual
            ['c3#ir_r3_k5_s2_e3_c40_se4_nre'],    # custom name ‘c3#’ for the last repeated layer, size 1/8 of input
            ['ir_r1_k3_s2_e6_c80_nhs'],       # inverted residual with expansion 6, hard_swish
            ['ir_r1_k3_s1_e2.5_c80_nhs'],     # inverted residual with expansion 2.5, hard_swish
            ['ir_r2_k3_s1_e2.3_c80_nhs'],     # inverted residual with expansion 2.3, hard_swish
            ['c4#ir_r2_k3_s1_e6_c112_se4_nhs'],  # custom name ‘c4#’ for the last repeated layer, size 1/16 of input
            ['c5#ir_r3_k5_s2_e6_c160_se4_nhs'],  # custom name ‘c5#’ for the last repeated layer, size 1/32 of input
            ['cn_bn_r1_k1_s1_c960_nhs'],   # conv2d with batch norm, hard_swish,...
            ['avgpool_k7_s1'],             # average pooling
            ['cn_r1_k1_s1_c1280_nhs'],     # conv2d with hard_swish
        ]
    }

    model_ins = 'inputs#'

    model_outs = ['backbone#']
```

Python code:

```python
    from tf_netbuilder.builder import NetModule
    ...
    class MobilenetV3(tf.keras.Model):
        def __init__(self, num_classes):
            self.backbone = NetModule(
                    net_def=model_def,
                    inputs_stack_name=model_ins,
                    output_names=model_outs,
                    in_chs=[3], name='MobilenetV3')
            ...
    
        def call(self, inputs):
            x = self.backbone(inputs)
            ...
```
**NetBuilder** class is the engine of this library. It is nothing else than an implementation of **tf.keras.layers.Layer** so it can be embedded in your existing custom model.
Parameters:
* **net_def** - dictionary with definition of architecture
    * key - stack name (stacks are layer groups that can be referenced from other stacks)
    * value - list of lists of blocks (layers, operations)
* **inputs_stack** - name of a stack containing inputs definition. One stack should be reserved for inputs.
* **output_names** - list of names of stacks that we want at the output. This is useful if our model contains multiple branches or generally, we need multiple outputs from the model.
* **in_chs** - number of channels for each input. Note that this is a list because we can provide multiple inputs.
* **name** - custom name for the module.

Note:

All labels should end with the character ‘#’. This is an indicator that we are dealing with a label and it is also a separator in a block definition.

The first item in any stack should always be the block **select**. This is a way to set inputs for a stack. It may be model's input or other stack (reference by name).

Individual blocks can have custom name for example **c3#ir_r3...** This is useful if we want to use the tensor of that layer in an operation like concatenate, upscale, etc.

## Openpose singlenet with Mobilenet v3 as backbone

This is an example of a more complex model with concatenation, upsacling and multiple outputs.
The architecture is described in detail here: [Single-Network Whole-Body Pose Estimation](https://arxiv.org/abs/1909.13423)
Although the original implementation uses more stages and VGG as a backbone, I have created a smaller version based on MobilenetV3 and only 3 paf stages and one heatmap stage.
Here is the [training code](https://github.com/michalfaber/tensorflow_Realtime_Multi-Person_Pose_Estimation)

Definition of architecture

```python
    model_def = {
        'inputs#': [
            ['img#norm1']
        ],

        # Mobilenet v3 backbone

        'backbone#': [
            ['select:img#'],
            ['cn_bn_r1_k3_s2_c16_nhs'],
            ['ir_r1_k3_s1_e1_c16_nre'],
            ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],
            ['c3#ir_r3_k5_s2_e3_c40_se4_nre'],
            ['ir_r1_k3_s2_e6_c80_nhs'],
            ['ir_r1_k3_s1_e2.5_c80_nhs'],
            ['ir_r2_k3_s1_e2.3_c80_nhs'],
            ['c4#ir_r2_k3_s1_e6_c112_se4_nhs'],
            ['upscaled_c4#up_x2:'],
            ['cnct:c3#:upscaled_c4#']
        ],

        # PAF stages

        'stage_0#': [
            ['select:backbone#'],       
            ['cn2_r5_e1_c192_npre'],
            ['ir_r1_k1_s1_e1_c256_se4_npre'],
            ['hd_r1_k1_s1_c38']
        ],
        'stage_1#': [
            ['select:stage_0#:backbone#'],  # select 2 stacks
            ['cnct:'],                      # concatenate them
            ['cn2_r5_e1_c384_npre'],
            ['ir_r1_k1_s1_e1_c256_se4_npre'],
            ['hd_r1_k1_s1_c38']
        ],
        'stage_2#': [
            ['select:stage_1#:backbone#'],
            ['cnct:'],
            ['cn2_r5_e1_c384_npre'],
            ['ir_r1_k1_s1_e1_c512_se4_npre'],
            ['hd_r1_k1_s1_c38']
        ],

        # Heatmap stages

        'stage_3#': [
            ['select:stage_2#:backbone#'],
            ['cnct:'],
            ['cn2_r5_e1_c384_npre'],
            ['ir_r1_k1_s1_e1_c512_se4_npre'],
            ['hd_r1_k1_s1_c19']
        ],
    }

    model_ins = 'inputs#'

    model_outs = ['stage_0#', 'stage_1#', 'stage_2#', 'stage_3#']
```

Python code

```python
    from tf_netbuilder.builder import NetModule
    ...
    class OpenPoseSingleNet(tf.keras.Model):
        def __init__(self, in_chs):
            super(OpenPoseSingleNet, self).__init__()
    
            self.net = NetModule(
                    net_def=model_def,
                    inputs_stack_name=model_ins,
                    output_names=model_outs,
                    in_chs=[3], name='MobilenetV3')
    
        def call(self, inputs):
            x = self.net(inputs)
    
            return x    
```

Note:

This model has multiple stacks. How can we distinguish an operation from a layer ? Operation code ends with the character ':' and all its arguments (references to other stacks or blocks) should be separated by ':' as well. For example:

```python
    ['cnct:c3#:upscaled_c4#']
```

This is a concatenation operation denoted by a code **cnct:**. It has 2 arguments **c3#** and  **upscaled_c4#** which are references to other blocks.

Take a look at this special construct:

```python
    {
        'stage_2#': [
            ['select:stage_1#:backbone#'],
            ['cnct:'],
            ...
        ]
    }
```
Here we select **stage_1#** and **backbone#** as input to the **stage_2#** stack. The operation in the next line **cnct:** will perform concatenation on the previously selected stacks.

##  How to add more layers and operations

The current implementation contains quite a small set of layers and operations. Only those that I needed in my experiments with pose estimation models.
More layers and operation can be easily added. Take a look at the script **./tf_netbuilder/config.py** :

```python
    ...
    # registering parsers for arguments of all blocks
    
    NetBuilderConfig.add_parser("n", lambda arg: ACTIVATION_FUNCS[arg], "activation")
    NetBuilderConfig.add_parser("s", lambda arg: int(arg), "strides")
    NetBuilderConfig.add_parser("c", lambda arg: int(arg), "out_chs")
    NetBuilderConfig.add_parser("e", lambda arg: float(eval(arg)), "exp_ratio")
    NetBuilderConfig.add_parser("se", lambda arg: int(arg), "se_factor")
    NetBuilderConfig.add_parser("k", lambda arg: int(arg), "kernel_size")
    NetBuilderConfig.add_parser("bn", lambda arg: True, "batch_norm")   # no arg, just the flag
    # below is the internal parser and variable determining how many times a given layer should be repeated
    NetBuilderConfig.add_parser("r", lambda arg: int(arg), KEY_REPEATS_NUM)
    
    # registering blocks types - layers
    
    NetBuilderConfig.add_block_type("ir", InvertedResidual, prepare_ir_args)
    NetBuilderConfig.add_block_type("cn", ConvBnAct, prepare_cn_args)
    NetBuilderConfig.add_block_type("cn2", SmallerConv7x7, prepare_cn2_args)
    NetBuilderConfig.add_block_type("hd", tf.keras.layers.Conv2D, prepare_hd_args)
    NetBuilderConfig.add_block_type("avgpool", tf.keras.layers.AveragePooling2D, prepare_avgpool_args)
    NetBuilderConfig.add_block_type("maxpool", tf.keras.layers.MaxPooling2D, prepare_maxpool_args)
    
    # registering operations
    
    NetBuilderConfig.add_operation('cnct', ConcatOper)
    NetBuilderConfig.add_operation('mul', MulOper)
    NetBuilderConfig.add_operation('up_x2', UpscaleX2)
    
    # registering operations on inputs
    
    NetBuilderConfig.add_input_operation('norm1', NormalizationMinus1Plus1Input)
    NetBuilderConfig.add_input_operation('norm05', NormalizationMinus05Plus05Input)
    ...
```

In short:
* **add_parser** - parser for a single argument. For example k3 should be parsed to {‘kernel_size’: 3}
* **add_block_type** - here you can assign a layer class to a string code. Plus, additional function that prepares a set of arguments for a given layer.
* **add_operation** - code for operation and class implementing that operation.
* **add_input_operation** - code for special operation on inputs. Usually, this is kind of an input normalization etc.

## Try without installing the library

There is still work going on in this library so I obviously don’t recommend installing it system-wide. You can try it with just a few simple steps:

```sh
    git clone https://github.com/michalfaber/tf_netbuilder
    cd tf_netbuilder
    mkdir .venv
    virtualenv .venv/tf_netbuilder
    source .venv/tf_netbuilder/bin/activate
    pip install -r requirements.txt
    cd examples
    python eval_mobilenet_v3.py
    python eval_openpose_singlenet.py
```

##  Full doc

...soon
