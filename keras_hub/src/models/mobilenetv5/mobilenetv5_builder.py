import re
from copy import deepcopy

from keras_hub.src.models.mobilenet.mobilenet_backbone import ConvBnActBlock
from keras_hub.src.models.mobilenet.mobilenet_backbone import DepthwiseConvBlock
from keras_hub.src.models.mobilenet.mobilenet_backbone import (
    InvertedResidualBlock,
)
from keras_hub.src.models.mobilenet.util import adjust_channels
from keras_hub.src.models.mobilenetv5.mobilenetv5_attention import (
    MobileAttention,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import CondConvResidual
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import EdgeResidual
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import (
    UniversalInvertedResidual,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_utils import parse_ksize
from keras_hub.src.models.mobilenetv5.mobilenetv5_utils import round_channels


def decode_block_str(block_str):
    assert isinstance(block_str, str)
    ops = block_str.split("_")
    block_type = ops[0]
    ops = ops[1:]
    options = {}
    skip = None
    for op in ops:
        if op == "noskip":
            skip = False
        elif op == "skip":
            skip = True
        elif op.startswith("n"):
            key = op[0]
            v = op[1:]
            options[key] = v if v else "relu"
        else:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

    act_layer = options.get("n", "gelu")
    num_repeat = int(options["r"])

    block_args = dict(
        block_type=block_type,
        out_chs=int(options["c"]),
        stride=int(options["s"]),
        act_layer=act_layer,
    )

    if block_type == "ir":
        block_args.update(
            dict(
                dw_kernel_size=parse_ksize(options["k"]),
                exp_ratio=float(options["e"]),
                se_ratio=float(options.get("se", 0.0)),
                noskip=skip is False,
            )
        )
    elif block_type == "uir":
        start_kernel_size = parse_ksize(options.get("a", "0"))
        end_kernel_size = parse_ksize(options.get("p", "0"))
        block_args.update(
            dict(
                dw_kernel_size_start=start_kernel_size,
                dw_kernel_size_mid=parse_ksize(options["k"]),
                dw_kernel_size_end=end_kernel_size,
                exp_ratio=float(options["e"]),
                se_ratio=float(options.get("se", 0.0)),
                noskip=skip is False,
            )
        )
    elif block_type == "er":
        block_args.update(
            dict(
                exp_kernel_size=parse_ksize(options["k"]),
                pw_kernel_size=1,
                exp_ratio=float(options["e"]),
                se_ratio=float(options.get("se", 0.0)),
                noskip=skip is False,
            )
        )
    elif block_type in ("mqa", "mha"):
        key_dim_val = int(options.get("d", "64"))
        block_args.update(
            dict(
                num_heads=int(options.get("h", "12")),
                key_dim=key_dim_val,
                value_dim=key_dim_val,
                kv_stride=int(options.get("v", "1")),
                use_cpe=bool(int(options.get("cpe", "0"))),
            )
        )
    return block_args, num_repeat


def decode_arch_def(arch_def):
    arch_args = []
    for _, block_strings in enumerate(arch_def):
        stack_args = []
        for block_str in block_strings:
            ba, rep = decode_block_str(block_str)
            stack_args.extend([deepcopy(ba) for _ in range(rep)])
        arch_args.append(stack_args)
    return arch_args


def convert_arch_def_to_stackwise(arch_def):
    decoded_args = decode_arch_def(arch_def)
    stackwise_params = {
        k: []
        for k in [
            "stackwise_block_types",
            "stackwise_num_blocks",
            "stackwise_num_filters",
            "stackwise_strides",
            "stackwise_act_layers",
            "stackwise_exp_ratios",
            "stackwise_se_ratios",
            "stackwise_dw_kernel_sizes",
            "stackwise_dw_start_kernel_sizes",
            "stackwise_dw_end_kernel_sizes",
            "stackwise_exp_kernel_sizes",
            "stackwise_pw_kernel_sizes",
            "stackwise_num_heads",
            "stackwise_key_dims",
            "stackwise_value_dims",
            "stackwise_kv_strides",
            "stackwise_use_cpe",
        ]
    }
    for stack in decoded_args:
        stackwise_params["stackwise_num_blocks"].append(len(stack))
        current_stack_params = {
            k: [] for k in stackwise_params if k != "stackwise_num_blocks"
        }
        for block in stack:
            current_stack_params["stackwise_block_types"].append(
                block.get("block_type")
            )
            current_stack_params["stackwise_num_filters"].append(
                block.get("out_chs")
            )
            current_stack_params["stackwise_strides"].append(
                block.get("stride")
            )
            current_stack_params["stackwise_act_layers"].append(
                block.get("act_layer")
            )
            current_stack_params["stackwise_exp_ratios"].append(
                block.get("exp_ratio", 0.0)
            )
            current_stack_params["stackwise_se_ratios"].append(
                block.get("se_ratio", 0.0)
            )
            current_stack_params["stackwise_dw_kernel_sizes"].append(
                block.get("dw_kernel_size", block.get("dw_kernel_size_mid", 0))
            )
            current_stack_params["stackwise_dw_start_kernel_sizes"].append(
                block.get("dw_kernel_size_start", 0)
            )
            current_stack_params["stackwise_dw_end_kernel_sizes"].append(
                block.get("dw_kernel_size_end", 0)
            )
            current_stack_params["stackwise_exp_kernel_sizes"].append(
                block.get("exp_kernel_size", 0)
            )
            current_stack_params["stackwise_pw_kernel_sizes"].append(
                block.get("pw_kernel_size", 0)
            )
            current_stack_params["stackwise_num_heads"].append(
                block.get("num_heads", 0)
            )
            current_stack_params["stackwise_key_dims"].append(
                block.get("key_dim", 0)
            )
            current_stack_params["stackwise_value_dims"].append(
                block.get("value_dim", 0)
            )
            current_stack_params["stackwise_kv_strides"].append(
                block.get("kv_stride", 0)
            )
            current_stack_params["stackwise_use_cpe"].append(
                block.get("use_cpe", False)
            )
        for key, value in current_stack_params.items():
            stackwise_params[key].append(value)
    return stackwise_params


class MobileNetV5Builder:
    """Builds a MobileNetV5 model from a decoded architecture definition.

    This class takes a decoded architecture definition and constructs a list of
    network stages, where each stage is a list of blocks. It handles channel
    rounding, stride management, and feature extraction points.

    Args:
        output_stride: int. The desired output stride of the network.
        pad_type: str. The padding type for convolutions.
        round_chs_fn: callable. A function to round the number of channels.
        se_from_exp: bool. If `True`, SE channel reduction is based on the
            expanded channels.
        act_layer: str. The default activation function for blocks.
        norm_layer: str. The default normalization layer for blocks.
        aa_layer: keras.layers.Layer. An optional anti-aliasing layer.
        se_layer: keras.layers.Layer. The Squeeze-and-Excitation layer to use.
        drop_path_rate: float. The stochastic depth rate for the network.
        layer_scale_init_value: float. The initial value for layer scale.
        feature_location: str. Where to extract features from, either
            `"bottleneck"`, `"expansion"`, or `""`.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        output_stride=32,
        pad_type="same",
        round_chs_fn=round_channels,
        se_from_exp=False,
        act_layer="relu",
        norm_layer="batch_norm",
        aa_layer=None,
        se_layer=None,
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        feature_location="",
        data_format=None,
        channel_axis=None,
        dtype=None,
    ):
        self.output_stride = output_stride
        self.pad_type = pad_type
        self.data_format = data_format
        self.channel_axis = channel_axis
        self.round_chs_fn = round_chs_fn
        self.se_from_exp = se_from_exp
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.aa_layer = aa_layer
        self.se_layer = se_layer
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.dtype = dtype
        if feature_location == "depthwise":
            feature_location = "expansion"
        self.feature_location = feature_location
        assert feature_location in ("bottleneck", "expansion", "")
        self.in_chs = None
        self.features = []

    def _make_block(self, ba, block_idx, block_count):
        drop_path_rate = self.drop_path_rate * block_idx / block_count
        bt = ba.pop("block_type")
        ba["filters"] = self.round_chs_fn(ba.pop("out_chs"))
        s2d = ba.get("s2d", 0)
        if s2d > 0:
            ba["filters"] *= 4
        if "expansion_in_chs" in ba and ba["expansion_in_chs"]:
            ba["expansion_in_chs"] = self.round_chs_fn(ba["expansion_in_chs"])
        ba["pad_type"] = self.pad_type
        ba["act_layer"] = (
            ba.get("act_layer")
            if ba.get("act_layer") is not None
            else self.act_layer
        )
        assert ba["act_layer"] is not None
        ba["norm_layer"] = self.norm_layer
        ba["drop_path_rate"] = drop_path_rate
        ba["data_format"] = self.data_format
        ba["channel_axis"] = self.channel_axis
        ba["dtype"] = self.dtype
        if bt in ("ir", "er", "uir", "ds", "dsa"):
            se_ratio = ba.pop("se_ratio", None)
            if se_ratio and self.se_layer is not None:
                if not self.se_from_exp:
                    se_ratio /= ba.get("exp_ratio", 1.0)
                if s2d == 1:
                    se_ratio /= 4
                ba["se_layer"] = lambda channels: self.se_layer(
                    filters=channels,
                    bottleneck_filters=adjust_channels(channels * se_ratio),
                    squeeze_activation=ba["act_layer"],
                    excite_activation="sigmoid",
                    data_format=self.data_format,
                    dtype=self.dtype,
                )
            else:
                ba["se_layer"] = None
        ba.pop("aa_layer", None)
        if bt == "ir":
            padding = 0
            if ba["pad_type"].lower() in ("", "same"):
                kernel_size = ba["dw_kernel_size"]
                if isinstance(kernel_size, (list, tuple)):
                    kernel_size = kernel_size[0]
                padding = (kernel_size - 1) // 2
            block = (
                CondConvResidual(**ba)
                if ba.get("num_experts", 0) > 0
                else InvertedResidualBlock(
                    expansion=ba["exp_ratio"],
                    infilters=self.in_chs,
                    filters=ba["filters"],
                    kernel_size=ba["dw_kernel_size"],
                    stride=ba["stride"],
                    padding=padding,
                    squeeze_excite_ratio=ba.pop("se_ratio", None),
                    activation=ba["act_layer"],
                )
            )
        elif bt == "ds" or bt == "dsa":
            block = DepthwiseConvBlock(
                infilters=self.in_chs,
                filters=ba["filters"],
                kernel_size=ba["dw_kernel_size"],
                stride=ba["stride"],
                squeeze_excite_ratio=ba.pop("se_ratio", None),
                residual=not ba["noskip"],
                dtype=self.dtype,
            )
        elif bt == "er":
            block = EdgeResidual(**ba)
        elif bt == "cn":
            block = ConvBnActBlock(out_chs=ba.pop("filters"), **ba)
        elif bt == "uir":
            block = UniversalInvertedResidual(
                **ba, layer_scale_init_value=self.layer_scale_init_value
            )
        elif bt == "mqa":
            ba.pop("act_layer", None)
            block = MobileAttention(
                **ba,
                use_multi_query=True,
                layer_scale_init_value=self.layer_scale_init_value,
            )
        elif bt == "mha":
            ba.pop("act_layer", None)
            block = MobileAttention(
                **ba, layer_scale_init_value=self.layer_scale_init_value
            )
        else:
            raise ValueError(f"Unknown block type ({bt}) while building model.")
        self.in_chs = ba["filters"]
        return block

    def __call__(self, in_chs, model_block_args):
        self.in_chs = in_chs
        total_block_count = sum([len(x) for x in model_block_args])
        total_block_idx = 0
        current_stride = 2
        current_dilation = 1
        stages = []
        if model_block_args[0][0]["stride"] > 1:
            feature_info = dict(
                module="conv_stem",
                num_chs=in_chs,
                stage=0,
                reduction=current_stride,
            )
            self.features.append(feature_info)
        space2depth = 0
        for stack_idx, stack_args in enumerate(model_block_args):
            blocks = []
            for block_idx, block_args in enumerate(stack_args):
                last_block = block_idx + 1 == len(stack_args)
                in_chs_for_current_block = self.in_chs
                assert block_args["stride"] in (1, 2)
                if block_idx >= 1:
                    block_args["stride"] = 1
                if not space2depth and block_args.pop("s2d", False):
                    assert block_args["stride"] == 1
                    space2depth = 1
                if space2depth > 0:
                    if space2depth == 2 and block_args["stride"] == 2:
                        block_args["stride"] = 1
                        block_args["exp_ratio"] /= 4
                        space2depth = 0
                    else:
                        block_args["s2d"] = space2depth
                next_dilation = current_dilation
                if block_args["stride"] > 1:
                    next_output_stride = current_stride * block_args["stride"]
                    if next_output_stride > self.output_stride:
                        next_dilation = current_dilation * block_args["stride"]
                        block_args["stride"] = 1
                    else:
                        current_stride = next_output_stride
                block_args["dilation"] = current_dilation
                if next_dilation != current_dilation:
                    current_dilation = next_dilation
                block = self._make_block(
                    block_args.copy(), total_block_idx, total_block_count
                )
                blocks.append(block)
                if space2depth == 1:
                    space2depth = 2
                extract_features = False
                if last_block:
                    next_stack_idx = stack_idx + 1
                    extract_features = (
                        next_stack_idx >= len(model_block_args)
                        or model_block_args[next_stack_idx][0]["stride"] > 1
                    )
                if extract_features:
                    num_chs = 0
                    module_name = f"blocks.{stack_idx}.{block_idx}"
                    if self.feature_location == "expansion":
                        bt = block_args.get("block_type")
                        if bt in ["ir", "er", "uir"]:
                            exp_ratio = block_args.get("exp_ratio", 1.0)
                            num_chs = self.round_chs_fn(
                                in_chs_for_current_block * exp_ratio
                            )
                        else:
                            num_chs = in_chs_for_current_block
                    else:
                        num_chs = self.in_chs
                        module_name = f"blocks.{stack_idx}"

                    feature_info = dict(
                        stage=stack_idx + 1,
                        reduction=current_stride,
                        num_chs=num_chs,
                        module=module_name,
                    )
                    self.features.append(feature_info)
                total_block_idx += 1
            stages.append(blocks)
        return stages
