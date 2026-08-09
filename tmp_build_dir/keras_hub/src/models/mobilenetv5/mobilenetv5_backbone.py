import keras
from keras.src import saving

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mobilenet.mobilenet_backbone import SqueezeAndExcite2D
from keras_hub.src.models.mobilenetv5.mobilenetv5_blocks import (
    MobileNetV5MultiScaleFusionAdapter,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_builder import (
    MobileNetV5Builder,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import ConvNormAct
from keras_hub.src.models.mobilenetv5.mobilenetv5_utils import (
    feature_take_indices,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_utils import round_channels
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.MobileNetV5Backbone")
class MobileNetV5Backbone(Backbone):
    """MobileNetV5 backbone network.

    This class represents the backbone of the MobileNetV5 architecture, which
    can be used as a feature extractor for various downstream tasks.

    Args:
        stackwise_block_types: list of list of strings. The block type for each
            block in each stack.
        stackwise_num_blocks: list of ints. The number of blocks for each
            stack.
        stackwise_num_filters: list of list of ints. The number of filters for
            each block in each stack.
        stackwise_strides: list of list of ints. The stride for each block in
            each stack.
        stackwise_act_layers: list of list of strings. The activation function
            for each block in each stack.
        stackwise_exp_ratios: list of list of floats. The expansion ratio for
            each block in each stack.
        stackwise_se_ratios: list of list of floats. The SE ratio for each
            block in each stack.
        stackwise_dw_kernel_sizes: list of list of ints. The depthwise kernel
            size for each block in each stack.
        stackwise_dw_start_kernel_sizes: list of list of ints. The start
            depthwise kernel size for each `uir` block in each stack.
        stackwise_dw_end_kernel_sizes: list of list of ints. The end depthwise
            kernel size for each `uir` block in each stack.
        stackwise_exp_kernel_sizes: list of list of ints. The expansion kernel
            size for each `er` block in each stack.
        stackwise_pw_kernel_sizes: list of list of ints. The pointwise kernel
            size for each `er` block in each stack.
        stackwise_num_heads: list of list of ints. The number of heads for each
            `mqa` or `mha` block in each stack.
        stackwise_key_dims: list of list of ints. The key dimension for each
            `mqa` or `mha` block in each stack.
        stackwise_value_dims: list of list of ints. The value dimension for each
            `mqa` or `mha` block in each stack.
        stackwise_kv_strides: list of list of ints. The key-value stride for
            each `mqa` or `mha` block in each stack.
        stackwise_use_cpe: list of list of bools. Whether to use conditional
            position encoding for each `mqa` or `mha` block in each stack.
        filters: int. The number of input channels.
        stem_size: int. The number of channels in the stem convolution.
        stem_bias: bool. If `True`, a bias term is used in the stem
            convolution.
        fix_stem: bool. If `True`, the stem size is not rounded.
        num_features: int. The number of output features, used when `use_msfa`
            is `True`.
        pad_type: str. The padding type for convolutions.
        use_msfa: bool. If `True`, the Multi-Scale Fusion Adapter is used.
        msfa_indices: tuple. The indices of the feature maps to be used by the
            MSFA.
        msfa_output_resolution: int. The output resolution of the MSFA.
        act_layer: str. The activation function to use.
        norm_layer: str. The normalization layer to use.
        se_layer: keras.layers.Layer. The Squeeze-and-Excitation layer to use.
        se_from_exp: bool. If `True`, SE channel reduction is based on the
            expanded channels.
        round_chs_fn: callable. A function to round the number of channels.
        drop_path_rate: float. The stochastic depth rate.
        layer_scale_init_value: float. The initial value for layer scale.
        image_shape: tuple. The shape of the input image. Defaults to
            `(None, None, 3)`.
        data_format: str, The data format of the image channels. Can be either
            `"channels_first"` or `"channels_last"`. If `None` is specified,
            it will use the `image_data_format` value found in your Keras
            config file at `~/.keras/keras.json`. Defaults to `None`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights. Defaults to `None`.

    Example:
    ```python
    import keras
    from keras_hub.models import MobileNetV5Backbone

    # Randomly initialized backbone with a custom config.
    model_args = {
        "stackwise_block_types": [["er"], ["uir", "uir"]],
        "stackwise_num_blocks": [1, 2],
        "stackwise_num_filters": [[24], [48, 48]],
        "stackwise_strides": [[2], [2, 1]],
        "stackwise_act_layers": [["relu"], ["relu", "relu"]],
        "stackwise_exp_ratios": [[4.0], [6.0, 6.0]],
        "stackwise_se_ratios": [[0.0], [0.0, 0.0]],
        "stackwise_dw_kernel_sizes": [[0], [5, 5]],
        "stackwise_dw_start_kernel_sizes": [[0], [0, 0]],
        "stackwise_dw_end_kernel_sizes": [[0], [0, 0]],
        "stackwise_exp_kernel_sizes": [[3], [0, 0]],
        "stackwise_pw_kernel_sizes": [[1], [0, 0]],
        "stackwise_num_heads": [[0], [0, 0]],
        "stackwise_key_dims": [[0], [0, 0]],
        "stackwise_value_dims": [[0], [0, 0]],
        "stackwise_kv_strides": [[0], [0, 0]],
        "stackwise_use_cpe": [[False], [False, False]],
        "use_msfa": False,
    }
    model = MobileNetV5Backbone(**model_args)
    input_data = keras.ops.ones((1, 224, 224, 3))
    output = model(input_data)

    # Load the backbone from a preset and run a prediction.
    backbone = MobileNetV5Backbone.from_preset("mobilenetv5_300m_gemma3n")

    # Expected output shape = (1, 16, 16, 2048).
    outputs = backbone.predict(keras.ops.ones((1, 224, 224, 3)))
    ```
    """

    def __init__(
        self,
        stackwise_block_types,
        stackwise_num_blocks,
        stackwise_num_filters,
        stackwise_strides,
        stackwise_act_layers,
        stackwise_exp_ratios,
        stackwise_se_ratios,
        stackwise_dw_kernel_sizes,
        stackwise_dw_start_kernel_sizes,
        stackwise_dw_end_kernel_sizes,
        stackwise_exp_kernel_sizes,
        stackwise_pw_kernel_sizes,
        stackwise_num_heads,
        stackwise_key_dims,
        stackwise_value_dims,
        stackwise_kv_strides,
        stackwise_use_cpe,
        filters=3,
        stem_size=16,
        stem_bias=True,
        fix_stem=False,
        num_features=2048,
        pad_type="same",
        use_msfa=True,
        msfa_indices=(-2, -1),
        msfa_output_resolution=16,
        act_layer="gelu",
        norm_layer="rms_norm",
        se_layer=SqueezeAndExcite2D,
        se_from_exp=True,
        round_chs_fn=round_channels,
        drop_path_rate=0.0,
        layer_scale_init_value=None,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        block_args = []
        for i in range(len(stackwise_num_blocks)):
            stack_args = []
            for j in range(stackwise_num_blocks[i]):
                block_type = stackwise_block_types[i][j]
                args = {
                    "block_type": block_type,
                    "out_chs": stackwise_num_filters[i][j],
                    "stride": stackwise_strides[i][j],
                    "act_layer": stackwise_act_layers[i][j],
                }
                if block_type == "ir":
                    args.update(
                        {
                            "dw_kernel_size": stackwise_dw_kernel_sizes[i][j],
                            "exp_ratio": stackwise_exp_ratios[i][j],
                            "se_ratio": stackwise_se_ratios[i][j],
                        }
                    )
                elif block_type == "uir":
                    args.update(
                        {
                            "dw_kernel_size_mid": stackwise_dw_kernel_sizes[i][
                                j
                            ],
                            "dw_kernel_size_start": stackwise_dw_start_kernel_sizes[  # noqa: E501
                                i
                            ][j],
                            "dw_kernel_size_end": stackwise_dw_end_kernel_sizes[
                                i
                            ][j],
                            "exp_ratio": stackwise_exp_ratios[i][j],
                            "se_ratio": stackwise_se_ratios[i][j],
                        }
                    )
                elif block_type == "er":
                    args.update(
                        {
                            "exp_kernel_size": stackwise_exp_kernel_sizes[i][j],
                            "pw_kernel_size": stackwise_pw_kernel_sizes[i][j],
                            "exp_ratio": stackwise_exp_ratios[i][j],
                            "se_ratio": stackwise_se_ratios[i][j],
                        }
                    )
                elif block_type in ("mqa", "mha"):
                    args.update(
                        {
                            "num_heads": stackwise_num_heads[i][j],
                            "key_dim": stackwise_key_dims[i][j],
                            "value_dim": stackwise_value_dims[i][j],
                            "kv_stride": stackwise_kv_strides[i][j],
                            "use_cpe": stackwise_use_cpe[i][j],
                        }
                    )
                stack_args.append(args)
            block_args.append(stack_args)

        # === Layers ===
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        conv_stem = ConvNormAct(
            stem_size,
            kernel_size=3,
            stride=2,
            pad_type=pad_type,
            bias=stem_bias,
            norm_layer=norm_layer,
            act_layer=act_layer,
            name="conv_stem",
            data_format=data_format,
            channel_axis=channel_axis,
            dtype=dtype,
        )
        builder = MobileNetV5Builder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            data_format=data_format,
            channel_axis=channel_axis,
            dtype=dtype,
        )
        blocks = builder(stem_size, block_args)
        feature_info = builder.features
        msfa = None
        if use_msfa:
            msfa_indices_calc, _ = feature_take_indices(
                len(feature_info), msfa_indices
            )
            msfa_in_chs = [
                feature_info[mi]["num_chs"] for mi in msfa_indices_calc
            ]
            msfa = MobileNetV5MultiScaleFusionAdapter(
                in_chs=msfa_in_chs,
                filters=num_features,
                output_resolution=msfa_output_resolution,
                norm_layer=norm_layer,
                act_layer=act_layer,
                name="msfa",
                channel_axis=channel_axis,
                data_format=data_format,
                dtype=dtype,
            )

        # === Functional Model ===
        image_input = keras.layers.Input(shape=image_shape)
        x = conv_stem(image_input)
        if use_msfa:
            intermediates = []
            feat_idx = 0
            if feat_idx in msfa_indices_calc:
                intermediates.append(x)

            for stage in blocks:
                for block in stage:
                    x = block(x)
                feat_idx += 1
                if feat_idx in msfa_indices_calc:
                    intermediates.append(x)
            x = msfa(intermediates)
        else:
            for stage in blocks:
                for block in stage:
                    x = block(x)

        super().__init__(inputs=image_input, outputs=x, dtype=dtype, **kwargs)

        # === Config ===
        self.stackwise_block_types = stackwise_block_types
        self.stackwise_num_blocks = stackwise_num_blocks
        self.stackwise_num_filters = stackwise_num_filters
        self.stackwise_strides = stackwise_strides
        self.stackwise_act_layers = stackwise_act_layers
        self.stackwise_exp_ratios = stackwise_exp_ratios
        self.stackwise_se_ratios = stackwise_se_ratios
        self.stackwise_dw_kernel_sizes = stackwise_dw_kernel_sizes
        self.stackwise_dw_start_kernel_sizes = stackwise_dw_start_kernel_sizes
        self.stackwise_dw_end_kernel_sizes = stackwise_dw_end_kernel_sizes
        self.stackwise_exp_kernel_sizes = stackwise_exp_kernel_sizes
        self.stackwise_pw_kernel_sizes = stackwise_pw_kernel_sizes
        self.stackwise_num_heads = stackwise_num_heads
        self.stackwise_key_dims = stackwise_key_dims
        self.stackwise_value_dims = stackwise_value_dims
        self.stackwise_kv_strides = stackwise_kv_strides
        self.stackwise_use_cpe = stackwise_use_cpe
        self.filters = filters
        self.stem_size = stem_size
        self.stem_bias = stem_bias
        self.fix_stem = fix_stem
        self.num_features = num_features
        self.pad_type = pad_type
        self.use_msfa = use_msfa
        self.msfa_indices = msfa_indices
        self.msfa_output_resolution = msfa_output_resolution
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.se_layer = se_layer
        self.se_from_exp = se_from_exp
        self.round_chs_fn = round_chs_fn
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.image_shape = image_shape
        self.data_format = data_format
        self.channel_axis = channel_axis

    def get_config(self):
        config = {
            "stackwise_block_types": self.stackwise_block_types,
            "stackwise_num_blocks": self.stackwise_num_blocks,
            "stackwise_num_filters": self.stackwise_num_filters,
            "stackwise_strides": self.stackwise_strides,
            "stackwise_act_layers": self.stackwise_act_layers,
            "stackwise_exp_ratios": self.stackwise_exp_ratios,
            "stackwise_se_ratios": self.stackwise_se_ratios,
            "stackwise_dw_kernel_sizes": self.stackwise_dw_kernel_sizes,
            "stackwise_dw_start_kernel_sizes": self.stackwise_dw_start_kernel_sizes,  # noqa: E501
            "stackwise_dw_end_kernel_sizes": self.stackwise_dw_end_kernel_sizes,
            "stackwise_exp_kernel_sizes": self.stackwise_exp_kernel_sizes,
            "stackwise_pw_kernel_sizes": self.stackwise_pw_kernel_sizes,
            "stackwise_num_heads": self.stackwise_num_heads,
            "stackwise_key_dims": self.stackwise_key_dims,
            "stackwise_value_dims": self.stackwise_value_dims,
            "stackwise_kv_strides": self.stackwise_kv_strides,
            "stackwise_use_cpe": self.stackwise_use_cpe,
            "filters": self.filters,
            "stem_size": self.stem_size,
            "stem_bias": self.stem_bias,
            "fix_stem": self.fix_stem,
            "num_features": self.num_features,
            "pad_type": self.pad_type,
            "use_msfa": self.use_msfa,
            "msfa_indices": self.msfa_indices,
            "msfa_output_resolution": self.msfa_output_resolution,
            "act_layer": self.act_layer,
            "norm_layer": self.norm_layer,
            "se_from_exp": self.se_from_exp,
            "drop_path_rate": self.drop_path_rate,
            "layer_scale_init_value": self.layer_scale_init_value,
            "image_shape": self.image_shape,
            "data_format": self.data_format,
        }
        if self.round_chs_fn is not round_channels:
            config["round_chs_fn"] = saving.serialize_keras_object(
                self.round_chs_fn
            )
        if self.se_layer is not SqueezeAndExcite2D:
            config["se_layer"] = saving.serialize_keras_object(self.se_layer)
        return config

    @classmethod
    def from_config(cls, config):
        if "round_chs_fn" in config:
            config["round_chs_fn"] = saving.deserialize_keras_object(
                config["round_chs_fn"]
            )
        if "se_layer" in config:
            config["se_layer"] = saving.deserialize_keras_object(
                config["se_layer"]
            )
        return cls(**config)
