import keras

from keras_hub.src.models.d_fine.d_fine_encoder import DFineEncoder
from keras_hub.src.models.d_fine.d_fine_layers import DFineConvNormLayer
from keras_hub.src.models.d_fine.d_fine_layers import (
    DFineFeatureAggregationBlock,
)
from keras_hub.src.models.d_fine.d_fine_layers import DFineSCDown


class DFineHybridEncoder(keras.layers.Layer):
    """Hybrid encoder for the D-FINE model.

    This layer sits between the HGNetV2 backbone (`HGNetV2Backbone`) and the
    main `DFineDecoder`. It takes multi-scale feature maps from the backbone,
    optionally refines them with transformer-based `DFineEncoder` layers, and
    then fuses them using a Feature Pyramid Network (FPN) top-down pathway and a
    Path Aggregation Network (PAN) bottom-up pathway. The resulting enriched
    feature maps serve as the key and value inputs for the decoder's
    cross-attention mechanism.

    Args:
        encoder_in_channels: list of int, Input channel dimensions for each
            feature level from the backbone.
        feat_strides: list of int, Stride values for each feature level,
            indicating the downsampling factor relative to the input image.
        encoder_hidden_dim: int, Hidden dimension size used throughout the
            encoder for feature projection and attention computation.
        encode_proj_layers: list of int, Indices of feature levels to apply
            transformer encoding to. Not all levels need transformer
            processing.
        positional_encoding_temperature: float, Temperature parameter for
            sinusoidal positional embeddings used in transformer attention.
        eval_size: tuple or None, Fixed evaluation size `(height, width)` for
            consistent positional embeddings during inference. If `None`,
            dynamic sizing is used.
        normalize_before: bool, Whether to apply layer normalization before
            attention and feed-forward operations in transformer layers.
        num_attention_heads: int, Number of attention heads in multi-head
            attention mechanisms within transformer layers.
        dropout: float, Dropout probability applied to attention weights and
            feed-forward networks for regularization.
        layer_norm_eps: float, Small epsilon value for numerical stability in
            layer normalization operations.
        encoder_activation_function: str, Activation function used in
            transformer feed-forward networks (e.g., `"relu"`, `"gelu"`).
        activation_dropout: float, Dropout probability specifically applied to
            activation functions in feed-forward networks.
        encoder_ffn_dim: int, Hidden dimension size for feed-forward networks
            within transformer layers.
        num_encoder_layers: int, Number of transformer encoder layers to apply
            at each selected feature level.
        batch_norm_eps: float, Small epsilon value for numerical stability in
            batch normalization operations used in components.
        hidden_expansion: float, Expansion factor for hidden dimensions in
            `DFineFeatureAggregationBlock` blocks used in FPN and PAN pathways.
        depth_multiplier: float, Depth multiplier for scaling the number of
            blocks in `DFineFeatureAggregationBlock` modules.
        kernel_initializer: str or Initializer, optional, Initializer for
            the kernel weights of each layer. Defaults to
            `"glorot_uniform"`.
        bias_initializer: str or Initializer, optional, Initializer for
            the bias weights of each layer. Defaults to
            `"zeros"`.
        channel_axis: int, optional, The channel axis. Defaults to `None`.
        data_format: str, optional, The data format. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        encoder_in_channels,
        feat_strides,
        encoder_hidden_dim,
        encode_proj_layers,
        positional_encoding_temperature,
        eval_size,
        normalize_before,
        num_attention_heads,
        dropout,
        layer_norm_eps,
        encoder_activation_function,
        activation_dropout,
        encoder_ffn_dim,
        num_encoder_layers,
        batch_norm_eps,
        hidden_expansion,
        depth_multiplier,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        channel_axis=None,
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)

        self.encoder_in_channels = encoder_in_channels
        self.num_fpn_stages = len(self.encoder_in_channels) - 1
        self.feat_strides = feat_strides
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encode_proj_layers = encode_proj_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.eval_size = eval_size
        self.out_channels = [
            self.encoder_hidden_dim for _ in self.encoder_in_channels
        ]
        self.out_strides = self.feat_strides
        self.depth_multiplier = depth_multiplier
        self.num_encoder_layers = num_encoder_layers
        self.normalize_before = normalize_before
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps
        self.encoder_activation_function = encoder_activation_function
        self.activation_dropout_rate = activation_dropout
        self.encoder_ffn_dim = encoder_ffn_dim
        self.batch_norm_eps = batch_norm_eps
        self.hidden_expansion = hidden_expansion
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.channel_axis = channel_axis
        self.data_format = data_format

        self.encoder = [
            DFineEncoder(
                normalize_before=self.normalize_before,
                encoder_hidden_dim=self.encoder_hidden_dim,
                num_attention_heads=self.num_attention_heads,
                dropout=self.dropout_rate,
                layer_norm_eps=self.layer_norm_eps,
                encoder_activation_function=self.encoder_activation_function,
                activation_dropout=self.activation_dropout_rate,
                encoder_ffn_dim=self.encoder_ffn_dim,
                dtype=self.dtype_policy,
                num_encoder_layers=self.num_encoder_layers,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                name=f"d_fine_encoder_{i}",
            )
            for i in range(len(self.encode_proj_layers))
        ]

        self.lateral_convs = []
        self.fpn_blocks = []
        for i in range(len(self.encoder_in_channels) - 1, 0, -1):
            lateral_layer = DFineConvNormLayer(
                filters=self.encoder_hidden_dim,
                kernel_size=1,
                batch_norm_eps=self.batch_norm_eps,
                stride=1,
                groups=1,
                padding=0,
                activation_function=None,
                dtype=self.dtype_policy,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                channel_axis=self.channel_axis,
                name=f"lateral_conv_{i}",
            )
            self.lateral_convs.append(lateral_layer)
            num_blocks = round(3 * self.depth_multiplier)
            fpn_layer = DFineFeatureAggregationBlock(
                encoder_hidden_dim=self.encoder_hidden_dim,
                hidden_expansion=self.hidden_expansion,
                batch_norm_eps=self.batch_norm_eps,
                activation_function="silu",
                num_blocks=num_blocks,
                dtype=self.dtype_policy,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                channel_axis=self.channel_axis,
                name=f"fpn_block_{i}",
            )
            self.fpn_blocks.append(fpn_layer)

        self.downsample_convs = []
        self.pan_blocks = []
        for i in range(len(self.encoder_in_channels) - 1):
            num_blocks = round(3 * self.depth_multiplier)
            self.downsample_convs.append(
                DFineSCDown(
                    encoder_hidden_dim=self.encoder_hidden_dim,
                    batch_norm_eps=self.batch_norm_eps,
                    kernel_size=3,
                    stride=2,
                    dtype=self.dtype_policy,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    channel_axis=self.channel_axis,
                    name=f"downsample_conv_{i}",
                )
            )
            self.pan_blocks.append(
                DFineFeatureAggregationBlock(
                    encoder_hidden_dim=self.encoder_hidden_dim,
                    hidden_expansion=self.hidden_expansion,
                    batch_norm_eps=self.batch_norm_eps,
                    activation_function="silu",
                    num_blocks=num_blocks,
                    dtype=self.dtype_policy,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    channel_axis=self.channel_axis,
                    name=f"pan_block_{i}",
                )
            )

        self.upsample = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest",
            dtype=self.dtype_policy,
            data_format=self.data_format,
            name="upsample",
        )
        self.identity = keras.layers.Identity(
            dtype=self.dtype_policy, name="identity"
        )

    def build(self, input_shape):
        inputs_embeds_shapes = input_shape
        # Encoder layers.
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_map_shape = inputs_embeds_shapes[enc_ind]
                if self.data_format == "channels_last":
                    batch_s, h_s, w_s, c_s = feature_map_shape
                else:  # channels_first
                    batch_s, c_s, h_s, w_s = feature_map_shape
                if h_s is not None and w_s is not None:
                    seq_len_for_this_encoder = h_s * w_s
                else:
                    seq_len_for_this_encoder = None
                encoder_input_shape = (batch_s, seq_len_for_this_encoder, c_s)
                self.encoder[i].build(encoder_input_shape)
        # FPN and PAN pathways.
        # FPN (Top-down pathway).
        fpn_feature_maps_shapes = [inputs_embeds_shapes[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            lateral_conv.build(fpn_feature_maps_shapes[-1])
            shape_after_lateral_conv = lateral_conv.compute_output_shape(
                fpn_feature_maps_shapes[-1]
            )
            if self.data_format == "channels_last":
                batch_s, orig_h, orig_w, c = shape_after_lateral_conv
                target_h = orig_h * 2 if orig_h is not None else None
                target_w = orig_w * 2 if orig_w is not None else None
                shape_after_resize = (batch_s, target_h, target_w, c)
            else:
                batch_s, c, orig_h, orig_w = shape_after_lateral_conv
                target_h = orig_h * 2 if orig_h is not None else None
                target_w = orig_w * 2 if orig_w is not None else None
                shape_after_resize = (batch_s, c, target_h, target_w)
            backbone_feature_map_k_shape = inputs_embeds_shapes[
                self.num_fpn_stages - idx - 1
            ]
            shape_after_concat_fpn = list(shape_after_resize)
            shape_after_concat_fpn[self.channel_axis] += (
                backbone_feature_map_k_shape[self.channel_axis]
            )
            shape_after_concat_fpn = tuple(shape_after_concat_fpn)
            fpn_block.build(shape_after_concat_fpn)
            fpn_feature_maps_shapes.append(
                fpn_block.compute_output_shape(shape_after_concat_fpn)
            )
        # PAN (Bottom-up pathway).
        reversed_fpn_feature_maps_shapes = fpn_feature_maps_shapes[::-1]
        pan_feature_maps_shapes = [reversed_fpn_feature_maps_shapes[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks)
        ):
            downsample_conv.build(pan_feature_maps_shapes[-1])
            shape_after_downsample = downsample_conv.compute_output_shape(
                pan_feature_maps_shapes[-1]
            )
            fpn_shape = reversed_fpn_feature_maps_shapes[idx + 1]
            concat_shape = list(shape_after_downsample)
            concat_shape[self.channel_axis] += fpn_shape[self.channel_axis]
            pan_block.build(tuple(concat_shape))
            pan_feature_maps_shapes.append(
                pan_block.compute_output_shape(tuple(concat_shape))
            )
        super().build(input_shape)

    def call(
        self,
        inputs_embeds,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        training=None,
    ):
        hidden_states = [keras.ops.convert_to_tensor(t) for t in inputs_embeds]

        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        encoder_states_tuple = () if output_hidden_states else None
        all_attentions_tuple = () if output_attentions else None

        processed_maps = {}
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                current_feature_map = hidden_states[enc_ind]
                if output_hidden_states:
                    encoder_states_tuple = encoder_states_tuple + (
                        self.identity(current_feature_map),
                    )

                batch_size = keras.ops.shape(current_feature_map)[0]
                if self.data_format == "channels_last":
                    height = keras.ops.shape(current_feature_map)[1]
                    width = keras.ops.shape(current_feature_map)[2]
                    channels = keras.ops.shape(current_feature_map)[-1]
                    src_flatten = keras.ops.reshape(
                        current_feature_map,
                        (batch_size, height * width, channels),
                    )
                else:
                    channels = keras.ops.shape(current_feature_map)[1]
                    height = keras.ops.shape(current_feature_map)[2]
                    width = keras.ops.shape(current_feature_map)[3]

                    transposed_map = keras.ops.transpose(
                        current_feature_map, (0, 2, 3, 1)
                    )
                    src_flatten = keras.ops.reshape(
                        transposed_map,
                        (batch_size, height * width, channels),
                    )

                pos_embed = None
                if training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        width,
                        height,
                        self.encoder_hidden_dim,
                        self.positional_encoding_temperature,
                        dtype=self.compute_dtype,
                    )
                encoder_output = self.encoder[i](
                    src=src_flatten,
                    src_mask=attention_mask,
                    pos_embed=pos_embed,
                    output_attentions=output_attentions,
                    training=training,
                )
                if output_attentions:
                    processed_feature_map, layer_attentions = encoder_output
                else:
                    processed_feature_map, layer_attentions = (
                        encoder_output,
                        None,
                    )

                if self.data_format == "channels_last":
                    processed_maps[enc_ind] = keras.ops.reshape(
                        processed_feature_map,
                        (batch_size, height, width, self.encoder_hidden_dim),
                    )
                else:
                    reshaped_map = keras.ops.reshape(
                        processed_feature_map,
                        (batch_size, height, width, self.encoder_hidden_dim),
                    )
                    processed_maps[enc_ind] = keras.ops.transpose(
                        reshaped_map, (0, 3, 1, 2)
                    )

                if output_attentions and layer_attentions is not None:
                    all_attentions_tuple = all_attentions_tuple + (
                        layer_attentions,
                    )

        processed_hidden_states = []
        for i in range(len(hidden_states)):
            if i in processed_maps:
                processed_hidden_states.append(processed_maps[i])
            else:
                processed_hidden_states.append(hidden_states[i])
        if self.num_encoder_layers > 0:
            if output_hidden_states:
                encoder_states_tuple = encoder_states_tuple + (
                    self.identity(
                        processed_hidden_states[self.encode_proj_layers[-1]]
                    ),
                )
        else:
            processed_hidden_states = hidden_states
        fpn_inter_outputs = []
        y = processed_hidden_states[-1]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            backbone_feature_map_k = processed_hidden_states[
                self.num_fpn_stages - idx - 1
            ]
            y_lateral = lateral_conv(y, training=training)
            fpn_inter_outputs.append(y_lateral)
            y_upsampled = self.upsample(y_lateral, training=training)
            fused_feature_map_k = keras.ops.concatenate(
                [y_upsampled, backbone_feature_map_k],
                axis=self.channel_axis,
            )
            y = fpn_block(fused_feature_map_k, training=training)
        fpn_feature_maps = fpn_inter_outputs + [y]

        fpn_feature_maps = fpn_feature_maps[::-1]

        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks)
        ):
            top_pan_feature_map_k = pan_feature_maps[-1]
            fpn_feature_map_k = fpn_feature_maps[idx + 1]

            downsampled_feature_map_k = downsample_conv(
                top_pan_feature_map_k, training=training
            )
            fused_feature_map_k = keras.ops.concatenate(
                [downsampled_feature_map_k, fpn_feature_map_k],
                axis=self.channel_axis,
            )
            new_pan_feature_map_k = pan_block(
                fused_feature_map_k, training=training
            )
            pan_feature_maps.append(new_pan_feature_map_k)

        return tuple(
            v
            for v in [
                pan_feature_maps,
                encoder_states_tuple if output_hidden_states else None,
                all_attentions_tuple if output_attentions else None,
            ]
            if v is not None
        )

    @staticmethod
    def build_2d_sincos_position_embedding(
        width,
        height,
        embedding_dim=256,
        temperature=10000.0,
        dtype="float32",
    ):
        grid_w = keras.ops.arange(width, dtype=dtype)
        grid_h = keras.ops.arange(height, dtype=dtype)
        grid_w, grid_h = keras.ops.meshgrid(grid_w, grid_h, indexing="ij")
        if embedding_dim % 4 != 0:
            raise ValueError(
                "Embed dimension must be divisible by 4 for 2D sin-cos position"
                " embedding"
            )
        pos_dim = embedding_dim // 4
        omega = keras.ops.arange(pos_dim, dtype=dtype) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = keras.ops.matmul(
            keras.ops.reshape(grid_w, (-1, 1)),
            keras.ops.reshape(omega, (1, -1)),
        )
        out_h = keras.ops.matmul(
            keras.ops.reshape(grid_h, (-1, 1)),
            keras.ops.reshape(omega, (1, -1)),
        )

        concatenated_embeds = keras.ops.concatenate(
            [
                keras.ops.sin(out_w),
                keras.ops.cos(out_w),
                keras.ops.sin(out_h),
                keras.ops.cos(out_h),
            ],
            axis=1,
        )
        return keras.ops.expand_dims(concatenated_embeds, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_in_channels": self.encoder_in_channels,
                "feat_strides": self.feat_strides,
                "encoder_hidden_dim": self.encoder_hidden_dim,
                "encode_proj_layers": self.encode_proj_layers,
                "positional_encoding_temperature": self.positional_encoding_temperature,  # noqa: E501
                "eval_size": self.eval_size,
                "normalize_before": self.normalize_before,
                "num_attention_heads": self.num_attention_heads,
                "dropout": self.dropout_rate,
                "layer_norm_eps": self.layer_norm_eps,
                "encoder_activation_function": self.encoder_activation_function,
                "activation_dropout": self.activation_dropout_rate,
                "encoder_ffn_dim": self.encoder_ffn_dim,
                "num_encoder_layers": self.num_encoder_layers,
                "batch_norm_eps": self.batch_norm_eps,
                "hidden_expansion": self.hidden_expansion,
                "depth_multiplier": self.depth_multiplier,
                "kernel_initializer": self.kernel_initializer,
                "bias_initializer": self.bias_initializer,
                "channel_axis": self.channel_axis,
                "data_format": self.data_format,
            }
        )
        return config

    def compute_output_spec(
        self,
        inputs_embeds,
        attention_mask_spec=None,
        output_attentions=None,
        output_hidden_states=None,
        training=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )
        hidden_states_specs = list(inputs_embeds)
        encoder_states_tuple_specs = () if output_hidden_states else None
        all_attentions_tuple_specs = () if output_attentions else None
        processed_maps_specs = {}
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                current_feature_map_spec = hidden_states_specs[enc_ind]
                if output_hidden_states:
                    encoder_states_tuple_specs += (
                        self.identity(current_feature_map_spec),
                    )
                if self.data_format == "channels_last":
                    batch_size, h, w, c = current_feature_map_spec.shape
                else:
                    batch_size, c, h, w = current_feature_map_spec.shape
                seq_len = h * w if h is not None and w is not None else None
                src_flatten_spec = keras.KerasTensor(
                    (batch_size, seq_len, c), dtype=self.compute_dtype
                )
                pos_embed_spec = keras.KerasTensor(
                    (batch_size, seq_len, self.encoder_hidden_dim),
                    dtype=self.compute_dtype,
                )
                encoder_output_spec = self.encoder[i].compute_output_spec(
                    src=src_flatten_spec,
                    src_mask=attention_mask_spec,
                    pos_embed=pos_embed_spec,
                    output_attentions=output_attentions,
                )
                if output_attentions:
                    _, layer_attentions_spec = encoder_output_spec
                    all_attentions_tuple_specs += (layer_attentions_spec,)
                if self.data_format == "channels_last":
                    processed_maps_specs[enc_ind] = keras.KerasTensor(
                        (batch_size, h, w, self.encoder_hidden_dim),
                        dtype=self.compute_dtype,
                    )
                else:
                    processed_maps_specs[enc_ind] = keras.KerasTensor(
                        (batch_size, self.encoder_hidden_dim, h, w),
                        dtype=self.compute_dtype,
                    )
        processed_hidden_states_specs = []
        for i in range(len(hidden_states_specs)):
            if i in processed_maps_specs:
                processed_hidden_states_specs.append(processed_maps_specs[i])
            else:
                processed_hidden_states_specs.append(hidden_states_specs[i])
        if self.num_encoder_layers > 0:
            if output_hidden_states:
                encoder_states_tuple_specs += (
                    self.identity(
                        processed_hidden_states_specs[
                            self.encode_proj_layers[-1]
                        ]
                    ),
                )
        else:
            processed_hidden_states_specs = hidden_states_specs
        fpn_inter_outputs_specs = []
        y_spec = processed_hidden_states_specs[-1]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs, self.fpn_blocks)
        ):
            backbone_feature_map_k_spec = processed_hidden_states_specs[
                self.num_fpn_stages - idx - 1
            ]
            y_lateral_spec = keras.KerasTensor(
                lateral_conv.compute_output_shape(y_spec.shape),
                dtype=self.compute_dtype,
            )
            fpn_inter_outputs_specs.append(y_lateral_spec)
            y_upsampled_spec = keras.KerasTensor(
                self.upsample.compute_output_shape(y_lateral_spec.shape),
                dtype=self.compute_dtype,
            )
            concat_shape = list(y_upsampled_spec.shape)
            concat_shape[self.channel_axis] += (
                backbone_feature_map_k_spec.shape[self.channel_axis]
            )
            y_spec = keras.KerasTensor(
                fpn_block.compute_output_shape(tuple(concat_shape)),
                dtype=self.compute_dtype,
            )
        fpn_feature_maps_specs = fpn_inter_outputs_specs + [y_spec]
        fpn_feature_maps_specs = fpn_feature_maps_specs[::-1]
        pan_feature_maps_specs = [fpn_feature_maps_specs[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs, self.pan_blocks)
        ):
            top_pan_feature_map_k_spec = pan_feature_maps_specs[-1]
            fpn_feature_map_k_spec = fpn_feature_maps_specs[idx + 1]
            downsampled_feature_map_k_spec = keras.KerasTensor(
                downsample_conv.compute_output_shape(
                    top_pan_feature_map_k_spec.shape
                ),
                dtype=self.compute_dtype,
            )
            concat_shape = list(downsampled_feature_map_k_spec.shape)
            concat_shape[self.channel_axis] += fpn_feature_map_k_spec.shape[
                self.channel_axis
            ]
            new_pan_feature_map_k_spec = keras.KerasTensor(
                pan_block.compute_output_shape(tuple(concat_shape)),
                dtype=self.compute_dtype,
            )
            pan_feature_maps_specs.append(new_pan_feature_map_k_spec)
        outputs = [
            tuple(pan_feature_maps_specs),
        ]
        if output_hidden_states:
            outputs.append(encoder_states_tuple_specs)
        if output_attentions:
            outputs.append(all_attentions_tuple_specs)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
