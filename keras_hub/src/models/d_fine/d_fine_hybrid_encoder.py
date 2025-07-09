import keras

from keras_hub.src.models.d_fine.d_fine_encoder import DFineEncoder
from keras_hub.src.models.d_fine.d_fine_layers import DFineConvNormLayer
from keras_hub.src.models.d_fine.d_fine_layers import DFineRepNCSPELAN4
from keras_hub.src.models.d_fine.d_fine_layers import DFineSCDown


@keras.saving.register_keras_serializable(package="keras_hub")
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
        encoder_layers: int, Number of transformer encoder layers to apply at
            each selected feature level.
        batch_norm_eps: float, Small epsilon value for numerical stability in
            batch normalization operations used in components.
        hidden_expansion: float, Expansion factor for hidden dimensions in
            `DFineRepNCSPELAN4` blocks used in FPN and PAN pathways.
        depth_mult: float, Depth multiplier for scaling the number of blocks
            in `DFineRepNCSPELAN4` modules.
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
        encoder_layers,
        batch_norm_eps,
        hidden_expansion,
        depth_mult,
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        self.depth_mult = depth_mult
        self.encoder_layers_count = encoder_layers
        self.normalize_before = normalize_before
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout
        self.layer_norm_eps = layer_norm_eps
        self.encoder_activation_function = encoder_activation_function
        self.activation_dropout_rate = activation_dropout
        self.encoder_ffn_dim = encoder_ffn_dim
        self.batch_norm_eps = batch_norm_eps
        self.hidden_expansion = hidden_expansion

        self.encoder_list = [
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
                encoder_layers=self.encoder_layers_count,
                name=f"d_fine_encoder_{i}",
            )
            for i in range(len(self.encode_proj_layers))
        ]

        self.lateral_convs_list = []
        self.fpn_blocks_list = []
        for i in range(len(self.encoder_in_channels) - 1, 0, -1):
            lateral_layer = DFineConvNormLayer(
                in_channels=self.encoder_hidden_dim,
                out_channels=self.encoder_hidden_dim,
                kernel_size=1,
                batch_norm_eps=self.batch_norm_eps,
                stride=1,
                groups=1,
                padding=0,
                activation_function=None,
                dtype=self.dtype_policy,
                name=f"lateral_conv_{i}",
            )
            self.lateral_convs_list.append(lateral_layer)
            num_blocks = round(3 * self.depth_mult)
            fpn_layer = DFineRepNCSPELAN4(
                encoder_hidden_dim=self.encoder_hidden_dim,
                hidden_expansion=self.hidden_expansion,
                batch_norm_eps=self.batch_norm_eps,
                activation_function="silu",
                numb_blocks=num_blocks,
                dtype=self.dtype_policy,
                name=f"fpn_block_{i}",
            )
            self.fpn_blocks_list.append(fpn_layer)

        self.downsample_convs_list = []
        self.pan_blocks_list = []
        for i in range(len(self.encoder_in_channels) - 1):
            self.downsample_convs_list.append(
                DFineSCDown(
                    encoder_hidden_dim=self.encoder_hidden_dim,
                    batch_norm_eps=self.batch_norm_eps,
                    kernel_size=3,
                    stride=2,
                    dtype=self.dtype_policy,
                    name=f"downsample_conv_{i}",
                )
            )
            self.pan_blocks_list.append(
                DFineRepNCSPELAN4(
                    encoder_hidden_dim=self.encoder_hidden_dim,
                    hidden_expansion=self.hidden_expansion,
                    batch_norm_eps=self.batch_norm_eps,
                    activation_function="silu",
                    numb_blocks=num_blocks,
                    dtype=self.dtype_policy,
                    name=f"pan_block_{i}",
                )
            )

        self.upsample = keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest",
            dtype=self.dtype_policy,
            name="upsample",
        )

    def build(self, input_shape):
        inputs_embeds_list_shapes = input_shape
        # Encoder layers.
        if self.encoder_layers_count > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_map_shape = inputs_embeds_list_shapes[enc_ind]
                batch_s, h_s, w_s, c_s = feature_map_shape[:4]
                if h_s is not None and w_s is not None:
                    seq_len_for_this_encoder = h_s * w_s
                else:
                    seq_len_for_this_encoder = None
                encoder_input_shape = (batch_s, seq_len_for_this_encoder, c_s)
                self.encoder_list[i].build(encoder_input_shape)
        # FPN and PAN pathways.
        # FPN (Top-down pathway).
        fpn_feature_maps_shapes = [inputs_embeds_list_shapes[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs_list, self.fpn_blocks_list)
        ):
            lateral_conv.build(fpn_feature_maps_shapes[-1])
            shape_after_lateral_conv = lateral_conv.compute_output_shape(
                fpn_feature_maps_shapes[-1]
            )
            batch_s, orig_h, orig_w, c = shape_after_lateral_conv
            target_h = orig_h * 2 if orig_h is not None else None
            target_w = orig_w * 2 if orig_w is not None else None
            shape_after_resize = (
                batch_s,
                target_h,
                target_w,
                c,
            )
            backbone_feature_map_k_shape = inputs_embeds_list_shapes[
                self.num_fpn_stages - idx - 1
            ]
            concat_channels = (
                shape_after_resize[3] + backbone_feature_map_k_shape[3]
            )
            shape_after_concat_fpn = (
                shape_after_resize[0],
                shape_after_resize[1],
                shape_after_resize[2],
                concat_channels,
            )
            fpn_block.build(shape_after_concat_fpn)
            fpn_feature_maps_shapes.append(
                fpn_block.compute_output_shape(shape_after_concat_fpn)
            )
        # PAN (Bottom-up pathway).
        reversed_fpn_feature_maps_shapes = fpn_feature_maps_shapes[::-1]
        pan_feature_maps_shapes = [reversed_fpn_feature_maps_shapes[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs_list, self.pan_blocks_list)
        ):
            downsample_conv.build(pan_feature_maps_shapes[-1])
            shape_after_downsample = downsample_conv.compute_output_shape(
                pan_feature_maps_shapes[-1]
            )
            fpn_shape = reversed_fpn_feature_maps_shapes[idx + 1]
            concat_shape = list(shape_after_downsample)
            concat_shape[-1] += fpn_shape[-1]
            pan_block.build(tuple(concat_shape))
            pan_feature_maps_shapes.append(
                pan_block.compute_output_shape(tuple(concat_shape))
            )
        super().build(input_shape)

    def call(
        self,
        inputs_embeds_list,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        training=None,
    ):
        hidden_states_list = [
            keras.ops.convert_to_tensor(t) for t in inputs_embeds_list
        ]

        _output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        _output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else False
        )

        encoder_states_tuple = () if _output_hidden_states else None
        all_attentions_tuple = () if _output_attentions else None

        if self.encoder_layers_count > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                current_feature_map = hidden_states_list[enc_ind]
                if _output_hidden_states:
                    encoder_states_tuple = encoder_states_tuple + (
                        current_feature_map,
                    )

                batch_size = keras.ops.shape(current_feature_map)[0]
                height = keras.ops.shape(current_feature_map)[1]
                width = keras.ops.shape(current_feature_map)[2]

                src_flatten = keras.ops.reshape(
                    current_feature_map,
                    (batch_size, height * width, self.encoder_hidden_dim),
                )

                pos_embed = None
                if training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        width,
                        height,
                        self.encoder_hidden_dim,
                        self.positional_encoding_temperature,
                    )
                processed_feature_map, layer_attentions = self.encoder_list[i](
                    src=src_flatten,
                    src_mask=attention_mask,
                    pos_embed=pos_embed,
                    output_attentions=_output_attentions,
                    training=training,
                )

                hidden_states_list[enc_ind] = keras.ops.reshape(
                    processed_feature_map,
                    (batch_size, height, width, self.encoder_hidden_dim),
                )

                if _output_attentions and layer_attentions is not None:
                    all_attentions_tuple = all_attentions_tuple + (
                        layer_attentions,
                    )

            if _output_hidden_states:
                encoder_states_tuple = encoder_states_tuple + (
                    hidden_states_list[self.encode_proj_layers[-1]],
                )

        fpn_feature_maps_list = [hidden_states_list[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs_list, self.fpn_blocks_list)
        ):
            backbone_feature_map_k = hidden_states_list[
                self.num_fpn_stages - idx - 1
            ]
            top_fpn_feature_map_k = fpn_feature_maps_list[-1]

            top_fpn_feature_map_k = lateral_conv(
                top_fpn_feature_map_k, training=training
            )
            fpn_feature_maps_list[-1] = top_fpn_feature_map_k
            top_fpn_feature_map_resized_k = self.upsample(
                top_fpn_feature_map_k, training=training
            )

            fused_feature_map_k = keras.ops.concatenate(
                [top_fpn_feature_map_resized_k, backbone_feature_map_k], axis=-1
            )
            new_fpn_feature_map_k = fpn_block(
                fused_feature_map_k, training=training
            )
            fpn_feature_maps_list.append(new_fpn_feature_map_k)

        fpn_feature_maps_list = fpn_feature_maps_list[::-1]

        pan_feature_maps_list = [fpn_feature_maps_list[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs_list, self.pan_blocks_list)
        ):
            top_pan_feature_map_k = pan_feature_maps_list[-1]
            fpn_feature_map_k = fpn_feature_maps_list[idx + 1]

            downsampled_feature_map_k = downsample_conv(
                top_pan_feature_map_k, training=training
            )
            fused_feature_map_k = keras.ops.concatenate(
                [downsampled_feature_map_k, fpn_feature_map_k], axis=-1
            )
            new_pan_feature_map_k = pan_block(
                fused_feature_map_k, training=training
            )
            pan_feature_maps_list.append(new_pan_feature_map_k)

        return tuple(
            v
            for v in [
                pan_feature_maps_list,
                encoder_states_tuple if _output_hidden_states else None,
                all_attentions_tuple if _output_attentions else None,
            ]
            if v is not None
        )

    @staticmethod
    def build_2d_sincos_position_embedding(
        width, height, embed_dim=256, temperature=10000.0
    ):
        grid_w = keras.ops.arange(width, dtype="float32")
        grid_h = keras.ops.arange(height, dtype="float32")
        grid_w, grid_h = keras.ops.meshgrid(grid_w, grid_h, indexing="ij")
        if embed_dim % 4 != 0:
            raise ValueError(
                "Embed dimension must be divisible by 4 for 2D sin-cos position"
                " embedding"
            )
        pos_dim = embed_dim // 4
        omega = keras.ops.arange(pos_dim, dtype="float32") / pos_dim
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
                "encoder_layers": self.encoder_layers_count,
                "batch_norm_eps": self.batch_norm_eps,
                "hidden_expansion": self.hidden_expansion,
                "depth_mult": self.depth_mult,
            }
        )
        return config

    def compute_output_shape(self, inputs_embeds_list_shapes):
        encoder_output_shapes = []
        for i, enc_ind in enumerate(self.encode_proj_layers):
            input_shape_for_encoder = inputs_embeds_list_shapes[enc_ind]
            batch_s, h_s, w_s, c_s = input_shape_for_encoder
            if h_s is not None and w_s is not None:
                seq_len_for_this_encoder = h_s * w_s
            else:
                seq_len_for_this_encoder = None
            encoder_input_shape_reshaped = (
                batch_s,
                seq_len_for_this_encoder,
                c_s,
            )
            _, enc_attn_shape = self.encoder_list[i].compute_output_shape(
                encoder_input_shape_reshaped
            )
            enc_hidden_shape_original = (batch_s, h_s, w_s, c_s)
            encoder_output_shapes.append(
                (enc_hidden_shape_original, enc_attn_shape)
            )
        encoder_states_tuple_shapes = []
        all_attentions_tuple_shapes = []
        if self.encoder_layers_count > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                encoder_states_tuple_shapes.append(encoder_output_shapes[i][0])
                all_attentions_tuple_shapes.append(encoder_output_shapes[i][1])
            encoder_states_tuple_shapes.append(encoder_output_shapes[-1][0])
        fpn_feature_maps_shapes = [inputs_embeds_list_shapes[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(
            zip(self.lateral_convs_list, self.fpn_blocks_list)
        ):
            shape_after_lateral_conv = lateral_conv.compute_output_shape(
                fpn_feature_maps_shapes[-1]
            )
            batch_s, orig_h, orig_w, c = shape_after_lateral_conv
            target_h = orig_h * 2 if orig_h is not None else None
            target_w = orig_w * 2 if orig_w is not None else None
            shape_after_resize = (
                shape_after_lateral_conv[0],
                target_h,
                target_w,
                c,
            )
            backbone_feature_map_k_shape = inputs_embeds_list_shapes[
                self.num_fpn_stages - idx - 1
            ]
            shape_after_concat_fpn = (
                shape_after_resize[0],
                shape_after_resize[1],
                shape_after_resize[2],
                shape_after_resize[3] + backbone_feature_map_k_shape[3],
            )
            shape_after_fpn_block = fpn_block.compute_output_shape(
                shape_after_concat_fpn
            )
            fpn_feature_maps_shapes.append(shape_after_fpn_block)
        reversed_fpn_feature_maps_shapes = fpn_feature_maps_shapes[::-1]
        pan_feature_maps_shapes = [reversed_fpn_feature_maps_shapes[0]]
        for idx, (downsample_conv, pan_block) in enumerate(
            zip(self.downsample_convs_list, self.pan_blocks_list)
        ):
            shape_after_downsample_conv = downsample_conv.compute_output_shape(
                pan_feature_maps_shapes[-1]
            )
            fpn_feature_map_k_shape = reversed_fpn_feature_maps_shapes[idx + 1]
            shape_after_concat_pan = (
                shape_after_downsample_conv[0],
                shape_after_downsample_conv[1],
                shape_after_downsample_conv[2],
                shape_after_downsample_conv[3] + fpn_feature_map_k_shape[3],
            )
            shape_after_pan_block = pan_block.compute_output_shape(
                shape_after_concat_pan
            )
            pan_feature_maps_shapes.append(shape_after_pan_block)
        final_pan_shapes_tuple = tuple(pan_feature_maps_shapes)
        final_encoder_states_tuple_shapes = tuple(encoder_states_tuple_shapes)
        final_all_attentions_tuple_shapes = tuple(all_attentions_tuple_shapes)
        return (
            final_pan_shapes_tuple,
            final_encoder_states_tuple_shapes,
            final_all_attentions_tuple_shapes,
        )
