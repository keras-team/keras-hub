from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam3.sam3_layers import SAM3Attention
from keras_hub.src.models.sam3.sam3_utils import create_bidirectional_mask
from keras_hub.src.utils.keras_utils import standardize_data_format


class SAM3MaskEmbedder(layers.Layer):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)

        self.layers = [
            layers.Dense(
                self.hidden_dim, dtype=self.dtype_policy, name="layer_0"
            ),
            layers.Dense(
                self.hidden_dim, dtype=self.dtype_policy, name="layer_1"
            ),
            layers.Dense(
                self.hidden_dim, dtype=self.dtype_policy, name="layer_2"
            ),
        ]
        self.activation = layers.ReLU(
            dtype=self.dtype_policy, name="activation"
        )

    def build(self, queries_shape):
        hidden_state_shape = queries_shape
        self.activation.build(hidden_state_shape)
        for layer in self.layers:
            layer.build(hidden_state_shape)
            hidden_state_shape = layer.compute_output_shape(hidden_state_shape)

    def call(self, queries, training=None):
        hidden_states = queries
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, training=training)
            if i < len(self.layers) - 1:
                hidden_states = self.activation(
                    hidden_states, training=training
                )
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config

    def compute_output_shape(self, queries_shape):
        hidden_state_shape = list(queries_shape)
        hidden_state_shape[-1] = self.hidden_dim
        return hidden_state_shape


class SAM3PixelDecoder(layers.Layer):
    def __init__(
        self, num_upsampling_stages, hidden_dim, data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_upsampling_stages = int(num_upsampling_stages)
        self.hidden_dim = int(hidden_dim)
        self.data_format = standardize_data_format(data_format)

        # Create conv layers and norms for FPN.
        self.pad_layers = [
            layers.ZeroPadding2D(
                padding=1,
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"pad_layer_{i}",
            )
            for i in range(self.num_upsampling_stages)
        ]
        self.conv_layers = [
            layers.Conv2D(
                self.hidden_dim,
                3,
                1,
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"conv_layer_{i}",
            )
            for i in range(self.num_upsampling_stages)
        ]
        self.norms = [
            layers.GroupNormalization(
                8, epsilon=1e-5, dtype=self.dtype_policy, name=f"norm_{i}"
            )
            for i in range(self.num_upsampling_stages)
        ]

    def build(self, backbone_features_shapes):
        self.sizes = []
        for i, feature_shape in enumerate(
            reversed(backbone_features_shapes[:-1])
        ):
            if self.data_format == "channels_last":
                self.sizes.append(
                    (int(feature_shape[1]), int(feature_shape[2]))
                )
            else:
                self.sizes.append(
                    (int(feature_shape[2]), int(feature_shape[3]))
                )
            pad_layer = self.pad_layers[i]
            conv_layer = self.conv_layers[i]
            norm_layer = self.norms[i]
            pad_layer.build(feature_shape)
            feature_shape = pad_layer.compute_output_shape(feature_shape)
            conv_layer.build(feature_shape)
            feature_shape = conv_layer.compute_output_shape(feature_shape)
            norm_layer.build(feature_shape)

    def call(self, backbone_features, training=None):
        prev_fpn = backbone_features[-1]
        for i, feature in enumerate(reversed(backbone_features[:-1])):
            prev_fpn = ops.image.resize(
                prev_fpn,
                size=self.sizes[i],
                interpolation="nearest",
                data_format=self.data_format,
            )
            prev_fpn = ops.add(prev_fpn, feature)
            prev_fpn = self.pad_layers[i](prev_fpn, training=training)
            prev_fpn = self.conv_layers[i](prev_fpn, training=training)
            prev_fpn = self.norms[i](prev_fpn, training=training)
            prev_fpn = ops.relu(prev_fpn)
        return prev_fpn

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_upsampling_stages": self.num_upsampling_stages,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config

    def compute_output_shape(self, backbone_features_shapes):
        return backbone_features_shapes[0]


@keras_hub_export("keras_hub.layers.SAM3MaskDecoder")
class SAM3MaskDecoder(layers.Layer):
    """A mask decoder for the Segment Anything Model 3 (SAM3).

    This layer generates segmentation masks given the object queries from the
    DETR decoder and fused features. It uses a pixel decoder to upsample
    backbone features and predicts instance masks and semantic segmentation.

    Args:
        num_upsampling_stages: int. The number of upsampling stages in the
            pixel decoder.
        hidden_dim: int. The hidden dimension of the decoder.
        num_heads: int. The number of attention heads.
        dropout_rate: float. The dropout rate for attention. Defaults to `0.0`.
        layer_norm_epsilon: float. The epsilon value for layer normalization.
            Defaults to `1e-6`.
        data_format: str. The data format, either `"channels_last"` or
            `"channels_first"`.
    """

    def __init__(
        self,
        num_upsampling_stages,
        hidden_dim,
        num_heads,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_upsampling_stages = int(num_upsampling_stages)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        self.data_format = standardize_data_format(data_format)

        self.pixel_decoder = SAM3PixelDecoder(
            num_upsampling_stages=self.num_upsampling_stages,
            hidden_dim=self.hidden_dim,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="pixel_decoder",
        )
        self.mask_embedder = SAM3MaskEmbedder(
            hidden_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="mask_embedder",
        )
        self.instance_projection = layers.Conv2D(
            self.hidden_dim,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="instance_projection",
        )
        self.semantic_projection = layers.Conv2D(
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="semantic_projection",
        )
        self.prompt_cross_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="prompt_cross_attn",
        )
        self.prompt_cross_attn_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="prompt_cross_attn_norm",
        )
        self.prompt_cross_attn_dropout = layers.Dropout(
            self.dropout_rate,
            dtype=self.dtype_policy,
            name="prompt_cross_attn_dropout",
        )

    def build(
        self,
        decoder_queries_shape,
        backbone_features_shape,
        encoder_hidden_states_shape,
        prompt_features_shape,
        prompt_masks_shape,
    ):
        if self.data_format == "channels_last":
            self.height = int(backbone_features_shape[-1][1])
            self.width = int(backbone_features_shape[-1][2])
        else:
            self.height = int(backbone_features_shape[-1][2])
            self.width = int(backbone_features_shape[-1][3])
        self.prompt_cross_attn_norm.build(encoder_hidden_states_shape)
        self.prompt_cross_attn.build(
            encoder_hidden_states_shape,
            prompt_features_shape,
            prompt_features_shape,
        )
        self.prompt_cross_attn_dropout.build(encoder_hidden_states_shape)
        # _embed_pixels.
        encoder_visual_embeds_shape = [
            encoder_hidden_states_shape[0],
            self.height * self.width,
            encoder_hidden_states_shape[-1],
        ]
        backbone_features_shape = list(backbone_features_shape)
        backbone_features_shape[-1] = encoder_visual_embeds_shape
        self.pixel_decoder.build(backbone_features_shape)
        pixel_embeds_shape = self.pixel_decoder.compute_output_shape(
            backbone_features_shape
        )
        self.instance_projection.build(pixel_embeds_shape)
        self.mask_embedder.build(decoder_queries_shape)
        self.semantic_projection.build(pixel_embeds_shape)

    def _embed_pixels(self, backbone_features, encoder_hidden_states):
        spatial_dim = self.height * self.width
        encoder_visual_embed = encoder_hidden_states[:, :spatial_dim, :]
        encoder_visual_embed = ops.reshape(
            encoder_visual_embed, (-1, self.height, self.width, self.hidden_dim)
        )
        if self.data_format == "channels_first":
            encoder_visual_embed = ops.transpose(
                encoder_visual_embed, (0, 3, 1, 2)
            )
        backbone_features = list(backbone_features)
        backbone_features[-1] = encoder_visual_embed
        return self.pixel_decoder(backbone_features)

    def call(
        self,
        decoder_queries,
        backbone_features,
        encoder_hidden_states,
        prompt_features,
        prompt_masks,
        training=None,
    ):
        # Cross-attention: encoder features attend to prompt features.
        residual = encoder_hidden_states
        normed_hidden_states = self.prompt_cross_attn_norm(
            encoder_hidden_states, training=training
        )
        cross_attn_mask = create_bidirectional_mask(
            normed_hidden_states, prompt_masks
        )
        attn_output = self.prompt_cross_attn(
            query=normed_hidden_states,
            key=prompt_features,
            value=prompt_features,
            attention_mask=cross_attn_mask,
            training=training,
        )
        encoder_hidden_states = ops.add(
            residual,
            self.prompt_cross_attn_dropout(attn_output, training=training),
        )

        # Process backbone features through FPN to get pixel embeddings.
        pixel_embed = self._embed_pixels(
            backbone_features, encoder_hidden_states
        )

        # Predict instance masks via dot product between query embeddings and
        # pixel embeddings.
        instance_embeds = self.instance_projection(
            pixel_embed, training=training
        )
        mask_embeddings = self.mask_embedder(decoder_queries, training=training)
        if self.data_format == "channels_last":
            pred_masks = ops.einsum(
                "bqc,bhwc->bhwq", mask_embeddings, instance_embeds
            )
        else:
            pred_masks = ops.einsum(
                "bqc,bchw->bqhw", mask_embeddings, instance_embeds
            )

        # Generate semantic segmentation.
        semantic_segs = self.semantic_projection(pixel_embed, training=training)
        return pred_masks, semantic_segs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_upsampling_stages": self.num_upsampling_stages,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(
        self,
        decoder_queries_shape,
        backbone_features_shape,
        encoder_hidden_states_shape,
        prompt_features_shape,
        prompt_masks_shape,
    ):
        batch_size = encoder_hidden_states_shape[0]
        if self.data_format == "channels_last":
            output_height = int(backbone_features_shape[0][1])
            output_width = int(backbone_features_shape[0][2])
            pred_masks_shape = [
                batch_size,
                output_height,
                output_width,
                self.hidden_dim,
            ]
            semantic_segs_shape = [batch_size, output_height, output_width, 1]
        else:
            output_height = int(backbone_features_shape[0][2])
            output_width = int(backbone_features_shape[0][3])
            pred_masks_shape = [
                batch_size,
                self.hidden_dim,
                output_height,
                output_width,
            ]
            semantic_segs_shape = [batch_size, 1, output_height, output_width]
        return pred_masks_shape, semantic_segs_shape
