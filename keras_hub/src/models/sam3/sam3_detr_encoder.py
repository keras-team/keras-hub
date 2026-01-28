from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.sam3.sam3_layers import SAM3MLP
from keras_hub.src.models.sam3.sam3_layers import SAM3Attention
from keras_hub.src.models.sam3.sam3_utils import create_bidirectional_mask


class SAM3DetrEncoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        hidden_activation="relu",
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_heads = int(num_heads)
        self.hidden_activation = hidden_activation
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.layer_norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm1",
        )
        self.self_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.dropout = layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="dropout"
        )
        self.cross_attn = SAM3Attention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="cross_attn",
        )
        self.layer_norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm2",
        )
        self.mlp = SAM3MLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            activation=self.hidden_activation,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.layer_norm3 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm3",
        )

    def build(
        self,
        vision_feats_shape,
        prompt_feats_shape,
        vision_pos_encodings_shape,
        prompt_cross_attn_masks_shape,
    ):
        self.layer_norm1.build(vision_feats_shape)
        self.self_attn.build(
            vision_feats_shape, vision_feats_shape, vision_feats_shape
        )
        self.dropout.build(vision_feats_shape)
        self.layer_norm2.build(vision_feats_shape)
        self.cross_attn.build(
            vision_feats_shape, prompt_feats_shape, prompt_feats_shape
        )
        self.layer_norm3.build(vision_feats_shape)
        self.mlp.build(vision_feats_shape)

    def call(
        self,
        vision_feats,
        prompt_feats,
        vision_pos_encodings,
        prompt_cross_attn_masks=None,
        training=None,
    ):
        residual = vision_feats
        hidden_states = self.layer_norm1(vision_feats, training=training)
        hidden_states_with_pos = ops.add(hidden_states, vision_pos_encodings)
        hidden_states = self.self_attn(
            query=hidden_states_with_pos,
            key=hidden_states_with_pos,
            value=hidden_states,
            training=training,
        )
        hidden_states = ops.add(
            self.dropout(hidden_states, training=training), residual
        )

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states, training=training)
        hidden_states = self.cross_attn(
            query=hidden_states,
            key=prompt_feats,
            value=prompt_feats,
            attention_mask=prompt_cross_attn_masks,
            training=training,
        )

        hidden_states = ops.add(
            self.dropout(hidden_states, training=training), residual
        )

        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states, training=training)
        hidden_states = self.mlp(hidden_states, training=training)
        hidden_states = ops.add(
            self.dropout(hidden_states, training=training), residual
        )
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(
        self,
        vision_feats_shape,
        prompt_feats_shape,
        vision_pos_encodings_shape,
        prompt_cross_attn_masks_shape,
    ):
        return vision_feats_shape


@keras_hub_export("keras_hub.layers.SAM3DetrEncoder")
class SAM3DetrEncoder(layers.Layer):
    """A DETR encoder for the Segment Anything Model 3 (SAM3).

    This layer implements a transformer-based encoder that fuses vision and
    prompt features. It processes flattened vision features and prompt features
    through multiple layers of self-attention and cross-attention.

    Args:
        num_layers: int. The number of transformer layers.
        hidden_dim: int. The hidden dimension of the transformer layers.
        intermediate_dim: int. The dimension of the intermediate layer in the
            transformer's MLP.
        num_heads: int. The number of attention heads.
        hidden_activation: str. The activation function for the transformer
            layers. Defaults to `"relu"`.
        dropout_rate: float. The dropout rate for the MLP and attention.
            Defaults to `0.0`.
        layer_norm_epsilon: float. The epsilon value for layer normalization.
            Defaults to `1e-6`.
    """

    def __init__(
        self,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        hidden_activation="relu",
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.num_heads = int(num_heads)
        self.hidden_activation = hidden_activation
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.layers = [
            SAM3DetrEncoderLayer(
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                hidden_activation=self.hidden_activation,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"layer_{i}",
            )
            for i in range(self.num_layers)
        ]

    def build(
        self,
        vision_features_shape,
        text_features_shape,
        vision_pos_embeds_shape,
        text_masks_shape,
    ):
        self.height = int(vision_features_shape[1])
        self.width = int(vision_features_shape[2])
        feature_flattened_shape = [
            vision_features_shape[0],
            vision_features_shape[1] * vision_features_shape[2],
            vision_features_shape[-1],
        ]
        for layer in self.layers:
            layer.build(
                feature_flattened_shape,
                text_features_shape,
                feature_flattened_shape,
                None,
            )

    def call(
        self,
        vision_features,
        text_features,
        vision_pos_embeds,
        text_masks,
        training=None,
    ):
        # Flatten multi-level features for encoder processing.
        batch_size = ops.shape(vision_features)[0]
        features_flattened = ops.reshape(
            vision_features, (batch_size, self.height * self.width, -1)
        )
        pos_embeds_flattened = ops.reshape(
            vision_pos_embeds, (batch_size, self.height * self.width, -1)
        )
        spatial_shapes = ops.array([[self.height, self.width]], dtype="int32")

        prompt_cross_attn_masks = create_bidirectional_mask(
            features_flattened, text_masks
        )
        hidden_states = features_flattened
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                prompt_feats=text_features,
                vision_pos_encodings=pos_embeds_flattened,
                prompt_cross_attn_masks=prompt_cross_attn_masks,
                training=training,
            )
        return (
            hidden_states,
            pos_embeds_flattened,
            text_features,
            spatial_shapes,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(
        self,
        vision_features_shape,
        text_features_shape,
        vision_pos_embeds_shape,
        text_masks_shape,
    ):
        features_flattened_shape = [
            vision_features_shape[0],
            vision_features_shape[1] * vision_features_shape[2],
            vision_features_shape[-1],
        ]
        spatial_shape = [1, 2]
        return (
            features_flattened_shape,
            features_flattened_shape,
            text_features_shape,
            spatial_shape,
        )
