import numpy as np
from keras import layers
from keras import ops

from keras_hub.src.models.sam3.sam3_layers import SAM3DecoderMLP


class SAM3DotProductScoring(layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.intermediate_dim = int(intermediate_dim)
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        self.text_mlp = SAM3DecoderMLP(
            num_layers=2,
            hidden_dim=self.intermediate_dim,
            output_dim=self.hidden_dim,
            dtype=self.dtype_policy,
            name="text_mlp",
        )
        self.text_mlp_dropout = layers.Dropout(
            self.dropout_rate, dtype=self.dtype_policy, name="text_mlp_dropout"
        )
        self.text_mlp_out_norm = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="text_mlp_out_norm",
        )

        # Projections for text and query features.
        self.text_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="text_proj"
        )
        self.query_proj = layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="query_proj"
        )

        # Scale factor for dot product.
        self.scale = float(1.0 / np.sqrt(self.hidden_dim))

        # Clamping to avoid numerical issues.
        self.clamp_max_val = 12.0

    def build(
        self, decoder_hidden_states_shape, text_features_shape, text_masks_shape
    ):
        self.text_mlp.build(text_features_shape)
        self.text_mlp_dropout.build(text_features_shape)
        self.text_mlp_out_norm.build(text_features_shape)
        pooled_text_shape = [text_features_shape[0], text_features_shape[-1]]
        self.text_proj.build(pooled_text_shape)
        self.query_proj.build(decoder_hidden_states_shape)

    def _pool_text_features(self, text_features, text_mask=None):
        if text_mask is None:
            # No padding, simple mean.
            return ops.mean(text_features, axis=1)

        is_valid = ops.expand_dims(
            ops.cast(text_mask, text_features.dtype), axis=-1
        )
        # Count valid tokens per batch.
        num_valid = ops.maximum(ops.sum(is_valid, axis=1), 1.0)
        # Mean pool only over valid tokens.
        return ops.divide(
            ops.sum(ops.multiply(text_features, is_valid), axis=1), num_valid
        )

    def call(
        self,
        decoder_hidden_states,
        text_features,
        text_masks=None,
        training=None,
    ):
        orig_text_features = text_features
        text_features = self.text_mlp(text_features, training=training)
        text_features = self.text_mlp_dropout(text_features, training=training)
        text_features = ops.add(text_features, orig_text_features)
        text_features = self.text_mlp_out_norm(text_features, training=training)

        pooled_text = self._pool_text_features(text_features, text_masks)

        proj_text = self.text_proj(pooled_text, training=training)
        proj_queries = self.query_proj(decoder_hidden_states, training=training)

        proj_text = ops.expand_dims(proj_text, axis=-1)
        scores = ops.matmul(proj_queries, ops.expand_dims(proj_text, axis=1))
        scores = ops.multiply(scores, self.scale)
        scores = ops.clip(scores, -self.clamp_max_val, self.clamp_max_val)
        return scores

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    def compute_output_shape(
        self, decoder_hidden_states_shape, text_features_shape, text_masks_shape
    ):
        batch_size = decoder_hidden_states_shape[0]
        num_layers = decoder_hidden_states_shape[1]
        num_queries = decoder_hidden_states_shape[2]
        return [batch_size, num_layers, num_queries, 1]
