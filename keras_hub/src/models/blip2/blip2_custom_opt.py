"""BLIP-2 custom OPT model."""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_opt_decoder import OPTDecoderBlock


def opt_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras.saving.register_keras_serializable(package="keras_hub")
class BLIP2OPTEmbeddings(keras.layers.Layer):
    """Embeddings for the BLIP-2 OPT language model.

    This layer combines learned token embeddings and learned position
    embeddings. Position embeddings are indexed starting from `position_offset`
    to account for the prepended Q-Former query tokens.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        hidden_dim: int. The dimensionality of the embeddings.
        max_sequence_length: int. The maximum sequence length supported.
        position_offset: int. The offset added to position indices.
        initializer_range: float. The standard deviation for the truncated
            normal initializer.
        dtype: The dtype of the layer.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        vocabulary_size,
        hidden_dim,
        max_sequence_length,
        position_offset,
        initializer_range,
        dtype,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        self.position_offset = position_offset
        self.initializer_range = initializer_range

        self.token_embedding = keras.layers.ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            tie_weights=True,
            embeddings_initializer=opt_kernel_initializer(initializer_range),
            dtype=dtype,
            name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            input_dim=max_sequence_length + 2,
            output_dim=hidden_dim,
            embeddings_initializer=opt_kernel_initializer(initializer_range),
            dtype=dtype,
            name="position_embedding",
        )

    def build(self, input_shape):
        self.token_embedding.build(input_shape)
        self.position_embedding.build((None, None))
        self.built = True

    def call(
        self,
        token_ids,
        position_ids=None,
        visual_position_ids=None,
        training=None,
    ):
        token_embeds = self.token_embedding(token_ids, training=training)
        if position_ids is None:
            seq_len = ops.shape(token_ids)[-1]
            position_ids = ops.expand_dims(
                ops.arange(
                    self.position_offset,
                    self.position_offset + seq_len,
                    dtype="int32",
                ),
                axis=0,
            )
        pos_embeds = self.position_embedding(position_ids)
        text_out = token_embeds + ops.cast(pos_embeds, token_embeds.dtype)

        if visual_position_ids is not None:
            vis_pos_embeds = self.position_embedding(visual_position_ids)
            return text_out, ops.cast(vis_pos_embeds, token_embeds.dtype)

        return text_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "max_sequence_length": self.max_sequence_length,
                "position_offset": self.position_offset,
                "initializer_range": self.initializer_range,
            }
        )
        return config


@keras_hub_export("keras_hub.models.BLIP2CustomOPT")
class BLIP2CustomOPT(keras.Model):
    """Custom OPT language model for BLIP-2.

    This model is a variant of the OPT decoder that accepts visual features
    from the Q-Former as a soft prompt prepended to the text tokens. It uses
    pre-normalization and provides a `call_with_cache` method for efficient
    autoregressive generation.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. Number of transformer layers.
        num_heads: int. Number of attention heads.
        hidden_dim: int. The dimensionality of the transformer layers.
        intermediate_dim: int. The dimensionality of the FFN layers.
        num_query_tokens: int. The number of visual query tokens prepended.
        dropout: float. Dropout probability.
        max_sequence_length: int. The maximum sequence length.
        qformer_hidden_dim: int. The dimensionality of Q-Former features.
        language_projection: `keras.layers.Layer`. Optional projection layer for
            visual features.
        initializer_range: float. Initializer range for weights.
        layer_norm_epsilon: float. Epsilon for layer normalization.
        dtype: The dtype of the model.
        name: The name of the model.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        num_query_tokens,
        dropout,
        max_sequence_length,
        qformer_hidden_dim,
        language_projection=None,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        dtype=None,
        name=None,
        **kwargs,
    ):
        # === Layers ===
        embeddings_layer = BLIP2OPTEmbeddings(
            vocabulary_size=vocabulary_size,
            hidden_dim=hidden_dim,
            max_sequence_length=max_sequence_length,
            position_offset=num_query_tokens + 2,
            initializer_range=initializer_range,
            dtype=dtype,
            name="embeddings_layer",
        )
        transformer_layers = [
            OPTDecoderBlock(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                name=f"transformer_layer_{i}",
            )
            for i in range(num_layers)
        ]
        layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="layer_norm",
        )
        if language_projection is None:
            language_projection = keras.layers.EinsumDense(
                equation="btd,df->btf",
                output_shape=(None, hidden_dim),
                bias_axes="f",
                name="language_projection",
                dtype=dtype,
            )

        # === Functional Model ===
        token_ids_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )
        inputs = {
            "token_ids": token_ids_input,
            "padding_mask": padding_mask_input,
        }

        if num_query_tokens > 0:
            qformer_features_input = keras.Input(
                shape=(num_query_tokens, qformer_hidden_dim),
                name="qformer_features",
            )
            inputs["qformer_features"] = qformer_features_input

            projected_qf = language_projection(qformer_features_input)

            vis_pos_ids = ops.expand_dims(
                ops.arange(2, 2 + num_query_tokens, dtype="int32"), axis=0
            )
            x, vis_pos_embeds = embeddings_layer(
                token_ids_input, visual_position_ids=vis_pos_ids
            )
            projected_qf = projected_qf + vis_pos_embeds

            x = ops.concatenate([projected_qf, x], axis=1)

            vis_mask = ops.cast(
                ops.ones_like(qformer_features_input[..., 0]), "bool"
            )
            full_padding_mask = ops.concatenate(
                [vis_mask, ops.cast(padding_mask_input, "bool")], axis=1
            )
        else:
            x = embeddings_layer(token_ids_input)  # only called once now
            full_padding_mask = padding_mask_input

        for layer in transformer_layers:
            x = layer(x, padding_mask=full_padding_mask)

        outputs = layer_norm(x)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_query_tokens = num_query_tokens
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.qformer_hidden_dim = qformer_hidden_dim
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon

        # === Track Layers ===
        self.embeddings_layer = embeddings_layer
        self.transformer_layers = transformer_layers
        self.layer_norm = layer_norm
        self.language_projection = language_projection

    def call_with_cache(self, x, padding_mask, cache, cache_update_index):
        updated_caches = []
        for i, layer in enumerate(self.transformer_layers):
            per_layer_cache = cache[:, i]
            x, new_layer_cache = layer(
                x,
                padding_mask=padding_mask,
                cache=per_layer_cache,
                cache_update_index=cache_update_index,
            )
            updated_caches.append(new_layer_cache)
        new_cache = ops.stack(updated_caches, axis=1)
        x = self.layer_norm(x)
        return x, new_cache

    @property
    def token_embedding(self):
        return self.embeddings_layer.token_embedding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_query_tokens": self.num_query_tokens,
                "dropout": self.dropout,
                "max_sequence_length": self.max_sequence_length,
                "qformer_hidden_dim": self.qformer_hidden_dim,
                "initializer_range": self.initializer_range,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "language_projection": keras.layers.serialize(
                    self.language_projection
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("language_projection") is not None:
            config["language_projection"] = keras.layers.deserialize(
                config["language_projection"]
            )
        return cls(**config)
