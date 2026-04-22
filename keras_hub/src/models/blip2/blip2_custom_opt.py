import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.blip2.blip2_opt_decoder import OPTDecoderBlock


def opt_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


class Blip2OPTEmbeddings(keras.layers.Layer):
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

        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
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

    def call(self, token_ids, position_ids=None, training=None):
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
        return token_embeds + ops.cast(pos_embeds, token_embeds.dtype)

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


@keras_hub_export("keras_hub.models.Blip2CustomOPT")
class Blip2CustomOPT(Backbone):
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
        language_projection,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        dtype=None,
        name=None,
        **kwargs,
    ):
        self.embeddings_layer = Blip2OPTEmbeddings(
            vocabulary_size=vocabulary_size,
            hidden_dim=hidden_dim,
            max_sequence_length=max_sequence_length,
            position_offset=num_query_tokens + 2,
            initializer_range=initializer_range,
            dtype=dtype,
            name="embeddings_layer",
        )

        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                OPTDecoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    dropout=dropout,
                    layer_norm_epsilon=layer_norm_epsilon,
                    dtype=dtype,
                    name=f"transformer_layer_{i}",
                )
            )

        self.layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="layer_norm",
        )

        if language_projection is None:
            self.language_projection = keras.layers.EinsumDense(
                equation="btd,df->btf",
                output_shape=(None, hidden_dim),
                bias_axes="f",
                name="language_projection",
                dtype=dtype,
            )
        elif isinstance(language_projection, dict):
            self.language_projection = keras.layers.deserialize(
                language_projection
            )
        else:
            self.language_projection = language_projection

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="bool", name="padding_mask"
        )

        x = self.embeddings_layer(token_id_input)
        inputs = {
            "token_ids": token_id_input,
            "padding_mask": padding_mask_input,
        }

        if num_query_tokens > 0:
            qf_input = keras.Input(
                shape=(num_query_tokens, qformer_hidden_dim),
                name="qformer_features",
            )
            inputs["qformer_features"] = qf_input

            projected_qf = self.language_projection(qf_input)

            pos_ids = ops.expand_dims(
                ops.arange(2, 2 + num_query_tokens, dtype="int32"), axis=0
            )
            pos_embeds = self.embeddings_layer.position_embedding(pos_ids)
            projected_qf = projected_qf + ops.cast(
                pos_embeds, projected_qf.dtype
            )

            x = ops.concatenate([projected_qf, x], axis=1)

            vis_mask = ops.cast(
                ops.ones_like(projected_qf[:, :, 0]), dtype="bool"
            )
            full_padding_mask = ops.concatenate(
                [vis_mask, padding_mask_input], axis=1
            )
        else:
            full_padding_mask = padding_mask_input

        for layer in self.transformer_layers:
            x = layer(x, padding_mask=full_padding_mask)

        output = self.layer_norm(x)

        super().__init__(
            inputs=inputs,
            outputs=output,
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

    @property
    def token_embedding(self):
        return self.embeddings_layer.token_embedding

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

    def default_lora_layer_names(self):
        return ["q_proj", "v_proj"]