import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_opt_decoder import OPTDecoderBlock


def opt_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras.saving.register_keras_serializable(package="keras_hub")
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


@keras_hub_export("keras_hub.models.BLIP2CustomOPT")
class BLIP2CustomOPT(keras.layers.Layer): 
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
        super().__init__(dtype=dtype, name=name, **kwargs)

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
        self._language_projection_arg = language_projection

        self.embeddings_layer = Blip2OPTEmbeddings(
            vocabulary_size=self.vocabulary_size,
            hidden_dim=self.hidden_dim,
            max_sequence_length=self.max_sequence_length,
            position_offset=self.num_query_tokens + 2,
            initializer_range=self.initializer_range,
            dtype=self.dtype_policy,
            name="embeddings_layer",
        )

    def build(self, input_shape):
        token_ids_shape = input_shape["token_ids"]

        # === Build embeddings ===
        self.embeddings_layer.build(token_ids_shape)

        # === Construct AND build transformer layers  ===
        self.transformer_layers = []
        for i in range(self.num_layers):
            block = OPTDecoderBlock(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                dropout=self.dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name=f"transformer_layer_{i}",
            )
            block.build((None, None, self.hidden_dim))
            self.transformer_layers.append(block)

        # === Construct AND build layer norm ===
        self.layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="layer_norm",
        )
        self.layer_norm.build((None, None, self.hidden_dim))

        # === Construct AND build projection ===
        self.projection_layer = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, self.hidden_dim),
            bias_axes="f",
            name="language_projection",
            dtype=self.dtype_policy,
        )
        self.projection_layer.build((None, None, self.qformer_hidden_dim))

        self.built = True

    def call(self, inputs, training=None):
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        qformer_features = inputs.get("qformer_features", None)

        x = self.embeddings_layer(token_ids, training=training)

        if qformer_features is not None:
            projected_qf = self.projection_layer(qformer_features)

            nq = ops.shape(qformer_features)[1]
            pos_ids = ops.expand_dims(
                ops.arange(2, 2 + nq, dtype="int32"), axis=0
            )
            pos_embeds = self.embeddings_layer.position_embedding(pos_ids)
            projected_qf = projected_qf + ops.cast(pos_embeds, projected_qf.dtype)

            x = ops.concatenate([projected_qf, x], axis=1)

            vis_mask = ops.ones((ops.shape(token_ids)[0], nq), dtype="bool")
            full_padding_mask = ops.concatenate([vis_mask, padding_mask], axis=1)
        else:
            full_padding_mask = padding_mask

        for layer in self.transformer_layers:
            x = layer(x, padding_mask=full_padding_mask, training=training)

        return self.layer_norm(x)

    def compute_output_shape(self, input_shape):
        token_ids_shape = input_shape["token_ids"]
        return (token_ids_shape[0], token_ids_shape[1], self.hidden_dim)

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
                "language_projection": self._language_projection_arg,
            }
        )
        return config