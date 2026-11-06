import keras
from keras import layers
from keras import ops


class NoNorm(layers.Layer):
    """
    NoNorm simply adds a trainable elementwise scale (gamma) and shift (beta)
    without actual normalization.

    Args:
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(self, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.gamma = self.add_weight(
            shape=(hidden_size,),
            initializer="ones",
            trainable=True,
            name="gamma",
        )
        self.beta = self.add_weight(
            shape=(hidden_size,),
            initializer="zeros",
            trainable=True,
            name="beta",
        )
        self.built = True

    def call(self, inputs):
        gamma = ops.cast(self.gamma, inputs.dtype)
        beta = ops.cast(self.beta, inputs.dtype)
        return inputs * gamma + beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_output_spec(self, inputs):
        return keras.KerasTensor(shape=inputs.shape, dtype=self.compute_dtype)


class MobileBertTrigramProjection(layers.Layer):
    """
    Projects the concatenation of next, current, and previous embeddings.

    Args:
        hidden_dim: int. The dimension of the output.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(self, hidden_dim, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.dense = layers.Dense(hidden_dim, dtype=dtype, name="dense")

    def build(self, input_shape):
        self.dense.build((input_shape[0], input_shape[1], input_shape[2] * 3))
        self.built = True

    def call(self, inputs):
        next_e = ops.pad(inputs[:, 1:, :], [[0, 0], [0, 1], [0, 0]])
        prev_e = ops.pad(inputs[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
        x = ops.concatenate([next_e, inputs, prev_e], axis=-1)
        return self.dense(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class MobileBertBottleneck(layers.Layer):
    """
    Bottleneck layer for MobileBERT.

    Args:
        intra_bottleneck_dim: int. The inner bottleneck dimension.
        normalization_type: string. "no_norm" or "layer_norm".
        layer_norm_epsilon: float. The epsilon value in layer norm.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        intra_bottleneck_dim,
        normalization_type="no_norm",
        layer_norm_epsilon=1e-12,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.intra_bottleneck_dim = intra_bottleneck_dim
        self.normalization_type = normalization_type
        self.layer_norm_epsilon = layer_norm_epsilon

        self.dense = layers.Dense(
            intra_bottleneck_dim, dtype=dtype, name="dense"
        )
        if normalization_type == "no_norm":
            self.layer_norm = NoNorm(dtype=dtype, name="layer_norm")
        else:
            self.layer_norm = layers.LayerNormalization(
                epsilon=layer_norm_epsilon, dtype=dtype, name="layer_norm"
            )

    def build(self, input_shape):
        self.dense.build(input_shape)
        self.layer_norm.build(
            (input_shape[0], input_shape[1], self.intra_bottleneck_dim)
        )
        self.built = True

    def call(self, inputs):
        x = self.dense(inputs)
        return self.layer_norm(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.intra_bottleneck_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intra_bottleneck_dim": self.intra_bottleneck_dim,
                "normalization_type": self.normalization_type,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class MobileBertAttention(layers.Layer):
    """
    Self-attention layer for MobileBERT.

    Args:
        num_heads: int. Number of attention heads.
        intra_bottleneck_dim: int. The inner bottleneck dimension.
        dropout: float. Dropout probability.
        normalization_type: string. "no_norm" or "layer_norm".
        layer_norm_epsilon: float. The epsilon value in layer norm.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        num_heads,
        intra_bottleneck_dim,
        dropout=0.0,
        normalization_type="no_norm",
        layer_norm_epsilon=1e-12,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.num_heads = num_heads
        self.intra_bottleneck_dim = intra_bottleneck_dim
        self.dropout = dropout
        self.normalization_type = normalization_type
        self.layer_norm_epsilon = layer_norm_epsilon

        self.query_dense = layers.Dense(
            intra_bottleneck_dim, dtype=dtype, name="query_dense"
        )
        self.key_dense = layers.Dense(
            intra_bottleneck_dim, dtype=dtype, name="key_dense"
        )
        self.value_dense = layers.Dense(
            intra_bottleneck_dim, dtype=dtype, name="value_dense"
        )
        self.output_dense = layers.Dense(
            intra_bottleneck_dim, dtype=dtype, name="output_dense"
        )

        if normalization_type == "no_norm":
            self.output_layer_norm = NoNorm(
                dtype=dtype, name="output_layer_norm"
            )
        else:
            self.output_layer_norm = layers.LayerNormalization(
                epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="output_layer_norm",
            )

        self.dropout_layer = layers.Dropout(
            dropout, dtype=dtype, name="dropout"
        )

    def build(self, query_shape, value_shape, **kwargs):
        self.query_dense.build(query_shape)
        self.key_dense.build(query_shape)
        self.value_dense.build(value_shape)
        self.output_dense.build(
            (query_shape[0], query_shape[1], self.intra_bottleneck_dim)
        )
        self.output_layer_norm.build(
            (query_shape[0], query_shape[1], self.intra_bottleneck_dim)
        )
        self.built = True

    def call(self, query, value, residual, padding_mask=None, training=None):
        batch_size = ops.shape(query)[0]
        seq_length = ops.shape(query)[1]
        head_dim = self.intra_bottleneck_dim // self.num_heads

        q = self.query_dense(query)
        k = self.key_dense(query)
        v = self.value_dense(value)

        q = ops.reshape(q, (batch_size, seq_length, self.num_heads, head_dim))
        k = ops.reshape(k, (batch_size, seq_length, self.num_heads, head_dim))
        v = ops.reshape(v, (batch_size, seq_length, self.num_heads, head_dim))

        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores / ops.cast(
            ops.sqrt(ops.cast(head_dim, q.dtype)), q.dtype
        )

        if padding_mask is not None:
            mask = ops.cast(padding_mask, q.dtype)
            mask = mask[:, None, None, :]
            scores = scores + (ops.cast(1.0, q.dtype) - mask) * ops.cast(
                -1e9, q.dtype
            )

        attention_probs = ops.softmax(scores, axis=-1)
        attention_probs = self.dropout_layer(attention_probs, training=training)

        attention_output = ops.matmul(attention_probs, v)
        attention_output = ops.transpose(attention_output, (0, 2, 1, 3))
        attention_output = ops.reshape(
            attention_output,
            (batch_size, seq_length, self.intra_bottleneck_dim),
        )

        out = self.output_dense(attention_output)
        out = out + ops.cast(residual, out.dtype)
        return self.output_layer_norm(out)

    def compute_output_shape(self, query_shape, value_shape, **kwargs):
        return (query_shape[0], query_shape[1], self.intra_bottleneck_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "intra_bottleneck_dim": self.intra_bottleneck_dim,
                "dropout": self.dropout,
                "normalization_type": self.normalization_type,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class MobileBertFeedForward(layers.Layer):
    """
    Feed-forward network for MobileBERT.

    Args:
        intermediate_dim: int. The hidden dimension of the FFN.
        output_dim: int. The output dimension of the FFN.
        activation: string. The activation function.
        normalization_type: string. "no_norm" or "layer_norm".
        layer_norm_epsilon: float. The epsilon value in layer norm.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        intermediate_dim,
        output_dim,
        activation="relu",
        normalization_type="no_norm",
        layer_norm_epsilon=1e-12,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.activation = keras.activations.get(activation)
        self.normalization_type = normalization_type
        self.layer_norm_epsilon = layer_norm_epsilon

        self.intermediate_dense = layers.Dense(
            intermediate_dim, dtype=dtype, name="intermediate_dense"
        )
        self.output_dense = layers.Dense(
            output_dim, dtype=dtype, name="output_dense"
        )

        if normalization_type == "no_norm":
            self.output_layer_norm = NoNorm(
                dtype=dtype, name="output_layer_norm"
            )
        else:
            self.output_layer_norm = layers.LayerNormalization(
                epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="output_layer_norm",
            )

    def build(self, input_shape):
        self.intermediate_dense.build(input_shape)
        self.output_dense.build(
            (input_shape[0], input_shape[1], self.intermediate_dim)
        )
        self.output_layer_norm.build(
            (input_shape[0], input_shape[1], self.output_dim)
        )
        self.built = True

    def call(self, inputs):
        x = self.intermediate_dense(inputs)
        x = self.activation(x)
        x = self.output_dense(x)
        x = x + ops.cast(inputs, x.dtype)
        return self.output_layer_norm(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "output_dim": self.output_dim,
                "activation": keras.activations.serialize(self.activation),
                "normalization_type": self.normalization_type,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class MobileBertEncoderLayer(layers.Layer):
    """
    Encoder layer for MobileBERT.

    Args:
        hidden_dim: int. The hidden dimension of the transformer.
        intra_bottleneck_dim: int. The inner bottleneck dimension.
        intermediate_dim: int. The hidden dimension of the FFNs.
        num_heads: int. Number of attention heads.
        num_feedforward_networks: int. Number of stacked FFNs.
        activation: string. The activation function.
        normalization_type: string. "no_norm" or "layer_norm".
        layer_norm_epsilon: float. The epsilon value in layer norm.
        dropout: float. Dropout probability for the layer.
        attention_dropout: float. Dropout probability for the attention probs.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        hidden_dim,
        intra_bottleneck_dim,
        intermediate_dim,
        num_heads,
        num_feedforward_networks=4,
        activation="relu",
        normalization_type="no_norm",
        layer_norm_epsilon=1e-12,
        dropout=0.0,
        attention_dropout=0.1,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.intra_bottleneck_dim = intra_bottleneck_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.num_feedforward_networks = num_feedforward_networks
        self.activation = keras.activations.get(activation)
        self.normalization_type = normalization_type
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        self.bottleneck_input = MobileBertBottleneck(
            intra_bottleneck_dim,
            normalization_type=normalization_type,
            layer_norm_epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="bottleneck_input",
        )
        self.bottleneck_attention = MobileBertBottleneck(
            intra_bottleneck_dim,
            normalization_type=normalization_type,
            layer_norm_epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="bottleneck_attention",
        )
        self.attention = MobileBertAttention(
            num_heads=num_heads,
            intra_bottleneck_dim=intra_bottleneck_dim,
            dropout=attention_dropout,
            normalization_type=normalization_type,
            layer_norm_epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="attention",
        )

        self.ffn_layers = []
        for i in range(num_feedforward_networks):
            self.ffn_layers.append(
                MobileBertFeedForward(
                    intermediate_dim=intermediate_dim,
                    output_dim=intra_bottleneck_dim,
                    activation=activation,
                    normalization_type=normalization_type,
                    layer_norm_epsilon=layer_norm_epsilon,
                    dtype=dtype,
                    name=f"ffn_{i}",
                )
            )

        self.output_bottleneck_dense = layers.Dense(
            hidden_dim, dtype=dtype, name="output_bottleneck_dense"
        )
        self.output_bottleneck_dropout = layers.Dropout(
            dropout, dtype=dtype, name="output_bottleneck_dropout"
        )
        if normalization_type == "no_norm":
            self.output_bottleneck_layer_norm = NoNorm(
                dtype=dtype, name="output_bottleneck_layer_norm"
            )
        else:
            self.output_bottleneck_layer_norm = layers.LayerNormalization(
                epsilon=layer_norm_epsilon,
                dtype=dtype,
                name="output_bottleneck_layer_norm",
            )

    def build(self, input_shape):
        self.bottleneck_input.build(input_shape)
        self.bottleneck_attention.build(input_shape)
        bottleneck_shape = (
            input_shape[0],
            input_shape[1],
            self.intra_bottleneck_dim,
        )

        self.attention.build(
            query_shape=bottleneck_shape, value_shape=input_shape
        )
        for i in range(self.num_feedforward_networks):
            self.ffn_layers[i].build(bottleneck_shape)

        self.output_bottleneck_dense.build(bottleneck_shape)
        self.output_bottleneck_layer_norm.build(input_shape)
        self.built = True

    def call(self, hidden_states, padding_mask=None, training=None):
        residual_in = self.bottleneck_input(hidden_states)
        query = self.bottleneck_attention(hidden_states)

        attention_out = self.attention(
            query=query,
            value=hidden_states,
            residual=residual_in,
            padding_mask=padding_mask,
            training=training,
        )

        x = attention_out
        for i in range(self.num_feedforward_networks):
            x = self.ffn_layers[i](x)

        x = self.output_bottleneck_dense(x)
        x = self.output_bottleneck_dropout(x, training=training)
        x = x + ops.cast(hidden_states, x.dtype)
        return self.output_bottleneck_layer_norm(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intra_bottleneck_dim": self.intra_bottleneck_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "num_feedforward_networks": self.num_feedforward_networks,
                "activation": keras.activations.serialize(self.activation),
                "normalization_type": self.normalization_type,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config


class MobileBertMaskedLMHead(layers.Layer):
    """
    Masked Language Modeling head for MobileBERT.

    Args:
        token_embedding: A `keras_hub.layers.ReversibleEmbedding` instance.
            Used to project from the `embedding_dim` of the model to the output
            `vocabulary_size`.
        embedding_dim: int. The output dimension of the token embedding.
        hidden_dim: int. The dimension of the transformer outputs.
        vocabulary_size: int. The size of the token vocabulary.
        layer_norm_epsilon: float. The epsilon value in layer norm.
        activation: string. The activation function.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for model computations and weights.
    """

    def __init__(
        self,
        token_embedding=None,
        embedding_dim=128,
        hidden_dim=512,
        vocabulary_size=None,
        layer_norm_epsilon=1e-12,
        activation="relu",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.token_embedding = token_embedding
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation = keras.activations.get(activation)

        if vocabulary_size is None and token_embedding is None:
            raise ValueError(
                "One of `vocabulary_size` or `token_embedding` must be set."
            )

        if token_embedding:
            if vocabulary_size and vocabulary_size != token_embedding.input_dim:
                raise ValueError("`vocabulary_size` should match embedding.")
            self.vocabulary_size = token_embedding.input_dim
        else:
            self.vocabulary_size = vocabulary_size

        self.transform_dense = layers.Dense(
            hidden_dim, dtype=dtype, name="transform_dense"
        )
        self.transform_layer_norm = layers.LayerNormalization(
            epsilon=layer_norm_epsilon, dtype=dtype, name="transform_layer_norm"
        )

    def build(self, inputs_shape, mask_positions_shape=None):
        self.transform_dense.build(inputs_shape)
        self.transform_layer_norm.build(
            (inputs_shape[0], inputs_shape[1], self.hidden_dim)
        )

        self.extra_kernel = self.add_weight(
            shape=(self.hidden_dim - self.embedding_dim, self.vocabulary_size),
            initializer="zeros",
            trainable=True,
            name="extra_kernel",
        )
        self.bias = self.add_weight(
            shape=(self.vocabulary_size,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        self.built = True

    def call(self, inputs, mask_positions=None):
        x = inputs
        if mask_positions is not None:
            mask_positions = ops.expand_dims(mask_positions, axis=-1)
            x = ops.take_along_axis(x, mask_positions, axis=1)

        x = self.transform_dense(x)
        x = self.activation(x)
        x = self.transform_layer_norm(x)

        x_emb = x[..., : self.embedding_dim]
        x_extra = x[..., self.embedding_dim :]

        if self.token_embedding is not None:
            logits_emb = self.token_embedding(x_emb, reverse=True)
        else:
            raise ValueError("`token_embedding` must be set to call.")

        logits_extra = ops.matmul(
            x_extra, ops.cast(self.extra_kernel, x_extra.dtype)
        )

        outputs = ops.cast(logits_emb, logits_extra.dtype) + logits_extra
        outputs = ops.cast(outputs, self.compute_dtype)
        return outputs + ops.cast(self.bias, self.compute_dtype)

    def compute_output_shape(self, inputs_shape, mask_positions_shape=None):
        if mask_positions_shape is not None:
            return (
                inputs_shape[0],
                mask_positions_shape[1],
                self.vocabulary_size,
            )
        return (inputs_shape[0], inputs_shape[1], self.vocabulary_size)

    @classmethod
    def from_config(cls, config):
        embedding = config.get("token_embedding")
        if embedding:
            config["token_embedding"] = keras.layers.deserialize(embedding)
        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        embedding_config = None
        if self.token_embedding:
            embedding_config = keras.layers.serialize(self.token_embedding)
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "activation": keras.activations.serialize(self.activation),
                "token_embedding": embedding_config,
            }
        )
        return config
