import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.t5.t5_multi_head_attention import T5MultiHeadAttention
from keras_hub.src.utils.keras_utils import clone_initializer


class MPNetRelativePositionBias(keras.layers.Layer):
    """Bucketed relative position bias shared by all MPNet encoder layers.

    MPNet adds a learned, per-head bias to the attention scores, based on the
    bucketed distance between the key and the query positions. The bucketing
    is the bidirectional scheme from T5. The bias is computed once from
    `arange(sequence_length)`, so it does not depend on padding, and the same
    bias is passed to every encoder layer.

    Args:
        num_heads: int. The number of attention heads.
        num_buckets: int. The number of relative position buckets. Defaults
            to `32`.
        max_distance: int. Relative distances at or beyond this value share
            the last bucket. Defaults to `128`.
        initializer: string or `keras.initializers` initializer. The
            initializer for the bias table. Defaults to `"glorot_uniform"`.

    Call arguments:
        inputs: The hidden states, of shape
            `(batch_size, sequence_length, hidden_dim)`. Only the sequence
            length is used.

    Returns:
        A tensor of shape `(1, num_heads, sequence_length, sequence_length)`.
    """

    def __init__(
        self,
        num_heads,
        num_buckets=32,
        max_distance=128,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.initializer = keras.initializers.get(initializer)

    def build(self, inputs_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_buckets, self.num_heads),
            initializer=self.initializer,
        )
        self.built = True

    def call(self, inputs):
        sequence_length = ops.shape(inputs)[1]
        position = ops.arange(sequence_length, dtype="int32")
        # `relative_position[i, j] = j - i`, the key position minus the query
        # position.
        relative_position = position[None, :] - position[:, None]
        buckets = T5MultiHeadAttention._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        values = ops.take(self.embeddings, buckets, axis=0)
        values = ops.transpose(values, (2, 0, 1))
        return ops.expand_dims(values, axis=0)

    def compute_output_shape(self, inputs_shape):
        sequence_length = inputs_shape[1]
        return (1, self.num_heads, sequence_length, sequence_length)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "num_buckets": self.num_buckets,
                "max_distance": self.max_distance,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config


class MPNetAttention(keras.layers.Layer):
    """Multi-head self-attention with an additive relative position bias.

    The scaled dot-product attention scores are offset by `position_bias`
    before the softmax. The attention is computed explicitly, rather than with
    `ops.dot_product_attention`, because an additive float bias is not
    supported on all backends.

    Args:
        num_heads: int. The number of attention heads. The hidden size of the
            inputs must be divisible by `num_heads`.
        dropout: float. Dropout probability applied to the attention
            probabilities. Defaults to `0.0`.
        kernel_initializer: string or `keras.initializers` initializer. The
            kernel initializer for the projection layers. Defaults to
            `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer. The bias
            initializer for the projection layers. Defaults to `"zeros"`.

    Call arguments:
        hidden_states: The input tensor, of shape
            `(batch_size, sequence_length, hidden_dim)`.
        position_bias: Optional tensor broadcastable to
            `(batch_size, num_heads, sequence_length, sequence_length)`, added
            to the attention scores.
        attention_mask: Optional tensor of shape
            `(batch_size, 1, sequence_length)` or
            `(batch_size, sequence_length, sequence_length)`, where `1` marks a
            key that can be attended to.
        training: bool. Whether the layer is in training mode.
    """

    def __init__(
        self,
        num_heads,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dropout = dropout
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        if hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"The hidden size of the inputs ({hidden_dim}) must be "
                f"divisible by `num_heads` ({self.num_heads})."
            )
        self.head_dim = hidden_dim // self.num_heads

        def projection(name):
            # The names `query_dense` and `value_dense` are the default
            # targets of `Backbone.enable_lora()`.
            layer = keras.layers.EinsumDense(
                equation="btd,dnh->btnh",
                output_shape=(None, self.num_heads, self.head_dim),
                bias_axes="nh",
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                dtype=self.dtype_policy,
                name=name,
            )
            layer.build(inputs_shape)
            return layer

        self.query_dense = projection("query_dense")
        self.key_dense = projection("key_dense")
        self.value_dense = projection("value_dense")
        self.output_dense = keras.layers.EinsumDense(
            equation="btnh,nhd->btd",
            output_shape=(None, hidden_dim),
            bias_axes="d",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="output_dense",
        )
        self.output_dense.build(
            (None, None, self.num_heads, self.head_dim),
        )
        # The softmax always runs in float32 for numerical stability.
        self.softmax = keras.layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )
        self.dropout_layer = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="attention_dropout",
        )
        self.built = True

    def call(
        self,
        hidden_states,
        position_bias=None,
        attention_mask=None,
        training=None,
    ):
        query = self.query_dense(hidden_states)
        key = self.key_dense(hidden_states)
        value = self.value_dense(hidden_states)

        # Same order of operations as the reference implementation: scale the
        # scores, add the position bias, then mask. Done in float32.
        scores = ops.einsum("bqnh,bknh->bnqk", query, key)
        scores = ops.cast(scores, "float32") / math.sqrt(self.head_dim)
        if position_bias is not None:
            scores = scores + ops.cast(position_bias, "float32")
        if attention_mask is not None:
            # Add a head axis.
            attention_mask = ops.expand_dims(attention_mask, axis=1)
        probs = self.softmax(scores, mask=attention_mask)
        probs = ops.cast(probs, self.compute_dtype)
        probs = self.dropout_layer(probs, training=training)

        context = ops.einsum("bnqk,bknh->bqnh", probs, value)
        return self.output_dense(context)

    def compute_output_shape(self, inputs_shape):
        return inputs_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config


class MPNetTransformerLayer(keras.layers.Layer):
    """An MPNet encoder block.

    This block has the same post-layer-normalization structure as
    `keras_hub.layers.TransformerEncoder`, except that the self-attention
    scores are offset by a relative position bias.

    Args:
        num_heads: int. The number of attention heads.
        intermediate_dim: int. The hidden size of the feedforward network.
        dropout: float. Dropout probability applied to the attention
            probabilities and to the outputs of the attention and the
            feedforward blocks. Defaults to `0.0`.
        activation: string or callable. The activation of the feedforward
            network. Defaults to `"gelu"` (the exact, erf-based GELU).
        layer_norm_epsilon: float. The epsilon of the layer normalization
            layers. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer. The
            kernel initializer for the dense and attention layers. Defaults to
            `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer. The bias
            initializer for the dense and attention layers. Defaults to
            `"zeros"`.

    Call arguments:
        inputs: The input tensor, of shape
            `(batch_size, sequence_length, hidden_dim)`.
        padding_mask: Optional boolean or int tensor of shape
            `(batch_size, sequence_length)`, where `1` marks a real token and
            `0` marks padding.
        position_bias: Optional tensor broadcastable to
            `(batch_size, num_heads, sequence_length, sequence_length)`, added
            to the attention scores.
        training: bool. Whether the layer is in training mode.
    """

    def __init__(
        self,
        num_heads,
        intermediate_dim,
        dropout=0.0,
        activation="gelu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]

        # Self attention block.
        self.self_attention_layer = MPNetAttention(
            num_heads=self.num_heads,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )
        self.self_attention_layer.build(inputs_shape)
        self.self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )
        self.self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self.self_attention_layer_norm.build(inputs_shape)

        # Feedforward block.
        self.feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self.feedforward_intermediate_dense.build(inputs_shape)
        self.feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self.feedforward_output_dense.build(tuple(intermediate_shape))
        self.feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self.feedforward_layer_norm.build(inputs_shape)
        self.built = True

    def call(
        self,
        inputs,
        padding_mask=None,
        position_bias=None,
        training=None,
    ):
        attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, None
        )

        # Self attention block.
        residual = inputs
        x = self.self_attention_layer(
            inputs,
            position_bias=position_bias,
            attention_mask=attention_mask,
            training=training,
        )
        x = self.self_attention_dropout(x, training=training)
        x = self.self_attention_layer_norm(x + residual)

        # Feedforward block.
        residual = x
        x = self.feedforward_intermediate_dense(x)
        x = self.feedforward_output_dense(x)
        x = self.feedforward_dropout(x, training=training)
        return self.feedforward_layer_norm(x + residual)

    def compute_output_shape(self, inputs_shape):
        return inputs_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config
