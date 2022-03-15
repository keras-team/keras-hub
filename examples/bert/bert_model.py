# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bert model and layer implementations.

We should work to replace this with components from the keras-nlp library.
"""

import tensorflow as tf
from tensorflow import keras

import keras_nlp


def make_attention_mask(inputs, mask):
    """Make a 3D attention mask from a 2D input mask.

    Given `inputs` with shape `[batch, from_seq_length, ...]` and a mask with
    shape `[batch_size, to_seq_length]`, this will output a mask with dtype
    matching `inputs` with shape `[batch, from_seq_length, to_seq_length]`.

    Args:
        inputs: The inputs to the attention layer.
        mask: The 2D input mask.
    """
    inputs_shape = tf.shape(inputs)
    batch_size = inputs_shape[0]
    from_seq_length = inputs_shape[1]
    to_seq_length = tf.shape(mask)[1]
    mask = tf.reshape(mask, [batch_size, 1, to_seq_length])
    mask = tf.cast(mask, dtype=inputs.dtype)
    return tf.ones([batch_size, from_seq_length, 1], dtype=inputs.dtype) * mask


class TransformerEncoderBlock(keras.layers.Layer):
    """TransformerEncoderBlock layer.

    This layer implements the Transformer Encoder from
    "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
    which combines a `keras.layers.MultiHeadAttention` layer with a
    two-layer feedforward network.

    Args:
        num_attention_heads: Number of attention heads.
        inner_size: The output dimension of the first Dense layer in a
            two-layer feedforward network.
        inner_activation: The activation for the first Dense layer in a
            two-layer feedforward network.
        output_range: the sequence output range, [0, output_range) for
            slicing the target sequence. `None` means the target sequence is
            not sliced.
        kernel_initializer: Initializer for dense layer kernels.
        bias_initializer: Initializer for dense layer biases.
        kernel_regularizer: Regularizer for dense layer kernels.
        bias_regularizer: Regularizer for dense layer biases.
        activity_regularizer: Regularizer for dense layer activity.
        kernel_constraint: Constraint for dense layer kernels.
        bias_constraint: Constraint for dense layer kernels.
        use_bias: Whether to enable use_bias in attention layer. If set
            False, use_bias in attention layer is disabled.
        norm_first: Whether to normalize inputs to attention and
            intermediate dense layers. If set False, output of attention and
            intermediate dense layers is normalized.
        norm_epsilon: Epsilon value to initialize normalization layers.
        hidden_dropout: Dropout probability for the post-attention and
            output dropout.
        attention_dropout: Dropout probability for within the attention
            layer.
        inner_dropout: Dropout probability for the first Dense layer in a
        two-layer feedforward network.
        attention_initializer: Initializer for kernels of attention layers.
            If set `None`, attention layers use kernel_initializer as
            initializer for kernel.
        attention_axes: axes over which the attention is applied. `None`
            means attention over all axes, but batch, heads, and features.
        **kwargs: keyword arguments.

    References:
        [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
        [BERT: Pre-training of Deep Bidirectional Transformers for Language
        Understanding](https://arxiv.org/abs/1810.04805)
    """

    def __init__(
        self,
        num_attention_heads,
        inner_size,
        inner_activation,
        output_range=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        use_bias=True,
        norm_first=False,
        norm_epsilon=1e-12,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        inner_dropout=0.0,
        attention_initializer=None,
        attention_axes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._num_heads = num_attention_heads
        self._inner_size = inner_size
        self._inner_activation = inner_activation
        self._attention_dropout = attention_dropout
        self._attention_dropout_rate = attention_dropout
        self._hidden_dropout = hidden_dropout
        self._hidden_dropout_rate = hidden_dropout
        self._output_range = output_range
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = keras.regularizers.get(
            activity_regularizer
        )
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._inner_dropout = inner_dropout
        if attention_initializer:
            self._attention_initializer = keras.initializers.get(
                attention_initializer
            )
        else:
            self._attention_initializer = self._kernel_initializer
        self._attention_axes = attention_axes

    def build(self, input_shape):
        if isinstance(input_shape, tf.TensorShape):
            input_tensor_shape = input_shape
        elif isinstance(input_shape, (list, tuple)):
            input_tensor_shape = tf.TensorShape(input_shape[0])
        else:
            raise ValueError(
                f"Unknown input shape type. Received: {type(input_shape)}"
            )
        einsum_equation = "abc,cd->abd"
        if len(input_tensor_shape.as_list()) > 3:
            einsum_equation = "...bc,cd->...bd"
        hidden_size = input_tensor_shape[-1]
        if hidden_size % self._num_heads != 0:
            raise ValueError(
                f"The input size {hidden_size} is not a multiple of the number "
                f"of attention heads {self._num_heads}"
            )
        self._attention_head_size = int(hidden_size // self._num_heads)
        common_kwargs = dict(
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        self._attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._attention_head_size,
            dropout=self._attention_dropout,
            use_bias=self._use_bias,
            kernel_initializer=self._attention_initializer,
            attention_axes=self._attention_axes,
            name="self_attention",
            **common_kwargs,
        )
        self._attention_dropout = keras.layers.Dropout(
            rate=self._hidden_dropout
        )
        # Use float32 in layernorm for numeric stability. It is probably safe in
        # mixed_float16, but we haven't validated this yet.
        self._attention_layer_norm = keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32,
        )
        self._intermediate_dense = keras.layers.experimental.EinsumDense(
            einsum_equation,
            output_shape=(None, self._inner_size),
            bias_axes="d",
            kernel_initializer=self._kernel_initializer,
            name="intermediate",
            **common_kwargs,
        )
        policy = keras.mixed_precision.global_policy()
        if policy.name == "mixed_bfloat16":
            # bfloat16 causes BERT with the LAMB optimizer to not converge
            # as well, so we use float32.
            # TODO(b/154538392): Investigate this.
            policy = tf.float32
        self._intermediate_activation_layer = keras.layers.Activation(
            self._inner_activation, dtype=policy
        )
        self._inner_dropout_layer = keras.layers.Dropout(
            rate=self._inner_dropout
        )
        self._output_dense = keras.layers.experimental.EinsumDense(
            einsum_equation,
            output_shape=(None, hidden_size),
            bias_axes="d",
            name="output",
            kernel_initializer=self._kernel_initializer,
            **common_kwargs,
        )
        self._hidden_dropout = keras.layers.Dropout(rate=self._hidden_dropout)
        # Use float32 in layernorm for numeric stability.
        self._output_layer_norm = keras.layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def get_config(self):
        config = {
            "num_attention_heads": self._num_heads,
            "inner_size": self._inner_size,
            "inner_activation": self._inner_activation,
            "hidden_dropout": self._hidden_dropout_rate,
            "attention_dropout": self._attention_dropout_rate,
            "output_range": self._output_range,
            "kernel_initializer": keras.initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self._bias_initializer
            ),
            "kernel_regularizer": keras.regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": keras.regularizers.serialize(
                self._bias_regularizer
            ),
            "activity_regularizer": keras.regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": keras.constraints.serialize(
                self._kernel_constraint
            ),
            "bias_constraint": keras.constraints.serialize(
                self._bias_constraint
            ),
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
            "inner_dropout": self._inner_dropout,
            "attention_initializer": keras.initializers.serialize(
                self._attention_initializer
            ),
            "attention_axes": self._attention_axes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, query, key_value=None, attention_mask=None):
        """Transformer self-attention encoder block call.

        Args:
            query: The query for the multi-head attention layer.
            key_value: Optional key/value tensor for multi-head attention. If
                none supplied, the query will also be used.
            attention_mask: Optional mask for the multi-head attention layer.

        Returns:
            An output tensor with the same dimensions as input/query tensor.
        """
        if self._output_range:
            if self._norm_first:
                source_tensor = query[:, 0 : self._output_range, :]
                query = self._attention_layer_norm(query)
                if key_value is not None:
                    key_value = self._attention_layer_norm(key_value)
            target_tensor = query[:, 0 : self._output_range, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 0 : self._output_range, :]
        else:
            if self._norm_first:
                source_tensor = query
                query = self._attention_layer_norm(query)
                if key_value is not None:
                    key_value = self._attention_layer_norm(key_value)
            target_tensor = query

        if key_value is None:
            key_value = query
        # TODO(mattdangerw): Use the build in masking mechanism.
        attention_output = self._attention_layer(
            query=target_tensor, value=key_value, attention_mask=attention_mask
        )
        attention_output = self._attention_dropout(attention_output)
        if self._norm_first:
            attention_output = source_tensor + attention_output
        else:
            attention_output = self._attention_layer_norm(
                target_tensor + attention_output
            )
        if self._norm_first:
            source_attention_output = attention_output
            attention_output = self._output_layer_norm(attention_output)
        inner_output = self._intermediate_dense(attention_output)
        inner_output = self._intermediate_activation_layer(inner_output)
        inner_output = self._inner_dropout_layer(inner_output)
        layer_output = self._output_dense(inner_output)
        layer_output = self._hidden_dropout(layer_output)

        if self._norm_first:
            return source_attention_output + layer_output

        # During mixed precision training, layer norm output is always fp32 for
        # now. Casts fp32 for the subsequent add.
        layer_output = tf.cast(layer_output, tf.float32)
        return self._output_layer_norm(layer_output + attention_output)


# TODO(mattdangerw): This class is needed for TPU friendly embeddings, we should
# remove it entirely and fix tf.keras.layers.Embedding as needed.
class OnDeviceEmbedding(keras.layers.Layer):
    """Performs an embedding lookup suitable for TPU devices.

    This layer uses either tf.gather or tf.one_hot to translate integer indices
    to float embeddings.

    Args:
        vocab_size: Number of elements in the vocabulary.
        embedding_width: Output size of the embedding layer.
        initializer: The initializer to use for the embedding weights. Defaults
            to "glorot_uniform".
        use_one_hot: Whether to use tf.one_hot over tf.gather for the embedding
            lookup. Defaults to False (that is, using tf.gather). Setting this
            option to True may improve performance, especially on small
            vocabulary sizes, but will generally require more memory.
      scale_factor: Whether to scale the output embeddings. Defaults to None
        (that is, not to scale). Setting this option to a float will let values
        in output embeddings multiplied by scale_factor.
    """

    def __init__(
        self,
        vocab_size,
        embedding_width,
        initializer="glorot_uniform",
        use_one_hot=False,
        scale_factor=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._vocab_size = vocab_size
        self._embedding_width = embedding_width
        self._initializer = initializer
        self._use_one_hot = use_one_hot
        self._scale_factor = scale_factor

    def get_config(self):
        config = {
            "vocab_size": self._vocab_size,
            "embedding_width": self._embedding_width,
            "initializer": self._initializer,
            "use_one_hot": self._use_one_hot,
            "scale_factor": self._scale_factor,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            "embeddings",
            shape=[self._vocab_size, self._embedding_width],
            initializer=self._initializer,
            dtype=tf.float32,
        )

        super().build(input_shape)

    def call(self, inputs):
        flat_inputs = tf.reshape(inputs, [-1])
        if self._use_one_hot:
            one_hot_data = tf.one_hot(
                flat_inputs, depth=self._vocab_size, dtype=self.compute_dtype
            )
            embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            embeddings = tf.gather(self.embeddings, flat_inputs)
        embeddings = tf.reshape(
            embeddings,
            tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0),
        )
        embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
        if self._scale_factor:
            embeddings *= self._scale_factor
        return embeddings


class BertModel(keras.Model):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default values for this object are taken from the BERT-Base
    implementation in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding".

    Args:
        vocab_size: The size of the token vocabulary.
        num_layers: The number of transformer layers.
        hidden_size: The size of the transformer hidden layers.
        hidden_dropout: Dropout probability for the post-attention and output
            dropout.
        num_attention_heads: The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        attention_dropout: The dropout rate to use for the attention layers
            within the transformer layers.
        inner_size: The output dimension of the first Dense layer in a two-layer
            feedforward network for each transformer.
        inner_activation: The activation for the first Dense layer in a
            two-layer feedforward network for each transformer.
        initializer_range: The initialzer range to use for a truncated normal
            initializer.
        max_sequence_length: The maximum sequence length that this encoder can
            consume. If None, max_sequence_length uses the value from sequence
            length. This determines the variable shape for positional
            embeddings.
        type_vocab_size: The number of types that the 'segment_ids' input can
            take.
        norm_first: Whether to normalize inputs to attention and intermediate
            dense layers. If set False, output of attention and intermediate
            dense layers is normalized.
    """

    def __init__(
        self,
        vocab_size,
        num_layers=12,
        hidden_size=768,
        hidden_dropout=0.1,
        num_attention_heads=12,
        attention_dropout=0.1,
        inner_size=3072,
        inner_activation="gelu",
        initializer_range=0.02,
        max_sequence_length=512,
        type_vocab_size=2,
        norm_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        activation = keras.activations.get(inner_activation)
        initializer = keras.initializers.TruncatedNormal(
            stddev=initializer_range
        )
        initializer = keras.initializers.get(initializer)

        self._embedding_layer = OnDeviceEmbedding(
            vocab_size=vocab_size,
            embedding_width=hidden_size,
            initializer=initializer,
            name="word_embeddings",
        )

        self._position_embedding_layer = keras_nlp.layers.PositionEmbedding(
            initializer=initializer,
            max_length=max_sequence_length,
            name="position_embedding",
        )

        self._type_embedding_layer = OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=hidden_size,
            initializer=initializer,
            use_one_hot=True,
            name="type_embeddings",
        )

        self._embedding_norm_layer = keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )

        self._embedding_dropout = keras.layers.Dropout(
            rate=hidden_dropout, name="embedding_dropout"
        )

        self._transformer_layers = []
        for i in range(num_layers):
            layer = TransformerEncoderBlock(
                num_attention_heads=num_attention_heads,
                inner_size=inner_size,
                inner_activation=inner_activation,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                norm_first=norm_first,
                kernel_initializer=initializer,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        self._config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "max_sequence_length": max_sequence_length,
            "type_vocab_size": type_vocab_size,
            "inner_size": inner_size,
            "inner_activation": keras.activations.serialize(activation),
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
            "initializer_range": initializer_range,
            "norm_first": norm_first,
        }
        self.inputs = dict(
            input_ids=keras.Input(shape=(None,), dtype=tf.int32),
            input_mask=keras.Input(shape=(None,), dtype=tf.int32),
            segment_ids=keras.Input(shape=(None,), dtype=tf.int32),
        )

    def call(self, inputs):
        if isinstance(inputs, dict):
            input_ids = inputs.get("input_ids")
            input_mask = inputs.get("input_mask")
            segment_ids = inputs.get("segment_ids")
        else:
            raise ValueError(f"Inputs should be a dict. Received: {inputs}.")

        word_embeddings = None
        word_embeddings = self._embedding_layer(input_ids)
        position_embeddings = self._position_embedding_layer(word_embeddings)
        type_embeddings = self._type_embedding_layer(segment_ids)

        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self._embedding_norm_layer(embeddings)
        embeddings = self._embedding_dropout(embeddings)

        attention_mask = make_attention_mask(embeddings, input_mask)

        x = embeddings
        for layer in self._transformer_layers:
            x = layer(x, attention_mask=attention_mask)
        return x

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def get_config(self):
        return dict(self._config)
