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

import tensorflow as tf
from tensorflow import keras


class SelfAttentionMask(tf.keras.layers.Layer):
    """Create 3D attention mask from a 2D tensor mask.

    inputs[0]: from_tensor: 2D or 3D Tensor of shape
        [batch_size, from_seq_length, ...].
    inputs[1]: to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """

    def call(self, inputs, to_mask=None):
        if isinstance(inputs, list) and to_mask is None:
            to_mask = inputs[1]
            inputs = inputs[0]
        from_shape = tf.shape(inputs)
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]

        to_shape = tf.shape(to_mask)
        to_seq_length = to_shape[1]

        to_mask = tf.cast(
            tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
            dtype=inputs.dtype,
        )

        # We don't assume that `from_tensor` is a mask (although it could be).
        # We don't actually care if we attend *from* padding tokens (only *to*
        # padding) tokens so we create a tensor of all ones.
        #
        # `broadcast_ones` = [batch_size, from_seq_length, 1]
        broadcast_ones = tf.ones(
            shape=[batch_size, from_seq_length, 1], dtype=inputs.dtype
        )

        # Here we broadcast along two dimensions to create the mask.
        mask = broadcast_ones * to_mask

        return mask


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """TransformerEncoderBlock layer.

    This layer implements the Transformer Encoder from
    "Attention Is All You Need". (https://arxiv.org/abs/1706.03762),
    which combines a `tf.keras.layers.MultiHeadAttention` layer with a
    two-layer feedforward network.

    References:
      [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
      [BERT: Pre-training of Deep Bidirectional Transformers for Language
       Understanding](https://arxiv.org/abs/1810.04805)
    """

    def __init__(
        self,
        num_attention_heads,
        inner_dim,
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
        output_dropout=0.0,
        attention_dropout=0.0,
        inner_dropout=0.0,
        attention_initializer=None,
        attention_axes=None,
        **kwargs
    ):
        """Initializes `TransformerEncoderBlock`.

        Args:
            num_attention_heads: Number of attention heads.
            inner_dim: The output dimension of the first Dense layer in a
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
            output_dropout: Dropout probability for the post-attention and
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
        """
        util.filter_kwargs(kwargs)
        super().__init__(**kwargs)

        self._num_heads = num_attention_heads
        self._inner_dim = inner_dim
        self._inner_activation = inner_activation
        self._attention_dropout = attention_dropout
        self._attention_dropout_rate = attention_dropout
        self._output_dropout = output_dropout
        self._output_dropout_rate = output_dropout
        self._output_range = output_range
        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = tf.keras.regularizers.get(
            activity_regularizer
        )
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._inner_dropout = inner_dropout
        if attention_initializer:
            self._attention_initializer = tf.keras.initializers.get(
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
                "The type of input shape argument is not supported, got: %s"
                % type(input_shape)
            )
        einsum_equation = "abc,cd->abd"
        if len(input_tensor_shape.as_list()) > 3:
            einsum_equation = "...bc,cd->...bd"
        hidden_size = input_tensor_shape[-1]
        if hidden_size % self._num_heads != 0:
            raise ValueError(
                "The input size (%d) is not a multiple of the number of "
                "attention heads (%d)" % (hidden_size, self._num_heads)
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
        self._attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._attention_head_size,
            dropout=self._attention_dropout,
            use_bias=self._use_bias,
            kernel_initializer=self._attention_initializer,
            attention_axes=self._attention_axes,
            name="self_attention",
            **common_kwargs
        )
        self._attention_dropout = tf.keras.layers.Dropout(
            rate=self._output_dropout
        )
        # Use float32 in layernorm for numeric stability.
        # It is probably safe in mixed_float16, but we haven't validated this
        # yet.
        self._attention_layer_norm = tf.keras.layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32,
        )
        self._intermediate_dense = tf.keras.layers.experimental.EinsumDense(
            einsum_equation,
            output_shape=(None, self._inner_dim),
            bias_axes="d",
            kernel_initializer=self._kernel_initializer,
            name="intermediate",
            **common_kwargs
        )
        policy = tf.keras.mixed_precision.global_policy()
        if policy.name == "mixed_bfloat16":
            # bfloat16 causes BERT with the LAMB optimizer to not converge
            # as well, so we use float32.
            # TODO(b/154538392): Investigate this.
            policy = tf.float32
        self._intermediate_activation_layer = tf.keras.layers.Activation(
            self._inner_activation, dtype=policy
        )
        self._inner_dropout_layer = tf.keras.layers.Dropout(
            rate=self._inner_dropout
        )
        self._output_dense = tf.keras.layers.experimental.EinsumDense(
            einsum_equation,
            output_shape=(None, hidden_size),
            bias_axes="d",
            name="output",
            kernel_initializer=self._kernel_initializer,
            **common_kwargs
        )
        self._output_dropout = tf.keras.layers.Dropout(
            rate=self._output_dropout
        )
        # Use float32 in layernorm for numeric stability.
        self._output_layer_norm = tf.keras.layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype=tf.float32,
        )

        super(TransformerEncoderBlock, self).build(input_shape)

    def get_config(self):
        config = {
            "num_attention_heads": self._num_heads,
            "inner_dim": self._inner_dim,
            "inner_activation": self._inner_activation,
            "output_dropout": self._output_dropout_rate,
            "attention_dropout": self._attention_dropout_rate,
            "output_range": self._output_range,
            "kernel_initializer": tf.keras.initializers.serialize(
                self._kernel_initializer
            ),
            "bias_initializer": tf.keras.initializers.serialize(
                self._bias_initializer
            ),
            "kernel_regularizer": tf.keras.regularizers.serialize(
                self._kernel_regularizer
            ),
            "bias_regularizer": tf.keras.regularizers.serialize(
                self._bias_regularizer
            ),
            "activity_regularizer": tf.keras.regularizers.serialize(
                self._activity_regularizer
            ),
            "kernel_constraint": tf.keras.constraints.serialize(
                self._kernel_constraint
            ),
            "bias_constraint": tf.keras.constraints.serialize(
                self._bias_constraint
            ),
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
            "inner_dropout": self._inner_dropout,
            "attention_initializer": tf.keras.initializers.serialize(
                self._attention_initializer
            ),
            "attention_axes": self._attention_axes,
        }
        base_config = super(TransformerEncoderBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Transformer self-attention encoder block call.

        Args:
            inputs: a single tensor or a list of tensors.
                `input tensor` as the single sequence of embeddings.
                [`input tensor`, `attention mask`] to have the additional
                    attention mask.
                [`query tensor`, `key value tensor`, `attention mask`] to have
                    separate input streams for the query, and key/value to the
                    multi-head attention.

        Returns:
          An output tensor with the same dimensions as input/query tensor.
        """
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 2:
                input_tensor, attention_mask = inputs
                key_value = None
            elif len(inputs) == 3:
                input_tensor, key_value, attention_mask = inputs
            else:
                raise ValueError(
                    "Unexpected inputs to %s with length at %d"
                    % (self.__class__, len(inputs))
                )
        else:
            input_tensor, key_value, attention_mask = (inputs, None, None)

        if self._output_range:
            if self._norm_first:
                source_tensor = input_tensor[:, 0 : self._output_range, :]
                input_tensor = self._attention_layer_norm(input_tensor)
                if key_value is not None:
                    key_value = self._attention_layer_norm(key_value)
            target_tensor = input_tensor[:, 0 : self._output_range, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 0 : self._output_range, :]
        else:
            if self._norm_first:
                source_tensor = input_tensor
                input_tensor = self._attention_layer_norm(input_tensor)
                if key_value is not None:
                    key_value = self._attention_layer_norm(key_value)
            target_tensor = input_tensor

        if key_value is None:
            key_value = input_tensor
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
        layer_output = self._output_dropout(layer_output)

        if self._norm_first:
            return source_attention_output + layer_output

        # During mixed precision training, layer norm output is always fp32 for
        # now. Casts fp32 for the subsequent add.
        layer_output = tf.cast(layer_output, tf.float32)
        return self._output_layer_norm(layer_output + attention_output)


class PositionEmbedding(tf.keras.layers.Layer):
    """Creates a positional embedding.

    Example:
    ```python
    position_embedding = PositionEmbedding(max_length=100)
    inputs = tf.keras.Input((100, 32), dtype=tf.float32)
    outputs = position_embedding(inputs)
    ```

    Args:
        max_length: The maximum size of the dynamic sequence.
        initializer: The initializer to use for the embedding weights. Defaults
            to "glorot_uniform".
        seq_axis: The axis of the input tensor where we add the embeddings.

    Reference: This layer creates a positional embedding as described in
    [BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding](https://arxiv.org/abs/1810.04805).
    """

    def __init__(
        self, max_length, initializer="glorot_uniform", seq_axis=1, **kwargs
    ):

        super(PositionEmbedding, self).__init__(**kwargs)
        if max_length is None:
            raise ValueError("`max_length` must be an Integer, not `None`.")
        self._max_length = max_length
        self._initializer = tf.keras.initializers.get(initializer)
        self._seq_axis = seq_axis

    def get_config(self):
        config = {
            "max_length": self._max_length,
            "initializer": tf.keras.initializers.serialize(self._initializer),
            "seq_axis": self._seq_axis,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        dimension_list = input_shape.as_list()
        width = dimension_list[-1]
        weight_sequence_length = self._max_length

        self._position_embeddings = self.add_weight(
            "embeddings",
            shape=[weight_sequence_length, width],
            initializer=self._initializer,
        )

        super(PositionEmbedding, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        actual_seq_len = input_shape[self._seq_axis]
        position_embeddings = self._position_embeddings[:actual_seq_len, :]
        new_shape = [1 for _ in inputs.get_shape().as_list()]
        new_shape[self._seq_axis] = actual_seq_len
        new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
        position_embeddings = tf.reshape(position_embeddings, new_shape)
        return tf.broadcast_to(position_embeddings, input_shape)


class OnDeviceEmbedding(tf.keras.layers.Layer):
    """Performs an embedding lookup suitable for accelerator devices.

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
            (that is, not to scale). Setting this option to a float will let
            values in output embeddings multiplied by scale_factor.
    """

    def __init__(
        self,
        vocab_size,
        embedding_width,
        initializer="glorot_uniform",
        use_one_hot=False,
        scale_factor=None,
        **kwargs
    ):

        super(OnDeviceEmbedding, self).__init__(**kwargs)
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
        base_config = super(OnDeviceEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            "embeddings",
            shape=[self._vocab_size, self._embedding_width],
            initializer=self._initializer,
            dtype=tf.float32,
        )

        super(OnDeviceEmbedding, self).build(input_shape)

    def call(self, inputs):
        flat_inputs = tf.reshape(inputs, [-1])
        if self._use_one_hot:
            dtype = self._compute_dtype
            if not tf.dtypes.as_dtype(dtype).is_floating:
                # TensorFlow 1 compatibility. In TF1, self._compute_dtype is
                # int32 instead of a floating-point dtype, as the dtype is
                # inferred from the dtype of the inputs
                dtype = tf.float32
            one_hot_data = tf.one_hot(
                flat_inputs, depth=self._vocab_size, dtype=dtype
            )
            embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            embeddings = tf.gather(self.embeddings, flat_inputs)

        embeddings = tf.reshape(
            embeddings,
            # Work around b/142213824: prefer concat to shape over a Python
            # list.
            tf.concat([tf.shape(inputs), [self._embedding_width]], axis=0),
        )
        embeddings.set_shape(inputs.shape.as_list() + [self._embedding_width])
        if self._scale_factor:
            embeddings *= self._scale_factor
        return embeddings

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def embedding_width(self):
        return self._embedding_width


class BertEncoderV2(tf.keras.layers.Layer):
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
        hidden_size: The size of the transformer hidden layers.
        num_layers: The number of transformer layers.
        num_attention_heads: The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        max_sequence_length: The maximum sequence length that this encoder can
            consume. If None, max_sequence_length uses the value from sequence
            length. This determines the variable shape for positional
            embeddings.
        type_vocab_size: The number of types that the 'type_ids' input can take.
        inner_dim: The output dimension of the first Dense layer in a two-layer
          feedforward network for each transformer.
        inner_activation: The activation for the first Dense layer in a
            two-layer feedforward network for each transformer.
        output_dropout: Dropout probability for the post-attention and output
          dropout.
        attention_dropout: The dropout rate to use for the attention layers
            within the transformer layers.
        initializer: The initialzer to use for all weights in this encoder.
        output_range: The sequence output range, [0, output_range), by slicing
            the target sequence of the last transformer layer. `None` means the
            entire target sequence will attend to the source sequence, which
            yields the full output.
        embedding_width: The width of the word embeddings. If the embedding
            width is not equal to hidden size, embedding parameters will be
            factorized into two matrices in the shape of ['vocab_size',
            'embedding_width'] and ['embedding_width', 'hidden_size']
            ('embedding_width' is usually much smaller than 'hidden_size').
        embedding_layer: An optional Layer instance which will be called to
            generate embeddings for the input word IDs.
        norm_first: Whether to normalize inputs to attention and intermediate
            dense layers. If set False, output of attention and intermediate
            dense layers is normalized.
        with_dense_inputs: Whether to accept dense embeddings as the input.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_attention_heads: int = 12,
        max_sequence_length: int = 512,
        type_vocab_size: int = 16,
        inner_dim: int = 3072,
        inner_activation: _Activation = _approx_gelu,
        output_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        initializer: _Initializer = tf.keras.initializers.TruncatedNormal(
            stddev=0.02
        ),
        output_range: Optional[int] = None,
        embedding_width: Optional[int] = None,
        embedding_layer: Optional[tf.keras.layers.Layer] = None,
        norm_first: bool = False,
        with_dense_inputs: bool = False,
        **kwargs
    ):
        # Pops kwargs that are used in V1 implementation.
        if "dict_outputs" in kwargs:
            kwargs.pop("dict_outputs")
        if "return_all_encoder_outputs" in kwargs:
            kwargs.pop("return_all_encoder_outputs")
        if "intermediate_size" in kwargs:
            inner_dim = kwargs.pop("intermediate_size")
        if "activation" in kwargs:
            inner_activation = kwargs.pop("activation")
        if "dropout_rate" in kwargs:
            output_dropout = kwargs.pop("dropout_rate")
        if "attention_dropout_rate" in kwargs:
            attention_dropout = kwargs.pop("attention_dropout_rate")
        super().__init__(**kwargs)

        activation = tf.keras.activations.get(inner_activation)
        initializer = tf.keras.initializers.get(initializer)

        if embedding_width is None:
            embedding_width = hidden_size

        if embedding_layer is None:
            self._embedding_layer = layers.OnDeviceEmbedding(
                vocab_size=vocab_size,
                embedding_width=embedding_width,
                initializer=initializer,
                name="word_embeddings",
            )
        else:
            self._embedding_layer = embedding_layer

        self._position_embedding_layer = layers.PositionEmbedding(
            initializer=initializer,
            max_length=max_sequence_length,
            name="position_embedding",
        )

        self._type_embedding_layer = layers.OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=embedding_width,
            initializer=initializer,
            use_one_hot=True,
            name="type_embeddings",
        )

        self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )

        self._embedding_dropout = tf.keras.layers.Dropout(
            rate=output_dropout, name="embedding_dropout"
        )

        # We project the 'embedding' output to 'hidden_size' if it is not
        # already 'hidden_size'.
        self._embedding_projection = None
        if embedding_width != hidden_size:
            self._embedding_projection = (
                tf.keras.layers.experimental.EinsumDense(
                    "...x,xy->...y",
                    output_shape=hidden_size,
                    bias_axes="y",
                    kernel_initializer=initializer,
                    name="embedding_projection",
                )
            )

        self._transformer_layers = []
        self._attention_mask_layer = layers.SelfAttentionMask(
            name="self_attention_mask"
        )
        for i in range(num_layers):
            layer = layers.TransformerEncoderBlock(
                num_attention_heads=num_attention_heads,
                inner_dim=inner_dim,
                inner_activation=inner_activation,
                output_dropout=output_dropout,
                attention_dropout=attention_dropout,
                norm_first=norm_first,
                output_range=output_range if i == num_layers - 1 else None,
                kernel_initializer=initializer,
                name="transformer/layer_%d" % i,
            )
            self._transformer_layers.append(layer)

        self._pooler_layer = tf.keras.layers.Dense(
            units=hidden_size,
            activation="tanh",
            kernel_initializer=initializer,
            name="pooler_transform",
        )

        self._config = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "max_sequence_length": max_sequence_length,
            "type_vocab_size": type_vocab_size,
            "inner_dim": inner_dim,
            "inner_activation": tf.keras.activations.serialize(activation),
            "output_dropout": output_dropout,
            "attention_dropout": attention_dropout,
            "initializer": tf.keras.initializers.serialize(initializer),
            "output_range": output_range,
            "embedding_width": embedding_width,
            "embedding_layer": embedding_layer,
            "norm_first": norm_first,
            "with_dense_inputs": with_dense_inputs,
        }
        if with_dense_inputs:
            self.inputs = dict(
                input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
                input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
                input_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
                dense_inputs=tf.keras.Input(
                    shape=(None, embedding_width), dtype=tf.float32
                ),
                dense_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
                dense_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
            )
        else:
            self.inputs = dict(
                input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
                input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
                input_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
            )

    def call(self, inputs):
        word_embeddings = None
        if isinstance(inputs, dict):
            word_ids = inputs.get("input_word_ids")
            mask = inputs.get("input_mask")
            type_ids = inputs.get("input_type_ids")
            word_embeddings = inputs.get("input_word_embeddings", None)

            dense_inputs = inputs.get("dense_inputs", None)
            dense_mask = inputs.get("dense_mask", None)
            dense_type_ids = inputs.get("dense_type_ids", None)
        else:
            raise ValueError("Unexpected inputs type to %s." % self.__class__)

        if word_embeddings is None:
            word_embeddings = self._embedding_layer(word_ids)

        if dense_inputs is not None:
            # Concat the dense embeddings at sequence end.
            word_embeddings = tf.concat([word_embeddings, dense_inputs], axis=1)
            type_ids = tf.concat([type_ids, dense_type_ids], axis=1)
            mask = tf.concat([mask, dense_mask], axis=1)

        # absolute position embeddings.
        position_embeddings = self._position_embedding_layer(word_embeddings)
        type_embeddings = self._type_embedding_layer(type_ids)

        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self._embedding_norm_layer(embeddings)
        embeddings = self._embedding_dropout(embeddings)

        if self._embedding_projection is not None:
            embeddings = self._embedding_projection(embeddings)

        attention_mask = self._attention_mask_layer(embeddings, mask)

        encoder_outputs = []
        x = embeddings
        for layer in self._transformer_layers:
            x = layer([x, attention_mask])
            encoder_outputs.append(x)

        last_encoder_output = encoder_outputs[-1]
        first_token_tensor = last_encoder_output[:, 0, :]
        pooled_output = self._pooler_layer(first_token_tensor)

        return dict(
            sequence_output=encoder_outputs[-1],
            pooled_output=pooled_output,
            encoder_outputs=encoder_outputs,
        )

    def get_embedding_table(self):
        return self._embedding_layer.embeddings

    def get_embedding_layer(self):
        return self._embedding_layer

    def get_config(self):
        return dict(self._config)

    @property
    def transformer_layers(self):
        """List of Transformer layers in the encoder."""
        return self._transformer_layers

    @property
    def pooler_layer(self):
        """The pooler dense layer after the transformer layers."""
        return self._pooler_layer

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if (
            "embedding_layer" in config
            and config["embedding_layer"] is not None
        ):
            warn_string = (
                "You are reloading a model that was saved with a "
                "potentially-shared embedding layer object. If you contine to "
                "train this model, the embedding layer will no longer be "
                "shared. To work around this, load the model outside of the "
                "Keras API."
            )
            print("WARNING: " + warn_string)
            logging.warn(warn_string)

        return cls(**config)
