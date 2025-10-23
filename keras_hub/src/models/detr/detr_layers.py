import math

from keras import Layer
from keras import activations
from keras import layers
from keras import ops


class DetrFrozenBatchNormalization(Layer):
    """BatchNormalization with fixed affine + batch stats.
    Based on https://github.com/facebookresearch/detr/blob/master/models/backbone.py.
    """

    def __init__(self, num_features, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.epsilon = epsilon

    def build(self):
        self.weight = self.add_weight(
            shape=(self.num_features,),
            initializer="ones",
            trainable=False,
            name="weight",
        )
        self.bias = self.add_weight(
            shape=(self.num_features,),
            initializer="zeros",
            trainable=False,
            name="bias",
        )
        self.running_mean = self.add_weight(
            shape=(self.num_features,),
            initializer="zeros",
            trainable=False,
            name="running_mean",
        )
        self.running_var = self.add_weight(
            shape=(self.num_features,),
            initializer="ones",
            trainable=False,
            name="running_var",
        )

    def call(self, inputs):
        weight = ops.reshape(self.weight, (1, 1, 1, -1))
        bias = ops.reshape(self.bias, (1, 1, 1, -1))
        running_mean = ops.reshape(self.running_mean, (1, 1, 1, -1))
        running_var = ops.reshape(self.running_var, (1, 1, 1, -1))

        scale = weight * ops.rsqrt(running_var + self.epsilon)
        bias = bias - running_mean * scale
        return inputs * scale + bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_features": self.num_features, "epsilon": self.epsilon}
        )
        return config


class DetrSinePositionEmbedding(Layer):
    def __init__(
        self, embedding_dim=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, inputs):
        if input is None:
            raise ValueError("No pixel mask provided")

        y_embed = ops.cumsum(inputs, axis=1, dtype="float32")
        x_embed = ops.cumsum(inputs, axis=2, dtype="float32")
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = ops.arange(self.embedding_dim, dtype="float32")
        dim_t = self.temperature ** (
            2 * ops.floor(dim_t / 2) / self.embedding_dim
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ops.stack(
            (ops.sin(pos_x[:, :, :, ::2]), ops.cos(pos_x[:, :, :, 1::2])),
            axis=4,
        )
        pos_y = ops.stack(
            (ops.sin(pos_y[:, :, :, ::2]), ops.cos(pos_y[:, :, :, 1::2])),
            axis=4,
        )

        shape_x = list(ops.shape(pos_x))
        pos_x = ops.reshape(pos_x, shape_x[:-2] + [-1])
        shape_y = list(ops.shape(pos_y))
        pos_y = ops.reshape(pos_y, shape_y[:-2] + [-1])

        pos = ops.concatenate((pos_y, pos_x), axis=3)
        pos = ops.transpose(pos, [0, 3, 1, 2])
        return pos

    def compute_output_shape(self, input_shape):
        """Compute output shape: (batch, embedding_dim*2, height, width)."""
        batch_size, height, width = input_shape
        return (batch_size, self.embedding_dim * 2, height, width)


# Functional version of the code based on https://github.com/tensorflow/models/blob/master/official/projects/detr/modeling/detr.py


def position_embedding_sine(
    attention_mask,
    num_pos_features=256,
    temperature=10000.0,
    normalize=True,
    scale=2 * math.pi,
):
    """Sine-based positional embeddings for 2D images.

    Args:
      attention_mask: a `bool` Tensor specifying the size of the input image to
        the Transformer and which elements are padded, of size [batch_size,
        height, width]
      num_pos_features: a `int` specifying the number of positional features,
        should be equal to the hidden size of the Transformer network
      temperature: a `float` specifying the temperature of the positional
        embedding. Any type that is converted to a `float` can also be accepted.
      normalize: a `bool` determining whether the positional embeddings
        should be normalized between [0, scale] before application
        of the sine and cos functions.
      scale: a `float` if normalize is True specifying the
        scale embeddings before application of the embedding function.

    Returns:
      embeddings: a `float` tensor of the same shape as input_tensor specifying
        the positional embeddings based on sine features.
    """
    if num_pos_features % 2 != 0:
        raise ValueError(
            "Number of embedding features (num_pos_features) must be even when "
            "column and row embeddings are concatenated."
        )
    num_pos_features = num_pos_features // 2

    # Produce row and column embeddings based on total size of the image
    # <tf.float>[batch_size, height, width]
    row_embedding = ops.cumsum(attention_mask, 1)
    col_embedding = ops.cumsum(attention_mask, 2)

    if normalize:
        eps = 1e-6
        row_embedding = row_embedding / (row_embedding[:, -1:, :] + eps) * scale
        col_embedding = col_embedding / (col_embedding[:, :, -1:] + eps) * scale

    dim_t = ops.arange(num_pos_features, dtype=row_embedding.dtype)
    dim_t = ops.power(temperature, 2 * (dim_t // 2) / num_pos_features)

    # Creates positional embeddings for each row and column position
    # <tf.float>[batch_size, height, width, num_pos_features]
    pos_row = ops.expand_dims(row_embedding, -1) / dim_t
    pos_col = ops.expand_dims(col_embedding, -1) / dim_t
    pos_row = ops.stack(
        [ops.sin(pos_row[:, :, :, 0::2]), ops.cos(pos_row[:, :, :, 1::2])],
        axis=4,
    )
    pos_col = ops.stack(
        [ops.sin(pos_col[:, :, :, 0::2]), ops.cos(pos_col[:, :, :, 1::2])],
        axis=4,
    )

    # Reshape to flatten the last two dimensions
    # pos_row/pos_col shape: (batch, height, width, num_pos_features/2, 2)
    # We want: (batch, height, width, num_pos_features)
    shape = list(ops.shape(pos_row))
    final_shape = shape[:-2] + [-1]
    pos_row = ops.reshape(pos_row, final_shape)
    pos_col = ops.reshape(pos_col, final_shape)
    output = ops.concatenate([pos_row, pos_col], -1)

    return output


class DetrTransformerEncoder(layers.Layer):
    """
    Adapted from
    https://github.com/tensorflow/models/blob/master/official/projects/detr/modeling/transformer.py
    """

    def __init__(
        self,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        activation="relu",
        dropout_rate=0.0,
        attentiondropout_rate=0.0,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.attentiondropout_rate = attentiondropout_rate
        self.use_bias = use_bias
        self.norm_first = norm_first
        self.norm_epsilon = norm_epsilon
        self.intermediate_dropout = intermediate_dropout

    def build(self, input_shape):
        self.encoder_layers = []
        for i in range(self.num_layers):
            self.encoder_layers.append(
                DetrTransformerEncoderBlock(
                    num_attention_heads=self.num_attention_heads,
                    inner_dim=self.intermediate_size,
                    inner_activation=self.activation,
                    output_dropout=self.dropout_rate,
                    attention_dropout=self.attentiondropout_rate,
                    use_bias=self.use_bias,
                    norm_first=self.norm_first,
                    norm_epsilon=self.norm_epsilon,
                    inner_dropout=self.intermediate_dropout,
                )
            )
        self.output_normalization = layers.LayerNormalization(
            epsilon=self.norm_epsilon, dtype="float32"
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "attentiondropout_rate": self.attentiondropout_rate,
            "use_bias": self.use_bias,
            "norm_first": self.norm_first,
            "norm_epsilon": self.norm_epsilon,
            "intermediate_dropout": self.intermediate_dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(
        self, encoder_inputs, attention_mask=None, pos_embed=None, training=None
    ):
        for layer_idx in range(self.num_layers):
            encoder_inputs = self.encoder_layers[layer_idx](
                [encoder_inputs, attention_mask, pos_embed], training=training
            )

        return encoder_inputs


class DetrTransformerEncoderBlock(layers.Layer):
    """
    Adapted from
    https://github.com/tensorflow/models/blob/master/official/projects/detr/modeling/transformer.py
    """

    def __init__(
        self,
        num_attention_heads,
        inner_dim,
        inner_activation,
        use_bias=True,
        norm_first=False,
        norm_epsilon=1e-12,
        output_dropout=0.0,
        attention_dropout=0.0,
        inner_dropout=0.0,
        attention_axes=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_attention_heads
        self.inner_dim = inner_dim
        self.inner_activation = inner_activation
        self.attention_dropout = attention_dropout
        self.attentiondropout_rate = attention_dropout
        self.output_dropout = output_dropout
        self.outputdropout_rate = output_dropout
        self.use_bias = use_bias
        self.norm_first = norm_first
        self.norm_epsilon = norm_epsilon
        self.inner_dropout = inner_dropout
        self.attention_axes = attention_axes

    def build(self, input_shape):
        hidden_size = input_shape[-1][-1]
        if hidden_size % self.num_heads != 0:
            raise ValueError(
                "The input size (%d) is not a multiple of "
                "the number of attention heads (%d)"
                % (hidden_size, self.num_heads)
            )
        self.attention_head_size = int(hidden_size // self.num_heads)

        self.attention_layer = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.attention_head_size,
            dropout=self.attention_dropout,
            use_bias=self.use_bias,
            attention_axes=self.attention_axes,
            name="self_attention",
        )
        self.attention_dropout = layers.Dropout(rate=self.output_dropout)
        self.attention_layer_norm = layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self.norm_epsilon,
            dtype="float32",
        )
        self.intermediate_dense = layers.Dense(
            self.inner_dim,
            activation=self.inner_activation,
            use_bias=self.use_bias,
            name="intermediate",
        )

        self.inner_dropout_layer = layers.Dropout(rate=self.inner_dropout)
        self.output_dense = layers.Dense(
            hidden_size,
            use_bias=self.use_bias,
            name="output",
        )
        self.output_dropout = layers.Dropout(rate=self.output_dropout)
        self.output_layer_norm = layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self.norm_epsilon,
            dtype="float32",
        )

        super().build(input_shape)

    def get_config(self):
        config = {
            "num_attention_heads": self.num_heads,
            "inner_dim": self.inner_dim,
            "inner_activation": self.inner_activation,
            "output_dropout": self.outputdropout_rate,
            "attention_dropout": self.attentiondropout_rate,
            "use_bias": self.use_bias,
            "norm_first": self.norm_first,
            "norm_epsilon": self.norm_epsilon,
            "inner_dropout": self.inner_dropout,
            "attention_axes": self.attention_axes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        input_tensor, attention_mask, pos_embed = inputs

        if self.norm_first:
            source_tensor = input_tensor
            input_tensor = self.attention_layer_norm(input_tensor)
        target_tensor = input_tensor

        attention_output = self.attention_layer(
            query=target_tensor + pos_embed,
            key=input_tensor + pos_embed,
            value=input_tensor,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = self.attention_dropout(
            attention_output, training=training
        )
        if self.norm_first:
            attention_output = source_tensor + attention_output
        else:
            attention_output = self.attention_layer_norm(
                target_tensor + attention_output
            )
        if self.norm_first:
            source_attention_output = attention_output
            attention_output = self.output_layer_norm(attention_output)

        inner_output = self.intermediate_dense(attention_output)
        inner_output = self.inner_dropout_layer(inner_output, training=training)
        layer_output = self.output_dense(inner_output)
        layer_output = self.output_dropout(layer_output, training=training)

        if self.norm_first:
            return source_attention_output + layer_output

        return self.output_layer_norm(layer_output + attention_output)


class DetrTransformerDecoder(layers.Layer):
    """
    Adapted from
    https://github.com/tensorflow/models/blob/master/official/projects/detr/modeling/transformer.py
    """

    def __init__(
        self,
        num_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        activation="relu",
        dropout_rate=0.0,
        attentiondropout_rate=0.0,
        use_bias=True,
        norm_first=False,
        norm_epsilon=1e-5,
        intermediate_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.attentiondropout_rate = attentiondropout_rate
        self.use_bias = use_bias
        self.norm_first = norm_first
        self.norm_epsilon = norm_epsilon
        self.intermediate_dropout = intermediate_dropout

    def build(self, input_shape):
        self.decoder_layers = []
        for i in range(self.num_layers):
            self.decoder_layers.append(
                DetrTransformerDecoderBlock(
                    num_attention_heads=self.num_attention_heads,
                    intermediate_size=self.intermediate_size,
                    intermediate_activation=self.activation,
                    dropout_rate=self.dropout_rate,
                    attentiondropout_rate=self.attentiondropout_rate,
                    use_bias=self.use_bias,
                    norm_first=self.norm_first,
                    norm_epsilon=self.norm_epsilon,
                    intermediate_dropout=self.intermediate_dropout,
                    name=("layer_%d" % i),
                )
            )
        self.output_normalization = layers.LayerNormalization(
            epsilon=self.norm_epsilon, dtype="float32"
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "attentiondropout_rate": self.attentiondropout_rate,
            "use_bias": self.use_bias,
            "norm_first": self.norm_first,
            "norm_epsilon": self.norm_epsilon,
            "intermediate_dropout": self.intermediate_dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(
        self,
        target,
        memory,
        self_attention_mask=None,
        cross_attention_mask=None,
        return_all_decoder_outputs=False,
        input_pos_embed=None,
        memory_pos_embed=None,
        training=None,
    ):
        output_tensor = target
        decoder_outputs = []
        for layer_idx in range(self.num_layers):
            transformer_inputs = [
                output_tensor,
                memory,
                cross_attention_mask,
                self_attention_mask,
                input_pos_embed,
                memory_pos_embed,
            ]

            output_tensor = self.decoder_layers[layer_idx](
                transformer_inputs, training=training
            )

            if return_all_decoder_outputs:
                decoder_outputs.append(self.output_normalization(output_tensor))

        if return_all_decoder_outputs:
            return decoder_outputs
        else:
            return self.output_normalization(output_tensor)

    def compute_output_shape(self, input_shape):
        # input_shape is for target: (batch, seq_len, hidden_dim)
        return input_shape


class DetrTransformerDecoderBlock(layers.Layer):
    """
    Adapted from
    https://github.com/tensorflow/models/blob/master/official/projects/detr/modeling/transformer.py
    """

    def __init__(
        self,
        num_attention_heads,
        intermediate_size,
        intermediate_activation,
        dropout_rate=0.0,
        attentiondropout_rate=0.0,
        use_bias=True,
        norm_first=False,
        norm_epsilon=1e-5,
        intermediate_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_activation = activations.get(intermediate_activation)
        self.dropout_rate = dropout_rate
        self.attentiondropout_rate = attentiondropout_rate

        self.use_bias = use_bias
        self.norm_first = norm_first
        self.norm_epsilon = norm_epsilon
        self.intermediate_dropout = intermediate_dropout

    def build(self, input_shape):
        # List of lists
        input_shape = input_shape[0]
        if len(input_shape) != 3:
            raise ValueError(
                "TransformerLayer expects a three-dimensional input of "
                "shape [batch, sequence, width]."
            )
        hidden_size = input_shape[2]
        if hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the "
                "number of attention heads (%d)"
                % (hidden_size, self.num_attention_heads)
            )
        self.attention_head_size = int(hidden_size) // self.num_attention_heads

        # Self attention.
        self.self_attention = layers.MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.attention_head_size,
            dropout=self.attentiondropout_rate,
            use_bias=self.use_bias,
            name="self_attention",
        )
        self.self_attention_output_dense = layers.EinsumDense(
            "abc,cd->abd",
            output_shape=(None, hidden_size),
            bias_axes="d",
            name="output",
        )
        self.self_attention_dropout = layers.Dropout(rate=self.dropout_rate)
        self.self_attention_layer_norm = layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self.norm_epsilon,
            dtype="float32",
        )
        # Encoder-decoder attention.
        self.encdec_attention = layers.MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.attention_head_size,
            dropout=self.attentiondropout_rate,
            output_shape=hidden_size,
            use_bias=self.use_bias,
            name="encdec",
        )

        self.encdec_attention_dropout = layers.Dropout(rate=self.dropout_rate)
        self.encdec_attention_layer_norm = layers.LayerNormalization(
            name="encdec_output_layer_norm",
            axis=-1,
            epsilon=self.norm_epsilon,
            dtype="float32",
        )

        # Feed-forward projection.
        self.intermediate_dense = layers.EinsumDense(
            "abc,cd->abd",
            output_shape=(None, self.intermediate_size),
            bias_axes="d",
            name="intermediate",
        )
        self.intermediate_activation_layer = layers.Activation(
            self.intermediate_activation
        )
        self.intermediate_dropout_layer = layers.Dropout(
            rate=self.intermediate_dropout
        )
        self.output_dense = layers.EinsumDense(
            "abc,cd->abd",
            output_shape=(None, hidden_size),
            bias_axes="d",
            name="output",
        )
        self.output_dropout = layers.Dropout(rate=self.dropout_rate)
        self.output_layer_norm = layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self.norm_epsilon,
            dtype="float32",
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "dropout_rate": self.dropout_rate,
            "attentiondropout_rate": self.attentiondropout_rate,
            "use_bias": self.use_bias,
            "norm_first": self.norm_first,
            "norm_epsilon": self.norm_epsilon,
            "intermediate_dropout": self.intermediate_dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        (
            input_tensor,
            memory,
            attention_mask,
            self_attention_mask,
            input_pos_embed,
            memory_pos_embed,
        ) = inputs
        source_tensor = input_tensor
        if self.norm_first:
            input_tensor = self.self_attention_layer_norm(input_tensor)
        self_attention_output = self.self_attention(
            query=input_tensor + input_pos_embed,
            key=input_tensor + input_pos_embed,
            value=input_tensor,
            attention_mask=self_attention_mask,
            training=training,
        )
        self_attention_output = self.self_attention_dropout(
            self_attention_output, training=training
        )
        if self.norm_first:
            self_attention_output = source_tensor + self_attention_output
        else:
            self_attention_output = self.self_attention_layer_norm(
                input_tensor + self_attention_output
            )
        if self.norm_first:
            source_self_attention_output = self_attention_output
            self_attention_output = self.encdec_attention_layer_norm(
                self_attention_output
            )
        cross_attn_inputs = dict(
            query=self_attention_output + input_pos_embed,
            key=memory + memory_pos_embed,
            value=memory,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = self.encdec_attention(**cross_attn_inputs)
        attention_output = self.encdec_attention_dropout(
            attention_output, training=training
        )
        if self.norm_first:
            attention_output = source_self_attention_output + attention_output
        else:
            attention_output = self.encdec_attention_layer_norm(
                self_attention_output + attention_output
            )
        if self.norm_first:
            source_attention_output = attention_output
            attention_output = self.output_layer_norm(attention_output)

        intermediate_output = self.intermediate_dense(attention_output)
        intermediate_output = self.intermediate_activation_layer(
            intermediate_output
        )
        intermediate_output = self.intermediate_dropout_layer(
            intermediate_output, training=training
        )
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output, training=training)
        if self.norm_first:
            layer_output = source_attention_output + layer_output
        else:
            layer_output = self.output_layer_norm(
                layer_output + attention_output
            )
        return layer_output

    def compute_output_shape(self, input_shape):
        # input_shape is a list: [input_tensor_shape, memory_shape, ...]
        # Return shape of input_tensor (unchanged)
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape


class CreateSelfAttentionMask(layers.Layer):
    """Creates self-attention mask of ones for decoder queries."""

    def __init__(self, num_queries, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

    def call(self, inputs):
        batch_size = ops.shape(inputs)[0]
        return ops.ones((batch_size, self.num_queries, self.num_queries))

    def compute_output_shape(self, input_shape):
        if input_shape is None:
            return (None, self.num_queries, self.num_queries)
        batch_size = input_shape[0]
        return (batch_size, self.num_queries, self.num_queries)

    def get_config(self):
        config = super().get_config()
        config.update({"num_queries": self.num_queries})
        return config


class CreateCrossAttentionMask(layers.Layer):
    """Tiles encoder mask for cross-attention in decoder."""

    def __init__(self, num_queries, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

    def call(self, mask):
        # mask shape: (batch, seq_len)
        # output: (batch, num_queries, seq_len)
        return ops.tile(ops.expand_dims(mask, axis=1), (1, self.num_queries, 1))

    def compute_output_shape(self, input_shape):
        if input_shape is None:
            return (None, self.num_queries, None)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        return (batch_size, self.num_queries, seq_len)

    def get_config(self):
        config = super().get_config()
        config.update({"num_queries": self.num_queries})
        return config


class DetrQueryEmbedding(layers.Layer):
    """Learnable object query embeddings for DETR decoder.

    Creates a set of learnable embeddings that serve as object queries
    in the DETR decoder. Each query will predict one object (or no-object).

    Args:
        num_queries: int. Number of object queries.
        hidden_dim: int. Dimensionality of each query embedding.
        **kwargs: Additional arguments passed to Layer.
    """

    def __init__(self, num_queries, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.query_embed = self.add_weight(
            name="query_embed",
            shape=(self.num_queries, self.hidden_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Tile query embeddings to match batch size.

        Args:
            inputs: Input tensor (used only to extract batch size).
                Shape: (batch, ...)

        Returns:
            Query embeddings: (batch, num_queries, hidden_dim)
        """
        batch_size = ops.shape(inputs)[0]
        return ops.tile(
            ops.expand_dims(self.query_embed, axis=0), (batch_size, 1, 1)
        )

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return (batch_size, self.num_queries, self.hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_queries": self.num_queries,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


class DETRMLPPredictionHead(layers.Layer):
    """MLP head for bounding box coordinate prediction.

    A 3-layer MLP that predicts normalized bounding box coordinates
    [cx, cy, w, h] in range [0, 1].

    Args:
        hidden_dim: int. Input/hidden dimension, default 256
        output_dim: int. Output dimension, default 4 (bbox coords)
        num_layers: int. Number of linear layers, default 3
    """

    def __init__(self, hidden_dim=256, output_dim=4, num_layers=3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

    def build(self, input_shape):
        self.layers_list = []
        for i in range(self.num_layers - 1):
            self.layers_list.append(
                layers.Dense(
                    self.hidden_dim, activation="relu", name=f"layer_{i}"
                )
            )
        # Final layer outputs bbox coords (no activation, will apply
        # sigmoid later)
        self.layers_list.append(
            layers.Dense(self.output_dim, name=f"layer_{self.num_layers - 1}")
        )
        super().build(input_shape)

    def call(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
            }
        )
        return config


class DETRTransformer(Layer):
    """Encoder and Decoder of DETR."""

    def __init__(
        self,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def build(self, input_shape=None):
        if self.num_encoder_layers > 0:
            self.encoder = DetrTransformerEncoder(
                attentiondropout_rate=self.dropout_rate,
                dropout_rate=self.dropout_rate,
                intermediate_dropout=self.dropout_rate,
                norm_first=False,
                num_layers=self.num_encoder_layers,
                num_attention_heads=self.num_attention_heads,
                intermediate_size=self.intermediate_size,
            )
        else:
            self.encoder = None

        self.decoder = DetrTransformerDecoder(
            attentiondropout_rate=self.dropout_rate,
            dropout_rate=self.dropout_rate,
            intermediate_dropout=self.dropout_rate,
            norm_first=False,
            num_layers=self.num_decoder_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
        )
        super().build(input_shape)

    def get_config(self):
        return {
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout_rate": self.dropout_rate,
        }

    def call(self, inputs):
        sources = inputs["inputs"]
        targets = inputs["targets"]
        pos_embed = inputs["pos_embed"]
        mask = inputs["mask"]
        input_shape = ops.shape(sources)
        source_attention_mask = ops.tile(
            ops.expand_dims(mask, axis=1), [1, input_shape[1], 1]
        )

        if self.encoder is not None:
            memory = self.encoder(
                sources,
                attention_mask=source_attention_mask,
                pos_embed=pos_embed,
            )
        else:
            memory = sources

        target_shape = ops.shape(targets)
        cross_attention_mask = ops.tile(
            ops.expand_dims(mask, axis=1), [1, target_shape[1], 1]
        )
        target_shape = ops.shape(targets)

        decoded = self.decoder(
            ops.zeros_like(targets),
            memory,
            self_attention_mask=ops.ones(
                (target_shape[0], target_shape[1], target_shape[1])
            ),
            cross_attention_mask=cross_attention_mask,
            return_all_decoder_outputs=True,
            input_pos_embed=targets,
            memory_pos_embed=pos_embed,
        )
        return decoded


class ExpandMaskLayer(layers.Layer):
    """Expand 1D mask to 2D attention mask."""

    def call(self, mask_flat):
        num_features = ops.shape(mask_flat)[1]
        return ops.tile(
            ops.expand_dims(mask_flat, axis=1), (1, num_features, 1)
        )

    def compute_output_shape(self, input_shape):
        batch_size, num_features = input_shape
        return (batch_size, num_features, num_features)


class ResizeMaskLayer(layers.Layer):
    """Resize mask to target spatial dimensions."""

    def call(self, inputs):
        mask, target_tensor = inputs
        target_shape = ops.shape(target_tensor)
        h, w = target_shape[1], target_shape[2]
        return ops.image.resize(mask, (h, w), interpolation="nearest")

    def compute_output_shape(self, input_shape):
        mask_shape, target_shape = input_shape
        return (mask_shape[0], target_shape[1], target_shape[2], mask_shape[3])
