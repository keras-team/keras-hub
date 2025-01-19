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

    def call(self, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")

        y_embed = ops.cumsum(pixel_mask, axis=1, dtype="float32")
        x_embed = ops.cumsum(pixel_mask, axis=2, dtype="float32")
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

        pos_x = ops.reshape(
            pos_x, [pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], -1]
        )
        pos_y = ops.reshape(
            pos_y, [pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], -1]
        )

        pos = ops.concatenate((pos_y, pos_x), axis=3)
        pos = ops.transpose(pos, [0, 3, 1, 2])
        return pos


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
        attention_dropout_rate=0.0,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._intermediate_dropout = intermediate_dropout

    def build(self, input_shape):
        self.encoder_layers = []
        for i in range(self.num_layers):
            self.encoder_layers.append(
                DetrTransformerEncoderBlock(
                    num_attention_heads=self.num_attention_heads,
                    inner_dim=self._intermediate_size,
                    inner_activation=self._activation,
                    output_dropout=self._dropout_rate,
                    attention_dropout=self._attention_dropout_rate,
                    use_bias=self._use_bias,
                    norm_first=self._norm_first,
                    norm_epsilon=self._norm_epsilon,
                    inner_dropout=self._intermediate_dropout,
                )
            )
        self.output_normalization = layers.LayerNormalization(
            epsilon=self._norm_epsilon, dtype="float32"
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self._intermediate_size,
            "activation": self._activation,
            "dropout_rate": self._dropout_rate,
            "attention_dropout_rate": self._attention_dropout_rate,
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
            "intermediate_dropout": self._intermediate_dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, encoder_inputs, attention_mask=None, pos_embed=None):
        for layer_idx in range(self.num_layers):
            encoder_inputs = self.encoder_layers[layer_idx](
                [encoder_inputs, attention_mask, pos_embed]
            )

        output_tensor = encoder_inputs
        output_tensor = self.output_normalization(output_tensor)

        return output_tensor


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
        output_range=None,
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

        self._num_heads = num_attention_heads
        self._inner_dim = inner_dim
        self._inner_activation = inner_activation
        self._attention_dropout = attention_dropout
        self._attention_dropout_rate = attention_dropout
        self._output_dropout = output_dropout
        self._output_dropout_rate = output_dropout
        self._output_range = output_range
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._inner_dropout = inner_dropout
        self._attention_axes = attention_axes

    def build(self, input_shape):
        einsum_equation = "abc,cd->abd"
        if len(len(input_shape)) > 3:
            einsum_equation = "...bc,cd->...bd"

        hidden_size = input_shape[-1]
        if hidden_size % self._num_heads != 0:
            raise ValueError(
                "The input size (%d) is not a multiple of "
                "the number of attention heads (%d)"
                % (hidden_size, self._num_heads)
            )
        self._attention_head_size = int(hidden_size // self._num_heads)

        self._attention_layer = layers.MultiHeadAttention(
            num_heads=self._num_heads,
            key_dim=self._attention_head_size,
            dropout=self._attention_dropout,
            use_bias=self._use_bias,
            attention_axes=self._attention_axes,
            name="self_attention",
        )
        self._attention_dropout = layers.Dropout(rate=self._output_dropout)
        self._attention_layer_norm = layers.LayerNormalization(
            name="self_attention_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype="float32",
        )
        self._intermediate_dense = layers.EinsumDense(
            einsum_equation,
            output_shape=(None, self._inner_dim),
            bias_axes="d",
            name="intermediate",
        )

        self._intermediate_activation_layer = layers.Activation(
            self._inner_activation
        )
        self._inner_dropout_layer = layers.Dropout(rate=self._inner_dropout)
        self._output_dense = layers.EinsumDense(
            einsum_equation,
            output_shape=(None, hidden_size),
            bias_axes="d",
            name="output",
        )
        self._output_dropout = layers.Dropout(rate=self._output_dropout)
        self._output_layer_norm = layers.LayerNormalization(
            name="output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
            dtype="float32",
        )

        super().build(input_shape)

    def get_config(self):
        config = {
            "num_attention_heads": self._num_heads,
            "inner_dim": self._inner_dim,
            "inner_activation": self._inner_activation,
            "output_dropout": self._output_dropout_rate,
            "attention_dropout": self._attention_dropout_rate,
            "output_range": self._output_range,
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
            "inner_dropout": self._inner_dropout,
            "attention_axes": self._attention_axes,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        input_tensor, attention_mask, pos_embed = inputs

        key_value = None

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
            query=target_tensor + pos_embed,
            key=key_value + pos_embed,
            value=key_value,
            attention_mask=attention_mask,
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

        return self._output_layer_norm(layer_output + attention_output)


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
        attention_dropout_rate=0.0,
        use_bias=False,
        norm_first=True,
        norm_epsilon=1e-6,
        intermediate_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._intermediate_dropout = intermediate_dropout

    def build(self, input_shape):
        self.decoder_layers = []
        for i in range(self.num_layers):
            self.decoder_layers.append(
                DetrTransformerDecoderBlock(
                    num_attention_heads=self.num_attention_heads,
                    intermediate_size=self._intermediate_size,
                    intermediate_activation=self._activation,
                    dropout_rate=self._dropout_rate,
                    attention_dropout_rate=self._attention_dropout_rate,
                    use_bias=self._use_bias,
                    norm_first=self._norm_first,
                    norm_epsilon=self._norm_epsilon,
                    intermediate_dropout=self._intermediate_dropout,
                    name=("layer_%d" % i),
                )
            )
        self.output_normalization = layers.LayerNormalization(
            epsilon=self._norm_epsilon, dtype="float32"
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self._intermediate_size,
            "activation": self._activation,
            "dropout_rate": self._dropout_rate,
            "attention_dropout_rate": self._attention_dropout_rate,
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
            "intermediate_dropout": self._intermediate_dropout,
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

            output_tensor = self.decoder_layers[layer_idx](transformer_inputs)

            if return_all_decoder_outputs:
                decoder_outputs.append(self.output_normalization(output_tensor))

        if return_all_decoder_outputs:
            return decoder_outputs
        else:
            return self.output_normalization(output_tensor)


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
        attention_dropout_rate=0.0,
        use_bias=True,
        norm_first=False,
        norm_epsilon=1e-12,
        intermediate_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.intermediate_activation = activations.get(intermediate_activation)
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._intermediate_dropout = intermediate_dropout

        self._cross_attention_cls = layers.MultiHeadAttention

    def build(self, input_shape):
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
            dropout=self.attention_dropout_rate,
            use_bias=self._use_bias,
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
            epsilon=self._norm_epsilon,
            dtype="float32",
        )
        # Encoder-decoder attention.
        self.encdec_attention = self._cross_attention_cls(
            num_heads=self.num_attention_heads,
            key_dim=self.attention_head_size,
            dropout=self.attention_dropout_rate,
            output_shape=hidden_size,
            use_bias=self._use_bias,
            name="attention/encdec",
        )

        self.encdec_attention_dropout = layers.Dropout(rate=self.dropout_rate)
        self.encdec_attention_layer_norm = layers.LayerNormalization(
            name="attention/encdec_output_layer_norm",
            axis=-1,
            epsilon=self._norm_epsilon,
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
        self._intermediate_dropout_layer = layers.Dropout(
            rate=self._intermediate_dropout
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
            epsilon=self._norm_epsilon,
            dtype="float32",
        )
        super().build(input_shape)

    def get_config(self):
        config = {
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
            "intermediate_dropout": self._intermediate_dropout,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        (
            input_tensor,
            memory,
            attention_mask,
            self_attention_mask,
            input_pos_embed,
            memory_pos_embed,
        ) = inputs
        source_tensor = input_tensor
        if self._norm_first:
            input_tensor = self.self_attention_layer_norm(input_tensor)
        self_attention_output = self.self_attention(
            query=input_tensor + input_pos_embed,
            key=input_tensor + input_pos_embed,
            value=input_tensor,
            attention_mask=self_attention_mask,
        )
        self_attention_output = self.self_attention_dropout(
            self_attention_output
        )
        if self._norm_first:
            self_attention_output = source_tensor + self_attention_output
        else:
            self_attention_output = self.self_attention_layer_norm(
                input_tensor + self_attention_output
            )
        if self._norm_first:
            source_self_attention_output = self_attention_output
            self_attention_output = self.encdec_attention_layer_norm(
                self_attention_output
            )
        cross_attn_inputs = dict(
            query=self_attention_output + input_pos_embed,
            key=memory + memory_pos_embed,
            value=memory,
            attention_mask=attention_mask,
        )
        attention_output = self.encdec_attention(**cross_attn_inputs)
        attention_output = self.encdec_attention_dropout(attention_output)
        if self._norm_first:
            attention_output = source_self_attention_output + attention_output
        else:
            attention_output = self.encdec_attention_layer_norm(
                self_attention_output + attention_output
            )
        if self._norm_first:
            source_attention_output = attention_output
            attention_output = self.output_layer_norm(attention_output)

        intermediate_output = self.intermediate_dense(attention_output)
        intermediate_output = self.intermediate_activation_layer(
            intermediate_output
        )
        intermediate_output = self._intermediate_dropout_layer(
            intermediate_output
        )
        layer_output = self.output_dense(intermediate_output)
        layer_output = self.output_dropout(layer_output)
        if self._norm_first:
            layer_output = source_attention_output + layer_output
        else:
            layer_output = self.output_layer_norm(
                layer_output + attention_output
            )
        return layer_output
