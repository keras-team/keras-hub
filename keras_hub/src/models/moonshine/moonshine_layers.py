import keras


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineInvFreqInitializer(keras.initializers.Initializer):
    """
    Moonshine inverse frequency initializer.

    Initializes weights for computing inverse frequencies used in rotary
    embeddings. It generates a tensor of inverse frequency values based on the
    specified dimension and base.

    Args:
        dim (int): The dimensionality for which to compute inverse frequencies.
        Typically, this should be an even number.
        base (float, optional): The base value used in computing the inverse
        frequencies. Defaults to 10000.

    Returns:
        A tensor of shape (dim // 2,) containing the inverse frequency values.
    """

    def __init__(self, dim, base=10000):
        self.dim = dim
        self.base = base

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = "float32"
        exponents = keras.ops.arange(0, self.dim, 2, dtype=dtype) / self.dim
        return 1.0 / (self.base**exponents)

    def get_config(self):
        return {"dim": self.dim, "base": self.base}


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineRotaryEmbedding(keras.layers.Layer):
    """
    Moonshine rotary embedding layer. Computes rotary positional embeddings
    using precomputed inverse frequencies. The layer stores the inverse
    frequency weights as a non-trainable parameter and uses them to compute
    sinusoidal embeddings based on input positions.

    Args:
        dim (int): The number of dimensions for which rotary embeddings are
        computed.
        base (float, optional): Base value for computing inverse frequencies.
        Defaults to 10000.

    Returns:
        A tensor containing rotary embeddings reshaped to the appropriate output
        dimensions.
    """

    def __init__(self, dim, base=10000, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.base = base

    def build(self, input_shape):
        super().build(input_shape)
        # Create and track the non-trainable weight immediately.
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(self.dim // 2,),
            initializer=MoonshineInvFreqInitializer(self.dim, self.base),
            trainable=False,
        )

    def call(self, t):
        # Note: Cannot compute inv_freq on the fly here instead of storing it as
        # a weight. Causes NoneType error.
        t_cast = keras.ops.cast(t, keras.ops.dtype(self.inv_freq))
        freqs = keras.ops.einsum("i,j->ij", t_cast, self.inv_freq)
        freqs = keras.ops.stack((freqs, freqs), axis=-1)
        shape_list = list(keras.ops.shape(freqs))
        shape_list[-2:] = [-1]
        return keras.ops.reshape(freqs, shape_list)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "base": self.base})
        return config


class MoonshineArange(keras.layers.Layer):
    """
    Moonshine arange layer.
    The behavior is adapted from keras.ops.arange. The input is squeezed to
    obtain a scalar which is then used as the end value for the range.

    Returns:
        A 1-D tensor containing values from 0 to the scalar derived from input.
    """

    def call(self, inputs):
        return keras.ops.arange(
            keras.ops.squeeze(inputs), dtype=self.compute_dtype
        )

    def compute_output_spec(self, input_shape):
        return keras.KerasTensor((None,), dtype=self.compute_dtype)


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineSwiGLU(keras.layers.Layer):
    """
    Moonshine SwiGLU feedforward layer.

    Implements a SwiGLU feedforward activation block. The layer applies a dense
    projection that outputs 2 * multiplier * hidden_dim units, splits the output
    into two halves, applies a SiLU activation to the gating half, multiplies
    the two halves elementwise, and projects the result back to hidden_dim.

    Args:
        hidden_dim (int): The input and output dimensionality.
        multiplier (int): The multiplicative factor for the intermediate dense
        layer.

    Returns:
        A tensor with the same last dimension as the input (hidden_dim).
    """

    def __init__(
        self,
        hidden_dim,
        multiplier,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.multiplier = multiplier
        # First dense layer produces 2 * multiplier * hidden_dim outputs.
        self.dense_1 = keras.layers.Dense(
            hidden_dim * multiplier * 2, use_bias=True, name="dense_1"
        )
        # Activation layer using "silu" (Swish activation)
        self.activation = keras.layers.Activation("silu", name="activation")
        # Second dense layer projects back to hidden_dim.
        self.dense_2 = keras.layers.Dense(
            hidden_dim, use_bias=True, name="dense_2"
        )

    def build(self, input_shape):
        super().build(input_shape)
        # Build the first dense layer using the original input shape.
        self.dense_1.build(input_shape)
        # After dense_1, the output shape becomes: (..., 2 * multiplier *
        # hidden_dim).
        # When splitting, each part will have shape (..., multiplier *
        # hidden_dim).
        new_input_shape = list(input_shape)
        new_input_shape[-1] = self.hidden_dim * self.multiplier
        self.dense_2.build(tuple(new_input_shape))

    def call(self, inputs):
        x = self.dense_1(inputs)
        x1, gate = keras.ops.split(x, 2, axis=-1)
        x = x1 * self.activation(gate)
        return self.dense_2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "multiplier": self.multiplier,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineLinearGeLU(keras.layers.Layer):
    """
    Moonshine Linear GeLU feedforward layer.

    Implements a feedforward block that applies a linear projection, followed by
    a GeLU activation, and a second linear projection back to the original
    hidden dimension. This layer serves as an alternative to SwiGLU in the
    Moonshine architecture.

    Args:
        hidden_dim (int): The dimensionality of the input and output.
        multiplier (int, optional): The factor by which the hidden dimension is
        expanded in the intermediate dense layer. Defaults to 4.

    Returns:
        A tensor with the same last dimension as the input (hidden_dim).
    """

    def __init__(
        self,
        hidden_dim,
        multiplier=4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.multiplier = multiplier
        # Taken from pretrained weights.
        # First dense layer: output dimension is hidden_dim * multiplier.
        self.dense_1 = keras.layers.Dense(
            hidden_dim * multiplier, use_bias=True, name="dense_1"
        )
        # Activation layer using "gelu"
        self.activation = keras.layers.Activation("gelu", name="activation")
        # Second dense layer: output dimension is hidden_dim.
        self.dense_2 = keras.layers.Dense(
            hidden_dim, use_bias=True, name="dense_2"
        )

    def build(self, input_shape):
        super().build(input_shape)
        # Build the first dense layer with the given input shape.
        self.dense_1.build(input_shape)
        # The output of dense_1 will have its last dimension = hidden_dim *
        # multiplier.
        # Use that output shape to build the second dense layer.
        dense1_output_shape = self.dense_1.compute_output_shape(input_shape)
        self.dense_2.build(dense1_output_shape)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.activation(x)
        return self.dense_2(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "multiplier": self.multiplier,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineReversibleEmbedding(keras.layers.Layer):
    """
    Moonshine reversible embedding.

    An embedding layer that maps discrete token indices to continuous embedding
    vectors and supports reversible operations to project hidden representations
    back into vocabulary logits. This reversible functionality enables the layer
    to serve both as an encoder for input tokens and as a decoder output
    projection in transformer models.

    Args:
        vocab_size (int): The size of the vocabulary.
        hidden_dim (int): The dimensionality of the embedding vectors.
        embeddings_initializer (str or callable, optional): Initializer for the
        embedding weights. Defaults to "uniform".
        embeddings_regularizer (str or callable, optional): Regularizer for the
        embedding weights. Defaults to None.
        embeddings_constraint (str or callable, optional): Constraint for the
        embedding weights. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base layer.
    """

    def __init__(
        self,
        vocab_size,
        hidden_dim,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embeddings_initializer = keras.initializers.get(
            embeddings_initializer
        )
        self.embeddings_regularizer = keras.regularizers.get(
            embeddings_regularizer
        )
        self.embeddings_constraint = keras.constraints.get(
            embeddings_constraint
        )

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=[self.vocab_size, self.hidden_dim],
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, reverse=False):
        if reverse:
            kernel = keras.ops.transpose(
                keras.ops.convert_to_tensor(self.embeddings)
            )
            return keras.ops.matmul(inputs, kernel)
        else:
            return keras.ops.take(self.embeddings, inputs, axis=0)

    def compute_output_shape(self, input_shape):
        if not input_shape:
            raise ValueError("Input shape must be non-empty")

        output_shape = list(input_shape)
        if hasattr(self, "_reverse") and self._reverse:
            output_shape[-1] = self.vocab_size
        else:
            output_shape.append(self.hidden_dim)
        return tuple(output_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_dim": self.hidden_dim,
                "embeddings_initializer": keras.initializers.serialize(
                    self.embeddings_initializer
                ),
                "embeddings_regularizer": keras.regularizers.serialize(
                    self.embeddings_regularizer
                ),
                "embeddings_constraint": keras.constraints.serialize(
                    self.embeddings_constraint
                ),
            }
        )
        return config
