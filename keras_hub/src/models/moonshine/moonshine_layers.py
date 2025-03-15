import keras


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineInvFreqInitializer(keras.initializers.Initializer):
    """
    Moonshine inverse frequency initializer.

    Initializes weights for computing inverse frequencies used in rotary
    embeddings. It generates a tensor of inverse frequency values based on the
    specified dimension and base value.

    Args:
        inv_freq_dim: int. The number of dimensions for which to compute inverse
            frequencies. This should be an even number, representing half the
            number of features.
        max_position_embeddings: int. The maximum sequence length the model will
            process, used to control the scale of positional embeddings.
        base_value: float, optional. The exponential base value used in
            computing the inverse frequencies. Higher values produce longer
            wavelengths. Defaults to 10000.
        scaling_factor: float, optional. A multiplier applied to the inverse
            frequencies to control the scale of the embeddings. Defaults to 1.0.
    """

    # References:
    # Defined and formulated based on the UsefulSensors implementation of the
    # InvFreqInitializer class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L164).

    def __init__(
        self,
        inv_freq_dim,
        max_position_embeddings,
        base_value=10000,
        scaling_factor=1.0,
    ):
        self.inv_freq_dim = inv_freq_dim
        self.max_position_embeddings = max_position_embeddings
        self.base_value = base_value
        self.scaling_factor = scaling_factor

    def __call__(self, shape, dtype=None, **kwargs):
        if dtype is None:
            dtype = self.compute_dtype

        inv_freq = 1.0 / (
            self.base_value
            ** (
                keras.ops.arange(0, self.inv_freq_dim, dtype=dtype)
                / self.inv_freq_dim
            )
        )
        return inv_freq * self.scaling_factor

    def get_config(self):
        return {
            "inv_freq_dim": self.inv_freq_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "base_value": self.base_value,
            "scaling_factor": self.scaling_factor,
        }


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineRotaryEmbedding(keras.layers.Layer):
    """
    Moonshine rotary embedding layer.

    Computes rotary positional embeddings using precomputed inverse frequencies.
    Supports two RoPE types: "default" and "dynamic".

    - **Default RoPE**: Applies rotary embeddings to a fraction of dimensions
      controlled by `partial_rotary_factor`.
    - **Dynamic RoPE**: Updates frequencies dynamically based on sequence length
      aligning functionally with the Hugging Face implementation.

    The layer stores inverse frequency weights as a non-trainable parameter and
    computes sinusoidal embeddings based on input positions. Unlike KerasHub's
    `RotaryEmbedding` class, this implementation explicitly requires `head_dim`
    and applies `partial_rotary_factor` for selective rotary embedding, whereas
    KerasHub uses `max_wavelength` without partial application. Additionally,
    this version employs a custom initializer (`MoonshineInvFreqInitializer`)
    for frequency computation, while KerasHub's implementation computes
    frequencies on-the-fly.

    Args:
        head_dim: int. The dimensionality of each attention head, determining
            the feature space for rotary embeddings.
        max_position_embeddings: int, optional. The maximum sequence length the
            model can process, controlling the positional embedding scale.
            Defaults to 2048.
        base_value: float, optional. Base value for computing inverse
            frequencies. Higher values result in longer wavelengths. Defaults to
            10000.
        rope_scaling: dict, optional. Configuration for RoPE scaling, such as
            `{"rope_type": "default"}` or `{"rope_type": "dynamic"}`.
            Defaults to `{"rope_type": "default"}` if None.
        partial_rotary_factor: float, optional. The fraction of `head_dim`
            dimensions that receive rotary embeddings, balancing rotary and
            non-rotary components. Defaults to 0.62.
        dtype: string, optional. The data type for model computations and
            weights. Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    # References:
    # Based on the UsefulSensors implementation of the RotaryEmbedding class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L176).
    # Incorporates dynamic RoPE concepts from the Hugging Face implementation (https://github.com/huggingface/transformers/blob/bc30dd1efb99f571d45b2e2131a555d09285ddd8/src/transformers/models/moonshine/modeling_moonshine.py#L311C1).

    def __init__(
        self,
        head_dim,
        max_position_embeddings=2048,
        base_value=10000,
        rope_scaling=None,
        partial_rotary_factor=0.62,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base_value = base_value
        self.partial_rotary_factor = partial_rotary_factor

        if rope_scaling is None:
            rope_scaling = {"rope_type": "default"}
        self.rope_scaling = rope_scaling
        self.rope_type = rope_scaling.get("rope_type", "default")

        if self.rope_type == "default":
            self.scaling_factor = 1.0
            self.attention_scaling = 1.0
        elif "dynamic" in self.rope_type:
            self.scaling_factor = 1.0  # Initial scaling, updated dynamically
            self.attention_scaling = 1.0
        else:
            raise NotImplementedError(
                f"rope_type '{self.rope_type}' not implemented"
            )

    def build(self, input_shape):
        # Create and track the non-trainable weight immediately.
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        rotary_dim = (rotary_dim // 2) * 2
        rotary_dim = rotary_dim // 2

        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(rotary_dim,),
            initializer=MoonshineInvFreqInitializer(
                rotary_dim,
                self.max_position_embeddings,
                self.base_value,
                self.scaling_factor,
            ),
            trainable=False,
            dtype=self.dtype,
        )
        self.original_inv_freq = keras.ops.convert_to_tensor(self.inv_freq)
        self.current_inv_freq = self.original_inv_freq
        self.max_seq_len_cached = self.max_position_embeddings
        self.built = True

    def call(self, t, position_ids=None):
        if "dynamic" in self.rope_type:
            if position_ids is None:
                position_ids = keras.ops.expand_dims(t, axis=0)
            seq_len = keras.ops.max(position_ids) + 1
            if seq_len > self.max_seq_len_cached:
                scaling = self.max_position_embeddings / seq_len
                self.current_inv_freq = self.original_inv_freq * scaling
                self.max_seq_len_cached = seq_len
            elif (
                seq_len < self.max_position_embeddings
                and self.max_seq_len_cached > self.max_position_embeddings
            ):
                self.current_inv_freq = self.original_inv_freq
                self.max_seq_len_cached = self.max_position_embeddings

            pos_cast = keras.ops.cast(position_ids, self.dtype)
            freqs = pos_cast[:, :, None] * self.current_inv_freq[None, None, :]
            emb = keras.ops.concatenate((freqs, freqs), axis=-1)
            shape_list = list(keras.ops.shape(emb))
            shape_list[0] = -1
            shape_list[-2:] = [-1]
            return keras.ops.reshape(emb, shape_list)
        else:
            # Original "default" behavior.
            t_cast = keras.ops.cast(t, keras.ops.dtype(self.inv_freq))
            freqs = keras.ops.einsum("i,j->ij", t_cast, self.inv_freq)
            emb = keras.ops.stack((freqs, freqs), axis=-1)
            shape_list = list(keras.ops.shape(emb))
            shape_list[-2:] = [-1]
            return keras.ops.reshape(emb, shape_list)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "head_dim": self.head_dim,
                "max_position_embeddings": self.max_position_embeddings,
                "base_value": self.base_value,
                "rope_scaling": self.rope_scaling,
                "partial_rotary_factor": self.partial_rotary_factor,
                "dtype": self.dtype,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineArange(keras.layers.Layer):
    """
    Moonshine arange layer.

    This layer serves as a wrapper around the `arange()` function, ensuring
    compatibility with pretrained Moonshine weights. The Hugging Face
    implementation expects `arange()` to be encapsulated within a layer rather
    than directly using `keras.ops.arange()`. Without this wrapper, weight
    mapping from pretrained Moonshine models would fail.

    Args:
        inputs: Tensor. A tensor containing the upper limit for the range.
            This value will be squeezed to extract a scalar.
    """

    # References:
    # Based on the UsefulSensors implementation of the Arange class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L196).

    def __init__(self, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)

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

    Implements a SwiGLU-based feedforward block. The layer applies a dense
    projection with `2 * feedforward_expansion_factor * hidden_dim` units,
    splits the output into two halves, applies a SiLU activation to the
    gating half, performs elementwise multiplication, and projects the
    result back to `hidden_dim`.

    Args:
        hidden_dim: int. The input and output dimensionality of the layer.
        feedforward_expansion_factor: int. Expansion factor for the intermediate
            dense layer, determining how much the representation is widened
            before projecting back to `hidden_dim`.
        kernel_initializer: optional. Initializer for the kernel weight matrix.
            Defaults to `keras.initializers.GlorotUniform()`.
        dtype: string, optional. Data type for computations and weights.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    # References:
    # Based on the UsefulSensors implementation of the FFSwigLU class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L102).

    def __init__(
        self,
        hidden_dim,
        feedforward_expansion_factor,
        kernel_initializer=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.kernel_initializer = (
            kernel_initializer or keras.initializers.GlorotUniform()
        )
        # First dense layer produces '2 * feedforward_expansion_factor *
        # hidden_dim' outputs.
        self.dense_1 = keras.layers.Dense(
            hidden_dim * feedforward_expansion_factor * 2,
            use_bias=True,
            name="dense_1",
            dtype=self.dtype,
            kernel_initializer=self.kernel_initializer,
        )
        # Activation layer using "silu" (Swish activation).
        self.activation = keras.layers.Activation(
            "silu",
            name="activation",
            dtype=self.dtype,
        )
        # Second dense layer projects back to hidden_dim.
        self.dense_2 = keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            name="dense_2",
            dtype=self.dtype,
            kernel_initializer=self.kernel_initializer,
        )

    def build(self, input_shape):
        super().build(input_shape)
        # Build the first dense layer using the original input shape.
        self.dense_1.build(input_shape)
        # After dense_1, the output shape becomes: (..., 2 *
        # feedforward_expansion_factor * hidden_dim).
        # When splitting, each part will have shape (...,
        # feedforward_expansion_factor * hidden_dim).
        new_input_shape = list(input_shape)
        new_input_shape[-1] = (
            self.hidden_dim * self.feedforward_expansion_factor
        )
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
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dtype": self.dtype,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineLinearGeLU(keras.layers.Layer):
    """
    Moonshine Linear GeLU feedforward layer.

    Implements a feedforward block that applies a linear projection,
    followed by a GeLU activation, and a second linear projection back
    to the original hidden dimension. This serves as an alternative
    to SwiGLU in the Moonshine architecture.

    Args:
        hidden_dim: int. The input and output dimensionality of the layer.
        feedforward_expansion_factor: int. Expansion factor for the
            intermediate dense layer, determining how much the representation
            is widened before projecting back to `hidden_dim`. Defaults to 4.
        kernel_initializer: optional. Initializer for the kernel weight matrix.
            Defaults to `keras.initializers.GlorotUniform()`.
        dtype: string, optional. Data type for computations and weights.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    # References:
    # Based on the UsefulSensors implementation of the FFLinearGelu class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L82).

    def __init__(
        self,
        hidden_dim,
        feedforward_expansion_factor=4,
        kernel_initializer=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.kernel_initializer = (
            kernel_initializer or keras.initializers.GlorotUniform()
        )
        # Taken from pretrained weights.
        # First dense layer: output dimension is 'hidden_dim *
        # feedforward_expansion_factor'.
        self.dense_1 = keras.layers.Dense(
            hidden_dim * feedforward_expansion_factor,
            use_bias=True,
            name="dense_1",
            dtype=self.dtype,
            kernel_initializer=self.kernel_initializer,
        )
        # Activation layer using "gelu".
        self.activation = keras.layers.Activation(
            "gelu",
            name="activation",
            dtype=self.dtype,
        )
        # Second dense layer: output dimension is 'hidden_dim'.
        self.dense_2 = keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            name="dense_2",
            dtype=self.dtype,
            kernel_initializer=self.kernel_initializer,
        )

    def build(self, input_shape):
        super().build(input_shape)
        # Build the first dense layer with the given input shape.
        self.dense_1.build(input_shape)
        # The output of 'dense_1' will have its last dimension = 'hidden_dim *
        # feedforward_expansion_factor'.
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
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dtype": self.dtype,
            }
        )
        return config
