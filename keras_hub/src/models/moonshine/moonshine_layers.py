import keras

from keras_hub.src.utils.keras_utils import clone_initializer


def moonshine_kernel_initializer(initializer_range=0.02):
    return keras.initializers.TruncatedNormal(stddev=initializer_range)


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
    KerasHub uses `max_wavelength` without partial application.

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
    # Based on the UsefulSensors implementation of the RotaryEmbedding class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L176-L193).
    # Incorporates dynamic RoPE concepts from the Hugging Face implementation (https://github.com/huggingface/transformers/blob/bc30dd1efb99f571d45b2e2131a555d09285ddd8/src/transformers/models/moonshine/modeling_moonshine.py#L311-L369).

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
        if rotary_dim <= 0:
            raise ValueError(
                f"Calculated rotary_dim ({rotary_dim}) must be a positive even "
                f"number. Check head_dim ({self.head_dim}) and "
                f"partial_rotary_factor ({self.partial_rotary_factor})."
            )
        rotary_dim_half = rotary_dim // 2

        # Compute inv_freq.
        inv_freq = 1.0 / (
            self.base_value
            ** (
                keras.ops.arange(0, rotary_dim_half, dtype=self.dtype)
                / rotary_dim_half
            )
        )
        inv_freq = inv_freq * self.scaling_factor

        # Set the non-trainable weight using the computed tensor.
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=(rotary_dim_half,),
            initializer=keras.initializers.Constant(inv_freq),
            trainable=False,
            dtype=self.dtype,
        )
        self.original_inv_freq = keras.ops.convert_to_tensor(self.inv_freq)
        self.max_sequence_length_cached = self.max_position_embeddings
        self.built = True

    def call(self, t, position_ids=None):
        # "Dynamic" RoPE behavior.
        if "dynamic" in self.rope_type:
            if position_ids is None:
                position_ids = keras.ops.expand_dims(t, axis=0)
            seq_len = keras.ops.max(position_ids) + 1
            if seq_len > self.max_position_embeddings:
                scaling = keras.ops.cast(
                    self.max_position_embeddings, self.dtype
                ) / keras.ops.cast(seq_len, self.dtype)
            else:
                scaling = keras.ops.cast(1.0, self.dtype)
            current_inv_freq = self.original_inv_freq * scaling
            if seq_len > self.max_sequence_length_cached:
                self.max_sequence_length_cached = seq_len
            elif (
                seq_len < self.max_position_embeddings
                and self.max_sequence_length_cached
                > self.max_position_embeddings
            ):
                self.max_sequence_length_cached = self.max_position_embeddings

            pos_cast = keras.ops.cast(position_ids, self.dtype)
            freqs = pos_cast[:, :, None] * current_inv_freq[None, None, :]
            cos = keras.ops.cos(freqs) * self.attention_scaling
            sin = keras.ops.sin(freqs) * self.attention_scaling
            return cos, sin
        # Original "default" behavior.
        else:
            t_cast = keras.ops.cast(t, keras.ops.dtype(self.inv_freq))
            original_shape = keras.ops.shape(t_cast)
            is_generation_step = (
                len(original_shape) == 2 and original_shape[1] == 1
            )
            if is_generation_step:
                # (batch, 1) -> Squeeze to (batch,) for einsum "i,j->ij".
                t_cast_for_einsum = keras.ops.squeeze(t_cast, axis=1)
                freqs = keras.ops.einsum(
                    "i,j->ij", t_cast_for_einsum, self.inv_freq
                )  # Shape (batch, rotary_dim_half)
            elif len(original_shape) == 1:
                t_cast_for_einsum = t_cast
                freqs = keras.ops.einsum(
                    "i,j->ij", t_cast_for_einsum, self.inv_freq
                )  # Shape (seq_len, rotary_dim_half)
            else:
                raise ValueError(
                    f"Unexpected shape for input 't' in "
                    f"MoonshineRotaryEmbedding default path: {original_shape}. "
                    "Expected (seq_len,) or (batch, 1)."
                )
            emb = keras.ops.stack((freqs, freqs), axis=-1)
            shape_list = list(keras.ops.shape(emb))
            shape_list[-2:] = [-1]
            emb_flat = keras.ops.reshape(emb, shape_list)
            if is_generation_step:
                final_emb = keras.ops.expand_dims(emb_flat, axis=1)
            else:
                final_emb = keras.ops.expand_dims(emb_flat, axis=0)
            return final_emb

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
class MoonshineMLP(keras.layers.Layer):
    """
    Moonshine MLP layer.

    Implements a Multi-Layer Perceptron (MLP) for Moonshine models with support
    for both `SwiGLU` and `LinearGeLU` activation patterns. The MLP consists of
    two dense layers with an activation function in between, expanding the input
    dimension before projecting back to the original dimension.

    Args:
        hidden_dim: int. The dimensionality of the input and output tensors.
        feedforward_expansion_factor: float. The factor by which to expand the
            hidden dimension in the intermediate layer.
        use_swiglu_activation: bool, optional. If `True`, uses SwiGLU activation
            (SiLU with gating). If `False`, uses standard GeLU activation.
            Defaults to `True`.
        initializer_range: float, optional. The standard deviation for kernel
            initialization. Defaults to 0.02.
        dtype: string, optional. The data type for model computations and
            weights. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    # References:
    # Based on the HuggingFace implementation of the MoonshineEncoderMLP and
    # MoonshineDecoderMLP classes (https://github.com/huggingface/transformers/blob/fc8764c9a618add64c33e83720f974750bcd0978/src/transformers/models/moonshine/modeling_moonshine.py#L66-L94).

    def __init__(
        self,
        hidden_dim,
        feedforward_expansion_factor,
        use_swiglu_activation=True,
        initializer_range=0.02,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation
        self.kernel_initializer = moonshine_kernel_initializer(
            initializer_range=initializer_range
        )
        self.initializer_range = initializer_range

        if use_swiglu_activation:
            # First dense layer produces (2 * feedforward_expansion_factor *
            # hidden_dim) outputs.
            self.dense_1 = keras.layers.Dense(
                int(hidden_dim * feedforward_expansion_factor * 2),
                use_bias=True,
                name="dense_1",
                dtype=self.dtype,
                kernel_initializer=clone_initializer(self.kernel_initializer),
            )
            # Activation layer using "silu" (Swish activation).
            self.activation = keras.layers.Activation(
                "silu", name="activation", dtype=self.dtype
            )
        else:
            # Taken from pretrained weights.
            # First dense layer: output dimension is (hidden_dim *
            # feedforward_expansion_factor).
            self.dense_1 = keras.layers.Dense(
                int(hidden_dim * feedforward_expansion_factor),
                use_bias=True,
                name="dense_1",
                dtype=self.dtype,
                kernel_initializer=clone_initializer(self.kernel_initializer),
            )
            self.activation = keras.layers.Activation(
                "gelu", name="activation", dtype=self.dtype
            )

        # Second dense layer projects back to hidden_dim.
        self.dense_2 = keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            name="dense_2",
            dtype=self.dtype,
            kernel_initializer=clone_initializer(self.kernel_initializer),
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
        if self.use_swiglu_activation:
            x1, gate = keras.ops.split(x, 2, axis=-1)
            activated_gate = self.activation(gate)
            x = x1 * activated_gate
        else:
            x = self.activation(x)
        output = self.dense_2(x)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
                "initializer_range": self.initializer_range,
                "dtype": self.dtype,
            }
        )
        return config
