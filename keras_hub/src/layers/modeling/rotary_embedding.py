import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.RotaryEmbedding")
class RotaryEmbedding(keras.layers.Layer):
    """Rotary positional encoding layer.

    This layer encodes absolute positional information with a rotation
    matrix. It calculates the rotary encoding with a mix of sine and
    cosine functions with geometrically increasing wavelengths.
    Defined and formulated in
    [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4).
    The input must be a tensor with shape a sequence dimension and a feature
    dimension. Typically, this will either an input with shape
    `(batch_size, sequence_length, feature_length)` or
    `(batch_size, sequence_length, num_heads, feature_length)`.
    This layer will return a new tensor with the rotary embedding applied to
    the input tensor.

    Args:
        max_wavelength: int. The maximum angular wavelength of the sine/cosine
            curves.
        scaling_factor: float. The scaling factor used to scale positions of
            the tokens.
        rope_type: str. The type of RoPE scaling to apply. Supported types:
            "linear", "dynamic", "yarn". Defaults to "linear".
        beta_fast: float. Beta fast parameter for YaRN scaling. Only used
            when rope_type="yarn". Defaults to 32.0.
        beta_slow: float. Beta slow parameter for YaRN scaling. Only used
            when rope_type="yarn". Defaults to 1.0.
        original_max_position_embeddings: int. Original maximum position
            embeddings for YaRN scaling. Only used when rope_type="yarn".
            Defaults to 4096.
        truncate: bool. Whether to apply truncation for YaRN scaling. Only used
            when rope_type="yarn". Defaults to False.
        sequence_axis: int. Sequence axis in the input tensor.
        feature_axis: int. Feature axis in the input tensor.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        inputs: The tensor inputs to apply the embedding to. This can have
            any shape, but must contain both a sequence and feature axis. The
            rotary embedding will be applied to `inputs` and returned.
        start_index: An integer or integer tensor. The starting position to
            compute the rotary embedding from. This is useful during cached
            decoding, where each position is predicted separately in a loop.
        positions: Tensor of shape `(sequence_length,)` or
            `(batch_size, sequence_length)`. Custom positions for the input
            sequence. If specified, this tensor will be used to
            compute the rotary embedding, and the `start_index` argument will
            be ignored. This is useful for cases with non-standard positions.

    Examples:

    ```python
    batch_size = 16
    feature_length = 18
    sequence_length = 256
    num_heads = 8

    # No multi-head dimension.
    tensor = np.ones((batch_size, sequence_length, feature_length))
    rot_emb_layer = RotaryEmbedding()
    tensor_rot = rot_emb_layer(tensor)

    # With multi-head dimension.
    tensor = np.ones((batch_size, sequence_length, num_heads, feature_length))
    tensor_rot = rot_emb_layer(tensor)
    ```

    References:
     - [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v4)
    """

    def __init__(
        self,
        max_wavelength=10000,
        scaling_factor=1.0,
        rope_type="linear",
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=4096,
        truncate=False,
        sequence_axis=1,
        feature_axis=-1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.sequence_axis = sequence_axis
        self.feature_axis = feature_axis
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type

        # YaRN-specific parameters (only used when rope_type="yarn")
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.original_max_position_embeddings = original_max_position_embeddings
        self.truncate = truncate
        self.built = True

    def _normalize_axes(self, input_shape):
        """Normalize and validate axis indices for the given input shape."""
        rank = len(input_shape)

        # Normalize negative indices
        sequence_axis = self.sequence_axis
        feature_axis = self.feature_axis

        if sequence_axis < 0:
            sequence_axis += rank
        if feature_axis < 0:
            feature_axis += rank

        if sequence_axis < 0 or sequence_axis >= rank:
            raise ValueError(
                f"sequence_axis {self.sequence_axis} "
                f"is out of range for input with rank {rank}"
            )
        if feature_axis < 0 or feature_axis >= rank:
            raise ValueError(
                f"feature_axis {self.feature_axis} "
                f"is out of range for input with rank {rank}"
            )
        if sequence_axis == feature_axis:
            raise ValueError("sequence_axis and feature_axis must be different")

        return sequence_axis, feature_axis

    def _validate_rotary_dimension(self, rotary_dim):
        if rotary_dim % 2 != 0:
            raise ValueError(
                f"Rotary dimension must be even, got {rotary_dim}."
                "The rotary embedding splits the feature dimension "
                "into two halves. Consider using a different feature "
                "dimension or padding."
            )

    def call(self, inputs, start_index=0, positions=None):
        input_shape = ops.shape(inputs)
        sequence_axis, feature_axis = self._normalize_axes(input_shape)

        rotary_dim = input_shape[feature_axis]
        self._validate_rotary_dimension(rotary_dim)

        # Take care of unbatched `positions`.
        if positions is not None:
            if len(ops.shape(positions)) == 1:
                positions = ops.expand_dims(positions, axis=0)

        inputs = ops.moveaxis(inputs, (feature_axis, sequence_axis), (-1, 1))
        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            inputs, start_index, positions
        )
        output = self._apply_rotary_pos_emb(inputs, cos_emb, sin_emb)
        return ops.moveaxis(output, (-1, 1), (feature_axis, sequence_axis))

    def _apply_rotary_pos_emb(self, tensor, cos_emb, sin_emb):
        x1, x2 = ops.split(tensor, 2, axis=-1)
        # Avoid `ops.concatenate` for now, to avoid a obscure bug with XLA
        # compilation on jax. We should be able to remove this once the
        # following PR is in all jax releases we care about:
        # https://github.com/openxla/xla/pull/7875
        half_rot_tensor = ops.stack((-x2, x1), axis=-2)
        half_rot_tensor = ops.reshape(half_rot_tensor, ops.shape(tensor))
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def _compute_positions(self, inputs, start_index=0):
        seq_len = ops.shape(inputs)[1]
        positions = ops.arange(seq_len, dtype="float32")
        return positions + ops.cast(start_index, dtype="float32")

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        """Compute cos & sin RoPE embeddings with optional YaRN scaling.
        Uses tensor ops only to remain JIT/backends friendly.
        """
        batch_axis = 0
        sequence_axis = 1
        feature_axis = len(inputs.shape) - 1

        rotary_dim = ops.shape(inputs)[feature_axis]
        inverse_freq = self._get_inverse_freq(rotary_dim)

        if positions is None:
            positions = self._compute_positions(inputs, start_index)
            positions = ops.expand_dims(
                positions, axis=batch_axis
            )  # shape (1, seq_len)
        else:
            positions = ops.cast(positions, "float32")
            if len(ops.shape(positions)) == 1:
                positions = ops.expand_dims(positions, axis=batch_axis)

        if (
            self.rope_type == "yarn"
            and self.truncate
            and self.original_max_position_embeddings is not None
        ):
            positions = ops.minimum(
                positions,
                ops.cast(self.original_max_position_embeddings, "float32"),
            )

        freq = ops.einsum("bi,j->bij", positions, inverse_freq)

        embedding = ops.stack((freq, freq), axis=-2)
        embedding = ops.reshape(
            embedding, (*ops.shape(freq)[:-1], ops.shape(freq)[-1] * 2)
        )

        for axis in range(len(inputs.shape)):
            if axis not in (batch_axis, sequence_axis, feature_axis):
                embedding = ops.expand_dims(embedding, axis)

        cos_emb = ops.cast(ops.cos(embedding), self.compute_dtype)
        sin_emb = ops.cast(ops.sin(embedding), self.compute_dtype)

        if self.rope_type == "yarn":
            # YaRN temperature scaling
            factor = ops.add(
                ops.multiply(
                    ops.cast(0.1, self.compute_dtype),
                    ops.log(ops.cast(self.scaling_factor, self.compute_dtype)),
                ),
                ops.cast(1.0, self.compute_dtype),
            )
            cos_emb = cos_emb * factor
            sin_emb = sin_emb * factor
        return cos_emb, sin_emb

    def _get_inverse_freq(self, rotary_dim):
        """Return inverse frequencies."""
        idx = ops.arange(0, rotary_dim, 2, dtype="float32")
        denom = ops.cast(rotary_dim, "float32")
        freq_range = idx / denom
        inv = ops.power(ops.cast(self.max_wavelength, "float32"), -freq_range)

        if self.rope_type == "linear":
            return inv / ops.cast(self.scaling_factor, "float32")
        elif self.rope_type == "dynamic":
            exponent = ops.cast(rotary_dim, "float32") / ops.cast(
                max(1, rotary_dim - 2), "float32"
            )
            return inv / ops.power(
                ops.cast(self.scaling_factor, "float32"), exponent
            )
        elif self.rope_type == "yarn":
            return self._get_yarn_inverse_freq(rotary_dim)
        else:
            return inv

    def _get_yarn_inverse_freq(self, rotary_dim):
        # Get the base (rope_theta equivalent) from max_wavelength
        base = ops.cast(self.max_wavelength, "float32")

        # Compute base frequencies: base ** (idx / dim)
        idx = ops.arange(0, rotary_dim, 2, dtype="float32")
        pos_freqs = ops.power(base, idx / ops.cast(rotary_dim, "float32"))

        # Compute interpolation and extrapolation frequencies
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (
            ops.cast(self.scaling_factor, "float32") * pos_freqs
        )

        # Find correction range
        beta_fast = ops.cast(self.beta_fast, "float32")
        beta_slow = ops.cast(self.beta_slow, "float32")

        # Find correction dimensions for beta_fast and beta_slow
        def find_correction_dim_tensor(num_rotations, dim):
            max_pos = ops.cast(self.original_max_position_embeddings, "float32")
            return (dim * ops.log(max_pos / (num_rotations * 2 * np.pi))) / (
                2 * ops.log(base)
            )

        low = find_correction_dim_tensor(
            beta_fast, ops.cast(rotary_dim, "float32")
        )
        high = find_correction_dim_tensor(
            beta_slow, ops.cast(rotary_dim, "float32")
        )

        # Apply truncation if specified
        if self.truncate:
            low = ops.floor(low)
            high = ops.ceil(high)

        # Clamp to valid range
        low = ops.maximum(low, ops.cast(0, "float32"))
        high = ops.minimum(high, ops.cast(rotary_dim // 2 - 1, "float32"))

        # Linear ramp function
        dim_half = rotary_dim // 2
        idx_half = ops.arange(0, dim_half, dtype="float32")

        # Prevent singularity
        diff = high - low
        diff = ops.maximum(diff, ops.cast(0.001, "float32"))

        linear_func = (idx_half - low) / diff
        ramp_func = ops.clip(linear_func, 0, 1)

        # Apply the ramp to get extrapolation factor
        inv_freq_extrapolation_factor = 1 - ramp_func

        # Combine interpolation and extrapolation
        scaled_inverse_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )

        return scaled_inverse_freq

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
                "scaling_factor": self.scaling_factor,
                "rope_type": self.rope_type,
                "beta_fast": self.beta_fast,
                "beta_slow": self.beta_slow,
                "original_max_position_embeddings": (
                    self.original_max_position_embeddings
                ),
                "truncate": self.truncate,
                "sequence_axis": self.sequence_axis,
                "feature_axis": self.feature_axis,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
