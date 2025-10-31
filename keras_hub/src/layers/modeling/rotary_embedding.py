import keras
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
        rope_scaling: dict. Configuration for RoPE scaling following HuggingFace
            standard. Supported scaling types: "default", "linear", "dynamic", "yarn".
            For any scaling type, required parameters:
            - type: str, scaling type ("linear", "dynamic", "yarn")
            - factor: float, scaling factor for context extension

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
        sequence_axis=1,
        feature_axis=-1,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.scaling_factor = scaling_factor
        self.rope_scaling = rope_scaling or {}
        self._parse_rope_scaling()

        # Store original axis values for validation
        self._original_sequence_axis = sequence_axis
        self._original_feature_axis = feature_axis

    def _parse_rope_scaling(self):
        """Parse rope_scaling configuration following HuggingFace standard."""
        if not self.rope_scaling:
            self.rope_type = "default"
            self.rope_factor = self.scaling_factor  # Use scaling_factor when no rope_scaling
            return

        # Support full HuggingFace rope_scaling parameters
        self.rope_type = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
        self.rope_factor = self.rope_scaling.get("factor", 1.0)

        # YaRN-specific parameters
        if self.rope_type == "yarn":
            self.beta_fast = self.rope_scaling.get("beta_fast", 32.0)
            self.beta_slow = self.rope_scaling.get("beta_slow", 1.0)
            self.original_max_position_embeddings = self.rope_scaling.get("original_max_position_embeddings", 4096)
            self.truncate = self.rope_scaling.get("truncate", False)
        else:
            # Set defaults for non-YaRN types
            self.beta_fast = None
            self.beta_slow = None
            self.original_max_position_embeddings = None
            self.truncate = None

    def _normalize_axes(self, input_shape):
        """Normalize and validate axis indices for the given input shape."""
        rank = len(input_shape)

        # Normalize negative indices
        sequence_axis = self._original_sequence_axis
        feature_axis = self._original_feature_axis

        if sequence_axis < 0:
            sequence_axis += rank
        if feature_axis < 0:
            feature_axis += rank

        # Validate axis indices
        if sequence_axis < 0 or sequence_axis >= rank:
            raise ValueError(f"sequence_axis {self._original_sequence_axis} is out of range for input with rank {rank}")
        if feature_axis < 0 or feature_axis >= rank:
            raise ValueError(f"feature_axis {self._original_feature_axis} is out of range for input with rank {rank}")
        if sequence_axis == feature_axis:
            raise ValueError("sequence_axis and feature_axis must be different")

        return sequence_axis, feature_axis

    def _validate_rotary_dimension(self, rotary_dim):
        """Validate that rotary dimension is even and handle odd dimensions."""
        if rotary_dim % 2 != 0:
            raise ValueError(
                f"Rotary dimension must be even, got {rotary_dim}. "
                "The rotary embedding splits the feature dimension into two halves. "
                "Consider using a different feature dimension or padding."
            )

    def call(self, inputs, start_index=0, positions=None):
        # Normalize and validate axes
        input_shape = ops.shape(inputs)
        sequence_axis, feature_axis = self._normalize_axes(input_shape)

        # Validate rotary dimension
        rotary_dim = input_shape[feature_axis]
        self._validate_rotary_dimension(rotary_dim)

        # Take care of unbatched `positions`.
        if positions is not None:
            if len(ops.shape(positions)) == 1:
                positions = ops.expand_dims(positions, axis=0)

        inputs = ops.moveaxis(
            inputs, (feature_axis, sequence_axis), (-1, 1)
        )
        cos_emb, sin_emb = self._compute_cos_sin_embedding(
            inputs, start_index, positions
        )
        output = self._apply_rotary_pos_emb(inputs, cos_emb, sin_emb)
        return ops.moveaxis(
            output, (-1, 1), (feature_axis, sequence_axis)
        )

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
        positions = ops.arange(seq_len, dtype=self.compute_dtype)
        return positions + ops.cast(start_index, dtype=self.compute_dtype)

    def _compute_cos_sin_embedding(self, inputs, start_index=0, positions=None):
        """Compute cos & sin RoPE embeddings with optional YaRN scaling.
        Uses tensor ops only to remain JIT/backends friendly.
        """
        batch_axis = 0
        sequence_axis = 1
        feature_axis = len(inputs.shape) - 1

        # rotary_dim should be half of the last feature axis (HF-style: rotate pairs)
        rotary_dim = ops.shape(inputs)[feature_axis]
        # Validate evenness
        try:
            # best-effort check when running eagerly; if unavailable this will be a no-op
            if int(rotary_dim) % 2 != 0:
                raise ValueError("Rotary embedding requires even feature dimension (last axis).")
        except Exception:
            pass

        # Get inverse frequencies using the appropriate scaling method (linear, dynamic, yarn, etc.)
        inverse_freq = self._get_inverse_freq(rotary_dim)

        # positions handling
        if positions is None:
            seq_len = ops.shape(inputs)[sequence_axis]
            positions = ops.arange(seq_len, dtype=self.compute_dtype)
            positions = positions + ops.cast(start_index, self.compute_dtype)
            positions = ops.expand_dims(positions, axis=0)  # shape (1, seq_len)
        else:
            # ensure float dtype and batch dim
            positions = ops.cast(positions, self.compute_dtype)
            if len(ops.shape(positions)) == 1:
                positions = ops.expand_dims(positions, axis=0)

        # Apply truncation for YaRN if specified
        if self.rope_type == "yarn" and self.truncate and self.original_max_position_embeddings is not None:
            positions = ops.minimum(
                positions,
                ops.cast(self.original_max_position_embeddings, self.compute_dtype)
            )

        # compute outer product positions x inverse_freq -> shape (batch?, seq_len, rotary_dim//2)
        # If positions has batch dim, einsum handles it
        freq = ops.einsum("bi,j->bij", positions, inverse_freq)

        # stack to interleave sin/cos dims and reshape to full rotary dim
        embedding = ops.stack((freq, freq), axis=-2)
        embedding = ops.reshape(embedding, (*ops.shape(freq)[:-1], ops.shape(freq)[-1] * 2))

        # Expand embedding to match inputs rank (insert axes for any non-batch/seq/feature dims)
        for axis in range(len(inputs.shape)):
            if axis not in (batch_axis, sequence_axis, feature_axis):
                embedding = ops.expand_dims(embedding, axis)

        cos_emb = ops.cast(ops.cos(embedding), self.compute_dtype)
        sin_emb = ops.cast(ops.sin(embedding), self.compute_dtype)

        # YaRN temperature scaling: implement in tensor ops
        if self.rope_type == "yarn":
            # t = (0.1 * ln(s) + 1)^2
            # make sure s > 0
            small = ops.cast(1e-6, self.compute_dtype)
            s_safe = ops.maximum(ops.cast(self.rope_factor, self.compute_dtype), small)
            t = ops.square(ops.add(ops.multiply(ops.cast(0.1, self.compute_dtype), ops.log(s_safe)),
                                   ops.cast(1.0, self.compute_dtype)))
            sqrt_t = ops.sqrt(t)

            # HF/YaRN descriptions indicate a temperature scaling applied to cos/sin embeddings,
            # equivalently scaling the logits. We implement the sqrt scaling on cos/sin.
            cos_emb = cos_emb * sqrt_t
            sin_emb = sin_emb * sqrt_t

        return cos_emb, sin_emb

    def _get_inverse_freq(self, rotary_dim):
        """Return inverse frequencies per HF convention (tensor-returning, uses compute_dtype)."""
        # rotary_dim expected to be python int or small tensor; create idx with dtype
        dtype = self.compute_dtype
        idx = ops.arange(0, rotary_dim, 2, dtype=dtype)
        denom = ops.cast(rotary_dim, dtype)
        freq_range = idx / denom
        inv = ops.power(ops.cast(self.max_wavelength, dtype), -freq_range)

        # apply rope_scaling variants
        if self.rope_type == "default":
            return inv
        elif self.rope_type == "linear":
            # linear: divide inverse freqs by factor (consistent with HF linear scaling semantics)
            return inv / ops.cast(self.rope_factor, dtype)
        elif self.rope_type == "dynamic":
            # dynamic (NTK-aware) fallback conservative implementation:
            # HF dynamic implementation uses NTK-by-parts; use a practical scaling to approximate.
            # Here we conservatively divide by rope_factor^(rotary_dim/(rotary_dim-2))
            exponent = ops.cast(rotary_dim, dtype) / ops.cast(max(1, rotary_dim - 2), dtype)
            return inv / ops.power(ops.cast(self.rope_factor, dtype), exponent)
        elif self.rope_type == "yarn":
            # Delegate to more advanced YaRN inverse freq routine
            return self._get_yarn_inverse_freq(inv, rotary_dim)
        else:
            return inv

    def _get_yarn_inverse_freq(self, base_inverse_freq, rotary_dim):
        """YaRN NTK-by-parts style inverse frequency scaling (tensor-friendly).
           This follows the YaRN paper and common porting decisions used in HF forks.
        """
        dtype = self.compute_dtype
        s = ops.cast(self.rope_factor, dtype)
        
        # Get the base (rope_theta equivalent) from max_wavelength
        base = ops.cast(self.max_wavelength, dtype)
        
        # Compute base frequencies: base ** (idx / dim)
        idx = ops.arange(0, rotary_dim, 2, dtype=dtype)
        pos_freqs = ops.power(base, idx / ops.cast(rotary_dim, dtype))
        
        # Compute interpolation and extrapolation frequencies
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (s * pos_freqs)
        
        # Find correction range using the same logic as the correct implementation
        if self.beta_fast is not None and self.beta_slow is not None and self.original_max_position_embeddings is not None:
            L = ops.cast(self.original_max_position_embeddings, dtype)
            beta_fast = ops.cast(self.beta_fast, dtype)
            beta_slow = ops.cast(self.beta_slow, dtype)
            
            # Find correction dimensions for beta_fast and beta_slow
            def find_correction_dim_tensor(num_rotations, dim, base_val, max_pos):
                return (dim * ops.log(max_pos / (num_rotations * 2 * 3.141592653589793))) / (2 * ops.log(base_val))
            
            low = find_correction_dim_tensor(beta_fast, ops.cast(rotary_dim, dtype), base, L)
            high = find_correction_dim_tensor(beta_slow, ops.cast(rotary_dim, dtype), base, L)
            
            # Apply truncation if specified
            if self.truncate:
                low = ops.floor(low)
                high = ops.ceil(high)
            
            # Clamp to valid range
            low = ops.maximum(low, ops.cast(0, dtype))
            high = ops.minimum(high, ops.cast(rotary_dim // 2 - 1, dtype))
            
            # Linear ramp function
            dim_half = rotary_dim // 2
            idx_half = ops.arange(0, dim_half, dtype=dtype)
            
            # Prevent singularity
            diff = high - low
            diff = ops.maximum(diff, ops.cast(0.001, dtype))
            
            linear_func = (idx_half - low) / diff
            ramp_func = ops.clip(linear_func, 0, 1)
            
            # Apply the ramp to get extrapolation factor
            inv_freq_extrapolation_factor = 1 - ramp_func
            
            # Combine interpolation and extrapolation
            scaled_inverse_freq = (
                inv_freq_interpolation * (1 - inv_freq_extrapolation_factor) +
                inv_freq_extrapolation * inv_freq_extrapolation_factor
            )
        else:
            # Fallback to simple scaling
            alpha = ops.power(s, ops.cast(rotary_dim, dtype) / ops.cast(max(1, rotary_dim - 2), dtype))
            scaled_inverse_freq = base_inverse_freq / alpha

        return scaled_inverse_freq

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_wavelength": self.max_wavelength,
                "scaling_factor": self.scaling_factor,
                "sequence_axis": self._original_sequence_axis,
                "feature_axis": self._original_feature_axis,
                "rope_scaling": self.rope_scaling,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape