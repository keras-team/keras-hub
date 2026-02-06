from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding


class MultimodalRotaryEmbedding(RotaryEmbedding):
    """Multimodal Rotary Position Embedding (M-RoPE) for vision-language models.

    M-RoPE extends standard RoPE to handle multimodal inputs by dividing the
    head dimension into three sections:
    - Text section: Standard 1D position encoding for text tokens
    - Temporal section: Position encoding for time/frame dimension (video/audio)
    - Spatial section: Position encoding for spatial dimensions (image patches)

    For text-only tokens, all three sections use the same position ID, making it
    equivalent to standard RoPE. For vision/audio tokens, each section gets
    independent position IDs for temporal and spatial modeling.

    Args:
        mrope_section: tuple of 3 ints. Dimension allocation for
            (text, temporal, spatial).
            For example, [24, 20, 20] means:
            - 24 dims for text positions
            - 20 dims for temporal positions
            - 20 dims for spatial positions
            Total must equal head_dim // 2 (since RoPE uses pairs).
        max_wavelength: int. The maximum angular wavelength. Defaults to 10000.
        scaling_factor: float. Scaling factor for positions. Defaults to 1.0.
        sequence_axis: int. Sequence axis in input tensor. Defaults to 1.
        feature_axis: int. Feature axis in input tensor. Defaults to -1.
        **kwargs: Additional arguments passed to parent RotaryEmbedding.

    Examples:
    ```python
    import numpy as np
    import keras

    # Initialize M-RoPE with section [24, 20, 20] for 128-dim heads
    mrope = MultimodalRotaryEmbedding(
        mrope_section=[24, 20, 20],
        max_wavelength=1000000,
    )

    q = keras.random.normal((2, 10, 32, 128))
    k = keras.random.normal((2, 10, 32, 128))

    text_pos = np.arange(10)
    position_ids = np.stack([
        text_pos,
        text_pos,
        text_pos,
    ], axis=0)
    position_ids = np.expand_dims(position_ids, 1).repeat(2, axis=1)

    q_embed, k_embed = mrope.apply_multimodal_rotary_embedding(
        q, k, position_ids
    )

    # Vision tokens: different positions per section
    # For image patches: text_pos=constant, temporal_pos=0, spatial_pos varies
    ```
    """

    def __init__(
        self,
        mrope_section,
        max_wavelength=10000,
        scaling_factor=1.0,
        attention_scaling=1.0,
        sequence_axis=1,
        feature_axis=-1,
        **kwargs,
    ):
        super().__init__(
            max_wavelength=max_wavelength,
            scaling_factor=scaling_factor,
            sequence_axis=sequence_axis,
            feature_axis=feature_axis,
            **kwargs,
        )

        if len(mrope_section) != 3:
            raise ValueError(
                f"mrope_section must have 3 elements "
                f"(text, temporal, spatial), got {len(mrope_section)}"
            )

        self.mrope_section = tuple(mrope_section)
        self.attention_scaling = attention_scaling

        self.total_rope_dim = sum(mrope_section)

    def apply_multimodal_rotary_embedding(self, query, key, position_ids):
        """Apply M-RoPE to query and key tensors.

        Args:
            query: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            key: Key tensor of shape (batch, seq_len, num_heads, head_dim)
            position_ids: Position IDs of shape (3, batch, seq_len) where:
                position_ids[0] = text positions
                position_ids[1] = temporal positions
                position_ids[2] = spatial positions

        Returns:
            Tuple of (query_embed, key_embed) with M-RoPE applied.
        """
        # Compute frequency matrices for each modality section
        batch_size = ops.shape(query)[0]
        seq_len = ops.shape(query)[1]

        # Compute inverse frequencies for the full head dimension
        head_dim_half = sum(self.mrope_section)
        idx = ops.arange(0, head_dim_half * 2, 2, dtype="float32")
        denom = ops.cast(head_dim_half * 2, "float32")
        freq_range = idx / denom
        inverse_freq = ops.power(
            ops.cast(self.max_wavelength, "float32"), -freq_range
        )
        inverse_freq = inverse_freq / ops.cast(self.scaling_factor, "float32")

        position_ids_float = ops.cast(position_ids, "float32")

        # Expand for broadcasting:
        # position_ids (3, batch, seq_len) -> (3, batch, seq_len, 1)
        # inverse_freq (head_dim_half,) -> (1, 1, 1, head_dim_half)
        position_ids_expanded = ops.expand_dims(position_ids_float, axis=-1)
        inverse_freq_expanded = ops.reshape(
            inverse_freq, (1, 1, 1, head_dim_half)
        )

        # Compute frequencies: (3, batch, seq_len, head_dim_half)
        freqs_stacked = position_ids_expanded * inverse_freq_expanded

        # Apply interleaved M-RoPE to reorganize frequency layout
        freqs_interleaved = self._apply_interleaved_mrope(
            freqs_stacked, self.mrope_section
        )

        # Duplicate frequencies for cos/sin pairs and compute embeddings
        # Shape: (batch, seq_len, head_dim)
        embedding = ops.stack([freqs_interleaved, freqs_interleaved], axis=-1)
        embedding = ops.reshape(
            embedding, (batch_size, seq_len, sum(self.mrope_section) * 2)
        )

        # Apply attention scaling to cos/sin embeddings
        cos_full = ops.cos(embedding) * self.attention_scaling
        sin_full = ops.sin(embedding) * self.attention_scaling
        cos_full = ops.cast(cos_full, self.compute_dtype)
        sin_full = ops.cast(sin_full, self.compute_dtype)

        # Expand for broadcasting with (batch, seq_len, num_heads, head_dim)
        cos_full = ops.expand_dims(
            cos_full, axis=2
        )  # (batch, seq_len, 1, head_dim)
        sin_full = ops.expand_dims(sin_full, axis=2)

        # Apply rotary embedding
        query_embed = self._apply_rotary_pos_emb_single(
            query, cos_full, sin_full
        )
        key_embed = self._apply_rotary_pos_emb_single(key, cos_full, sin_full)

        return query_embed, key_embed

    def _apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved M-RoPE to reorganize frequency layout.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.

        Args:
            freqs: Frequency matrices of shape
                (3, batch, seq_len, head_dim_half) where dim 0
                corresponds to [text, temporal, spatial]
                All 3 matrices have the same dimension but were computed with
                different position IDs.
            mrope_section: Tuple of (text_dim, temporal_dim, spatial_dim)

        Returns:
            Interleaved frequencies of shape
            (batch, seq_len, sum(mrope_section))
        """
        freqs_t = freqs[0]

        head_dim_half = sum(mrope_section)
        indices_list = []
        interleaved_length = min(mrope_section[1], mrope_section[2]) * 3

        for pos in range(interleaved_length):
            if pos % 3 == 0:
                # Text dimension
                indices_list.append(freqs[0][..., pos : pos + 1])
            elif pos % 3 == 1:
                # Temporal dimension
                indices_list.append(freqs[1][..., pos : pos + 1])
            else:
                # Spatial dimension
                indices_list.append(freqs[2][..., pos : pos + 1])

        # Remaining positions will be all from text dimension
        if interleaved_length < head_dim_half:
            indices_list.append(freqs_t[..., interleaved_length:])

        # Concatenate all selected frequencies
        result = ops.concatenate(indices_list, axis=-1)

        return result

    def _apply_rotary_pos_emb_single(self, tensor, cos_emb, sin_emb):
        """Apply rotary position embedding to a single tensor.

        Args:
            tensor: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            cos_emb: Cosine embedding of shape (batch, seq_len, 1, head_dim)
            sin_emb: Sine embedding of shape (batch, seq_len, 1, head_dim)

        Returns:
            Tensor with rotary embedding applied.
        """
        x1, x2 = ops.split(tensor, 2, axis=-1)

        half_rot_tensor = ops.stack([-x2, x1], axis=-2)
        half_rot_tensor = ops.reshape(half_rot_tensor, ops.shape(tensor))

        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mrope_section": self.mrope_section,
                "attention_scaling": self.attention_scaling,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
