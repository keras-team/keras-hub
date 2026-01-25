"""Multimodal Rotary Position Embedding (M-RoPE) for Qwen3-Omni.

This implements the M-RoPE mechanism used in Qwen3-Omni for multimodal inputs.
M-RoPE divides the head dimension into three sections for text, temporal, and spatial
position encodings, enabling effective multimodal position representation.

Reference:
- Qwen2-VL: https://qwenlm.github.io/blog/qwen2-vl/
- HuggingFace implementation: transformers/models/qwen2_vl/modeling_qwen2_vl.py
"""

import keras
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
        mrope_section: tuple of 3 ints. Dimension allocation for (text, temporal, spatial).
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
    
    # Text tokens: all sections use same position
    # Shape: (batch, seq_len, num_heads, head_dim)
    q = keras.random.normal((2, 10, 32, 128))
    k = keras.random.normal((2, 10, 32, 128))
    
    # Position IDs shape: (batch, seq_len, 3)
    # For text: [[0, 0, 0], [1, 1, 1], [2, 2, 2], ...]
    position_ids = np.stack([
        np.arange(10),  # text positions
        np.arange(10),  # temporal positions (same for text)
        np.arange(10),  # spatial positions (same for text)
    ], axis=-1)
    position_ids = np.expand_dims(position_ids, 0).repeat(2, axis=0)
    
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
                f"mrope_section must have 3 elements (text, temporal, spatial), "
                f"got {len(mrope_section)}"
            )
        
        self.mrope_section = tuple(mrope_section)
        
        # Validate that mrope_section dimensions are consistent
        # Note: Each section uses pairs (cos/sin), so total = sum * 2
        total_dim = sum(mrope_section) * 2
        # This will be validated against head_dim when build() is called in parent attention layer
        self.total_rope_dim = sum(mrope_section)
    
    def apply_multimodal_rotary_embedding(
        self, query, key, position_ids
    ):
        """Apply M-RoPE to query and key tensors.
        
        Args:
            query: Query tensor of shape (batch, seq_len, num_heads, head_dim)
            key: Key tensor of shape (batch, seq_len, num_heads, head_dim)
            position_ids: Position IDs of shape (batch, seq_len, 3) where the last
                dimension contains [text_pos, temporal_pos, spatial_pos]
        
        Returns:
            Tuple of (query_embed, key_embed) with M-RoPE applied.
        """
        # Compute cos/sin embeddings for each modality section
        # position_ids shape: (batch, seq_len, 3)
        batch_size = ops.shape(query)[0]
        seq_len = ops.shape(query)[1]
        num_heads = ops.shape(query)[2]
        head_dim = ops.shape(query)[3]
        
        # Split position IDs into three modalities
        # Each has shape (batch, seq_len)
        text_pos = position_ids[:, :, 0]
        temporal_pos = position_ids[:, :, 1]
        spatial_pos = position_ids[:, :, 2]
        
        # Compute embeddings for each section
        # mrope_section = [text_dim, temporal_dim, spatial_dim]
        mrope_section_doubled = [s * 2 for s in self.mrope_section]
        
        cos_embeds = []
        sin_embeds = []
        
        for pos, dim in zip(
            [text_pos, temporal_pos, spatial_pos],
            self.mrope_section
        ):
            # Compute cos/sin for this section's dimension
            cos_emb, sin_emb = self._compute_cos_sin_for_section(
                pos, dim, batch_size, seq_len
            )
            cos_embeds.append(cos_emb)
            sin_embeds.append(sin_emb)
        
        # Concatenate embeddings: (batch, seq_len, head_dim)
        cos_full = ops.concatenate(cos_embeds, axis=-1)
        sin_full = ops.concatenate(sin_embeds, axis=-1)
        
        # Expand for broadcasting with (batch, seq_len, num_heads, head_dim)
        cos_full = ops.expand_dims(cos_full, axis=2)  # (batch, seq_len, 1, head_dim)
        sin_full = ops.expand_dims(sin_full, axis=2)
        
        # Apply rotary embedding
        query_embed = self._apply_rotary_pos_emb_single(query, cos_full, sin_full)
        key_embed = self._apply_rotary_pos_emb_single(key, cos_full, sin_full)
        
        return query_embed, key_embed
    
    def _compute_cos_sin_for_section(self, positions, section_dim, batch_size, seq_len):
        """Compute cos/sin embeddings for one section of M-RoPE.
        
        Args:
            positions: Position IDs of shape (batch, seq_len)
            section_dim: Dimension allocated to this section (e.g., 24 for text)
            batch_size: Batch size
            seq_len: Sequence length
        
        Returns:
            Tuple of (cos_emb, sin_emb) each with shape (batch, seq_len, section_dim*2)
        """
        # Create inverse frequencies for this section dimension
        # section_dim corresponds to half the actual dimension (pairs)
        rotary_dim = section_dim * 2
        
        # Compute inverse frequencies
        idx = ops.arange(0, rotary_dim, 2, dtype="float32")
        denom = ops.cast(rotary_dim, "float32")
        freq_range = idx / denom
        inverse_freq = ops.power(
            ops.cast(self.max_wavelength, "float32"),
            -freq_range
        )
        inverse_freq = inverse_freq / ops.cast(self.scaling_factor, "float32")
        
        # positions shape: (batch, seq_len)
        # inverse_freq shape: (section_dim,)
        positions_float = ops.cast(positions, "float32")
        
        # Compute frequencies: (batch, seq_len, section_dim)
        # Using einsum for batched outer product
        freq = ops.einsum("bs,d->bsd", positions_float, inverse_freq)
        
        # Create cos/sin embeddings by repeating each frequency
        # Stack and reshape to get (batch, seq_len, section_dim*2)
        embedding = ops.stack([freq, freq], axis=-1)
        embedding = ops.reshape(
            embedding,
            (batch_size, seq_len, rotary_dim)
        )
        
        cos_emb = ops.cast(ops.cos(embedding), self.compute_dtype)
        sin_emb = ops.cast(ops.sin(embedding), self.compute_dtype)
        
        return cos_emb, sin_emb
    
    def _apply_rotary_pos_emb_single(self, tensor, cos_emb, sin_emb):
        """Apply rotary position embedding to a single tensor.
        
        Implements: output = tensor * cos + rotate_half(tensor) * sin
        
        Args:
            tensor: Input tensor of shape (batch, seq_len, num_heads, head_dim)
            cos_emb: Cosine embedding of shape (batch, seq_len, 1, head_dim)
            sin_emb: Sine embedding of shape (batch, seq_len, 1, head_dim)
        
        Returns:
            Tensor with rotary embedding applied.
        """
        # Split tensor into two halves and rotate
        x1, x2 = ops.split(tensor, 2, axis=-1)
        
        # Rotate half: stack [-x2, x1] and reshape
        half_rot_tensor = ops.stack([-x2, x1], axis=-2)
        half_rot_tensor = ops.reshape(half_rot_tensor, ops.shape(tensor))
        
        # Apply rotation: tensor * cos + rotate_half(tensor) * sin
        return (tensor * cos_emb) + (half_rot_tensor * sin_emb)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "mrope_section": self.mrope_section,
        })
        return config
    
    def compute_output_shape(self, input_shape):
        # Returns same shape as input
        return input_shape
