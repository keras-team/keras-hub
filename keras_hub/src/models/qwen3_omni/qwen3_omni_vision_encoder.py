import math

import keras
import numpy as np
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export


class Qwen3OmniVisionPatchEmbed(layers.Layer):
    """3D patch embedding layer for Qwen3-Omni vision encoder.

    Converts video or image input into patches using 3D convolution.
    For images, the temporal dimension is 1. For videos, the temporal
    dimension represents frames.

    Args:
        patch_size: int. The spatial patch size (height and width).
        temporal_patch_size: int. The temporal patch size (frames).
        in_channels: int. The number of input channels (e.g., 3 for RGB).
        embed_dim: int. The output embedding dimension.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(
        self,
        patch_size,
        temporal_patch_size,
        in_channels,
        embed_dim,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = layers.Conv3D(
            filters=embed_dim,
            kernel_size=(temporal_patch_size, patch_size, patch_size),
            strides=(temporal_patch_size, patch_size, patch_size),
            use_bias=True,
            data_format="channels_last",
            dtype=dtype,
            name="proj",
        )

    def build(self, input_shape):
        self.proj.build(input_shape)
        self.built = True

    def call(self, pixel_values):
        """Forward pass.

        Args:
            pixel_values: Tensor with shape
                `(batch_size, temporal, height, width, channels)`.

        Returns:
            Tensor with shape `(batch_size, num_patches, embed_dim)`.
        """
        hidden_states = self.proj(pixel_values)
        batch_size = ops.shape(hidden_states)[0]
        embed_dim = ops.shape(hidden_states)[-1]
        hidden_states = ops.reshape(hidden_states, [batch_size, -1, embed_dim])
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "in_channels": self.in_channels,
                "embed_dim": self.embed_dim,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        temporal = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        num_patches = (
            (temporal // self.temporal_patch_size)
            * (height // self.patch_size)
            * (width // self.patch_size)
        )
        return (batch_size, num_patches, self.embed_dim)


class Qwen3OmniVisionRotaryEmbedding(layers.Layer):
    """Rotary position embedding for Qwen3-Omni vision encoder.

    Computes 2D rotary position embeddings from spatial position indices.
    Unlike the text M-RoPE, this operates on (row, col) position pairs.

    Args:
        dim: int. The embedding dimension (head_dim // 2).
        theta: float. The base frequency. Defaults to `10000.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(self, dim, theta=10000.0, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (
            theta ** (np.arange(0, dim, 2, dtype="float32") / dim)
        )
        self._inv_freq = inv_freq

    def call(self, seqlen):
        """Compute rotary frequency table.

        Args:
            seqlen: int. The maximum spatial extent.

        Returns:
            Tensor with shape `(seqlen, dim // 2)`.
        """
        seq = ops.arange(seqlen, dtype="float32")
        inv_freq = ops.convert_to_tensor(self._inv_freq, dtype="float32")
        freqs = ops.einsum("i,j->ij", seq, inv_freq)
        return freqs

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "theta": self.theta})
        return config


def _rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : ops.shape(x)[-1] // 2]
    x2 = x[..., ops.shape(x)[-1] // 2 :]
    return ops.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb_vision(q, k, cos, sin):
    """Apply rotary position embeddings to query and key for vision.

    Args:
        q: Query tensor of shape `(batch, num_heads, seq_len, head_dim)`.
        k: Key tensor of shape `(batch, num_heads, seq_len, head_dim)`.
        cos: Cosine embedding of shape `(seq_len, head_dim)`.
        sin: Sine embedding of shape `(seq_len, head_dim)`.

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied.
    """
    # Reshape for broadcasting: (1, 1, seq_len, head_dim)
    cos = ops.expand_dims(ops.expand_dims(cos, axis=0), axis=0)
    sin = ops.expand_dims(ops.expand_dims(sin, axis=0), axis=0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3OmniVisionAttention(layers.Layer):
    """Multi-head attention for Qwen3-Omni vision encoder.

    Uses fused QKV projection and applies 2D rotary position embeddings.
    Attention is non-causal (bidirectional).

    Args:
        hidden_size: int. The hidden dimension.
        num_heads: int. The number of attention heads.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(self, hidden_size, num_heads, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5

        self.qkv = layers.Dense(
            hidden_size * 3,
            use_bias=True,
            dtype=dtype,
            name="qkv",
        )
        self.proj = layers.Dense(
            hidden_size,
            use_bias=True,
            dtype=dtype,
            name="proj",
        )

    def build(self, input_shape):
        self.qkv.build(input_shape)
        proj_shape = list(input_shape)
        proj_shape[-1] = self.hidden_size
        self.proj.build(proj_shape)
        self.built = True

    def call(self, hidden_states, position_embeddings=None, training=False):
        """Forward pass.

        Args:
            hidden_states: Tensor of shape
                `(batch_size, seq_len, hidden_size)`.
            position_embeddings: Tuple of (cos, sin) tensors for RoPE,
                each of shape `(seq_len, head_dim)`.
            training: bool. Whether in training mode.

        Returns:
            Tensor of shape `(batch_size, seq_len, hidden_size)`.
        """
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Fused QKV projection
        qkv = self.qkv(hidden_states)
        qkv = ops.reshape(
            qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim]
        )
        qkv = ops.transpose(qkv, [2, 0, 3, 1, 4])
        query, key, value = ops.split(qkv, 3, axis=0)
        query = ops.squeeze(query, axis=0)
        key = ops.squeeze(key, axis=0)
        value = ops.squeeze(value, axis=0)

        # Apply rotary position embeddings
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = _apply_rotary_pos_emb_vision(query, key, cos, sin)

        # Scaled dot-product attention (non-causal)
        attn_weights = (
            ops.matmul(query, ops.transpose(key, [0, 1, 3, 2])) * self.scaling
        )
        attn_weights = ops.softmax(ops.cast(attn_weights, "float32"), axis=-1)
        attn_weights = ops.cast(attn_weights, self.compute_dtype)

        attn_output = ops.matmul(attn_weights, value)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(
            attn_output, [batch_size, seq_len, self.hidden_size]
        )
        attn_output = self.proj(attn_output)
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class Qwen3OmniVisionMLP(layers.Layer):
    """Feed-forward MLP for Qwen3-Omni vision encoder.

    Args:
        hidden_size: int. The hidden dimension.
        intermediate_size: int. The MLP intermediate dimension.
        hidden_act: string. The activation function name.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(
        self, hidden_size, intermediate_size, hidden_act, dtype=None, **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.fc1 = layers.Dense(
            intermediate_size,
            use_bias=True,
            dtype=dtype,
            name="fc1",
        )
        if hidden_act in ("gelu_pytorch_tanh", "gelu_approximate"):
            self.act_fn = lambda x: keras.activations.gelu(x, approximate=True)
        else:
            self.act_fn = layers.Activation(
                hidden_act, dtype=dtype, name="act_fn"
            )
        self.fc2 = layers.Dense(
            hidden_size,
            use_bias=True,
            dtype=dtype,
            name="fc2",
        )

    def build(self, input_shape):
        self.fc1.build(input_shape)
        mid_shape = list(input_shape)
        mid_shape[-1] = self.intermediate_size
        if hasattr(self.act_fn, "build"):
            self.act_fn.build(mid_shape)
        self.fc2.build(mid_shape)
        self.built = True

    def call(self, hidden_states):
        return self.fc2(self.act_fn(self.fc1(hidden_states)))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class Qwen3OmniVisionPatchMerger(layers.Layer):
    """Spatial patch merger for Qwen3-Omni vision encoder.

    Merges spatially adjacent patches and projects to the output dimension.
    When `use_postshuffle_norm` is True, LayerNorm is applied after merging
    (used for deepstack mergers). When False, LayerNorm is applied before
    merging (used for the main merger).

    Args:
        hidden_size: int. The hidden dimension of the vision encoder.
        spatial_merge_size: int. The spatial merge factor.
        out_hidden_size: int. The output projection dimension.
        use_postshuffle_norm: bool. Whether to apply LayerNorm after
            spatial merging. Defaults to `False`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(
        self,
        hidden_size,
        spatial_merge_size,
        out_hidden_size,
        use_postshuffle_norm=False,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm

        merge_dim = hidden_size * (spatial_merge_size**2)
        norm_dim = merge_dim if use_postshuffle_norm else hidden_size
        self.ln_q = layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="ln_q",
        )
        self._norm_dim = norm_dim
        self._merge_dim = merge_dim

        self.mlp_fc1 = layers.Dense(
            merge_dim,
            use_bias=True,
            dtype=dtype,
            name="mlp_fc1",
        )
        self.mlp_act = layers.Activation("gelu", dtype=dtype, name="mlp_act")
        self.mlp_fc2 = layers.Dense(
            out_hidden_size,
            use_bias=True,
            dtype=dtype,
            name="mlp_fc2",
        )

    def build(self, input_shape):
        self.ln_q.build([None, self._norm_dim])
        self.mlp_fc1.build([None, self._merge_dim])
        self.mlp_act.build([None, self._merge_dim])
        self.mlp_fc2.build([None, self._merge_dim])
        self.built = True

    def call(self, hidden_states):
        """Forward pass.

        Args:
            hidden_states: Tensor of shape `(num_tokens, hidden_size)`.

        Returns:
            Tensor of shape `(num_merged_tokens, out_hidden_size)`.
        """
        if self.use_postshuffle_norm:
            hidden_states = ops.reshape(hidden_states, [-1, self._merge_dim])
            hidden_states = self.ln_q(hidden_states)
        else:
            hidden_states = self.ln_q(hidden_states)
            hidden_states = ops.reshape(hidden_states, [-1, self._merge_dim])

        hidden_states = self.mlp_fc1(hidden_states)
        hidden_states = self.mlp_act(hidden_states)
        hidden_states = self.mlp_fc2(hidden_states)
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "spatial_merge_size": self.spatial_merge_size,
                "out_hidden_size": self.out_hidden_size,
                "use_postshuffle_norm": self.use_postshuffle_norm,
            }
        )
        return config


class Qwen3OmniVisionBlock(layers.Layer):
    """Vision transformer block for Qwen3-Omni.

    Implements a Vision Transformer (ViT) block with pre-normalization,
    multi-head attention with rotary position embeddings, and a
    feed-forward MLP.

    Args:
        hidden_size: int. The hidden dimension.
        num_heads: int. The number of attention heads.
        intermediate_size: int. The MLP intermediate dimension.
        hidden_act: string. The activation function name.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the layer's computations and weights.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        hidden_act="gelu",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="norm1",
        )
        self.attn = Qwen3OmniVisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dtype=dtype,
            name="attn",
        )
        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6,
            dtype=dtype,
            name="norm2",
        )
        self.mlp = Qwen3OmniVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            dtype=dtype,
            name="mlp",
        )

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.attn.build(input_shape)
        self.norm2.build(input_shape)
        self.mlp.build(input_shape)
        self.built = True

    def call(
        self,
        hidden_states,
        position_embeddings=None,
        training=False,
    ):
        """Forward pass.

        Args:
            hidden_states: Tensor with shape
                `(batch_size, sequence_length, hidden_size)`.
            position_embeddings: Tuple of (cos, sin) for RoPE, or None.
            training: bool. Whether the layer is in training mode.

        Returns:
            Tensor with shape `(batch_size, sequence_length, hidden_size)`.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            training=training,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "hidden_act": self.hidden_act,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


@keras_hub_export("keras_hub.models.Qwen3OmniVisionEncoder")
class Qwen3OmniVisionEncoder(keras.layers.Layer):
    """Vision encoder for Qwen3-Omni.

    This encoder processes image and video inputs using a Vision Transformer
    (ViT) architecture with:
    - 3D patch embedding for spatiotemporal features
    - Learnable position embeddings with bilinear interpolation
    - 2D rotary position embeddings (RoPE) in attention
    - Vision transformer blocks
    - Spatial patch merging with output projection
    - Deepstack intermediate feature collection

    Args:
        depth: int. The number of transformer layers. Defaults to `27`.
        hidden_size: int. The hidden dimension. Defaults to `1152`.
        hidden_act: string. The activation function name.
            Defaults to `"gelu_pytorch_tanh"`.
        intermediate_size: int. The MLP intermediate dimension.
            Defaults to `4304`.
        num_heads: int. The number of attention heads. Defaults to `16`.
        in_channels: int. The number of input channels. Defaults to `3`.
        patch_size: int. The spatial patch size. Defaults to `16`.
        spatial_merge_size: int. The spatial merge factor for downsampling.
            Defaults to `2`.
        temporal_patch_size: int. The temporal patch size for videos.
            Defaults to `2`.
        out_hidden_size: int. The output projection dimension.
            Defaults to `3584`.
        num_position_embeddings: int. Number of position embeddings in the
            learnable embedding table. Defaults to `2304`.
        deepstack_visual_indexes: list of int. Layer indices at which to
            collect deepstack intermediate features. Defaults to
            `[8, 16, 24]`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to use
            for the model's computations and weights.

    Example:
    ```python
    import numpy as np
    import keras_hub

    # Create encoder
    vision_encoder = keras_hub.models.Qwen3OmniVisionEncoder(
        hidden_size=1152,
        depth=27,
        num_heads=16,
        intermediate_size=4304,
        out_hidden_size=3584,
    )

    pixel_values = np.random.uniform(size=(1, 2, 14, 14, 3))
    grid_thw = np.array([[1, 14, 14]])
    output = vision_encoder({
        "pixel_values": pixel_values,
        "grid_thw": grid_thw,
    })
    ```
    """

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=None,
        dtype=None,
        **kwargs,
    ):
        # Call parent init FIRST (required for PyTorch backend)
        super().__init__(dtype=dtype, **kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes or [
            8,
            16,
            24,
        ]

        self.num_grid_per_side = int(math.sqrt(num_position_embeddings))
        self.spatial_merge_unit = spatial_merge_size * spatial_merge_size

        # === Patch embedding ===
        self.patch_embed = Qwen3OmniVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
            dtype=dtype,
            name="patch_embed",
        )

        # === Learnable position embeddings ===
        self.pos_embed = layers.Embedding(
            input_dim=num_position_embeddings,
            output_dim=hidden_size,
            dtype=dtype,
            name="pos_embed",
        )

        # === Vision rotary position embeddings ===
        head_dim = hidden_size // num_heads
        self.rotary_pos_emb = Qwen3OmniVisionRotaryEmbedding(
            head_dim // 2,
            dtype=dtype,
            name="rotary_pos_emb",
        )

        # === Vision transformer blocks ===
        self.blocks = []
        for i in range(depth):
            block = Qwen3OmniVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                dtype=dtype,
                name=f"block_{i}",
            )
            self.blocks.append(block)

        # === Main patch merger (pre-shuffle norm) ===
        self.merger = Qwen3OmniVisionPatchMerger(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
            use_postshuffle_norm=False,
            dtype=dtype,
            name="merger",
        )

        # === Deepstack mergers (post-shuffle norm) ===
        self.merger_list = []
        for i in range(len(self.deepstack_visual_indexes)):
            merger = Qwen3OmniVisionPatchMerger(
                hidden_size=hidden_size,
                spatial_merge_size=spatial_merge_size,
                out_hidden_size=out_hidden_size,
                use_postshuffle_norm=True,
                dtype=dtype,
                name=f"merger_list_{i}",
            )
            self.merger_list.append(merger)

    def build(self, input_shape=None):
        # Build patch embedding with a representative shape
        patch_shape = (
            None,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        self.patch_embed.build(patch_shape)

        # Build position embedding
        self.pos_embed.build([None])

        # Build transformer blocks
        block_shape = (None, None, self.hidden_size)
        for block in self.blocks:
            block.build(block_shape)

        # Build mergers
        merger_shape = (None, self.hidden_size)
        self.merger.build(merger_shape)
        for merger in self.merger_list:
            merger.build(merger_shape)

        self.built = True

    def _compute_rot_pos_emb(self, grid_thw):
        """Compute 2D rotary position embeddings from grid_thw.

        For each image/video in the batch, computes (row, col) position
        indices accounting for the spatial merge pattern, then looks up
        rotary frequencies.

        Args:
            grid_thw: Integer tensor of shape `(num_images, 3)` where each
                row is `(temporal, height, width)` in patch units.

        Returns:
            Tuple of (cos, sin) tensors, each of shape
            `(total_tokens, head_dim)`.
        """
        merge_size = self.spatial_merge_size

        # Find max spatial extent for frequency table
        grid_h = grid_thw[:, 1]
        grid_w = grid_thw[:, 2]
        max_hw = ops.cast(ops.max(ops.maximum(grid_h, grid_w)), "int32")
        freq_table = self.rotary_pos_emb(max_hw)

        # Build (row, col) position indices for all tokens
        pos_ids_list = []
        for idx in range(ops.shape(grid_thw)[0]):
            t = ops.cast(grid_thw[idx, 0], "int32")
            h = ops.cast(grid_thw[idx, 1], "int32")
            w = ops.cast(grid_thw[idx, 2], "int32")
            merged_h = h // merge_size
            merged_w = w // merge_size

            block_rows = ops.arange(merged_h, dtype="int32")
            block_cols = ops.arange(merged_w, dtype="int32")
            intra_row = ops.arange(merge_size, dtype="int32")
            intra_col = ops.arange(merge_size, dtype="int32")

            # Full-resolution row positions:
            # block_rows[:, None, None, None] * merge_size
            #     + intra_row[None, None, :, None]
            row_idx = ops.reshape(
                block_rows, [-1, 1, 1, 1]
            ) * merge_size + ops.reshape(intra_row, [1, 1, -1, 1])
            col_idx = ops.reshape(
                block_cols, [1, -1, 1, 1]
            ) * merge_size + ops.reshape(intra_col, [1, 1, 1, -1])

            # Broadcast to (merged_h, merged_w, merge_size, merge_size)
            row_idx = ops.broadcast_to(
                row_idx,
                [merged_h, merged_w, merge_size, merge_size],
            )
            col_idx = ops.broadcast_to(
                col_idx,
                [merged_h, merged_w, merge_size, merge_size],
            )

            row_idx = ops.reshape(row_idx, [-1])
            col_idx = ops.reshape(col_idx, [-1])

            # Stack to (num_spatial_tokens, 2)
            coords = ops.stack([row_idx, col_idx], axis=-1)

            # Repeat for temporal frames
            if t > 1:
                coords = ops.tile(coords, [t, 1])

            pos_ids_list.append(coords)

        pos_ids = ops.concatenate(pos_ids_list, axis=0)

        # Look up rotary embeddings: (total_tokens, 2, dim//2)
        # -> (total_tokens, dim)
        embeddings = ops.take(freq_table, pos_ids, axis=0)
        embeddings = ops.reshape(embeddings, [ops.shape(embeddings)[0], -1])

        # Double the frequencies and compute cos/sin
        emb = ops.concatenate([embeddings, embeddings], axis=-1)
        cos = ops.cos(emb)
        sin = ops.sin(emb)
        return cos, sin

    def _fast_pos_embed_interpolate(self, grid_thw):
        """Bilinear interpolation of learnable position embeddings.

        Given variable-resolution grids, performs bilinear interpolation
        over the 2D embedding table, then reorders tokens to match the
        spatial merge pattern.

        Args:
            grid_thw: Integer tensor of shape `(num_images, 3)`.

        Returns:
            Tensor of shape `(total_tokens, hidden_size)`.
        """
        grid_ts = grid_thw[:, 0]
        grid_hs = grid_thw[:, 1]
        grid_ws = grid_thw[:, 2]
        merge_size = self.spatial_merge_size
        n = self.num_grid_per_side

        pos_embed_weight = self.pos_embed.embeddings

        patch_pos_embeds_list = []
        for i in range(ops.shape(grid_thw)[0]):
            t = ops.cast(grid_ts[i], "int32")
            h = ops.cast(grid_hs[i], "int32")
            w = ops.cast(grid_ws[i], "int32")

            h_idxs = ops.cast(ops.linspace(0.0, float(n - 1), h), "float32")
            w_idxs = ops.cast(ops.linspace(0.0, float(n - 1), w), "float32")

            h_floor = ops.cast(ops.floor(h_idxs), "int32")
            w_floor = ops.cast(ops.floor(w_idxs), "int32")
            h_ceil = ops.minimum(h_floor + 1, n - 1)
            w_ceil = ops.minimum(w_floor + 1, n - 1)

            dh = h_idxs - ops.cast(h_floor, "float32")
            dw = w_idxs - ops.cast(w_floor, "float32")

            # 4-corner indices into the (n*n,) embedding table
            # Shape: each is (h, w) -> flattened to (h*w,)
            base_h_floor = h_floor * n
            base_h_ceil = h_ceil * n

            # (h, 1) + (1, w) -> (h, w) via broadcasting
            idx_tl = ops.reshape(base_h_floor, [-1, 1]) + ops.reshape(
                w_floor, [1, -1]
            )
            idx_tr = ops.reshape(base_h_floor, [-1, 1]) + ops.reshape(
                w_ceil, [1, -1]
            )
            idx_bl = ops.reshape(base_h_ceil, [-1, 1]) + ops.reshape(
                w_floor, [1, -1]
            )
            idx_br = ops.reshape(base_h_ceil, [-1, 1]) + ops.reshape(
                w_ceil, [1, -1]
            )

            # Bilinear weights: (h, 1) * (1, w) -> (h, w)
            w_tl = ops.reshape(1.0 - dh, [-1, 1]) * ops.reshape(
                1.0 - dw, [1, -1]
            )
            w_tr = ops.reshape(1.0 - dh, [-1, 1]) * ops.reshape(dw, [1, -1])
            w_bl = ops.reshape(dh, [-1, 1]) * ops.reshape(1.0 - dw, [1, -1])
            w_br = ops.reshape(dh, [-1, 1]) * ops.reshape(dw, [1, -1])

            # Flatten and gather
            idx_tl = ops.reshape(idx_tl, [-1])
            idx_tr = ops.reshape(idx_tr, [-1])
            idx_bl = ops.reshape(idx_bl, [-1])
            idx_br = ops.reshape(idx_br, [-1])
            w_tl = ops.reshape(w_tl, [-1, 1])
            w_tr = ops.reshape(w_tr, [-1, 1])
            w_bl = ops.reshape(w_bl, [-1, 1])
            w_br = ops.reshape(w_br, [-1, 1])

            pos_embed = (
                ops.take(pos_embed_weight, idx_tl, axis=0) * w_tl
                + ops.take(pos_embed_weight, idx_tr, axis=0) * w_tr
                + ops.take(pos_embed_weight, idx_bl, axis=0) * w_bl
                + ops.take(pos_embed_weight, idx_br, axis=0) * w_br
            )

            # Repeat for temporal frames: (h*w, hidden) -> (t*h*w, hidden)
            pos_embed = ops.tile(pos_embed, [t, 1])

            # Reorder to spatial merge pattern:
            # (t, h//m, w//m, m, m, hidden) -> permute -> flatten
            pos_embed = ops.reshape(
                pos_embed,
                [
                    t,
                    h // merge_size,
                    merge_size,
                    w // merge_size,
                    merge_size,
                    -1,
                ],
            )
            pos_embed = ops.transpose(pos_embed, [0, 1, 3, 2, 4, 5])
            pos_embed = ops.reshape(pos_embed, [-1, self.hidden_size])

            patch_pos_embeds_list.append(pos_embed)

        return ops.concatenate(patch_pos_embeds_list, axis=0)

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: dict with keys:
                - `"pixel_values"`: Tensor of shape
                    `(total_patches, temporal_patch_size, patch_size,
                    patch_size, in_channels)` — pre-chunked patches.
                - `"grid_thw"`: Integer tensor of shape
                    `(num_images_or_videos, 3)` — the temporal, height, and
                    width of each image/video in patch grid units.
            training: bool. Whether the model is in training mode.

        Returns:
            dict with keys:
                - `"last_hidden_state"`: Tensor of shape
                    `(1, total_tokens, hidden_size)`.
                - `"pooler_output"`: Tensor of shape
                    `(1, total_merged_tokens, out_hidden_size)`.
                - `"deepstack_features"`: List of tensors, each of shape
                    `(1, total_merged_tokens, out_hidden_size)`.
        """
        pixel_values = inputs["pixel_values"]
        grid_thw = inputs["grid_thw"]

        # Patch embedding: (total_patches, t, h, w, c) -> (1, total, hidden)
        hidden_states = self.patch_embed(pixel_values)
        if len(ops.shape(hidden_states)) == 2:
            hidden_states = ops.expand_dims(hidden_states, axis=0)

        # Add interpolated position embeddings
        pos_embeds = self._fast_pos_embed_interpolate(grid_thw)
        pos_embeds = ops.expand_dims(pos_embeds, axis=0)
        hidden_states = hidden_states + pos_embeds

        # Compute rotary position embeddings
        cos, sin = self._compute_rot_pos_emb(grid_thw)
        position_embeddings = (cos, sin)

        # Apply transformer blocks and collect deepstack features
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                training=training,
            )
            if layer_num in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_num)
                # Squeeze batch for merger, then re-add
                hs_2d = ops.squeeze(hidden_states, axis=0)
                deepstack_feat = self.merger_list[ds_idx](hs_2d)
                deepstack_feat = ops.expand_dims(deepstack_feat, axis=0)
                deepstack_feature_lists.append(deepstack_feat)

        hs_2d = ops.squeeze(hidden_states, axis=0)
        merged_hidden_states = self.merger(hs_2d)
        merged_hidden_states = ops.expand_dims(merged_hidden_states, axis=0)

        return {
            "last_hidden_state": hidden_states,
            "pooler_output": merged_hidden_states,
            "deepstack_features": deepstack_feature_lists,
        }

    def compute_output_spec(self, input_spec, **kwargs):
        """Compute output shape for symbolic tracing."""
        pixel_values_spec = input_spec["pixel_values"]
        num_patches = None
        return {
            "last_hidden_state": keras.KerasTensor(
                shape=(1, num_patches, self.hidden_size),
                dtype=pixel_values_spec.dtype,
            ),
            "pooler_output": keras.KerasTensor(
                shape=(1, num_patches, self.out_hidden_size),
                dtype=pixel_values_spec.dtype,
            ),
            "deepstack_features": [
                keras.KerasTensor(
                    shape=(1, num_patches, self.out_hidden_size),
                    dtype=pixel_values_spec.dtype,
                )
                for _ in self.deepstack_visual_indexes
            ],
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depth": self.depth,
                "hidden_size": self.hidden_size,
                "hidden_act": self.hidden_act,
                "intermediate_size": self.intermediate_size,
                "num_heads": self.num_heads,
                "in_channels": self.in_channels,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "temporal_patch_size": self.temporal_patch_size,
                "out_hidden_size": self.out_hidden_size,
                "num_position_embeddings": self.num_position_embeddings,
                "deepstack_visual_indexes": self.deepstack_visual_indexes,
            }
        )
        return config
