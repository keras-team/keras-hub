import math

import keras
from keras import ops

from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import Qwen3_5VisionMLP
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionPatchEmbed,
)
from keras_hub.src.models.qwen3_5.qwen3_5_vision_encoder import (
    Qwen3_5VisionRotaryEmbedding,
)

# Queries per attention call. Bounds the frame mask at `(chunk, seq)`.
_QUERY_CHUNK = 4096


class Qwen3_VLVisionAttention(keras.layers.Layer):
    """Multi-head self-attention with 2D rotary embeddings.

    Args:
        hidden_size: int. Embedding dimension.
        num_heads: int. Number of attention heads.
    """

    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = keras.layers.Dense(
            hidden_size * 3,
            use_bias=True,
            dtype=self.dtype_policy,
            name="qkv",
        )
        self.proj = keras.layers.Dense(
            hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="proj",
        )

    def build(self, input_shape):
        self.qkv.build(input_shape)
        self.proj.build(input_shape)
        self.built = True

    def _apply_rotary(self, x, cos, sin):
        # x: (batch, seq, heads, head_dim); cos/sin: (batch, seq, head_dim).
        cos = ops.expand_dims(cos, axis=2)
        sin = ops.expand_dims(sin, axis=2)
        half = self.head_dim // 2
        rotated = ops.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return x * cos + rotated * sin

    def call(self, x, cos, sin, frame_id):
        """Run attention.

        Args:
            x: Tensor `(batch, seq, hidden_size)`.
            cos: Tensor `(batch, seq, head_dim)`.
            sin: Tensor `(batch, seq, head_dim)`.
            frame_id: Int tensor `(batch, seq)`. A patch attends to the
                patches that share its frame id.
        """
        shape = ops.shape(x)
        qkv = ops.reshape(
            self.qkv(x), (shape[0], shape[1], 3, self.num_heads, self.head_dim)
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self._apply_rotary(q, cos, sin)
        k = self._apply_rotary(k, cos, sin)
        # The frame mask is `(chunk, seq)` rather than `(seq, seq)`: queries
        # go through in chunks so a long batch of frames does not need a dense
        # square mask. Lengths that are not known statically use one chunk.
        seq_len = shape[1]
        if isinstance(seq_len, int) and seq_len > _QUERY_CHUNK:
            bounds = [
                (start, min(start + _QUERY_CHUNK, seq_len))
                for start in range(0, seq_len, _QUERY_CHUNK)
            ]
        else:
            bounds = [(0, None)]
        outputs = []
        for start, stop in bounds:
            mask = frame_id[:, start:stop, None] == frame_id[:, None, :]
            outputs.append(
                ops.dot_product_attention(
                    q[:, start:stop],
                    k,
                    v,
                    mask=ops.expand_dims(mask, axis=1),
                    scale=1.0 / math.sqrt(self.head_dim),
                )
            )
        out = outputs[0] if len(outputs) == 1 else ops.concatenate(outputs, 1)
        out = ops.reshape(out, (shape[0], shape[1], self.hidden_size))
        return self.proj(out)

    def compute_output_spec(self, x, cos, sin, frame_id):
        # Attention keeps the shape of the hidden states, so skip tracing the
        # chunked loop symbolically.
        return keras.KerasTensor(
            shape=tuple(x.shape[:-1]) + (self.hidden_size,),
            dtype=self.compute_dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
            }
        )
        return config


class Qwen3_VLVisionBlock(keras.layers.Layer):
    """Pre-norm transformer block of the vision tower.

    Args:
        hidden_size: int. Embedding dimension.
        num_heads: int. Number of attention heads.
        intermediate_size: int. MLP hidden dimension.
        layer_norm_epsilon: float. Epsilon of both layer norms.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, dtype=self.dtype_policy, name="norm1"
        )
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, dtype=self.dtype_policy, name="norm2"
        )
        self.attn = Qwen3_VLVisionAttention(
            hidden_size, num_heads, dtype=self.dtype_policy, name="attn"
        )
        self.mlp = Qwen3_5VisionMLP(
            hidden_size,
            intermediate_size,
            dtype=self.dtype_policy,
            name="mlp",
        )

    def build(self, input_shape):
        self.norm1.build(input_shape)
        self.norm2.build(input_shape)
        self.attn.build(input_shape)
        self.mlp.build(input_shape)
        self.built = True

    def call(self, x, cos, sin, frame_id):
        x = x + self.attn(self.norm1(x), cos, sin, frame_id)
        return x + self.mlp(self.norm2(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class Qwen3_VLVisionPatchMerger(keras.layers.Layer):
    """Merges `spatial_merge_size**2` neighbouring patches into one token.

    The final merger normalizes each patch over `hidden_size` and then groups
    patches. The deepstack mergers (`postshuffle_norm=True`) group first and
    normalize over the grouped dimension.

    Args:
        hidden_size: int. Vision hidden dimension.
        spatial_merge_size: int. Side length of the merged patch window.
        out_hidden_size: int. Output dimension.
        postshuffle_norm: bool. Whether to normalize after grouping.
        layer_norm_epsilon: float. Epsilon of the layer norm.
    """

    def __init__(
        self,
        hidden_size,
        spatial_merge_size,
        out_hidden_size,
        postshuffle_norm=False,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.out_hidden_size = out_hidden_size
        self.postshuffle_norm = postshuffle_norm
        self.layer_norm_epsilon = layer_norm_epsilon
        self.merged_size = hidden_size * spatial_merge_size**2
        self.norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon, dtype=self.dtype_policy, name="norm"
        )
        self.linear_fc1 = keras.layers.Dense(
            self.merged_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="linear_fc1",
        )
        self.linear_fc2 = keras.layers.Dense(
            out_hidden_size,
            use_bias=True,
            dtype=self.dtype_policy,
            name="linear_fc2",
        )

    def build(self, input_shape=None):
        norm_dim = (
            self.merged_size if self.postshuffle_norm else self.hidden_size
        )
        self.norm.build((None, norm_dim))
        self.linear_fc1.build((None, self.merged_size))
        self.linear_fc2.build((None, self.merged_size))
        self.built = True

    def call(self, x):
        """Merge patches.

        Args:
            x: Tensor `(batch, num_patches, hidden_size)`, ordered so that
                every `spatial_merge_size**2` consecutive patches form one
                merge window.

        Returns:
            Tensor `(batch, num_patches // spatial_merge_size**2,
            out_hidden_size)`.
        """
        shape = ops.shape(x)
        if not self.postshuffle_norm:
            x = self.norm(x)
        # An explicit count keeps empty inputs valid, where -1 is ambiguous.
        merged_count = shape[1] // self.spatial_merge_size**2
        x = ops.reshape(x, (shape[0], merged_count, self.merged_size))
        if self.postshuffle_norm:
            x = self.norm(x)
        x = self.linear_fc1(x)
        x = ops.gelu(x, approximate=False)
        return self.linear_fc2(x)

    def compute_output_spec(self, x):
        # The merged token count is dynamic, so skip tracing the reshape.
        return keras.KerasTensor(
            shape=(x.shape[0], None, self.out_hidden_size),
            dtype=self.compute_dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "spatial_merge_size": self.spatial_merge_size,
                "out_hidden_size": self.out_hidden_size,
                "postshuffle_norm": self.postshuffle_norm,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


class Qwen3_VLVisionTokenLayout(keras.layers.Layer):
    """Locate every patch inside its item, frame and merge window.

    Args:
        spatial_merge_size: int. Side length of the merge window.

    Call arguments:
        pixel_values: Tensor `(batch, num_patches, ...)`. Only the patch
            count is used.
        grid_thw: int Tensor `(batch, num_items, 3)`.

    Returns:
        Tuple `(row, col, h, w, frame_id)` of `(batch, num_patches)` int32
        tensors. `row`/`col` are patch coordinates inside the frame, `h` and
        `w` the frame size of the owning item, and `frame_id` is unique per
        frame (`-1` for padding patches).
    """

    def __init__(self, spatial_merge_size, **kwargs):
        super().__init__(**kwargs)
        self.spatial_merge_size = spatial_merge_size

    def build(self, pixel_values_shape=None, grid_thw_shape=None):
        self.built = True

    def call(self, pixel_values, grid_thw):
        merge = self.spatial_merge_size
        grid_thw = ops.cast(grid_thw, "int32")
        t, h, w = grid_thw[..., 0], grid_thw[..., 1], grid_thw[..., 2]
        ends = ops.cumsum(t * h * w, axis=1)
        starts = ends - t * h * w
        frame_starts = ops.cumsum(t, axis=1) - t

        # Patch index without needing a static number of patches.
        ones = ops.ones_like(pixel_values[:, :, 0, 0, 0, 0], dtype="int32")
        index = ops.cumsum(ones, axis=1) - 1
        valid = index < ends[:, -1:]
        item = ops.sum(
            ops.cast(index[:, :, None] >= ends[:, None, :], "int32"), axis=-1
        )
        last_item = ops.sum(ops.ones_like(t), axis=1, keepdims=True) - 1
        item = ops.minimum(item, last_item)

        def gather(x):
            return ops.take_along_axis(x, item, axis=1)

        h_p, w_p = gather(h), gather(w)
        area = ops.maximum(h_p * w_p, 1)
        local = index - gather(starts)
        frame = local // area
        within = local % area

        # Within a frame, patches are grouped into merge windows.
        windows_per_row = ops.maximum(w_p // merge, 1)
        window = within // (merge * merge)
        inner = within % (merge * merge)
        row = (window // windows_per_row) * merge + inner // merge
        col = (window % windows_per_row) * merge + inner % merge
        frame_id = ops.where(valid, gather(frame_starts) + frame, -1)
        return row, col, h_p, w_p, frame_id

    def compute_output_spec(self, pixel_values, grid_thw):
        # Each output needs its own tensor so the graph tells them apart.
        return tuple(
            keras.KerasTensor(pixel_values.shape[:2], dtype="int32")
            for _ in range(5)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"spatial_merge_size": self.spatial_merge_size})
        return config


class Qwen3_VLVisionPatchEmbedding(Qwen3_5VisionPatchEmbed):
    """Batched Conv3D patch embedding, shared with Qwen3.5.

    The Qwen3.5 layer takes one patch per row. This one takes
    `(batch, num_patches, T, P, P, C)` and returns `(batch, num_patches,
    hidden_size)`.
    """

    def call(self, pixel_values):
        pixel_values = ops.cast(pixel_values, "float32")
        shape = ops.shape(pixel_values)
        patch_shape = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
            self.in_channels,
        )
        x = ops.reshape(pixel_values, (-1,) + patch_shape)
        # Conv3D on an empty batch returns a wrong shape on TensorFlow, so a
        # zero row is appended and dropped again.
        x = ops.concatenate([x, ops.zeros((1,) + patch_shape, x.dtype)], axis=0)
        x = super().call(x)[:-1]
        return ops.reshape(x, (shape[0], shape[1], self.hidden_size))

    def compute_output_spec(self, pixel_values):
        return keras.KerasTensor(
            shape=tuple(pixel_values.shape[:2]) + (self.hidden_size,),
            dtype="float32",
        )


class Qwen3_VLVisionPositionEmbedding(keras.layers.Layer):
    """Adds the learned position table, bilinearly sampled per frame.

    Args:
        num_position_embeddings: int. Rows of the table. Must be a perfect
            square.
        hidden_size: int. Embedding dimension.

    Call arguments:
        x: Tensor `(batch, num_patches, hidden_size)`.
        row, col, h, w: int Tensors `(batch, num_patches)` from
            `Qwen3_VLVisionTokenLayout`.
    """

    def __init__(self, num_position_embeddings, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.num_position_embeddings = num_position_embeddings
        self.hidden_size = hidden_size
        self.num_grid_per_side = int(math.isqrt(num_position_embeddings))

    def build(self, x_shape=None, *index_shapes):
        self.embeddings = self.add_weight(
            shape=(self.num_position_embeddings, self.hidden_size),
            initializer="uniform",
            dtype=self.variable_dtype,
            name="embeddings",
        )
        self.built = True

    def call(self, x, row, col, h, w):
        side = self.num_grid_per_side

        def coordinate(index, size):
            step = (side - 1) / ops.cast(ops.maximum(size - 1, 1), "float32")
            return ops.cast(index, "float32") * step

        row_pos, col_pos = coordinate(row, h), coordinate(col, w)
        row_floor = ops.floor(row_pos)
        col_floor = ops.floor(col_pos)
        row_frac = row_pos - row_floor
        col_frac = col_pos - col_floor
        row_lo = ops.cast(row_floor, "int32")
        col_lo = ops.cast(col_floor, "int32")
        row_hi = ops.minimum(row_lo + 1, side - 1)
        col_hi = ops.minimum(col_lo + 1, side - 1)

        def lookup(r, c, weight):
            rows = ops.take(self.embeddings, r * side + c, axis=0)
            return rows * weight[..., None]

        return x + (
            lookup(row_lo, col_lo, (1 - row_frac) * (1 - col_frac))
            + lookup(row_lo, col_hi, (1 - row_frac) * col_frac)
            + lookup(row_hi, col_lo, row_frac * (1 - col_frac))
            + lookup(row_hi, col_hi, row_frac * col_frac)
        )

    def compute_output_spec(self, x, row, col, h, w):
        return keras.KerasTensor(x.shape, dtype=x.dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_position_embeddings": self.num_position_embeddings,
                "hidden_size": self.hidden_size,
            }
        )
        return config


class Qwen3_VLVisionRotary(Qwen3_5VisionRotaryEmbedding):
    """2D rotary tables from patch coordinates.

    Args:
        head_dim: int. Size of an attention head.
        theta: float. Base wavelength. Defaults to `10000.0`.

    Returns:
        `(cos, sin)`, each `(batch, num_patches, head_dim)`.
    """

    def build(self, row_shape=None, col_shape=None):
        self.built = True

    def call(self, row, col):
        inv_freq = ops.convert_to_tensor(self._inv_freq_vals, dtype="float32")
        row_freqs = ops.cast(row, "float32")[..., None] * inv_freq
        col_freqs = ops.cast(col, "float32")[..., None] * inv_freq
        freqs = ops.concatenate([row_freqs, col_freqs], axis=-1)
        emb = ops.concatenate([freqs, freqs], axis=-1)
        return ops.cos(emb), ops.sin(emb)

    def compute_output_spec(self, row, col):
        shape = tuple(row.shape[:2]) + (self.head_dim,)
        return (
            keras.KerasTensor(shape, dtype="float32"),
            keras.KerasTensor(shape, dtype="float32"),
        )


class Qwen3_VLVisionEncoder(keras.Model):
    """Vision tower of Qwen3-VL.

    Patches are embedded with a Conv3D, summed with a bilinearly interpolated
    learned position table, and run through `depth` transformer blocks with
    2D rotary embeddings. Attention is restricted to each frame of each item.
    The output of the blocks listed in `deepstack_visual_indexes` is also
    passed through extra mergers for injection into the language model.

    Patches must be in merge-window-major order: for each frame, the patches
    of every `spatial_merge_size x spatial_merge_size` window are consecutive.

    The tower always runs in float32 regardless of `dtype`.

    Args:
        depth: int. Number of transformer blocks.
        hidden_size: int. Vision hidden dimension.
        num_heads: int. Number of attention heads.
        intermediate_size: int. MLP hidden dimension.
        out_hidden_size: int. Output dimension, equal to the text model's
            hidden size.
        patch_size: int. Spatial patch size in pixels. Defaults to `16`.
        temporal_patch_size: int. Frames per patch. Defaults to `2`.
        in_channels: int. Input channels. Defaults to `3`.
        spatial_merge_size: int. Side length of the merge window. Defaults to
            `2`.
        num_position_embeddings: int. Size of the learned position table. Must
            be a perfect square. Defaults to `2304`.
        deepstack_visual_indexes: tuple of int. Zero-based indices of the
            blocks whose outputs feed the deepstack mergers. Defaults to
            `(8, 16, 24)`.
        layer_norm_epsilon: float. Epsilon of all layer norms. Defaults to
            `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. Ignored in favor
            of float32.

    This is a functional model, so `summary()` lists the layers and the
    parameter count.

    Call arguments:
        inputs: A dict with `"pixel_values"`, a tensor `(batch, num_patches,
            temporal_patch_size, patch_size, patch_size, in_channels)`, and
            `"image_grid_thw"`, an int tensor `(batch, num_items, 3)` with
            `(t, h, w)` in patch units before merging. Rows of zeros are
            treated as padding.

    Returns:
        A list `[features, *deepstack_features]`. `features` has shape
        `(batch, num_patches // spatial_merge_size**2, out_hidden_size)`.
        There is one `deepstack_features` tensor of the same shape per entry
        of `deepstack_visual_indexes`.
    """

    def __init__(
        self,
        depth,
        hidden_size,
        num_heads,
        intermediate_size,
        out_hidden_size,
        patch_size=16,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        num_position_embeddings=2304,
        deepstack_visual_indexes=(8, 16, 24),
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        if any(not 0 <= i < depth for i in tuple(deepstack_visual_indexes)):
            raise ValueError(
                "`deepstack_visual_indexes` must all be in `[0, depth)`. "
                f"Received: {tuple(deepstack_visual_indexes)} with "
                f"depth={depth}"
            )
        num_grid_per_side = int(math.isqrt(num_position_embeddings))
        if num_grid_per_side**2 != num_position_embeddings:
            raise ValueError(
                "`num_position_embeddings` must be a perfect square. "
                f"Received: {num_position_embeddings}"
            )
        deepstack_visual_indexes = tuple(deepstack_visual_indexes)
        head_dim = hidden_size // num_heads

        # === Functional Model ===
        pixel_values = keras.Input(
            shape=(
                None,
                temporal_patch_size,
                patch_size,
                patch_size,
                in_channels,
            ),
            dtype="float32",
            name="pixel_values",
        )
        grid_thw = keras.Input(
            shape=(None, 3), dtype="int32", name="image_grid_thw"
        )
        row, col, h, w, frame_id = Qwen3_VLVisionTokenLayout(
            spatial_merge_size, dtype="float32", name="token_layout"
        )(pixel_values, grid_thw)
        x = Qwen3_VLVisionPatchEmbedding(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            dtype="float32",
            name="patch_embed",
        )(pixel_values)
        x = Qwen3_VLVisionPositionEmbedding(
            num_position_embeddings,
            hidden_size,
            dtype="float32",
            name="pos_embed",
        )(x, row, col, h, w)
        cos, sin = Qwen3_VLVisionRotary(
            head_dim, dtype="float32", name="rotary"
        )(row, col)
        deepstack_features = []
        for i in range(depth):
            x = Qwen3_VLVisionBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype="float32",
                name=f"blocks_{i}",
            )(x, cos, sin, frame_id)
            if i in deepstack_visual_indexes:
                position = deepstack_visual_indexes.index(i)
                deepstack_features.append(
                    Qwen3_VLVisionPatchMerger(
                        hidden_size=hidden_size,
                        spatial_merge_size=spatial_merge_size,
                        out_hidden_size=out_hidden_size,
                        postshuffle_norm=True,
                        layer_norm_epsilon=layer_norm_epsilon,
                        dtype="float32",
                        name=f"deepstack_merger_list_{position}",
                    )(x)
                )
        features = Qwen3_VLVisionPatchMerger(
            hidden_size=hidden_size,
            spatial_merge_size=spatial_merge_size,
            out_hidden_size=out_hidden_size,
            postshuffle_norm=False,
            layer_norm_epsilon=layer_norm_epsilon,
            dtype="float32",
            name="merger",
        )(x)

        super().__init__(
            inputs={"pixel_values": pixel_values, "image_grid_thw": grid_thw},
            outputs=[features] + deepstack_features,
            dtype="float32",
            **kwargs,
        )

        # === Config ===
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.spatial_merge_size = spatial_merge_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.layer_norm_epsilon = layer_norm_epsilon
        self.head_dim = head_dim
        self.num_grid_per_side = num_grid_per_side

    @property
    def patch_embed(self):
        return self.get_layer("patch_embed")

    @property
    def pos_embed(self):
        return self.get_layer("pos_embed")

    @property
    def blocks(self):
        return [self.get_layer(f"blocks_{i}") for i in range(self.depth)]

    @property
    def merger(self):
        return self.get_layer("merger")

    @property
    def deepstack_merger_list(self):
        return [
            self.get_layer(f"deepstack_merger_list_{i}")
            for i in range(len(self.deepstack_visual_indexes))
        ]

    def get_config(self):
        return {
            "name": self.name,
            "trainable": self.trainable,
            "depth": self.depth,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "out_hidden_size": self.out_hidden_size,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "in_channels": self.in_channels,
            "spatial_merge_size": self.spatial_merge_size,
            "num_position_embeddings": self.num_position_embeddings,
            "deepstack_visual_indexes": list(self.deepstack_visual_indexes),
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "dtype": keras.dtype_policies.serialize(self.dtype_policy),
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
