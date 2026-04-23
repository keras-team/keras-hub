"""Layers used by `HieraBackbone`.

The layers are laid out to mirror the weight naming of the Hiera image
encoder shipped with SAM2 (`image_encoder.trunk.*` in the HuggingFace
`facebook/sam2-hiera-*-hf` safetensors), so that a future HuggingFace
converter can port weights without per-layer renaming.

Reference implementations:
- https://github.com/facebookresearch/sam2/blob/main/sam2/modeling/backbones/hieradet.py
- https://github.com/facebookresearch/hiera
"""

import keras
from keras import ops


def _do_pool(x, stride):
    """Strided max-pool on a channels-last `(B, H, W, C)` tensor."""
    if stride is None or (stride[0] == 1 and stride[1] == 1):
        return x
    return ops.nn.max_pool(
        x,
        pool_size=stride,
        strides=stride,
        padding="valid",
        data_format="channels_last",
    )


def _window_partition(x, window_size):
    """Partition `(B, H, W, C)` into `(B * num_windows, ws, ws, C)`.

    Pads `H` and `W` up to a multiple of `window_size` if needed, and returns
    the (possibly padded) `(Hp, Wp)` so that `window_unpartition` can strip
    the padding.
    """
    b, h, w, c = (
        ops.shape(x)[0],
        ops.shape(x)[1],
        ops.shape(x)[2],
        ops.shape(x)[3],
    )
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = ops.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    hp, wp = h + pad_h, w + pad_w
    x = ops.reshape(
        x,
        (
            b,
            hp // window_size,
            window_size,
            wp // window_size,
            window_size,
            c,
        ),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = ops.reshape(x, (-1, window_size, window_size, c))
    return windows, (hp, wp)


def _window_unpartition(windows, window_size, pad_hw, hw):
    """Inverse of `_window_partition`. Strips any padding introduced there."""
    hp, wp = pad_hw
    h, w = hw
    c = ops.shape(windows)[-1]
    b = ops.shape(windows)[0] // ((hp // window_size) * (wp // window_size))
    x = ops.reshape(
        windows,
        (b, hp // window_size, wp // window_size, window_size, window_size, c),
    )
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    x = ops.reshape(x, (b, hp, wp, c))
    if hp != h or wp != w:
        x = x[:, :h, :w, :]
    return x


class HieraAbsolutePositionEmbedding(keras.layers.Layer):
    """Adds SAM2-Hiera's two learned positional embeddings to a feature map.

    The first is a low-resolution background grid bicubically interpolated
    to the feature map size. The second is a window-sized grid that is
    tiled across the feature map. Weight names match
    `trunk.pos_embed` and `trunk.pos_embed_window`.
    """

    def __init__(
        self,
        embed_dim,
        background_size,
        window_size,
        feature_map_size,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.background_size = tuple(background_size)
        self.window_size = int(window_size)
        self.feature_map_size = tuple(feature_map_size)

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1,) + self.background_size + (self.embed_dim,),
            initializer="zeros",
            trainable=True,
        )
        self.pos_embed_window = self.add_weight(
            name="pos_embed_window",
            shape=(1, self.window_size, self.window_size, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.built = True

    def call(self, inputs):
        # Explicit `convert_to_tensor` so the JAX backend resolves the
        # Keras variable into a tracer-compatible array before `ops.image`
        # sees it; calling `ops.image.resize` directly on a `KerasVariable`
        # trips `__jax_array__` abstractification.
        pos_embed = ops.convert_to_tensor(self.pos_embed)
        pos_embed_window = ops.convert_to_tensor(self.pos_embed_window)
        pos_embed_full = ops.image.resize(
            pos_embed,
            size=self.feature_map_size,
            interpolation="bicubic",
        )
        tile_h = self.feature_map_size[0] // self.window_size
        tile_w = self.feature_map_size[1] // self.window_size
        pos_embed_window_full = ops.tile(
            pos_embed_window, (1, tile_h, tile_w, 1)
        )
        # If `feature_map_size` is not an exact multiple of `window_size`,
        # the tiled grid is smaller than the feature map; right-pad so the
        # add against `inputs` broadcasts cleanly.
        pad_h = self.feature_map_size[0] - tile_h * self.window_size
        pad_w = self.feature_map_size[1] - tile_w * self.window_size
        if pad_h > 0 or pad_w > 0:
            pos_embed_window_full = ops.pad(
                pos_embed_window_full,
                [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
            )
        # Position-embedding variables are stored at variable_dtype (fp32
        # under mixed precision); cast to the block's compute dtype before
        # adding so the residual path does not introduce a dtype mismatch.
        pos_embed_full = ops.cast(pos_embed_full, self.compute_dtype)
        pos_embed_window_full = ops.cast(
            pos_embed_window_full, self.compute_dtype
        )
        return inputs + pos_embed_full + pos_embed_window_full

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "background_size": self.background_size,
                "window_size": self.window_size,
                "feature_map_size": self.feature_map_size,
            }
        )
        return config


class HieraPatchEmbedding(keras.layers.Layer):
    """7x7 stride-4 patch embedding used by the Hiera trunk.

    This matches SAM2's `image_encoder.trunk.patch_embed.proj`: a single
    `Conv2D(embed_dim, kernel_size=7, strides=4, padding="same")`.
    """

    def __init__(
        self,
        embed_dim,
        kernel_size=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def build(self, input_shape):
        self.proj = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="same",
            dtype=self.dtype_policy,
            name="proj",
        )
        self.proj.build(input_shape)
        self.built = True

    def call(self, inputs):
        return self.proj(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            }
        )
        return config


class HieraMLP(keras.layers.Layer):
    """Two-layer GELU MLP, matching `trunk.blocks.i.mlp.layers`."""

    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        self.layers_0 = keras.layers.Dense(
            self.hidden_dim, dtype=self.dtype_policy, name="layers_0"
        )
        self.layers_0.build(input_shape)
        intermediate_shape = list(input_shape)
        intermediate_shape[-1] = self.hidden_dim
        self.layers_1 = keras.layers.Dense(
            self.output_dim, dtype=self.dtype_policy, name="layers_1"
        )
        self.layers_1.build(tuple(intermediate_shape))
        self.built = True

    def call(self, inputs):
        x = self.layers_0(inputs)
        x = keras.activations.gelu(x, approximate=False)
        x = self.layers_1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {"hidden_dim": self.hidden_dim, "output_dim": self.output_dim}
        )
        return config


class HieraMultiScaleAttention(keras.layers.Layer):
    """Multi-scale attention used by each Hiera block.

    Matches `trunk.blocks.i.attn` in SAM2 — a fused QKV linear, optional
    strided pooling on `q` (for the first block of each pooling stage), and
    an output projection.
    """

    def __init__(self, dim_in, dim_out, num_heads, q_stride=None, **kwargs):
        super().__init__(**kwargs)
        if dim_out % num_heads != 0:
            raise ValueError(
                f"`dim_out` ({dim_out}) must be divisible by `num_heads` "
                f"({num_heads})."
            )
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim**-0.5
        self.q_stride = tuple(q_stride) if q_stride is not None else None

    def build(self, input_shape):
        self.qkv = keras.layers.Dense(
            3 * self.dim_out, dtype=self.dtype_policy, name="qkv"
        )
        self.qkv.build(input_shape)
        proj_input_shape = list(input_shape)
        proj_input_shape[-1] = self.dim_out
        self.proj = keras.layers.Dense(
            self.dim_out, dtype=self.dtype_policy, name="proj"
        )
        self.proj.build(tuple(proj_input_shape))
        self.built = True

    def call(self, x):
        b = ops.shape(x)[0]
        h = ops.shape(x)[1]
        w = ops.shape(x)[2]

        qkv = self.qkv(x)  # (B, H, W, 3 * dim_out)
        qkv = ops.reshape(qkv, (b, h * w, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, H*W, head_dim)

        if self.q_stride is not None:
            # Pool q back into a spatial map, strided-pool it, then flatten
            # again. k and v keep the pre-pool spatial size.
            q_spatial = ops.reshape(
                ops.transpose(q, (0, 2, 1, 3)), (b, h, w, self.dim_out)
            )
            q_spatial = _do_pool(q_spatial, self.q_stride)
            h_out = h // self.q_stride[0]
            w_out = w // self.q_stride[1]
            q = ops.reshape(
                q_spatial, (b, h_out * w_out, self.num_heads, self.head_dim)
            )
            q = ops.transpose(q, (0, 2, 1, 3))
        else:
            h_out, w_out = h, w

        # Explicit einsum attention: `ops.dot_product_attention` on the TF
        # backend asserts equal target/source sequence lengths, which the
        # pooled-`q`, full-`k`/`v` case violates.
        attn_scores = ops.einsum("bntd,bnsd->bnts", q, k) * self.scale
        # Compute the softmax in float32 for numerical stability, then cast
        # back so the subsequent matmul keeps the block's compute dtype.
        attn_scores = ops.cast(
            ops.softmax(ops.cast(attn_scores, "float32"), axis=-1),
            self.compute_dtype,
        )
        attn_output = ops.einsum("bnts,bnsd->bntd", attn_scores, v)
        # (B, num_heads, H_out*W_out, head_dim)
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(attn_output, (b, h_out, w_out, self.dim_out))
        return self.proj(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "q_stride": self.q_stride,
            }
        )
        return config


class HieraBlock(keras.layers.Layer):
    """One Hiera transformer block.

    Mirrors `trunk.blocks.i` in SAM2:
      - `norm1` before attention.
      - Window-partition (skipped when `window_size == 0`, i.e. global attn).
      - `attn` with optional `q_stride` pooling in the first block of each
        pooling stage.
      - A residual whose skip path applies `proj` when `dim_in != dim_out`,
        followed by the same `q_stride` pooling on the skip.
      - `norm2` + `mlp` with another residual.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        q_stride=None,
        window_size=0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.q_stride = tuple(q_stride) if q_stride is not None else None
        self.window_size = window_size
        self.layer_norm_epsilon = layer_norm_epsilon

    def build(self, input_shape):
        self.norm1 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="norm1",
        )
        self.norm1.build(input_shape)

        self.attn = HieraMultiScaleAttention(
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            num_heads=self.num_heads,
            q_stride=self.q_stride,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.attn.build(input_shape)

        out_shape = list(input_shape)
        out_shape[-1] = self.dim_out
        self.norm2 = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="norm2",
        )
        self.norm2.build(tuple(out_shape))

        self.mlp = HieraMLP(
            hidden_dim=int(self.dim_out * self.mlp_ratio),
            output_dim=self.dim_out,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(tuple(out_shape))

        if self.dim_in != self.dim_out:
            # Residual projection for the first block of each pooling stage.
            self.proj = keras.layers.Dense(
                self.dim_out, dtype=self.dtype_policy, name="proj"
            )
            self.proj.build(input_shape)
        else:
            self.proj = None

        self.built = True

    def call(self, x):
        shortcut = x
        x = self.norm1(x)

        if self.dim_in != self.dim_out:
            # Project and pool the skip path so it matches attention's output.
            shortcut = _do_pool(self.proj(x), self.q_stride)
        elif self.q_stride is not None:
            shortcut = _do_pool(shortcut, self.q_stride)

        window_size = self.window_size
        if window_size > 0:
            h, w = ops.shape(x)[1], ops.shape(x)[2]
            x, pad_hw = _window_partition(x, window_size)

        x = self.attn(x)

        if self.q_stride is not None:
            # Attention pooled `q`, so the effective window and feature map
            # sizes shrink.
            window_size = window_size // self.q_stride[0] if window_size else 0
            if window_size > 0:
                h = h // self.q_stride[0]
                w = w // self.q_stride[1]
                pad_hw = (
                    pad_hw[0] // self.q_stride[0],
                    pad_hw[1] // self.q_stride[1],
                )

        if window_size > 0:
            x = _window_unpartition(x, window_size, pad_hw, (h, w))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim_in": self.dim_in,
                "dim_out": self.dim_out,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "q_stride": self.q_stride,
                "window_size": self.window_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
