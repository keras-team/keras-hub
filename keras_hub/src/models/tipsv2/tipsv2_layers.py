"""Shared layers for TIPSv2 vision-language model.

Implements the building blocks for both the DINOv2-style ViT vision encoder
and the custom text encoder with sinusoidal positional embeddings.

References:
- HF: google/tipsv2-b14 (image_encoder.py, text_encoder.py)
- GitHub: google-deepmind/tips/pytorch
"""

import math

import keras
from keras import layers
from keras import ops


class TIPSv2LayerScale(keras.layers.Layer):
    """Layer scale: element-wise multiplication by a learnable vector.

    Used in the vision encoder to scale residual connections.
    Initialized to `init_values` (typically 1.0 for TIPSv2).

    Args:
        dim: int. Dimension of the layer scale vector.
        init_values: float. Initial value for all elements. Defaults to
            `1e-5`.
    """

    def __init__(self, dim, init_values=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.init_values = init_values

    def build(self, input_shape):
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
        )
        self.built = True

    def call(self, x):
        return x * ops.cast(self.gamma, dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "init_values": self.init_values,
            }
        )
        return config


class TIPSv2PatchEmbedding(keras.layers.Layer):
    """2D image to patch embedding: (B, H, W, C) -> (B, N, D).

    Applies a Conv2D with kernel_size=patch_size and stride=patch_size,
    then flattens the spatial dimensions.

    Args:
        hidden_dim: int. Output embedding dimension.
        patch_size: int. Size of each square patch.
        image_size: int. Expected input image size (height=width).
        data_format: str. Data format for the Conv2D layer.
    """

    def __init__(
        self,
        hidden_dim,
        patch_size,
        image_size,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = layers.Conv2D(
            filters=hidden_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            data_format=data_format,
            dtype=self.dtype_policy,
            name="projection",
        )

    def build(self, input_shape):
        self.projection.build(input_shape)
        self.built = True

    def call(self, x):
        # x: (B, H, W, C)
        x = self.projection(x)  # (B, H', W', D)
        # Flatten spatial dims.
        shape = ops.shape(x)
        x = ops.reshape(x, (shape[0], -1, self.hidden_dim))  # (B, N, D)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_size": self.patch_size,
                "image_size": self.image_size,
            }
        )
        return config


class TIPSv2VisionAttention(keras.layers.Layer):
    """Multi-head attention with fused QKV projection.

    Follows the DINOv2 attention implementation with a single linear
    layer for Q, K, V projection.

    Args:
        dim: int. Input/output dimension.
        num_heads: int. Number of attention heads.
        qkv_bias: bool. Whether to use bias in the QKV projection.
            Defaults to `True`.
        proj_bias: bool. Whether to use bias in the output projection.
            Defaults to `True`.
        attn_drop: float. Attention dropout rate. Defaults to `0.0`.
        proj_drop: float. Output projection dropout rate. Defaults to `0.0`.
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.proj_bias = proj_bias
        self.attn_drop_rate = attn_drop
        self.proj_drop_rate = proj_drop

        self.qkv = layers.Dense(
            dim * 3,
            use_bias=qkv_bias,
            dtype=self.dtype_policy,
            name="qkv",
        )
        self.attn_drop = layers.Dropout(attn_drop, dtype=self.dtype_policy)
        self.proj = layers.Dense(
            dim,
            use_bias=proj_bias,
            dtype=self.dtype_policy,
            name="proj",
        )
        self.proj_drop = layers.Dropout(proj_drop, dtype=self.dtype_policy)

    def build(self, input_shape):
        self.qkv.build(input_shape)
        self.proj.build(input_shape)
        self.built = True

    def call(self, x, training=None):
        shape = ops.shape(x)
        b, n = shape[0], shape[1]

        qkv = self.qkv(x)  # (B, N, 3 * D)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # (B,H,N,N)
        attn = attn * self.scale
        attn = ops.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = ops.matmul(attn, v)  # (B, H, N, Dh)
        x = ops.transpose(x, (0, 2, 1, 3))  # (B, N, H, Dh)
        x = ops.reshape(x, (b, n, self.dim))

        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "proj_bias": self.proj_bias,
                "attn_drop": self.attn_drop_rate,
                "proj_drop": self.proj_drop_rate,
            }
        )
        return config


class TIPSv2MLP(keras.layers.Layer):
    """Transformer MLP (two-layer FFN with GELU activation).

    Args:
        hidden_features: int. Hidden layer dimension.
        out_features: int. Output dimension. If None, same as input.
        drop: float. Dropout rate. Defaults to `0.0`.
    """

    def __init__(
        self,
        hidden_features,
        out_features=None,
        drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.drop_rate = drop

    def build(self, input_shape):
        in_features = input_shape[-1]
        out_features = self.out_features or in_features

        self.fc1 = layers.Dense(
            self.hidden_features,
            dtype=self.dtype_policy,
            name="fc1",
        )
        self.fc1.build(input_shape)
        self.fc2 = layers.Dense(
            out_features,
            dtype=self.dtype_policy,
            name="fc2",
        )
        self.fc2.build((*input_shape[:-1], self.hidden_features))
        self.drop = layers.Dropout(self.drop_rate, dtype=self.dtype_policy)
        self.built = True

    def call(self, x, training=None):
        x = self.fc1(x)
        x = ops.gelu(x, approximate=False)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.out_features or input_shape[-1]
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
                "drop": self.drop_rate,
            }
        )
        return config


class TIPSv2SwiGLU(keras.layers.Layer):
    """SwiGLU FFN layer following DINOv2 implementation.

    Uses a single linear layer to project to 2*hidden, then chunks into
    two halves: one goes through SiLU, then element-wise multiply.

    Args:
        hidden_features: int. Hidden dimension (before the 2/3 adjustment).
        out_features: int. Output dimension. If None, same as input.
    """

    def __init__(
        self,
        hidden_features=None,
        out_features=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_features = hidden_features
        self.out_features = out_features

    def build(self, input_shape):
        in_features = input_shape[-1]
        out_features = self.out_features or in_features
        hidden_features = self.hidden_features or in_features
        # Match HF's SwiGLUFFNFused rounding.
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.w12 = layers.Dense(
            2 * hidden_features,
            dtype=self.dtype_policy,
            name="w12",
        )
        self.w12.build(input_shape)
        self.w3 = layers.Dense(
            out_features,
            dtype=self.dtype_policy,
            name="w3",
        )
        self.w3.build((*input_shape[:-1], hidden_features))
        self._hidden_features_actual = hidden_features
        self.built = True

    def call(self, x, training=None):
        x12 = self.w12(x)
        x1 = x12[..., : self._hidden_features_actual]
        x2 = x12[..., self._hidden_features_actual :]
        hidden = ops.silu(x1) * x2
        return self.w3(hidden)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.out_features or input_shape[-1]
        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_features": self.hidden_features,
                "out_features": self.out_features,
            }
        )
        return config


class TIPSv2VisionBlock(keras.layers.Layer):
    """Transformer block for the TIPSv2 vision encoder.

    Pre-norm architecture with LayerScale and optional DropPath.

    Args:
        dim: int. Hidden dimension.
        num_heads: int. Number of attention heads.
        mlp_ratio: float. Ratio of MLP hidden dim to embedding dim.
            Defaults to `4.0`.
        qkv_bias: bool. Use bias in QKV projection. Defaults to `True`.
        init_values: float. LayerScale init value. None disables
            LayerScale. Defaults to `None`.
        ffn_layer: str. FFN type, either "mlp" or "swiglu".
            Defaults to `"mlp"`.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        init_values=None,
        ffn_layer="mlp",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.init_values = init_values
        self.ffn_layer = ffn_layer

    def build(self, input_shape):
        self.norm1 = layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy, name="norm1"
        )
        self.norm1.build(input_shape)

        self.attn = TIPSv2VisionAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.attn.build(input_shape)

        if self.init_values is not None and self.init_values > 0:
            self.ls1 = TIPSv2LayerScale(
                self.dim,
                init_values=self.init_values,
                dtype=self.dtype_policy,
                name="ls1",
            )
            self.ls1.build(input_shape)
        else:
            self.ls1 = None

        self.norm2 = layers.LayerNormalization(
            epsilon=1e-6, dtype=self.dtype_policy, name="norm2"
        )
        self.norm2.build(input_shape)

        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        if self.ffn_layer == "swiglu":
            self.mlp = TIPSv2SwiGLU(
                hidden_features=mlp_hidden_dim,
                dtype=self.dtype_policy,
                name="mlp",
            )
        else:
            self.mlp = TIPSv2MLP(
                hidden_features=mlp_hidden_dim,
                dtype=self.dtype_policy,
                name="mlp",
            )
        self.mlp.build(input_shape)

        if self.init_values is not None and self.init_values > 0:
            self.ls2 = TIPSv2LayerScale(
                self.dim,
                init_values=self.init_values,
                dtype=self.dtype_policy,
                name="ls2",
            )
            self.ls2.build(input_shape)
        else:
            self.ls2 = None

        self.built = True

    def call(self, x, training=None):
        # Attention residual.
        residual = self.attn(self.norm1(x), training=training)
        if self.ls1 is not None:
            residual = self.ls1(residual)
        x = x + residual

        # FFN residual.
        residual = self.mlp(self.norm2(x), training=training)
        if self.ls2 is not None:
            residual = self.ls2(residual)
        x = x + residual
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio,
                "qkv_bias": self.qkv_bias,
                "init_values": self.init_values,
                "ffn_layer": self.ffn_layer,
            }
        )
        return config


# ── Text Encoder Layers ──────────────────────────────────────────────


class TIPSv2SinusoidalPositionEmbedding(keras.layers.Layer):
    """Sinusoidal positional embedding (computed, not learned).

    Generates position embeddings using sin/cos functions with geometric
    timescale progression, matching the TIPSv2 text encoder reference.

    Args:
        embedding_dim: int. Dimension of the positional embedding.
        min_timescale: int. Minimum timescale. Defaults to `1`.
        max_timescale: int. Maximum timescale. Defaults to `10000`.
    """

    def __init__(
        self,
        embedding_dim,
        min_timescale=1,
        max_timescale=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def call(self, seq_length):
        num_timescales = self.embedding_dim // 2
        log_timescale_increment = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / max(num_timescales - 1, 1)

        inv_timescales = self.min_timescale * ops.exp(
            ops.cast(ops.arange(num_timescales), "float32")
            * (-log_timescale_increment)
        )
        # positions: (1, seq_length, 1)
        position = ops.cast(ops.arange(seq_length), "float32")[None, :, None]
        # inv_timescales: (1, 1, num_timescales)
        inv_timescales = inv_timescales[None, None, :]

        scaled_time = position * inv_timescales  # (1, seq, num_ts)
        signal = ops.concatenate(
            [ops.sin(scaled_time), ops.cos(scaled_time)], axis=-1
        )  # (1, seq, embedding_dim) or (1, seq, embedding_dim-1)

        # Pad if embedding_dim is odd.
        if self.embedding_dim % 2 != 0:
            signal = ops.pad(signal, [[0, 0], [0, 0], [0, 1]])

        return signal

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "min_timescale": self.min_timescale,
                "max_timescale": self.max_timescale,
            }
        )
        return config


class TIPSv2TextMLP(keras.layers.Layer):
    """Text encoder MLP with ReLU activation and padding mask support.

    The padding mask is applied after both the activation and the
    output projection, matching the HF reference behavior.

    Args:
        mlp_dim: int. Hidden dimension.
        d_model: int. Input/output dimension.
    """

    def __init__(self, mlp_dim, d_model, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.d_model = d_model

    def build(self, input_shape):
        self.c_fc = layers.Dense(
            self.mlp_dim,
            dtype=self.dtype_policy,
            name="c_fc",
        )
        self.c_fc.build(input_shape)
        self.c_proj = layers.Dense(
            self.d_model,
            dtype=self.dtype_policy,
            name="c_proj",
        )
        self.c_proj.build((*input_shape[:-1], self.mlp_dim))
        self.built = True

    def call(self, inputs, mask):
        """Apply MLP with masking.

        Args:
            inputs: Tensor of shape (B, seq_len, D).
            mask: Tensor of shape (B, seq_len) with 1 for valid, 0 for
                padding.
        """
        x = self.c_fc(inputs)
        x = ops.relu(x)
        x = x * mask[..., None]  # First masking.
        x = self.c_proj(x)
        x = x * mask[..., None]  # Second masking.
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mlp_dim": self.mlp_dim,
                "d_model": self.d_model,
            }
        )
        return config


class TIPSv2TextAttention(keras.layers.Layer):
    """Multi-head attention for the TIPSv2 text encoder.

    Uses a fused in_proj (Q/K/V concatenated) and applies a padding mask.
    The mask converts padding positions to additive -inf.

    Args:
        d_model: int. Model dimension.
        num_heads: int. Number of attention heads.
    """

    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

    def build(self, input_shape):
        # Fused in_proj for Q, K, V.
        self.in_proj = layers.Dense(
            self.d_model * 3,
            dtype=self.dtype_policy,
            name="in_proj",
        )
        self.in_proj.build(input_shape)
        self.out_proj = layers.Dense(
            self.d_model,
            dtype=self.dtype_policy,
            name="out_proj",
        )
        self.out_proj.build(input_shape)
        self.built = True

    def call(self, x, mask):
        """Apply masked multi-head attention.

        Args:
            x: Tensor of shape (B, seq_len, D).
            mask: Tensor of shape (B, seq_len) with 1 for valid, 0 for
                padding.
        """
        shape = ops.shape(x)
        b, seq_len = shape[0], shape[1]

        qkv = self.in_proj(x)  # (B, seq, 3*D)
        qkv = ops.reshape(qkv, (b, seq_len, 3, self.num_heads, self.head_dim))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, S, Dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # (B,H,S,S)
        attn = attn * self.scale

        # Build additive attention mask from padding mask.
        # mask: (B, seq) → attn_mask: (B, 1, 1, seq)
        attn_mask = mask[:, None, None, :]
        attn_mask = ops.where(
            attn_mask > 0,
            ops.zeros_like(attn),
            ops.full_like(attn, float("-inf")),
        )
        attn = attn + attn_mask

        attn = ops.softmax(attn, axis=-1)
        x = ops.matmul(attn, v)  # (B, H, S, Dh)
        x = ops.transpose(x, (0, 2, 1, 3))  # (B, S, H, Dh)
        x = ops.reshape(x, (b, seq_len, self.d_model))

        x = self.out_proj(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
            }
        )
        return config


class TIPSv2TextBlock(keras.layers.Layer):
    """Transformer block for the TIPSv2 text encoder.

    Pre-norm architecture with masked attention and masked MLP.

    Args:
        d_model: int. Model dimension.
        num_heads: int. Number of attention heads.
        mlp_dim: int. MLP hidden dimension.
    """

    def __init__(self, d_model, num_heads, mlp_dim, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

    def build(self, input_shape):
        self.ln_1 = layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy, name="ln_1"
        )
        self.ln_1.build(input_shape)

        self.attn = TIPSv2TextAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dtype=self.dtype_policy,
            name="attn",
        )
        self.attn.build(input_shape)

        self.ln_2 = layers.LayerNormalization(
            epsilon=1e-5, dtype=self.dtype_policy, name="ln_2"
        )
        self.ln_2.build(input_shape)

        self.mlp = TIPSv2TextMLP(
            mlp_dim=self.mlp_dim,
            d_model=self.d_model,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(input_shape)
        self.built = True

    def call(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x), mask)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
            }
        )
        return config
