import math

import keras
from keras import ops
from keras import random


class OverlappingPatchingAndEmbedding(keras.layers.Layer):
    def __init__(self, project_dim=32, patch_size=7, stride=4, **kwargs):
        """Overlapping Patching and Embedding layer.

        Differs from `PatchingAndEmbedding` in that the patch size does not
        affect the sequence length. It's fully derived from the `stride`
        parameter. Additionally, no positional embedding is done
        as part of the layer - only a projection using a `Conv2D` layer.

        Args:
            project_dim: integer, the dimensionality of the projection.
                Defaults to `32`.
            patch_size: integer, the size of the patches to encode.
                Defaults to `7`.
            stride: integer, the stride to use for the patching before
                projection. Defaults to `5`.
        """
        super().__init__(**kwargs)

        self.project_dim = project_dim
        self.patch_size = patch_size
        self.stride = stride

        padding_size = self.patch_size // 2

        self.padding = keras.layers.ZeroPadding2D(
            padding=(padding_size, padding_size)
        )
        self.proj = keras.layers.Conv2D(
            filters=project_dim,
            kernel_size=patch_size,
            strides=stride,
            padding="valid",
        )
        self.norm = keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        x = self.padding(x)
        x = self.proj(x)
        x = ops.reshape(x, (-1, x.shape[1] * x.shape[2], x.shape[3]))
        x = self.norm(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "project_dim": self.project_dim,
                "patch_size": self.patch_size,
                "stride": self.stride,
            }
        )
        return config


class HierarchicalTransformerEncoder(keras.layers.Layer):
    """Hierarchical transformer encoder block implementation as a Keras Layer.

    The layer uses `SegFormerMultiheadAttention` as a `MultiHeadAttention`
    alternative for computational efficiency, and is meant to be used
    within the SegFormer architecture.

    Args:
        project_dim: integer, the dimensionality of the projection of the
            encoder, and output of the `SegFormerMultiheadAttention` layer.
            Due to the residual addition the input dimensionality has to be
            equal to the output dimensionality.
        num_heads: integer, the number of heads for the
            `SegFormerMultiheadAttention` layer.
        drop_prob: float, the probability of dropping a random
            sample using the `DropPath` layer. Defaults to `0.0`.
        layer_norm_epsilon: float, the epsilon for
            `LayerNormalization` layers. Defaults to `1e-06`
        sr_ratio: integer, the ratio to use within
            `SegFormerMultiheadAttention`. If set to > 1, a `Conv2D`
            layer is used to reduce the length of the sequence.
            Defaults to `1`.
    """

    def __init__(
        self,
        project_dim,
        num_heads,
        sr_ratio=1,
        drop_prob=0.0,
        layer_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.project_dim = project_dim
        self.num_heads = num_heads
        self.drop_prop = drop_prob

        self.norm1 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.attn = SegFormerMultiheadAttention(
            project_dim, num_heads, sr_ratio
        )
        self.drop_path = DropPath(drop_prob)
        self.norm2 = keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.mlp = MixFFN(
            channels=project_dim,
            mid_channels=int(project_dim * 4),
        )

    def build(self, input_shape):
        super().build(input_shape)
        self.H = ops.sqrt(ops.cast(input_shape[1], "float32"))
        self.W = ops.sqrt(ops.cast(input_shape[2], "float32"))

    def call(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mlp": keras.saving.serialize_keras_object(self.mlp),
                "project_dim": self.project_dim,
                "num_heads": self.num_heads,
                "drop_prop": self.drop_prop,
            }
        )
        return config


class MixFFN(keras.layers.Layer):
    def __init__(self, channels, mid_channels):
        super().__init__()
        self.fc1 = keras.layers.Dense(mid_channels)
        self.dwconv = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
        )
        self.fc2 = keras.layers.Dense(channels)

    def call(self, x):
        x = self.fc1(x)
        shape = ops.shape(x)
        H, W = int(math.sqrt(shape[1])), int(math.sqrt(shape[1]))
        B, C = shape[0], shape[2]
        x = ops.reshape(x, (B, H, W, C))
        x = self.dwconv(x)
        x = ops.reshape(x, (B, -1, C))
        x = ops.nn.gelu(x)
        x = self.fc2(x)
        return x


class SegFormerMultiheadAttention(keras.layers.Layer):
    def __init__(self, project_dim, num_heads, sr_ratio):
        """Efficient MultiHeadAttention implementation as a Keras layer.

        A huge bottleneck in scaling transformers is the self-attention layer
        with an O(n^2) complexity.

        SegFormerMultiheadAttention performs a sequence reduction (SR) operation
        with a given ratio, to reduce the sequence length before performing key
        and value projections, reducing the O(n^2) complexity to O(n^2/R) where
        R is the sequence reduction ratio.

        Args:
            project_dim: integer, the dimensionality of the projection
                of the `SegFormerMultiheadAttention` layer.
            num_heads: integer, the number of heads to use in the
                attention computation.
            sr_ratio: integer, the sequence reduction ratio to perform
                on the sequence before key and value projections.
        """
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.scale = (project_dim // num_heads) ** -0.5
        self.q = keras.layers.Dense(project_dim)
        self.k = keras.layers.Dense(project_dim)
        self.v = keras.layers.Dense(project_dim)
        self.proj = keras.layers.Dense(project_dim)
        self.dropout = keras.layers.Dropout(0.1)
        self.proj_drop = keras.layers.Dropout(0.1)

        if sr_ratio > 1:
            self.sr = keras.layers.Conv2D(
                filters=project_dim,
                kernel_size=sr_ratio,
                strides=sr_ratio,
            )
            self.norm = keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        input_shape = ops.shape(x)
        H, W = int(math.sqrt(input_shape[1])), int(math.sqrt(input_shape[1]))
        B, N, C = input_shape[0], input_shape[1], input_shape[2]

        q = self.q(x)
        q = ops.reshape(
            q,
            (
                input_shape[0],
                input_shape[1],
                self.num_heads,
                input_shape[2] // self.num_heads,
            ),
        )
        q = ops.transpose(q, [0, 2, 1, 3])

        if self.sr_ratio > 1:
            x = ops.reshape(
                x,
                (B, H, W, C),
            )
            x = self.sr(x)
            x = ops.reshape(x, [B, -1, C])
            x = self.norm(x)

        k = self.k(x)
        v = self.v(x)

        k = ops.transpose(
            ops.reshape(
                k,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        v = ops.transpose(
            ops.reshape(
                v,
                [B, -1, self.num_heads, C // self.num_heads],
            ),
            [0, 2, 1, 3],
        )

        attn = (q @ ops.transpose(k, [0, 1, 3, 2])) * self.scale
        attn = ops.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)

        attn = attn @ v
        attn = ops.reshape(
            ops.transpose(attn, [0, 2, 1, 3]),
            [B, N, C],
        )

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x


class DropPath(keras.layers.Layer):
    """Implements the DropPath layer.

    DropPath randomly drops samples during
    training with a probability of `rate`. Note that this layer drops individual
    samples within a batch and not the entire batch, whereas StochasticDepth
    randomly drops the entire batch.

    Args:
        rate: float, the probability of the residual branch being dropped.
        seed: (Optional) integer. Used to create a random seed.
    """

    def __init__(self, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
        self._seed_val = seed
        self.seed = random.SeedGenerator(seed=seed)

    def call(self, x, training=None):
        if self.rate == 0.0 or not training:
            return x
        else:
            batch_size = x.shape[0] or ops.shape(x)[0]
            drop_map_shape = (batch_size,) + (1,) * (len(x.shape) - 1)
            drop_map = ops.cast(
                random.uniform(drop_map_shape, seed=self.seed) > self.rate,
                x.dtype,
            )
            x = x / (1.0 - self.rate)
            x = x * drop_map
            return x

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self._seed_val})
        return config
