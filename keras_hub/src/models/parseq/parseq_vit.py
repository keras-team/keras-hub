import keras
import numpy as np
from keras import layers
from keras import ops

LAYERNORM_EPSILON = 1e-5


class DropPath(layers.Layer):
    """
    Drop paths (Stochastic Depth) per sample.
    """

    def __init__(self, drop_prob=0.0, seed=42, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(seed=seed)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if (not training) or self.drop_prob == 0.0:
            return x

        # Compute drop
        keep_prob = 1.0 - self.drop_prob
        batch_size = ops.shape(x)[0]
        rank = len(ops.shape(x))

        # We'll broadcast along spatial / sequence dimensions
        if rank == 2:
            # (B, C)
            shape = [batch_size, 1]
        elif rank == 3:
            # (B, N, C)
            shape = [batch_size, 1, 1]
        else:
            # E.g. (B, H, W, C)
            shape = [batch_size, 1, 1, 1]

        random_tensor = keep_prob + keras.random.uniform(
            shape, seed=self.seed_generator, dtype=x.dtype
        )
        binary_tensor = ops.floor(random_tensor)
        output = x / keep_prob * binary_tensor
        return output


class Mlp(layers.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features=None,
        out_features=None,
        act="gelu",
        drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else in_features
        )

        self.fc1 = layers.Dense(hidden_features, activation=act)
        self.drop1 = layers.Dropout(drop)
        self.fc2 = layers.Dense(out_features)
        self.drop2 = layers.Dropout(drop)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


def to_2tuple(x):
    """Helper function that replicates an integer into a tuple (x, x)."""
    return (x, x) if isinstance(x, int) else x


class PatchEmbed(layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, **kwargs
    ):
        super().__init__(**kwargs)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.num_patches = num_patches

        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def call(self, x):
        """
        x: (B, H, W, C) with H==self.img_size[0], W==self.img_size[1]
        """
        # Check image size
        input_shape = ops.shape(x)
        H, W = input_shape[1], input_shape[2]

        if H is not None and H != self.img_size[0]:
            raise ValueError("Input height must match model.")
        if W is not None and W != self.img_size[1]:
            raise ValueError("Input width must match model.")

        # 1) conv -> shape (B, H//p, W//p, embed_dim)
        x = self.proj(x)

        # 2) flatten spatial dims -> (B, (H//p)*(W//p), embed_dim)
        B, Hp, Wp, C = ops.unstack(ops.shape(x))
        x = ops.reshape(x, (B, Hp * Wp, C))

        return x


class Attention(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=attn_drop,
            use_bias=qkv_bias,
        )
        self.proj_drop = layers.Dropout(proj_drop)

    def call(self, x, training=False):
        # Apply MultiHeadAttention
        x = self.mha(x, x)  # Self-attention (query, key, value are the same)
        # Projection and Dropout
        x = self.proj_drop(x, training=training)
        return x


class Block(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act="gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.norm1 = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else (lambda x: x)
        )

        self.norm2 = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act=act,
            drop=drop,
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else (lambda x: x)
        )

    def call(self, x, training=False):
        # (B, N, C) -> add shortcut
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path1(x)
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path2(x)
        return x


class VisionTransformer(keras.Model):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        class_num=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Class token + position embedding
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, num_patches, embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.pos_drop = layers.Dropout(drop_rate)

        # Stochastic depth schedules
        dpr_values = np.linspace(0, drop_path_rate, depth)

        # Transformer Encoder Blocks
        self.blocks = []

        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_values[i],
            )
            self.blocks.append(block)

        self.norm = layers.LayerNormalization(epsilon=LAYERNORM_EPSILON)

        # Classifier head
        if class_num:
            self.head = layers.Dense(class_num)
        else:
            self.head = None

    def call(self, x, training=False):
        """
        x shape: (B, H, W, C), channels-last
        """

        # 1) Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 2) Add pos_embed
        x = x + self.pos_embed  # shape broadcast (1, num_patches, C)

        # 3) Dropout
        x = self.pos_drop(x, training=training)

        # 4) Pass through Transformer blocks
        for blk in self.blocks:
            x = blk(x, training=training)

        # 5) Normalize
        x = self.norm(x, training=training)

        # 6) Classification head
        if self.head is not None:
            x = self.head(
                x, training=training
            )  # (B, num_patches, class_num) if class_num>0
        return x
