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
        name="mlp",
        **kwargs,
    ):
        super().__init__(**kwargs, name=name)
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else in_features
        )

        self.fc1 = layers.Dense(
            hidden_features, activation=act, name=f"{name}_dense1"
        )
        self.drop1 = layers.Dropout(drop, name=f"{name}_dropout1")
        self.fc2 = layers.Dense(out_features, name=f"{name}_dense2")
        self.drop2 = layers.Dropout(drop, name=f"{name}_dropout2")

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class PatchEmbed(layers.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        name="patchembed",
        **kwargs,
    ):
        super().__init__(**kwargs, name=name)
        patch_size = (
            (patch_size, patch_size)
            if isinstance(patch_size, int)
            else patch_size
        )
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
            name=f"{name}_conv",
        )

    def call(self, x):
        """
        x: (B, H, W, C) with H==self.img_size[0], W==self.img_size[1]
        """
        # Check image size
        input_shape = ops.shape(x)
        H, W = input_shape[1], input_shape[2]

        if H is not None and H % self.patch_size[0] != 0:
            raise ValueError("Input height must fit `patch_size`.")
        if W is not None and W % self.patch_size[1] != 0:
            raise ValueError("Input width must fit `patch_size`.")

        # 1) conv -> shape (B, H//p, W//p, embed_dim)
        x = self.proj(x)

        # 2) flatten spatial dims -> (B, (H//p)*(W//p), embed_dim)
        B, Hp, Wp, C = ops.shape(x)
        x = ops.reshape(x, (B, Hp * Wp, C))

        return x


class Attention(layers.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        name="attention",
        **kwargs,
    ):
        super().__init__(**kwargs, name=name)
        self.num_heads = num_heads
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim // num_heads,
            dropout=attn_drop,
            use_bias=qkv_bias,
            name=f"{name}_mha",
        )
        self.proj_drop = layers.Dropout(proj_drop, name=f"{name}_dropout")

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
        mlp_dim,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act="gelu",
        name="block",
        **kwargs,
    ):
        super().__init__(**kwargs, name=name)

        self.norm1 = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_norm1"
        )
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name=f"{name}_attn",
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else (lambda x: x)
        )

        self.norm2 = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_norm2"
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_dim,
            act=act,
            drop=drop,
            name=f"{name}_mlp",
        )

    def call(self, x, training=False):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(keras.Model):
    def __init__(
        self,
        patch_size=16,
        class_num=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        name="vit",
        **kwargs,
    ):
        super().__init__(**kwargs, name=name)
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            embed_dim=embed_dim,
            name=f"{name}_embed",
        )
        self.pos_drop = layers.Dropout(drop_rate, name=f"{name}_dropout")

        # Stochastic depth schedules
        dpr_values = np.linspace(0, drop_path_rate, depth)

        # Transformer Encoder Blocks
        self.blocks = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_values[i],
                name=f"{name}_block{i}",
            )
            for i in range(depth)
        ]

        self.norm = layers.LayerNormalization(
            epsilon=LAYERNORM_EPSILON, name=f"{name}_norm"
        )

        # Classifier head
        if class_num:
            self.head = layers.Dense(class_num, name=f"{name}_head")
        else:
            self.head = None

    def build(self, input_shape):
        # Add class token + positional embeddings
        height, width = input_shape[1:3]
        if height is not None and height % self.patch_size[0] != 0:
            raise ValueError("Input height must fit `patch_size`.")
        if width is not None and width % self.patch_size[1] != 0:
            raise ValueError("Input width must fit `patch_size`.")
        num_patches = (height // self.patch_size[0]) * (
            width // self.patch_size[1]
        )
        self.cls_token = self.add_weight(
            name=f"{self.name}_cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )
        self.pos_embed = self.add_weight(
            name=f"{self.name}_pos_embed",
            shape=(1, num_patches, self.embed_dim),
            initializer="zeros",
            trainable=True,
        )

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
