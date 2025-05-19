import keras
from keras import ops
import numpy as np


class SmolVLMVisionEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        image_size,
        patch_size,
        hidden_size,
        num_channels=3,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = hidden_size
        self.patch_size = patch_size
        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side**2

        self.patch_embedding = keras.layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            name="patch_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            self.num_patches, self.embed_dim, name="position_embedding"
        )

    def compute_position_ids(self, patch_attention_mask):
        batch_size = ops.shape(patch_attention_mask)[0]
        max_nb_patches_h, max_nb_patches_w = (
            patch_attention_mask.shape[1],
            patch_attention_mask.shape[2],
        )
        max_patches = max_nb_patches_h * max_nb_patches_w

        def compute_per_image(p_mask):
            nb_patches_h = ops.sum(ops.any(p_mask, axis=1), dtype="int32")
            nb_patches_w = ops.sum(ops.any(p_mask, axis=0), dtype="int32")
            frac_h = ops.linspace(0.0, 1.0 - 1e-6, nb_patches_h)
            frac_w = ops.linspace(0.0, 1.0 - 1e-6, nb_patches_w)
            boundaries = ops.linspace(0.0, 1.0, self.num_patches_per_side + 1)[
                1:-1
            ]
            bucket_h = ops.searchsorted(boundaries, frac_h, side="right")
            bucket_w = ops.searchsorted(boundaries, frac_w, side="right")
            rows, cols = ops.where(p_mask)
            pos_ids = (
                bucket_h[rows] * self.num_patches_per_side + bucket_w[cols]
            )
            indices = rows * max_nb_patches_w + cols
            pos_ids_full = ops.scatter(
                indices=ops.expand_dims(indices, axis=1),
                values=pos_ids,
                shape=[max_patches],
            )
            return pos_ids_full

        position_ids = ops.map(
            compute_per_image, xs=patch_attention_mask, dtype="int32"
        )
        return position_ids

    def call(self, inputs):
        pixel_values, patch_attention_mask = inputs
        batch_size = ops.shape(pixel_values)[0]
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = ops.reshape(patch_embeds, [batch_size, -1, self.embed_dim])
        position_ids = self.compute_position_ids(patch_attention_mask)
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings


class SmolVLMVisionAttention(keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.q_proj = keras.layers.Dense(self.embed_dim, name="q_proj")
        self.k_proj = keras.layers.Dense(self.embed_dim, name="k_proj")
        self.v_proj = keras.layers.Dense(self.embed_dim, name="v_proj")
        self.out_proj = keras.layers.Dense(self.embed_dim, name="out_proj")

    def call(self, hidden_states, attention_mask=None, training=False):
        batch_size, seq_length = (
            ops.shape(hidden_states)[0],
            ops.shape(hidden_states)[1],
        )

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = ops.reshape(
            queries, [batch_size, seq_length, self.num_heads, self.head_dim]
        )
        keys = ops.reshape(
            keys, [batch_size, seq_length, self.num_heads, self.head_dim]
        )
        values = ops.reshape(
            values, [batch_size, seq_length, self.num_heads, self.head_dim]
        )

        queries = ops.transpose(
            queries, [0, 2, 1, 3]
        )  # [batch, heads, seq, head_dim]
        keys = ops.transpose(keys, [0, 2, 1, 3])
        values = ops.transpose(values, [0, 2, 1, 3])

        attn_weights = (
            ops.matmul(queries, ops.transpose(keys, [0, 1, 3, 2])) * self.scale
        )
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)
        if training:
            attn_weights = ops.dropout(attn_weights, self.dropout)

        attn_output = ops.matmul(attn_weights, values)
        attn_output = ops.transpose(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(
            attn_output, [batch_size, seq_length, self.embed_dim]
        )
        return self.out_proj(attn_output)


class SmolVLMVisionMLP(keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act="gelu",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.fc1 = keras.layers.Dense(intermediate_size, name="fc1")
        self.activation_fn = keras.layers.Activation(
            hidden_act, name="activation"
        )
        self.fc2 = keras.layers.Dense(hidden_size, name="fc2")

    def call(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


# Your updated Encoder Layer
class SmolVLMEncoderLayer(keras.keras.layers.Layer):
    """Single transformer encoder layer with attention and MLP."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        attention_dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps

        self.self_attn = SmolVLMVisionAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            dtype=dtype,
            name="self_attn",
        )
        self.layer_norm1 = keras.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, dtype=dtype, name="layer_norm1"
        )
        self.mlp = SmolVLMVisionMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            dtype=dtype,
            name="mlp",
        )
        self.layer_norm2 = keras.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, dtype=dtype, name="layer_norm2"
        )

    def call(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "intermediate_size": self.intermediate_size,
                "attention_dropout": self.attention_dropout,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
            }
        )
        return config


# Updated Vision Encoder with integrated image preprocessing
class SmolVLMVisionEncoder(keras.Model):
    """Complete SmolVLM vision encoder with integrated image preprocessing."""

    def __init__(
        self,
        image_size,
        patch_size,
        hidden_size,
        num_attention_heads,
        num_hidden_layers,
        intermediate_size,
        text_hidden_size,  # For connector projection
        scale_factor=2,  # For connector pixel shuffle
        attention_dropout=0.0,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        num_channels=3,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.num_channels = num_channels
        self.text_hidden_size = text_hidden_size

        self.embeddings = SmolVLMVisionEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_channels=num_channels,
            dtype=dtype,
            name="embeddings",
        )
        self.layers = [
            SmolVLMEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                attention_dropout=attention_dropout,
                hidden_act=hidden_act,
                layer_norm_eps=layer_norm_eps,
                dtype=dtype,
                name=f"layer_{i}",
            )
            for i in range(num_hidden_layers)
        ]
        self.post_layernorm = keras.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, dtype=dtype, name="post_layernorm"
        )

        # Connector projection layer
        self.connector_proj = keras.layers.Dense(
            text_hidden_size, use_bias=False, name="connector_proj", dtype=dtype
        )

        # Handle dtype policy
        if hasattr(keras.dtype_policies, "get"):
            self.dtype_policy = keras.dtype_policies.get(dtype)
        else:
            dtype = dtype or keras.config.dtype_policy().name
            self.dtype_policy = keras.dtype_policies.DTypePolicy(dtype)

    def preprocess_images(self, pixel_values, pixel_attention_mask):
        """Preprocess pixel_values and pixel_attention_mask as in PyTorch implementation."""
        batch_size, num_images = (
            ops.shape(pixel_values)[0],
            ops.shape(pixel_values)[1],
        )
        # Reshape: (batch_size, num_images, C, H, W) -> (batch_size * num_images, C, H, W)
        pixel_values = ops.reshape(pixel_values, [-1, *pixel_values.shape[2:]])

        # Remove padding images (full zeros)
        nb_values_per_image = np.prod(pixel_values.shape[1:])
        real_images_mask = ops.sum(ops.abs(pixel_values), axis=[1, 2, 3]) != 0
        # If no real images, keep one (first) image
        real_images_mask = ops.cond(
            pred=ops.any(real_images_mask),
            true_fn=lambda: real_images_mask,
            false_fn=lambda: ops.scatter(
                indices=[[0]],
                values=[True],
                shape=[ops.shape(real_images_mask)[0]],
            ),
        )
        pixel_values = pixel_values[real_images_mask]

        # Handle pixel_attention_mask
        if pixel_attention_mask is None:
            pixel_attention_mask = ops.ones(
                [
                    ops.shape(pixel_values)[0],
                    pixel_values.shape[2],
                    pixel_values.shape[3],
                ],
                dtype="bool",
            )
        else:
            pixel_attention_mask = ops.reshape(
                pixel_attention_mask, [-1, *pixel_attention_mask.shape[2:]]
            )
            pixel_attention_mask = pixel_attention_mask[real_images_mask]

        # Compute patch attention mask
        patches_h = ops.slice(
            pixel_attention_mask,
            start_indices=[0, 0, 0],
            shape=[-1, -1, pixel_attention_mask.shape[2], 1],
        )
        patches_h = ops.unstack(patches_h, axis=2)
        patches_w = [
            ops.slice(
                h,
                start_indices=[0, 0, 0],
                shape=[-1, h.shape[1], -1, self.patch_size],
            )
            for h in patches_h
        ]
        patches_w = ops.stack(patches_w, axis=3)
        patch_attention_mask = ops.sum(patches_w, axis=[-1, -2]) > 0

        return pixel_values, patch_attention_mask

    def call(self, pixel_values, pixel_attention_mask):
        # Preprocess images and attention mask
        pixel_values, patch_attention_mask = self.preprocess_images(
            pixel_values, pixel_attention_mask
        )

        # Embeddings
        hidden_states = self.embeddings([pixel_values, patch_attention_mask])

        # Prepare attention mask for encoder
        seq_len = ops.shape(hidden_states)[1]
        attention_mask = ops.reshape(
            patch_attention_mask, [-1, seq_len]
        )  # (batch_size, seq_len)
        attention_mask = ops.where(
            attention_mask, 0.0, -1e9
        )  # 0 for valid, -inf for padded
        attention_mask = ops.expand_dims(
            attention_mask, axis=1
        )  # (batch_size, 1, seq_len)
        attention_mask = ops.expand_dims(
            attention_mask, axis=1
        )  # (batch_size, 1, 1, seq_len)

        # Encoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final normalization
        hidden_states = self.post_layernorm(hidden_states)

        # Connector logic: pixel shuffle and projection
        hidden_states = self.pixel_shuffle(hidden_states)
        hidden_states = self.connector_proj(hidden_states)

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "patch_size": self.patch_size,
                "hidden_size": self.hidden_size,
                "num_attention_heads": self.num_attention_heads,
                "num_hidden_layers": self.num_hidden_layers,
                "intermediate_size": self.intermediate_size,
                "attention_dropout": self.attention_dropout,
                "hidden_act": self.hidden_act,
                "layer_norm_eps": self.layer_norm_eps,
                "num_channels": self.num_channels,
            }
        )
        return config
