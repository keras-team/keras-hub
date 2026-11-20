"""TIPSv2 Text Encoder.

A transformer text encoder with sinusoidal positional embeddings,
ReLU-activated masked MLP, and masked global average pooling.
"""

import math

import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.tipsv2.tipsv2_layers import TIPSv2TextBlock


class TIPSv2SinusoidalPositionEmbedding(keras.layers.Layer):
    """Sinusoidal positional embedding (computed, not learned).

    Takes an input tensor and returns a position embedding tensor of the
    same shape, computed using sin/cos functions. This layer broadcasts
    the embedding to match the input's batch and sequence dimensions.

    Args:
        max_sequence_length: int. Maximum supported sequence length.
        embedding_dim: int. Dimension of the positional embedding.
        min_timescale: int. Minimum timescale. Defaults to `1`.
        max_timescale: int. Maximum timescale. Defaults to `10000`.
    """

    def __init__(
        self,
        max_sequence_length,
        embedding_dim,
        min_timescale=1,
        max_timescale=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def build(self, input_shape):
        # Pre-compute the full position embedding table at build time.
        num_timescales = self.embedding_dim // 2
        log_timescale_increment = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / max(num_timescales - 1, 1)

        inv_timescales = self.min_timescale * ops.exp(
            ops.cast(ops.arange(num_timescales), "float32")
            * (-log_timescale_increment)
        )
        position = ops.cast(ops.arange(self.max_sequence_length), "float32")[
            :, None
        ]
        inv_timescales = inv_timescales[None, :]

        scaled_time = position * inv_timescales  # (max_len, num_ts)
        signal = ops.concatenate(
            [ops.sin(scaled_time), ops.cos(scaled_time)], axis=-1
        )  # (max_len, embedding_dim)

        # Pad if embedding_dim is odd.
        if self.embedding_dim % 2 != 0:
            signal = ops.pad(signal, [[0, 0], [0, 1]])

        # Store as non-trainable weight for serialization.
        self.pos_table = self.add_weight(
            name="pos_table",
            shape=(self.max_sequence_length, self.embedding_dim),
            initializer="zeros",
            trainable=False,
        )
        self.pos_table.assign(signal)
        self.built = True

    def call(self, inputs):
        """Return position embeddings matching input sequence length.

        Args:
            inputs: Tensor of shape (B, seq_len, D).

        Returns:
            Tensor of shape (B, seq_len, D) with position embeddings.
        """
        seq_length = ops.shape(inputs)[-2]
        # Slice the pre-computed table to match seq_length.
        # Explicitly convert Variable to tensor for JAX tracing.
        pos_table = ops.convert_to_tensor(self.pos_table)
        pos_embed = ops.slice(
            pos_table, (0, 0), (seq_length, self.embedding_dim)
        )
        return ops.broadcast_to(pos_embed, ops.shape(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_sequence_length": self.max_sequence_length,
                "embedding_dim": self.embedding_dim,
                "min_timescale": self.min_timescale,
                "max_timescale": self.max_timescale,
            }
        )
        return config


@keras_hub_export("keras_hub.models.TIPSv2TextEncoder")
class TIPSv2TextEncoder(Backbone):
    """TIPSv2 text encoder.

    A transformer encoder for text that uses sinusoidal positional embeddings,
    masked multi-head attention, ReLU-activated masked MLP, and global
    average pooling over non-padding tokens.

    The default constructor gives a fully customizable, randomly initialized
    model. To load preset architectures and weights, use `from_preset`.

    Args:
        vocabulary_size: int. Size of the token vocabulary.
        embedding_dim: int. Token embedding dimension.
        hidden_dim: int. Transformer hidden dimension.
        num_layers: int. Number of transformer blocks.
        num_heads: int. Number of attention heads.
        intermediate_dim: int. MLP hidden dimension.
        max_sequence_length: int. Maximum input sequence length.
            Defaults to `64`.
        scale_sqrt_depth: bool. Whether to scale embeddings by
            sqrt(embedding_dim). Defaults to `True`.
        dtype: str or Policy. Dtype for computations. Defaults to `None`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    encoder = keras_hub.models.TIPSv2TextEncoder(
        vocabulary_size=32000,
        embedding_dim=768,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
    )
    token_ids = np.ones((1, 64), dtype="int32")
    padding_mask = np.ones((1, 64), dtype="int32")
    output = encoder({"token_ids": token_ids, "padding_mask": padding_mask})
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dim,
        hidden_dim,
        num_layers,
        num_heads,
        intermediate_dim,
        max_sequence_length=64,
        scale_sqrt_depth=True,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            dtype=dtype,
            name="token_embedding",
        )
        self.pos_embedding = TIPSv2SinusoidalPositionEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            dtype=dtype,
            name="pos_embedding",
        )
        self.text_blocks = []
        for i in range(num_layers):
            block = TIPSv2TextBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                mlp_dim=intermediate_dim,
                dtype=dtype,
                name=f"block_{i}",
            )
            self.text_blocks.append(block)
        self.ln_final = layers.LayerNormalization(
            epsilon=1e-5, dtype=dtype, name="ln_final"
        )

        # === Functional Model ===
        token_id_input = layers.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = layers.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Token embedding + scaling.
        x = self.token_embedding(token_id_input)
        if scale_sqrt_depth:
            x = layers.Rescaling(
                scale=embedding_dim**0.5,
                dtype=dtype,
                name="embedding_scale",
            )(x)

        # Add sinusoidal positional embeddings.
        pos_emb = self.pos_embedding(x)
        x = x + pos_emb

        # Build attention mask: 1 for valid, 0 for padding.
        mask = ops.cast(padding_mask_input, dtype=x.dtype)

        # Transformer blocks with mask.
        for block in self.text_blocks:
            x = block(x, mask)

        # Final layer norm.
        x = self.ln_final(x)

        # Masked global average pooling.
        # Zero out padding positions, sum, divide by valid count.
        x = x * mask[..., None]
        valid_count = ops.sum(mask, axis=1, keepdims=True) + 1e-8
        x = ops.sum(x, axis=1) / valid_count  # (B, D)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.max_sequence_length = max_sequence_length
        self.scale_sqrt_depth = scale_sqrt_depth

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "embedding_dim": self.embedding_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "max_sequence_length": self.max_sequence_length,
                "scale_sqrt_depth": self.scale_sqrt_depth,
            }
        )
        return config
