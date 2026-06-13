import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm


@keras_hub_export("keras_hub.models.Gemma4UnifiedVisionEmbedder")
class Gemma4UnifiedVisionEmbedder(keras.Model):
    """Lightweight encoder-free vision embedder for Gemma4 Unified (12B).

    Projects pre-merged image patches directly into the language model's
    hidden space via LN → Dense → LN → pos_emb → LN →
    RMSNorm → Linear.

    Args:
        hidden_dim: int. Output embedding dimension (must match the text
            backbone's `hidden_dim`).
        model_patch_size: int. Spatial size of each merged model patch in
            pixels (e.g. 48 = 16 × 3 for teacher `patch_size=16` and
            `pooling_kernel_size=3`).
        mm_posemb_size: int. Number of entries in each axis of the learned
            factorized position embedding table.
        num_soft_tokens: int. Maximum number of soft (vision) tokens per
            image after patch merging.
        pooling_kernel_size: int. Kernel size used during teacher-patch
            merging. Stored for configuration only; the actual merge
            happens in the image converter.
        patch_size: int. Teacher patch size in pixels. Defaults to `16`.
        layer_norm_epsilon: float. Epsilon for LayerNorm layers.
            Defaults to `1e-6`.
        dtype: string or `keras.mixed_precision.DTypePolicy`. Compute dtype.

    Example:
    ```python
    import numpy as np

    embedder = keras_hub.models.Gemma4UnifiedVisionEmbedder(
        hidden_dim=3840,
        model_patch_size=48,
        mm_posemb_size=1120,
        num_soft_tokens=280,
        pooling_kernel_size=3,
        patch_size=16,
    )
    pixel_values = np.ones((1, 1, 280, 6912), dtype="float32")
    pixel_position_ids = np.zeros((1, 1, 280, 2), dtype="int32")
    output = embedder(
        {"pixel_values": pixel_values, "pixel_position_ids": pixel_position_ids}
    )
    # output.shape == (1, 1, 280, 3840)
    ```
    """

    def __init__(
        self,
        hidden_dim,
        model_patch_size,
        mm_posemb_size,
        num_soft_tokens,
        pooling_kernel_size=3,
        patch_size=16,
        layer_norm_epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        patch_dim = model_patch_size * model_patch_size * 3

        # === Functional Model ===
        pixel_values_input = keras.Input(
            shape=(None, None, patch_dim),
            name="pixel_values",
        )
        pixel_position_ids_input = keras.Input(
            shape=(None, None, 2),
            dtype="int32",
            name="pixel_position_ids",
        )

        # --- Patch projection: LN₁ → Dense → LN₂ ---
        patch_ln1 = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="patch_ln1",
        )
        patch_dense = keras.layers.Dense(
            hidden_dim,
            use_bias=True,
            dtype=dtype,
            name="patch_dense",
        )
        patch_ln2 = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="patch_ln2",
        )

        x = patch_ln1(pixel_values_input)
        x = patch_dense(x)
        x = patch_ln2(x)

        # Factorized 2-D positional embedding: separate X and Y tables.
        pos_embedding_x = keras.layers.Embedding(
            mm_posemb_size,
            hidden_dim,
            dtype=dtype,
            name="pos_embedding_x",
        )
        pos_embedding_y = keras.layers.Embedding(
            mm_posemb_size,
            hidden_dim,
            dtype=dtype,
            name="pos_embedding_y",
        )

        pos_x = pixel_position_ids_input[..., 0]  # (..., N)
        pos_y = pixel_position_ids_input[..., 1]  # (..., N)

        # Clamp -1 padding to 0 for safe lookup, then mask.
        pos_x_safe = ops.maximum(pos_x, 0)
        pos_y_safe = ops.maximum(pos_y, 0)

        pe_x = pos_embedding_x(pos_x_safe)  # (..., N, hidden_dim)
        pe_y = pos_embedding_y(pos_y_safe)  # (..., N, hidden_dim)

        # Mask out padding positions (where original pos was -1).
        valid_x = ops.cast(ops.expand_dims(pos_x != -1, axis=-1), pe_x.dtype)
        valid_y = ops.cast(ops.expand_dims(pos_y != -1, axis=-1), pe_y.dtype)
        pos_emb = pe_x * valid_x + pe_y * valid_y

        x = x + pos_emb

        # --- Post-position norm ---
        pos_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="pos_norm",
        )
        x = pos_norm(x)

        # Multimodal embedder: RMSNorm → Linear (maps vision to LM space).
        embedding_pre_projection_norm = Gemma4VNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="embedding_pre_projection_norm",
        )
        embedding_projection = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="embedding_projection",
        )
        x = embedding_pre_projection_norm(x)
        x = embedding_projection(x)

        outputs = x
        super().__init__(
            inputs={
                "pixel_values": pixel_values_input,
                "pixel_position_ids": pixel_position_ids_input,
            },
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.hidden_dim = hidden_dim
        self.model_patch_size = model_patch_size
        self.mm_posemb_size = mm_posemb_size
        self.num_soft_tokens = num_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.patch_size = patch_size
        self.layer_norm_epsilon = layer_norm_epsilon

    @property
    def num_vision_tokens_per_image(self):
        """Maximum number of vision soft tokens per image."""
        return self.num_soft_tokens

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "model_patch_size": self.model_patch_size,
                "mm_posemb_size": self.mm_posemb_size,
                "num_soft_tokens": self.num_soft_tokens,
                "pooling_kernel_size": self.pooling_kernel_size,
                "patch_size": self.patch_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
