import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma4.gemma4_layers import Gemma4VNorm


@keras_hub_export("keras_hub.models.Gemma4UnifiedVisionEmbedder")
class Gemma4UnifiedVisionEmbedder(keras.Model):
    """Lightweight encoder-free vision embedder for Gemma4 Unified (12B).

    Unlike the tower-based `Gemma4VisionEncoder` used by 2B/4B/26B/31B
    models, this embedder projects pre-merged image patches directly into
    the language model's hidden space via a single linear projection and
    learned 2-D positional embeddings. There is no separate ViT encoder.

    The image preprocessing pipeline (resize → patchify → merge k×k teacher
    patches → pad) produces `pixel_values` of shape
    `(batch, num_images, max_soft_tokens, model_patch_size² × 3)` and
    `pixel_position_ids` of shape `(batch, num_images, max_soft_tokens, 2)`.

    This embedder then:
    1. Projects the flattened patch pixels via a dense layer to `hidden_dim`.
    2. Adds 2-D learned positional embeddings looked up from
       `pos_embedding_table` using the XY position IDs.
    3. Applies a parameter-free RMS norm (`Gemma4VNorm`).

    Args:
        hidden_dim: int. Output embedding dimension (must match the text
            backbone's `hidden_dim`).
        model_patch_size: int. Spatial size of each merged model patch in
            pixels (e.g. 48 = 16 × 3 for teacher `patch_size=16` and
            `pooling_kernel_size=3`).
        mm_posemb_size: int. Number of entries in the learned position
            embedding table. Should be ≥ max possible flattened XY index.
        num_soft_tokens: int. Maximum number of soft (vision) tokens per
            image after patch merging.
        pooling_kernel_size: int. Kernel size used during teacher-patch
            merging. Stored for configuration only; the actual merge happens
            in the image converter.
        patch_size: int. Teacher patch size in pixels. Stored for
            configuration only.
        layer_norm_epsilon: float. Epsilon for the post-projection norm.
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
        # Vision embedder always runs in float32 for consistency.
        if hasattr(dtype, "variable_dtype"):
            dtype = "float32"
        elif dtype is not None and dtype != "float32":
            dtype = "float32"

        input_dim = model_patch_size * model_patch_size * 3

        # === Functional Model ===
        pixel_values_input = keras.Input(
            shape=(None, None, input_dim),
            name="pixel_values",
        )
        pixel_position_ids_input = keras.Input(
            shape=(None, None, 2),
            dtype="int32",
            name="pixel_position_ids",
        )

        # Linear projection: (model_patch_size² × 3) → hidden_dim
        embedding_projection = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            dtype=dtype,
            name="embedding_projection",
        )
        x = embedding_projection(pixel_values_input)

        # 2-D positional embedding (XY → flat 1-D index → lookup).
        pos_embedding_table = keras.layers.Embedding(
            mm_posemb_size,
            hidden_dim,
            dtype=dtype,
            name="pos_embedding_table",
        )

        # Flatten XY → 1-D index. Position IDs of -1 indicate padding;
        # clamp to 0 and mask later.
        pos_x = pixel_position_ids_input[..., 0]  # (..., N)
        pos_y = pixel_position_ids_input[..., 1]  # (..., N)

        # Clamp -1 padding positions to 0.
        pos_x_safe = ops.maximum(pos_x, 0)
        pos_y_safe = ops.maximum(pos_y, 0)

        # Flat index: idx = x * num_patches_y + y (dynamic, matches HF).
        num_patches_y = ops.cast(
            ops.maximum(ops.max(pos_y_safe) + 1, 1), "int32"
        )
        flat_pos = pos_x_safe * num_patches_y + pos_y_safe

        pos_emb = pos_embedding_table(flat_pos)
        x = x + pos_emb

        # Post-projection norm (parameter-free RMSNorm / VNorm).
        post_norm = Gemma4VNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name="embedding_post_projection_norm",
        )
        x = post_norm(x)

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
