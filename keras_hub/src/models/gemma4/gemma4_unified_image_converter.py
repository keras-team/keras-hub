import math

from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4AspectRatioResizing,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


def _patches_merge_numpy(patches, positions_xy, length):
    """Merge k×k groups of teacher patches into model patches (NumPy path).

    Given `L` input patches of dimension `D = patch_size² × 3`, merge groups
    of `k×k` spatially adjacent patches into `length` output patches of
    dimension `(k × patch_size)² × 3`.

    Args:
        patches: (batch, L, D)
        positions_xy: (batch, L, 2)  — integer XY positions (-1 for padding)
        length: target number of output patches (= max_soft_tokens)

    Returns:
        merged_patches: (batch, length, k²×D)
        merged_positions: (batch, length, 2)
    """
    import numpy as np

    batch_size = patches.shape[0]
    L = patches.shape[1]
    D = patches.shape[2]
    patch_size = int(math.isqrt(D // 3))
    k = int(math.isqrt(L // length))

    all_merged_patches = []
    all_merged_positions = []
    for b in range(batch_size):
        p = patches[b]  # (L, D)
        pos = positions_xy[b]  # (L, 2)

        # Compute target ordering: group by kernel
        max_x = pos[:, 0].max() + 1
        kernel_x = pos[:, 0] // k
        kernel_y = pos[:, 1] // k

        num_from_tl = k * k * kernel_x + k * max_x * kernel_y
        within_kernel_x = pos[:, 0] % k
        within_kernel_y = pos[:, 1] % k
        num_from_tl_kernel = within_kernel_x + within_kernel_y * k
        target_ordering = num_from_tl_kernel + num_from_tl

        perm = np.argsort(target_ordering)
        kernel_ordered = p[perm]  # (L, D)

        # Reshape to merge
        kernel_ordered = kernel_ordered.reshape(
            length, k, k, patch_size, patch_size, 3
        )
        # Rearrange: (l, ky, kx, py, px, c) → (l, ky*py, kx*px, c)
        kernel_ordered = kernel_ordered.transpose(0, 1, 3, 2, 4, 5)
        merged = kernel_ordered.reshape(
            length, k * patch_size * k * patch_size * 3
        )

        # Compute merged positions
        kernel_pos = pos[perm]  # (L, 2)
        kernel_pos = kernel_pos.reshape(length, k * k, 2).astype(np.float64)

        # Compute merged position: divide by k to get model-level grid coords.
        # For non-padded patches, compute new positions
        new_pos = kernel_pos // k
        # Take min valid position per kernel
        new_pos = new_pos.min(axis=1).astype(np.int32)

        all_merged_patches.append(merged)
        all_merged_positions.append(new_pos)

    return np.stack(all_merged_patches), np.stack(all_merged_positions)


@keras_hub_export("keras_hub.layers.Gemma4UnifiedImageConverter")
class Gemma4UnifiedImageConverter(ImageConverter):
    """Preprocess raw images for Gemma4 Unified (12B) vision inputs.

    Unlike `Gemma4ImageConverter` which produces teacher-level patches,
    this converter additionally **merges** groups of `pooling_kernel_size²`
    teacher patches into larger model patches, matching the encoder-free
    architecture of the Gemma4 Unified 12B model.

    Pipeline:
    1. Aspect-ratio-preserving resize to the nearest valid resolution
    2. Patchify into `patch_size`-px teacher patches
    3. **Merge** k×k teacher patches into model patches
    4. Pad merged patches to `max_soft_tokens`

    Output:
    - `pixel_values`: `(batch, max_soft_tokens, model_patch_size²×3)`
    - `pixel_position_ids`: `(batch, max_soft_tokens, 2)`

    Args:
        patch_size: int. Teacher patch size in pixels. Defaults to `16`.
        max_soft_tokens: int. Maximum merged soft tokens per image. Defaults
            to `280`.
        pooling_kernel_size: int. Number of teacher patches merged along each
            spatial axis. Defaults to `3`.
        **kwargs: Additional keyword arguments forwarded to
            `keras_hub.layers.ImageConverter`.
    """

    backbone_cls = Gemma4Backbone

    def __init__(
        self,
        patch_size=16,
        max_soft_tokens=280,
        pooling_kernel_size=3,
        **kwargs,
    ):
        # Always do image preprocessing in float32.
        kwargs.pop("dtype", None)
        dtype = "float32"
        super().__init__(dtype=dtype, **kwargs)
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.model_patch_size = patch_size * pooling_kernel_size

        # Reuse the same aspect-ratio resizing logic as the standard
        # Gemma4ImageConverter.
        self.resizing = Gemma4AspectRatioResizing(
            patch_size=self.patch_size,
            max_soft_tokens=self.max_soft_tokens,
            pooling_kernel_size=self.pooling_kernel_size,
        )

    @preprocessing_function
    def call(self, inputs):
        import numpy as np

        # --- Resize ---
        if isinstance(inputs, dict):
            x = self.resizing.call(inputs["images"])
        else:
            x = self.resizing.call(inputs)

        # Apply scale/offset normalization.
        if self.scale is not None:
            scale = self._expand_non_channel_dims(self.scale, x)
            x, scale = self._convert_types(x, scale, self.compute_dtype)
            x = x * scale
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, x)
            x, offset = self._convert_types(x, offset, x.dtype)
            x = x + offset

        # --- Patchify into teacher patches ---
        ps = self.patch_size
        k = self.pooling_kernel_size
        max_teacher_patches = self.max_soft_tokens * (k**2)

        # Convert to numpy for the merge logic (preprocessing runs in eager).
        x_np = np.array(x) if not isinstance(x, np.ndarray) else x
        batch_size = x_np.shape[0]
        h, w = x_np.shape[1], x_np.shape[2]
        n_h = h // ps
        n_w = w // ps

        # (batch, n_h, ps, n_w, ps, 3) → (batch, n_h*n_w, ps*ps*3)
        x_np = x_np.reshape(batch_size, n_h, ps, n_w, ps, 3)
        x_np = x_np.transpose(0, 1, 3, 2, 4, 5)
        teacher_patches = x_np.reshape(batch_size, n_h * n_w, 3 * ps * ps)

        # Build teacher-level position IDs: (x, y) for each patch
        col_ids = np.arange(n_w, dtype=np.int32)  # x
        row_ids = np.arange(n_h, dtype=np.int32)  # y
        col_grid = np.tile(col_ids.reshape(1, n_w), (n_h, 1))
        row_grid = np.tile(row_ids.reshape(n_h, 1), (1, n_w))
        positions = np.stack(
            [col_grid.reshape(-1), row_grid.reshape(-1)], axis=-1
        )
        positions = np.tile(positions[np.newaxis], (batch_size, 1, 1))

        # Pad teacher patches to max_teacher_patches
        current = n_h * n_w
        pad_len = max_teacher_patches - current
        if pad_len > 0:
            p_pad = np.zeros(
                (batch_size, pad_len, 3 * ps * ps),
                dtype=teacher_patches.dtype,
            )
            teacher_patches = np.concatenate([teacher_patches, p_pad], axis=1)
            pos_pad = np.full(
                (batch_size, pad_len, 2), -1, dtype=positions.dtype
            )
            positions = np.concatenate([positions, pos_pad], axis=1)

        # --- Merge teacher patches → model patches ---
        merged_patches, merged_positions = _patches_merge_numpy(
            teacher_patches, positions, self.max_soft_tokens
        )

        # Convert back to framework tensors
        pixel_values = ops.convert_to_tensor(
            merged_patches, dtype=self.compute_dtype
        )
        pixel_position_ids = ops.convert_to_tensor(
            merged_positions, dtype="int32"
        )

        outputs = {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }

        if isinstance(inputs, dict):
            inputs.update(outputs)
            return inputs
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
