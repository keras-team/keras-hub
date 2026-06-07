import math

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4AspectRatioResizing,
)
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function

# Large value used to push padding patches to the end during argsort.
_PADDING_SORT_SENTINEL = 1_000_000_000


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

    batch_size = patches.shape[0]
    L = patches.shape[1]
    D = patches.shape[2]
    patch_size = int(math.isqrt(D // 3))
    k = int(math.isqrt(L // length))

    if batch_size == 0:
        merged_dim = k * patch_size * k * patch_size * 3
        return (
            np.empty((0, length, merged_dim), dtype=patches.dtype),
            np.empty((0, length, 2), dtype=np.int32),
        )

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

        # Ensure padding patches (where pos is -1) are sorted to the end
        is_padding = (pos[:, 0] == -1) & (pos[:, 1] == -1)
        target_ordering = np.where(
            is_padding, _PADDING_SORT_SENTINEL, target_ordering
        )

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

        # Compute merged positions. Mask out -1 padding entries so they
        # don't corrupt the min; restore -1 for fully-padding groups.
        kernel_pos = pos[perm]  # (L, 2)
        kernel_pos = kernel_pos.reshape(length, k * k, 2)
        is_pad = (kernel_pos == -1).all(axis=-1, keepdims=True)
        # Replace padding coords with large value for min, then divide.
        safe_pos = np.where(is_pad, np.iinfo(np.int32).max, kernel_pos)
        new_pos = (safe_pos // k).min(axis=1)
        # Restore -1 where all k² patches in a group were padding.
        all_pad = is_pad.all(axis=1)  # (length, 1)
        new_pos = np.where(all_pad, -1, new_pos).astype(np.int32)

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

    Example:
    ```python
    import numpy as np

    converter = keras_hub.layers.Gemma4UnifiedImageConverter(
        patch_size=16,
        max_soft_tokens=280,
        pooling_kernel_size=3,
    )
    images = np.random.rand(1, 768, 768, 3).astype("float32") * 255
    output = converter(images)
    # output["pixel_values"].shape == (1, 280, 6912)
    # output["pixel_position_ids"].shape == (1, 280, 2)
    ```
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
        # --- Resize ---
        if isinstance(inputs, dict):
            x = self.resizing.call(inputs["images"])
        else:
            x = self.resizing.call(inputs)

        # Apply scale/offset normalization.
        # Gemma4 Unified always expects pixels in [0, 1].  When no explicit
        # scale is configured (e.g. after a serialization round-trip that lost
        # the value), default to 1/255 rescaling so raw uint8 images work.
        if self.scale is not None:
            scale = self._expand_non_channel_dims(self.scale, x)
            x, scale = self._convert_types(x, scale, self.compute_dtype)
            x = x * scale
        else:
            if in_tf_function():
                import tensorflow as tf

                x = tf.cast(x, self.compute_dtype) / 255.0
            else:
                x = ops.cast(x, self.compute_dtype) / 255.0
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, x)
            x, offset = self._convert_types(x, offset, x.dtype)
            x = x + offset

        ps = self.patch_size
        k = self.pooling_kernel_size
        max_teacher_patches = self.max_soft_tokens * (k**2)

        if in_tf_function():
            import tensorflow as tf

            shape = tf.shape(x)
            batch_size = shape[0]
            h = shape[1]
            w = shape[2]
            n_h = h // ps
            n_w = w // ps

            # --- Patchify into teacher patches ---
            x = tf.reshape(x, (batch_size, n_h, ps, n_w, ps, 3))
            x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
            teacher_patches = tf.reshape(
                x, (batch_size, n_h * n_w, 3 * ps * ps)
            )

            # Build teacher-level position IDs: (x, y)
            col_ids = tf.range(n_w, dtype="int32")
            row_ids = tf.range(n_h, dtype="int32")
            col_grid = tf.tile(tf.reshape(col_ids, (1, n_w)), (n_h, 1))
            row_grid = tf.tile(tf.reshape(row_ids, (n_h, 1)), (1, n_w))
            positions = tf.stack(
                [tf.reshape(col_grid, [-1]), tf.reshape(row_grid, [-1])],
                axis=-1,
            )
            positions = tf.tile(
                tf.expand_dims(positions, 0), [batch_size, 1, 1]
            )

            # Pad teacher patches to max_teacher_patches
            current = n_h * n_w
            pad_len = tf.cast(max_teacher_patches, tf.int32) - current
            p_pad = tf.zeros(
                (batch_size, pad_len, 3 * ps * ps),
                dtype=teacher_patches.dtype,
            )
            teacher_patches = tf.concat([teacher_patches, p_pad], axis=1)
            teacher_patches = tf.ensure_shape(
                teacher_patches,
                [None, max_teacher_patches, 3 * ps * ps],
            )
            pos_pad = tf.fill((batch_size, pad_len, 2), -1)
            positions = tf.concat([positions, pos_pad], axis=1)
            positions = tf.ensure_shape(
                positions, [None, max_teacher_patches, 2]
            )

            # --- Merge teacher patches → model patches (TF path) ---
            merged_patches, merged_positions = self._patches_merge_tf(
                teacher_patches, positions, self.max_soft_tokens
            )
            pixel_values = tf.ensure_shape(
                merged_patches,
                [None, self.max_soft_tokens, (k * ps) ** 2 * 3],
            )
            pixel_position_ids = tf.ensure_shape(
                merged_positions, [None, self.max_soft_tokens, 2]
            )
        else:
            x_np = x if isinstance(x, np.ndarray) else ops.convert_to_numpy(x)
            batch_size = x_np.shape[0]
            h, w = x_np.shape[1], x_np.shape[2]
            n_h = h // ps
            n_w = w // ps

            # --- Patchify into teacher patches ---
            x_np = x_np.reshape(batch_size, n_h, ps, n_w, ps, 3)
            x_np = x_np.transpose(0, 1, 3, 2, 4, 5)
            teacher_patches = x_np.reshape(batch_size, n_h * n_w, 3 * ps * ps)

            # Build teacher-level position IDs: (x, y)
            col_ids = np.arange(n_w, dtype=np.int32)
            row_ids = np.arange(n_h, dtype=np.int32)
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
                teacher_patches = np.concatenate(
                    [teacher_patches, p_pad], axis=1
                )
                pos_pad = np.full(
                    (batch_size, pad_len, 2), -1, dtype=positions.dtype
                )
                positions = np.concatenate([positions, pos_pad], axis=1)

            # --- Merge teacher patches → model patches (NumPy path) ---
            merged_patches, merged_positions = _patches_merge_numpy(
                teacher_patches, positions, self.max_soft_tokens
            )

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

    def _patches_merge_tf(self, patches, positions_xy, length):
        """Merge k×k groups of teacher patches (TF graph-safe path).

        This reimplements `_patches_merge_numpy` using pure `tf.*` ops
        so it can execute inside `tf.data.Dataset.map` or `tf.function`.
        """
        import tensorflow as tf

        ps = self.patch_size
        k = self.pooling_kernel_size

        def _merge_single(args):
            p, pos = args  # (L, D), (L, 2)

            # Compute target ordering: group by kernel
            max_x = tf.reduce_max(pos[:, 0]) + 1
            kernel_x = pos[:, 0] // k
            kernel_y = pos[:, 1] // k

            num_from_tl = k * k * kernel_x + k * max_x * kernel_y
            within_kernel_x = pos[:, 0] % k
            within_kernel_y = pos[:, 1] % k
            num_from_tl_kernel = within_kernel_x + within_kernel_y * k
            target_ordering = num_from_tl_kernel + num_from_tl

            # Ensure padding patches are sorted to the end
            is_padding = tf.math.logical_and(
                tf.equal(pos[:, 0], -1), tf.equal(pos[:, 1], -1)
            )
            target_ordering = tf.where(
                is_padding,
                tf.cast(_PADDING_SORT_SENTINEL, target_ordering.dtype),
                target_ordering,
            )

            perm = tf.argsort(target_ordering)
            kernel_ordered = tf.gather(p, perm)

            # Reshape to merge
            kernel_ordered = tf.reshape(
                kernel_ordered, (length, k, k, ps, ps, 3)
            )
            kernel_ordered = tf.transpose(kernel_ordered, (0, 1, 3, 2, 4, 5))
            merged = tf.reshape(kernel_ordered, (length, k * ps * k * ps * 3))

            # Compute merged positions. Mask out -1 padding entries so
            # they don't corrupt the min; restore -1 for all-pad groups.
            kernel_pos = tf.gather(pos, perm)
            kernel_pos = tf.reshape(kernel_pos, (length, k * k, 2))
            is_pad = tf.reduce_all(
                tf.equal(kernel_pos, -1), axis=-1, keepdims=True
            )
            large_val = tf.constant(
                2147483647, dtype=kernel_pos.dtype
            )  # int32 max
            safe_pos = tf.where(is_pad, large_val, kernel_pos)
            new_pos = tf.reduce_min(safe_pos // k, axis=1)
            all_pad = tf.reduce_all(is_pad, axis=1)  # (length, 1)
            new_pos = tf.where(all_pad, -1, new_pos)
            new_pos = tf.cast(new_pos, tf.int32)

            return merged, new_pos

        merged, positions = tf.map_fn(
            _merge_single,
            (patches, positions_xy),
            fn_output_signature=(
                tf.TensorSpec((length, (k * ps) ** 2 * 3), patches.dtype),
                tf.TensorSpec((length, 2), tf.int32),
            ),
        )
        return merged, positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "max_soft_tokens": self.max_soft_tokens,
                "pooling_kernel_size": self.pooling_kernel_size,
            }
        )
        return config
