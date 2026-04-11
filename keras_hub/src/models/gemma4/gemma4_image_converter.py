import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


class Gemma4AspectRatioResizing(keras.layers.Layer):
    def __init__(
        self, patch_size, max_soft_tokens, pooling_kernel_size, **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.height = 1  # Dummy value to satisfy ImageConverter checks
        self.width = 1  # Dummy value to satisfy ImageConverter checks

    def call(self, inputs):
        # inputs: (B, H, W, 3) where H, W can be dynamic Tensors or static
        from keras_hub.src.utils.tensor_utils import in_tf_function

        side_mult = self.pooling_kernel_size * self.patch_size
        max_patches = self.max_soft_tokens * (self.pooling_kernel_size**2)
        target_px = max_patches * (self.patch_size**2)

        if in_tf_function():
            # Inside a tf.data graph trace or tf.function: use pure TF ops to
            # avoid torch/jax dispatch failing on symbolic TF tensors.
            import tensorflow as tf

            height = tf.shape(inputs)[1]
            width = tf.shape(inputs)[2]
            total_px = height * width
            factor = tf.sqrt(
                tf.cast(target_px, "float32") / tf.cast(total_px, "float32")
            )
            ideal_height = factor * tf.cast(height, "float32")
            ideal_width = factor * tf.cast(width, "float32")
            side_mult_f = tf.cast(side_mult, "float32")
            target_height = tf.cast(
                tf.floor(ideal_height / side_mult_f) * side_mult_f, "int32"
            )
            target_width = tf.cast(
                tf.floor(ideal_width / side_mult_f) * side_mult_f, "int32"
            )
            target_height = tf.maximum(target_height, side_mult)
            target_width = tf.maximum(target_width, side_mult)
            float_inputs = tf.cast(inputs, "float32")
            resized = tf.image.resize(
                float_inputs,
                size=(target_height, target_width),
                method=tf.image.ResizeMethod.BICUBIC,
                antialias=True,
            )
            return tf.clip_by_value(resized, 0.0, 255.0)
        else:
            # Eager mode: use backend-agnostic ops (works with concrete
            # numpy/torch/jax arrays).
            height = ops.shape(inputs)[1]
            width = ops.shape(inputs)[2]
            total_px = height * width
            factor = ops.sqrt(
                ops.cast(target_px, "float32") / ops.cast(total_px, "float32")
            )
            ideal_height = factor * ops.cast(height, "float32")
            ideal_width = factor * ops.cast(width, "float32")
            target_height = ops.cast(
                ops.floor(ideal_height / ops.cast(side_mult, "float32"))
                * ops.cast(side_mult, "float32"),
                "int32",
            )
            target_width = ops.cast(
                ops.floor(ideal_width / ops.cast(side_mult, "float32"))
                * ops.cast(side_mult, "float32"),
                "int32",
            )
            target_height = ops.maximum(target_height, side_mult)
            target_width = ops.maximum(target_width, side_mult)
            float_inputs = ops.cast(inputs, "float32")
            # Resize using bicubic interpolation with antialias=True.
            #
            # HF's AutoProcessor uses PIL Image.resize(..., Image.BICUBIC),
            # which is a Catmull-Rom spline evaluated in C without antialias.
            # `ops.image.resize` uses the backend's GPU-friendly bicubic
            # kernel, which differs near sharp edges (max|Δ| ≈ 22/255 with
            # antialias=True vs ≈ 35/255 without). A perfect match is not
            # achievable without falling back to PIL, which would break
            # tf.function and GPU execution.
            resized = ops.image.resize(
                float_inputs,
                size=(target_height, target_width),
                interpolation="bicubic",
                antialias=True,
            )
            return ops.clip(resized, 0.0, 255.0)


@keras_hub_export("keras_hub.layers.Gemma4ImageConverter")
class Gemma4ImageConverter(ImageConverter):
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

        # Overwrite fixed resizing with aspect-ratio preserving resizing
        self.resizing = Gemma4AspectRatioResizing(
            patch_size=self.patch_size,
            max_soft_tokens=self.max_soft_tokens,
            pooling_kernel_size=self.pooling_kernel_size,
        )

    @preprocessing_function
    def call(self, inputs):
        # Perform aspect-ratio-preserving resize by calling the resizing
        # layer's `call()` method directly rather than via Keras `__call__`.
        # This bypasses Keras dtype conversion which fails when a symbolic
        # tf.Tensor (created inside tf.data.Dataset.map) meets the torch
        # backend's `convert_to_tensor` path.
        if isinstance(inputs, dict):
            x = self.resizing.call(inputs["images"])
        else:
            x = self.resizing.call(inputs)

        # Apply scale/offset from the parent ImageConverter (normalization).
        # This replicates what super().call() would do, without going through
        # Keras __call__ which fails on symbolic TF tensors in graph mode.
        if self.scale is not None:
            scale = self._expand_non_channel_dims(self.scale, x)
            x, scale = self._convert_types(x, scale, self.compute_dtype)
            x = x * scale
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, x)
            x, offset = self._convert_types(x, offset, x.dtype)
            x = x + offset

        mapped_x = x

        ps = self.patch_size
        max_patches = self.max_soft_tokens * (self.pooling_kernel_size**2)

        from keras_hub.src.utils.tensor_utils import in_tf_function

        if in_tf_function():
            import tensorflow as tf

            # Shape computations via TF ops (graph-safe).
            shape = tf.shape(mapped_x)
            batch_size = shape[0]
            h = shape[1]
            w = shape[2]
            n_h = h // ps
            n_w = w // ps

            mapped_x = tf.reshape(mapped_x, (batch_size, n_h, ps, n_w, ps, 3))
            mapped_x = tf.transpose(mapped_x, (0, 1, 3, 2, 4, 5))
            pixel_values_unpadded = tf.reshape(
                mapped_x, (batch_size, n_h * n_w, 3 * ps * ps)
            )

            row_ids = tf.range(n_h, dtype="int32")
            col_ids = tf.range(n_w, dtype="int32")

            col_grid = tf.tile(tf.reshape(col_ids, (1, n_w)), (n_h, 1))
            row_grid = tf.tile(tf.reshape(row_ids, (n_h, 1)), (1, n_w))

            pixel_position_ids_unpadded = tf.stack(
                [tf.reshape(col_grid, [-1]), tf.reshape(row_grid, [-1])],
                axis=-1,
            )
            pixel_position_ids_unpadded = tf.tile(
                tf.expand_dims(pixel_position_ids_unpadded, 0),
                [batch_size, 1, 1],
            )

            # Always pad to max_patches and set the static shape so that
            # tf.data.Dataset.map can infer concrete element specs.
            current_patches = n_h * n_w
            pad_len = tf.cast(max_patches, tf.int32) - current_patches
            p_val_pad = tf.zeros(
                (batch_size, pad_len, 3 * ps * ps),
                dtype=pixel_values_unpadded.dtype,
            )
            pixel_values = tf.concat([pixel_values_unpadded, p_val_pad], axis=1)
            pixel_values = tf.ensure_shape(
                pixel_values, [None, max_patches, 3 * ps * ps]
            )
            pos_pad = tf.fill((batch_size, pad_len, 2), -1)
            pixel_position_ids = tf.concat(
                [pixel_position_ids_unpadded, pos_pad], axis=1
            )
            pixel_position_ids = tf.ensure_shape(
                pixel_position_ids, [None, max_patches, 2]
            )

        else:
            # Shape computations via backend-agnostic ops (eager mode).
            shape = ops.shape(mapped_x)
            batch_size = shape[0]
            h = shape[1]
            w = shape[2]
            n_h = h // ps
            n_w = w // ps

            mapped_x = ops.reshape(mapped_x, (batch_size, n_h, ps, n_w, ps, 3))
            mapped_x = ops.transpose(mapped_x, (0, 1, 3, 2, 4, 5))
            pixel_values_unpadded = ops.reshape(
                mapped_x, (batch_size, n_h * n_w, 3 * ps * ps)
            )

            # Build (x, y) position indices for each patch.
            row_ids = ops.arange(n_h, dtype="int32")  # y
            col_ids = ops.arange(n_w, dtype="int32")  # x

            col_grid = ops.tile(ops.reshape(col_ids, (1, n_w)), (n_h, 1))
            row_grid = ops.tile(ops.reshape(row_ids, (n_h, 1)), (1, n_w))

            pixel_position_ids_unpadded = ops.stack(
                [ops.reshape(col_grid, [-1]), ops.reshape(row_grid, [-1])],
                axis=-1,
            )
            pixel_position_ids_unpadded = ops.tile(
                ops.expand_dims(pixel_position_ids_unpadded, 0),
                [batch_size, 1, 1],
            )

            # Pad to max_patches with a regular Python conditional (safe in
            # eager mode where n_h and n_w are concrete scalars).
            current_patches = n_h * n_w
            pad_len = max_patches - current_patches

            if pad_len > 0:
                p_val_pad = ops.zeros(
                    (batch_size, pad_len, 3 * ps * ps),
                    dtype=pixel_values_unpadded.dtype,
                )
                pixel_values = ops.concatenate(
                    [pixel_values_unpadded, p_val_pad], axis=1
                )
                pos_pad = (
                    ops.ones(
                        (batch_size, pad_len, 2),
                        dtype=pixel_position_ids_unpadded.dtype,
                    )
                    * -1
                )
                pixel_position_ids = ops.concatenate(
                    [pixel_position_ids_unpadded, pos_pad], axis=1
                )
            else:
                pixel_values = pixel_values_unpadded
                pixel_position_ids = pixel_position_ids_unpadded

        outputs = {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }

        if isinstance(inputs, dict):
            # Update the original dict to preserve other keys (like text)
            inputs.update(outputs)
            return inputs
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
