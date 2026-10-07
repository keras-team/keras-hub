# Add this import
from keras import backend
from keras import layers
from keras import ops
from keras import random


class RandomElasticDeformation3D(layers.Layer):
    """
    A high-performance 3D elastic deformation layer optimized for TPUs.
    """

    def __init__(
        self,
        grid_size=(4, 4, 4),
        alpha=35.0,
        sigma=2.5,
        data_format="channels_last",
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.seed = seed
        self.alpha = alpha
        self.sigma = sigma
        self.data_format = data_format
        self._rng = random.SeedGenerator(seed) if seed is not None else None
        if data_format not in ["channels_last", "channels_first"]:
            message = (
                "`data_format` must be one of 'channels_last' or "
                f"'channels_first'. Received: {self.data_format}"
            )
            raise ValueError(message)

    def build(self, input_shape):
        self._alpha_tensor = ops.convert_to_tensor(
            self.alpha, dtype=self.compute_dtype
        )
        self._sigma_tensor = ops.convert_to_tensor(
            self.sigma, dtype=self.compute_dtype
        )
        kernel_size = ops.cast(
            2 * ops.round(3 * self._sigma_tensor) + 1, dtype="int32"
        )
        ax = ops.arange(
            -ops.cast(kernel_size // 2, self.compute_dtype) + 1.0,
            ops.cast(kernel_size // 2, self.compute_dtype) + 1.0,
        )
        kernel_1d = ops.exp(-(ax**2) / (2.0 * self._sigma_tensor**2))
        self.kernel_1d = kernel_1d / ops.sum(kernel_1d)
        self.built = True

    def _separable_gaussian_filter_3d(self, tensor):
        depth_kernel = ops.reshape(self.kernel_1d, (-1, 1, 1, 1, 1))
        tensor = ops.conv(
            tensor, ops.cast(depth_kernel, dtype=tensor.dtype), padding="same"
        )
        height_kernel = ops.reshape(self.kernel_1d, (1, -1, 1, 1, 1))
        tensor = ops.conv(
            tensor, ops.cast(height_kernel, dtype=tensor.dtype), padding="same"
        )
        width_kernel = ops.reshape(self.kernel_1d, (1, 1, -1, 1, 1))
        tensor = ops.conv(
            tensor, ops.cast(width_kernel, dtype=tensor.dtype), padding="same"
        )
        return tensor

    def call(self, inputs):
        image_volume, label_volume = inputs
        original_image_dtype = image_volume.dtype
        original_label_dtype = label_volume.dtype
        compute_dtype = self.compute_dtype

        was_batched = True
        if len(image_volume.shape) == 4:
            was_batched = False
            image_volume = ops.expand_dims(image_volume, axis=0)
            label_volume = ops.expand_dims(label_volume, axis=0)

        image_volume = ops.cast(image_volume, dtype=compute_dtype)
        label_volume = ops.cast(label_volume, dtype=compute_dtype)

        input_shape = ops.shape(image_volume)
        B, D, H, W, C = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        )

        if self._rng is not None:
            coarse_flow = random.uniform(
                shape=(
                    B,
                    self.grid_size[0],
                    self.grid_size[1],
                    self.grid_size[2],
                    3,
                ),
                minval=-1,
                maxval=1,
                dtype=compute_dtype,
                seed=self._rng,
            )
        else:
            coarse_flow = random.uniform(
                shape=(
                    B,
                    self.grid_size[0],
                    self.grid_size[1],
                    self.grid_size[2],
                    3,
                ),
                minval=-1,
                maxval=1,
                dtype=compute_dtype,
            )

        flow = coarse_flow
        flow_shape = ops.shape(flow)
        flow = ops.reshape(
            flow,
            (flow_shape[0] * flow_shape[1], flow_shape[2], flow_shape[3], 3),
        )
        flow = ops.image.resize(flow, (H, W), interpolation="bicubic")
        flow = ops.reshape(flow, (flow_shape[0], flow_shape[1], H, W, 3))
        flow = ops.transpose(flow, (0, 2, 3, 1, 4))
        flow_shape = ops.shape(flow)
        flow = ops.reshape(
            flow,
            (
                flow_shape[0] * flow_shape[1] * flow_shape[2],
                flow_shape[3],
                1,
                3,
            ),
        )
        flow = ops.image.resize(flow, (D, 1), interpolation="bicubic")
        flow = ops.reshape(
            flow, (flow_shape[0], flow_shape[1], flow_shape[2], D, 3)
        )
        flow = ops.transpose(flow, (0, 3, 1, 2, 4))

        flow_components = ops.unstack(flow, axis=-1)
        smoothed_components = []
        for component in flow_components:
            smoothed_components.append(
                ops.squeeze(
                    self._separable_gaussian_filter_3d(
                        ops.expand_dims(component, axis=-1)
                    ),
                    axis=-1,
                )
            )
        smoothed_flow = ops.stack(smoothed_components, axis=-1)

        flow = smoothed_flow * self._alpha_tensor
        grid_d, grid_h, grid_w = ops.meshgrid(
            ops.arange(D, dtype=compute_dtype),
            ops.arange(H, dtype=compute_dtype),
            ops.arange(W, dtype=compute_dtype),
            indexing="ij",
        )
        grid = ops.stack([grid_d, grid_h, grid_w], axis=-1)
        warp_grid = ops.expand_dims(grid, 0) + flow

        batched_coords = ops.transpose(warp_grid, (0, 4, 1, 2, 3))

        def perform_map(elems):
            image_slice, label_slice, coords = elems
            deformed_channels = []
            image_slice_transposed = ops.transpose(image_slice, (3, 0, 1, 2))
            # The channel dimension C is a static value when the graph is built
            for c in range(C):
                deformed_channels.append(
                    ops.image.map_coordinates(
                        image_slice_transposed[c], coords, order=1
                    )
                )
            deformed_image_slice = ops.stack(deformed_channels, axis=0)
            deformed_image_slice = ops.transpose(
                deformed_image_slice, (1, 2, 3, 0)
            )
            label_channel = ops.squeeze(label_slice, axis=-1)
            deformed_label_channel = ops.image.map_coordinates(
                label_channel, coords, order=0
            )
            deformed_label_slice = ops.expand_dims(
                deformed_label_channel, axis=-1
            )
            return deformed_image_slice, deformed_label_slice

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            deformed_image, deformed_label = tf.map_fn(
                perform_map,
                elems=(image_volume, label_volume, batched_coords),
                dtype=(compute_dtype, compute_dtype),
            )
        elif backend.backend() == "jax":
            import jax

            deformed_image, deformed_label = jax.lax.map(
                perform_map, xs=(image_volume, label_volume, batched_coords)
            )
        else:
            deformed_images_list = []
            deformed_labels_list = []
            for i in range(B):
                img_slice, lbl_slice = perform_map(
                    (image_volume[i], label_volume[i], batched_coords[i])
                )
                deformed_images_list.append(img_slice)
                deformed_labels_list.append(lbl_slice)
            deformed_image = ops.stack(deformed_images_list, axis=0)
            deformed_label = ops.stack(deformed_labels_list, axis=0)

        deformed_image = ops.cast(deformed_image, original_image_dtype)
        deformed_label = ops.cast(deformed_label, original_label_dtype)

        if not was_batched:
            deformed_image = ops.squeeze(deformed_image, axis=0)
            deformed_label = ops.squeeze(deformed_label, axis=0)

        return deformed_image, deformed_label

    def compute_output_shape(self, input_shape):
        image_shape, label_shape = input_shape
        return image_shape, label_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "grid_size": self.grid_size,
                "alpha": self.alpha,
                "sigma": self.sigma,
                "data_format": self.data_format,
                "seed": self.seed,
            }
        )
        return config
