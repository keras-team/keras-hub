from keras import ops
from keras import layers
from keras import random

class RandomElasticDeformation3D(layers.Layer):
    """
    A high-performance 3D elastic deformation layer optimized for TPUs.
    
    This implementation leverages the layer's compute_dtype (e.g., bfloat16) 
    to potentially halve memory bandwidth requirements and uses a vectorized 
    mapping for maximum speed.
    """
    def __init__(self,
                 grid_size=(4, 4, 4),
                 alpha=35.0,
                 sigma=2.5,
                 data_format="channels_last",
                 **kwargs):
        super().__init__(**kwargs)

        self.grid_size = grid_size
        self.alpha = alpha
        self.sigma = sigma
        self.data_format = data_format
        if data_format not in ["channels_last", "channels_first"]:
            raise ValueError(
                "`data_format` must be one of 'channels_last' or "
                f"'channels_first'. Received: {data_format}"
            )
            
    def build(self, input_shape):
        """Create tensor state in build to respect the layer's dtype."""
        self._alpha_tensor = ops.convert_to_tensor(self.alpha, dtype=self.compute_dtype)
        self._sigma_tensor = ops.convert_to_tensor(self.sigma, dtype=self.compute_dtype)
        
        # Pre-compute the 1D Gaussian kernel
        kernel_size = ops.cast(2 * ops.round(3 * self._sigma_tensor) + 1, dtype="int32")
        ax = ops.arange(-ops.cast(kernel_size // 2, self.compute_dtype) + 1.0,
                        ops.cast(kernel_size // 2, self.compute_dtype) + 1.0)
        kernel_1d = ops.exp(-(ax**2) / (2.0 * self._sigma_tensor**2))
        self.kernel_1d = kernel_1d / ops.sum(kernel_1d)
        self.built = True

    def _separable_gaussian_filter_3d(self, tensor):
        """Apply a 3D Gaussian filter using separable 1D convolutions."""
        depth_kernel = ops.reshape(self.kernel_1d, (-1, 1, 1, 1, 1))
        tensor = ops.conv(tensor, ops.cast(depth_kernel, dtype=tensor.dtype), padding='same')

        height_kernel = ops.reshape(self.kernel_1d, (1, -1, 1, 1, 1))
        tensor = ops.conv(tensor, ops.cast(height_kernel, dtype=tensor.dtype), padding='same')

        width_kernel = ops.reshape(self.kernel_1d, (1, 1, -1, 1, 1))
        tensor = ops.conv(tensor, ops.cast(width_kernel, dtype=tensor.dtype), padding='same')
        
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
        B, D, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        C = input_shape[4]

        # 1. Create a coarse random flow field.
        coarse_flow = random.uniform(
            shape=(B, self.grid_size[0], self.grid_size[1], self.grid_size[2], 3),
            minval=-1, maxval=1, dtype=compute_dtype
        )

        # 2. Upsample the flow field.
        flow = coarse_flow
        flow_shape = ops.shape(flow)
        flow = ops.reshape(flow, (flow_shape[0] * flow_shape[1], flow_shape[2], flow_shape[3], 3))
        flow = ops.image.resize(flow, (H, W), interpolation="bicubic")
        flow = ops.reshape(flow, (flow_shape[0], flow_shape[1], H, W, 3))
        flow = ops.transpose(flow, (0, 2, 3, 1, 4))
        flow_shape = ops.shape(flow)
        flow = ops.reshape(flow, (flow_shape[0] * flow_shape[1] * flow_shape[2], flow_shape[3], 1, 3))
        flow = ops.image.resize(flow, (D, 1), interpolation="bicubic")
        flow = ops.reshape(flow, (flow_shape[0], flow_shape[1], flow_shape[2], D, 3))
        flow = ops.transpose(flow, (0, 3, 1, 2, 4))

        # 3. Apply Gaussian smoothing.
        flow_components = ops.unstack(flow, axis=-1)
        smoothed_components = []
        for component in flow_components:
            component_reshaped = ops.expand_dims(component, axis=-1)
            smoothed_component = self._separable_gaussian_filter_3d(component_reshaped)
            smoothed_components.append(ops.squeeze(smoothed_component, axis=-1))
        smoothed_flow = ops.stack(smoothed_components, axis=-1)
        
        # 4. Scale the flow field and create warp grid.
        flow = smoothed_flow * self._alpha_tensor
        grid_d, grid_h, grid_w = ops.meshgrid(
            ops.arange(D, dtype=compute_dtype),
            ops.arange(H, dtype=compute_dtype),
            ops.arange(W, dtype=compute_dtype),
            indexing='ij'
        )
        grid = ops.stack([grid_d, grid_h, grid_w], axis=-1)
        warp_grid = ops.expand_dims(grid, 0) + flow
        

        batched_coords = ops.transpose(warp_grid, (0, 4, 1, 2, 3))


        deformed_images_batched = []
        for i in range(B):

            image_slice = image_volume[i] 
            coords = batched_coords[i]      

 
            image_slice_transposed = ops.transpose(image_slice, (3, 0, 1, 2))
            
            deformed_channels = []
            for c in range(C):

                deformed_channel = ops.image.map_coordinates(
                    image_slice_transposed[c], coords, order=1
                )
                deformed_channels.append(deformed_channel)
            
            # Stack and transpose back to (D, H, W, C)
            deformed_image_slice = ops.stack(deformed_channels, axis=0)
            deformed_images_batched.append(ops.transpose(deformed_image_slice, (1, 2, 3, 0)))

        deformed_image = ops.stack(deformed_images_batched, axis=0)

        # Process Labels: loop over the batch dimension.
        deformed_labels_batched = []
        for i in range(B):
            label_slice = label_volume[i] 
            coords = batched_coords[i]     
            

            label_channel = ops.squeeze(label_slice, axis=-1)
            deformed_label_channel = ops.image.map_coordinates(
                label_channel, coords, order=0
            )

            deformed_labels_batched.append(ops.expand_dims(deformed_label_channel, axis=-1))

        deformed_label = ops.stack(deformed_labels_batched, axis=0)
        


        deformed_image = ops.cast(deformed_image, original_image_dtype)
        deformed_label = ops.cast(deformed_label, original_label_dtype)

        if not was_batched:
            deformed_image = ops.squeeze(deformed_image, axis=0)
            deformed_label = ops.squeeze(deformed_label, axis=0)

        return deformed_image, deformed_label

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer."""
        image_shape, label_shape = input_shape
        return image_shape, label_shape

    def get_config(self):
        config = super().get_config()
        config.update({
            "grid_size": self.grid_size,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "data_format": self.data_format,
        })
        return config