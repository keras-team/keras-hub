import tensorflow as tf

class RandomElasticDeformation3D(tf.keras.layers.Layer):
    """
    A high-performance 3D elastic deformation layer optimized for TPUs and GPUs.
    ... (docstring is the same) ...
    """
    def __init__(self,
                 grid_size=(4, 4, 4),
                 alpha=35.0,
                 sigma=2.5,
                 data_format="DHWC",
                 **kwargs):
        super().__init__(**kwargs)
        self.grid_size = grid_size
        self.alpha = tf.constant(alpha, dtype=tf.bfloat16)
        self.sigma = tf.constant(sigma, dtype=tf.bfloat16)
        if data_format not in ["DHWC", "HWDC"]:
            raise ValueError("`data_format` must be one of 'DHWC' or 'HWDC'")
        self.data_format = data_format

    def _separable_gaussian_filter_3d(self, tensor, sigma):

        kernel_size = tf.cast(2 * tf.round(3 * sigma) + 1, dtype=tf.int32)
        ax = tf.range(-tf.cast(kernel_size // 2, tf.bfloat16) + 1.0,
                      tf.cast(kernel_size // 2, tf.bfloat16) + 1.0)
        kernel_1d = tf.exp(-(ax**2) / (2.0 * self.sigma**2))
        kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
        filter_d = tf.cast(tf.reshape(kernel_1d, [-1, 1, 1, 1, 1]), dtype=tensor.dtype)
        filter_h = tf.cast(tf.reshape(kernel_1d, [1, -1, 1, 1, 1]), dtype=tensor.dtype)
        filter_w = tf.cast(tf.reshape(kernel_1d, [1, 1, -1, 1, 1]), dtype=tensor.dtype)
        tensor = tf.nn.convolution(tensor, filter_d, strides=1, padding='SAME')
        tensor = tf.nn.convolution(tensor, filter_h, strides=1, padding='SAME')
        tensor = tf.nn.convolution(tensor, filter_w, strides=1, padding='SAME')
        return tensor

    def call(self, inputs):
        image_volume, label_volume = inputs
        original_image_dtype = image_volume.dtype

        was_batched = True
        if image_volume.shape.rank == 4:
            was_batched = False
            image_volume = tf.expand_dims(image_volume, axis=0)
            label_volume = tf.expand_dims(label_volume, axis=0)

        if self.data_format == "HWDC":
            image_volume = tf.transpose(image_volume, perm=[0, 3, 1, 2, 4])
            label_volume = tf.transpose(label_volume, perm=[0, 3, 1, 2, 4])

        image_volume = tf.cast(image_volume, dtype=tf.bfloat16)
        input_shape = tf.shape(image_volume)
        B, D, H, W = input_shape[0], input_shape[1], input_shape[2], input_shape[3]

        coarse_flow = tf.random.uniform(
            shape=(B, self.grid_size[0], self.grid_size[1], self.grid_size[2], 3),
            minval=-1, maxval=1, dtype=tf.bfloat16)

        flow = tf.reshape(coarse_flow, [B * self.grid_size[0], self.grid_size[1], self.grid_size[2], 3])
        flow = tf.image.resize(flow, size=[H, W], method='bicubic')
        flow = tf.reshape(flow, [B, self.grid_size[0], H, W, 3])
        flow = tf.transpose(flow, perm=[0, 2, 3, 1, 4])
        flow = tf.reshape(flow, [B * H * W, self.grid_size[0], 3])
        flow = tf.image.resize(tf.expand_dims(flow, axis=1), size=[1, D], method='bicubic')
        flow = tf.squeeze(flow, axis=1)
        flow = tf.reshape(flow, [B, H, W, D, 3])
        flow = tf.transpose(flow, perm=[0, 3, 1, 2, 4])


        flow = tf.cast(flow, dtype=tf.bfloat16)
        
        flow_components = tf.unstack(flow, axis=-1)
        smoothed_components = []
        for component in flow_components:
            smoothed_component = self._separable_gaussian_filter_3d(
                component[..., tf.newaxis], self.sigma
            )
            smoothed_components.append(smoothed_component[..., 0])
        smoothed_flow = tf.stack(smoothed_components, axis=-1)
        

        flow = smoothed_flow * self.alpha

        grid_d, grid_h, grid_w = tf.meshgrid(
            tf.range(D, dtype=tf.bfloat16),
            tf.range(H, dtype=tf.bfloat16),
            tf.range(W, dtype=tf.bfloat16),
            indexing='ij'
        )
        grid = tf.stack([grid_d, grid_h, grid_w], axis=-1)
        

        warp_grid = tf.expand_dims(grid, 0) + flow
        
        warp_grid_floor = tf.floor(warp_grid)
        t = warp_grid - warp_grid_floor

        d0 = tf.cast(warp_grid_floor[..., 0], tf.int32); h0 = tf.cast(warp_grid_floor[..., 1], tf.int32); w0 = tf.cast(warp_grid_floor[..., 2], tf.int32)
        d1 = tf.clip_by_value(d0 + 1, 0, D - 1); h1 = tf.clip_by_value(h0 + 1, 0, H - 1); w1 = tf.clip_by_value(w0 + 1, 0, W - 1)
        d0 = tf.clip_by_value(d0, 0, D - 1); h0 = tf.clip_by_value(h0, 0, H - 1); w0 = tf.clip_by_value(w0, 0, W - 1)

        c000 = tf.gather_nd(image_volume, tf.stack([d0, h0, w0], axis=-1), batch_dims=1); c001 = tf.gather_nd(image_volume, tf.stack([d0, h0, w1], axis=-1), batch_dims=1)
        c010 = tf.gather_nd(image_volume, tf.stack([d0, h1, w0], axis=-1), batch_dims=1); c011 = tf.gather_nd(image_volume, tf.stack([d0, h1, w1], axis=-1), batch_dims=1)
        c100 = tf.gather_nd(image_volume, tf.stack([d1, h0, w0], axis=-1), batch_dims=1); c101 = tf.gather_nd(image_volume, tf.stack([d1, h0, w1], axis=-1), batch_dims=1)
        c110 = tf.gather_nd(image_volume, tf.stack([d1, h1, w0], axis=-1), batch_dims=1); c111 = tf.gather_nd(image_volume, tf.stack([d1, h1, w1], axis=-1), batch_dims=1)

        td, th, tw = t[..., 0:1], t[..., 1:2], t[..., 2:3]
        c00 = c000*(1-tw) + c001*tw; c01 = c010*(1-tw) + c011*tw; c10 = c100*(1-tw) + c101*tw; c11 = c110*(1-tw) + c111*tw
        c0 = c00*(1-th) + c01*th; c1 = c10*(1-th) + c11*th
        deformed_image = c0*(1-td) + c1*td
        deformed_image = tf.cast(deformed_image, original_image_dtype)

        nearest_indices_float = tf.round(warp_grid)
        nearest_d = tf.clip_by_value(tf.cast(nearest_indices_float[..., 0], tf.int32), 0, D - 1)
        nearest_h = tf.clip_by_value(tf.cast(nearest_indices_float[..., 1], tf.int32), 0, H - 1)
        nearest_w = tf.clip_by_value(tf.cast(nearest_indices_float[..., 2], tf.int32), 0, W - 1)
        deformed_label = tf.gather_nd(label_volume, tf.stack([nearest_d, nearest_h, nearest_w], axis=-1), batch_dims=1)

        if self.data_format == "HWDC":
            deformed_image = tf.transpose(deformed_image, perm=[0, 2, 3, 1, 4])
            deformed_label = tf.transpose(deformed_label, perm=[0, 2, 3, 1, 4])

        if not was_batched:
            deformed_image = tf.squeeze(deformed_image, axis=0)
            deformed_label = tf.squeeze(deformed_label, axis=0)

        return deformed_image, deformed_label