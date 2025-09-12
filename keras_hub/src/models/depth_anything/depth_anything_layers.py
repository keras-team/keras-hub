from keras import layers
from keras import ops

from keras_hub.src.models.depth_anything.interpolate import interpolate
from keras_hub.src.utils.keras_utils import standardize_data_format


class DepthAnythingTokenToImage(layers.Layer):
    """A layer that converts tokens into images.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        patch_height: int. The height of each patch.
        patch_width: int. The width of each patch.
        num_cls_tokens: int. The number of class tokens at the beginning of
            the sequence. Defaults to `1`.
        num_register_tokens: int. The number of register tokens after the
            class tokens. Defaults to `0`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        hidden_dim,
        patch_height,
        patch_width,
        num_cls_tokens=1,
        num_register_tokens=0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.num_cls_tokens = int(num_cls_tokens)
        self.num_register_tokens = int(num_register_tokens)
        self.data_format = standardize_data_format(data_format)
        # Always use channels_last for reshaping first.
        self.target_shape = (
            self.patch_height,
            self.patch_width,
            self.hidden_dim,
        )

    def call(self, inputs):
        # Remove the cls token.
        x = inputs[:, self.num_cls_tokens + self.num_register_tokens :, ...]

        x = ops.reshape(x, (ops.shape(x)[0],) + self.target_shape)
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "patch_height": self.patch_height,
                "patch_width": self.patch_width,
                "num_cls_tokens": self.num_cls_tokens,
                "num_register_tokens": self.num_register_tokens,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0], *self.target_shape]
        if self.data_format == "channels_first":
            output_shape = [
                output_shape[0],
                output_shape[3],
                output_shape[1],
                output_shape[2],
            ]
        return output_shape


class DepthAnythingReassembleLayer(layers.Layer):
    """A layer that resizes the input images.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        factor: float. The resizing factor. If `factor > 1`, the layer upsamples
            the input. If `factor < 1`, the layer downsamples the input. If
            `factor == 1`, the layer only applies a linear projection.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, factor, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.factor = float(factor)
        self.data_format = standardize_data_format(data_format)

        self.projection = layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=1,
            data_format=self.data_format,
            use_bias=True,
            dtype=self.dtype_policy,
            name="projection",
        )
        if self.factor > 1:
            self.padding = layers.Identity(
                dtype=self.dtype_policy, name="padding"
            )
            self.resize = layers.Conv2DTranspose(
                filters=self.hidden_dim,
                kernel_size=int(self.factor),
                strides=int(self.factor),
                data_format=self.data_format,
                use_bias=True,
                dtype=self.dtype_policy,
                name="resize",
            )
        elif self.factor == 1:
            self.padding = layers.Identity(
                dtype=self.dtype_policy, name="padding"
            )
            self.resize = layers.Identity(
                dtype=self.dtype_policy, name="resize"
            )
        elif self.factor < 1:
            self.padding = layers.ZeroPadding2D(
                padding=(1, 1),
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name="padding",
            )
            self.resize = layers.Conv2D(
                filters=self.hidden_dim,
                kernel_size=3,
                strides=int(1 / self.factor),
                data_format=self.data_format,
                use_bias=True,
                dtype=self.dtype_policy,
                name="resize",
            )

    def build(self, inputs_shape):
        self.projection.build(inputs_shape)
        inputs_shape = self.projection.compute_output_shape(inputs_shape)
        self.padding.build(inputs_shape)
        inputs_shape = self.padding.compute_output_shape(inputs_shape)
        self.resize.build(inputs_shape)

    def call(self, inputs, training=None):
        x = self.projection(inputs, training=training)
        x = self.padding(x, training=training)
        return self.resize(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "factor": self.factor,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.data_format == "channels_first":
            output_shape[1] = self.hidden_dim
            output_shape[2] = int(output_shape[2] * self.factor)
            output_shape[3] = int(output_shape[3] * self.factor)
        else:
            output_shape[1] = int(output_shape[1] * self.factor)
            output_shape[2] = int(output_shape[2] * self.factor)
            output_shape[3] = self.hidden_dim
        return output_shape


class DepthAnythingPreActResidualLayer(layers.Layer):
    """A ReLU + Conv2D layer.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.data_format = standardize_data_format(data_format)

        self.activation1 = layers.ReLU(
            dtype=self.dtype_policy, name="activation1"
        )
        self.padding1 = layers.ZeroPadding2D(
            padding=(1, 1),
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="padding1",
        )
        self.convolution1 = layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=1,
            data_format=self.data_format,
            use_bias=True,
            dtype=self.dtype_policy,
            name="convolution1",
        )
        self.activation2 = layers.ReLU(
            dtype=self.dtype_policy, name="activation2"
        )
        self.padding2 = layers.ZeroPadding2D(
            padding=(1, 1),
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="padding2",
        )
        self.convolution2 = layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=3,
            strides=1,
            data_format=self.data_format,
            use_bias=True,
            dtype=self.dtype_policy,
            name="convolution2",
        )

    def build(self, inputs_shape):
        self.activation1.build(inputs_shape)
        self.padding1.build(inputs_shape)
        inputs_shape = self.padding1.compute_output_shape(inputs_shape)
        self.convolution1.build(inputs_shape)
        inputs_shape = self.convolution1.compute_output_shape(inputs_shape)
        self.activation2.build(inputs_shape)
        self.padding2.build(inputs_shape)
        inputs_shape = self.padding2.compute_output_shape(inputs_shape)
        self.convolution2.build(inputs_shape)

    def call(self, inputs, training=None):
        residual = inputs
        x = self.activation1(inputs, training=training)
        x = self.padding1(x, training=training)
        x = self.convolution1(x, training=training)
        x = self.activation2(x, training=training)
        x = self.padding2(x, training=training)
        x = self.convolution2(x, training=training)
        return ops.add(x, residual)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape


class DepthAnythingFeatureFusionLayer(layers.Layer):
    """A layer that fuses the incoming features.

    Args:
        hidden_dim: int. The number of units in the hidden layers.
        size: tuple of int. The target size of the output feature map.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(self, hidden_dim, size, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.size = tuple(int(s) for s in size)
        self.data_format = standardize_data_format(data_format)

        self.residual_layer1 = DepthAnythingPreActResidualLayer(
            hidden_dim=self.hidden_dim,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="residual_layer1",
        )
        self.residual_layer2 = DepthAnythingPreActResidualLayer(
            hidden_dim=self.hidden_dim,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="residual_layer2",
        )
        self.projection = layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=1,
            data_format=self.data_format,
            use_bias=True,
            dtype=self.dtype_policy,
            name="projection",
        )

    def build(self, inputs_shape):
        self.residual_layer1.build(inputs_shape)
        self.residual_layer2.build(inputs_shape)
        inputs_shape = list(inputs_shape)
        if self.data_format == "channels_last":
            inputs_shape[1] = self.size[0]
            inputs_shape[2] = self.size[1]
        else:
            inputs_shape[2] = self.size[0]
            inputs_shape[3] = self.size[1]
        self.projection.build(inputs_shape)

    def call(self, inputs, residual=None, training=None):
        if residual is not None:
            inputs = ops.add(
                inputs, self.residual_layer1(residual, training=training)
            )

        x = self.residual_layer2(inputs, training=training)
        x = interpolate(x, size=self.size, data_format=self.data_format)
        return self.projection(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "size": self.size,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        input_shape = self.residual_layer2.compute_output_shape(input_shape)
        input_shape = list(input_shape)
        if self.data_format == "channels_last":
            input_shape[1] = self.size[0]
            input_shape[2] = self.size[1]
        else:
            input_shape[2] = self.size[0]
            input_shape[3] = self.size[1]
        return self.projection.compute_output_shape(input_shape)


class DepthAnythingNeck(layers.Layer):
    """A DepthAnything neck layer.

    Args:
        patch_size: int. The size of one side of each patch.
        image_size: tuple of ints. The (height, width) of the input images.
        backbone_hidden_dim: int. The number of units in the backbone layers.
        neck_hidden_dims: List of int. The number of units in each neck layer.
        reassemble_factors: List of float. The resizing factor in each neck
            layer.
        fusion_hidden_dim: int. The number of units in the fusion layers.
        num_cls_tokens: int. The number of class tokens at the beginning of
            the sequence. Defaults to `1`.
        num_register_tokens: int. The number of register tokens after the
            class tokens. Defaults to `0`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        patch_size,
        image_size,
        backbone_hidden_dim,
        neck_hidden_dims,
        reassemble_factors,
        fusion_hidden_dim,
        num_cls_tokens=1,
        num_register_tokens=0,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.backbone_hidden_dim = int(backbone_hidden_dim)
        self.neck_hidden_dims = tuple(int(d) for d in neck_hidden_dims)
        self.reassemble_factors = tuple(float(f) for f in reassemble_factors)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.num_cls_tokens = int(num_cls_tokens)
        self.num_register_tokens = int(num_register_tokens)
        self.data_format = standardize_data_format(data_format)
        if len(self.neck_hidden_dims) != len(self.reassemble_factors):
            raise ValueError(
                "`DepthAnythingNeck` expects the length of `neck_hidden_dims` "
                "and `reassemble_factors` to be the same. "
                f"Received: neck_hidden_dims={neck_hidden_dims}, "
                f"reassemble_factors={reassemble_factors}"
            )

        # Calculate the patch sizes for token to image layers.
        patch_height = self.image_size[0] // self.patch_size
        patch_width = self.image_size[1] // self.patch_size
        # Calculate the sizes for fusion layers.
        fusion_sizes = [
            (int(patch_height * factor), int(patch_width * factor))
            for factor in reversed(self.reassemble_factors[:-1])
        ]
        fusion_sizes = fusion_sizes + [
            (fusion_sizes[-1][0] * 2, fusion_sizes[-1][1] * 2)
        ]

        self.token_to_images = [
            DepthAnythingTokenToImage(
                hidden_dim=backbone_hidden_dim,
                patch_height=patch_height,
                patch_width=patch_width,
                num_cls_tokens=num_cls_tokens,
                num_register_tokens=num_register_tokens,
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"token_to_images_{i}",
            )
            for i in range(len(self.neck_hidden_dims))
        ]
        self.reassemble_stage = [
            DepthAnythingReassembleLayer(
                hidden_dim=hidden_dim,
                factor=factor,
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"reassemble_stage_{i}",
            )
            for i, (hidden_dim, factor) in enumerate(
                zip(self.neck_hidden_dims, self.reassemble_factors)
            )
        ]
        self.paddings = [
            layers.ZeroPadding2D(
                padding=(1, 1),
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"paddings_{i}",
            )
            for i in range(len(self.neck_hidden_dims))
        ]
        self.convs = [
            layers.Conv2D(
                filters=self.fusion_hidden_dim,
                kernel_size=3,
                data_format=self.data_format,
                use_bias=False,
                dtype=self.dtype_policy,
                name=f"convs_{i}",
            )
            for i in range(len(self.neck_hidden_dims))
        ]
        self.fusion_stage = [
            DepthAnythingFeatureFusionLayer(
                hidden_dim=self.fusion_hidden_dim,
                size=size,
                data_format=self.data_format,
                dtype=self.dtype_policy,
                name=f"fusion_stage_{i}",
            )
            for i, size in enumerate(fusion_sizes)
        ]

    def build(self, inputs_shape):
        outputs_shape = []
        # Reassemble stage.
        for i, shape in enumerate(inputs_shape):
            self.token_to_images[i].build(shape)
            shape = self.token_to_images[i].compute_output_shape(shape)
            self.reassemble_stage[i].build(shape)
            shape = self.reassemble_stage[i].compute_output_shape(shape)
            outputs_shape.append(shape)
        # Convs.
        for i, shape in enumerate(outputs_shape):
            self.convs[i].build(shape)
            shape = self.convs[i].compute_output_shape(shape)
            outputs_shape[i] = shape
        # Fusion stage.
        for i, shape in enumerate(reversed(outputs_shape)):
            self.fusion_stage[i].build(shape)

    def call(self, inputs, training=None):
        # Reassemble stage.
        xs = [
            self.reassemble_stage[i](
                self.token_to_images[i](x), training=training
            )
            for i, x in enumerate(inputs)
        ]
        # Convs.
        xs = [
            self.convs[i](self.paddings[i](x), training=training)
            for i, x in enumerate(xs)
        ]
        # Fusion stage.
        fused_xs = []
        fused_x = None
        for i, x in enumerate(reversed(xs)):
            if fused_x is None:
                fused_x = self.fusion_stage[i](
                    x, residual=None, training=training
                )
            else:
                fused_x = self.fusion_stage[i](
                    fused_x, residual=x, training=training
                )
            fused_xs.append(fused_x)
        return fused_xs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "image_size": self.image_size,
                "backbone_hidden_dim": self.backbone_hidden_dim,
                "neck_hidden_dims": self.neck_hidden_dims,
                "reassemble_factors": self.reassemble_factors,
                "fusion_hidden_dim": self.fusion_hidden_dim,
                "num_cls_tokens": self.num_cls_tokens,
                "num_register_tokens": self.num_register_tokens,
            }
        )
        return config


class DepthAnythingDepthEstimationHead(layers.Layer):
    """A DepthAnything neck layer.

    Args:
        patch_size: int. The size of one side of each patch.
        patch_height: int. The height of each patch.
        patch_width: int. The width of each patch.
        hidden_dim: int. The number of units in the hidden layers.
        fusion_hidden_dim: int. The number of units in the fusion layers.
        head_hidden_dim: int. The number of units in the head layers.
        head_in_index: int. The index of the feature map to be used as input
            to the head.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `dtype` etc.
    """

    def __init__(
        self,
        patch_size,
        patch_height,
        patch_width,
        fusion_hidden_dim,
        head_hidden_dim,
        head_in_index,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.fusion_hidden_dim = int(fusion_hidden_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_in_index = int(head_in_index)
        self.data_format = standardize_data_format(data_format)

        # Calculate the interpolate size.
        self.interpolate_size = (
            int(self.patch_height * self.patch_size),
            int(self.patch_width * self.patch_size),
        )

        self.padding1 = layers.ZeroPadding2D(
            padding=(1, 1),
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="padding1",
        )
        self.conv1 = layers.Conv2D(
            filters=self.fusion_hidden_dim // 2,
            kernel_size=3,
            data_format=self.data_format,
            use_bias=True,
            dtype=self.dtype_policy,
            name="conv1",
        )
        self.padding2 = layers.ZeroPadding2D(
            padding=(1, 1),
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="padding2",
        )
        self.conv2 = layers.Conv2D(
            filters=self.head_hidden_dim,
            kernel_size=3,
            data_format=self.data_format,
            use_bias=True,
            dtype=self.dtype_policy,
            name="conv2",
        )
        self.activation1 = layers.ReLU(
            dtype=self.dtype_policy, name="activation1"
        )
        self.conv3 = layers.Conv2D(
            filters=1,
            kernel_size=1,
            data_format=self.data_format,
            use_bias=True,
            dtype=self.dtype_policy,
            name="conv3",
        )

    def build(self, inputs_shape):
        inputs_shape = inputs_shape[self.head_in_index]
        self.padding1.build(inputs_shape)
        inputs_shape = self.padding1.compute_output_shape(inputs_shape)
        self.conv1.build(inputs_shape)
        inputs_shape = self.conv1.compute_output_shape(inputs_shape)
        inputs_shape = list(inputs_shape)
        if self.data_format == "channels_last":
            inputs_shape[1] = self.interpolate_size[0]
            inputs_shape[2] = self.interpolate_size[1]
        else:
            inputs_shape[2] = self.interpolate_size[0]
            inputs_shape[3] = self.interpolate_size[1]
        self.padding2.build(inputs_shape)
        inputs_shape = self.padding2.compute_output_shape(inputs_shape)
        self.conv2.build(inputs_shape)
        inputs_shape = self.conv2.compute_output_shape(inputs_shape)
        self.activation1.build(inputs_shape)
        self.conv3.build(inputs_shape)
        inputs_shape = self.conv3.compute_output_shape(inputs_shape)

    def call(self, inputs, training=None):
        x = inputs[self.head_in_index]
        x = self.padding1(x, training=training)
        x = self.conv1(x, training=training)
        x = interpolate(
            x, size=self.interpolate_size, data_format=self.data_format
        )
        x = self.padding2(x, training=training)
        x = self.conv2(x, training=training)
        x = self.activation1(x, training=training)
        return self.conv3(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "patch_height": self.patch_height,
                "patch_width": self.patch_width,
                "fusion_hidden_dim": self.fusion_hidden_dim,
                "head_hidden_dim": self.head_hidden_dim,
                "head_in_index": self.head_in_index,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[self.head_in_index]
        if self.data_format == "channels_last":
            output_shape = [
                input_shape[0],
                int(self.patch_height * self.patch_size),
                int(self.patch_width * self.patch_size),
                1,
            ]
        else:
            output_shape = [
                input_shape[0],
                1,
                int(self.patch_height * self.patch_size),
                int(self.patch_width * self.patch_size),
            ]
        return output_shape
