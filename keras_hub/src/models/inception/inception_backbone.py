import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.feature_pyramid_backbone import FeaturePyramidBackbone
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.InceptionBackbone")
class InceptionBackbone(FeaturePyramidBackbone):
    """GoogleNet (Inception v1) core network with hyperparameters.

    This class implements a GoogleNet (Inception v1) backbone as described in 
    [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)(CVPR 2015).
    The architecture is characterized by its unique Inception modules that process
    input at different scales simultaneously using multiple filter sizes in parallel.

    Args:
        initial_filters: list of ints. The number of filters for the initial
            convolutional layers.
        initial_strides: list of ints. The strides for the initial convolutional
            layers.
        inception_config: list of lists. Each inner list represents an inception
            block configuration with [1x1_filters, 3x3_reduce_filters, 3x3_filters,
            5x5_reduce_filters, 5x5_filters, pool_proj_filters].
        aux_classifiers: boolean. Whether to include auxiliary classifiers or not.
            Note: In backbone mode, these are typically not used.
        image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the
            inputs. `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch_size, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the model's computations and weights.

    Examples:
    ```python
    input_data = np.random.uniform(0, 1, size=(2, 224, 224, 3))

    # Pretrained GoogleNet backbone.
    model = keras_hub.models.GoogleNetBackbone.from_preset("googlenet_imagenet")
    model(input_data)

    # Randomly initialized GoogleNet backbone with a custom config.
    model = keras_hub.models.GoogleNetBackbone(
        initial_filters=[64, 192],
        initial_strides=[2, 1],
        inception_config=[
            # Inception 3a
            [64, 96, 128, 16, 32, 32],
            # Inception 3b
            [128, 128, 192, 32, 96, 64],
        ],
        aux_classifiers=False,
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        initial_filters,
        initial_strides,
        inception_config,
        aux_classifiers=False,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        if len(initial_filters) != len(initial_strides):
            raise ValueError(
                "The length of `initial_filters` and `initial_strides` must be the same. "
                f"Received: initial_filters={initial_filters}, "
                f"initial_strides={initial_strides}."
            )
        
        for i, config in enumerate(inception_config):
            if len(config) != 6:
                raise ValueError(
                    "Each inception config should have 6 values: "
                    "[1x1_filters, 3x3_reduce_filters, 3x3_filters, "
                    "5x5_reduce_filters, 5x5_filters, pool_proj_filters]. "
                    f"Received for inception block {i}: {config}"
                )

        data_format = standardize_data_format(data_format)
        bn_axis = -1 if data_format == "channels_last" else 1
        
        # === Functional Model ===
        image_input = layers.Input(shape=image_shape)
        x = image_input  # Intermediate result.
        
        # Initial convolution layers
        for i, (filters, stride) in enumerate(zip(initial_filters, initial_strides)):
            if stride > 1:
                x = layers.ZeroPadding2D(
                    padding=(3, 3) if i == 0 else (1, 1),
                    data_format=data_format,
                    dtype=dtype,
                    name=f"conv{i+1}_pad",
                )(x)
                padding = "valid"
            else:
                padding = "same"
                
            x = layers.Conv2D(
                filters,
                kernel_size=7 if i == 0 else 3,
                strides=stride,
                padding=padding,
                use_bias=False,
                data_format=data_format,
                dtype=dtype,
                name=f"conv{i+1}",
            )(x)
            x = layers.BatchNormalization(
                axis=bn_axis,
                epsilon=1e-5,
                momentum=0.9,
                dtype=dtype,
                name=f"conv{i+1}_bn",
            )(x)
            x = layers.Activation("relu", dtype=dtype, name=f"conv{i+1}_relu")(x)
            
            # Max pooling after first conv layer
            if i == 0:
                x = layers.ZeroPadding2D(
                    (1, 1), data_format=data_format, dtype=dtype, name=f"pool{i+1}_pad"
                )(x)
                x = layers.MaxPooling2D(
                    3,
                    strides=2,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"pool{i+1}_pool",
                )(x)
        
        # Max pooling after initial conv layers
        x = layers.ZeroPadding2D(
            (1, 1), data_format=data_format, dtype=dtype, name="pool2_pad"
        )(x)
        x = layers.MaxPooling2D(
            3,
            strides=2,
            data_format=data_format,
            dtype=dtype,
            name="pool2_pool",
        )(x)
        
        # Inception blocks
        pyramid_outputs = {}
        for i, config in enumerate(inception_config):
            block_level = i // 2 + 3  # Inception blocks start at level 3
            block_name = f"inception_{block_level}{chr(97 + i % 2)}"  # a, b, c, etc.
            
            x = apply_inception_module(
                x,
                config[0],  # 1x1 filters
                config[1],  # 3x3 reduce filters
                config[2],  # 3x3 filters
                config[3],  # 5x5 reduce filters
                config[4],  # 5x5 filters
                config[5],  # pool proj filters
                data_format=data_format,
                dtype=dtype,
                name=block_name,
            )
            
            # Add to pyramid outputs after each 2 inception blocks (each level)
            if i % 2 == 1 or i == len(inception_config) - 1:
                pyramid_level = block_level
                pyramid_outputs[f"P{pyramid_level}"] = x
                
                # Max pooling between inception levels (except after the last one)
                if i < len(inception_config) - 1 and i % 2 == 1:
                    x = layers.ZeroPadding2D(
                        (1, 1), 
                        data_format=data_format, 
                        dtype=dtype, 
                        name=f"pool{pyramid_level}_pad"
                    )(x)
                    x = layers.MaxPooling2D(
                        3,
                        strides=2,
                        data_format=data_format,
                        dtype=dtype,
                        name=f"pool{pyramid_level}_pool",
                    )(x)
            
            # Add auxiliary classifiers if requested (typically after 4a and 4d)
            if aux_classifiers and (
                (block_level == 4 and i % 2 == 0) or  # After 4a
                (block_level == 4 and i % 2 == 1)     # After 4d
            ):
                aux_output = apply_auxiliary_classifier(
                    x,
                    data_format=data_format,
                    dtype=dtype,
                    name=f"aux_{block_name}",
                )
                # In backbone mode, we don't actually use these outputs
        
        # Apply global average pooling at the end
        x = layers.GlobalAveragePooling2D(
            data_format=data_format, dtype=dtype, name="avg_pool"
        )(x)

        super().__init__(
            inputs=image_input,
            outputs=x,  # Main output is the global average pooled features
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.initial_filters = initial_filters
        self.initial_strides = initial_strides
        self.inception_config = inception_config
        self.aux_classifiers = aux_classifiers
        self.image_shape = image_shape
        self.pyramid_outputs = pyramid_outputs
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_filters": self.initial_filters,
                "initial_strides": self.initial_strides,
                "inception_config": self.inception_config,
                "aux_classifiers": self.aux_classifiers,
                "image_shape": self.image_shape,
            }
        )
        return config


def apply_inception_module(
    x,
    filters_1x1,
    filters_3x3_reduce,
    filters_3x3,
    filters_5x5_reduce,
    filters_5x5,
    filters_pool_proj,
    data_format=None,
    dtype=None,
    name=None,
):
    """Applies an Inception module.
    
    The Inception module processes input at different scales simultaneously 
    using multiple filter sizes in parallel.
    
    Args:
        x: Tensor. The input tensor to pass through the inception module.
        filters_1x1: int. The number of filters in the 1x1 convolution branch.
        filters_3x3_reduce: int. The number of filters in the 3x3 reduce convolution.
        filters_3x3: int. The number of filters in the 3x3 convolution.
        filters_5x5_reduce: int. The number of filters in the 5x5 reduce convolution.
        filters_5x5: int. The number of filters in the 5x5 convolution.
        filters_pool_proj: int. The number of filters in the pool projection.
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
            (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the module.
            
    Returns:
        The output tensor for the Inception module.
    """
    data_format = data_format or keras.config.image_data_format()
    bn_axis = -1 if data_format == "channels_last" else 1
    
    # 1x1 branch
    branch1 = layers.Conv2D(
        filters_1x1,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_1x1_conv",
    )(x)
    branch1 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_1x1_bn",
    )(branch1)
    branch1 = layers.Activation("relu", dtype=dtype, name=f"{name}_1x1_relu")(branch1)
    
    # 3x3 branch
    branch2 = layers.Conv2D(
        filters_3x3_reduce,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_3x3_reduce_conv",
    )(x)
    branch2 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_3x3_reduce_bn",
    )(branch2)
    branch2 = layers.Activation("relu", dtype=dtype, name=f"{name}_3x3_reduce_relu")(branch2)
    branch2 = layers.Conv2D(
        filters_3x3,
        3,
        padding="same",
        use_bias=False,
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_3x3_conv",
    )(branch2)
    branch2 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_3x3_bn",
    )(branch2)
    branch2 = layers.Activation("relu", dtype=dtype, name=f"{name}_3x3_relu")(branch2)
    
    # 5x5 branch
    branch3 = layers.Conv2D(
        filters_5x5_reduce,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_5x5_reduce_conv",
    )(x)
    branch3 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_5x5_reduce_bn",
    )(branch3)
    branch3 = layers.Activation("relu", dtype=dtype, name=f"{name}_5x5_reduce_relu")(branch3)
    branch3 = layers.Conv2D(
        filters_5x5,
        5,
        padding="same",
        use_bias=False,
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_5x5_conv",
    )(branch3)
    branch3 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_5x5_bn",
    )(branch3)
    branch3 = layers.Activation("relu", dtype=dtype, name=f"{name}_5x5_relu")(branch3)
    
    # Pool branch
    branch4 = layers.MaxPooling2D(
        3,
        strides=1,
        padding="same",
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_pool",
    )(x)
    branch4 = layers.Conv2D(
        filters_pool_proj,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_pool_proj_conv",
    )(branch4)
    branch4 = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_pool_proj_bn",
    )(branch4)
    branch4 = layers.Activation("relu", dtype=dtype, name=f"{name}_pool_proj_relu")(branch4)
    
    # Concatenate all branches
    return layers.Concatenate(
        axis=bn_axis, dtype=dtype, name=f"{name}_concat"
    )([branch1, branch2, branch3, branch4])


def apply_auxiliary_classifier(
    x,
    data_format=None,
    dtype=None,
    name=None,
):
    """Applies an auxiliary classifier.
    
    This function implements the auxiliary classifiers used in GoogleNet to help
    with the vanishing gradient problem during training.
    
    Args:
        x: Tensor. The input tensor to pass through the auxiliary classifier.
        data_format: `None` or str. the ordering of the dimensions in the
            inputs. Can be `"channels_last"`
            (`(batch_size, height, width, channels)`) or`"channels_first"`
            (`(batch_size, channels, height, width)`).
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The dtype
            to use for the models computations and weights.
        name: str. A prefix for the layer names used in the classifier.
            
    Returns:
        The output tensor for the auxiliary classifier.
    """
    data_format = data_format or keras.config.image_data_format()
    bn_axis = -1 if data_format == "channels_last" else 1
    
    x = layers.AveragePooling2D(
        5,
        strides=3,
        padding="valid",
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_avg_pool",
    )(x)
    x = layers.Conv2D(
        128,
        1,
        padding="same",
        use_bias=False,
        data_format=data_format,
        dtype=dtype,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        epsilon=1e-5,
        momentum=0.9,
        dtype=dtype,
        name=f"{name}_bn",
    )(x)
    x = layers.Activation("relu", dtype=dtype, name=f"{name}_relu")(x)
    
    # Note: In a backbone, we typically don't use the classification layers
    # These would normally include flatten, dense (1024), dropout, and final dense layer
    
    return x