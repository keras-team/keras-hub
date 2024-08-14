# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.resnet.resnet_backbone import ResNetBackbone


@keras_nlp_export("keras_nlp.models.ResNetFeaturePyramidBackbone")
class ResNetFeaturePyramidBackbone(ResNetBackbone):
    """ResNet and ResNetV2 core network with hyperparameters.

    This class implements a ResNet backbone as described in [Deep Residual
    Learning for Image Recognition](https://arxiv.org/abs/1512.03385)(
    CVPR 2016), [Identity Mappings in Deep Residual Networks](
    https://arxiv.org/abs/1603.05027)(ECCV 2016) and [ResNet strikes back: An
    improved training procedure in timm](https://arxiv.org/abs/2110.00476)(
    NeurIPS 2021 Workshop).

    The difference in ResNet and ResNetV2 rests in the structure of their
    individual building blocks. In ResNetV2, the batch normalization and
    ReLU activation precede the convolution layers, as opposed to ResNet where
    the batch normalization and ReLU activation are applied after the
    convolution layers.

    Args:
        stackwise_num_filters: list of ints. The number of filters for each
            stack.
        stackwise_num_blocks: list of ints. The number of blocks for each stack.
        stackwise_num_strides: list of ints. The number of strides for each
            stack.
        block_type: str. The block type to stack. One of `"basic_block"` or
            `"bottleneck_block"`. Use `"basic_block"` for ResNet18 and ResNet34.
            Use `"bottleneck_block"` for ResNet50, ResNet101 and ResNet152.
        use_pre_activation: boolean. Whether to use pre-activation or not.
            `True` for ResNetV2, `False` for ResNet.
        include_rescaling: boolean. If `True`, rescale the input using
            `Rescaling(1 / 255.0)` layer. If `False`, do nothing. Defaults to
            `True`.
        input_image_shape: tuple. The input shape without the batch size.
            Defaults to `(None, None, 3)`.
        pooling: `None` or str. Pooling mode for feature extraction. Defaults
            to `"avg"`.
            - `None` means that the output of the model will be the 4D tensor
                from the last convolutional block.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional block, resulting in a 2D
                tensor.
            - `max` means that global max pooling will be applied to the
                output of the last convolutional block, resulting in a 2D
                tensor.
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
            to use for the models computations and weights.
        output_keys: `None` or list of strs. Keys to use for the outputs of
            the model. Defaults to `None`, meaning that all
            `self.pyramid_outputs` will be used.

    Examples:
    ```python
    input_data = np.ones((2, 224, 224, 3), dtype="float32")

    # Pretrained ResNet feature pyramid backbone.
    model = keras_nlp.models.ResNetFeaturePyramidBackbone.from_preset(
        "resnet50"
    )
    model(input_data)

    # Randomly initialized ResNetV2 feature pyramidbackbone with a custom config.
    model = keras_nlp.models.ResNetBackbone(
        stackwise_num_filters=[64, 64, 64],
        stackwise_num_blocks=[2, 2, 2],
        stackwise_num_strides=[1, 2, 2],
        block_type="basic_block",
        use_pre_activation=True,
        pooling="avg",
        output_keys=["P3", "P4"]
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        stackwise_num_filters,
        stackwise_num_blocks,
        stackwise_num_strides,
        block_type,
        use_pre_activation=False,
        include_rescaling=True,
        input_image_shape=(None, None, 3),
        pooling="avg",
        data_format=None,
        dtype=None,
        output_keys=None,
        **kwargs,
    ):
        super().__init__(
            stackwise_num_filters,
            stackwise_num_blocks,
            stackwise_num_strides,
            block_type,
            use_pre_activation,
            include_rescaling,
            input_image_shape,
            pooling,
            data_format,
            dtype,
            **kwargs,
        )

        output_keys = output_keys or self.pyramid_outputs.keys()
        outputs = {}
        for k in output_keys:
            try:
                output = self.pyramid_outputs[k]
            except KeyError:
                raise KeyError(
                    f"'{k}' not in self.pyramid_outputs. The available keys "
                    f"are: {list(self.pyramid_outputs.keys())}"
                )
            outputs[k] = output

        super(ResNetBackbone, self).__init__(
            inputs=self.inputs, outputs=outputs, dtype=dtype, **kwargs
        )
