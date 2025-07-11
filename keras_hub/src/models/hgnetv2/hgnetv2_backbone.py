import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_encoder import HGNetV2Encoder
from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2Embeddings
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.HGNetV2Backbone")
class HGNetV2Backbone(Backbone):
    """This class represents a Keras Backbone of the HGNetV2 model.

    This class implements an HGNetV2 backbone architecture, a convolutional
    neural network (CNN) optimized for GPU efficiency. HGNetV2 is frequently
    used as a lightweight CNN backbone in object detection pipelines like
    RT-DETR and YOLO variants, delivering strong performance on classification
    and detection tasks, with speed-ups and accuracy gains compared to larger
    CNN backbones.

    Args:
        depths: list of ints, the number of blocks in each stage.
        embedding_size: int, the size of the embedding layer.
        hidden_sizes: list of ints, the sizes of the hidden layers.
        stem_channels: list of ints, the channels for the stem part.
        hidden_act: str, the activation function for hidden layers.
        use_learnable_affine_block: bool, whether to use learnable affine
            transformations.
        stackwise_stage_filters: list of tuples, where each tuple contains
            configuration for a stage: (stage_in_channels, stage_mid_channels,
            stage_out_channels, stage_num_blocks, stage_num_of_layers,
            stage_kernel_size).
            - stage_in_channels: int, input channels for the stage
            - stage_mid_channels: int, middle channels for the stage
            - stage_out_channels: int, output channels for the stage
            - stage_num_blocks: int, number of blocks in the stage
            - stage_num_of_layers: int, number of layers in each block
            - stage_kernel_size: int, kernel size for the stage
        apply_downsample: list of bools, whether to downsample in each stage.
        use_lightweight_conv_block: list of bools, whether to use HGNetV2
            lightweight convolutional blocks in each stage.
        image_shape: tuple, the shape of the input image without the batch size.
            Defaults to `(None, None, 3)`.
        data_format: `None` or str, the data format ('channels_last' or
            'channels_first'). If not specified, defaults to the
            `image_data_format` value in your Keras config.
        out_features: list of str or `None`, the names of the output features to
            return. If `None`, returns all available features from all stages.
            Defaults to `None`.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`, the data
            type for computations and weights.

    Examples:
    ```python
    import numpy as np
    from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
    input_data = np.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone.
    model = keras_hub.models.HGNetV2Backbone.from_preset(
        "hgnetv2_b5_ssld_stage2_ft_in1k"
    )
    model(input_data)

    # Randomly initialized backbone with a custom config.
    model = HGNetV2Backbone(
        depths=[1, 2, 4],
        embedding_size=32,
        hidden_sizes=[64, 128, 256],
        stem_channels=[3, 16, 32],
        hidden_act="relu",
        use_learnable_affine_block=False,
        stackwise_stage_filters=[
            (32, 16, 64, 1, 1, 3),     # Stage 0
            (64, 32, 128, 2, 1, 3),    # Stage 1
            (128, 64, 256, 4, 1, 3),   # Stage 2
        ],
        apply_downsample=[False, True, True],
        use_lightweight_conv_block=[False, False, False],
        image_shape=(224, 224, 3),
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        depths,
        embedding_size,
        hidden_sizes,
        stem_channels,
        hidden_act,
        use_learnable_affine_block,
        stackwise_stage_filters,
        apply_downsample,
        use_lightweight_conv_block,
        image_shape=(None, None, 3),
        data_format=None,
        out_features=None,
        dtype=None,
        **kwargs,
    ):
        name = kwargs.get("name", None)
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        self.image_shape = image_shape
        (
            stage_in_channels,
            stage_mid_channels,
            stage_out_filters,
            stage_num_blocks,
            stage_num_of_layers,
            stage_kernel_size,
        ) = zip(*stackwise_stage_filters)

        # === Layers ===
        self.embedder_layer = HGNetV2Embeddings(
            stem_channels=stem_channels,
            hidden_act=hidden_act,
            use_learnable_affine_block=use_learnable_affine_block,
            data_format=data_format,
            channel_axis=channel_axis,
            name=f"{name}_embedder" if name else "embedder",
            dtype=dtype,
        )
        self.encoder_layer = HGNetV2Encoder(
            stage_in_channels=stage_in_channels,
            stage_mid_channels=stage_mid_channels,
            stage_out_channels=stage_out_filters,
            stage_num_blocks=stage_num_blocks,
            stage_num_of_layers=stage_num_of_layers,
            apply_downsample=apply_downsample,
            use_lightweight_conv_block=use_lightweight_conv_block,
            stage_kernel_size=stage_kernel_size,
            use_learnable_affine_block=use_learnable_affine_block,
            data_format=data_format,
            channel_axis=channel_axis,
            name=f"{name}_encoder" if name else "encoder",
            dtype=dtype,
        )
        self.stage_names = ["stem"] + [
            f"stage{i + 1}" for i in range(len(stackwise_stage_filters))
        ]
        self.out_features = (
            out_features if out_features is not None else self.stage_names
        )

        # === Functional Model ===
        pixel_values = keras.layers.Input(
            shape=image_shape, name="pixel_values_input"
        )
        embedding_output = self.embedder_layer(pixel_values)
        all_encoder_hidden_states_tuple = self.encoder_layer(embedding_output)
        feature_maps_output = {
            stage_name: all_encoder_hidden_states_tuple[idx]
            for idx, stage_name in enumerate(self.stage_names)
            if stage_name in self.out_features
        }
        super().__init__(
            inputs=pixel_values, outputs=feature_maps_output, **kwargs
        )

        # === Config ===
        self.depths = depths
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.stem_channels = stem_channels
        self.hidden_act = hidden_act
        self.use_learnable_affine_block = use_learnable_affine_block
        self.stackwise_stage_filters = stackwise_stage_filters
        self.apply_downsample = apply_downsample
        self.use_lightweight_conv_block = use_lightweight_conv_block
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "depths": self.depths,
                "embedding_size": self.embedding_size,
                "hidden_sizes": self.hidden_sizes,
                "stem_channels": self.stem_channels,
                "hidden_act": self.hidden_act,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "stackwise_stage_filters": self.stackwise_stage_filters,
                "apply_downsample": self.apply_downsample,
                "use_lightweight_conv_block": self.use_lightweight_conv_block,
                "image_shape": self.image_shape,
                "out_features": self.out_features,
                "data_format": self.data_format,
            }
        )
        return config
