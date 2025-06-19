import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.hgnetv2.hgnetv2_encoder import HGNetV2Encoder
from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2Embeddings
from keras_hub.src.utils.keras_utils import standardize_data_format


@keras_hub_export("keras_hub.models.HGNetV2Backbone")
class HGNetV2Backbone(Backbone):
    """This class represents a Keras Backbone of the HGNetV2 model.

    This class implements an HGNetV2 backbone architecture.

    Args:
        initializer_range: float, the range for initializing weights.
        depths: list of ints, the number of blocks in each stage.
        embedding_size: int, the size of the embedding layer.
        hidden_sizes: list of ints, the sizes of the hidden layers.
        stem_channels: list of ints, the channels for the stem part.
        hidden_act: str, the activation function for hidden layers.
        use_learnable_affine_block: bool, whether to use learnable affine
            transformations.
        num_channels: int, the number of channels in the input image.
        stage_in_channels: list of ints, the input channels for each stage.
        stage_mid_channels: list of ints, the middle channels for each stage.
        stage_out_channels: list of ints, the output channels for each stage.
        stage_num_blocks: list of ints, the number of blocks in each stage.
        stage_numb_of_layers: list of ints, the number of layers in each block.
        stage_downsample: list of bools, whether to downsample in each stage.
        stage_light_block: list of bools, whether to use light blocks in each
            stage.
        stage_kernel_size: list of ints, the kernel sizes for each stage.
        image_shape: tuple, the shape of the input image without the batch size.
            Defaults to `(None, None, 3)`.
        data_format: `None` or str, the data format ('channels_last' or
            'channels_first'). If not specified, defaults to the
            `image_data_format` value in your Keras config.
        dtype: `None` or str or `keras.mixed_precision.DTypePolicy`, the data
            type for computations and weights.

    Examples:
    ```python
    import numpy as np
    from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
    input_data = np.ones(shape=(8, 224, 224, 3))

    # Pretrained backbone.
    model = keras_hub.models.HGNetV2Backbone.from_preset(
        "hgnetv2_b5.ssld_stage2_ft_in1k"
    )
    model(input_data)

    # Randomly initialized backbone with a custom config.
    model = HGNetV2Backbone(
        initializer_range=0.02,
        depths=[1, 2, 4],
        embedding_size=32,
        hidden_sizes=[64, 128, 256],
        stem_channels=[3, 16, 32],
        hidden_act="relu",
        use_learnable_affine_block=False,
        num_channels=3,
        stage_in_channels=[32, 64, 128],
        stage_mid_channels=[16, 32, 64],
        stage_out_channels=[64, 128, 256],
        stage_num_blocks=[1, 2, 4],
        stage_numb_of_layers=[1, 1, 1],
        stage_downsample=[False, True, True],
        stage_light_block=[False, False, False],
        stage_kernel_size=[3, 3, 3],
        image_shape=(224, 224, 3),
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        initializer_range,
        depths,
        embedding_size,
        hidden_sizes,
        stem_channels,
        hidden_act,
        use_learnable_affine_block,
        num_channels,
        stage_in_channels,
        stage_mid_channels,
        stage_out_channels,
        stage_num_blocks,
        stage_numb_of_layers,
        stage_downsample,
        stage_light_block,
        stage_kernel_size,
        image_shape=(None, None, 3),
        data_format=None,
        dtype=None,
        **kwargs,
    ):
        name = kwargs.get("name", None)
        data_format = standardize_data_format(data_format)
        channel_axis = -1 if data_format == "channels_last" else 1
        self.image_shape = image_shape

        # === Layers ===
        self.embedder_layer = HGNetV2Embeddings(
            stem_channels=stem_channels,
            hidden_act=hidden_act,
            use_learnable_affine_block=use_learnable_affine_block,
            num_channels=num_channels,
            data_format=data_format,
            channel_axis=channel_axis,
            name=f"{name}_embedder" if name else "embedder",
            dtype=dtype,
        )
        self.encoder_layer = HGNetV2Encoder(
            stage_in_channels=stage_in_channels,
            stage_mid_channels=stage_mid_channels,
            stage_out_channels=stage_out_channels,
            stage_num_blocks=stage_num_blocks,
            stage_numb_of_layers=stage_numb_of_layers,
            stage_downsample=stage_downsample,
            stage_light_block=stage_light_block,
            stage_kernel_size=stage_kernel_size,
            use_learnable_affine_block=use_learnable_affine_block,
            data_format=data_format,
            channel_axis=channel_axis,
            name=f"{name}_encoder" if name else "encoder",
            dtype=dtype,
        )
        self.stage_names = [f"stage{i}" for i in range(len(stage_in_channels))]
        self.out_features = self.stage_names

        # === Functional Model ===
        pixel_values = keras.layers.Input(
            shape=image_shape, name="pixel_values_input"
        )
        embedding_output = self.embedder_layer(pixel_values)
        all_encoder_hidden_states_tuple = self.encoder_layer(embedding_output)
        feature_maps_output = {
            stage_name: all_encoder_hidden_states_tuple[idx + 1]
            for idx, stage_name in enumerate(self.stage_names)
            if stage_name in self.out_features
        }
        super().__init__(
            inputs=pixel_values, outputs=feature_maps_output, **kwargs
        )

        # === Config ===
        self.initializer_range = initializer_range
        self.depths = depths
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.stem_channels = stem_channels
        self.hidden_act = hidden_act
        self.use_learnable_affine_block = use_learnable_affine_block
        self.num_channels = num_channels
        self.stage_in_channels = stage_in_channels
        self.stage_mid_channels = stage_mid_channels
        self.stage_out_channels = stage_out_channels
        self.stage_num_blocks = stage_num_blocks
        self.stage_numb_of_layers = stage_numb_of_layers
        self.stage_downsample = stage_downsample
        self.stage_light_block = stage_light_block
        self.stage_kernel_size = stage_kernel_size
        self.data_format = data_format

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initializer_range": self.initializer_range,
                "depths": self.depths,
                "embedding_size": self.embedding_size,
                "hidden_sizes": self.hidden_sizes,
                "stem_channels": self.stem_channels,
                "hidden_act": self.hidden_act,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "num_channels": self.num_channels,
                "stage_in_channels": self.stage_in_channels,
                "stage_mid_channels": self.stage_mid_channels,
                "stage_out_channels": self.stage_out_channels,
                "stage_num_blocks": self.stage_num_blocks,
                "stage_numb_of_layers": self.stage_numb_of_layers,
                "stage_downsample": self.stage_downsample,
                "stage_light_block": self.stage_light_block,
                "stage_kernel_size": self.stage_kernel_size,
                "image_shape": self.image_shape,
                "data_format": self.data_format,
            }
        )
        return config
