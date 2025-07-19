import keras

from keras_hub.src.models.hgnetv2.hgnetv2_layers import HGNetV2Stage


@keras.saving.register_keras_serializable(package="keras_hub")
class HGNetV2Encoder(keras.layers.Layer):
    """This class represents the encoder of the HGNetV2 model.

    This class implements the encoder part of the HGNetV2 architecture, which
    consists of multiple stages. Each stage is an instance of `HGNetV2Stage`,
    and the encoder processes the input through these stages sequentially,
    collecting the hidden states at each stage.

    Args:
        stage_in_channels: A list of integers, specifying the input channels
            for each stage.
        stage_mid_channels: A list of integers, specifying the mid channels for
            each stage.
        stage_out_channels: A list of integers, specifying the output channels
            for each stage.
        stage_num_blocks: A list of integers, specifying the number of blocks
            in each stage.
        stage_num_of_layers: A list of integers, specifying the number of
            layers in each block of each stage.
        apply_downsample: A list of booleans or integers, indicating whether to
            downsample in each stage.
        use_lightweight_conv_block: A list of booleans, indicating whether to
            use HGNetV2 lightweight convolutional blocks in each stage.
        stage_kernel_size: A list of integers or tuples, specifying the kernel
            size for each stage.
        use_learnable_affine_block: A boolean, indicating whether to use
            learnable affine transformations in the blocks.
        data_format: `None` or str. If specified, either `"channels_last"` or
            `"channels_first"`. The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch_size, height, width, channels)` while `"channels_first"`
            corresponds to inputs with shape `(batch_size, channels, height,
            width)`. It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`. If you never set it,
            then it will be `"channels_last"`.
        channel_axis: int, the axis that represents the channels.
        **kwargs: Additional keyword arguments passed to the parent class.
    """

    def __init__(
        self,
        stage_in_channels,
        stage_mid_channels,
        stage_out_channels,
        stage_num_blocks,
        stage_num_of_layers,
        apply_downsample,
        use_lightweight_conv_block,
        stage_kernel_size,
        use_learnable_affine_block,
        data_format=None,
        channel_axis=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage_in_channels = stage_in_channels
        self.stage_mid_channels = stage_mid_channels
        self.stage_out_channels = stage_out_channels
        self.stage_num_blocks = stage_num_blocks
        self.stage_num_of_layers = stage_num_of_layers
        self.apply_downsample = apply_downsample
        self.use_lightweight_conv_block = use_lightweight_conv_block
        self.stage_kernel_size = stage_kernel_size
        self.use_learnable_affine_block = use_learnable_affine_block
        self.data_format = data_format
        self.channel_axis = channel_axis

        self.stages_list = []
        for stage_idx in range(len(self.stage_in_channels)):
            stage_layer = HGNetV2Stage(
                stage_in_channels=self.stage_in_channels,
                stage_mid_channels=self.stage_mid_channels,
                stage_out_channels=self.stage_out_channels,
                stage_num_blocks=self.stage_num_blocks,
                stage_num_of_layers=self.stage_num_of_layers,
                apply_downsample=self.apply_downsample,
                use_lightweight_conv_block=self.use_lightweight_conv_block,
                stage_kernel_size=self.stage_kernel_size,
                use_learnable_affine_block=self.use_learnable_affine_block,
                stage_index=stage_idx,
                data_format=self.data_format,
                channel_axis=self.channel_axis,
                drop_path=0.0,
                name=f"{self.name}_stage_{stage_idx}"
                if self.name
                else f"stage_{stage_idx}",
                dtype=self.dtype,
            )
            self.stages_list.append(stage_layer)

    def build(self, input_shape):
        super().build(input_shape)
        current_input_shape = input_shape
        for stage_keras_layer in self.stages_list:
            stage_keras_layer.build(current_input_shape)
            current_input_shape = stage_keras_layer.compute_output_shape(
                current_input_shape
            )

    def call(
        self,
        hidden_state,
        training=None,
    ):
        all_hidden_states_list = []
        current_hidden_state = hidden_state
        for stage_keras_layer in self.stages_list:
            all_hidden_states_list.append(current_hidden_state)
            current_hidden_state = stage_keras_layer(
                current_hidden_state, training=training
            )
        all_hidden_states_list.append(current_hidden_state)
        return tuple(all_hidden_states_list)

    def compute_output_shape(self, input_shape):
        current_shape = input_shape
        all_hidden_shapes = [input_shape]
        for stage_keras_layer in self.stages_list:
            current_shape = stage_keras_layer.compute_output_shape(
                current_shape
            )
            all_hidden_shapes.append(current_shape)
        return tuple(all_hidden_shapes)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stage_in_channels": self.stage_in_channels,
                "stage_mid_channels": self.stage_mid_channels,
                "stage_out_channels": self.stage_out_channels,
                "stage_num_blocks": self.stage_num_blocks,
                "stage_num_of_layers": self.stage_num_of_layers,
                "apply_downsample": self.apply_downsample,
                "use_lightweight_conv_block": self.use_lightweight_conv_block,
                "stage_kernel_size": self.stage_kernel_size,
                "use_learnable_affine_block": self.use_learnable_affine_block,
                "data_format": self.data_format,
                "channel_axis": self.channel_axis,
            }
        )
        return config
