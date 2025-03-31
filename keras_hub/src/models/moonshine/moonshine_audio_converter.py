import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.audio_converter import AudioConverter
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_layers import (
    moonshine_kernel_initializer,
)
from keras_hub.src.utils.keras_utils import clone_initializer


@keras_hub_export("keras_hub.layers.MoonshineAudioConverter")
class MoonshineAudioConverter(AudioConverter):
    """Moonshine preprocessor and audio converter layer.

    This layer processes raw audio waveforms for the Moonshine ASR model. Audio
    is formatted as a batched tensor at a 16kHz sample rate and validated for
    length (0.1 to 64 seconds). The layer downsamples and extracts key features
    from the audio signal through a series of convolutional operations,
    normalization, and nonlinear activations.

    Args:
        filter_dim: int. The number of filters for the first convolutional
            layer. This influences the dimensionality of the feature extraction
            pipeline and determines the richness of the audio representation.
        sampling_rate: int, optional. The audio sampling rate in Hz. Defaults to
            16,000.
        padding_value: float, optional. The value for padding. Defaults to 0.0.
        do_normalize: bool, optional. Whether to normalize inputs. Defaults to
            False.
        return_attention_mask: bool, optional. Whether to return an attention
            mask. Defaults to True.
        initializer_range: float, optional. The standard deviation for kernel
            initialization. Defaults to 0.02.
        **kwargs: Additional keyword arguments passed to the base AudioConverter
            class for customizing the underlying preprocessing behavior.

    Examples:
    ```python
    import keras
    from keras_hub.layers import MoonshineAudioConverter

    # Create a dummy audio input (1 second at 16kHz).
    dummy_audio = keras.ops.convert_to_tensor(
        [[0.1] * 16000],
        dtype="float32"
    )
    dummy_audio = keras.ops.expand_dims(dummy_audio, axis=-1)

    # Initialize the preprocessor.
    preprocessor = MoonshineAudioConverter(filter_dim=256)

    # Process the audio.
    features = preprocessor(dummy_audio)

    # Output shapes.
    print(features["input_values"].shape)  # Expected: (1, 40, 256)
    print(features["attention_mask"].shape)  # Expected: (1, 40)
    ```
    """

    # References:
    # Defined and formulated based on the Hugging Face implementation of the
    # Wav2Vec2FeatureExtractor class (https://github.com/huggingface/transformers/blob/66f29aaaf55c8fe0c3dbcd24beede2ca4effac56/src/transformers/models/wav2vec2/feature_extraction_wav2vec2.py#L31-L243)
    # and the convolutional layer structure defined in the UsefulSensors
    # implementation of the AudioPreprocessor class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L6-L32).

    backbone_cls = MoonshineBackbone

    def __init__(
        self,
        filter_dim,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=False,
        return_attention_mask=True,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filter_dim = filter_dim
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.do_normalize = do_normalize
        self.return_attention_mask = return_attention_mask
        self.initializer_range = initializer_range
        self.kernel_initializer = moonshine_kernel_initializer(
            initializer_range=initializer_range
        )

        self.conv1 = keras.layers.Conv1D(
            filters=filter_dim,
            kernel_size=127,
            strides=64,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
        )
        self.tanh = keras.layers.Activation("tanh")
        self.group_norm = keras.layers.GroupNormalization(
            groups=1,
            axis=-1,
            epsilon=1e-5,
        )
        self.conv2 = keras.layers.Conv1D(
            filters=2 * filter_dim,
            kernel_size=7,
            strides=3,
            padding="valid",
            kernel_initializer=clone_initializer(self.kernel_initializer),
        )
        self.gelu1 = keras.layers.Activation("gelu")
        self.conv3 = keras.layers.Conv1D(
            filters=filter_dim,
            kernel_size=3,
            strides=2,
            padding="valid",
            kernel_initializer=clone_initializer(self.kernel_initializer),
        )
        self.gelu2 = keras.layers.Activation("gelu")

    def build(self, input_shape):
        self.conv1.build((None, None, 1))
        self.group_norm.build((None, None, self.filter_dim))
        self.conv2.build((None, None, self.filter_dim))
        self.conv3.build((None, None, 2 * self.filter_dim))
        self.built = True

    def call(
        self,
        inputs,
        sampling_rate=None,
        padding=None,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        # Standardize inputs.
        inputs = keras.ops.convert_to_tensor(inputs)
        # Validate sampling rate.
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate {self.sampling_rate}, got "
                f"{sampling_rate}"
            )

        # Ensure inputs are (batch_size, time_steps, 1).
        if keras.ops.ndim(inputs) == 2:
            inputs = keras.ops.expand_dims(inputs, axis=-1)
        elif keras.ops.ndim(inputs) != 3 or keras.ops.shape(inputs)[-1] != 1:
            raise ValueError(
                "Inputs must be mono audio: (batch_size, time_steps, 1)"
            )

        # Get original length and validate duration.
        original_length = keras.ops.shape(inputs)[1]
        # Handle padding.
        if padding == "longest":
            max_length = original_length
        elif padding == "max_length" and max_length is None:
            max_length = original_length
        if max_length is not None:
            if pad_to_multiple_of:
                max_length = (
                    (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of
                ) * pad_to_multiple_of
            if original_length < max_length:
                padding_amount = max_length - original_length
                inputs = keras.ops.pad(
                    inputs,
                    [(0, 0), (0, padding_amount), (0, 0)],
                    constant_values=self.padding_value,
                )

        # Normalize if enabled.
        if self.do_normalize:
            mean = keras.ops.mean(inputs, axis=1, keepdims=True)
            var = keras.ops.var(inputs, axis=1, keepdims=True)
            inputs = (inputs - mean) / keras.ops.sqrt(var + 1e-7)

        # Apply convolutional feature extraction.
        x = self.conv1(inputs)
        x = self.tanh(x)
        x = self.group_norm(x)
        x = self.conv2(x)
        x = self.gelu1(x)
        x = self.conv3(x)
        features = self.gelu2(x)

        # Generate attention mask.
        output_length = keras.ops.shape(features)[1]
        attention_mask = None
        if self.return_attention_mask:
            # Calculate mask length through the network's downsampling ops.
            # Step 1: First conv layer (conv1).
            conv1_out = (original_length - 127 + 1) / 64
            # Step 2: Second conv layer (conv2).
            conv2_out = (conv1_out - 7 + 1) / 3
            # Step 3: Third conv layer (conv3).
            conv3_out = (conv2_out - 3 + 1) / 2

            # Apply ceil() to get the final mask length as an int.
            mask_length = keras.ops.cast(
                keras.ops.ceil(keras.ops.cast(conv3_out, "float32")), "int32"
            )
            # Broadcast the mask length to match the batch size.
            batch_size = keras.ops.shape(inputs)[0]
            mask_length = keras.ops.broadcast_to(mask_length, [batch_size])
            indices = keras.ops.arange(output_length, dtype="int32")
            attention_mask = keras.ops.cast(
                indices[None, :] < mask_length[:, None], dtype="int32"
            )

        output = {"input_values": features}
        if attention_mask is not None:
            output["attention_mask"] = attention_mask

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_dim": self.filter_dim,
                "sampling_rate": self.sampling_rate,
                "padding_value": self.padding_value,
                "do_normalize": self.do_normalize,
                "return_attention_mask": self.return_attention_mask,
                "initializer_range": self.initializer_range,
            }
        )
        return config
