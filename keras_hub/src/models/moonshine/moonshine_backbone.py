import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.moonshine.moonshine_decoder import (
    MoonshineDecoderBlock,
)
from keras_hub.src.models.moonshine.moonshine_encoder import (
    MoonshineEncoderBlock,
)
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineRotaryEmbedding,
)
from keras_hub.src.models.moonshine.moonshine_layers import (
    moonshine_kernel_initializer,
)


def compute_output_lengths(input_lengths):
    lengths = keras.ops.cast(input_lengths, "float32")
    lengths = keras.ops.floor((lengths - 127) / 64) + 1
    lengths = keras.ops.floor((lengths - 7) / 3) + 1
    lengths = keras.ops.floor((lengths - 3) / 2) + 1
    return keras.ops.maximum(keras.ops.cast(lengths, "int32"), 0)


@keras.saving.register_keras_serializable(package="keras_hub")
class ComputeAttentionMask(keras.layers.Layer):
    def call(self, features_for_shape, output_lengths):
        max_output_length = keras.ops.shape(features_for_shape)[1]
        indices = keras.ops.arange(max_output_length, dtype="int32")
        attention_mask = indices[None, :] < output_lengths[:, None]
        attention_mask = keras.ops.cast(attention_mask, "bool")
        return attention_mask

    def compute_output_shape(self, input_shapes):
        batch_dim = None
        if isinstance(input_shapes, (list, tuple)) and len(input_shapes) > 0:
            features_shape = input_shapes[0]
            if (
                isinstance(features_shape, (list, tuple))
                and len(features_shape) > 0
            ):
                batch_dim = features_shape[0]
        return (batch_dim, None)


@keras.saving.register_keras_serializable(package="keras_hub")
class Arange(keras.layers.Layer):
    def call(self, inputs):
        sequence_length = keras.ops.shape(inputs)[1]
        return keras.ops.arange(sequence_length, dtype="int32")


@keras_hub_export("keras_hub.models.MoonshineBackbone")
class MoonshineBackbone(Backbone):
    """Moonshine backbone with integrated audio feature extraction.

    This class implements an encoder-decoder backbone, as used in the Moonshine
    ASR system. It includes initial convolutional layers for audio feature
    extraction followed by `MoonshineEncoderBlock` instances for processing
    these features and `MoonshineDecoderBlock` instances for generating output
    sequences.

    Args:
        vocabulary_size: int. The size of the vocabulary for the embedding
            layers.
        filter_dim: int. The number of filters for the initial convolutional
            feature extractor layers. Typically matches `hidden_dim`.
        encoder_num_layers: int. The number of stacked encoder blocks.
        decoder_num_layers: int. The number of stacked decoder blocks.
        hidden_dim: int. The dimensionality of the model's hidden
            representations and embeddings.
        intermediate_dim: int. The dimensionality of the intermediate
            representations in feedforward networks.
        encoder_num_heads: int. The number of attention heads in the encoder's
            multi-head attention.
        decoder_num_heads: int. The number of attention heads in the decoder's
            multi-head attention.
        feedforward_expansion_factor: int, optional. A multiplier applied to
            `intermediate_dim` to determine the total width of the feedforward
            network. Defaults to 4.
        encoder_use_swiglu_activation: bool, optional. When True, uses SwiGLU
            in the encoder feedforward network. Defaults to False.
        decoder_use_swiglu_activation: bool, optional. When True, uses SwiGLU
            in the decoder feedforward network. Defaults to True.
        max_position_embeddings: int, optional. The maximum sequence length for
            position embeddings. Defaults to 2048.
        pad_head_dim_to_multiple_of: int, optional. If specified, pads the head
            dimension to be a multiple of this value for performance
            optimization. Defaults to None.
        partial_rotary_factor: float, optional. The fraction of dimensions to
            apply rotary position embeddings to. Defaults to 0.62.
        dropout: float, optional. The dropout probability for input dropout
            layers. Defaults to 0.0.
        initializer_range: float, optional. The standard deviation of the
            truncated normal initializer for weights. Defaults to 0.02.
        rope_theta: float, optional. The base frequency for rotary position
            embeddings. Defaults to 10,000.0.
        attention_bias: bool, optional. Whether to use bias in attention
            mechanisms. Defaults to False.
        attention_dropout: float, optional. The dropout probability for
            attention mechanisms. Defaults to 0.0.
        dtype: str, optional. The dtype to use for model computations and
            weights. Defaults to None.

    Examples:
    ```python
    import numpy as np
    import keras
    from keras_hub.models import MoonshineBackbone

    # Create random input data for demonstration.
    # Input is now raw-ish audio features (e.g., from MoonshineAudioConverter).
    encoder_raw_input_values = np.random.rand(1, 16000, 1).astype("float32")
    # Mask corresponding to the raw input time dimension
    encoder_padding_mask = np.ones((1, 16000), dtype="bool")
    decoder_token_ids = np.random.randint(
        0, 1000, size=(1, 20), dtype="int32"
    )
    decoder_padding_mask = np.ones((1, 20), dtype="bool")

    # Initialize the Moonshine backbone with specific parameters.
    backbone = MoonshineBackbone(
        vocabulary_size=10000,
        filter_dim=256,
        encoder_num_layers=6,
        decoder_num_layers=6,
        hidden_dim=256,
        intermediate_dim=512,
        encoder_num_heads=8,
        decoder_num_heads=8,
        feedforward_expansion_factor=4,
        decoder_use_swiglu_activation=True,
        encoder_use_swiglu_activation=False,
    )

    # Forward pass through the model.
    outputs = backbone(
        {
            "encoder_input_values": encoder_raw_input_values,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
    )

    # Display the outputs.
    print("Encoder output shape:", outputs["encoder_sequence_output"].shape)
    print("Decoder output shape:", outputs["decoder_sequence_output"].shape)
    ```
    """

    # References:
    # Feature Extractor: UsefulSensors implementation (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L6-L32)
    # Transformer Backbone: Hugging Face implementation (https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1326-L1486).

    def __init__(
        self,
        vocabulary_size,
        filter_dim,
        encoder_num_layers,
        decoder_num_layers,
        hidden_dim,
        intermediate_dim,
        encoder_num_heads,
        decoder_num_heads,
        feedforward_expansion_factor=4,
        encoder_use_swiglu_activation=False,
        decoder_use_swiglu_activation=True,
        max_position_embeddings=2048,
        pad_head_dim_to_multiple_of=None,
        partial_rotary_factor=0.62,
        dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        # ==== Layers ====
        self._compute_mask_layer = ComputeAttentionMask(
            name="compute_attention_mask"
        )

        # Feature extractor layers.
        self.conv1 = keras.layers.Conv1D(
            filters=filter_dim,
            kernel_size=127,
            strides=64,
            use_bias=False,
            padding="valid",
            kernel_initializer=moonshine_kernel_initializer(
                initializer_range=initializer_range
            ),
            name="conv1",
            dtype=dtype,
        )
        self.group_norm = keras.layers.GroupNormalization(
            groups=1,
            axis=-1,
            epsilon=1e-5,
            center=True,
            scale=True,
            name="group_norm",
            dtype=dtype,
        )
        self.tanh_after_conv1 = keras.layers.Activation(
            "tanh", name="tanh_after_conv1", dtype=dtype
        )
        self.conv2 = keras.layers.Conv1D(
            filters=2 * filter_dim,
            kernel_size=7,
            strides=3,
            use_bias=True,
            padding="valid",
            kernel_initializer=moonshine_kernel_initializer(
                initializer_range=initializer_range
            ),
            name="conv2",
            dtype=dtype,
        )
        self.gelu_after_conv2 = keras.layers.Activation(
            "gelu", name="gelu_after_conv2", dtype=dtype
        )
        self.conv3 = keras.layers.Conv1D(
            filters=filter_dim,
            kernel_size=3,
            strides=2,
            use_bias=True,
            padding="valid",
            kernel_initializer=moonshine_kernel_initializer(
                initializer_range=initializer_range
            ),
            name="conv3",
            dtype=dtype,
        )
        self.gelu_after_conv3 = keras.layers.Activation(
            "gelu", name="gelu_after_conv3", dtype=dtype
        )

        # Transformer layers.
        encoder_head_dim = hidden_dim // encoder_num_heads
        if pad_head_dim_to_multiple_of:
            encoder_head_dim = (
                (encoder_head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        decoder_head_dim = hidden_dim // decoder_num_heads
        if pad_head_dim_to_multiple_of:
            decoder_head_dim = (
                (decoder_head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        # Embedding layer for decoder.
        self.token_embedding = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=moonshine_kernel_initializer(
                initializer_range=initializer_range
            ),
            name="token_embedding",
            dtype=dtype,
        )

        # Rotary embeddings for encoder and decoder.
        self.encoder_rotary_embedding = MoonshineRotaryEmbedding(
            head_dim=encoder_head_dim,
            max_position_embeddings=max_position_embeddings,
            partial_rotary_factor=partial_rotary_factor,
            base_value=rope_theta,
            name="encoder_rotary_embedding",
            dtype=dtype,
        )

        self.decoder_rotary_embedding = MoonshineRotaryEmbedding(
            head_dim=decoder_head_dim,
            max_position_embeddings=max_position_embeddings,
            partial_rotary_factor=partial_rotary_factor,
            base_value=rope_theta,
            name="decoder_rotary_embedding",
            dtype=dtype,
        )

        # Dropout for encoder.
        self.encoder_dropout = keras.layers.Dropout(
            dropout, name="encoder_dropout", dtype=dtype
        )
        # Dropout for decoder.
        self.decoder_dropout = keras.layers.Dropout(
            dropout, name="decoder_dropout", dtype=dtype
        )

        # Encoder blocks.
        self.encoder_blocks = []
        for i in range(encoder_num_layers):
            encoder_block = MoonshineEncoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=encoder_num_heads,
                feedforward_expansion_factor=feedforward_expansion_factor,
                use_swiglu_activation=encoder_use_swiglu_activation,
                pad_head_dim_to_multiple_of=pad_head_dim_to_multiple_of,
                initializer_range=initializer_range,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                name=f"encoder_block_{i}",
                dtype=dtype,
            )
            self.encoder_blocks.append(encoder_block)

        # Layer normalization for encoder.
        self.encoder_final_layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=False,
            scale=True,
            name="encoder_final_layer_norm",
            dtype=dtype,
        )

        # Decoder blocks.
        self.decoder_blocks = []
        for i in range(decoder_num_layers):
            decoder_block = MoonshineDecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=decoder_num_heads,
                feedforward_expansion_factor=feedforward_expansion_factor,
                use_swiglu_activation=decoder_use_swiglu_activation,
                pad_head_dim_to_multiple_of=pad_head_dim_to_multiple_of,
                initializer_range=initializer_range,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                name=f"decoder_block_{i}",
                dtype=dtype,
            )
            self.decoder_blocks.append(decoder_block)

        # Layer normalization for decoder.
        self.decoder_post_norm = keras.layers.LayerNormalization(
            epsilon=1e-5,
            center=False,
            scale=True,
            name="decoder_post_norm",
            dtype=dtype,
        )

        # === Functional Model ===
        encoder_raw_input_values = keras.Input(
            shape=(None, 1), name="encoder_input_values", dtype=dtype
        )
        encoder_input_padding_mask = keras.Input(
            shape=(None,), name="encoder_padding_mask", dtype="bool"
        )
        decoder_input = keras.Input(
            shape=(None,), name="decoder_token_ids", dtype="int32"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), name="decoder_padding_mask", dtype="bool"
        )

        # Feature extraction.
        encoder_hidden_states = self.conv1(encoder_raw_input_values)
        encoder_hidden_states = self.tanh_after_conv1(encoder_hidden_states)
        encoder_hidden_states = self.group_norm(encoder_hidden_states)
        encoder_hidden_states = self.conv2(encoder_hidden_states)
        encoder_hidden_states = self.gelu_after_conv2(encoder_hidden_states)
        encoder_hidden_states = self.conv3(encoder_hidden_states)
        encoder_hidden_states = self.gelu_after_conv3(encoder_hidden_states)

        # Compute mask for encoder features.
        original_lengths = keras.ops.sum(
            keras.ops.cast(encoder_input_padding_mask, "int32"), axis=1
        )
        output_lengths = compute_output_lengths(original_lengths)
        encoder_attention_mask = self._compute_mask_layer(
            encoder_hidden_states, output_lengths
        )

        # Encoder.
        encoder_positions = Arange(name="encoder_positions")(
            encoder_hidden_states
        )
        encoder_rotary_emb = self.encoder_rotary_embedding(encoder_positions)
        encoder_hidden_states = self.encoder_dropout(encoder_hidden_states)
        for encoder_block in self.encoder_blocks:
            encoder_hidden_states = encoder_block(
                encoder_hidden_states,
                encoder_rotary_emb,
                attention_mask=encoder_attention_mask,
            )
        encoder_output = self.encoder_final_layer_norm(encoder_hidden_states)

        # Decoder.
        decoder_positions = Arange(name="decoder_positions")(decoder_input)
        decoder_rotary_emb = self.decoder_rotary_embedding(decoder_positions)
        decoder_hidden_states = self.token_embedding(decoder_input)
        decoder_hidden_states = self.decoder_dropout(decoder_hidden_states)
        for decoder_block in self.decoder_blocks:
            decoder_hidden_states = decoder_block(
                decoder_sequence=decoder_hidden_states,
                encoder_sequence=encoder_output,
                rotary_embedding=decoder_rotary_emb,
                decoder_padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_attention_mask,
            )
        decoder_output = self.decoder_post_norm(decoder_hidden_states)

        super().__init__(
            inputs={
                "encoder_input_values": encoder_raw_input_values,
                "encoder_padding_mask": encoder_input_padding_mask,
                "decoder_token_ids": decoder_input,
                "decoder_padding_mask": decoder_padding_mask,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
                "encoder_attention_mask": encoder_attention_mask,
            },
            dtype=dtype,
            **kwargs,
        )

        # ==== Config ====
        self.vocabulary_size = vocabulary_size
        self.filter_dim = filter_dim
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.encoder_use_swiglu_activation = encoder_use_swiglu_activation
        self.decoder_use_swiglu_activation = decoder_use_swiglu_activation
        self.max_position_embeddings = max_position_embeddings
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of
        self.partial_rotary_factor = partial_rotary_factor
        self.dropout = dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "filter_dim": self.filter_dim,
                "encoder_num_layers": self.encoder_num_layers,
                "decoder_num_layers": self.decoder_num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "encoder_num_heads": self.encoder_num_heads,
                "decoder_num_heads": self.decoder_num_heads,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "encoder_use_swiglu_activation": self.encoder_use_swiglu_activation,  # noqa: E501
                "decoder_use_swiglu_activation": self.decoder_use_swiglu_activation,  # noqa: E501
                "max_position_embeddings": self.max_position_embeddings,
                "pad_head_dim_to_multiple_of": self.pad_head_dim_to_multiple_of,
                "partial_rotary_factor": self.partial_rotary_factor,
                "dropout": self.dropout,
                "initializer_range": self.initializer_range,
                "rope_theta": self.rope_theta,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "dtype": self.dtype,
            }
        )
        return config
