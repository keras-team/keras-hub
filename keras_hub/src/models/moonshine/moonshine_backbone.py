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
from keras_hub.src.utils.keras_utils import clone_initializer


class Arange(keras.layers.Layer):
    def call(self, inputs):
        sequence_length = keras.ops.shape(inputs)[1]
        return keras.ops.arange(sequence_length, dtype="int32")


@keras_hub_export("keras_hub.models.MoonshineBackbone")
class MoonshineBackbone(Backbone):
    """Moonshine backbone for speech recognition.

    This class implements an encoder-decoder backbone, as used in the Moonshine
    ASR system. It combines `MoonshineEncoderBlock` instances for processing
    input sequences and `MoonshineDecoderBlock` instances for generating output
    sequences.

    Args:
        vocabulary_size: int. The size of the vocabulary for the embedding
            layers.
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
        use_swiglu_activation: bool, optional. When True, uses the SwiGLU
            activation in the feedforward network for improved performance.
            Defaults to False.
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
        rope_scaling: dict, optional. The scaling configuration for rotary
            position embeddings. Defaults to None.
        dtype: str, optional. The dtype to use for model computations and
            weights. Defaults to None.

    Examples:
    ```python
    # Create random input data for demonstration.
    encoder_input_values = np.random.rand(1, 100, 256).astype("float32")
    decoder_token_ids = np.random.randint(
        0, 1000, size=(1, 20), dtype="int32"
    )

    # Initialize the Moonshine backbone with specific parameters.
    backbone = MoonshineBackbone(
        vocabulary_size=10000,
        encoder_num_layers=6,
        decoder_num_layers=6,
        hidden_dim=256,
        intermediate_dim=512,
        encoder_num_heads=8,
        decoder_num_heads=8,
        feedforward_expansion_factor=4,
        use_swiglu_activation=True,
    )

    # Forward pass through the model.
    outputs = backbone(
        {
            "encoder_input_values": encoder_input_values,
            "decoder_token_ids": decoder_token_ids,
        }
    )

    # Display the outputs.
    print("Encoder output shape:", outputs["encoder_sequence_output"].shape)
    print("Decoder output shape:", outputs["decoder_sequence_output"].shape)
    ```
    """

    # References:
    # Defined and formulated based on the Hugging Face implementation of the
    # MoonshineModel class (https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1326-L1486).

    def __init__(
        self,
        vocabulary_size,
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
        rope_scaling=None,
        dtype=None,
        **kwargs,
    ):
        # ==== Config ====
        self.vocabulary_size = vocabulary_size
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
        self.rope_scaling = rope_scaling
        self.embeddings_initializer = moonshine_kernel_initializer(
            initializer_range=initializer_range
        )

        # ==== Layers ====
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
            embeddings_initializer=clone_initializer(
                self.embeddings_initializer
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
            rope_scaling=rope_scaling,
            name="encoder_rotary_embedding",
            dtype=dtype,
        )

        self.decoder_rotary_embedding = MoonshineRotaryEmbedding(
            head_dim=decoder_head_dim,
            max_position_embeddings=max_position_embeddings,
            partial_rotary_factor=partial_rotary_factor,
            base_value=rope_theta,
            rope_scaling=rope_scaling,
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
        encoder_input = keras.Input(
            shape=(None, hidden_dim), name="encoder_input_values", dtype=dtype
        )
        decoder_input = keras.Input(
            shape=(None,), name="decoder_token_ids", dtype="int32"
        )
        encoder_padding_mask = keras.Input(
            shape=(None,), name="encoder_padding_mask", dtype="bool"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), name="decoder_padding_mask", dtype="bool"
        )

        # Encoder.
        encoder_positions = Arange(name="encoder_positions")(encoder_input)
        encoder_rotary_emb = self.encoder_rotary_embedding(encoder_positions)
        encoder_hidden_states = self.encoder_dropout(encoder_input)
        for encoder_block in self.encoder_blocks:
            encoder_hidden_states = encoder_block(
                encoder_hidden_states,
                encoder_rotary_emb,
                attention_mask=encoder_padding_mask,
            )
        encoder_output = self.encoder_final_layer_norm(encoder_hidden_states)

        # Decoder.
        decoder_positions = Arange(name="decoder_positions")(decoder_input)
        decoder_rotary_emb = self.decoder_rotary_embedding(decoder_positions)
        decoder_hidden_states = self.token_embedding(decoder_input)
        decoder_hidden_states = self.decoder_dropout(decoder_hidden_states)
        for decoder_block in self.decoder_blocks:
            decoder_hidden_states, _, _, _, _ = decoder_block(
                [decoder_hidden_states, encoder_output, decoder_rotary_emb],
                decoder_attention_mask=decoder_padding_mask,
                encoder_attention_mask=encoder_padding_mask,
            )
        decoder_output = self.decoder_post_norm(decoder_hidden_states)

        super().__init__(
            inputs={
                "encoder_input_values": encoder_input,
                "decoder_token_ids": decoder_input,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_padding_mask": decoder_padding_mask,
            },
            outputs={
                "encoder_sequence_output": encoder_output,
                "decoder_sequence_output": decoder_output,
            },
            dtype=dtype,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
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
                "rope_scaling": self.rope_scaling,
                "dtype": self.dtype,
            }
        )
        return config

    # Use the MoonshineBackbone class as part of a trainable model.
    def logits(self, decoder_hidden_states):
        return self.token_embedding(decoder_hidden_states, reverse=True)
