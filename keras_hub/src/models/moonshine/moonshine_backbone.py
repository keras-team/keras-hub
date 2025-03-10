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

    Returns:
        Dictionary: A dictionary containing:
            encoder_sequence_output: A tensor of shape (batch_size,
                encoder_sequence_length, hidden_dim) representing the output of
                the encoder.
            decoder_sequence_output: A tensor of shape (batch_size,
                decoder_sequence_length, vocabulary_size) representing the
                output logits from the decoder.

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

    Additionally, you can use the `fit()` method to train the model as follows.

    ```python
    batch_size = 1
    encoder_seq_length = 100
    decoder_seq_length = 20
    hidden_dim = 256
    vocabulary_size = 10000
    x_train = {
        "encoder_input_values": np.random.rand(2, 100, 256).astype("float32"),
        "decoder_token_ids": np.random.randint(
            0, 10000, size=(2, 20), dtype="int32"
        ),
    }
    y_train = np.random.randint(0, 10000, size=(2, 20), dtype="int32")

    # Compile and train.
    backbone.compile(
        optimizer="adam",
        loss={"decoder_sequence_output": "sparse_categorical_crossentropy"},
        metrics={"decoder_sequence_output": "accuracy"},
    )
    backbone.fit(
        x_train, {"decoder_sequence_output": y_train}, epochs=1, batch_size=2
    )
    ```

    ## References
    Defined and formulated based on the
    [Hugging Face implementation of the MoonshineForConditionalGeneration](https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1509)
    class.
    """

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
        # ==== Layers ====
        encoder_head_dim = hidden_dim // encoder_num_heads
        if pad_head_dim_to_multiple_of is not None:
            encoder_head_dim = (
                (encoder_head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        decoder_head_dim = hidden_dim // decoder_num_heads
        if pad_head_dim_to_multiple_of is not None:
            decoder_head_dim = (
                (decoder_head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

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

        # Embedding layer for decoder.
        self.embedding_layer = ReversibleEmbedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=keras.initializers.RandomNormal(
                stddev=initializer_range
            ),
            name="embedding_layer",
            dtype=dtype,
        )

        # Encoder blocks.
        self.encoder_blocks = [
            MoonshineEncoderBlock(
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
            for i in range(encoder_num_layers)
        ]

        # Decoder blocks.
        self.decoder_blocks = [
            MoonshineDecoderBlock(
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
            for i in range(decoder_num_layers)
        ]

        # Layer normalization.
        self.encoder_final_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="encoder_final_layer_norm",
            dtype=dtype,
        )
        self.decoder_post_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="decoder_post_norm",
            dtype=dtype,
        )

        self.encoder_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="encoder_dropout",
        )
        self.decoder_dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name="decoder_dropout",
        )

        # ==== Functional Model ====
        encoder_input_values = keras.Input(
            shape=(None, hidden_dim), dtype=dtype, name="encoder_input_values"
        )
        decoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )

        # Rotary embeddings.
        pos_indices = keras.ops.arange(max_position_embeddings)
        encoder_rotary_emb = self.encoder_rotary_embedding(pos_indices)
        decoder_rotary_emb = self.decoder_rotary_embedding(pos_indices)

        # Encoder.
        x = self.encoder_dropout(encoder_input_values)
        for block in self.encoder_blocks:
            x = block(x, encoder_rotary_emb)
        encoder_output = self.encoder_final_layer_norm(x)

        # Decoder.
        x = self.embedding_layer(decoder_token_ids)
        x = self.decoder_dropout(x)
        for block in self.decoder_blocks:
            x, _, _, _, _ = block(
                [x, encoder_output, decoder_rotary_emb], use_cache=False
            )
        x = self.decoder_post_norm(x)
        decoder_logits = self.embedding_layer(x, reverse=True)
        encoder_sequence_output = keras.layers.Lambda(
            lambda x: x, name="encoder_sequence_output"
        )(encoder_output)
        decoder_sequence_output = keras.layers.Lambda(
            lambda x: x, name="decoder_sequence_output"
        )(decoder_logits)

        super().__init__(
            inputs={
                "encoder_input_values": encoder_input_values,
                "decoder_token_ids": decoder_token_ids,
            },
            outputs={
                "encoder_sequence_output": encoder_sequence_output,
                "decoder_sequence_output": decoder_sequence_output,
            },
            dtype=dtype,
            **kwargs,
        )

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
