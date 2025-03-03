import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.moonshine.moonshine_decoder import MoonshineDecoder
from keras_hub.src.models.moonshine.moonshine_encoder import MoonshineEncoder


@keras_hub_export("keras_hub.models.MoonshineBackbone")
class MoonshineBackbone(Backbone):
    """
    Moonshine backbone for speech recognition.

    This class implements an encoder-decoder backbone as used in the Moonshine
    ASR system. It combines a `MoonshineEncoder` for processing input sequences
    and a `MoonshineDecoder` for generating output sequences. Equivalent in
    functionality to the [MoonshineForConditionalGeneration](https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1509)
    class from the Hugging Face implementation.

    Args:
        vocabulary_size: int, Size of the vocabulary for the embedding layers.
        encoder_num_layers: int, Number of stacked encoder blocks.
        decoder_num_layers: int, Number of stacked decoder blocks.
        hidden_dim: int, Dimensionality of the model's hidden representations
            and embeddings.
        intermediate_dim: int, Dimensionality of the intermediate
            representations in feedforward networks.
        encoder_num_heads: int, Number of attention heads in encoder's
            multi-head attention.
        decoder_num_heads: int, Number of attention heads in decoder's
            multi-head attention.
        feedforward_expansion_factor: int, optional, Multiplier applied to
            intermediate_dim to determine the total width of the feed-forward
            network. Defaults to 4.
        use_swiglu_activation: bool, optional, When True, uses SwiGLU activation
            in the feed-forward network for improved performance. Defaults to
            False.
        max_position_embeddings: int, optional, Maximum sequence length for
            position embeddings. Defaults to 2048.
        pad_head_dim_to_multiple_of: int, optional, If specified, pads the head
            dimension to be a multiple of this value for performance
            optimization. Defaults to None.
        partial_rotary_factor: float, optional, Fraction of dimensions to apply
            rotary position embeddings to. Defaults to 0.62.
        dropout: float, optional, Dropout probability for the model. Defaults to
            0.0.
        dtype: string or `keras.mixed_precision.DTypePolicy`, optional, The
            dtype to use for model computations and weights. Defaults to None.

    Returns:
        Dictionary containing:
            - encoder_sequence_output: The output tensor from the encoder with
                shape (batch_size, encoder_sequence_length, hidden_dim).
            - decoder_sequence_output: The output logits from the decoder with
                shape (batch_size, decoder_sequence_length, vocabulary_size).

    Examples:

    ```python
    import numpy as np
    from keras_hub.models import MoonshineBackbone

    # Create input data.
    encoder_input_values = np.random.rand(1, 100, 256).astype("float32")
    encoder_attention_mask = np.ones((1, 100), dtype="int32")
    decoder_token_ids = np.random.randint(0, 1000, size=(1, 20), dtype="int32")
    decoder_padding_mask = np.ones((1, 20), dtype="int32")

    # Initialize the model.
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

    # Forward pass.
    outputs = backbone({
        "encoder_input_values": encoder_input_values,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_token_ids": decoder_token_ids,
        "decoder_padding_mask": decoder_padding_mask
    })

    print(outputs["encoder_sequence_output"].shape)
    print(outputs["decoder_sequence_output"].shape)
    ```

    Alternatively, you can load a pre-trained model using `from_preset()` and
    use the `fit()` method to train the model:

    ```python
    import numpy as np
    from keras_hub.models import MoonshineBackbone

    # Prepare dummy training data.
    batch_size = 1
    encoder_seq_length = 100
    decoder_seq_length = 20
    hidden_dim = 256
    vocabulary_size = 10000
    x_train = {
        "encoder_input_values": np.random.rand(
            batch_size, encoder_seq_length, hidden_dim
        ).astype("float32"),
        "encoder_attention_mask": np.ones(
            (batch_size, encoder_seq_length), dtype="int32"
        ),
        "decoder_token_ids": np.random.randint(
            0, vocabulary_size, size=(batch_size, decoder_seq_length),
            dtype="int32"
        ),
        "decoder_padding_mask": np.ones(
            (batch_size, decoder_seq_length), dtype="int32"
        ),
    }
    y_train = np.random.randint(
        0, vocabulary_size, size=(batch_size, decoder_seq_length), dtype="int32"
    )
    y_train_dict = {"decoder_sequence_output": y_train}

    # Instantiate the model.
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
        dtype="float32",
    )
    x_train_list = [
        x_train["encoder_input_values"],
        x_train["encoder_attention_mask"],
        x_train["decoder_token_ids"],
        x_train["decoder_padding_mask"],
    ]

    # Train the model.
    backbone.compile(
        optimizer="adam",
        loss={
            "encoder_sequence_output": None,
            "decoder_sequence_output": "sparse_categorical_crossentropy",
        },
        metrics={"decoder_sequence_output": "accuracy"},
    )
    print("Training MoonshineBackbone...")
    backbone.fit(x_train_list, y_train_dict, epochs=1, batch_size=2)
    print("Training completed successfully.")
    ```
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
        use_swiglu_activation=False,
        max_position_embeddings=2048,
        pad_head_dim_to_multiple_of=None,
        partial_rotary_factor=0.62,
        dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        # ==== Layers ====
        self.encoder = MoonshineEncoder(
            num_layers=encoder_num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=encoder_num_heads,
            feedforward_expansion_factor=feedforward_expansion_factor,
            use_swiglu_activation=use_swiglu_activation,
            max_position_embeddings=max_position_embeddings,
            pad_head_dim_to_multiple_of=pad_head_dim_to_multiple_of,
            partial_rotary_factor=partial_rotary_factor,
            name="encoder",
            dtype=dtype,
        )

        self.decoder = MoonshineDecoder(
            num_layers=decoder_num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=decoder_num_heads,
            vocabulary_size=vocabulary_size,
            feedforward_expansion_factor=feedforward_expansion_factor,
            use_swiglu_activation=use_swiglu_activation,
            max_position_embeddings=max_position_embeddings,
            pad_head_dim_to_multiple_of=pad_head_dim_to_multiple_of,
            partial_rotary_factor=partial_rotary_factor,
            name="decoder",
            dtype=dtype,
        )

        self.encoder_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            dtype=dtype,
            name="encoder_layer_norm",
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
        encoder_attention_mask = keras.Input(
            shape=(None,), dtype="int32", name="encoder_attention_mask"
        )
        decoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="decoder_padding_mask"
        )

        # Encoder.
        encoder_sequence_length = ops.sum(encoder_attention_mask, axis=1)
        encoder_input_values = self.encoder_dropout(encoder_input_values)
        encoder_output = self.encoder(
            [encoder_input_values, encoder_sequence_length]
        )
        encoder_output = self.encoder_layer_norm(encoder_output)

        # Decoder.
        decoder_sequence_length = ops.sum(decoder_padding_mask, axis=1)
        decoder_token_ids = self.decoder_dropout(decoder_token_ids)
        decoder_hidden_states = self.decoder(
            [
                decoder_token_ids,
                encoder_output,
                decoder_sequence_length,
            ]
        )

        decoder_logits = self.decoder.embedding_layer(
            decoder_hidden_states, reverse=True
        )
        encoder_sequence_output = keras.layers.Lambda(
            lambda x: x, name="encoder_sequence_output"
        )(encoder_output)
        decoder_sequence_output = keras.layers.Lambda(
            lambda x: x, name="decoder_sequence_output"
        )(decoder_logits)

        super().__init__(
            inputs=[
                encoder_input_values,
                encoder_attention_mask,
                decoder_token_ids,
                decoder_padding_mask,
            ],
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
        self.use_swiglu_activation = use_swiglu_activation
        self.max_position_embeddings = max_position_embeddings
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of
        self.partial_rotary_factor = partial_rotary_factor
        self.dropout = dropout

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
                "use_swiglu_activation": self.use_swiglu_activation,
                "max_position_embeddings": self.max_position_embeddings,
                "pad_head_dim_to_multiple_of": self.pad_head_dim_to_multiple_of,
                "partial_rotary_factor": self.partial_rotary_factor,
                "dropout": self.dropout,
                "dtype": self.dtype,
            }
        )
        return config
