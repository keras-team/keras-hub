import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.edrec.edrec_layers import EdRecDecoderBlock
from keras_hub.src.models.edrec.edrec_layers import EdRecEncoderBlock


@keras_hub_export("keras_hub.models.EdRecBackbone")
class EdRecBackbone(Backbone):
    """EdRec Backbone model.

    Args:
        vocab_size: int, size of the vocabulary.
        num_layers_enc: int, number of encoder layers.
        num_layers_dec: int, number of decoder layers.
        hidden_dim: int, hidden dimension (d_model).
        intermediate_dim: int, intermediate dimension (d_ff).
        num_heads: int, number of attention heads.
        dropout: float, dropout rate.
        epsilon: float, epsilon for simple RMSNorm.
    """

    def __init__(
        self,
        vocab_size,
        num_layers_enc,
        num_layers_dec,
        hidden_dim,
        intermediate_dim,
        num_heads,
        dropout=0.0,
        epsilon=1e-6,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.vocab_size = vocab_size
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon

        # Embeddings
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_dim,
            dtype=dtype,
            name="embedding",
        )

        # Encoder
        self.encoder_layers = []
        for i in range(num_layers_enc):
            self.encoder_layers.append(
                EdRecEncoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    dropout_rate=dropout,
                    epsilon=epsilon,
                    dtype=dtype,
                    name=f"encoder_layer_{i}",
                )
            )

        # Decoder
        self.decoder_layers = []
        for i in range(num_layers_dec):
            self.decoder_layers.append(
                EdRecDecoderBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    intermediate_dim=intermediate_dim,
                    dropout_rate=dropout,
                    epsilon=epsilon,
                    dtype=dtype,
                    name=f"decoder_layer_{i}",
                )
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "num_layers_enc": self.num_layers_enc,
                "num_layers_dec": self.num_layers_dec,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "epsilon": self.epsilon,
            }
        )
        return config

    def call(
        self,
        inputs,
        training=False,
    ):
        # inputs can be a dict or tuple.
        # Expected keys: encoder_token_ids, decoder_token_ids
        # Optional: encoder_padding_mask, decoder_padding_mask

        encoder_token_ids = inputs["encoder_token_ids"]
        decoder_token_ids = inputs.get("decoder_token_ids")
        encoder_padding_mask = inputs.get("encoder_padding_mask")
        decoder_padding_mask = inputs.get("decoder_padding_mask")

        # Embed encoder
        x_enc = self.embedding(encoder_token_ids)

        if encoder_padding_mask is None:
            encoder_padding_mask = ops.not_equal(encoder_token_ids, 0)

        # Run Encoder
        for layer in self.encoder_layers:
            x_enc = layer(
                x_enc,
                padding_mask=encoder_padding_mask,
                training=training,
            )

        # If decoder is present
        x_dec = None
        if decoder_token_ids is not None:
            x_dec = self.embedding(decoder_token_ids)

            if decoder_padding_mask is None:
                decoder_padding_mask = ops.not_equal(decoder_token_ids, 0)

            for layer in self.decoder_layers:
                x_dec = layer(
                    x_dec,
                    encoder_outputs=x_enc,
                    decoder_padding_mask=decoder_padding_mask,
                    encoder_padding_mask=encoder_padding_mask,
                    training=training,
                )
            return {
                "encoder_sequence_output": x_enc,
                "decoder_sequence_output": x_dec,
            }

        return {
            "encoder_sequence_output": x_enc,
        }

    @property
    def token_embedding(self):
        return self.embedding
