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
        # === Layers ===
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_dim,
            dtype=dtype,
            name="embedding",
        )
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

        # === Functional Model ===
        encoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="encoder_token_ids"
        )
        decoder_token_ids = keras.Input(
            shape=(None,), dtype="int32", name="decoder_token_ids"
        )
        encoder_padding_mask = keras.Input(
            shape=(None,), dtype="bool", name="encoder_padding_mask"
        )
        decoder_padding_mask = keras.Input(
            shape=(None,), dtype="bool", name="decoder_padding_mask"
        )

        # Encoder
        x_enc = self.embedding(encoder_token_ids)
        
        # Use provided mask or compute from tokens (if 0 is padding)
        # Note: Functional API inputs are symbolic. We can default to masking 0s if mask is not passed?
        # In Functional API, we have to wire all inputs.
        # But we can make flexible inputs via `inputs` dict in `super().__init__`.
        # However, for a rigid Functional model, we usually define one path.
        # Let's support both explicit mask and implicit mask logic if possible?
        # Actually, standard KerasHub backbones usually take explicit masks in `inputs`.
        # The computation of default mask usually happens OUTSIDE or we assume it's passed.
        # Let's look at `BartBackbone`. It takes `encoder_padding_mask` as input.
        # If the user doesn't pass it, it might fail or we need to handle it.
        # But `BartBackbone` defines explicit Input for it.
        # Let's assume explicit masks for now as per `edrec_backbone_test.py`.
        
        # Helper to compute mask if needed (though difficult in pure Functional construction if conditional)
        # We will assume masks are passed for the functional trace.
        
        for layer in self.encoder_layers:
            x_enc = layer(
                x_enc,
                padding_mask=encoder_padding_mask,
            )
            
        # Decoder
        x_dec = self.embedding(decoder_token_ids)
        for layer in self.decoder_layers:
            x_dec = layer(
                x_dec,
                encoder_outputs=x_enc,
                decoder_padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_padding_mask,
            )

        super().__init__(
            inputs={
                "encoder_token_ids": encoder_token_ids,
                "decoder_token_ids": decoder_token_ids,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_padding_mask": decoder_padding_mask,
            },
            outputs={
                "encoder_sequence_output": x_enc,
                "decoder_sequence_output": x_dec,
            },
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocab_size = vocab_size
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.epsilon = epsilon

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

    @property
    def token_embedding(self):
        return self.embedding
