import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.parseq.parseq_decoder import PARSeqDecode


@keras_hub_export("keras_hub.models.PARSeqBackbone")
class PARSeqBackbone(Backbone):
    """Scene Text Detection with PARSeq.

    Performs OCR in natural scenes using the PARSeq model described in [Scene
    Text Recognition with Permuted Autoregressive Sequence Models](
    https://arxiv.org/abs/2207.06966). PARSeq is a ViT-based model that allows
    iterative decoding by performing an autoregressive decoding phase, followed
    by a refinement phase.
    """

    def __init__(
        self,
        image_encoder,
        vocabulary_size,
        max_label_length,
        decoder_hidden_dim,
        num_decoder_layers,
        num_decoder_heads,
        decoder_mlp_dim,
        dropout_rate=0.1,
        attention_dropout=0.1,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.image_encoder = image_encoder
        self.decode = PARSeqDecode(
            vocabulary_size=vocabulary_size,
            max_label_length=max_label_length,
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            hidden_dim=decoder_hidden_dim,
            mlp_dim=decoder_mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            name="decoder",
        )
        self.head = keras.layers.Dense(
            vocabulary_size - 2,  # We don't predict <bos> nor <pad>
        )

        # === Functional Model ===
        image_input = self.image_encoder.input

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        memory = self.image_encoder(image_input)
        target_out = self.decode(
            token_id_input, memory, target_padding_mask=padding_mask_input
        )
        logits = self.head(target_out)

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.max_label_length = max_label_length
        self.decoder_hidden_dim = decoder_hidden_dim
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.decoder_mlp_dim = decoder_mlp_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout

        super().__init__(
            inputs={
                "images": image_input,
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=logits,
            dtype=dtype,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder": keras.layers.serialize(self.image_encoder),
                "vocabulary_size": self.vocabulary_size,
                "max_label_length": self.max_label_length,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "num_decoder_layers": self.num_decoder_layers,
                "num_decoder_heads": self.num_decoder_heads,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
            }
        )