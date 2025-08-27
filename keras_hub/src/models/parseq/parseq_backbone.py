import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.parseq.parseq_decoder import PARSeqDecoder


@keras_hub_export("keras_hub.models.PARSeqBackbone")
class PARSeqBackbone(Backbone):
    """Scene Text Detection with PARSeq.

    Performs OCR in natural scenes using the PARSeq model described in [Scene
    Text Recognition with Permuted Autoregressive Sequence Models](
    https://arxiv.org/abs/2207.06966). PARSeq is a ViT-based model that allows
    iterative decoding by performing an autoregressive decoding phase, followed
    by a refinement phase.

    Args:
        image_encoder: keras.Model. The image encoder model.
        vocabulary_size: int. The size of the vocabulary.
        max_label_length: int. The maximum length of the label sequence.
        decoder_hidden_dim: int. The dimension of the decoder hidden layers.
        num_decoder_layers: int. The number of decoder layers.
        num_decoder_heads: int. The number of attention heads in the decoder.
        decoder_mlp_dim: int. The dimension of the decoder MLP hidden layer.
        dropout_rate: float. The dropout rate for the decoder network.
            Defaults to `0.1`.
        attention_dropout: float. The dropout rate for the attention weights.
            Defaults to `0.1`.
        dtype: str. `None`, str, or `keras.mixed_precision.DTypePolicy`. The
            dtype to use for the computations and weights.
        **kwargs: Additional keyword arguments passed to the base
            `keras.Model` constructor.
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
        self.decoder = PARSeqDecoder(
            vocabulary_size=vocabulary_size,
            max_label_length=max_label_length,
            num_layers=num_decoder_layers,
            num_heads=num_decoder_heads,
            hidden_dim=decoder_hidden_dim,
            mlp_dim=decoder_mlp_dim,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            name="decoder",
            dtype=dtype,
        )
        self.head = keras.layers.Dense(
            vocabulary_size - 2,  # We don't predict <bos> nor <pad>
            dtype=dtype,
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
        target_out = self.decoder(
            token_id_input, memory, padding_mask=padding_mask_input
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
                "image_encoder": keras.layers.serialize(self.image_encoder),
                "vocabulary_size": self.vocabulary_size,
                "max_label_length": self.max_label_length,
                "decoder_hidden_dim": self.decoder_hidden_dim,
                "num_decoder_layers": self.num_decoder_layers,
                "num_decoder_heads": self.num_decoder_heads,
                "decoder_mlp_dim": self.decoder_mlp_dim,
                "dropout_rate": self.dropout_rate,
                "attention_dropout": self.attention_dropout,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "image_encoder": keras.layers.deserialize(
                    config["image_encoder"]
                ),
            }
        )

        return super().from_config(config)
