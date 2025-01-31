import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone


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
        decode_autoregressive=True,
        alphabet_size=97,
        max_text_length=25,
        num_decoder_layers=1,
        num_decoder_heads=12,
        dropout_rate=0.1,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.image_encoder = image_encoder

        image_input = self.image_encoder.input
        output = self.image_encoder(image_input)

        # === Config ===
        self.decode_autoregressive = decode_autoregressive
        self.alphabet_size = alphabet_size
        self.max_text_length = max_text_length
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_heads = num_decoder_heads
        self.dropout_rate = dropout_rate

        super().__init__(
            inputs=image_input,
            outputs=output,
            dtype=dtype,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder": keras.layers.serialize(self.image_encoder),
                "alphabet_size": self.alphabet_size,
                "max_text_length": self.max_text_length,
                "num_decoder_layers": self.num_decoder_layers,
                "num_decoder_heads": self.num_decoder_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
