import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_flan_t5_tokenizer import (
    BLIP2FlanT5Tokenizer,
)
from keras_hub.src.models.blip2.blip2_image_converter import BLIP2ImageConverter
from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.BLIP2Seq2SeqLMPreprocessor")
class BLIP2Seq2SeqLMPreprocessor(Seq2SeqLMPreprocessor):
    """Multimodal seq2seq preprocessor for the BLIP-2 Flan-T5 variant.

    This preprocessing layer is meant for use with
    `keras_hub.models.BLIP2Seq2SeqLM`, the encoder-decoder (Flan-T5) BLIP-2
    task. Following the keras-hub seq2seq convention it takes a dictionary with
    `"encoder_text"` and (optionally) `"decoder_text"` keys, and additionally
    accepts an `"images"` key when an `image_converter` is configured.

    The encoder text is fed to the T5 encoder (alongside the Q-Former visual
    soft-prompt produced by the backbone), and the decoder text is the
    teacher-forced target during training, or the decoder prompt during
    generation. It returns outputs in the standard `(x, y, sample_weight)`
    format, where `y` is the decoder sequence shifted one step left.

    For use with generation, the layer also exposes `generate_preprocess()`
    and `generate_postprocess()`, which are called implicitly by
    `keras_hub.models.BLIP2Seq2SeqLM.generate()`.

    Args:
        tokenizer: A `keras_hub.models.BLIP2FlanT5Tokenizer` instance.
        image_converter: A `keras_hub.models.BLIP2ImageConverter` instance, or
            `None`. If `None`, the preprocessor operates in text-only mode.
        encoder_sequence_length: int. The length of the packed encoder inputs.
            Defaults to `512`.
        decoder_sequence_length: int. The length of the packed decoder inputs.
            Defaults to `512`.

    Examples:
    ```python
    preprocessor = keras_hub.models.BLIP2Seq2SeqLMPreprocessor.from_preset(
        "blip2_flan_t5_xl"
    )

    # Image + text input.
    preprocessor({
        "images": np.random.randint(0, 256, (2, 224, 224, 3)),
        "encoder_text": ["a photo of a cat", "a photo of a dog"],
        "decoder_text": ["a cat sitting", "a running dog"],
    })

    # Prepare an image and prompt for generation.
    preprocessor.generate_preprocess({
        "images": np.random.randint(0, 256, (224, 224, 3)),
        "encoder_text": "a photo of",
    })
    ```

    References:
        - [Li et al., 2023](https://arxiv.org/abs/2301.12597)
    """

    backbone_cls = BLIP2Backbone
    tokenizer_cls = BLIP2FlanT5Tokenizer
    image_converter_cls = BLIP2ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        encoder_sequence_length=512,
        decoder_sequence_length=512,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            encoder_sequence_length=encoder_sequence_length,
            decoder_sequence_length=decoder_sequence_length,
            **kwargs,
        )
        self.image_converter = image_converter
        self.text_only_model = self.image_converter is None

    def build(self, input_shape):
        # The encoder receives `text + </s>` (no leading start token). The
        # decoder is seeded with T5's `decoder_start_token_id`, which is the
        # pad token (id 0), matching the HuggingFace Flan-T5 convention.
        self.encoder_packer = StartEndPacker(
            start_value=None,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.encoder_sequence_length,
            return_padding_mask=True,
        )
        self.decoder_packer = StartEndPacker(
            start_value=self.tokenizer.pad_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    def _parse_inputs(self, x):
        if isinstance(x, dict):
            images = x.get("images")
            encoder_text = x.get("encoder_text")
            decoder_text = x.get("decoder_text")
        else:
            images = None
            encoder_text = x
            decoder_text = None

        if images is not None and self.text_only_model:
            raise ValueError(
                "The initialized preprocessor is text-only, but `images` is "
                "not `None`. Pass `image_converter` to enable vision inputs."
            )
        return images, encoder_text, decoder_text

    def _preprocess_images(self, images):
        return self.image_converter(images)

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        images, encoder_text, decoder_text = self._parse_inputs(x)

        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            self.tokenizer(encoder_text),
            sequence_length=encoder_sequence_length,
        )
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            self.tokenizer(decoder_text),
            sequence_length=decoder_sequence_length + 1,
        )

        x_out = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        if images is not None:
            x_out["images"] = self._preprocess_images(images)

        # Target is the decoder sequence shifted one step to the left.
        y = decoder_token_ids[..., 1:]
        sample_weight = decoder_padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x_out, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        images, encoder_text, decoder_text = self._parse_inputs(x)
        if decoder_text is None:
            # Initialize an empty prompt for the decoder.
            decoder_text = tf.fill((tf.shape(encoder_text)[0],), "")

        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            self.tokenizer(encoder_text),
            sequence_length=encoder_sequence_length,
        )
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            self.tokenizer(decoder_text),
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )

        x_out = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
        if images is not None:
            x_out["images"] = self._preprocess_images(images)
        return x_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": (
                    keras.layers.serialize(self.image_converter)
                    if self.image_converter is not None
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("image_converter") is not None:
            config["image_converter"] = keras.layers.deserialize(
                config["image_converter"]
            )
        return super().from_config(config)
