"""BLIP-2 causal LM preprocessor."""

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_image_converter import BLIP2ImageConverter
from keras_hub.src.models.blip2.blip2_tokenizer import BLIP2Tokenizer
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.BLIP2CausalLMPreprocessor")
class BLIP2CausalLMPreprocessor(CausalLMPreprocessor):
    """Multimodal preprocessor for the BLIP-2 Causal LM.

    This preprocessing layer is meant for use with
    `keras_hub.models.BLIP2CausalLM`. It can be configured in two ways:
    text-only and text + vision, based on whether the passed value of
    `image_converter` is `None`. For the former, it takes in batches of
    strings. For the latter, it takes in batches of images and strings.
    It returns outputs in a `(x, y, sample_weight)` format, where the `y`
    label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this
    preprocessor is attached to a `keras_hub.models.BLIP2CausalLM` instance,
    these methods will be called implicitly in `generate()`. They can also be
    called standalone (e.g. to precompute preprocessing inputs for generation
    in a separate process).

    Args:
        tokenizer: A `keras_hub.models.BLIP2Tokenizer` instance.
        image_converter: A `keras_hub.models.BLIP2ImageConverter` instance, or
            `None`. If `None`, the preprocessor operates in text-only mode.
        sequence_length: int. The maximum length of the packed token sequence.
            Defaults to `512`.
        add_start_token: bool. Whether to prepend the tokenizer start token to
            each input sequence. Defaults to `True`.
        add_end_token: bool. Whether to append the tokenizer end token to each
            input sequence. Defaults to `True`.

    Examples:
    ```python
    # Load from a preset.
    preprocessor = keras_hub.models.BLIP2CausalLMPreprocessor.from_preset(
        "blip2_opt_2_7b"
    )

    # Text-only input.
    preprocessor({"text": ["A photo of a cat", "A photo of a dog"]})

    # Image + text input.
    preprocessor({
        "images": np.random.randint(0, 256, (2, 500, 500, 3)),
        "text": ["A photo of a cat", "A photo of a dog"],
    })

    # Prepare tokens for generation.
    preprocessor.generate_preprocess({
        "images": np.random.randint(0, 256, (500, 500, 3)),
        "text": "A photo of",
    })

    # Map generation outputs back to strings.
    preprocessor.generate_postprocess({
        "token_ids": np.array([[0, 4, 5, 2, 1, 1]]),
        "padding_mask": np.array([[1, 1, 1, 1, 0, 0]]),
    })
    ```

    References:
        - [Li et al., 2023](https://arxiv.org/abs/2301.12597)
    """

    backbone_cls = BLIP2Backbone
    tokenizer_cls = BLIP2Tokenizer
    image_converter_cls = BLIP2ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=512,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter
        self.text_only_model = self.image_converter is None

    def _preprocess_images(self, images):
        """Run the image converter on a batch of raw images."""
        return self.image_converter(images)

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if isinstance(x, dict):
            images = x.get("images")
            text = x.get("text")
        else:
            images = None
            text = x

        if images is not None and self.text_only_model:
            raise ValueError(
                "The initialized preprocessor is text-only, but `images` is "
                "not `None`. Pass `image_converter` to enable vision inputs."
            )

        processed = super().call(
            text,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )
        x_text, y_label, sw = processed

        x_out = {
            "token_ids": x_text["token_ids"],
            "padding_mask": x_text["padding_mask"],
        }
        if images is not None:
            x_out["images"] = self._preprocess_images(images)

        return x_out, y_label, sw

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        """Convert inputs to token ids and masks for generation.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).

        Accepts either raw string/image inputs (for generation from scratch) or
        an already-preprocessed dict containing ``token_ids`` and
        ``padding_mask`` (e.g. the output of a previous `call()` used directly
        as generation input).  In the latter case the dict is returned as-is so
        that the generation loop can reuse pre-tokenised inputs without trying
        to re-tokenise a ``None`` text field.
        """
        # Fast-path: input is already tokenised (has token_ids).
        # This happens when callers pass the output of `call()` directly to
        # `generate()` instead of raw text/image data.
        if isinstance(x, dict) and "token_ids" in x:
            return x

        if isinstance(x, dict):
            images = x.get("images")
            text = x.get("text")
        else:
            images = None
            text = x

        if images is not None and self.text_only_model:
            raise ValueError(
                "The initialized preprocessor is text-only, but `images` is "
                "not `None`. Pass `image_converter` to enable vision inputs."
            )

        x_text = super().generate_preprocess(
            text, sequence_length=sequence_length
        )

        x_out = {
            "token_ids": x_text["token_ids"],
            "padding_mask": x_text["padding_mask"],
        }
        if images is not None:
            x_out["images"] = self._preprocess_images(images)

        return x_out

    @preprocessing_function
    def generate_postprocess(self, x):
        """Convert token id output back to strings.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        return super().generate_postprocess(x)

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
