import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.moondream.moondream_backbone import MoondreamBackbone
from keras_hub.src.models.moondream.moondream_image_converter import (
    MoondreamImageConverter,
)
from keras_hub.src.models.moondream.moondream_tokenizer import (
    MoondreamTokenizer,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.MoondreamCausalLMPreprocessor")
class MoondreamCausalLMPreprocessor(CausalLMPreprocessor):
    """Moondream Causal LM preprocessor.

    This preprocessing layer handles tokenization of text prompts and optional
    image conversion for the Moondream model. It tokenizes prompts and
    responses, packs them into fixed-length sequences, and optionally resizes
    and normalizes images using a `MoondreamImageConverter`.

    This layer can be used directly with `fit()`, `predict()`, `evaluate()`,
    and `generate()` when attached to a `MoondreamCausalLM` model.

    Args:
        tokenizer: A `keras_hub.models.MoondreamTokenizer` instance.
        image_converter: A `keras_hub.layers.MoondreamImageConverter` instance,
            or `None`. If `None`, images will be passed through without
            preprocessing. Defaults to `None`.
        sequence_length: int. The length of the packed token sequences.
            Defaults to `1024`.
        add_start_token: bool. Whether to prepend the tokenizer's start token
            to each sequence. Defaults to `True`.
        add_end_token: bool. Whether to append the tokenizer's end token to
            each sequence. Defaults to `True`.

    Examples:
    ```python
    import numpy as np
    import keras_hub

    # Training preprocessing (images + prompts + responses).
    tokenizer = keras_hub.models.MoondreamTokenizer.from_preset("moondream2")
    preprocessor = keras_hub.models.MoondreamCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=512,
    )
    x = {
        "images": np.random.rand(2, 378, 378, 3).astype("float32"),
        "prompts": ["Describe this image.", "What is in the photo?"],
        "responses": ["A cat sitting on a mat.", "A dog in a park."],
    }
    x, y, sample_weight = preprocessor(x)

    # Inference preprocessing.
    x = {
        "images": np.random.rand(2, 378, 378, 3).astype("float32"),
        "prompts": ["Describe this image.", "What is in the photo?"],
    }
    x = preprocessor.generate_preprocess(x)
    ```
    """

    backbone_cls = MoondreamBackbone
    tokenizer_cls = MoondreamTokenizer
    image_converter_cls = MoondreamImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
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

    def build(self, input_shape):
        # Defer packer creation to `build()` so that tokenizer assets have
        # loaded when restoring a saved model.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        images = x.get("images", None)
        prompts = x["prompts"]
        responses = x["responses"]

        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        if images is not None and self.image_converter is not None:
            images = self.image_converter(images)

        # Pad with one extra token to account for the truncation below.
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1

        # The last token does not have a next token, so we truncate it out.
        x_out = {
            "token_ids": token_ids[..., :-1],
            "response_mask": response_mask[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }
        if images is not None:
            x_out["images"] = images

        # Target `y` will be the next token.
        y = token_ids[..., 1:]
        # Only compute the loss for labels in the response.
        sample_weight = response_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x_out, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert inputs to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask. Unlike calling the layer for training, this method does not
        compute labels and will never append an end token to the end of the
        sequence (as generation is expected to continue at the end of the
        inputted prompt).

        Args:
            x: A dict with keys `"images"` (optional) and `"prompts"`. Prompts
                are strings or batched strings.
            sequence_length: int. Override the layer's configured
                `sequence_length` for this call.
        """
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length

        images = x.get("images", None)
        prompts = self.tokenizer(x["prompts"])

        if images is not None and self.image_converter is not None:
            images = self.image_converter(images)

        if "responses" in x:
            responses = self.tokenizer(x["responses"])
            segments = (prompts, responses)
        else:
            segments = (prompts,)

        token_ids, _ = self.packer(
            segments,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        out = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if images is not None:
            out["images"] = images
        return out

    def generate_postprocess(self, x):
        """Convert integer token output back to strings after generation."""
        token_ids = x["token_ids"]
        padding_mask = x["padding_mask"]
        # Only decode tokens that belong to the generated response.
        ids_to_decode = token_ids
        mask = padding_mask
        return self.tokenizer.detokenize(
            ids_to_decode * mask + self.tokenizer.pad_token_id * (1 - mask)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": (
                    keras.saving.serialize_keras_object(self.image_converter)
                    if self.image_converter is not None
                    else None
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "image_converter" in config and config["image_converter"] is not None:
            config["image_converter"] = keras.saving.deserialize_keras_object(
                config["image_converter"]
            )
        return super().from_config(config)
