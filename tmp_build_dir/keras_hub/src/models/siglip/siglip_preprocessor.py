import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.siglip.siglip_backbone import SigLIPBackbone
from keras_hub.src.models.siglip.siglip_image_converter import (
    SigLIPImageConverter,
)
from keras_hub.src.models.siglip.siglip_tokenizer import SigLIPTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.SigLIPPreprocessor")
class SigLIPPreprocessor(CausalLMPreprocessor):
    """SigLIP preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.SigLIPBackbone`. By default, it will take in batches of
    strings and images, and return token ids and resized images.

    Args:
        tokenizer: A `keras_hub.models.SigLIPTokenizer` instance.
        image_converter: A `keras_hub.models.SigLIPImageConverter` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Defaults to `False`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Defaults to `True`.
        canonicalize_text: If `True`, the input strings will be canonicalized
            (converted to lowercase, punctuation removed, and stripped).

    Call arguments:
        x: A dict with `"prompts"` and `"images"` keys, where `"prompts"` is
            `tf.Tensor` or list of python strings and `"images"` are the image
            tensors.
        y: Label data. Should always be `None` since SigLIP doesn't need the
            label to calculate the loss.
        sample_weight: Label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.SigLIPPreprocessor.from_preset(
        "siglip_base_patch16_224"
    )

    # Tokenize the sentence and preprocess the image.
    preprocessor(
        {
            "prompts": "The quick brown fox jumped.",
            "images": np.ones(shape=(123, 123, 3)),
        }
    )

    # Tokenize a batch of sentences and preprocess a batch of images.
    preprocessor(
        {
            "prompts": ["The quick brown fox jumped.", "The fox slept."],
            "images": np.ones(shape=(2, 123, 123, 3)),
        }
    )
    ```
    """

    backbone_cls = SigLIPBackbone
    tokenizer_cls = SigLIPTokenizer
    image_converter_cls = SigLIPImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=64,
        add_start_token=False,
        add_end_token=True,
        canonicalize_text=True,
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
        self.canonicalize_text = bool(canonicalize_text)

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    def canonicalize_inputs(self, inputs):
        # Ref: https://github.com/google-research/big_vision/blob/main/big_vision/evaluators/proj/image_text/prompt_engineering.py
        inputs = tf.convert_to_tensor(inputs)
        # Do lower case.
        inputs = tf.strings.lower(inputs)
        # Remove punctuation.
        inputs = tf.strings.regex_replace(
            inputs,
            (
                r"!|\"|#|\$|%|&|\\|'|\(|\)|\*|\+|,|-|\.|/|:|;|<|=|>|\?|@|\[|\\|"
                r"\]|\^|_|`|{|\||}|~"
            ),
            "",
        )
        inputs = tf.strings.regex_replace(inputs, r"\s+", " ")
        inputs = tf.strings.strip(inputs)
        return inputs

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        images, prompts = x["images"], x["prompts"]
        if self.canonicalize_text:
            prompts = self.canonicalize_inputs(prompts)
        prompts = self.tokenizer(prompts)
        if self.image_converter:
            images = self.image_converter(images)
        token_ids, padding_mask = self.packer(
            prompts,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "images": images,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "canonicalize_text": self.canonicalize_text,
            }
        )
        return config
