import keras
import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_image_converter import BLIP2ImageConverter
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_hub.src.tokenizers.tokenizer import Tokenizer
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
        tokenizer: A `keras_hub.models.BLIP2OPTTokenizer`,
            `keras_hub.models.BLIP2FlanT5Tokenizer`, or
            `keras_hub.models.BLIP2VicunaTokenizer` instance. The actual
            tokenizer is resolved from the preset config when using
            `from_preset()`.
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
    tokenizer_cls = Tokenizer
    image_converter_cls = BLIP2ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        qformer_tokenizer=None,
        sequence_length=512,
        qformer_sequence_length=64,
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
        # The (BERT) Q-Former tokenizer is only present for instruction-aware
        # InstructBLIP variants; it tokenizes the instruction fed to the
        # Q-Former, separately from the language-model `tokenizer`.
        self.qformer_tokenizer = qformer_tokenizer
        self.qformer_sequence_length = qformer_sequence_length
        self.instruction_aware = qformer_tokenizer is not None
        self.qformer_packer = None
        self.text_only_model = self.image_converter is None

    def build(self, input_shape):
        super().build(input_shape)
        if self.instruction_aware:
            self.qformer_packer = StartEndPacker(
                start_value=self.qformer_tokenizer.cls_token_id,
                end_value=self.qformer_tokenizer.sep_token_id,
                pad_value=self.qformer_tokenizer.pad_token_id,
                sequence_length=self.qformer_sequence_length,
                return_padding_mask=True,
            )

    def _preprocess_images(self, images):
        return self.image_converter(images)

    def _preprocess_qformer(self, text, sequence_length=None):
        """Tokenize the instruction for the (instruction-aware) Q-Former."""
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.qformer_sequence_length
        qf_ids, qf_mask = self.qformer_packer(
            self.qformer_tokenizer(text),
            sequence_length=sequence_length,
            add_start_value=True,
            add_end_value=True,
        )
        return {
            "qformer_token_ids": qf_ids,
            "qformer_padding_mask": qf_mask,
        }

    def _parse_inputs(self, x):
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
        return images, text

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        images, text = self._parse_inputs(x)

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
        if self.instruction_aware and text is not None:
            x_out.update(self._preprocess_qformer(text))

        return x_out, y_label, sw

    @preprocessing_function
    def generate_preprocess(self, x, sequence_length=None):
        if isinstance(x, dict) and "token_ids" in x:
            return x

        images, text = self._parse_inputs(x)

        if isinstance(self.tokenizer, T5Tokenizer):
            if not self.built:
                self.build(None)
            sequence_length = sequence_length or self.sequence_length
            token_ids, padding_mask = self.packer(
                self.tokenizer(text),
                sequence_length=sequence_length,
                add_start_value=False,
                add_end_value=True,
            )
            x_text = {"token_ids": token_ids, "padding_mask": padding_mask}
        else:
            x_text = super().generate_preprocess(
                text, sequence_length=sequence_length
            )

        x_out = {
            "token_ids": x_text["token_ids"],
            "padding_mask": x_text["padding_mask"],
        }
        if images is not None:
            x_out["images"] = self._preprocess_images(images)
        if self.instruction_aware and text is not None:
            x_out.update(self._preprocess_qformer(text))

        return x_out

    @preprocessing_function
    def generate_postprocess(self, x):
        # The OPT (byte-pair) tokenizer has a pure-Python detokenize path that
        # accepts a ragged list of sequences. The Flan-T5 SentencePiece
        # tokenizer only detokenizes via TensorFlow, which cannot convert a
        # ragged Python list (produced by batched generation with
        # different-length outputs). Build a ragged tensor explicitly for it.
        if not isinstance(self.tokenizer, T5Tokenizer):
            return super().generate_postprocess(x)

        if not self.built:
            self.build(None)

        token_ids = keras.ops.convert_to_numpy(x["token_ids"]).astype("int32")
        padding_mask = keras.ops.convert_to_numpy(x["padding_mask"]).astype(
            "bool"
        )
        ids_to_strip = getattr(self.tokenizer, "special_token_ids", [])
        mask = padding_mask
        for token_id in ids_to_strip:
            mask = mask & (token_ids != token_id)

        if token_ids.ndim == 1:
            stripped = token_ids[mask].tolist()
        else:
            stripped = tf.ragged.constant(
                [
                    token_ids[i][mask[i]].tolist()
                    for i in range(token_ids.shape[0])
                ],
                dtype="int32",
            )
        return self.tokenizer.detokenize(stripped)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": (
                    keras.layers.serialize(self.image_converter)
                    if self.image_converter is not None
                    else None
                ),
                "qformer_tokenizer": (
                    keras.layers.serialize(self.qformer_tokenizer)
                    if self.qformer_tokenizer is not None
                    else None
                ),
                "qformer_sequence_length": self.qformer_sequence_length,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if config.get("image_converter") is not None:
            config["image_converter"] = keras.layers.deserialize(
                config["image_converter"]
            )
        if config.get("qformer_tokenizer") is not None:
            config["qformer_tokenizer"] = keras.layers.deserialize(
                config["qformer_tokenizer"]
            )
        return super().from_config(config)
