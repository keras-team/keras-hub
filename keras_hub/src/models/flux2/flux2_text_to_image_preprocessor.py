from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.flux2.flux2_backbone import Flux2Backbone
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.models.text_to_image_preprocessor import (
    TextToImagePreprocessor,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

# The Qwen3 chat template for a single user turn, rendered with
# `add_generation_prompt=True, enable_thinking=False`, exactly as
# `Flux2KleinPipeline._get_qwen3_prompt_embeds` applies it.
_PROMPT_PREFIX = "<|im_start|>user\n"
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_TEMPLATE_SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")


@keras_hub_export("keras_hub.models.Flux2TextToImagePreprocessor")
class Flux2TextToImagePreprocessor(TextToImagePreprocessor):
    """FLUX.2 [klein] text-to-image preprocessor.

    Wraps each prompt in the Qwen3 chat template used by FLUX.2 [klein]
    (a single user turn, followed by an assistant turn with thinking
    disabled), tokenizes it, and right-pads or truncates the result to
    `sequence_length` tokens, matching the reference pipeline's
    `padding="max_length", truncation=True`.

    This layer is meant for use with `keras_hub.models.Flux2TextToImage`.

    Args:
        tokenizer: A `keras_hub.models.Qwen3Tokenizer` instance. Its
            `unsplittable_tokens` must include the chat-template special
            tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`),
            which is the case for tokenizers converted from the Hugging Face
            `tokenizer.json`.
        sequence_length: int. The padded length of the token sequence.
            Defaults to `512`.

    Call arguments:
        x: A string, a list of strings, or a string tensor of prompts.
        sequence_length: Optional int. Overrides `sequence_length`.

    Example:
    ```python
    preprocessor = keras_hub.models.Flux2TextToImagePreprocessor.from_preset(
        "flux2_klein_4b"
    )
    preprocessor("A cat holding a sign that says hello world")
    preprocessor(["a red bicycle", "a blue house"])
    ```
    """

    backbone_cls = Flux2Backbone
    tokenizer_cls = Qwen3Tokenizer

    def __init__(self, tokenizer, sequence_length=512, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self.sequence_length = sequence_length

    def build(self, input_shape):
        # Deferred so that tokenizer assets are loaded when restoring.
        unsplittable = getattr(self.tokenizer, "unsplittable_tokens", None)
        missing = [
            token
            for token in _TEMPLATE_SPECIAL_TOKENS
            if token not in (unsplittable or [])
        ]
        if missing:
            raise ValueError(
                "The tokenizer must treat the chat-template tokens "
                f"{list(_TEMPLATE_SPECIAL_TOKENS)} as unsplittable, otherwise "
                f"they are tokenized as plain text. Missing: {missing}. "
                "Pass them in `unsplittable_tokens` when constructing the "
                "tokenizer (tokenizers converted from a Hugging Face "
                "`tokenizer.json` include them)."
            )
        self.packer = StartEndPacker(
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    def _apply_chat_template(self, x):
        if tf is not None and isinstance(x, (tf.Tensor, tf.RaggedTensor)):
            return tf.strings.join([_PROMPT_PREFIX, x, _PROMPT_SUFFIX])
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        if isinstance(x, str):
            return _PROMPT_PREFIX + x + _PROMPT_SUFFIX
        return [self._apply_chat_template(prompt) for prompt in x]

    def _generate_preprocess_python(self, x, sequence_length=None):
        if not self.built:
            self.build(None)
        token_ids = self.tokenizer(self._apply_chat_template(x))
        # The whole templated sequence is truncated from the right, like
        # Hugging Face's `truncation=True`, so no start/end values are added.
        token_ids, padding_mask = self.packer(
            token_ids,
            sequence_length=sequence_length or self.sequence_length,
            add_start_value=False,
            add_end_value=False,
        )
        return {"token_ids": token_ids, "padding_mask": padding_mask}

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        return self._generate_preprocess_python(
            x, sequence_length=sequence_length
        )

    def generate_preprocess(self, x, sequence_length=None):
        """Convert prompts to padded token ids and a padding mask."""
        if self._use_tf_workflow():
            return self._generate_preprocess_tf(
                x, sequence_length=sequence_length
            )
        return self._generate_preprocess_python(
            x, sequence_length=sequence_length
        )

    def call(self, x, sequence_length=None):
        return self.generate_preprocess(x, sequence_length=sequence_length)

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length})
        return config

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        if self.packer is not None:
            self.packer.sequence_length = value
