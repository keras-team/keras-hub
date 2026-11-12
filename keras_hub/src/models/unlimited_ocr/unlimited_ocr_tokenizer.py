import tokenizers
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

# Copied from the `pre_tokenizer` entry of the checkpoint `tokenizer.json`.
# The first rule isolates groups of up to three digits. The second rule
# isolates runs of CJK and kana characters. The third rule splits words and
# punctuation. The GPT-2 pattern of the base class matches none of them.
DIGIT_PATTERN = r"\p{N}{1,3}"
CJK_PATTERN = "[一-龥぀-ゟ゠-ヿ]+"
TEXT_PATTERN = (
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"
    r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
    r"| ?[\p{P}\p{S}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Markers that appear in prompts and in OCR output. The tokenizer keeps each
# marker as one token. They are not special tokens, so `generate()` keeps them
# in the output text.
OCR_MARKER_TOKENS = (
    "<|ref|>",
    "<|/ref|>",
    "<|det|>",
    "<|/det|>",
    "<|grounding|>",
    "<td>",
    "</td>",
    "<tr>",
    "</tr>",
    "<|User|>",
    "<|Assistant|>",
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.UnlimitedOCRTokenizer",
        "keras_hub.models.UnlimitedOCRTokenizer",
    ]
)
class UnlimitedOCRTokenizer(BytePairTokenizer):
    """An Unlimited-OCR byte-level BPE tokenizer.

    The tokenizer uses the three regular expression rules of the checkpoint
    `tokenizer.json` as its pre-tokenizer. The tokenizer keeps the image
    token and the OCR marker tokens, such as `"<|det|>"` and
    `"<|grounding|>"`, as single tokens.

    Args:
        vocabulary: dict or `None`. Maps token strings to integer ids.
        merges: list or `None`. The BPE merge rules.
        bos_token: str. The beginning-of-sequence token.
        eos_token: str. The end-of-sequence token.
        pad_token: str. The padding token.
        image_token: str. The placeholder token of the image features.
            Defaults to `"<image>"`.
        unsplittable_tokens: list or `None`. The tokens that the tokenizer
            never splits. If `None`, the tokenizer uses its special tokens
            and the OCR marker tokens.

    Example:

    ```python
    tokenizer = keras_hub.models.UnlimitedOCRTokenizer.from_preset(
        "unlimited_ocr"
    )

    # Unbatched input.
    tokenizer("<image>document parsing.")

    # Batched input.
    tokenizer(["<image>document parsing.", "Multi page parsing."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("<image>document parsing."))
    ```
    """

    backbone_cls = UnlimitedOCRBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        pad_token="<｜▁pad▁｜>",
        image_token="<image>",
        unsplittable_tokens=None,
        **kwargs,
    ):
        self._add_special_token(bos_token, "start_token")
        self._add_special_token(eos_token, "end_token")
        self._add_special_token(pad_token, "pad_token")
        self._add_special_token(image_token, "image_token")
        if unsplittable_tokens is None:
            unsplittable_tokens = sorted(self.special_tokens) + list(
                OCR_MARKER_TOKENS
            )
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=unsplittable_tokens,
            **kwargs,
        )

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        super()._set_vocabulary_and_merges_tokenizers(vocabulary, merges)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    tokenizers.Regex(DIGIT_PATTERN), behavior="isolated"
                ),
                pre_tokenizers.Split(
                    tokenizers.Regex(CJK_PATTERN), behavior="isolated"
                ),
                pre_tokenizers.Split(
                    tokenizers.Regex(TEXT_PATTERN), behavior="isolated"
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space, use_regex=False
                ),
            ]
        )

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        # The base class splits text with the GPT-2 pattern in the graph. That
        # gives other ids for this checkpoint. This method calls the
        # `tokenizers` backend from the graph instead.
        self._maybe_initialized_tokenizers()

        def _encode(string_tensor):
            values = string_tensor.numpy().tolist()
            strings = [v.decode("utf-8") for v in values]
            encodings = self._tokenizer.encode_batch(
                strings, add_special_tokens=False
            )
            return tf.ragged.constant(
                [e.ids for e in encodings], dtype=self.compute_dtype
            )

        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)
        tokens = tf.py_function(
            _encode,
            [inputs],
            Tout=tf.RaggedTensorSpec(
                shape=[None, None],
                dtype=self.compute_dtype,
                ragged_rank=1,
            ),
        )
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(
                shape=output_shape, default_value=self.pad_token_id
            )
        if unbatched:
            tokens = tokens[0]
        return tokens

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bos_token": self.start_token,
                "eos_token": self.end_token,
                "pad_token": self.pad_token,
                "image_token": self.image_token,
            }
        )
        return config
