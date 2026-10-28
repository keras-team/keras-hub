import warnings

import tokenizers
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

# The `pre_tokenizer` regex from `openai/privacy-filter`'s `tokenizer.json`
# (the o200k pattern). It differs from the GPT-2 split in `BytePairTokenizer`
# on digit runs (`\p{N}{1,3}` vs ` ?\p{N}+`), on whitespace before a word
# (`\s+(?!\S)`) and on case-aware word boundaries. The model was trained on
# these ids, so they matter for the predicted labels.
O200K_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.OpenAIPrivacyFilterTokenizer",
        "keras_hub.models.OpenAIPrivacyFilterTokenizer",
    ]
)
class OpenAIPrivacyFilterTokenizer(BytePairTokenizer):
    """An OpenAI Privacy Filter tokenizer using Byte-Pair Encoding.

    This tokenizer layer provides an implementation of a BPE tokenizer
    for the OpenAI Privacy Filter model. The Python workflow splits text with
    the o200k regex the model was trained with, so its ids match the
    Hugging Face tokenizer. The TensorFlow workflow (`tf.data` pipelines and
    `tf.function` tracing) keeps the GPT-2 split of `BytePairTokenizer`,
    because RE2 cannot express the o200k pattern; it is an approximation
    that differs from the Python workflow on digit runs and on runs of
    whitespace before a word.

    Args:
        vocabulary: string or dict, maps token to integer ids.
        merges: string or list, contains the merge rule.
    """

    backbone_cls = OpenAIPrivacyFilterBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        self.start_token_id = None
        self._warned_tf_path = False
        self._add_special_token("<|endoftext|>", "pad_token")
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

    def _tokenize_tf(self, inputs):
        if not getattr(self, "_warned_tf_path", False):
            warnings.warn(
                "OpenAIPrivacyFilterTokenizer is taking the tf.data path. "
                "This uses an approximated regex that differs from the model's "
                "training regex on digit runs and whitespace, which may affect "
                "predictions."
            )
            self._warned_tf_path = True
        return super()._tokenize_tf(inputs)

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        super()._set_vocabulary_and_merges_tokenizers(vocabulary, merges)
        if getattr(self, "_tokenizer", None) is None:
            return
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=tokenizers.Regex(O200K_SPLIT_PATTERN),
                    behavior="isolated",
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=False, use_regex=False
                ),
            ]
        )
