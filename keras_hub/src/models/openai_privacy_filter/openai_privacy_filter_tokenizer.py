import tokenizers
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import SPECIAL_WHITESPACES
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.byte_pair_tokenizer import split_strings_for_bpe
from keras_hub.src.utils.tensor_utils import tf

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

# The same pattern for the TensorFlow workflow. RE2 has no lookahead, so the
# `\s+(?!\S)` alternative is emulated with the `६` marker: `_split_strings_tf`
# inserts `६` before horizontal non-space whitespace preceding a letter/mark,
# and `split_strings_for_bpe` inserts `६` before spaces preceding non-spaces
# (excluding `६`), consumed by the second-to-last alternative and removed again
# after splitting. `६` is therefore excluded from the character classes.
O200K_TF_SPLIT_PATTERN_1 = (
    r"[^\r\n\p{L}\p{N}६]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|[^\r\n\p{L}\p{N}६]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    r"|\p{N}{1,3}"
    rf"| ?[^\s\p{{L}}\p{{N}}६{SPECIAL_WHITESPACES}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    rf"|[\s{SPECIAL_WHITESPACES}]+[\n\r\t\f६{SPECIAL_WHITESPACES}]"
    rf"|[\s{SPECIAL_WHITESPACES}]+"
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
    `tf.function` tracing) uses an RE2 form of the same pattern and gives
    the same ids, with these known differences:

    - Whitespace at the very end of a text is split differently.
    - Whitespace that contains a carriage return, a vertical tab or a
      non-ASCII space (such as a no-break space, em space or line
      separator) can be split differently.
    - `६` (U+096C), which the splitter uses internally as a marker, is
      dropped from the text.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.OpenAIPrivacyFilterTokenizer.from_preset(
        "openai_privacy_filter_en"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```

    Args:
        vocabulary: string or dict, maps token to integer ids.
        merges: string or list, contains the merge rule.
        add_prefix_space: bool. Must be `False` (the default). The o200k
            split runs before the prefix space is added, so `True` would make
            the Python and TensorFlow workflows disagree.
    """

    backbone_cls = OpenAIPrivacyFilterBackbone

    def __init__(self, vocabulary=None, merges=None, **kwargs):
        if kwargs.get("add_prefix_space", False):
            raise ValueError(
                "`add_prefix_space=True` is not supported by "
                "`OpenAIPrivacyFilterTokenizer`. The o200k split runs before "
                "the prefix space is added, so the Python and TensorFlow "
                "workflows would return different ids. Received: "
                f"add_prefix_space={kwargs['add_prefix_space']}"
            )
        self.start_token_id = None
        self._add_special_token("<|endoftext|>", "pad_token")
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

    def _split_strings_tf(self, inputs):
        inputs = tf.strings.regex_replace(
            inputs,
            rf"([\t\f\v{SPECIAL_WHITESPACES}])([\p{{L}}\p{{M}}])",
            r"६\1\2",
        )
        return split_strings_for_bpe(
            inputs,
            self.unsplittable_tokens,
            split_pattern_1=O200K_TF_SPLIT_PATTERN_1,
        )

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        super()._set_vocabulary_and_merges_tokenizers(vocabulary, merges)
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
