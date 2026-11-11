import re

import tokenizers
from tokenizers import normalizers
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gpt_neo_x.gpt_neo_x_backbone import GPTNeoXBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.byte_pair_tokenizer import create_static_hashtable
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None

# A private use character marking a split point in the TensorFlow path. It is
# removed from the text, so it is assumed not to appear in any input.
_SPLIT_MARKER = "\ue000"
# Unicode whitespace, as a regex class body. This is what `\s` means in the
# GPT-2 pre-tokenizer; RE2's `\s` is ASCII only and leaves out the vertical
# tab.
_WHITESPACE = (
    r"\s\x{0b}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}"
    r"\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}"
)
# The GPT-2 pre-tokenizer pattern, without its lookahead. See `_tokenize_tf`.
_SPLIT_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+"
    + rf"| ?[^{_WHITESPACE}\p{{L}}\p{{N}}{_SPLIT_MARKER}]+|[{_WHITESPACE}]+"
)


def _whitespace_run_tokens(vocabulary):
    """Vocabulary tokens made only of two or more spaces, longest first."""
    runs = [t for t in vocabulary if len(t) > 1 and not t.strip(" ")]
    return sorted(runs, key=len, reverse=True)


def _fill_rows(rows, values, mask):
    """Set row `i` of the ragged `rows` to `[values[i]]` where `mask[i]`.

    The masked rows must be empty.
    """
    filled = tf.RaggedTensor.from_row_lengths(
        tf.boolean_mask(values, mask), tf.cast(mask, "int64")
    )
    return tf.concat([rows, filled], axis=1)


@keras_hub_export(
    [
        "keras_hub.tokenizers.GPTNeoXTokenizer",
        "keras_hub.models.GPTNeoXTokenizer",
    ]
)
class GPTNeoXTokenizer(BytePairTokenizer):
    """A GPTNeoX tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by GPTNeoX
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a GPTNeoX preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
    """

    backbone_cls = GPTNeoXBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # GPTNeoX uses the same start as end token, i.e., "<|endoftext|>".
        self._add_special_token("<|endoftext|>", "end_token")
        self._add_special_token("<|endoftext|>", "start_token")
        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

    def set_vocabulary_and_merges(self, vocabulary, merges):
        super().set_vocabulary_and_merges(vocabulary, merges)
        vocabulary = self.vocabulary or {}
        # GPT-NeoX and Pythia vocabularies put "<|endoftext|>" at id 0 and
        # have a dedicated "<|padding|>" token. Padding with id 0 there would
        # make every padded position look like an end token, and
        # `generate()` would stop before producing anything.
        self.pad_token_id = vocabulary.get("<|padding|>", 0)
        # These vocabularies also have one token per run of 2 to 24 spaces,
        # which is matched in the text before byte-pair encoding. This is the
        # regex for them, longest first.
        self._whitespace_run_pattern = "|".join(
            f" {{{len(run)}}}" for run in _whitespace_run_tokens(vocabulary)
        )

    def _keep_padding_token_whole(self, vocabulary):
        # Match "<|padding|>" in text as one token, like "<|endoftext|>".
        if (
            "<|padding|>" in vocabulary
            and "<|padding|>" not in self.unsplittable_tokens
        ):
            self.unsplittable_tokens = [
                *self.unsplittable_tokens,
                "<|padding|>",
            ]

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        self._keep_padding_token_whole(vocabulary)
        super()._set_vocabulary_and_merges_tokenizers(vocabulary, merges)
        # Use the normalizer and the GPT-2 pre-tokenizer that GPT-NeoX and
        # Pythia checkpoints declare in `tokenizer.json`.
        self._tokenizer.normalizer = normalizers.NFC()
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=self.add_prefix_space, use_regex=True
        )
        self._tokenizer.add_tokens(
            [
                tokenizers.AddedToken(token, normalized=True)
                for token in _whitespace_run_tokens(vocabulary)
            ]
        )

    def _set_vocabulary_and_merges_tf(self, vocabulary, merges):
        self._keep_padding_token_whole(vocabulary)
        super()._set_vocabulary_and_merges_tf(vocabulary, merges)
        # Whitespace run tokens are stored as raw spaces. Detokenization
        # decodes byte-level characters, so spell them that way there.
        self.id_to_token_map = create_static_hashtable(
            list(self.vocabulary.values()),
            [token.replace(" ", "Ġ") for token in self.vocabulary],
            default="",
        )

    def _split_words_tf(self, inputs):
        """Split a batch of strings into words, as the GPT-2 tokenizer does.

        `BytePairTokenizer` splits whitespace the way Llama 3 does. This
        follows GPT-2 and matches the unsplittable and whitespace run tokens
        first, as the Python workflow does.
        """
        whole = [re.escape(token) for token in self.unsplittable_tokens]
        whole = "|".join(filter(None, [*whole, self._whitespace_run_pattern]))
        # The marker never survives a split, so it stands in for "no match".
        pieces = tf_text.regex_split(inputs, whole or _SPLIT_MARKER, whole)
        flat_pieces = pieces.flat_values
        is_whole = tf.strings.regex_full_match(
            flat_pieces, whole or _SPLIT_MARKER
        )

        text = tf.where(is_whole, "", flat_pieces)
        if self.add_prefix_space:
            # As in the Python workflow, every piece of text that does not
            # start with a space gets one.
            text = tf.strings.regex_replace(text, "^([^ ])", r" \1")
        # GPT-2 matches whitespace with `\s+(?!\S)|\s+`: a run of whitespace
        # followed by text gives up its last character. RE2 has no
        # lookahead, so mark that split point first.
        text = tf.strings.regex_replace(
            text,
            rf"([{_WHITESPACE}])([{_WHITESPACE}])([^{_WHITESPACE}])",
            rf"\1{_SPLIT_MARKER}\2\3",
        )
        words = tf_text.regex_split(
            text, f"{_SPLIT_PATTERN}|{_SPLIT_MARKER}", _SPLIT_PATTERN
        )
        words = _fill_rows(words, flat_pieces, is_whole)
        return pieces.with_flat_values(words).merge_dims(1, 2)

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs = tf_text.normalize_utf8(tf.convert_to_tensor(inputs), "NFC")
        if inputs.shape.rank > 1:
            return super()._tokenize_tf(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        words = self._split_words_tf(inputs)
        flat_words = words.flat_values

        # Byte-pair encode each word, as the base class does.
        cache_lookup = self.cache.lookup(flat_words)
        cache_mask = cache_lookup == ""

        def process_unseen_words():
            unseen_words = tf.boolean_mask(flat_words, cache_mask)
            self._bpe_merge_and_update_cache_tf(unseen_words)
            return self.cache.lookup(flat_words)

        tokenized_words = tf.cond(
            tf.math.reduce_any(cache_mask & (flat_words != "")),
            process_unseen_words,
            lambda: cache_lookup,
        )
        # The cache joins the tokens of a word with spaces, so a whitespace
        # run token cannot go through it. Those words are already one token.
        is_run = tf.strings.regex_full_match(
            flat_words, self._whitespace_run_pattern or _SPLIT_MARKER
        )
        tokens = tf.strings.split(
            tf.where(is_run, "", tokenized_words), sep=" "
        )
        # Splitting "" on an explicit separator gives one empty token.
        tokens = tf.ragged.boolean_mask(tokens, tokens != "")
        tokens = _fill_rows(tokens, flat_words, is_run)
        padding = ""
        if is_int_dtype(self.compute_dtype):
            tokens = self.token_to_id_map.lookup(tokens)
            padding = self.pad_token_id
        tokens = tf.RaggedTensor.from_row_splits(
            tokens.flat_values,
            tf.gather(tokens.row_splits, words.row_splits),
        )

        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(default_value=padding, shape=output_shape)
        if unbatched:
            tokens = tf.squeeze(tokens, 0)
        return tokens
