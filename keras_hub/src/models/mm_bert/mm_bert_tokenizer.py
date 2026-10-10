"""KerasHub tokenizer for mmBERT."""

import functools
import json
import os

import regex as re
import tokenizers
from tokenizers import decoders
from tokenizers import models
from tokenizers import normalizers
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mm_bert.mm_bert_backbone import MMBertBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import VOCAB_FILENAME
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.byte_pair_tokenizer import create_static_hashtable
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None

# Added tokens are isolated with this sentinel before the text is normalized,
# like the reference tokenizer, which declares them with `normalized: false`.
TOKEN_SENTINEL = "\x03"
BYTE_TOKEN_FORMAT = "<0x{:02X}>"
TOKEN_ATOM_PATTERN = f"{TOKEN_SENTINEL}[^{TOKEN_SENTINEL}]*{TOKEN_SENTINEL}"
ATOM_PATTERN = f"{TOKEN_ATOM_PATTERN}|▁[^▁{TOKEN_SENTINEL}]*"
# The merges are saved as JSON because some of them contain newlines, which
# one line per merge cannot represent.
MERGES_JSON_FILENAME = "merges.json"


@functools.lru_cache(maxsize=None)
def _added_token_pattern(tokens):
    """Compile a regex that matches any of `tokens`, longest first."""
    return re.compile(
        "("
        + "|".join(
            re.escape(token) for token in sorted(tokens, key=len, reverse=True)
        )
        + ")"
    )


@keras_hub_export(
    [
        "keras_hub.tokenizers.MMBertTokenizer",
        "keras_hub.models.MMBertTokenizer",
    ]
)
class MMBertTokenizer(BytePairTokenizer):
    """mmBERT byte-level BPE tokenizer.

    mmBERT ships the Gemma-2 SentencePiece vocabulary converted to a byte-level
    byte-pair vocabulary, along with the preprocessing pipeline of that
    conversion:

    * a `Replace(" ", "▁")` normalizer,
    * a `Metaspace(replacement="▁", prepend_scheme="always")` pre-tokenizer,
      which prefixes every word with `▁`,
    * 249 added tokens that are matched in the unnormalized text, and so split
      the input (this is why the character after a newline is tokenized as
      `▁b`, while the character after a tab is not),
    * byte fallback: a character that the vocabulary has no piece for is
      tokenized as one `<0xXX>` piece per byte of its UTF-8 encoding.

    This tokenizer configures the special token defaults required by mmBERT
    (`<pad>`, `<mask>`, `<bos>`, `<eos>` and `<unk>`) and implements the
    pipeline above for both the TensorFlow graph path and the `tokenizers`
    path.

    Args:
        vocabulary: dict or string. A dictionary mapping string tokens to
            integer IDs, or a file path to a json-serialized vocabulary map.
        merges: list or string. A list of byte pair merge rule strings, or a
            file path to a text merge rule list. Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent
            `BytePairTokenizer` class.

    Examples:
    ```python
    import keras_hub

    # Load the tokenizer directly from the original checkpoint.
    tokenizer = keras_hub.models.MMBertTokenizer.from_preset(
        "hf://jhu-clsp/mmBERT-base"
    )

    # Encode raw text strings to integer ID tokens.
    token_ids = tokenizer("The quick brown fox.")
    ```
    """

    backbone_cls = MMBertBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        pad_token = "<pad>"
        mask_token = "<mask>"
        bos_token = "<bos>"
        eos_token = "<eos>"
        unk_token = "<unk>"

        # Registered before `super().__init__()` so that
        # `_update_special_token_ids` sees them and raises on a vocabulary
        # that is missing one, rather than silently leaving the id `None`.
        self._add_special_token(pad_token, "pad_token")
        self._add_special_token(mask_token, "mask_token")
        self._add_special_token(bos_token, "bos_token")
        self._add_special_token(eos_token, "eos_token")
        self._add_special_token(unk_token, "unk_token")

        unsplittable_tokens = list(kwargs.pop("unsplittable_tokens", []))

        for token in (
            pad_token,
            mask_token,
            bos_token,
            eos_token,
            unk_token,
        ):
            if token not in unsplittable_tokens:
                unsplittable_tokens.append(token)

        kwargs["unsplittable_tokens"] = unsplittable_tokens

        # `▁` is added by the pipeline below.
        kwargs["add_prefix_space"] = False

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        self.file_assets = [VOCAB_FILENAME, MERGES_JSON_FILENAME]

    def save_assets(self, dir_path):
        """Save the vocabulary and the merge table as JSON."""
        os.makedirs(dir_path, exist_ok=True)

        with open(
            os.path.join(dir_path, VOCAB_FILENAME), "w", encoding="utf-8"
        ) as file:
            file.write(json.dumps(dict(self.vocabulary)))
        with open(
            os.path.join(dir_path, MERGES_JSON_FILENAME), "w", encoding="utf-8"
        ) as file:
            json.dump(list(self.merges), file)

    def load_assets(self, dir_path):
        """Load the vocabulary and the merge table."""
        vocabulary_path = os.path.join(dir_path, VOCAB_FILENAME)
        merges_path = os.path.join(dir_path, MERGES_JSON_FILENAME)

        with open(merges_path, encoding="utf-8") as file:
            merges = json.load(file)

        self.set_vocabulary_and_merges(vocabulary_path, merges)

    def _set_vocabulary_and_merges_tf(self, vocabulary, merges):
        super()._set_vocabulary_and_merges_tf(vocabulary, merges)

        # The byte fallback pieces of the vocabulary, e.g. `<0xE4>`.
        byte_tokens = [BYTE_TOKEN_FORMAT.format(i) for i in range(256)]
        raw_bytes = [bytes([i]) for i in range(256)]

        self.byte_to_token = create_static_hashtable(
            raw_bytes, byte_tokens, default=""
        )
        self.token_to_byte = create_static_hashtable(
            byte_tokens, raw_bytes, default=""
        )

    def _set_vocabulary_and_merges_tokenizers(self, vocabulary, merges):
        super()._set_vocabulary_and_merges_tokenizers(vocabulary, merges)

        merges_pairs = []

        for merge in merges:
            merge = str(merge)

            if "#version:" in merge.lstrip():
                continue

            merges_pairs.append(tuple(merge.split(" ")))

        # The BPE model, normalizer, pre-tokenizer and decoder below are the
        # ones declared by the checkpoint's `tokenizer.json`.
        self._tokenizer = tokenizers.Tokenizer(
            models.BPE(
                vocab=vocabulary,
                merges=merges_pairs,
                unk_token=self.unk_token,
                fuse_unk=True,
                byte_fallback=True,
            )
        )

        if self.unsplittable_tokens:
            self._tokenizer.add_special_tokens(list(self.unsplittable_tokens))

        if self.mask_token:
            # `<mask>` is the one token declared with `lstrip=True` in the
            # checkpoint's `tokenizer_config.json`.
            self._tokenizer.add_tokens(
                [tokenizers.AddedToken(self.mask_token, lstrip=True)]
            )

        self._tokenizer.normalizer = normalizers.Replace(" ", "▁")
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme="always"
        )
        self._tokenizer.decoder = decoders.Sequence(
            [
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )

    def _split_atoms(self, inputs):
        """Split a batch of strings into Metaspace atoms.

        Added tokens are matched in the unnormalized text and isolated, and
        every atom that does not already start with `▁` is prefixed with one,
        which is what `Metaspace(prepend_scheme="always")` does to each of the
        pieces it produces.
        """
        text = inputs

        if self.unsplittable_tokens:
            # Longest token first, so that e.g. `"\n\n"` wins over `"\n"`.
            pattern = "|".join(
                re.escape(token)
                for token in sorted(
                    self.unsplittable_tokens, key=len, reverse=True
                )
            )
            text = tf.strings.regex_replace(
                text,
                "(" + pattern + ")",
                TOKEN_SENTINEL + "\\1" + TOKEN_SENTINEL,
            )

            # `<mask>` is declared with `lstrip=True` in the checkpoint's
            # `tokenizer_config.json`, so the whitespace before it is dropped.
            # Wrapping keeps e.g. the `"\n\n"` added token intact.
            mask = TOKEN_SENTINEL + self.mask_token + TOKEN_SENTINEL
            text = tf.strings.regex_replace(
                text, r"\s+" + re.escape(mask), mask
            )

        text = tf.strings.regex_replace(text, " ", "▁")
        atoms = tf_text.regex_split(text, ATOM_PATTERN, ATOM_PATTERN)

        while atoms.shape.rank > 2:
            atoms = atoms.merge_dims(1, 2)

        flat_atoms = atoms.flat_values
        is_token = tf.strings.regex_full_match(flat_atoms, TOKEN_ATOM_PATTERN)

        # Unwrap the isolated added tokens.
        flat_lengths = tf.strings.length(flat_atoms)
        flat_atoms = tf.where(
            is_token,
            tf.strings.substr(
                flat_atoms,
                tf.ones_like(flat_lengths),
                tf.maximum(flat_lengths - 2, 0),
            ),
            flat_atoms,
        )

        needs_prefix = tf.logical_and(
            tf.logical_not(is_token),
            tf.logical_and(
                tf.not_equal(flat_atoms, ""),
                tf.logical_not(tf.strings.regex_full_match(flat_atoms, "▁.*")),
            ),
        )

        flat_atoms = tf.where(
            needs_prefix,
            tf.strings.join(["▁", flat_atoms]),
            flat_atoms,
        )

        return tf.RaggedTensor.from_row_splits(flat_atoms, atoms.row_splits)

    def _bpe_merge_and_update_cache_tf(self, tokens):
        """Byte-pair merge unseen tokens, with byte fallback.

        The parent maps every byte of a word through `byte2unicode`, which
        cannot express this vocabulary: mmBERT spells an unmergeable byte as a
        `<0xXX>` piece, so a character that is not in the vocabulary falls back
        to the pieces of its UTF-8 encoding.
        """
        chars = tf.strings.unicode_split(tokens, "UTF-8")
        flat_chars = chars.flat_values
        byte_pieces = tf.strings.reduce_join(
            self.byte_to_token.lookup(tf.strings.bytes_split(flat_chars)),
            axis=1,
            separator=" ",
        )
        is_known = tf.not_equal(self.token_to_id_map.lookup(flat_chars), -1)
        words = tf.RaggedTensor.from_row_splits(
            tf.where(is_known, flat_chars, byte_pieces),
            chars.row_splits,
        )
        tokenized_words = self._bpe_merge_tf(words)

        # For each word, join all its tokens by a whitespace for hashing.
        tokenized_words = tf.strings.reduce_join(
            tokenized_words, axis=1, separator=" "
        )
        self.cache.insert(tokens, tokenized_words)

    def tokenize(self, inputs):
        inputs = self._lstrip_mask_token_space(inputs)
        return super().tokenize(inputs)

    def _lstrip_mask_token_space(self, inputs):
        """Drop the whitespace before `<mask>` in plain string inputs.

        Strings are tokenized by the `tokenizers` pipeline, while tensors go
        through `_tokenize_tf`, which trims the whitespace itself.
        """
        if isinstance(inputs, str):
            return self._lstrip_mask_token_text(inputs)

        if isinstance(inputs, (list, tuple)) and all(
            isinstance(item, str) for item in inputs
        ):
            return [self._lstrip_mask_token_text(item) for item in inputs]

        return inputs

    def _lstrip_mask_token_text(self, text):
        """Trim the whitespace in front of every `<mask>` token.

        The reference tokenizer declares `<mask>` with `lstrip=True`. Older
        `tokenizers` releases ignore that flag for tokens that are also
        special tokens, so the trimming is done here. Added tokens are matched
        first, which keeps the whitespace of tokens like `"\n\n"` intact.
        """
        if self.mask_token not in text:
            return text

        if not self.unsplittable_tokens:
            return re.sub(
                r"\s+(?=" + re.escape(self.mask_token) + r")", "", text
            )

        # Plain text pieces land on the even indices of the split, matched
        # added tokens on the odd ones.
        atoms = _added_token_pattern(tuple(self.unsplittable_tokens)).split(
            text
        )

        for index in range(0, len(atoms) - 1, 2):
            if atoms[index + 1] == self.mask_token:
                atoms[index] = atoms[index].rstrip()

        return "".join(atoms)

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0

        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        if inputs.shape.rank > 1:
            raise ValueError(
                "`tokenize()` inputs should be a string, list of strings, or "
                f"string tensor with rank < 2. Received: {inputs}"
            )

        raw_tokens = self._split_atoms(inputs)

        token_row_splits = raw_tokens.row_splits
        flat_tokens = raw_tokens.flat_values

        # Check cache.
        cache_lookup = self.cache.lookup(flat_tokens)
        cache_mask = cache_lookup == ""
        has_unseen_words = tf.math.reduce_any(
            (cache_lookup == "") & (flat_tokens != "")
        )

        def process_unseen_tokens():
            unseen_tokens = tf.boolean_mask(flat_tokens, cache_mask)
            self._bpe_merge_and_update_cache_tf(unseen_tokens)
            return self.cache.lookup(flat_tokens)

        # If `has_unseen_words == True`, it means not all tokens are in cache,
        # we will process the unseen tokens. Otherwise return the cache lookup.
        tokenized_words = tf.cond(
            has_unseen_words,
            process_unseen_tokens,
            lambda: cache_lookup,
        )
        tokens = tf.strings.split(tokenized_words, sep=" ")

        if self.compute_dtype != tf.string:
            # Encode merged tokens.
            tokens = self.token_to_id_map.lookup(tokens)

        # Unflatten to match input.
        tokens = tf.RaggedTensor.from_row_splits(
            tokens.flat_values,
            tf.gather(tokens.row_splits, token_row_splits),
        )

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        # Convert to a dense output if input in scalar
        if unbatched:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])

        return tokens

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        inputs = tf.cast(inputs, self.dtype)
        tokens = self.id_to_token_map.lookup(inputs)
        flat_tokens = tokens.flat_values

        # `<0xXX>` pieces hold raw bytes, so replace them by their byte before
        # joining the sequence back together.
        raw_bytes = self.token_to_byte.lookup(flat_tokens)
        is_byte = tf.not_equal(raw_bytes, "")
        text = tf.strings.reduce_join(
            tf.RaggedTensor.from_row_splits(
                tf.where(is_byte, raw_bytes, flat_tokens),
                tokens.row_splits,
            ),
            axis=-1,
            separator="",
        )
        outputs = tf.strings.regex_replace(text, "▁", " ")

        if unbatched:
            outputs = tf.squeeze(outputs, 0)

        return outputs

    @property
    def start_token_id(self):
        return self.bos_token_id

    @property
    def end_token_id(self):
        return self.eos_token_id
