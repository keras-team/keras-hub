import numpy as np
import tokenizers
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
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


@keras_hub_export(
    [
        "keras_hub.tokenizers.UnlimitedOCRTokenizer",
        "keras_hub.models.UnlimitedOCRTokenizer",
    ]
)
class UnlimitedOCRTokenizer(BytePairTokenizer):
    """An Unlimited-OCR byte-level BPE tokenizer.

    The tokenizer uses the three regular expression rules of the checkpoint
    `tokenizer.json` as its pre-tokenizer. The tokenizer keeps the special
    tokens as single tokens. The checkpoint converter also keeps every added
    token of `tokenizer.json`, such as `"<|det|>"`, as a single token.

    The `generate()` text drops the special tokens: the start, end, padding
    and image tokens, and the chat tokens `"<|User|>"`, `"<|Assistant|>"` and
    `"<|EOT|>"`. The text keeps the layout markers, such as `"<|det|>"` and
    `"<|/det|>"`. In each block of the output, the markers enclose the
    category and the bounding box. The recognized text follows the closing
    marker. The text keeps the markers on purpose, so that a caller can
    separate the category and box from the text.

    Args:
        vocabulary: dict or `None`. Maps token strings to integer ids.
        merges: list or `None`. The BPE merge rules.
        bos_token: str. The beginning-of-sequence token.
        eos_token: str. The end-of-sequence token.
        pad_token: str. The padding token.
        image_token: str. The placeholder token of the image features.
            Defaults to `"<image>"`.
        user_token: str. The chat token that starts a user turn.
            Defaults to `"<|User|>"`.
        assistant_token: str. The chat token that starts an assistant turn.
            Defaults to `"<|Assistant|>"`.
        eot_token: str. The chat token that ends a turn.
            Defaults to `"<|EOT|>"`.
        unsplittable_tokens: list or `None`. Strings that the tokenizer never
            splits, such as `"<|det|>"` and `"<|/det|>"`. The checkpoint
            converter passes the added tokens of `tokenizer.json`. A
            tokenizer without this argument splits the markers.

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
        user_token="<|User|>",
        assistant_token="<|Assistant|>",
        eot_token="<|EOT|>",
        **kwargs,
    ):
        self._add_special_token(bos_token, "start_token")
        self._add_special_token(eos_token, "end_token")
        self._add_special_token(pad_token, "pad_token")
        self._add_special_token(image_token, "image_token")
        self._add_special_token(user_token, "user_token")
        self._add_special_token(assistant_token, "assistant_token")
        self._add_special_token(eot_token, "eot_token")
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)

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

    def _encode_strings(self, strings):
        """Encode a flat list of strings to one list of ids or tokens each."""
        strings = [
            s.decode("utf-8") if isinstance(s, bytes) else s for s in strings
        ]
        encodings = self._tokenizer.encode_batch(
            strings, add_special_tokens=False
        )
        if is_string_dtype(self.compute_dtype):
            return [e.tokens for e in encodings]
        return [e.ids for e in encodings]

    def _tokenize_tokenizers(self, inputs):
        # The base class reads each row of a rank 2 input as a sentence pair.
        # This method encodes each string of a rank 2 or higher input alone.
        if tf is not None and isinstance(inputs, tf.Tensor):
            array = inputs.numpy()
        else:
            array = np.asarray(inputs, dtype=object)
        if array.ndim < 2:
            return super()._tokenize_tokenizers(inputs)
        self._maybe_initialized_tokenizers()
        rows = self._encode_strings(array.reshape(-1).tolist())
        if self.sequence_length:
            length = self.sequence_length
            pad_value = self._get_pad_value()
            rows = [
                row[:length] + [pad_value] * (length - len(row)) for row in rows
            ]
            if is_int_dtype(self.compute_dtype):
                return np.array(rows, dtype=self.compute_dtype).reshape(
                    array.shape + (length,)
                )
        # Restore the leading shape of the input as nested lists.
        flat_rows = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            flat_rows[i] = row
        return flat_rows.reshape(array.shape).tolist()

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        # The base class splits text with the GPT-2 pattern in the graph. That
        # gives other ids for this checkpoint. This method calls the
        # `tokenizers` backend from the graph instead.
        self._maybe_initialized_tokenizers()

        def _encode(string_tensor):
            rows = self._encode_strings(string_tensor.numpy().tolist())
            # `tf.ragged.constant([])` gives a dense tensor, so build the
            # ragged tensor from the row lengths.
            return tf.RaggedTensor.from_row_lengths(
                tf.constant(
                    [t for row in rows for t in row],
                    dtype=self.compute_dtype,
                    shape=[sum(len(row) for row in rows)],
                ),
                tf.constant([len(row) for row in rows], dtype="int64"),
            )

        inputs = tf.convert_to_tensor(inputs)
        rank = inputs.shape.rank
        if rank is None:
            raise ValueError(
                "`tokenize()` inputs must have a known rank. "
                f"Received: {inputs}"
            )
        # Flatten the input to one dimension for the backend.
        flat_inputs = tf.reshape(inputs, [-1])
        tokens = tf.py_function(
            _encode,
            [flat_inputs],
            Tout=tf.RaggedTensorSpec(
                shape=[None, None],
                dtype=self.compute_dtype,
                ragged_rank=1,
            ),
        )
        if self.sequence_length:
            tokens = tokens.to_tensor(
                shape=[None, self.sequence_length],
                default_value=self._get_pad_value(),
            )
            if rank != 1:
                shape = tf.concat([tf.shape(inputs), [self.sequence_length]], 0)
                tokens = tf.reshape(tokens, shape)
                tokens.set_shape(inputs.shape.concatenate(self.sequence_length))
            return tokens
        # Restore the leading shape of the input as ragged rows.
        if rank == 0:
            return tokens[0]
        for size in reversed(range(1, rank)):
            tokens = tf.RaggedTensor.from_uniform_row_length(
                tokens, tf.cast(tf.shape(inputs)[size], "int64")
            )
        return tokens

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        # The base class maps ids to bytes with the GPT-2 table. That table
        # drops the characters `｜` and `▁`. This method calls the
        # `tokenizers` backend from the graph instead.
        self._maybe_initialized_tokenizers()
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)

        def _decode(flat_values, row_splits):
            flat_values = flat_values.numpy().tolist()
            row_splits = row_splits.numpy().tolist()
            token_ids = [
                flat_values[start:end]
                for start, end in zip(row_splits[:-1], row_splits[1:])
            ]
            return tf.constant(
                self._tokenizer.decode_batch(
                    token_ids, skip_special_tokens=False
                ),
                dtype="string",
            )

        outputs = tf.py_function(
            _decode,
            [inputs.flat_values, inputs.row_splits],
            Tout=tf.string,
        )
        outputs.set_shape([None])
        if unbatched:
            outputs = outputs[0]
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "bos_token": self.start_token,
                "eos_token": self.end_token,
                "pad_token": self.pad_token,
                "image_token": self.image_token,
                "user_token": self.user_token,
                "assistant_token": self.assistant_token,
                "eot_token": self.eot_token,
            }
        )
        return config
