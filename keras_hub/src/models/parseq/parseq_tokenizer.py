import os
import re
from typing import Iterable

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None

PARSEQ_VOCAB = list(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"
    "\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

VOCAB_FILENAME = "vocabulary.txt"


@keras_hub_export(
    [
        "keras_hub.tokenizers.PARSeqTokenizer",
        "keras_hub.models.PARSeqTokenizer",
    ]
)
class PARSeqTokenizer(tokenizer.Tokenizer):
    """A Tokenizer for PARSeq models, designed for OCR tasks.

    This tokenizer converts strings into sequences of integer IDs or string
    tokens, and vice-versa. It supports various preprocessing steps such as
    whitespace removal, Unicode normalization, and limiting the maximum label
    length. It also provides functionality to save and load the vocabulary
    from a file.

    Args:
        vocabulary: str. A string or iterable representing the vocabulary to
            use. If a string, it's treated as the path to a vocabulary file.
            If an iterable, it's treated as a list of characters forming
            the vocabulary. Defaults to `PARSEQ_VOCAB`.
        remove_whitespace: bool. Whether to remove whitespace characters from
            the input. Defaults to `True`.
        normalize_unicode: bool. Whether to normalize Unicode characters in the
            input using NFKD normalization and remove non-ASCII characters.
            Defaults to `True`.
        max_label_length: int. The maximum length of the tokenized output.
            Longer labels will be truncated. Defaults to `25`.
        dtype: str. The data type of the tokenized output. Must be an integer
            type (e.g., "int32") or a string type ("string").
            Defaults to `"int32"`.
        **kwargs: Additional keyword arguments passed to the base
            `keras.layers.Layer` constructor.
    """

    def __init__(
        self,
        vocabulary=PARSEQ_VOCAB,
        remove_whitespace=True,
        normalize_unicode=True,
        max_label_length=25,
        dtype="int32",
        **kwargs,
    ):
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )
        super().__init__(dtype=dtype, **kwargs)
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.max_label_length = max_label_length
        self.file_assets = [VOCAB_FILENAME]

        self.set_vocabulary(vocabulary)

    def save_assets(self, dir_path):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        with open(path, "w", encoding="utf-8") as file:
            for token in self.vocabulary:
                file.write(f"{token}\n")

    def load_assets(self, dir_path):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        self.set_vocabulary(path)

    def set_vocabulary(self, vocabulary):
        """Set the tokenizer vocabulary to a file or list of strings."""
        if vocabulary is None:
            self.vocabulary = None
            return

        if isinstance(vocabulary, str):
            with open(vocabulary, "r", encoding="utf-8") as file:
                self.vocabulary = [line.rstrip() for line in file]
                self.vocabulary = "".join(self.vocabulary)
        elif isinstance(vocabulary, Iterable):
            self.vocabulary = "".join(vocabulary)
        else:
            raise ValueError(
                "Vocabulary must be an file path or list of terms. "
                f"Received: vocabulary={vocabulary}"
            )

        self.lowercase_only = self.vocabulary == self.vocabulary.lower()
        self.uppercase_only = self.vocabulary == self.vocabulary.upper()
        escaped_charset = re.escape(self.vocabulary)  # Escape for safe regex
        self.unsupported_regex = f"[^{escaped_charset}]"
        self._itos = ("[E]",) + tuple(self.vocabulary) + ("[B]", "[P]")
        self._stoi = {s: i for i, s in enumerate(self._itos)}

        self._add_special_token("[B]", "start_token")
        self._add_special_token("[E]", "end_token")
        self._add_special_token("[P]", "pad_token")
        # Create lookup tables.
        self.char_to_id = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self._stoi.keys()),
                values=list(self._stoi.values()),
                key_dtype=tf.string,
                value_dtype=tf.int32,
            ),
            default_value=self._stoi["[E]"],
        )
        self.id_to_char = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self._stoi.values()),
                values=list(self._stoi.keys()),
                key_dtype=tf.int32,
                value_dtype=tf.string,
            ),
            default_value=self.pad_token,
        )

    def get_vocabulary(self):
        """Get the tokenizer vocabulary as a list of strings tokens."""
        return list(self.vocabulary)

    def id_to_token(self, id):
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return self._itos[id]

    def token_to_id(self, token):
        return self._stoi[token]

    def _preprocess(self, inputs):
        """Performs preprocessing include only characters from ASCII."""
        if self.remove_whitespace:
            inputs = tf.strings.regex_replace(inputs, r"\s+", "")

        if self.normalize_unicode:
            inputs = tf_text.normalize_utf8(inputs, normalization_form="NFKD")
            inputs = tf.strings.regex_replace(inputs, r"[^!-~]", "")

        if self.lowercase_only:
            inputs = tf.strings.lower(inputs)
        elif self.uppercase_only:
            inputs = tf.strings.upper(inputs)

        inputs = tf.strings.regex_replace(inputs, self.unsupported_regex, "")
        inputs = tf.strings.substr(inputs, 0, self.max_label_length)

        return inputs

    @preprocessing_function
    def tokenize(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        inputs = tf.map_fn(
            self._preprocess, inputs, fn_output_signature=tf.string
        )

        token_ids = tf.cond(
            tf.size(inputs) > 0,
            lambda: self.char_to_id.lookup(
                tf.strings.unicode_split(inputs, "UTF-8")
            ),
            lambda: tf.RaggedTensor.from_row_splits(
                values=tf.constant([], dtype=tf.int32),
                row_splits=tf.constant([0], dtype=tf.int64),
            ),
        )
        if unbatched:
            token_ids = tf.squeeze(token_ids, 0)
            tf.ensure_shape(token_ids, shape=[self.max_label_length])
        return token_ids

    @preprocessing_function
    def detokenize(self, inputs):
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        # tf-text sentencepiece does not handle int64.
        inputs = tf.cast(inputs, "int32")
        outputs = self.id_to_char.lookup(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        return len(self.vocabulary) + 3

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.max_label_length,),
            dtype=self.compute_dtype,
        )
