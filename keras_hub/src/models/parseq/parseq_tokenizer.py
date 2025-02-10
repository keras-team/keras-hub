import re

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

PARSEQ_VOCAB = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"
    "\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export(
    [
        "keras_hub.tokenizers.PARSeqTokenizer",
        "keras_hub.models.PARSeqTokenizer",
    ]
)
class PARSeqTokenizer(tokenizer.Tokenizer):
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
        self._add_special_token("[E]", "start_token")
        self._add_special_token("[B]", "end_token")
        self._add_special_token("[P]", "pad_token")
        super().__init__(dtype=dtype, **kwargs)
        self.vocabulary = vocabulary
        self.target_charset = tf.convert_to_tensor(vocabulary, dtype=tf.string)
        self.lowercase_only = self.target_charset == tf.strings.lower(
            self.target_charset
        )
        self.uppercase_only = self.target_charset == tf.strings.upper(
            self.target_charset
        )
        escaped_charset = re.escape(vocabulary)  # Escape for safe regex
        self.unsupported_regex = f"[^{escaped_charset}]"
        self._itos = (self.start_token,) + tuple(vocabulary) + (self.end_token,)
        self._stoi = {s: i for i, s in enumerate(self._itos)}

        # Create lookup tables.
        self.char_to_id = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self._stoi.keys()),
                values=list(self._stoi.values()),
                key_dtype=tf.string,
                value_dtype=tf.int32,
            ),
            default_value=0,
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

        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.max_label_length = max_label_length

    def id_to_token(self, id):
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return self._itos[id]

    def token_to_id(self, token):
        return self._stoi[token]

    def _preprocess(self, label):
        """Performs preprocessing include only characters from ASCII."""
        if self.remove_whitespace:
            label = tf.strings.regex_replace(label, r"\s+", "")

        if self.normalize_unicode:
            label = tf.strings.regex_replace(label, "[^!-~]", "")

        if self.lowercase_only:
            label = tf.strings.lower(label)
        elif self.uppercase_only:
            label = tf.strings.upper(label)

        label = tf.strings.regex_replace(label, self.unsupported_regex, "")
        label = tf.strings.substr(label, 0, self.max_label_length)

        return label

    @preprocessing_function
    def tokenize(self, inputs):
        self._check_vocabulary()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        inputs = tf.map_fn(self._preprocess, inputs, dtype=tf.string)

        if tf.size(inputs) > 0:
            chars = tf.strings.unicode_split(inputs, "UTF-8")
            token_ids = self.char_to_id.lookup(chars)
            token_ids = tf.cast(token_ids, dtype=tf.int32)
        else:
            token_ids = tf.ragged.constant([], dtype=tf.int32)

        return token_ids

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return len(self.vocabulary)

    def _check_vocabulary(self):
        if self.vocabulary is None:
            raise ValueError(
                "No vocabulary has been set for PARSeqTokenizer. Make sure "
                "to pass a `vocabulary` argument when creating the layer."
            )
