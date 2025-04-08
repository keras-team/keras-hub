import re

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

PARSEQ_VOCAB = (
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"
    "\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
)

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None


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
        if not isinstance(vocabulary, str):
            raise ValueError(
                "vocabulary must be string of unique characters. "
                f" Received:vocabulary={vocabulary}"
            )
        super().__init__(dtype=dtype, **kwargs)
        self.vocabulary = vocabulary
        self.lowercase_only = vocabulary == vocabulary.lower()
        self.uppercase_only = vocabulary == vocabulary.upper()
        escaped_charset = re.escape(vocabulary)  # Escape for safe regex
        self.unsupported_regex = f"[^{escaped_charset}]"
        self._itos = ("[E]",) + tuple(vocabulary) + ("[B]", "[P]")
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

        inputs = tf.map_fn(self._preprocess, inputs, dtype=tf.string)

        if tf.size(inputs) > 0:
            chars = tf.strings.unicode_split(inputs, "UTF-8")
            token_ids = self.char_to_id.lookup(chars)
        else:
            token_ids = tf.ragged.constant([], dtype=tf.int32)

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
