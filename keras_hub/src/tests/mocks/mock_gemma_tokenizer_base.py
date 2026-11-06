import keras
import numpy as np

from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.tensor_utils import assert_tf_installed
from keras_hub.src.utils.tensor_utils import canonicalize_python_string_inputs
from keras_hub.src.utils.tensor_utils import canonicalize_python_token_inputs
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import restore_outer_shape
from keras_hub.src.utils.tensor_utils import tf


class MockGemmaTokenizerBase(Tokenizer):
    """Shared base class for Mock Gemma tokenizers."""

    _strip_text = False
    _space_padded_tokens = ()
    _ASCII_WHITESPACE = " \t\n\r\x0b\x0c"

    def __init__(
        self,
        proto=None,
        sequence_length=None,
        dtype="int32",
        add_bos=False,
        add_eos=False,
        **kwargs,
    ):
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)

        self.proto = proto
        self.sequence_length = sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos

        self._setup_tokens()
        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")
        self._build_vocabulary_dicts()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proto": None,
                "sequence_length": self.sequence_length,
                "add_bos": self.add_bos,
                "add_eos": self.add_eos,
            }
        )
        return config

    def _build_vocabulary_dicts(self):
        self._python_string_to_id = {
            v: i for i, v in enumerate(self.vocabulary)
        }
        self._python_id_to_string = {
            i: v for i, v in enumerate(self.vocabulary)
        }

    def _maybe_initialized_tf(self):
        """Lazily builds the TF lookup tables so the mock constructs without
        TF; `tf.init_scope()` keeps them out of any enclosing trace."""
        if hasattr(self, "string_to_id"):
            return
        assert_tf_installed(self.__class__.__name__)
        with tf.init_scope():
            self.string_to_id = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    self.vocabulary, list(range(len(self.vocabulary)))
                ),
                default_value=3,
            )
            self.id_to_string = tf.lookup.StaticHashTable(
                tf.lookup.KeyValueTensorInitializer(
                    list(range(len(self.vocabulary))), self.vocabulary
                ),
                default_value="<unk>",
            )

    def vocabulary_size(self):
        return len(self.vocabulary)

    def get_vocabulary(self):
        return self.vocabulary

    def id_to_token(self, id):
        return self.vocabulary[id]

    def token_to_id(self, token):
        return self.vocabulary.index(token)

    def _tokenize_python(self, inputs):
        inputs, batched, outer_shape = canonicalize_python_string_inputs(inputs)

        batched_tokens = []
        for text in inputs:
            for token in self._space_padded_tokens:
                text = text.replace(token, f" {token} ")
            text = text.replace("  ", " ")
            if self._strip_text:
                text = text.strip(self._ASCII_WHITESPACE)

            sep_inputs = text.split(" ")
            tokens = [self._python_string_to_id.get(w, 3) for w in sep_inputs]

            if self.add_bos:
                tokens = [self.start_token_id] + tokens
            if self.add_eos:
                tokens = tokens + [self.end_token_id]

            if not is_int_dtype(self.compute_dtype):
                tokens = [self._python_id_to_string[t] for t in tokens]

            batched_tokens.append(tokens)

        if outer_shape is not None:
            return restore_outer_shape(batched_tokens, outer_shape)

        if not batched:
            if is_int_dtype(self.compute_dtype):
                return np.array(batched_tokens[0], dtype=self.compute_dtype)
            return batched_tokens[0]

        return batched_tokens

    def tokenize(self, inputs):
        if self._use_tf_workflow():
            return self._tokenize_tf(inputs)
        else:
            return self._tokenize_python(inputs)

    def _detokenize_python(self, inputs):
        inputs, batched = canonicalize_python_token_inputs(inputs)

        outputs_list = []
        for sample in inputs:
            strings = [
                self._python_id_to_string.get(id, "<unk>") for id in sample
            ]
            out_str = " ".join(strings)
            for token in [
                self.start_token,
                self.end_token,
                self.pad_token,
            ]:
                out_str = out_str.replace(token, "")
            out_str = out_str.strip(self._ASCII_WHITESPACE)
            outputs_list.append(out_str)

        if not batched:
            return outputs_list[0]
        return outputs_list

    def detokenize(self, inputs):
        if self._use_tf_workflow():
            return self._detokenize_tf(inputs)
        else:
            return self._detokenize_python(inputs)

    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.sequence_length,), dtype=self.compute_dtype
        )

    def _cast_tokens_tf(self, tokens):
        if is_int_dtype(self.compute_dtype):
            return tf.cast(tokens, self.compute_dtype)
        return self.id_to_string.lookup(tf.cast(tokens, "int32"))

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        inputs = tf.cast(inputs, "int32")

        outputs = self.id_to_string.lookup(inputs)
        outputs = tf.strings.reduce_join(outputs, axis=-1, separator=" ")

        for token in [
            self.start_token,
            self.end_token,
            self.pad_token,
        ]:
            outputs = tf.strings.regex_replace(outputs, token, "")

        outputs = tf.strings.strip(outputs)

        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs
