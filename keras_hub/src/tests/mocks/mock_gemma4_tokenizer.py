import re

import numpy as np

from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.tensor_utils import canonicalize_python_string_inputs
from keras_hub.src.utils.tensor_utils import canonicalize_python_token_inputs
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import restore_outer_shape
from keras_hub.src.utils.tensor_utils import tf


class MockGemma4Tokenizer(Tokenizer):
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

        self.vocabulary = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "<|image>",
            "<image|>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<|image|>",
            "the",
            "brown",
            "earth",
            "fox",
            "is",
            "quick",
            "round",
            "\n\n",
            "<|turn>",
            "<turn|>",
            "<|audio|>",
            "<|audio>",
            "<audio|>",
            "<|video|>",
            "<|video>",
            "<video|>",
        ]
        self._python_string_to_id = {
            v: i for i, v in enumerate(self.vocabulary)
        }
        self._python_id_to_string = {
            i: v for i, v in enumerate(self.vocabulary)
        }

        # Standard tokens.
        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")

        # Image placeholder token.
        self._add_special_token("<|image|>", "image_placeholder")

        # Boundary tokens used in the preprocessor.
        self._add_special_token("<|image>", "start_of_image_token")
        self._add_special_token("<image|>", "end_of_image_token")

        # Video tokens.
        self._add_special_token("<|video|>", "video_placeholder")
        self._add_special_token("<|video>", "start_of_video_token")
        self._add_special_token("<video|>", "end_of_video_token")

        self.sequence_length = sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.audio_placeholder_id = -1

    def _maybe_initialized_tf(self):
        """Builds the TF lookup tables the first time the TF path runs.

        The tables are not built in `__init__` so that this mock stays
        constructible when TensorFlow is absent. `tf.init_scope()` lifts
        their creation out of any enclosing `tf.function` trace (e.g. a
        `tf.data.Dataset.map`), which would otherwise capture them as
        symbolic tensors and fail on the second call.
        """
        if hasattr(self, "string_to_id"):
            return
        if tf is None:
            return
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

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        inputs = tf.strings.regex_replace(
            inputs,
            re.escape(self.start_of_image_token),
            f" {self.start_of_image_token} ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape(self.end_of_image_token),
            f" {self.end_of_image_token} ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape(self.image_placeholder),
            f" {self.image_placeholder} ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|audio>"),
            " <|audio> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<audio|>"),
            " <audio|> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|audio|>"),
            " <|audio|> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|video>"),
            " <|video> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<video|>"),
            " <video|> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|video|>"),
            " <|video|> ",
        )
        inputs = tf.strings.regex_replace(inputs, "  ", " ")
        inputs = tf.strings.strip(inputs)

        sep_inputs = tf.strings.split(inputs, sep=" ")
        tokens = self.string_to_id.lookup(sep_inputs)

        if self.add_bos:
            bos_tensor = tf.fill(
                value=self.start_token_id,
                dims=tokens.shape.as_list()[0:1] + [1],
            )
            tokens = tf.concat((bos_tensor, tokens), axis=-1)
        if self.add_eos:
            eos_tensor = tf.fill(
                value=self.end_token_id,
                dims=tokens.shape.as_list()[0:1] + [1],
            )
            tokens = tf.concat((tokens, eos_tensor), axis=-1)

        if unbatched:
            tokens = tf.squeeze(tokens, 0)

        return tokens

    def _tokenize_python(self, inputs):
        inputs, batched, outer_shape = canonicalize_python_string_inputs(inputs)

        batched_tokens = []
        for text in inputs:
            text = text.replace(
                self.start_of_image_token, f" {self.start_of_image_token} "
            )
            text = text.replace(
                self.end_of_image_token, f" {self.end_of_image_token} "
            )
            text = text.replace(
                self.image_placeholder, f" {self.image_placeholder} "
            )
            text = text.replace("<|audio>", " <|audio> ")
            text = text.replace("<audio|>", " <audio|> ")
            text = text.replace("<|audio|>", " <|audio|> ")
            text = text.replace("<|video>", " <|video> ")
            text = text.replace("<video|>", " <video|> ")
            text = text.replace("<|video|>", " <|video|> ")
            text = text.replace("  ", " ")
            text = text.strip()

            sep_inputs = text.split(" ")
            tokens = [self._python_string_to_id.get(w, 3) for w in sep_inputs]

            if self.add_bos:
                tokens = [self.start_token_id] + tokens
            if self.add_eos:
                tokens = tokens + [self.end_token_id]
            batched_tokens.append(tokens)

        if outer_shape is not None:
            return restore_outer_shape(batched_tokens, outer_shape)

        if not batched:
            return np.array(batched_tokens[0], dtype=self.compute_dtype)

        return batched_tokens

    def tokenize(self, inputs):
        if self._use_tf_workflow():
            return self._tokenize_tf(inputs)
        else:
            return self._tokenize_python(inputs)

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
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
            out_str = out_str.strip()
            outputs_list.append(out_str)

        if not batched:
            return outputs_list[0]
        return outputs_list

    def detokenize(self, inputs):
        if self._use_tf_workflow():
            return self._detokenize_tf(inputs)
        else:
            return self._detokenize_python(inputs)

    def __call__(self, inputs):
        return self.tokenize(inputs)
