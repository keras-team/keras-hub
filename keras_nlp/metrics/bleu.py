# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BLEU metric implementation."""

import collections
import math

import tensorflow as tf
from tensorflow import keras

from keras_nlp.utils.tf_utils import tensor_to_list

REPLACE_SUBSTRINGS = [
    ("<skipped>", ""),
    ("-\n", ""),
    ("\n", " "),
    ("&quot;", '"'),
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
]


REGEX_PATTERNS = [
    # language-dependent part (assuming Western languages)
    (r"([\{-\~\[-\` -\&\(-\+\:-\@\/])", r" \1 "),
    # tokenize period and comma unless preceded by a digit
    (r"([^0-9])([\.,])", r"\1 \2 "),
    # tokenize period and comma unless followed by a digit
    (r"([\.,])([^0-9])", r" \1 \2"),
    # tokenize dash when preceded by a digit
    (r"([0-9])(-)", r"\1 \2 "),
    # If last character is "." or ",", add space.
    (r"[\.,]$", r" \0 \1"),
    # one space only between words
    (r"\s+", r" "),
]


@keras.utils.register_keras_serializable(package="keras_nlp")
class Bleu(keras.metrics.Metric):
    """BLEU metric.

    This class implements the BLEU metric. BLEU is generally used to evaluate
    machine translation systems. By default, this implementation replicates
    SacreBLEU, but user-defined tokenizers can be passed to deal with other
    languages.

    For BLEU score, we count the number of matching n-grams in the candidate
    translation and the reference text. We find the "clipped count" of matching
    n-grams so as to not give a high score to a (reference, prediction) pair
    with redundant, repeated tokens. Secondly, BLEU score tends to reward
    shorter predictions more, which is why a brevity penalty is applied to
    penalise short predictions. For more details, see the following article:
    https://cloud.google.com/translate/automl/docs/evaluate#bleu.

    Note on input shapes:
    For unbatched inputs, `y_pred` should be a tensor of shape `()`, and
    `y_true` should be a tensor of shape `(num_references,)`. For batched
    inputs, `y_pred` should be a tensor of shape `(batch_size,)`,
    and `y_true` should be a tensor of shape `(batch_size, num_references)`. In
    case of batched inputs, `y_true` can also be a ragged tensor of shape
    `(batch_size, None)` if different samples have different number of
    references.

    Args:
        tokenizer: callable. A function that takes a string `tf.RaggedTensor`
            (of any shape), and tokenizes the strings in the tensor. If the
            tokenizer is not specified, the default tokenizer is used. The
            default tokenizer replicates the behaviour of SacreBLEU's
            `"tokenizer_13a"` tokenizer
            (https://github.com/mjpost/sacrebleu/blob/v2.1.0/sacrebleu/tokenizers/tokenizer_13a.py).
        max_order: int. The maximum n-gram order to use. For example, if
            `max_order` is set to 3, unigrams, bigrams, and trigrams will be
            considered. Defaults to 4.
        smooth: bool. Whether to apply Lin et al. 2004 smoothing to the BLEU
            score. Adds 1 to the matched n-gram count (i.e., numerator) and 1
            to the total n-gram count (i.e., denominator) for every order while
            calculating precision. Defaults to False.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    References:
        - [Papineni et al., 2002](https://aclanthology.org/P02-1040/)
        - [SacreBLEU](https://github.com/mjpost/sacrebleu)
        - [Lin et al., 2004](https://aclanthology.org/P04-1077/)
    """

    def __init__(
        self,
        tokenizer=None,
        max_order=4,
        smooth=False,
        dtype=None,
        name="bleu",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if not tf.as_dtype(self.dtype).is_floating:
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        self.tokenizer = tokenizer
        self.max_order = max_order
        self.smooth = smooth

        self._matches = self.add_weight(
            shape=(self.max_order,),
            name="bleu_matches",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._possible_matches = self.add_weight(
            shape=(self.max_order,),
            name="bleu_possible_matches",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._translation_length = self.add_weight(
            name="bleu_translation_length",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._reference_length = self.add_weight(
            name="bleu_reference_length",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._bleu = self.add_weight(
            name="bleu",
            initializer="zeros",
            dtype=self.dtype,
        )

    def _tokenizer(self, inputs):
        """
        Tokenizes the input strings. By default, replicates the behaviour of
        SacreBLEU's default tokenizer, namely, `tokenizer_13a`.
        """
        if self.tokenizer:
            return self.tokenizer(inputs)

        for pattern, replacement in REPLACE_SUBSTRINGS + REGEX_PATTERNS:
            inputs = tf.strings.regex_replace(
                input=inputs,
                pattern=pattern,
                rewrite=replacement,
                replace_global=True,
                name=None,
            )
        inputs = tf.strings.split(inputs)
        return inputs

    def _get_ngrams(self, segment, max_order):
        """Extracts all n-grams up to a given maximum order from an input segment.

        Uses Python ops. Inspired from
        https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py.

        Args:
            segment: list. Text segment from which n-grams will be
                extracted.
            max_order: int. Maximum length in tokens of the n-grams returned
                by this method.
        """
        ngram_counts = collections.Counter()
        for order in range(1, max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i : i + order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def _corpus_bleu(
        self,
        reference_corpus,
        translation_corpus,
        matches_by_order,
        possible_matches_by_order,
        translation_length,
        reference_length,
        max_order=4,
        smooth=False,
    ):
        """Corpus BLEU implementation using Python ops.

        Computes BLEU score of translated segments against one or more
        references. Inspired from
        https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py.

        Args:
            reference_corpus: list of lists of references for each
                translation. Each reference should be tokenized into a list
                of tokens.
            translation_corpus: list of translations to score. Each
                translation should be tokenized into a list of tokens.
            matches_by_order: list of floats containing the initial number
                of matches for each order.
            possible_matches_by_order: list of floats containing the initial
                number of possible matches for each order.
            translation_length: float. Initial number of tokens in all the
                translations.
            reference_length: float. Initial number of tokens in all the
                references.
            max_order: int. Maximum n-gram order to use when computing
                BLEU score.
            smooth: boolean. Whether or not to apply Lin et al. 2004
                smoothing.
        """
        for (references, translation) in zip(
            reference_corpus, translation_corpus
        ):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(
                    reference, max_order
                )
            translation_ngram_counts = self._get_ngrams(translation, max_order)
            overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram) - 1] += overlap[ngram]
            for order in range(1, max_order + 1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order - 1] += possible_matches

        precisions = [0] * max_order
        for i in range(0, max_order):
            if smooth:
                precisions[i] = (matches_by_order[i] + 1.0) / (
                    possible_matches_by_order[i] + 1.0
                )
            else:
                if possible_matches_by_order[i] > 0:
                    precisions[i] = (
                        float(matches_by_order[i])
                        / possible_matches_by_order[i]
                    )
                else:
                    precisions[i] = 0.0

        if min(precisions) > 0:
            p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
            geo_mean = math.exp(p_log_sum)
        else:
            geo_mean = 0

        ratio = float(translation_length) / reference_length

        if ratio > 1.0:
            bp = 1.0
        else:
            bp = math.exp(1 - 1.0 / ratio)

        bleu = geo_mean * bp

        return (
            bleu,
            matches_by_order,
            possible_matches_by_order,
            translation_length,
            reference_length,
        )

    def _calculate_bleu_score(self, references, translation):
        references = tensor_to_list(references)
        translation = tensor_to_list(translation)

        matches = self._matches.numpy()
        possible_matches = self._possible_matches.numpy()
        translation_length = self._translation_length.numpy()
        reference_length = self._reference_length.numpy()

        (
            bleu_score,
            matches,
            possible_matches,
            translation_length,
            reference_length,
        ) = self._corpus_bleu(
            reference_corpus=references,
            translation_corpus=translation,
            matches_by_order=matches,
            possible_matches_by_order=possible_matches,
            translation_length=translation_length,
            reference_length=reference_length,
            max_order=self.max_order,
            smooth=self.smooth,
        )
        return (
            tf.constant(bleu_score, dtype=self.dtype),
            tf.constant(matches, dtype=self.dtype),
            tf.constant(possible_matches, dtype=self.dtype),
            tf.constant(translation_length, dtype=self.dtype),
            tf.constant(reference_length, dtype=self.dtype),
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        def validate_and_fix_rank(inputs, tensor_name, base_rank=0):
            if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
                inputs = tf.convert_to_tensor(inputs)

            if inputs.shape.rank == base_rank:
                return inputs[tf.newaxis]
            elif inputs.shape.rank == base_rank + 1:
                return inputs
            elif inputs.shape.rank == base_rank + 2:
                if tf.shape(inputs)[-1] != 1:
                    raise ValueError(
                        f"{tensor_name} is of rank {input.shape.rank}. The "
                        f"last dimension must be of size 1."
                    )
                return tf.squeeze(inputs, axis=-1)
            else:
                raise ValueError(
                    f"{tensor_name} must be of rank {base_rank}, {base_rank+1} "
                    f"or {base_rank+2}. Found rank: {inputs.shape.rank}"
                )

        y_true = validate_and_fix_rank(y_true, "y_true", 1)
        y_pred = validate_and_fix_rank(y_pred, "y_pred", 0)

        # Tokenize the inputs.
        y_true = self._tokenizer(y_true)
        y_pred = self._tokenizer(y_pred)

        (
            bleu_score,
            matches,
            possible_matches,
            translation_length,
            reference_length,
        ) = tf.py_function(
            func=self._calculate_bleu_score,
            inp=[y_true, y_pred],
            Tout=[self.dtype, self.dtype, self.dtype, self.dtype, self.dtype],
        )

        self._matches.assign(matches)
        self._possible_matches.assign(possible_matches)
        self._translation_length.assign(translation_length)
        self._reference_length.assign(reference_length)
        self._bleu.assign(bleu_score)

    def result(self):
        return self._bleu

    def reset_state(self):
        self._matches.assign(
            tf.zeros(shape=(self.max_order,), dtype=self.dtype)
        )
        self._possible_matches.assign(
            tf.zeros(shape=(self.max_order,), dtype=self.dtype)
        )
        self._translation_length.assign(0.0)
        self._reference_length.assign(0.0)
        self._bleu.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": self.tokenizer,
                "max_order": self.max_order,
                "smooth": self.smooth,
            }
        )
        return config
