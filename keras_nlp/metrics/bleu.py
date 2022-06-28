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

import collections
import math

import tensorflow as tf
from tensorflow import keras

from keras_nlp.utils.tensor_utils import tensor_to_string_list

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


class Bleu(keras.metrics.Metric):
    """BLEU metric.

    This class implements the BLEU metric. BLEU is generally used to evaluate
    machine translation systems. Succinctly put, in BLEU score, we count the
    number of matching n-grams in the candidate translation to n-grams in the
    reference text. We find the "clipped count" of matching n-grams so as to not
    give a high score to a reference, prediction pair with repeated tokens.
    Secondly, BLEU score tends to reward shorter predictions more, which is why
    a brevity penalty is applied to penalise short predictions.

    Note on input shapes:
    For `y_true` and `y_pred`, this class supports scalar values and batch
    inputs of shapes `()`, `(batch_size,)` and `(batch_size, 1)`.

    Args:
        tokenizer: callable. A function that takes a string `tf.Tensor` (of
            any shape), and tokenizes the strings in the tensor. This function
            should use TensorFlow graph ops. If the tokenizer is not specified,
            the default tokenizer (`"tokenizer_13a"` present in the SacreBLEU
            package) will be used.
        max_order: int. The maximum n-gram order to use. For example, if
            `max_order` is set to 3, unigrams, bigrams, and trigrams will be
            considered. Defaults to 4.
        smooth: bool. Whether to apply Lin et al. 2004 smoothing to the BLEU
            score. Defaults to False.
        variant: string. Either `"corpus_bleu"` or `"sentence_bleu"`. The former
            computes the micro-average precision, which is equivalent to
            passing all samples (across batches) all at once. In other words,
            summing the numerators and denominators for each
            hypothesis-reference(s) pairs before the division (in order to
            calculate the precision). The latter is the macro-average BLEU score
            , which means that it computes the per sample BLEU score and
            averages it. Defaults to `"corpus_bleu"`.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.
    """

    def __init__(
        self,
        tokenizer=None,
        max_order=4,
        smooth=False,
        variant="corpus_bleu",
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

        if variant not in ("corpus_bleu", "sentence_bleu"):
            raise ValueError(
                "`variant` must be either 'corpus_bleu' or 'sentence_bleu'. "
                f"Received: variant={variant}"
            )

        def default_tokenizer(inputs):
            """
            Default tokenizer. Replicates the behaviour of SacreBLEU's
            default tokenizer, namely, `tokenizer_13a`.
            """
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

        if tokenizer is None:
            self.tokenizer = default_tokenizer
        else:
            self.tokenizer = tokenizer
        self.max_order = max_order
        self.smooth = smooth
        self.variant = variant

        if variant == "corpus_bleu":
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
        else:
            self._number_of_samples = self.add_weight(
                name="number_of_samples",
                initializer="zeros",
                dtype=self.dtype,
            )

        self._bleu = self.add_weight(
            name="bleu",
            initializer="zeros",
            dtype=self.dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        def validate_and_fix_rank(inputs, tensor_name, base_rank=0):
            if not isinstance(inputs, tf.Tensor):
                inputs = tf.convert_to_tensor(inputs)

            if inputs.shape.rank == base_rank:
                return inputs[tf.newaxis]
            elif inputs.shape.rank == base_rank + 1:
                return inputs
            else:
                raise ValueError(
                    f"{tensor_name} must be of rank {base_rank} or {base_rank+1}. "
                    f"Found rank: {inputs.shape.rank}"
                )

        def _get_ngrams(segment, max_order):
            """Extracts all n-grams upto a given maximum order from an input
            segment. Uses Python ops. Inspired from
            https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py.

            Args:
                segment: string. Text segment from which n-grams will be
                    extracted.
                max_order: int. Maximum length in tokens of the n-grams returned
                    by this methods.
            """
            ngram_counts = collections.Counter()
            for order in range(1, max_order + 1):
                for i in range(0, len(segment) - order + 1):
                    ngram = tuple(segment[i : i + order])
                    ngram_counts[ngram] += 1
            return ngram_counts

        def corpus_bleu(
            reference_corpus,
            translation_corpus,
            matches_by_order,
            possible_matches_by_order,
            translation_length,
            reference_length,
            max_order=4,
            smooth=False,
        ):
            """Computes BLEU score of translated segments against one or more
            references. Uses Python ops. Inspired from
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
                    merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
                translation_ngram_counts = _get_ngrams(translation, max_order)
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
                p_log_sum = sum(
                    (1.0 / max_order) * math.log(p) for p in precisions
                )
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

        def aggregate_sentence_bleu(
            reference_corpus,
            translation_corpus,
            max_order=4,
            smooth=False,
        ):
            """Computes the per-sample BLEU score and returns the aggregate of
            all samples. Uses Python ops.

            Args:
                reference_corpus: list of lists of references for each
                    translation. Each reference should be tokenized into a list
                    of tokens.
                translation_corpus: list of translations to score. Each
                    translation should be tokenized into a list of tokens.
                max_order: int. Maximum n-gram order to use when computing
                    BLEU score.
                smooth: boolean. Whether or not to apply Lin et al. 2004
                    smoothing.
            """
            bleu_score = 0.0
            for references, translation in zip(
                reference_corpus, translation_corpus
            ):
                bleu_score += corpus_bleu(
                    reference_corpus=[references],
                    translation_corpus=[translation],
                    matches_by_order=[0] * max_order,
                    possible_matches_by_order=[0] * max_order,
                    translation_length=0,
                    reference_length=0,
                    max_order=max_order,
                    smooth=smooth,
                )[0]
            return bleu_score

        def calculate_bleu_score(references, translation):
            references = tensor_to_string_list(references)
            translation = tensor_to_string_list(translation)

            if self.variant == "corpus_bleu":
                matches = self._matches.numpy().tolist()
                possible_matches = self._possible_matches.numpy().tolist()
                translation_length = self._translation_length.numpy()
                reference_length = self._reference_length.numpy()

                (
                    bleu_score,
                    matches,
                    possible_matches,
                    translation_length,
                    reference_length,
                ) = corpus_bleu(
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
            else:
                bleu_score = aggregate_sentence_bleu(
                    reference_corpus=references,
                    translation_corpus=translation,
                    max_order=self.max_order,
                    smooth=self.smooth,
                )
                return tf.constant(bleu_score, dtype=self.dtype)

        y_true = validate_and_fix_rank(y_true, "y_true", 1)
        y_pred = validate_and_fix_rank(y_pred, "y_pred", 0)

        if self.variant == "sentence_bleu":
            batch_size = tf.cast(tf.shape(y_true)[0], dtype=self.dtype)
            self._number_of_samples.assign_add(batch_size)

        # Tokenize the inputs.
        y_true = self.tokenizer(y_true)
        y_pred = self.tokenizer(y_pred)

        if self.variant == "corpus_bleu":
            (
                bleu_score,
                matches,
                possible_matches,
                translation_length,
                reference_length,
            ) = tf.py_function(
                func=calculate_bleu_score,
                inp=[y_true, y_pred],
                Tout=[
                    self.dtype,
                    self.dtype,
                    self.dtype,
                    self.dtype,
                    self.dtype,
                ],
            )

            self._matches.assign(matches)
            self._possible_matches.assign(possible_matches)
            self._translation_length.assign(translation_length)
            self._reference_length.assign(reference_length)
            self._bleu.assign(bleu_score)
        else:
            bleu_score = tf.py_function(
                func=calculate_bleu_score,
                inp=[y_true, y_pred],
                Tout=self.dtype,
            )
            self._bleu.assign_add(bleu_score)

    def result(self):
        if self.variant == "corpus_bleu":
            return self._bleu
        else:
            if self._number_of_samples == 0:
                return 0.0
            else:
                return self._bleu / self._number_of_samples

    def reset_state(self):
        self._matches.assign(0.0)
        self._possible_matches.assign(0.0)
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
                "variant": self.variant,
            }
        )
        return config
