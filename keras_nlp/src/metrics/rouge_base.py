# Copyright 2023 The KerasNLP Authors
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

import keras
from keras import ops

from keras_nlp.src.utils.tensor_utils import is_float_dtype
from keras_nlp.src.utils.tensor_utils import tensor_to_list

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


class RougeBase(keras.metrics.Metric):
    """ROUGE metric.

    This class implements two variants of the ROUGE metric - ROUGE-N,
    and ROUGE-L.

    Note on input shapes:
    For `y_true` and `y_pred`, this class supports scalar values and batch
    inputs of shapes `()`, `(batch_size,)` and `(batch_size, 1)`.

    Args:
        variant: string. One of "rougeN", "rougeL". For "rougeN", N lies in
            the range [1, 9]. Defaults to `"rouge2"`.
        use_stemmer: bool. Whether Porter Stemmer should be used to strip word
            suffixes to improve matching. Defaults to `False`.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
            not specified, it defaults to `"float32"`.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    References:
        - [Lin et al., 2004](https://aclanthology.org/W04-1013/)
    """

    def __init__(
        self,
        variant="rouge2",
        use_stemmer=False,
        dtype="float32",
        name="rouge",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if rouge_scorer is None:
            raise ImportError(
                f"{self.__class__.__name__} requires the `rouge_score` "
                "package. Please install it with `pip install rouge-score`."
            )

        if not is_float_dtype(dtype):
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        if variant not in tuple(
            ("rouge" + str(order) for order in range(1, 10))
        ) + ("rougeL",):
            raise ValueError(
                "Invalid variant of ROUGE. Should be one of: rougeN, rougeL, "
                "with N ranging from 1 to 9. Received: "
                f"variant={variant}"
            )

        self.variant = variant
        self.use_stemmer = use_stemmer

        # To-do: Add split_summaries and tokenizer options after the maintainers
        # of rouge_scorer have released a new version.
        self._rouge_scorer = rouge_scorer.RougeScorer(
            rouge_types=[self.variant],
            use_stemmer=use_stemmer,
        )

        self._rouge_precision = self.add_weight(
            shape=(),
            initializer="zeros",
            dtype=self.dtype,
            name="rouge_precision",
        )
        self._rouge_recall = self.add_weight(
            shape=(),
            initializer="zeros",
            dtype=self.dtype,
            name="rouge_recall",
        )
        self._rouge_f1_score = self.add_weight(
            shape=(),
            initializer="zeros",
            dtype=self.dtype,
            name="rouge_f1_score",
        )

        self._number_of_samples = self.add_weight(
            shape=(),
            initializer="zeros",
            dtype=self.dtype,
            name="number_of_samples",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Three possible shapes for y_true and y_pred: Python string,
        # [batch_size] and [batch_size, 1]. In the latter two cases, we have
        # strings in the tensor/list.

        def validate_and_fix_rank(inputs, tensor_name):
            if not isinstance(inputs, tf.Tensor):
                inputs = tf.convert_to_tensor(inputs)

            if inputs.shape.rank == 0:
                return inputs[tf.newaxis]
            elif inputs.shape.rank == 1:
                return inputs
            elif inputs.shape.rank == 2:
                if inputs.shape[1] != 1:
                    raise ValueError(
                        f"{tensor_name} must be of shape `[batch_size, 1]`. "
                        f"Found shape: {inputs.shape}"
                    )
                else:
                    return tf.squeeze(inputs, axis=1)
            else:
                raise ValueError(
                    f"{tensor_name} must be of rank 0 (scalar input), 1 or 2. "
                    f"Found rank: {inputs.shape.rank}"
                )

        y_true = validate_and_fix_rank(y_true, "y_true")
        y_pred = validate_and_fix_rank(y_pred, "y_pred")

        batch_size = tf.shape(y_true)[0]

        def calculate_rouge_score(reference, hypothesis):
            reference = tensor_to_list(reference)
            hypothesis = tensor_to_list(hypothesis)
            score = self._rouge_scorer.score(reference, hypothesis)[
                self.variant
            ]
            return score.precision, score.recall, score.fmeasure

        for batch_idx in range(batch_size):
            score = calculate_rouge_score(y_true[batch_idx], y_pred[batch_idx])
            self._rouge_precision.assign_add(score[0])
            self._rouge_recall.assign_add(score[1])
            self._rouge_f1_score.assign_add(score[2])

        self._number_of_samples.assign_add(
            ops.cast(batch_size, dtype=self.dtype)
        )

    def result(self):
        if self._number_of_samples == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }

        rouge_precision = self._rouge_precision / self._number_of_samples
        rouge_recall = self._rouge_recall / self._number_of_samples
        rouge_f1_score = self._rouge_f1_score / self._number_of_samples
        return {
            "precision": rouge_precision,
            "recall": rouge_recall,
            "f1_score": rouge_f1_score,
        }

    def reset_state(self):
        self._rouge_precision.assign(0.0)
        self._rouge_recall.assign(0.0)
        self._rouge_f1_score.assign(0.0)
        self._number_of_samples.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "variant": self.variant,
                "use_stemmer": self.use_stemmer,
            }
        )
        return config
