import keras

from keras_hub.src.metrics.metric_utils import validate_and_fix_rank
from keras_hub.src.utils.tensor_utils import is_float_dtype

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
        # [batch_size] and [batch_size, 1]. All are canonicalized to a Python
        # list of `batch_size` strings.
        y_true = validate_and_fix_rank(y_true, "y_true")
        y_pred = validate_and_fix_rank(y_pred, "y_pred")

        if len(y_true) != len(y_pred):
            raise ValueError(
                "`y_true` and `y_pred` must have the same batch size. "
                f"Received: len(y_true)={len(y_true)}, "
                f"len(y_pred)={len(y_pred)}"
            )

        for reference, hypothesis in zip(y_true, y_pred):
            score = self._rouge_scorer.score(reference, hypothesis)[
                self.variant
            ]
            self._rouge_precision.assign_add(score.precision)
            self._rouge_recall.assign_add(score.recall)
            self._rouge_f1_score.assign_add(score.fmeasure)

        self._number_of_samples.assign_add(len(y_true))

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
