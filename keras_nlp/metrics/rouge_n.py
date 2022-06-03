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

"""ROUGE-N metric implementation based on `keras.metrics.Metric`."""


from keras_nlp.metrics.rouge import RougeBase


class RougeN(RougeBase):
    """ROUGE-N metric.

    This class implements the ROUGE-N variant of the ROUGE metric. The ROUGE-N
    metric is traditionally used for evaluating summarisation systems.
    Succinctly put, ROUGE-N is a score based on the number of matching n-grams
    between the reference text and the hypothesis text.

    Note on input shapes:
    `y_true` and `y_pred` can be of the following types/shapes:
    1. Python string/scalar input
    2. Tensor/Python list
        a. rank 0
        b. rank 1 (every element in the tensor is a string)
        c. rank 2 (shape: `(batch_size, 1)`)

    Args:
        order: The order of n-grams which are to be matched. It should lie in
            range [1, 9]. Defaults to 2.
        metric_type: string. One of "precision", "recall", "f1_score". Defaults
            to "f1_score".
        use_stemmer: bool. Whether Porter Stemmer should be used to strip word
            suffixes to improve matching. Defaults to False.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    Examples:

    1. Various Input Types.
    1.1. Python string.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=2)
    >>> y_true = "the tiny little cat was found under the big funny bed"
    >>> y_pred = "the cat was under the bed"
    >>> rouge_n(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.26666668>

    1.2. rank 1 inputs.
    a. Python list.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=2)
    >>> y_true = [
    ...     "the tiny little cat was found under the big funny bed",
    ...     "i really love contributing to KerasNLP",
    ... ]
    >>> y_pred = [
    ...     "the cat was under the bed",
    ...     "i love contributing to KerasNLP",
    ... ]
    >>> rouge_n(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.4666667>

    b. Tensor.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=2)
    >>> y_true = tf.constant(
    ...     [
    ...         "the tiny little cat was found under the big funny bed",
    ...         "i really love contributing to KerasNLP",
    ...     ]
    ... )
    >>> y_pred = tf.constant(
    ...     [
    ...         "the cat was under the bed",
    ...         "i love contributing to KerasNLP",
    ...     ]
    ... )
    >>> rouge_n(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.4666667>

    1.3. rank 2 inputs.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=2)
    >>> y_true = tf.constant(
    ...     [
    ...         ["the tiny little cat was found under the big funny bed"],
    ...         ["i really love contributing to KerasNLP"],
    ...     ]
    ... )
    >>> y_pred = tf.constant(
    ...     [
    ...         ["the cat was under the bed"],
    ...         ["i love contributing to KerasNLP"],
    ...     ]
    ... )
    >>> rouge_n(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.4666667>

    2. Consider trigrams for calculating ROUGE-N.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=3)
    >>> y_true = tf.constant(
    ...     [
    ...         "the tiny little cat was found under the big funny bed",
    ...         "i really love contributing to KerasNLP",
    ...     ]
    ... )
    >>> y_pred = tf.constant(
    ...     [
    ...         "the cat was under the bed",
    ...         "i love contributing to KerasNLP",
    ...     ]
    ... )
    >>> rouge_n(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.2857143>

    3. Output the precision instead of the F1 Score.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=3, metric_type="precision")
    >>> y_true = tf.constant(
    ...     [
    ...         "the tiny little cat was found under the big funny bed",
    ...         "i really love contributing to KerasNLP",
    ...     ]
    ... )
    >>> y_pred = tf.constant(
    ...     [
    ...         "the cat was under the bed",
    ...         "i love contributing to KerasNLP",
    ...     ]
    ... )
    >>> rouge_n(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.33333334>

    4. Pass the metric to `model.compile()`.
    >>> inputs = keras.Input(shape=(), dtype='string')
    >>> outputs = tf.strings.lower(inputs)
    >>> model = keras.Model(inputs, outputs)
    >>> model.compile(metrics=[keras_nlp.metrics.RougeN()])
    >>> x = tf.constant(["HELLO THIS IS FUN"])
    >>> y = tf.constant(["hello this is awesome"])
    >>> metric_dict = model.evaluate(x, y, return_dict=True)
    >>> metric_dict["rouge-n"]
    0.6666666865348816
    """

    def __init__(
        self,
        order=2,
        metric_type="f1_score",
        use_stemmer=False,
        dtype=None,
        name="rouge-n",
        **kwargs,
    ):
        if order not in range(1, 10):
            raise ValueError(
                "Invalid `order` value. Should lie in the range [1, 9]."
                f"Received order={order}"
            )

        super().__init__(
            variant=f"rouge{order}",
            metric_type=metric_type,
            use_stemmer=use_stemmer,
            dtype=dtype,
            name=name,
            **kwargs,
        )

        self.order = order

    def get_config(self):
        config = super().get_config()
        del config["variant"]

        config.update(
            {
                "order": self.order,
            }
        )
        return config
