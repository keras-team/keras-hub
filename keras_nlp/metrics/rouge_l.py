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

"""ROUGE-L metric implementation based on `keras.metrics.Metric`."""


from keras_nlp.metrics.rouge import RougeBase


class RougeL(RougeBase):
    """ROUGE-L metric.

    This class implements the ROUGE-L variant of the ROUGE metric. The ROUGE-L
    metric is traditionally used for evaluating summarisation systems.
    Succinctly put, ROUGE-L is a score based on the length of the longest
    common subsequence present in the reference text and the hypothesis text.

    Note on input shapes:
    `y_true` and `y_pred` can be of the following types/shapes:
    1. Python string/scalar input
    2. Tensor/Python list
        a. rank 0
        b. rank 1 (every element in the tensor is a string)
        c. rank 2 (shape: `(batch_size, 1)`)

    Args:
        use_stemmer: bool. Whether Porter Stemmer should be used to strip word
            suffixes to improve matching. Defaults to False.
        metric_type: string. One of "precision", "recall", "f1_score". Defaults
            to "f1_score".
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    Examples:

    1. Various Input Types.
    1.1. Python string.
    >>> rouge_l = keras_nlp.metrics.RougeL()
    >>> y_true = "the tiny little cat was found under the big funny bed"
    >>> y_pred = "the cat was under the bed"
    >>> rouge_l(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.7058824>

    1.2. rank 1 inputs.
    a. Python list.
    >>> rouge_l = keras_nlp.metrics.RougeL()
    >>> y_true = [
    ...     "the tiny little cat was found under the big funny bed",
    ...     "i really love contributing to KerasNLP",
    ... ]
    >>> y_pred = [
    ...     "the cat was under the bed",
    ...     "i love contributing to KerasNLP",
    ... ]
    >>>
    >>> rouge_l(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.80748665>

    b. Tensor
    >>> rouge_l = keras_nlp.metrics.RougeL()
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
    >>> rouge_l(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.80748665>

    1.3. rank 2 inputs.
    >>> rouge_l = keras_nlp.metrics.RougeL()
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
    >>> rouge_l(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.80748665>

    3. Output the precision instead of the F1 Score.
    >>> rouge_l = keras_nlp.metrics.RougeL(metric_type="precision")
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
    >>> rouge_l(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=1.0>

    4. Pass the metric to `model.compile()`.
    >>> inputs = keras.Input(shape=(), dtype='string')
    >>> outputs = tf.strings.lower(inputs)
    >>> model = keras.Model(inputs, outputs)
    >>> model.compile(metrics=[keras_nlp.metrics.RougeL()])
    >>> x = tf.constant(["HELLO THIS IS FUN"])
    >>> y = tf.constant(["hello this is awesome"])
    >>> model.evaluate(x, y, return_dict=True)
    {'loss': 0.0, 'rouge-l': 0.75}
    """

    def __init__(
        self,
        metric_type="f1_score",
        use_stemmer=False,
        dtype=None,
        name="rouge-l",
        **kwargs,
    ):
        super().__init__(
            variant="rougeL",
            metric_type=metric_type,
            use_stemmer=use_stemmer,
            dtype=dtype,
            name=name,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        del config["variant"]
        return config
