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

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.metrics.rouge_base import RougeBase


@keras_nlp_export("keras_nlp.metrics.RougeN")
class RougeN(RougeBase):
    """ROUGE-N metric.

    This class implements the ROUGE-N variant of the ROUGE metric. The ROUGE-N
    metric is traditionally used for evaluating summarisation systems.
    Succinctly put, ROUGE-N is a score based on the number of matching n-grams
    between the reference text and the hypothesis text.

    Note on input shapes:
    For `y_true` and `y_pred`, this class supports scalar values and batch
    inputs of shapes `()`, `(batch_size,)` and `(batch_size, 1)`.

    Args:
        order: The order of n-grams which are to be matched. It should lie in
            range [1, 9]. Defaults to `2`.
        use_stemmer: bool. Whether Porter Stemmer should be used to strip word
            suffixes to improve matching. Defaults to `False`.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
               not specified, it defaults to `"float32"`.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    References:
        - [Lin et al., 2004](https://aclanthology.org/W04-1013/)

    Examples:

    1. Python string.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=2)
    >>> y_true = "the tiny little cat was found under the big funny bed"
    >>> y_pred = "the cat was under the bed"
    >>> rouge_n(y_true, y_pred)["f1_score"]
    <tf.Tensor: shape=(), dtype=float32, numpy=0.26666668>

    2. List inputs.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=2)
    >>> y_true = [
    ...     "the tiny little cat was found under the big funny bed",
    ...     "i really love contributing to KerasNLP",
    ... ]
    >>> y_pred = [
    ...     "the cat was under the bed",
    ...     "i love contributing to KerasNLP",
    ... ]
    >>> rouge_n(y_true, y_pred)["f1_score"]
    <tf.Tensor: shape=(), dtype=float32, numpy=0.4666667>

    3. 2D inputs.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=2)
    >>> y_true =[
    ...     ["the tiny little cat was found under the big funny bed"],
    ...     ["i really love contributing to KerasNLP"],
    ... ]
    >>> y_pred =[
    ...     ["the cat was under the bed"],
    ...     ["i love contributing to KerasNLP"],
    ... ]
    >>> rouge_n(y_true, y_pred)["f1_score"]
    <tf.Tensor: shape=(), dtype=float32, numpy=0.4666667>

    4. Trigrams.
    >>> rouge_n = keras_nlp.metrics.RougeN(order=3)
    >>> y_true = [
    ...     "the tiny little cat was found under the big funny bed",
    ...     "i really love contributing to KerasNLP",
    ... ]
    >>> y_pred = [
    ...     "the cat was under the bed",
    ...     "i love contributing to KerasNLP",
    ... ]
    >>> rouge_n(y_true, y_pred)["f1_score"]
    <tf.Tensor: shape=(), dtype=float32, numpy=0.2857143>
    """

    def __init__(
        self,
        order=2,
        use_stemmer=False,
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
            use_stemmer=use_stemmer,
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
