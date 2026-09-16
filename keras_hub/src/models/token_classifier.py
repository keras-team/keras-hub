import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.TokenClassifier")
class TokenClassifier(Task):
    """Base class for token-level classification tasks (e.g. NER, POS tagging).

    `TokenClassifier` tasks wrap a `keras_hub.models.Backbone` and a
    `keras_hub.models.Preprocessor` to create a model that produces per-token
    logits for sequence labeling. Unlike `TextClassifier` (which pools to a
    single vector), `TokenClassifier` retains the full `(batch, seq_len,
    num_classes)` output so every token position receives its own prediction.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    where `x` is a string (or preprocessed dict) and `y` is an integer tensor
    of shape `(batch, seq_len)` with values in `[0, num_classes)`. Padding
    positions are automatically excluded from the loss via the `padding_mask`.

    All `TokenClassifier` tasks include a `from_preset()` constructor for
    loading pre-trained configs and weights.

    Example:
    ```python
    # Load a pre-trained token classifier.
    classifier = keras_hub.models.TokenClassifier.from_preset(
        "openai_privacy_filter_en",
    )
    # Predict PII labels for a batch of sentences.
    classifier.predict(["My name is John and I live in Paris."])
    ```
    """

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        """Configures the `TokenClassifier` task for training.

        The `TokenClassifier` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value to these
        arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses
                `keras.optimizers.Adam(5e-5)`.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.SparseCategoricalCrossentropy` loss is applied
                per token. Padding tokens are masked via `sample_weight`.
            metrics: `"auto"`, or a list of metrics. Defaults to `"auto"`,
                where `keras.metrics.SparseCategoricalAccuracy` is used.
                For production NER evaluation, replace with seqeval F1.
            **kwargs: See `keras.Model.compile` for additional arguments.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        if loss == "auto":
            activation = getattr(self, "activation", None)
            activation = keras.activations.get(activation)
            from_logits = activation != keras.activations.softmax
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits)
        if metrics == "auto":
            metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )
