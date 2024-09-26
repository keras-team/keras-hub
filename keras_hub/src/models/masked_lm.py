import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.MaskedLM")
class MaskedLM(Task):
    """Base class for masked language modeling tasks.

    `MaskedLM` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    unsupervised fine-tuning with a masked language modeling loss.

    When calling `fit()`, all input will be tokenized, and random tokens in
    the input sequence will be masked. These positions of these masked tokens
    will be fed as an additional model input, and the original value of the
    tokens predicted by the model outputs.

    All `MaskedLM` tasks include a `from_preset()` constructor which can be used
    to load a pre-trained config and weights.

    Example:
    ```python
    # Load a Bert MaskedLM with pre-trained weights.
    masked_lm = keras_hub.models.MaskedLM.from_preset(
        "bert_base_en",
    )
    masked_lm.fit(train_ds)
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        **kwargs,
    ):
        """Configures the `MaskedLM` task for training.

        The `MaskedLM` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `weighted_metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Note that because training inputs include padded tokens which are
        excluded from the loss, it is almost always a good idea to compile with
        `weighted_metrics` and not `metrics`.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.SparseCategoricalCrossentropy` loss will be
                applied for the token classification `MaskedLM` task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            weighted_metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.SparseCategoricalAccuracy` will be
                applied to track the accuracy of the model at guessing masked
                token values. See `keras.Model.compile` and `keras.metrics` for
                more info on possible `weighted_metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        if loss == "auto":
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        if weighted_metrics == "auto":
            weighted_metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            **kwargs,
        )
