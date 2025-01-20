import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.ImageOCR")
class ImageOCR(Task):
    """Base class for all OCR tasks.

    `ImageOCR` tasks wrap a `keras_hub.models.Task` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    recognizing text in images.

    All `ImageOCR` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.
    """

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        """Configures the `ImageOCR` task for training.

        The `ImageOCR` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.SparseCategoricalCrossentropy` loss will be
                applied for the classification task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a list of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.SparseCategoricalAccuracy` will be
                applied to track the accuracy of the model during training.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(1e-4)
        if loss == "auto":
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        if metrics == "auto":
            metrics = [keras.metrics.SparseCategoricalAccuracy()]
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )
