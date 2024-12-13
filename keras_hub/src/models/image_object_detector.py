import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.ImageObjectDetector")
class ImageObjectDetector(Task):
    """Base class for all image object detection tasks.

    The `ImageObjectDetector` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    object detection. `ImageObjectDetector` tasks take an additional
    `num_classes` argument, controlling the number of predicted output classes.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a string and `y` is dictionary with `boxes` and
    `classes`.

    All `ImageObjectDetector` tasks include a `from_preset()` constructor which
    can be used to load a pre-trained config and weights.
    """

    def compile(
        self,
        optimizer="auto",
        box_loss="auto",
        classification_loss="auto",
        metrics=None,
        **kwargs,
    ):
        """Configures the `ImageObjectDetector` task for training.

        The `ImageObjectDetector` task extends the default compilation signature
        of `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            box_loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a
                `keras.losses.Huber` loss will be
                applied for the object detector task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            classification_loss: `"auto"`, a loss name, or a `keras.losses.Loss`
                instance. Defaults to `"auto"`, where a
                `keras.losses.BinaryFocalCrossentropy` loss will be
                applied for the object detector task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `a list of metrics to be evaluated by
                the model during training and testing. Defaults to `None`.
                See `keras.Model.compile` and `keras.metrics` for
                more info on possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(5e-5)
        if box_loss == "auto":
            box_loss = keras.losses.Huber(reduction="sum")
        if classification_loss == "auto":
            activation = getattr(self, "activation", None)
            activation = keras.activations.get(activation)
            from_logits = activation != keras.activations.sigmoid
            classification_loss = keras.losses.BinaryFocalCrossentropy(
                from_logits=from_logits, reduction="sum"
            )
        if metrics is not None:
            raise ValueError("User metrics not yet supported")

        losses = {
            "bbox_regression": box_loss,
            "cls_logits": classification_loss,
        }

        super().compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
            **kwargs,
        )
