import keras
from keras.losses import BinaryCrossentropy

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.losses.ciou_loss import CIoULoss
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.ObjectDetector")
class ObjectDetector(Task):
    """Base class for all object detection tasks."""

    def compile(
        self,
        optimizer="auto",
        box_loss="auto",
        classification_loss="auto",
        box_loss_weight=7.5,
        classification_loss_weight=0.5,
        metrics=None,
        **kwargs,
    ):
        """Configures the `ObjectDetector` task for training.

        The `ObjectDetector` task extends the default compilation signature of
        `keras.Model.compile` with a default `optimizer`, default loss
        functions `box_loss`, and `classification_loss` and default loss weights
        "box_loss_weight" and "classification_loss_weight".

        `compile()` mirrors the standard Keras `compile()` method, but has one
        key distinction -- two losses must be provided: `box_loss` and
        `classification_loss`.

        Args:
            box_loss: a Keras loss to use for box offset regression. A
                preconfigured loss is provided when the string "ciou" is passed.
            classification_loss: a Keras loss to use for box classification. A
                preconfigured loss is provided when the string
                "binary_crossentropy" is passed.
            box_loss_weight: (optional) float, a scaling factor for the box
                loss. Defaults to 7.5.
            classification_loss_weight: (optional) float, a scaling factor for
                the classification loss. Defaults to 0.5.
            kwargs: most other `keras.Model.compile()` arguments are supported
                and propagated to the `keras.Model` class.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(0.001)
        if box_loss == "auto":
            box_loss = CIoULoss(bounding_box_format="xyxy", reduction="sum")
        if classification_loss == "auto":
            classification_loss = BinaryCrossentropy(reduction="sum")
        if metrics is not None:
            raise ValueError("User metrics not yet supported")
        self.box_loss = box_loss
        self.classification_loss = classification_loss
        self.box_loss_weight = box_loss_weight
        self.classification_loss_weight = classification_loss_weight
        losses = {
            "box": self.box_loss,
            "class": self.classification_loss,
        }
        super().compile(loss=losses, **kwargs)
