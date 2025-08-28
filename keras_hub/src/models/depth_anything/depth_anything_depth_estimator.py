import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.depth_anything.depth_anything_backbone import (
    DepthAnythingBackbone,
)
from keras_hub.src.models.depth_anything.depth_anything_depth_estimator_preprocessor import (  # noqa: E501
    DepthAnythingDepthEstimatorPreprocessor,
)
from keras_hub.src.models.depth_anything.depth_anything_loss import (
    DepthAnythingLoss,
)
from keras_hub.src.models.depth_estimator import DepthEstimator


@keras_hub_export("keras_hub.models.DepthAnythingDepthEstimator")
class DepthAnythingDepthEstimator(DepthEstimator):
    backbone_cls = DepthAnythingBackbone
    preprocessor_cls = DepthAnythingDepthEstimatorPreprocessor

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        """Configures the `DepthEstimator` task for training.

        The `DepthEstimator` task extends the default compilation signature of
        `keras.Model.compile` with defaults for `optimizer`, `loss`, and
        `metrics`. To override these defaults, pass any value
        to these arguments during compilation.

        Args:
            optimizer: `"auto"`, an optimizer name, or a `keras.Optimizer`
                instance. Defaults to `"auto"`, which uses the default optimizer
                for the given model and task. See `keras.Model.compile` and
                `keras.optimizers` for more info on possible `optimizer` values.
            loss: `"auto"`, a loss name, or a `keras.losses.Loss` instance.
                Defaults to `"auto"`, where a `DepthAnythingLoss` loss will be
                applied for the depth estimation task. See
                `keras.Model.compile` and `keras.losses` for more info on
                possible `loss` values.
            metrics: `"auto"`, or a dict of metrics to be evaluated by
                the model during training and testing. Defaults to `"auto"`,
                where a `keras.metrics.RootMeanSquaredError` will be applied to
                track the accuracy of the model during training. See
                `keras.Model.compile` and `keras.metrics` for more info on
                possible `metrics` values.
            **kwargs: See `keras.Model.compile` for a full list of arguments
                supported by the compile method.
        """
        if optimizer == "auto":
            optimizer = keras.optimizers.AdamW(5e-5)
        if loss == "auto":
            loss = {
                "depths": DepthAnythingLoss(
                    min_depth=self.min_depth, max_depth=self.max_depth
                )
            }
        if metrics == "auto":
            metrics = {"depths": keras.metrics.RootMeanSquaredError()}
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )
