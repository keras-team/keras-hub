import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


class Multiplier(keras.layers.Layer):
    def __init__(self, multiplier=None, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = float(multiplier) if multiplier is not None else None

    def call(self, inputs):
        if self.multiplier is not None:
            inputs = keras.ops.multiply(inputs, self.multiplier)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "multiplier": self.multiplier,
            }
        )
        return config


@keras_hub_export("keras_hub.models.DepthEstimator")
class DepthEstimator(Task):
    """Base class for all depth estimation tasks.

    `DepthEstimator` tasks wrap a `keras_hub.models.Backbone` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    depth estimation.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    labels where `x` is a RGB image and `y` is a depth map. All `DepthEstimator`
    tasks include a `from_preset()` constructor which can be used to load a
    pre-trained config and weights.

    Args:
        backbone: A `keras_hub.models.Backbone` instance or a `keras.Model`.
        preprocessor: `None`, a `keras_hub.models.Preprocessor` instance,
            a `keras.Layer` instance, or a callable. If `None` no preprocessing
            will be applied to the inputs.
        depth_estimation_type: `"relative"` or `"metric"`. The type of depth map
            to use. `"relative"` depth maps are up-to-scale, while `"metric"`
            depth maps have metric meaning (e.g. in meters). Defaults to
            `"relative"`.
        min_depth: An float representing the minimum depth value. This value can
            be used to filter out invalid depth values during training. Defaults
            to `keras.config.epsilon()`.
        max_depth: An optional float representing the maximum depth value. This
            value can be used to filter out invalid depth values during
            training. When `depth_estimation_type="metric"`, the model's output
            will be scaled to the range `[0, max_depth]`.

    Examples:

    Call `predict()` to run inference.
    ```python
    # Load preset and train
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    depth_estimator = keras_hub.models.DepthEstimator.from_preset(
        "depth_anything_v2_small"
    )
    depth_estimator.predict(images)
    ```

    Call `fit()` on a single batch.
    ```python
    # Load preset and train
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    depths = np.random.uniform(0, 10, size=(2, 224, 224))
    depth_estimator = keras_hub.models.DepthEstimator.from_preset(
        "depth_anything_v2_small",
        depth_estimation_type="metric",
        max_depth=10.0,
    )
    depth_estimator.fit(x=images, y=depths, batch_size=2)
    ```

    Call `fit()` with custom loss, optimizer and backbone.
    ```python
    depth_estimator = keras_hub.models.DepthEstimator.from_preset(
        "depth_anything_v2_small",
        depth_estimation_type="metric",
        max_depth=10.0,
    )
    depth_estimator.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(5e-5),
    )
    depth_estimator.backbone.trainable = False
    depth_estimator.fit(x=images, y=depths, batch_size=2)
    ```

    Custom backbone.
    ```python
    images = np.random.randint(0, 256, size=(2, 224, 224, 3))
    depths = np.random.uniform(0, 10, size=(2, 224, 224))
    image_encoder = keras_hub.models.DINOV2Backbone.from_preset("dinov2_small")
    backbone = keras_hub.models.DepthAnythingBackbone(
        image_encoder=image_encoder,
        patch_size=image_encoder.patch_size,
        backbone_hidden_dim=image_encoder.hidden_dim,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_dims=[48, 96, 192, 384],
        fusion_hidden_dim=64,
        head_hidden_dim=32,
        head_in_index=-1,
    )
    depth_estimator = keras_hub.models.DepthEstimator(
        backbone=backbone,
        depth_estimation_type="metric",
        max_depth=10.0,
    )
    depth_estimator.fit(x=images, y=depths, batch_size=2)
    ```
    """

    def __init__(
        self,
        backbone,
        depth_estimation_type,
        min_depth=keras.config.epsilon(),
        max_depth=None,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        if depth_estimation_type == "relative":
            self.output_activation = keras.layers.ReLU(
                dtype=backbone.dtype_policy,
                name="output_activation",
            )
        elif depth_estimation_type == "metric":
            self.output_activation = keras.layers.Activation(
                activation="sigmoid",
                dtype=backbone.dtype_policy,
                name="output_activation",
            )
        else:
            raise ValueError(
                "`depth_estimation_type` should be either `'relative'` or "
                "`'metric'`. "
                f"Received: depth_estimation_type={depth_estimation_type}."
            )
        if max_depth is not None and depth_estimation_type != "metric":
            raise ValueError(
                "`max_depth` should only be set when "
                "`depth_estimation_type='metric'`. "
                f"Received: depth_estimation_type={depth_estimation_type}, "
                f"max_depth={max_depth}."
            )
        self.multiplier = Multiplier(
            multiplier=max_depth, dtype=backbone.dtype_policy, name="multiplier"
        )
        self.depths = keras.layers.Identity(
            dtype=backbone.dtype_policy, name="depths"
        )

        # === Config ===
        self.depth_estimation_type = depth_estimation_type
        self.min_depth = float(min_depth) if min_depth is not None else None
        self.max_depth = float(max_depth) if max_depth is not None else None

        # === Functional Model ===
        inputs = self.backbone.input
        depths = self.backbone(inputs)
        depths = self.output_activation(depths)
        depths = self.multiplier(depths)
        depths = self.depths(depths)
        outputs = {"depths": depths}
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "depth_estimation_type": self.depth_estimation_type,
                "min_depth": self.min_depth,
                "max_depth": self.max_depth,
            }
        )
        return config

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
                Defaults to `"auto"`, where a `keras.losses.MeanSquaredError`
                loss will be applied for the depth estimation task. See
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
            loss = {"depths": keras.losses.MeanSquaredError()}
        if metrics == "auto":
            metrics = {"depths": keras.metrics.RootMeanSquaredError()}
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )
