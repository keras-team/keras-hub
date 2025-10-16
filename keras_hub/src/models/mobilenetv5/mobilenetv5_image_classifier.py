import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.mobilenetv5.mobilenetv5_backbone import (
    MobileNetV5Backbone,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_image_classifier_preprocessor import (  # noqa: E501
    MobileNetV5ImageClassifierPreprocessor,
)
from keras_hub.src.models.mobilenetv5.mobilenetv5_layers import ConvNormAct
from keras_hub.src.models.mobilenetv5.mobilenetv5_utils import (
    SelectAdaptivePool2d,
)
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.MobileNetV5ImageClassifier")
class MobileNetV5ImageClassifier(ImageClassifier):
    """An end-to-end MobileNetV5 model for image classification.

    This model attaches a classification head to a `MobileNetV5Backbone`.
    The head consists of a global pooling layer, an optional convolutional
    head, a dropout layer, and a final dense classifier layer.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to image inputs during
    `fit()`, `predict()`, and `evaluate()`.

    Args:
        backbone: A `keras_hub.models.MobileNetV5Backbone` instance.
        num_classes: int. The number of classes for the classification head.
        preprocessor: A `keras_hub.models.ImageClassifierPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing.
        head_hidden_size: int. The number of channels in the convolutional
            head.
        global_pool: str. The type of global pooling to use.
        drop_rate: float. The dropout rate for the head.
        head_dtype: string or `keras.mixed_precision.DTypePolicy`. The dtype to
            use for the head computations and weights.

    Example:
    ```python
    import keras
    from keras_hub.models import MobileNetV5Backbone
    from keras_hub.models import MobileNetV5ImageClassifier

    # Randomly initialized task model with a custom config.
    model_args = {
        "stackwise_block_types": [["er"], ["uir", "uir"]],
        "stackwise_num_blocks": [1, 2],
        "stackwise_num_filters": [[24], [48, 48]],
        "stackwise_strides": [[2], [2, 1]],
        "stackwise_act_layers": [["relu"], ["relu", "relu"]],
        "stackwise_exp_ratios": [[4.0], [6.0, 6.0]],
        "stackwise_se_ratios": [[0.0], [0.0, 0.0]],
        "stackwise_dw_kernel_sizes": [[0], [5, 5]],
        "stackwise_dw_start_kernel_sizes": [[0], [0, 0]],
        "stackwise_dw_end_kernel_sizes": [[0], [0, 0]],
        "stackwise_exp_kernel_sizes": [[3], [0, 0]],
        "stackwise_pw_kernel_sizes": [[1], [0, 0]],
        "stackwise_num_heads": [[0], [0, 0]],
        "stackwise_key_dims": [[0], [0, 0]],
        "stackwise_value_dims": [[0], [0, 0]],
        "stackwise_kv_strides": [[0], [0, 0]],
        "stackwise_use_cpe": [[False], [False, False]],
        "use_msfa": False,
    }
    backbone = MobileNetV5Backbone(**model_args)
    model = MobileNetV5ImageClassifier(backbone, 1000)
    images = keras.ops.ones((1, 224, 224, 3))
    output = model.predict(images)
    ```
    """

    backbone_cls = MobileNetV5Backbone
    preprocessor_cls = MobileNetV5ImageClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        head_hidden_size=2048,
        global_pool="avg",
        drop_rate=0.0,
        head_dtype=None,
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy
        data_format = getattr(backbone, "data_format", "channels_last")

        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        if backbone.use_msfa:
            self.global_pool = SelectAdaptivePool2d(
                pool_type=global_pool, data_format=data_format, flatten=True
            )
            self.conv_head = None
            self.flatten = None
        else:
            self.global_pool = SelectAdaptivePool2d(
                pool_type=global_pool, data_format=data_format, flatten=False
            )
            self.conv_head = ConvNormAct(
                out_chs=head_hidden_size,
                kernel_size=1,
                pad_type="same",
                norm_layer=backbone.norm_layer,
                act_layer=backbone.act_layer,
                bias=False,
                name="conv_head",
                dtype=head_dtype,
            )
            self.flatten = keras.layers.Flatten(dtype=head_dtype)
        self.dropout = (
            keras.layers.Dropout(drop_rate, dtype=head_dtype)
            if drop_rate > 0.0
            else None
        )
        self.classifier = (
            keras.layers.Dense(num_classes, dtype=head_dtype, name="classifier")
            if num_classes > 0
            else keras.layers.Activation("linear", name="identity_classifier")
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        x = self.global_pool(x)
        if self.conv_head is not None:
            x = self.conv_head(x)
        if self.flatten is not None:
            x = self.flatten(x)
        if self.dropout is not None:
            x = self.dropout(x)
        outputs = self.classifier(x)
        Task.__init__(self, inputs=inputs, outputs=outputs, **kwargs)

        # === Config ===
        self.num_classes = num_classes
        self.head_hidden_size = head_hidden_size
        self.global_pool_type = global_pool
        self.drop_rate = drop_rate

    def get_config(self):
        config = Task.get_config(self)
        config.update(
            {
                "num_classes": self.num_classes,
                "head_hidden_size": self.head_hidden_size,
                "global_pool": self.global_pool_type,
                "drop_rate": self.drop_rate,
            }
        )
        return config
