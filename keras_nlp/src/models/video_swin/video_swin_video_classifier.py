# Copyright 2024 The KerasNLP Authors
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
import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.image_classifier import ImageClassifier
from keras_nlp.src.models.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)


@keras_nlp_export("keras_nlp.models.VideoSwinVideoClassifier")
class VideoSwinVideoClassifier(ImageClassifier):
    """VideoSwin Video classifier task model.

    Args:
        backbone: A `keras_nlp.models.VideoSwinBackbone` instance.
        num_classes: int. The number of classes to predict.
        activation: `None`, str or callable. The activation function to use on
            the `Dense` layer. Set `activation=None` to return the output
            logits. Defaults to `"softmax"`.
        head_dtype: `None` or str or `keras.mixed_precision.DTypePolicy`. The
            dtype to use for the classification head's computations and weights.

    To fine-tune with `fit()`, pass a dataset containing tuples of `(x, y)`
    where `x` is a tensor and `y` is a integer from `[0, num_classes)`.
    All `ImageClassifier` tasks include a `from_preset()` constructor which can
    be used to load a pre-trained config and weights.

    Custom backbone.
    ```python
    images = np.ones((2, 224, 224, 3), dtype="float32")
    labels = [0, 3]
    backbone = keras_nlp.models.VideoSwinBackbone(
        image_shape=(32, 224, 224, 3)
        include_rescaling=False,
    )
    classifier = keras_nlp.models.VideoSwinVideoClassifier(
        backbone=backbone,
        num_classes=4,
    )
    classifier.fit(x=images, y=labels, batch_size=2)
    ```
    """

    backbone_cls = VideoSwinBackbone

    def __init__(
        self,
        backbone,
        num_classes,
        activation="softmax",
        head_dtype=None,
        preprocessor=None,  # adding this dummy arg for saved model test
        # TODO: once preprocessor flow is figured out, this needs to be updated
        **kwargs,
    ):
        head_dtype = head_dtype or backbone.dtype_policy

        # === Layers ===
        self.backbone = backbone
        self.output_dense = keras.layers.Dense(
            num_classes,
            activation=activation,
            dtype=head_dtype,
            name="predictions",
        )

        # === Functional Model ===
        inputs = self.backbone.input
        x = self.backbone(inputs)
        outputs = self.output_dense(x)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.activation = activation

    def get_config(self):
        # Backbone serialized in `super`
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": self.activation,
            }
        )
        return config
