import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.bounding_box.converters import convert_format
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.ImageObjectDetectorPreprocessor")
class ImageObjectDetectorPreprocessor(Preprocessor):
    """Base class for object detector preprocessing layers.

    `ImageObjectDetectorPreprocessor` tasks wraps a
    `keras_hub.layers.Preprocessor` to create a preprocessing layer for
    object detection tasks. It is intended to be paired with a
    `keras_hub.models.ImageObjectDetector` task.

    All `ImageObjectDetectorPreprocessor` take three inputs, `x`, `y`, and
    `sample_weight`. `x`, the first input, should always be included. It can
    be a image or batch of images. See examples below. `y` and `sample_weight`
    are optional inputs that will be passed through unaltered. Usually, `y` will
    be the a dict of `{"boxes": Tensor(batch_size, num_boxes, 4),
    "classes": (batch_size, num_boxes)}.

    The layer will returns either `x`, an `(x, y)` tuple if labels were provided,
    or an `(x, y, sample_weight)` tuple if labels and sample weight were
    provided. `x` will be the input images after all model preprocessing has
    been applied.

    All `ImageObjectDetectorPreprocessor` tasks include a `from_preset()`
    constructor which can be used to load a pre-trained config and vocabularies.
    You can call the `from_preset()` constructor directly on this base class, in
    which case the correct class for your model will be automatically
    instantiated.

    Args:
        image_converter: Preprocessing pipeline for images.
        source_bounding_box_format: str. The format of the source bounding boxes.
            supported formats include:
                - `"rel_yxyx"`
                - `"rel_xyxy"`
                - `"rel_xywh"
            Defaults to `"rel_yxyx"`.
        target_bounding_box_format: str. TODO
            https://github.com/keras-team/keras-hub/issues/1907


    Examples.
    ```python
    preprocessor = keras_hub.models.ImageObjectDetectorPreprocessor.from_preset(
        "retinanet_resnet50",
    )

    # Resize a single image for resnet 50.
    x = np.ones((512, 512, 3))
    x = preprocessor(x)

    # Resize a labeled image.
    x, y = np.ones((512, 512, 3)), 1
    x, y = preprocessor(x, y)

    # Resize a batch of labeled images.
    x, y = [np.ones((512, 512, 3)), np.zeros((512, 512, 3))], [1, 0]
    x, y = preprocessor(x, y)

    # Use a `tf.data.Dataset`.
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(2)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    def __init__(
        self,
        target_bounding_box_format,
        source_bounding_box_format="rel_yxyx",
        image_converter=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if "rel" not in source_bounding_box_format:
            raise ValueError(
                f"Only relative bounding box formats are supported "
                f"Received source_bounding_box_format="
                f"`{source_bounding_box_format}`. "
                f"Please provide a source bounding box format from one of "
                f"the following `rel_xyxy` or `rel_yxyx` or `rel_xywh`. "
                f"Ensure that the provided ground truth bounding boxes are "
                f"normalized and relative to the image size. "
            )
        self.source_bounding_box_format = source_bounding_box_format
        self.target_bounding_box_format = target_bounding_box_format
        self.image_converter = image_converter

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        if self.image_converter:
            x = self.image_converter(x)

        if y is not None and keras.ops.is_tensor(y):
            y = convert_format(
                y,
                source=self.source_bounding_box_format,
                target=self.target_bounding_box_format,
                images=x,
            )
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "source_bounding_box_format": self.source_bounding_box_format,
                "target_bounding_box_format": self.target_bounding_box_format,
            }
        )

        return config
