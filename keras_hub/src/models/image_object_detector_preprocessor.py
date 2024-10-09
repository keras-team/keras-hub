import keras

from keras_hub.src.api_export import keras_hub_export
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
        image_converter=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        if self.image_converter:
            x = self.image_converter(x)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
