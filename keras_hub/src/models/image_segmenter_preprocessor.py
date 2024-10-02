import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.ImageSegmenterPreprocessor")
class ImageSegmenterPreprocessor(Preprocessor):
    """Base class for image segmentation preprocessing layers.

    `ImageSegmenterPreprocessor` wraps a
    `keras_hub.layers.ImageConverter` to create a preprocessing layer for
    image segmentation tasks. It is intended to be paired with a
    `keras_hub.models.ImageSegmenter` task.

    All `ImageSegmenterPreprocessor` instances take three inputs: `x`, `y`, and
    `sample_weight`.

    - `x`: The first input, should always be included. It can be an image or
      a batch of images.
    - `y`: (Optional) Usually the segmentation mask(s), will be passed through
      unaltered.
    - `sample_weight`: (Optional) Will be passed through unaltered.

    The layer will output either `x`, an `(x, y)` tuple if labels were provided,
    or an `(x, y, sample_weight)` tuple if labels and sample weight were
    provided. `x` will be the input images after all model preprocessing has
    been applied.

    All `ImageSegmenterPreprocessor` tasks include a `from_preset()`
    constructor which can be used to load a pre-trained config and vocabularies.
    You can call the `from_preset()` constructor directly on this base class, in
    which case the correct class for your model will be automatically
    instantiated.

    Examples.
    ```python
    preprocessor = keras_hub.models.ImageSegmenterPreprocessor.from_preset(
        "deeplabv3_resnet50",
    )

    # Resize a single image for the model.
    x = np.ones((512, 512, 3))
    x = preprocessor(x)

    # Resize an image and its mask.
    x, y = np.ones((512, 512, 3)), np.zeros((512, 512, 1))
    x, y = preprocessor(x, y)

    # Resize a batch of images and masks.
    x, y = [np.ones((512, 512, 3)), np.zeros((512, 512, 3))], [np.ones((512, 512, 1)), np.zeros((512, 512, 1))]
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
