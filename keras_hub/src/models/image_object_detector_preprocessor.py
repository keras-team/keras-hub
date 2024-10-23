import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.ImageObjectDetectorPreprocessor")
class ImageObjectDetectorPreprocessor(Preprocessor):
    """Base class for image segmentation preprocessing layers.

    `ImageObjectDetectorPreprocessor` wraps a
    `keras_hub.layers.ImageConverter` to create a preprocessing layer for
    image object detection tasks. It is intended to be paired with a
    `keras_hub.models.ImageObjectDetector` task.

    All `ImageObjectDetectorPreprocessor` instances take three inputs: `x`, `y`, and
    `sample_weight`.

    - `x`: The first input, should always be included. It can be an image or
      a batch of images.
    - `y`: (Optional) Usually the bounding boxes, if `resize_bboxes`
        is set to `True` this will be resized to input image shape else will be
        passed through unaltered.
    - `sample_weight`: (Optional) Will be passed through unaltered.

    The layer will output either `x`, an `(x, y)` tuple if labels were provided,
    or an `(x, y, sample_weight)` tuple if labels and sample weight were
    provided. `x` will be the input images after all model preprocessing has
    been applied.

    All `ImageObjectDetectorPreprocessor` tasks include a `from_preset()`
    constructor which can be used to load a pre-trained config.
    You can call the `from_preset()` constructor directly on this base class, in
    which case the correct class for your model will be automatically
    instantiated.

    Examples.
    ```python
    preprocessor = keras_hub.models.ImageObjectDetectorPreprocessor.from_preset(
        "yolov11_n",
    )

    # Resize a single image for the model.
    x = np.ones((512, 512, 3))
    x = preprocessor(x)
    ```

    """

    def __init__(
        self,
        image_converter=None,
        resize_bboxes=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter
        self.resize_bboxes = resize_bboxes

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        original_shape = x.shape
        if self.image_converter:
            x = self.image_converter(x)

        if y is not None and self.image_converter and self.resize_bboxes:
            # resize the bboxes
            # 1. Use the original shape for this
            # 2. Compute the would-be locations in the new size
            # 3. Scale to the new sizes
            print(original_shape)

        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
