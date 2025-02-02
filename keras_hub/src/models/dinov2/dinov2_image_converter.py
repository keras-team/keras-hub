from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.dinov2.dinov2_backbone import DinoV2Backbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.DinoV2ImageConverter")
class DinoV2ImageConverter(ImageConverter):
    """Converts images to the format expected by a DinoV2 model.

    This layer performs image normalization using mean and standard deviation
    values.

    Args:
        norm_mean: list or tuple of floats. Mean values for image normalization.
            Defaults to `[0.5, 0.5, 0.5]`.
        norm_std: list or tuple of floats. Standard deviation values for
            image normalization. Defaults to `[0.5, 0.5, 0.5]`.
        **kwargs: Additional keyword arguments passed to
            `keras_hub.layers.preprocessing.ImageConverter`.

    Examples:
    ```python
    import keras
    import numpy as np
    from keras_hub.src.layers import DinoV2ImageConverter

    # Example image (replace with your actual image data)
    image = np.random.rand(1, 224, 224, 3)  # Example: (B, H, W, C)

    # Create a DinoV2ImageConverter instance
    converter = DinoV2ImageConverter(
        image_size=(28,28),
        scale=1/255.
    )
    # Preprocess the image
    preprocessed_image = converter(image)
    ```
    """

    backbone_cls = DinoV2Backbone

    def __init__(
        self, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5], **kwargs
    ):
        super().__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    @preprocessing_function
    def call(self, inputs):
        x = super().call(inputs)
        # By default normalize using imagenet mean and std
        if self.norm_mean:
            x = x - self._expand_non_channel_dims(self.norm_mean, x)
        if self.norm_std:
            x = x / self._expand_non_channel_dims(self.norm_std, x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
            }
        )
        return config
