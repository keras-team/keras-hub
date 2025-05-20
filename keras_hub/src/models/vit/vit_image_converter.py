from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.vit.vit_backbone import ViTBackbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.ViTImageConverter")
class ViTImageConverter(ImageConverter):
    """Converts images to the format expected by a ViT model.

    This layer performs image normalization using mean and standard deviation
    values. By default, it uses the same normalization as the
    "google/vit-large-patch16-224" model on Hugging Face:
    `norm_mean=[0.5, 0.5, 0.5]` and `norm_std=[0.5, 0.5, 0.5]`
    ([reference](https://huggingface.co/google/vit-large-patch16-224/blob/main/preprocessor_config.json)).
    These defaults are suitable for models pretrained using this normalization.

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
    from keras_hub.src.layers import ViTImageConverter

    # Example image (replace with your actual image data)
    image = np.random.rand(1, 224, 224, 3)  # Example: (B, H, W, C)

    # Create a ViTImageConverter instance
    converter = ViTImageConverter(
        image_size=(28,28),
        scale=1/255.
    )
    # Preprocess the image
    preprocessed_image = converter(image)
    ```
    """

    backbone_cls = ViTBackbone

    def __init__(
        self, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5], **kwargs
    ):
        super().__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    @preprocessing_function
    def call(self, inputs):
        # TODO: Remove this whole function. Why can just use scale and offset
        # in the base class.
        x = super().call(inputs)
        if self.norm_mean:
            norm_mean = self._expand_non_channel_dims(self.norm_mean, x)
            x, norm_mean = self._convert_types(x, norm_mean, self.compute_dtype)
            x = x - norm_mean
        if self.norm_std:
            norm_std = self._expand_non_channel_dims(self.norm_std, x)
            x, norm_std = self._convert_types(x, norm_std, x.dtype)
            x = x / norm_std

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
