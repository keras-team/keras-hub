"""Image/video preprocessing converter for Qwen3-Omni."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)


@keras_hub_export("keras_hub.layers.Qwen3OmniImageConverter")
class Qwen3OmniImageConverter(ImageConverter):
    """Image/video preprocessing for Qwen3-Omni vision encoder.

    Converts images and videos to the format expected by Qwen3OmniBackbone.
    Handles resizing, normalization, and temporal patch extraction for videos.

    This layer follows the KerasHub ImageConverter pattern and automatically
    loads preprocessing config from the backbone preset.

    Args:
        **kwargs: Additional ImageConverter arguments.

    Examples:
    ```python
    # Create converter from backbone preset
    converter = keras_hub.models.Qwen3OmniImageConverter.from_preset(
        "qwen3_omni_instruct"
    )

    # Convert single image
    image = np.random.randint(0, 255, (224, 224, 3), dtype="uint8")
    processed = converter(image)

    # Convert batch of images
    images = [image1, image2, image3]
    processed_batch = converter(images)
    ```
    """

    backbone_cls = Qwen3OmniBackbone
