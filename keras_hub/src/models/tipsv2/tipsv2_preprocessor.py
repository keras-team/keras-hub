"""Preprocessor for TIPSv2 models."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.models.tipsv2.tipsv2_backbone import TIPSv2Backbone
from keras_hub.src.models.tipsv2.tipsv2_tokenizer import TIPSv2Tokenizer


@keras_hub_export("keras_hub.models.TIPSv2Preprocessor")
class TIPSv2Preprocessor(Preprocessor):
    """TIPSv2 preprocessing layer.

    This preprocessing layer provides functionality for preprocessing inputs
    for TIPSv2 models. It handles both image and text preprocessing.

    This layer will tokenize and pad/truncate text inputs, and resize/scale
    image inputs using the configured image converter.

    Args:
        tokenizer: `TIPSv2Tokenizer`. The tokenizer for text inputs.
        image_converter: `TIPSv2ImageConverter`. The image converter for
            image inputs.
        sequence_length: int. Maximum text sequence length. Defaults to
            `64`.

    Call arguments:
        x: dict. Input data containing "images" and/or text fields.
        y: optional label data.
        sample_weight: optional sample weights.
        sequence_length: optional override for max sequence length.

    Examples:
    ```python
    preprocessor = keras_hub.models.TIPSv2Preprocessor.from_preset(
        "tipsv2_b14"
    )
    preprocessor({
        "images": np.random.rand(1, 448, 448, 3).astype("float32"),
        "text": ["a photo of a cat"],
    })
    ```
    """

    backbone_cls = TIPSv2Backbone
    tokenizer_cls = TIPSv2Tokenizer
