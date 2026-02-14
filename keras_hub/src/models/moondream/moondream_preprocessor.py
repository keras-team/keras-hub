import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor


@keras_hub_export("keras_hub.models.MoondreamPreprocessor")
class MoondreamPreprocessor(CausalLMPreprocessor):
    """
    Moondream Causal LM Preprocessor.

    This class handles the preprocessing of images and text for the Moondream
    model. It combines image resizing/rescaling logic with text tokenization
    to prepare inputs for the model.

    Args:
        tokenizer: The tokenizer to be used for text inputs.
        image_converter: An optional layer or callable for image preprocessing
            (e.g., resizing, normalization).
        sequence_length: int. The context length for tokenization.
            Defaults to 1024.
        add_start_token: bool. Whether to add the start token.
            Defaults to True.
        add_end_token: bool. Whether to add the end token.
            Defaults to True.
        **kwargs: Standard Keras keyword arguments.

    Example:
    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.moondream.moondream_preprocessor import (
        MoondreamPreprocessor
    )

    # 1. Create a Mock Tokenizer
    class MockTokenizer:
        def __call__(self, x):
            return keras.ops.convert_to_tensor([[1, 2, 3]] * len(x))
        def detokenize(self, x):
            return x
        pass

    tokenizer = MockTokenizer()

    # 2. Create an Image Converter
    image_converter = keras.layers.Resizing(height=378, width=378)

    # 3. Instantiate Preprocessor
    preprocessor = MoondreamPreprocessor(
        tokenizer=tokenizer,
        image_converter=image_converter,
        sequence_length=128
    )

    # 4. Preprocess Data
    inputs = {
        "images": np.random.randint(0, 255, (2, 500, 500, 3)),
        "text": ["Describe this image.", "What is in the photo?"]
    }

    outputs = preprocessor(inputs)
    ```
    """

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter

    def call(self, x, y=None, sample_weight=None):
        if isinstance(x, dict):
            text_input = x.get("text", "")
            images = x.get("images", None)
        else:
            text_input = x
            images = None

        output = super().call(text_input, y=y, sample_weight=sample_weight)

        if isinstance(output, tuple):
            x_out = output[0]
        else:
            x_out = output

        if images is not None:
            if self.image_converter:
                images = self.image_converter(images)

            if isinstance(x_out, dict):
                x_out["images"] = images

        return output

    def generate_preprocess(self, x, sequence_length=None):
        if isinstance(x, dict):
            text_input = x.get("text", "")
            images = x.get("images", None)
        else:
            text_input = x
            images = None

        output = super().generate_preprocess(
            text_input, sequence_length=sequence_length
        )

        if images is not None:
            if self.image_converter:
                images = self.image_converter(images)
            output["images"] = images

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": keras.saving.serialize_keras_object(
                    self.image_converter
                ),
            }
        )
        return config
