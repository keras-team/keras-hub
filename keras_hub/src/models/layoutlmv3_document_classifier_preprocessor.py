"""LayoutLMv3 document classifier preprocessor implementation.

This module implements a preprocessor for the LayoutLMv3 document classifier.
"""

from typing import Dict, List, Optional, Union

from keras import backend, layers, ops
from keras.saving import register_keras_serializable
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor

from .layoutlmv3_tokenizer import LayoutLMv3Tokenizer

@keras_hub_export("keras_hub.models.LayoutLMv3DocumentClassifierPreprocessor")
class LayoutLMv3DocumentClassifierPreprocessor(Preprocessor):
    """Preprocessor for LayoutLMv3 document classifier.

    This preprocessor handles the preprocessing of text, layout, and image inputs
    for the LayoutLMv3 document classifier.

    Args:
        tokenizer: LayoutLMv3Tokenizer instance or string preset name.
        sequence_length: int, defaults to 512. Maximum sequence length.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
    ```python
    # Initialize preprocessor from preset
    preprocessor = LayoutLMv3DocumentClassifierPreprocessor.from_preset("layoutlmv3_base")

    # Preprocess document
    inputs = preprocessor({
        "text": "Document text",
        "bbox": [[0, 0, 100, 100]],
        "image": image_array
    })
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        image_size=(112, 112),
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            image_size=image_size,
            **kwargs,
        )

    def call(self, x, y=None, sample_weight=None):
        """Process the inputs.

        Args:
            x: A dictionary containing:
                - "text": A string or list of strings to tokenize.
                - "image": A numpy array or list of numpy arrays of shape (112, 112, 3).
                - "bbox": A list of bounding boxes for each token in the text.
            y: Any label data. Will be passed through unaltered.
            sample_weight: Any label weight data. Will be passed through unaltered.

        Returns:
            A tuple of (processed_inputs, y, sample_weight).
        """
        # Tokenize the text
        tokenized = self.tokenizer(x["text"])
        input_ids = tokenized["token_ids"]
        attention_mask = tokenized["attention_mask"]

        # Process bounding boxes
        bbox = x["bbox"]
        if isinstance(bbox, list):
            bbox = tf.ragged.constant(bbox)
        bbox = bbox.to_tensor(shape=(None, self.sequence_length, 4))

        # Process image
        image = x["image"]
        if isinstance(image, list):
            image = tf.stack(image)
        image = tf.cast(image, tf.float32)

        # Pad or truncate inputs
        input_ids = input_ids[:, : self.sequence_length]
        attention_mask = attention_mask[:, : self.sequence_length]
        bbox = bbox[:, : self.sequence_length]

        # Create padding mask
        padding_mask = tf.cast(attention_mask, tf.int32)

        # Return processed inputs
        processed_inputs = {
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "image": image,
        }

        return processed_inputs, y, sample_weight

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": keras.saving.serialize_keras_object(self.tokenizer),
                "sequence_length": self.sequence_length,
                "image_size": self.image_size,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config:
            config["tokenizer"] = keras.saving.deserialize_keras_object(
                config["tokenizer"]
            )
        return cls(**config)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate LayoutLMv3DocumentClassifierPreprocessor from preset.

        Args:
            preset: string. Must be one of "layoutlmv3_base", "layoutlmv3_large".

        Examples:
        ```python
        # Load preprocessor from preset
        preprocessor = LayoutLMv3DocumentClassifierPreprocessor.from_preset("layoutlmv3_base")
        ```
        """
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}"""
            )

        metadata = cls.presets[preset]
        config = metadata["config"]

        # Create tokenizer
        tokenizer = LayoutLMv3Tokenizer.from_preset(preset)

        # Create preprocessor
        preprocessor = cls(
            tokenizer=tokenizer,
            sequence_length=config["sequence_length"],
            image_size=config["image_size"],
            **kwargs,
        )

        return preprocessor 