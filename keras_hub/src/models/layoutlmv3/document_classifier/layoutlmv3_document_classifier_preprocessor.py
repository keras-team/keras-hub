"""LayoutLMv3 document classifier preprocessor.

This preprocessor inherits from Preprocessor and adds LayoutLMv3-specific
functionality for document classification.

Example:
```python
# Initialize the preprocessor
preprocessor = LayoutLMv3DocumentClassifierPreprocessor(
    tokenizer=LayoutLMv3Tokenizer.from_preset("layoutlmv3_base"),
    sequence_length=512,
    image_size=(112, 112),
)

# Preprocess input
features = {
    "text": ["Invoice #12345\nTotal: $100.00", "Receipt #67890\nTotal: $50.00"],
    "bbox": [
        [[0, 0, 100, 20], [0, 30, 100, 50]],  # Bounding boxes for first document
        [[0, 0, 100, 20], [0, 30, 100, 50]],  # Bounding boxes for second document
    ],
    "image": tf.random.uniform((2, 112, 112, 3)),  # Random images for demo
}
preprocessed = preprocessor(features)
```
"""

import os
import json
import tensorflow as tf
from keras.saving import register_keras_serializable
from keras.utils import register_keras_serializable
from keras_hub.src.models.preprocessor import Preprocessor
from .layoutlmv3_tokenizer import LayoutLMv3Tokenizer

import keras
from keras import layers
from keras.src.saving import register_keras_serializable

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export(
    [
        "keras_hub.models.LayoutLMv3DocumentClassifierPreprocessor",
        "keras_hub.models.LayoutLMv3Preprocessor",
    ]
)
@register_keras_serializable()
class LayoutLMv3DocumentClassifierPreprocessor(Preprocessor):
    """LayoutLMv3 document classifier preprocessor.
    
    This preprocessor inherits from Preprocessor and adds LayoutLMv3-specific
    functionality for document classification.
    
    Args:
        tokenizer: A LayoutLMv3Tokenizer instance.
        sequence_length: The maximum sequence length to use.
        image_size: A tuple of (height, width) for resizing images.
        **kwargs: Additional keyword arguments.
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