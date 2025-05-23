"""LayoutLMv3 document classifier implementation.

This module implements a document classification model using the LayoutLMv3 backbone.
"""

from typing import Dict, List, Optional, Union

from keras import backend, layers, ops
from keras.saving import register_keras_serializable
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone

from .layoutlmv3_backbone import LayoutLMv3Backbone
from .layoutlmv3_document_classifier_preprocessor import LayoutLMv3DocumentClassifierPreprocessor

@keras_hub_export("keras_hub.models.LayoutLMv3DocumentClassifier")
class LayoutLMv3DocumentClassifier(layers.Layer):
    """Document classifier using LayoutLMv3 backbone.

    This model uses the LayoutLMv3 backbone for document classification tasks,
    adding a classification head on top of the backbone's pooled output.

    Args:
        backbone: LayoutLMv3Backbone instance or string preset name.
        num_classes: int, defaults to 2. Number of output classes.
        dropout: float, defaults to 0.1. Dropout rate for the classification head.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:
    ```python
    # Initialize classifier from preset
    classifier = LayoutLMv3DocumentClassifier.from_preset("layoutlmv3_base")

    # Process document
    outputs = classifier({
        "input_ids": input_ids,
        "bbox": bbox,
        "attention_mask": attention_mask,
        "image": image
    })
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.num_classes = num_classes
        self.dropout = dropout

    def call(self, inputs):
        # Get backbone outputs
        backbone_outputs = self.backbone(inputs)
        sequence_output = backbone_outputs["sequence_output"]
        pooled_output = backbone_outputs["pooled_output"]

        # Classification head
        x = layers.Dropout(self.dropout)(pooled_output)
        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="classifier",
        )(x)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone": self.backbone,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        })
        return config

    @classmethod
    def from_preset(
        cls,
        preset,
        num_classes=2,
        dropout=0.1,
        **kwargs,
    ):
        """Create a LayoutLMv3 document classifier from a preset.

        Args:
            preset: string. Must be one of "layoutlmv3_base", "layoutlmv3_large".
            num_classes: int. Number of classes to classify documents into.
            dropout: float. Dropout probability for the classification head.
            **kwargs: Additional keyword arguments.

        Returns:
            A LayoutLMv3DocumentClassifier instance.
        """
        backbone = LayoutLMv3Backbone.from_preset(preset)
        return cls(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            **kwargs,
        ) 