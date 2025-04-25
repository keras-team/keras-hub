"""LayoutLMv3 document classifier task model."""

import tensorflow as tf
from tensorflow import keras

from keras_hub.src.models.layoutlmv3.layoutlmv3_backbone import LayoutLMv3Backbone


@keras.saving.register_keras_serializable(package="keras_hub")
class LayoutLMv3DocumentClassifier(keras.Model):
    """LayoutLMv3 document classifier task model.

    This model takes text, layout (bounding boxes) and image inputs and outputs
    document classification predictions.

    Args:
        backbone: A LayoutLMv3Backbone instance.
        num_classes: int. Number of classes to classify documents into.
        dropout: float. Dropout probability for the classification head.
        activation: str or callable. The activation function to use on the
            classification head.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        backbone,
        num_classes,
        dropout=0.1,
        activation="softmax",
        **kwargs,
    ):
        inputs = {
            "input_ids": keras.Input(shape=(None,), dtype=tf.int32),
            "bbox": keras.Input(shape=(None, 4), dtype=tf.int32),
            "attention_mask": keras.Input(shape=(None,), dtype=tf.int32),
            "image": keras.Input(shape=(None, None, 3), dtype=tf.float32),
        }

        # Get backbone outputs
        backbone_outputs = backbone(inputs)
        sequence_output = backbone_outputs["sequence_output"]
        pooled_output = backbone_outputs["pooled_output"]

        # Classification head
        x = keras.layers.Dropout(dropout)(pooled_output)
        outputs = keras.layers.Dense(
            num_classes,
            activation=activation,
            name="classifier",
        )(x)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        self.backbone = backbone
        self.num_classes = num_classes
        self.dropout = dropout
        self.activation = activation

    def get_config(self):
        config = super().get_config()
        config.update({
            "backbone": keras.saving.serialize_keras_object(self.backbone),
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "activation": self.activation,
        })
        return config

    @classmethod
    def from_preset(
        cls,
        preset,
        num_classes,
        dropout=0.1,
        activation="softmax",
        **kwargs,
    ):
        """Create a LayoutLMv3 document classifier from a preset.

        Args:
            preset: string. Must be one of "layoutlmv3_base", "layoutlmv3_large".
            num_classes: int. Number of classes to classify documents into.
            dropout: float. Dropout probability for the classification head.
            activation: str or callable. The activation function to use on the
                classification head.
            **kwargs: Additional keyword arguments.

        Returns:
            A LayoutLMv3DocumentClassifier instance.
        """
        backbone = LayoutLMv3Backbone.from_preset(preset)
        return cls(
            backbone=backbone,
            num_classes=num_classes,
            dropout=dropout,
            activation=activation,
            **kwargs,
        ) 