import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.models.modernbert.modern_bert_text_classifier_preprocessor import (  # noqa: E501
    ModernBertTextClassifierPreprocessor,
)
from keras_hub.src.models.text_classifier import TextClassifier


@keras_hub_export(
    [
        "keras_hub.models.ModernBertTextClassifier",
        "keras_hub.models.ModernBertClassifier",
    ]
)
class ModernBertTextClassifier(TextClassifier):
    """An end-to-end ModernBERT model for classification tasks.

    This model attaches a classification head to a
    `keras_hub.models.ModernBertBackbone` instance.

    The representation of the first token is used as the sequence-level
    representation for classification.

    Args:
        backbone: A `keras_hub.models.ModernBertBackbone` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A
            `keras_hub.models.ModernBertTextClassifierPreprocessor` or
            `None`. If `None`, inputs should already be preprocessed.
        activation: Optional `str` or callable. The activation function
            applied to the output. Set `activation="softmax"` to return
            probabilities.
        hidden_dim: int. The size of the intermediate pooler layer.
            Defaults to the backbone hidden dimension.
        dropout: float. Dropout probability applied to the pooled
            representation and classifier output.
    """

    backbone_cls = ModernBertBackbone
    preprocessor_cls = ModernBertTextClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        activation=None,
        hidden_dim=None,
        dropout=0.0,
        **kwargs,
    ):
        self.backbone = backbone
        self.preprocessor = preprocessor

        hidden_dim = hidden_dim or backbone.hidden_dim

        self.pooled_dropout = keras.layers.Dropout(
            dropout,
            dtype=backbone.dtype_policy,
            name="pooled_dropout",
        )

        self.pooled_dense = keras.layers.Dense(
            hidden_dim,
            activation="tanh",
            dtype=backbone.dtype_policy,
            name="pooled_dense",
        )

        self.output_dropout = keras.layers.Dropout(
            dropout,
            dtype=backbone.dtype_policy,
            name="output_dropout",
        )

        self.output_dense = keras.layers.Dense(
            num_classes,
            activation=activation,
            dtype=backbone.dtype_policy,
            name="logits",
        )

        inputs = backbone.input

        x = backbone(inputs)

        # Use the first token as the sequence representation.
        x = x[:, 0, :]

        x = self.pooled_dropout(x)
        x = self.pooled_dense(x)
        x = self.output_dropout(x)
        outputs = self.output_dense(x)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        self.num_classes = num_classes
        self.activation = keras.activations.get(activation)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "num_classes": self.num_classes,
                "activation": keras.activations.serialize(self.activation),
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
            }
        )

        return config
