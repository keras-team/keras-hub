import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_backbone import (  # noqa: E501
    OpenAIPrivacyFilterBackbone,
)
from keras_hub.src.models.openai_privacy_filter.openai_privacy_filter_preprocessor import (  # noqa: E501
    OpenAIPrivacyFilterPreprocessor,
)
from keras_hub.src.models.token_classifier import TokenClassifier


@keras_hub_export("keras_hub.models.OpenAIPrivacyFilterTokenClassifier")
class OpenAIPrivacyFilterTokenClassifier(TokenClassifier):
    """OpenAI Privacy Filter token classifier for PII detection.

    This model wraps `OpenAIPrivacyFilterBackbone` with a per-token
    classification head that produces logits for BIOES entity labels.

    The model detects 8 PII entity types (person, email, phone, address,
    date, URL, account number, secret) using 33 BIOES labels.

    Args:
        backbone: A `OpenAIPrivacyFilterBackbone` instance.
        num_classes: int. Number of output classes (BIOES labels).
        classifier_dropout: float. Dropout rate before the classification
            head. Defaults to `0.0`.
        activation: Optional `str` or callable. The activation function to use
            on the model outputs. Set `activation="softmax"` to return output
            probabilities. Defaults to `None`.
        preprocessor: A `OpenAIPrivacyFilterPreprocessor` or `None`. To change
            the maximum sequence length, set `preprocessor.sequence_length`.

    Example:
    ```python
    import keras_hub

    # Load pre-trained classifier.
    classifier = (
        keras_hub.models.OpenAIPrivacyFilterTokenClassifier.from_preset(
            "hf://openai/privacy-filter",
        )
    )
    # Predict PII labels.
    output = classifier.predict(["My name is John Smith."])
    ```
    """

    backbone_cls = OpenAIPrivacyFilterBackbone
    preprocessor_cls = OpenAIPrivacyFilterPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        classifier_dropout=0.0,
        activation=None,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        inputs = backbone.input
        x = backbone(inputs)

        self._classifier_dropout_layer = keras.layers.Dropout(
            rate=classifier_dropout,
            dtype=backbone.dtype_policy,
            name="classifier_dropout",
        )
        x = self._classifier_dropout_layer(x)

        self._classifier_dense = keras.layers.Dense(
            num_classes,
            activation=activation,
            dtype=backbone.dtype_policy,
            name="classifier",
        )
        outputs = self._classifier_dense(x)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.backbone = backbone
        self.num_classes = num_classes
        self.classifier_dropout = classifier_dropout
        self.activation = keras.activations.get(activation)
        self.preprocessor = preprocessor

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "classifier_dropout": self.classifier_dropout,
                "activation": keras.activations.serialize(self.activation),
            }
        )
        return config
