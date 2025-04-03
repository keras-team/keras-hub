from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.roberta.roberta_text_classifier import (
    RobertaTextClassifier,  # noqa: E501
)
from keras_hub.src.models.roformer_v2.roformer_v2_backbone import (
    RoformerV2Backbone,
)
from keras_hub.src.models.roformer_v2.roformer_v2_text_classifier_preprocessor import (  # noqa: E501
    RoformerV2TextClassifierPreprocessor,
)


@keras_hub_export("keras_hub.models.RoformerV2TextClassifier")
class RoformerV2TextClassifier(RobertaTextClassifier):
    """An end-to-end RoformerV2 model for classification tasks.

    This model attaches a classification head to
    `keras_hub.model.RoformerV2Backbone`, mapping from the backbone outputs
    to logits suitable for a classification task. For usage of this model with
    pre-trained weights, use the `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.RoformerV2Backbone` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A `keras_hub.models.RoformerV2TextClassifierPreprocessor`
            or `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        activation: Optional `str` or callable. The
            activation function to use on the model outputs. Set
            `activation="softmax"` to return output probabilities.
            Defaults to `None`.
        dropout: float. The dropout probability value, applied after the dense
            layer.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Pretrained classifier.
    classifier = keras_hub.models.RoformerV2TextClassifier.from_preset(
        "roformer_v2_base_zh",
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    classifier.predict(x=features, batch_size=2)

    # Re-compile (e.g., with a new learning rate).
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        jit_compile=True,
    )
    # Access backbone programmatically (e.g., to change `trainable`).
    classifier.backbone.trainable = False
    # Fit again.
    classifier.fit(x=features, y=labels, batch_size=2)
    ```

    Preprocessed integer data.
    ```python
    features = {
        "token_ids": np.ones(shape=(2, 12), dtype="int32"),
        "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
    }
    labels = [0, 3]

    # Pretrained classifier without preprocessing.
    classifier = keras_hub.models.RoformerV2TextClassifier.from_preset(
        "roformer_v2_base_zh",
        num_classes=4,
        preprocessor=None,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]
    tokenizer = keras_hub.models.RoformerV2Tokenizer(
        vocabulary=vocab,
    )
    preprocessor = keras_hub.models.RoformerV2TextClassifierPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.RoformerV2Backbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_wavelength=128,
        head_size=64,
    )
    classifier = keras_hub.models.RoformerV2TextClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```
    """

    backbone_cls = RoformerV2Backbone
    preprocessor_cls = RoformerV2TextClassifierPreprocessor
