import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_hub.src.models.deberta_v3.deberta_v3_backbone import (
    deberta_kernel_initializer,
)
from keras_hub.src.models.deberta_v3.deberta_v3_text_classifier_preprocessor import (  # noqa: E501
    DebertaV3TextClassifierPreprocessor,
)
from keras_hub.src.models.text_classifier import TextClassifier


@keras_hub_export(
    [
        "keras_hub.models.DebertaV3TextClassifier",
        "keras_hub.models.DebertaV3Classifier",
    ]
)
class DebertaV3TextClassifier(TextClassifier):
    """An end-to-end DeBERTa model for classification tasks.

    This model attaches a classification head to a
    `keras_hub.model.DebertaV3Backbone` model, mapping from the backbone
    outputs to logit output suitable for a classification task. For usage of
    this model with pre-trained weights, see the `from_preset()` method.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Note: `DebertaV3Backbone` has a performance issue on TPUs, and we recommend
    other models for TPU training and inference.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/microsoft/DeBERTa).

    Args:
        backbone: A `keras_hub.models.DebertaV3` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A `keras_hub.models.DebertaV3TextClassifierPreprocessor`
            or `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        activation: Optional `str` or callable. The
            activation function to use on the model outputs. Set
            `activation="softmax"` to return output probabilities.
            Defaults to `None`.
        hidden_dim: int. The size of the pooler layer.
        dropout: float. Dropout probability applied to the pooled output. For
            the second dropout layer, `backbone.dropout` is used.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Pretrained classifier.
    classifier = keras_hub.models.DebertaV3TextClassifier.from_preset(
        "deberta_v3_base_en",
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
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
    }
    labels = [0, 3]

    # Pretrained classifier without preprocessing.
    classifier = keras_hub.models.DebertaV3TextClassifier.from_preset(
        "deberta_v3_base_en",
        num_classes=4,
        preprocessor=None,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(features)
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=10,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="[PAD]",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        unk_piece="[UNK]",
    )
    tokenizer = keras_hub.models.DebertaV3Tokenizer(
        proto=bytes_io.getvalue(),
    )
    preprocessor = keras_hub.models.DebertaV3TextClassifierPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.DebertaV3Backbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    classifier = keras_hub.models.DebertaV3TextClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```
    """

    backbone_cls = DebertaV3Backbone
    preprocessor_cls = DebertaV3TextClassifierPreprocessor

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
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.pooled_dropout = keras.layers.Dropout(
            dropout,
            dtype=backbone.dtype_policy,
            name="pooled_dropout",
        )
        hidden_dim = hidden_dim or backbone.hidden_dim
        self.pooled_dense = keras.layers.Dense(
            hidden_dim,
            activation=keras.activations.gelu,
            dtype=backbone.dtype_policy,
            name="pooled_dense",
        )
        self.output_dropout = keras.layers.Dropout(
            backbone.dropout,
            dtype=backbone.dtype_policy,
            name="classifier_dropout",
        )
        self.output_dense = keras.layers.Dense(
            num_classes,
            kernel_initializer=deberta_kernel_initializer(),
            activation=activation,
            dtype=backbone.dtype_policy,
            name="logits",
        )

        # === Functional Model ===
        inputs = backbone.input
        x = backbone(inputs)[:, backbone.start_token_index, :]
        x = self.pooled_dropout(x)
        x = self.pooled_dense(x)
        x = self.output_dropout(x)
        outputs = self.output_dense(x)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.backbone = backbone
        self.preprocessor = preprocessor
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
