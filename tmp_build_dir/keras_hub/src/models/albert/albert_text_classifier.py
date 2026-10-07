import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.albert.albert_backbone import AlbertBackbone
from keras_hub.src.models.albert.albert_backbone import (
    albert_kernel_initializer,
)
from keras_hub.src.models.albert.albert_text_classifier_preprocessor import (
    AlbertTextClassifierPreprocessor,
)
from keras_hub.src.models.text_classifier import TextClassifier


@keras_hub_export(
    [
        "keras_hub.models.AlbertTextClassifier",
        "keras_hub.models.AlbertClassifier",
    ]
)
class AlbertTextClassifier(TextClassifier):
    """An end-to-end ALBERT model for classification tasks

    This model attaches a classification head to a
    `keras_hub.model.AlbertBackbone` backbone, mapping from the backbone outputs
    to logit output suitable for a classification task. For usage of this model
    with pre-trained weights, see the `from_preset()` method.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.AlertBackbone` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A `keras_hub.models.AlbertTextClassifierPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
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
    classifier = keras_hub.models.AlbertTextClassifier.from_preset(
        "albert_base_en_uncased",
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
    classifier = keras_hub.models.AlbertTextClassifier.from_preset(
        "albert_base_en_uncased",
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
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        user_defined_symbols="[MASK]",
    )
    tokenizer = keras_hub.models.AlbertTokenizer(
        proto=bytes_io.getvalue(),
    )
    preprocessor = keras_hub.models.AlbertTextClassifierPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.AlbertBackbone(
        vocabulary_size=tokenizer.vocabulary_size(),
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        embedding_dim=128,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    classifier = keras_hub.models.AlbertTextClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```
    """

    backbone_cls = AlbertBackbone
    preprocessor_cls = AlbertTextClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        activation=None,
        dropout=0.1,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.output_dense = keras.layers.Dense(
            num_classes,
            kernel_initializer=albert_kernel_initializer(),
            activation=activation,
            dtype=backbone.dtype_policy,
            name="logits",
        )
        self.output_dropout = keras.layers.Dropout(
            dropout,
            dtype=backbone.dtype_policy,
            name="output_dropout",
        )

        # === Functional Model ===
        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        pooled = self.output_dropout(pooled)
        outputs = self.output_dense(pooled)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
        self.num_classes = num_classes
        self.activation = keras.activations.get(activation)
        self.dropout = dropout

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        metrics="auto",
        **kwargs,
    ):
        if optimizer == "auto":
            optimizer = keras.optimizers.Adam(1e-5)
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": keras.activations.serialize(self.activation),
                "dropout": self.dropout,
            }
        )

        return config
