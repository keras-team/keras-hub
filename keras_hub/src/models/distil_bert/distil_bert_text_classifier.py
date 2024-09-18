# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    DistilBertBackbone,
)
from keras_hub.src.models.distil_bert.distil_bert_backbone import (
    distilbert_kernel_initializer,
)
from keras_hub.src.models.distil_bert.distil_bert_text_classifier_preprocessor import (
    DistilBertTextClassifierPreprocessor,
)
from keras_hub.src.models.text_classifier import TextClassifier


@keras_hub_export(
    [
        "keras_hub.models.DistilBertTextClassifier",
        "keras_hub.models.DistilBertClassifier",
    ]
)
class DistilBertTextClassifier(TextClassifier):
    """An end-to-end DistilBERT model for classification tasks.

    This model attaches a classification head to a
    `keras_hub.model.DistilBertBackbone` instance, mapping from the backbone
    outputs to logits suitable for a classification task. For usage of
    this model with pre-trained weights, see the `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/huggingface/transformers).

    Args:
        backbone: A `keras_hub.models.DistilBert` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A `keras_hub.models.DistilBertTextClassifierPreprocessor` or `None`. If
            `None`, this model will not apply preprocessing, and inputs should
            be preprocessed before calling the model.
        activation: Optional `str` or callable. The
            activation function to use on the model outputs. Set
            `activation="softmax"` to return output probabilities.
            Defaults to `None`.
        hidden_dim: int. The size of the pooler layer.
        dropout: float. The dropout probability value, applied after the first
            dense layer.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Use a shorter sequence length.
    preprocessor = keras_hub.models.DistilBertTextClassifierPreprocessor.from_preset(
        "distil_bert_base_en_uncased",
        sequence_length=128,
    )
    # Pretrained classifier.
    classifier = keras_hub.models.DistilBertTextClassifier.from_preset(
        "distil_bert_base_en_uncased",
        num_classes=4,
        preprocessor=preprocessor,
    )
    classifier.fit(x=features, y=labels, batch_size=2)

    # Re-compile (e.g., with a new learning rate)
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
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2)
    }
    labels = [0, 3]

    # Pretrained classifier without preprocessing.
    classifier = keras_hub.models.DistilBertTextClassifier.from_preset(
        "distil_bert_base_en_uncased",
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
    tokenizer = keras_hub.models.DistilBertTokenizer(
        vocabulary=vocab,
    )
    preprocessor = keras_hub.models.DistilBertTextClassifierPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.DistilBertBackbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    classifier = keras_hub.models.DistilBertTextClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```
    """

    backbone_cls = DistilBertBackbone
    preprocessor_cls = DistilBertTextClassifierPreprocessor

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        activation=None,
        hidden_dim=None,
        dropout=0.2,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        hidden_dim = hidden_dim or backbone.hidden_dim
        self.pooled_dense = keras.layers.Dense(
            hidden_dim,
            activation="relu",
            kernel_initializer=distilbert_kernel_initializer(),
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
            kernel_initializer=distilbert_kernel_initializer(),
            activation=activation,
            dtype=backbone.dtype_policy,
            name="logits",
        )

        # === Functional Model ===
        inputs = backbone.input
        x = backbone(inputs)[:, backbone.cls_token_index, :]
        x = self.pooled_dense(x)
        x = self.output_dropout(x)
        outputs = self.output_dense(x)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

        # === Config ===
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
