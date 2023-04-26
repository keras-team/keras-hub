# Copyright 2023 The KerasNLP Authors
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
"""DistilBERT classification model."""

import copy

from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.distil_bert.distil_bert_backbone import DistilBertBackbone
from keras_nlp.models.distil_bert.distil_bert_backbone import (
    distilbert_kernel_initializer,
)
from keras_nlp.models.distil_bert.distil_bert_preprocessor import (
    DistilBertPreprocessor,
)
from keras_nlp.models.distil_bert.distil_bert_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.DistilBertClassifier")
class DistilBertClassifier(Task):
    """An end-to-end DistilBERT model for classification tasks.

    This model attaches a classification head to a
    `keras_nlp.model.DistilBertBackbone` instance, mapping from the backbone
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
        backbone: A `keras_nlp.models.DistilBert` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A `keras_nlp.models.DistilBertPreprocessor` or `None`. If
            `None`, this model will not apply preprocessing, and inputs should
            be preprocessed before calling the model.
        activation: Optional `str` or callable, defaults to `None`. The
            activation function to use on the model outputs. Set
            `activation="softmax"` to return output probabilities.
        hidden_dim: int. The size of the pooler layer.
        dropout: float. The dropout probability value, applied after the first
            dense layer.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Use a shorter sequence length.
    preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
        "distil_bert_base_en_uncased",
        sequence_length=128,
    )
    # Pretrained classifier.
    classifier = keras_nlp.models.DistilBertClassifier.from_preset(
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
    # Access backbone programatically (e.g., to change `trainable`).
    classifier.backbone.trainable = False
    # Fit again.
    classifier.fit(x=features, y=labels, batch_size=2)
    ```

    Preprocessed integer data.
    ```python
    features = {
        "token_ids": tf.ones(shape=(2, 12), dtype=tf.int64),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
    }
    labels = [0, 3]

    # Pretrained classifier without preprocessing.
    classifier = keras_nlp.models.DistilBertClassifier.from_preset(
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
    tokenizer = keras_nlp.models.DistilBertTokenizer(
        vocabulary=vocab,
    )
    preprocessor = keras_nlp.models.DistilBertPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.DistilBertBackbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    classifier = keras_nlp.models.DistilBertClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    """

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
        inputs = backbone.input
        hidden_dim = hidden_dim or backbone.hidden_dim

        x = backbone(inputs)[:, backbone.cls_token_index, :]
        x = keras.layers.Dense(
            hidden_dim,
            activation="relu",
            kernel_initializer=distilbert_kernel_initializer(),
            name="pooled_dense",
        )(x)
        x = keras.layers.Dropout(dropout, name="classifier_dropout")(x)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=distilbert_kernel_initializer(),
            activation=activation,
            name="logits",
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )
        # All references to `self` below this line
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.num_classes = num_classes
        self.activation = keras.activations.get(activation)
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(
                from_logits=activation is None
            ),
            optimizer=keras.optimizers.Adam(5e-5),
            metrics=keras.metrics.SparseCategoricalAccuracy(),
            jit_compile=is_xla_compatible(self),
        )

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

    @classproperty
    def backbone_cls(cls):
        return DistilBertBackbone

    @classproperty
    def preprocessor_cls(cls):
        return DistilBertPreprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
