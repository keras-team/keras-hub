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
"""BERT classification model."""

import copy

from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.models.bert.bert_presets import classifier_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.keras_utils import clone_initializer
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.BertClassifier")
class BertClassifier(Task):
    """An end-to-end BERT model for classification tasks.

    This model attaches a classification head to a
    `keras_nlp.model.BertBackbone` instance, mapping from the backbone outputs
    to logits suitable for a classification task. For usage of this model with
    pre-trained weights, use the `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_nlp.models.BertBackbone` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A `keras_nlp.models.BertPreprocessor` or `None`. If
            `None`, this model will not apply preprocessing, and inputs should
            be preprocessed before calling the model.
        activation: Optional `str` or callable, defaults to `None`. The
            activation function to use on the model outputs. Set
            `activation="softmax"` to return output probabilities.
        dropout: float. The dropout probability value, applied after the dense
            layer.

    Examples:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Pretrained classifier.
    classifier = keras_nlp.models.BertClassifier.from_preset(
        "bert_base_en_uncased",
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    classifier.predict(x=features, batch_size=2)

    # Re-compile (e.g., with a learning rate schedule).
    schedule = keras.optimizers.schedules.CosineDecay(5e-5, decay_steps=10_000)
    classifier.compile(
        optimizer=keras_nlp.models.BertOptimizer(schedule),
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
        "segment_ids": tf.constant(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
    }
    labels = [0, 3]

    # Pretrained classifier without preprocessing.
    classifier = keras_nlp.models.BertClassifier.from_preset(
        "bert_base_en_uncased",
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
    tokenizer = keras_nlp.models.BertTokenizer(
        vocabulary=vocab,
    )
    preprocessor = keras_nlp.models.BertPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.BertBackbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    classifier = keras_nlp.models.BertClassifier(
        backbone=backbone,
        preprocessor=preprocessor,
        num_classes=4,
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes,
        preprocessor=None,
        activation="softmax",
        initializer="keras_nlp>BertInitializer",
        dropout=0.1,
        **kwargs,
    ):
        initializer = keras.initializers.get(initializer)
        activation = keras.activations.get(activation)

        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        pooled = keras.layers.Dropout(dropout)(pooled)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=clone_initializer(initializer),
            activation=activation,
            name="logits",
        )(pooled)
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
        self.activation = activation
        self.initializer = initializer
        self.dropout = dropout

        # Compile with defaults.
        self.compile()

    def compile(
        self,
        optimizer="keras_nlp>BertOptimizer",
        loss="sparse_categorical_crossentropy",
        metrics="sparse_categorical_accuracy",
        jit_compile=True,
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            jit_compile=jit_compile,
            **kwargs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "activation": keras.activations.serialize(self.activation),
                "initializer": keras.initializers.serialize(self.initializer),
                "dropout": self.dropout,
            }
        )
        return config

    @classproperty
    def backbone_cls(cls):
        return BertBackbone

    @classproperty
    def preprocessor_cls(cls):
        return BertPreprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets, **classifier_presets})
