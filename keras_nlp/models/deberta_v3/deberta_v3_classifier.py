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
"""DeBERTa classification model."""

import copy

from tensorflow import keras

from keras_nlp.models.deberta_v3.deberta_v3_backbone import DebertaV3Backbone
from keras_nlp.models.deberta_v3.deberta_v3_backbone import (
    deberta_kernel_initializer,
)
from keras_nlp.models.deberta_v3.deberta_v3_preprocessor import (
    DebertaV3Preprocessor,
)
from keras_nlp.models.deberta_v3.deberta_v3_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class DebertaV3Classifier(Task):
    """An end-to-end DeBERTa model for classification tasks.

    This model attaches a classification head to a
    `keras_nlp.model.DebertaV3Backbone` model, mapping from the backbone
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
        backbone: A `keras_nlp.models.DebertaV3` instance.
        num_classes: int. Number of classes to predict.
        hidden_dim: int. The size of the pooler layer.
        dropout: float. Dropout probability applied to the pooled output. For
            the second dropout layer, `backbone.dropout` is used.
        preprocessor: A `keras_nlp.models.DebertaV3Preprocessor` or `None`. If
            `None`, this model will not apply preprocessing, and inputs should
            be preprocessed before calling the model.

    Examples:

    Example usage.
    ```python
    # Define the preprocessed inputs.
    preprocessed_features = {
        "token_ids": tf.ones(shape=(2, 12), dtype=tf.int64),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(1, 12)),
    }
    labels = [0, 3]

    # Randomly initialized DeBERTa encoder
    backbone = keras_nlp.models.DebertaV3Backbone(
        vocabulary_size=128100,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
        bucket_size=6,
    )

    # Create a DeBERTa classifier and fit your data.
    classifier = keras_nlp.models.DebertaV3Classifier(
        backbone,
        num_classes=4,
        preprocessor=None,
    )
    classifier.fit(x=preprocessed_features, y=labels, batch_size=2)

    # Access backbone programatically (e.g., to change `trainable`)
    classifier.backbone.trainable = False
    ```

    Raw string inputs.
    ```python
    # Create a dataset with raw string features in an `(x, y)` format.
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Create a DebertaV3Classifier and fit your data.
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        "deberta_v3_base_en",
        num_classes=4,
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```

    Raw string inputs with customized preprocessing.
    ```python
    # Create a dataset with raw string features in an `(x, y)` format.
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 3]

    # Use a shorter sequence length.
    preprocessor = keras_nlp.models.DebertaV3Preprocessor.from_preset(
        "deberta_v3_base_en",
        sequence_length=128,
    )

    # Create a DebertaV3Classifier and fit your data.
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        "deberta_v3_base_en",
        num_classes=4,
        preprocessor=preprocessor,
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    classifier.fit(x=features, y=labels, batch_size=2)
    ```

    Preprocessed inputs.
    ```python
    # Create a dataset with preprocessed features in an `(x, y)` format.
    preprocessed_features = {
        "token_ids": tf.ones(shape=(2, 12), dtype=tf.int64),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
    }
    labels = [0, 3]

    # Create a DebertaV3Classifier and fit your data.
    classifier = keras_nlp.models.DebertaV3Classifier.from_preset(
        "deberta_v3_base_en",
        num_classes=4,
        preprocessor=None,
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    classifier.fit(x=preprocessed_features, y=labels, batch_size=2)
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes=2,
        hidden_dim=None,
        dropout=0.0,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        if hidden_dim is None:
            hidden_dim = backbone.hidden_dim

        x = backbone(inputs)[:, backbone.start_token_index, :]
        x = keras.layers.Dropout(dropout, name="pooled_dropout")(x)
        x = keras.layers.Dense(
            hidden_dim,
            activation=lambda x: keras.activations.gelu(x, approximate=False),
            name="pooled_dense",
        )(x)
        x = keras.layers.Dropout(backbone.dropout, name="classifier_dropout")(x)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=deberta_kernel_initializer(),
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
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Default compilation
        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(5e-5),
            metrics=keras.metrics.SparseCategoricalAccuracy(),
            jit_compile=is_xla_compatible(self),
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classproperty
    def backbone_cls(cls):
        return DebertaV3Backbone

    @classproperty
    def preprocessor_cls(cls):
        return DebertaV3Preprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
