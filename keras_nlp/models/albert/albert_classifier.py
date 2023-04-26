# Copyright 2022 The KerasNLP Authors
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
"""ALBERT classification model."""

import copy

from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.albert.albert_backbone import AlbertBackbone
from keras_nlp.models.albert.albert_backbone import albert_kernel_initializer
from keras_nlp.models.albert.albert_preprocessor import AlbertPreprocessor
from keras_nlp.models.albert.albert_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.AlbertClassifier")
class AlbertClassifier(Task):
    """An end-to-end ALBERT model for classification tasks

    This model attaches a classification head to a `keras_nlp.model.AlbertBackbone`
    backbone, mapping from the backbone outputs to logit output suitable for
    a classification task. For usage of this model with pre-trained weights, see
    the `from_preset()` method.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_nlp.models.AlertBackbone` instance.
        num_classes: int. Number of classes to predict.
        preprocessor: A `keras_nlp.models.AlbertPreprocessor` or `None`. If
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
    classifier = keras_nlp.models.AlbertClassifier.from_preset(
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
    classifier = keras_nlp.models.AlbertClassifier.from_preset(
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
    tokenizer = keras_nlp.models.AlbertTokenizer(
        proto=bytes_io.getvalue(),
    )
    preprocessor = keras_nlp.models.AlbertPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_nlp.models.AlbertBackbone(
        vocabulary_size=tokenizer.vocabulary_size(),
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        embedding_dim=128,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    classifier = keras_nlp.models.AlbertClassifier(
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
        activation=None,
        dropout=0.1,
        **kwargs,
    ):
        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        pooled = keras.layers.Dropout(dropout)(pooled)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=albert_kernel_initializer(),
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
        self._backbone = backbone
        self._preprocessor = preprocessor
        self.num_classes = num_classes
        self.activation = keras.activations.get(activation)
        self.dropout = dropout

        # Default compilation
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
                "dropout": self.dropout,
            }
        )

        return config

    @classproperty
    def backbone_cls(cls):
        return AlbertBackbone

    @classproperty
    def preprocessor_cls(cls):
        return AlbertPreprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets})
