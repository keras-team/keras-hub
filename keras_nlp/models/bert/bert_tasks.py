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
"""BERT task specific models and heads."""

import copy

from tensorflow import keras

from keras_nlp.models.bert.bert_models import Bert
from keras_nlp.models.bert.bert_models import bert_kernel_initializer
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.models.utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertClassifier(keras.Model):
    """BERT encoder model with a classification head.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_nlp.models.Bert` instance.
        num_classes: int. Number of classes to predict.

    Examples:
    ```python
    # Call classifier on the inputs.
    input_data = {{
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }}

    # Randomly initialized BERT encoder
    backbone = keras_nlp.models.Bert(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )
    classifier = keras_nlp.models.BertClassifier(backbone, 4, name="classifier")
    logits = classifier(input_data)

    # Access backbone programatically (e.g., to change `trainable`)
    classifier.backbone.trainable = False
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes=2,
        **kwargs,
    ):
        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=bert_kernel_initializer(),
            name="logits",
        )(pooled)
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        # All references to `self` below this line
        self._backbone = backbone
        self.num_classes = num_classes

    @property
    def backbone(self):
        """A `keras_nlp.models.Bert` instance providing the encoder submodel."""
        return self._backbone

    def get_config(self):
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "num_classes": self.num_classes,
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config:
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        return cls(**config)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        # Check if preset is backbone-only model
        if preset in Bert.presets:
            backbone = Bert.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets
        # Currently no classifier-level presets, so must throw.
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )


FROM_PRESET_DOCSTRING = """Instantiate BERT classification model from preset architecture and
    weights.

    Args:
        preset: string. Must be one of {names}.
        load_weights: Whether to load pre-trained weights into model. Defaults
            to `True`.

    Examples:
    ```python
    input_data = {{
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }}

    # Load backbone architecture and weights from preset
    classifier = BertClassifier.from_preset(
        "bert_base_uncased_en",
        num_classes=4,
    )
    output = classifier(input_data)

    # Load randomly initalized model from preset architecture
    classifier = BertClassifier.from_preset(
        "bert_base_uncased_en",
        load_weights=False,
        num_classes=4,
    )
    output = classifier(input_data)
    ```
    """

setattr(
    BertClassifier.from_preset.__func__,
    "__doc__",
    FROM_PRESET_DOCSTRING.format(names=", ".join(BertClassifier.presets)),
)
