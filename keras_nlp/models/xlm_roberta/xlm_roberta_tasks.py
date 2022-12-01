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
"""XLM-RoBERTa task specific models and heads."""

import copy

from tensorflow import keras

from keras_nlp.models.roberta.roberta_models import roberta_kernel_initializer
from keras_nlp.models.xlm_roberta.xlm_roberta_models import XLMRoberta
from keras_nlp.models.xlm_roberta.xlm_roberta_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
class XLMRobertaClassifier(keras.Model):
    """XLM-RoBERTa encoder model with a classification head.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq).

    Args:
        backbone: A `keras_nlp.models.XLMRoberta` instance.
        num_classes: int. Number of classes to predict.
        hidden_dim: int. The size of the pooler layer.

    Example usage:
    ```python
    # Randomly initialized XLM-RoBERTa encoder
    model = keras_nlp.models.XLMRoberta(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call classifier on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    classifier = keras_nlp.models.XLMRobertaClassifier(
        backbone=model,
        num_classes=4,
    )
    logits = classifier(input_data)
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes=2,
        hidden_dim=None,
        dropout=0.0,
        **kwargs,
    ):
        inputs = backbone.input
        if hidden_dim is None:
            hidden_dim = backbone.hidden_dim

        x = backbone(inputs)[:, backbone.start_token_index, :]
        x = keras.layers.Dropout(dropout, name="pooled_dropout")(x)
        x = keras.layers.Dense(
            hidden_dim, activation="tanh", name="pooled_dense"
        )(x)
        x = keras.layers.Dropout(dropout, name="classifier_dropout")(x)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=roberta_kernel_initializer(),
            name="logits",
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )
        # All references to `self` below this line
        self._backbone = backbone
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    @property
    def backbone(self):
        """A `keras_nlp.models.XLMRoberta` instance providing the encoder submodel."""
        return self._backbone

    def get_config(self):
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
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
    @format_docstring(names=", ".join(backbone_presets))
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Create a classification model from a preset architecture and weights.

        Args:
            preset: string. Must be one of {{names}}.
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        input_data = {
            "token_ids": tf.random.uniform(
                shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
            ),
            "padding_mask": tf.constant(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
            ),
        }

        # Load backbone architecture and weights from preset
        classifier = keras_nlp.models.XLMRobertaClassifier.from_preset(
            "xlm_roberta_base",
            num_classes=4,
        )
        output = classifier(input_data)

        # Load randomly initalized model from preset architecture
        classifier = keras_nlp.models.XLMRobertaClassifier.from_preset(
            "xlm_roberta_base",
            load_weights=False,
            num_classes=4,
        )
        output = classifier(input_data)
        ```
        """
        # Check if preset is backbone-only model
        if preset in XLMRoberta.presets:
            backbone = XLMRoberta.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets
        # Currently no classifier-level presets, so must throw.
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
