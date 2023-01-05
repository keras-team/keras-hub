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
"""BERT feature extraction model."""

import copy
import os

from tensorflow import keras

from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.utils.pipeline_model import PipelineModel
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring

PRESET_NAMES = ", ".join(list(backbone_presets))


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertFeatureExtractor(PipelineModel):
    """An end-to-end BERT model for feature extraction task. This model
    can't be compiled or fitted, and should be used as a submodel only.

    For usage of this model with pre-trained weights, see
    the `from_preset()` method.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to raw inputs during
    `fit()`, `predict()`, and `evaluate()`. This is done by default when
    creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_nlp.models.BertBackbone` instance.
        dropout: float. The dropout probability value, applied after the dense
            layer.
        preprocessor: A `keras_nlp.models.BertPreprocessor` or `None`. If
            `None`, this model will not apply preprocessing, and inputs should
            be preprocessed before calling the model.

    Examples:

    ```python
    # Call featurizer on the inputs.
    preprocessed_features = {
        "token_ids": tf.ones(shape=(2, 12), dtype=tf.int64),
        "segment_ids": tf.constant(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
    }

    # Randomly initialize a BERT backbone.
    backbone = keras_nlp.models.BertBackbone(
        vocabulary_size=30552,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Create a BERT featurizer and evaluate on data
    featurizer = keras_nlp.models.BertFeatureExtractor(
        backbone,
        preprocessor=None,
    )

    outputs = featurizer(preprocessed_features)

    # Using a classifier head (num_classes = 4)

    features = outputs
    outputs = keras.layers.Dense(4)(outputs["pooled_output"])
    classifier = keras.Model(features, outputs)

    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    labels = [0, 0, 0 , 3]

    classifier.fit(x=preprocessed_features, y=labels, batch_size=2)

    # Access backbone programatically (e.g., to change `trainable`)
    featurizer.backbone.trainable = False

    ```
    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        outputs = backbone(inputs)
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

    def preprocess_samples(self, x, y=None, sample_weight=None):
        return self.preprocessor(x, y=y, sample_weight=sample_weight)

    @property
    def backbone(self):
        """A `keras_nlp.models.BertBackbone` instance providing the encoder
        submodel.
        """
        return self._backbone

    @property
    def preprocessor(self):
        """A `keras_nlp.models.BertPreprocessor` for preprocessing inputs."""
        return self._preprocessor

    def get_config(self):
        return {
            "backbone": keras.layers.serialize(self.backbone),
            "preprocessor": keras.layers.serialize(self.preprocessor),
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        if "backbone" in config and isinstance(config["backbone"], dict):
            config["backbone"] = keras.layers.deserialize(config["backbone"])
        if "preprocessor" in config and isinstance(
            config["preprocessor"], dict
        ):
            config["preprocessor"] = keras.layers.deserialize(
                config["preprocessor"]
            )
        return cls(**config)

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets})

    @classmethod
    @format_docstring(names=PRESET_NAMES)
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Create a featue extraction model from a preset architecture and weights.

        By default, this method will automatically create a `preprocessor`
        layer to preprocess raw inputs during `fit()`, `predict()`, and
        `evaluate()`. If you would like to disable this behavior, pass
        `preprocessor=None`.

        Args:
            preset: string. Must be one of {{names}}.
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:

        Raw string inputs.
        ```python
        # Create a dataset with raw string features in an `(x, y)` format.
        features = ["The quick brown fox jumped.", "I forgot my homework."]

        # Create a BERT featurizer and evaluate on data
        featurizer = keras_nlp.models.BertFeatureExtractor.from_preset(
            "bert_base_en_uncased",
        )

        y_eval = featurizer(preprocessed_features)
        ```

        Raw string inputs with customized preprocessing.
        ```python
        # Create a dataset with raw string features in an `(x, y)` format.
        features = ["The quick brown fox jumped.", "I forgot my homework."]

        # Use a shorter sequence length.
        preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
            "bert_base_en_uncased",
            sequence_length=128,
        )

        # Create a BERT featurizer and evaluate on data
        featurizer = keras_nlp.models.BertFeatureExtractor.from_preset(
            "bert_base_en_uncased",
            preprocessor=preprocessor,
        )
        y_eval = featurizer(preprocessed_features)
        ```

        Preprocessed inputs.
        ```python
        # Create a dataset with preprocessed features in an `(x, y)` format.
        preprocessed_features = {
            "token_ids": tf.ones(shape=(2, 12), dtype=tf.int64),
            "segment_ids": tf.constant(
                [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
            ),
            "padding_mask": tf.constant(
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
            ),
        }

        # Create a BERT featurizer and evaluate on data
        featurizer = keras_nlp.models.BertFeatureExtractor.from_preset(
            "bert_base_en_uncased",
            preprocessor=None,
        )

        y_eval = featurizer(preprocessed_features)
        ```
        """
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        if "preprocessor" not in kwargs:
            kwargs["preprocessor"] = BertPreprocessor.from_preset(preset)

        # Check if preset is backbone-only model
        if preset in BertBackbone.presets:
            backbone = BertBackbone.from_preset(preset, load_weights)
            return cls(backbone, **kwargs)

        # Otherwise must be one of class presets
        metadata = cls.presets[preset]
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )

        model.load_weights(weights)
        return model
