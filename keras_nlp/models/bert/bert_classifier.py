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
"""BERT classification model."""

import copy

from tensorflow import keras

from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_backbone import bert_kernel_initializer
from keras_nlp.models.bert.bert_preprocessor import BertPreprocessor
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.models.bert.bert_presets import classifier_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring

PRESET_NAMES = ", ".join(list(backbone_presets) + list(classifier_presets))


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertClassifier(Task):
    """An end-to-end BERT model for classification tasks

    This model attaches a classification head to a `keras_nlp.model.BertBackbone`
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
        backbone: A `keras_nlp.models.BertBackbone` instance.
        num_classes: int. Number of classes to predict.
        dropout: float. The dropout probability value, applied after the dense
            layer.
        preprocessor: A `keras_nlp.models.BertPreprocessor` or `None`. If
            `None`, this model will not apply preprocessing, and inputs should
            be preprocessed before calling the model.

    Examples:

    ```python
    # Call classifier on the inputs.
    preprocessed_features = {
        "token_ids": tf.ones(shape=(2, 12), dtype=tf.int64),
        "segment_ids": tf.constant(
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2, shape=(2, 12)
        ),
    }
    labels = [0, 3]

    # Randomly initialize a BERT backbone.
    backbone = keras_nlp.models.BertBackbone(
        vocabulary_size=30552,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Create a BERT classifier and fit your data.
    classifier = keras_nlp.models.BertClassifier(
        backbone,
        num_classes=4,
        preprocessor=None,
    )
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    classifier.fit(x=preprocessed_features, y=labels, batch_size=2)

    # Access backbone programatically (e.g., to change `trainable`)
    classifier.backbone.trainable = False
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes=2,
        dropout=0.1,
        preprocessor=None,
        **kwargs,
    ):
        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        pooled = keras.layers.Dropout(dropout)(pooled)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=bert_kernel_initializer(),
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
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "name": self.name,
                "trainable": self.trainable,
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

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        return super().from_preset(
            preset=preset, load_weights=load_weights, **kwargs
        )


BertClassifier.from_preset.__func__.__doc__ = Task.from_preset.__doc__
format_docstring(
    model_task_name=BertClassifier.__name__,
    example_preset_name="bert_base_en_uncased",
    preset_names=PRESET_NAMES,
)(BertClassifier.from_preset.__func__)
