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

from tensorflow import keras

from keras_nlp.models.bert.bert_models import Bert
from keras_nlp.models.bert.bert_models import bert_kernel_initializer

# TODO(jbischof): Find more scalable way to list checkpoints.
CLASSIFIER_DOCSTRING = """BERT encoder model with a classification head.

    Args:
        backbone: A string or `keras_nlp.models.Bert` instance. If a string,
            should be one of {names}.
        num_classes: int. Number of classes to predict.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Examples:
    ```python
    # Randomly initialized BERT encoder
    model = keras_nlp.models.BertCustom(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

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
    classifier = keras_nlp.models.BertClassifier(model, 4, name="classifier")
    logits = classifier(input_data)

    # String backbone specification
    classifier = keras_nlp.models.BertClassifier(
        "bert_base_uncased_en", 4, name="classifier"
    )
    logits = classifier(input_data)
    ```
"""


@keras.utils.register_keras_serializable(package="keras_nlp")
class BertClassifier(keras.Model):
    def __init__(
        self,
        backbone="bert_base_uncased_en",
        num_classes=2,
        name=None,
        trainable=True,
    ):
        # Load backbone from string identifier
        # TODO(jbischof): create util function when ready to load backbones in
        # other task classes (e.g., load_backbone_from_string())
        if isinstance(backbone, str):
            if backbone not in Bert.presets:
                raise ValueError(
                    "`backbone` must be one of "
                    f"""{", ".join(Bert.presets)}. Received: {backbone}."""
                )
            backbone = Bert.from_preset(backbone)

        inputs = backbone.input
        pooled = backbone(inputs)["pooled_output"]
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=bert_kernel_initializer(),
            name="logits",
        )(pooled)
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs, outputs=outputs, name=name, trainable=trainable
        )
        # All references to `self` below this line
        self.backbone = backbone
        self.num_classes = num_classes

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


setattr(
    BertClassifier,
    "__doc__",
    CLASSIFIER_DOCSTRING.format(names=", ".join(Bert.presets)),
)
