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
"""DistilBERT task specific models and heads."""

from tensorflow import keras

from keras_nlp.models.distilbert.distilbert_models import (
    distilbert_kernel_initializer,
)


@keras.utils.register_keras_serializable(package="keras_nlp")
class DistilBertClassifier(keras.Model):
    """DistilBERT encoder model with a classification head.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/huggingface/transformers).

    Args:
        backbone: A `keras_nlp.models.DistilBert` instance.
        num_classes: int. Number of classes to predict.
        hidden_dim: int. The size of the pooler layer.

    Example usage:
    ```python
    # Randomly initialized DistilBERT encoder
    model = keras_nlp.models.DistilBert(
        vocabulary_size=30522,
        num_layers=6,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=512
    )

    # Call classifier on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    classifier = keras_nlp.models.DistilBertClassifier(model, 4)
    logits = classifier(input_data)
    ```
    """

    def __init__(
        self,
        backbone,
        num_classes,
        hidden_dim=None,
        dropout=0.2,
        **kwargs,
    ):
        inputs = backbone.input
        if hidden_dim is None:
            hidden_dim = backbone.hidden_dim

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
        """A `keras_nlp.models.DistilBert` instance providing the encoder submodel."""
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
