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

"""BERT model task heads and preconfigured versions."""

from tensorflow import keras

from keras_nlp.models.bert import BertBase
from keras_nlp.models.bert import BertLarge
from keras_nlp.models.bert import BertMedium
from keras_nlp.models.bert import BertSmall
from keras_nlp.models.bert import BertTiny
from keras_nlp.models.bert import bert_kernel_initializer

# Metadata for loading pretrained model weights.
backbone_checkpoints = {
    "tiny_uncased_en": {
        "model": BertTiny,
        "weights": "uncased_en",
        "description": (
            "Tiny size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
    },
    "small_uncased_en": {
        "model": BertSmall,
        "weights": "uncased_en",
        "description": (
            "Small size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
    },
    "medium_uncased_en": {
        "model": BertMedium,
        "weights": "uncased_en",
        "description": (
            "Medium size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
    },
    "base_uncased_en": {
        "model": BertBase,
        "weights": "uncased_en",
        "description": (
            "Base size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
    },
    "base_cased_en": {
        "model": BertBase,
        "weights": "cased_en",
        "description": (
            "Base size of Bert where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
    },
    "base_zh": {
        "model": BertBase,
        "weights": "zh",
        "description": ("Base size of BERT. Trained on Chinese Wikipedia."),
    },
    "base_multi_cased": {
        "model": BertBase,
        "weights": "multi_cased",
        "description": ("Base size of BERT. Trained on Chinese Wikipedia."),
    },
    "large_uncased_en": {
        "model": BertLarge,
        "weights": "uncased_en",
        "description": (
            "Large size of BERT where all input is lowercased. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
    },
    "large_cased_en": {
        "model": BertLarge,
        "weights": "cased_en",
        "description": (
            "Base size of Bert where case is maintained. "
            "Trained on English Wikipedia + BooksCorpus."
        ),
    },
}


class BertClassifier(keras.Model):
    """BERT encoder model with a classification head.

    Args:
        backbone: A string, `keras_nlp.models.BertCustom` or derivative such as
            `keras_nlp.models.BertBase` to encode inputs. If a string, should be
            one of `keras_nlp.models.bert_tasks.backbone_checkpoints`.
        num_classes: Int. Number of classes to predict.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
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
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }
    classifier = keras_nlp.models.BertClassifier(model, 4, name="classifier")
    logits = classifier(input_data)

    # String backbone specification
    classifier = keras_nlp.models.BertClassifier(
        "base_uncased_en", 4, name="classifier"
    )
    logits = classifier(input_data)
    ```
    """

    def __init__(
        self,
        backbone="base_uncased_en",
        num_classes=2,
        name=None,
        trainable=True,
    ):
        # Load backbone from string identifier
        if isinstance(backbone, str):
            if backbone not in backbone_checkpoints:
                raise ValueError(
                    "`backbone` must be one of "
                    f"""{", ".join(backbone_checkpoints.keys())}. """
                    f"Received: {backbone}"
                )
            backbone = backbone_checkpoints[backbone]["model"](
                backbone_checkpoints[backbone]["weights"]
            )

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
