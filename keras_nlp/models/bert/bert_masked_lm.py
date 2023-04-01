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
"""BERT masked LM model."""

import copy

from tensorflow import keras

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.masked_lm_head import MaskedLMHead
from keras_nlp.models.bert.bert_backbone import BertBackbone
from keras_nlp.models.bert.bert_backbone import bert_kernel_initializer
from keras_nlp.models.bert.bert_masked_lm_preprocessor import (
    BertMaskedLMPreprocessor,
)
from keras_nlp.models.bert.bert_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.keras_utils import is_xla_compatible
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.BertMaskedLM")
class BertMaskedLM(Task):
    """An end-to-end BERT model for the masked language modeling task.

    This model will train BERT on a masked language modeling task.
    The model will predict labels for a number of masked tokens in the
    input data. For usage of this model with pre-trained weights, see the
    `from_preset()` constructor.

    This model can optionally be configured with a `preprocessor` layer, in
    which case inputs can be raw string features during `fit()`, `predict()`,
    and `evaluate()`. Inputs will be tokenized and dynamically masked during
    training and evaluation. This is done by default when creating the model
    with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_nlp.models.BertBackbone` instance.
        preprocessor: A `keras_nlp.models.BertMaskedLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Example usage:

    Raw string data.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]

    # Pretrained language model.
    masked_lm = keras_nlp.models.BertMaskedLM.from_preset(
        "bert_base_en_uncased",
    )
    masked_lm.fit(x=features, batch_size=2)

    # Re-compile (e.g., with a new learning rate).
    masked_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        jit_compile=True,
    )
    # Access backbone programatically (e.g., to change `trainable`).
    masked_lm.backbone.trainable = False
    # Fit again.
    masked_lm.fit(x=features, batch_size=2)
    ```

    Preprocessed integer data.
    ```python
    # Create preprocessed batch where 0 is the mask token.
    features = {
        "token_ids": tf.constant(
            [[1, 2, 0, 4, 0, 6, 7, 8]] * 2, shape=(2, 8)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1]] * 2, shape=(2, 8)
        ),
        "mask_positions": tf.constant([[2, 4]] * 2, shape=(2, 2)),
        "segment_ids": tf.constant([[0, 0, 0, 0, 0, 0, 0, 0]] * 2, shape=(2, 8))
    }
    # Labels are the original masked values.
    labels = [[3, 5]] * 2

    masked_lm = keras_nlp.models.BertMaskedLM.from_preset(
        "bert_base_en_uncased",
        preprocessor=None,
    )
    masked_lm.fit(x=features, y=labels, batch_size=2)
    ```
    """

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        inputs = {
            **backbone.input,
            "mask_positions": keras.Input(
                shape=(None,), dtype="int32", name="mask_positions"
            ),
        }
        backbone_outputs = backbone(backbone.input)
        outputs = MaskedLMHead(
            vocabulary_size=backbone.vocabulary_size,
            embedding_weights=backbone.token_embedding.embeddings,
            intermediate_activation="gelu",
            kernel_initializer=bert_kernel_initializer(),
            name="mlm_head",
        )(backbone_outputs["sequence_output"], inputs["mask_positions"])

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
        self.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=keras.optimizers.Adam(5e-5),
            weighted_metrics=keras.metrics.SparseCategoricalAccuracy(),
            jit_compile=is_xla_compatible(self),
        )

    @classproperty
    def backbone_cls(cls):
        return BertBackbone

    @classproperty
    def preprocessor_cls(cls):
        return BertMaskedLMPreprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
