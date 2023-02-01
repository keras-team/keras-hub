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
"""RoBERTa masked lm model."""

import copy

from tensorflow import keras

from keras_nlp.layers.masked_lm_head import MaskedLMHead
from keras_nlp.models.roberta.roberta_backbone import RobertaBackbone
from keras_nlp.models.roberta.roberta_backbone import roberta_kernel_initializer
from keras_nlp.models.roberta.roberta_masked_lm_preprocessor import (
    RobertaMaskedLMPreprocessor,
)
from keras_nlp.models.roberta.roberta_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class RobertaMaskedLM(Task):
    """An end-to-end RoBERTa model for the masked language modeling task.

    This model will train RoBERTa on a masked language modeling task.
    The model will predict labels for a number of masked tokens in the
    input data. For usage of this model with pre-trained weights, see the
    `from_preset()` method.

    This model can optionally be configured with a `preprocessor` layer, in
    which case inputs can be raw string features during `fit()`, `predict()`,
    and `evaluate()`. Inputs will be tokenized and dynamically masked during
    training and evaluation. This is done by default when creating the model
    with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq).

    Args:
        backbone: A `keras_nlp.models.RobertaBackbone` instance.
        preprocessor: A `keras_nlp.models.RobertaMaskedLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Example usage:

    Raw string inputs and pretrained backbone.
    ```python
    # Create a dataset with raw string features. Labels are inferred.
    features = ["The quick brown fox jumped.", "I forgot my homework."]

    # Create a RobertaMaskedLM with a pretrained backbone and further train
    # on an MLM task.
    masked_lm = keras_nlp.models.RobertaMaskedLM.from_preset(
        "roberta_base_en",
    )
    masked_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    masked_lm.fit(x=features, batch_size=2)
    ```

    Preprocessed inputs and custom backbone.
    ```python
    # Create a preprocessed dataset where 0 is the mask token.
    preprocessed_features = {
        "token_ids": tf.constant(
            [[1, 2, 0, 4, 0, 6, 7, 8]] * 2, shape=(2, 8)
        ),
        "padding_mask": tf.constant(
            [[1, 1, 1, 1, 1, 1, 1, 1]] * 2, shape=(2, 8)
        ),
        "mask_positions": tf.constant([[2, 4]] * 2, shape=(2, 2))
    }
    # Labels are the original masked values.
    labels = [[3, 5]] * 2

    # Randomly initialize a RoBERTa encoder
    backbone = keras_nlp.models.RobertaBackbone(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )
    # Create a RoBERTa masked_lm and fit the data.
    masked_lm = keras_nlp.models.RobertaMaskedLM(
        backbone,
        preprocessor=None,
    )
    masked_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    masked_lm.fit(x=preprocessed_features, y=labels, batch_size=2)
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
            kernel_initializer=roberta_kernel_initializer(),
            name="mlm_head",
        )(backbone_outputs, inputs["mask_positions"])

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

    @classproperty
    def backbone_cls(cls):
        return RobertaBackbone

    @classproperty
    def preprocessor_cls(cls):
        return RobertaMaskedLMPreprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
