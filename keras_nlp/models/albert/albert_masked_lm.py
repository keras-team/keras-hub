# Copyright 2023 The KerasNLP Authors
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

"""ALBERT masked LM model."""

import copy

from tensorflow import keras

from keras_nlp.layers.masked_lm_head import MaskedLMHead
from keras_nlp.models.albert.albert_backbone import AlbertBackbone
from keras_nlp.models.albert.albert_backbone import albert_kernel_initializer
from keras_nlp.models.albert.albert_masked_lm_preprocessor import (
    AlbertMaskedLMPreprocessor,
)
from keras_nlp.models.albert.albert_presets import backbone_presets
from keras_nlp.models.task import Task
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class AlbertMaskedLM(Task):
    """An end-to-end ALBERT model for the masked language modeling task.

    This model will train ALBERT on a masked language modeling task.
    The model will predict labels for a number of masked tokens in the
    input data. For usage of this model with pre-trained weights, see the
    `from_preset()` method.

    This model can optionally be configured with a `preprocessor` layer, in
    which case inputs can be raw string features during `fit()`, `predict()`,
    and `evaluate()`. Inputs will be tokenized and dynamically masked during
    training and evaluation. This is done by default when creating the model
    with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_nlp.models.AlbertBackbone` instance.
        preprocessor: A `keras_nlp.models.AlbertMaskedLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Example usage:

    Raw string inputs and pretrained backbone.
    ```python
    # Create a dataset with raw string features. Labels are inferred.
    features = ["The quick brown fox jumped.", "I forgot my homework."]

    # Create a AlbertMaskedLM with a pretrained backbone and further train
    # on an MLM task.
    masked_lm = keras_nlp.models.AlbertMaskedLM.from_preset(
        "albert_base_en_uncased",
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
        "segment_ids": tf.constant(
            [[1, 0, 0, 4, 0, 6, 7, 8]] * 2, shape=(2, 8)
        ),
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

    # Randomly initialize a ALBERT encoder
    backbone = keras_nlp.models.AlbertBackbone(
        vocabulary_size=1000,
        num_layers=2,
        num_heads=2,
        embedding_dim=64,
        hidden_dim=64,
        intermediate_dim=128,
        max_sequence_length=128)

    # Create a ALBERT masked LM and fit the data.
    masked_lm = keras_nlp.models.AlbertMaskedLM(
        backbone,
        preprocessor=None,
    )
    masked_lm.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        jit_compile=True
    )
    masked_lm.fit(x=preprocessed_features, y=labels, batch_size=2)
    ```
    """

    def __init__(self, backbone, preprocessor=None, **kwargs):
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
            intermediate_activation=lambda x: keras.activations.gelu(
                x, approximate=True
            ),
            kernel_initializer=albert_kernel_initializer(),
            name="mlm_head",
        )(backbone_outputs["sequence_output"], inputs["mask_positions"])

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs
        )

        self.backbone = backbone
        self.preprocessor = preprocessor

    @classproperty
    def backbone_cls(cls):
        return AlbertBackbone

    @classproperty
    def preprocessor_cls(cls):
        return AlbertMaskedLMPreprocessor

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
