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
"""Albert masked lm model."""

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
            intermediate_activation="gelu",
            kernel_initializer=albert_kernel_initializer(),
            name="mlm_head",
        )(backbone_outputs, inputs["mask_positions"])

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
