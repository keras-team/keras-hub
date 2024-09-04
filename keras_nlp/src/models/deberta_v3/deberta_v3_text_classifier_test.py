# Copyright 2024 The KerasNLP Authors
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

import os

import pytest

from keras_nlp.src.models.deberta_v3.deberta_v3_backbone import (
    DebertaV3Backbone,
)
from keras_nlp.src.models.deberta_v3.deberta_v3_text_classifier import (
    DebertaV3TextClassifier,
)
from keras_nlp.src.models.deberta_v3.deberta_v3_text_classifier_preprocessor import (
    DebertaV3TextClassifierPreprocessor,
)
from keras_nlp.src.models.deberta_v3.deberta_v3_tokenizer import (
    DebertaV3Tokenizer,
)
from keras_nlp.src.tests.test_case import TestCase


class DebertaV3TextClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.preprocessor = DebertaV3TextClassifierPreprocessor(
            DebertaV3Tokenizer(
                # Generated using create_deberta_v3_test_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "deberta_v3_test_vocab.spm"
                )
            ),
            sequence_length=5,
        )
        self.backbone = DebertaV3Backbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.sequence_length,
        )
        self.init_kwargs = {
            "preprocessor": self.preprocessor,
            "backbone": self.backbone,
            "num_classes": 2,
        }
        self.train_data = (
            ["the quick brown fox.", "the slow brown fox."],  # Features.
            [1, 0],  # Labels.
        )
        self.input_data = self.preprocessor(*self.train_data)[0]

    def test_classifier_basics(self):
        self.run_task_test(
            cls=DebertaV3TextClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DebertaV3TextClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DebertaV3TextClassifier.presets:
            self.run_preset_test(
                cls=DebertaV3TextClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.input_data,
                expected_output_shape=(2, 2),
            )
