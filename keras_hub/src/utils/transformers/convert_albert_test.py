# Copyright 2024 The KerasHub Authors
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
import pytest

from keras_hub.src.models.albert.albert_backbone import AlbertBackbone
from keras_hub.src.models.albert.albert_text_classifier import (
    AlbertTextClassifier,
)
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.text_classifier import TextClassifier
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = AlbertTextClassifier.from_preset(
            "hf://albert/albert-base-v2", num_classes=2
        )
        prompt = "That movies was terrible."
        model.predict([prompt])

    @pytest.mark.large
    def test_class_detection(self):
        model = TextClassifier.from_preset(
            "hf://albert/albert-base-v2",
            num_classes=2,
            load_weights=False,
        )
        self.assertIsInstance(model, AlbertTextClassifier)
        model = Backbone.from_preset(
            "hf://albert/albert-base-v2",
            load_weights=False,
        )
        self.assertIsInstance(model, AlbertBackbone)

    # TODO: compare numerics with huggingface model
