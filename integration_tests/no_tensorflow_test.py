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

import unittest

import numpy as np

import keras_nlp


class NoTensorflow(unittest.TestCase):
    def test_backbone_works(self):
        backbone = keras_nlp.models.BertBackbone.from_preset(
            "bert_tiny_en_uncased",
        )
        backbone.predict(
            {
                "token_ids": np.ones((4, 128)),
                "padding_mask": np.ones((4, 128)),
                "segment_ids": np.ones((4, 128)),
            }
        )

    def test_tokenizer_errors(self):
        with self.assertRaises(Exception) as e:
            keras_nlp.models.BertTokenizer.from_preset(
                "bert_tiny_en_uncased",
            )
            self.assertTrue("pip install tensorflow-text" in e.exception)
