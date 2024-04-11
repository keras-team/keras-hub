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

from keras_nlp.models.gpt_neo_x.gpt_neo_x_tokenizer import GPTNeoXTokenizer
from keras_nlp.tests.test_case import TestCase


class GPTNeoXTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|endoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
            "special_tokens_in_strings": True,
        }
        self.input_data = [
            " airplane at airport<|endoftext|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=GPTNeoXTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[2, 3, 4, 2, 5, 6], [2, 3, 2, 5]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            GPTNeoXTokenizer(vocabulary=["a", "b", "c"], merges=[])
