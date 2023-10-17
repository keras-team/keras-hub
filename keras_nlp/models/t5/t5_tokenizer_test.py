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

import io

import sentencepiece
import tensorflow as tf

from keras_nlp.models.t5.t5_tokenizer import T5Tokenizer
from keras_nlp.tests.test_case import TestCase


class T5TokenizerTest(TestCase):
    def setUp(self):
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the earth is round"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=11,
            model_type="WORD",
            bos_id=-1,
            pad_id=0,
            eos_id=1,
            unk_id=2,
            pad_piece="<pad>",
            eos_piece="</s>",
            unk_piece="<unk>",
            user_defined_symbols="[MASK]",
        )
        self.init_kwargs = {"proto": bytes_io.getvalue()}
        self.input_data = ["the quick brown fox.", "the earth is round."]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=T5Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[4, 9, 5, 2], [4, 6, 8, 2]],
        )

    def test_errors_missing_special_tokens(self):
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=iter(["abc"]),
            model_writer=bytes_io,
            vocab_size=5,
            pad_id=-1,
            eos_id=-1,
            bos_id=-1,
        )
        with self.assertRaises(ValueError):
            T5Tokenizer(proto=bytes_io.getvalue())
