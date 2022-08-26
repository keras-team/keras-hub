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

"""Tests for loading pretrained word piece vocabularies."""

import tensorflow as tf
import keras_nlp
from keras_nlp.tokenizers.word_piece_tokenizer import SUPPORTED_VOCAB

class PretrainedWordPieceTokenizerTest(tf.test.TestCase):
    def test_download_vocabularies(self):
        for lang in SUPPORTED_VOCAB:
            print(lang)
            keras_nlp.tokenizers.WordPieceTokenizer(lang=lang, lowercase=False)
            keras_nlp.tokenizers.WordPieceTokenizer(lang=lang, lowercase=True)