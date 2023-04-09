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
"""Tests for XLM-RoBERTa masked language model."""

import io
import os

import pytest
import sentencepiece
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.xlm_roberta.xlm_roberta_backbone import XLMRobertaBackbone
from keras_nlp.models.xlm_roberta.xlm_roberta_masked_lm import (
    XLMRobertaMaskedLM,
)
from keras_nlp.models.xlm_roberta.xlm_roberta_masked_lm_preprocessor import (
    XLMRobertaMaskedLMPreprocessor,
)
from keras_nlp.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)


class XLMRobertaMaskedLMTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        bytes_io = io.BytesIO()
        vocab_data = tf.data.Dataset.from_tensor_slices(
            ["the quick brown fox", "the slow brown fox"]
        )
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=vocab_data.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=5,
            model_type="WORD",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            user_defined_symbols="[MASK]",
        )
        self.proto = bytes_io.getvalue()

        self.preprocessor = XLMRobertaMaskedLMPreprocessor(
            XLMRobertaTokenizer(proto=self.proto),
            sequence_length=5,
            mask_selection_length=2,
        )

        self.backbone = XLMRobertaBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=self.preprocessor.packer.sequence_length,
        )

        self.masked_lm = XLMRobertaMaskedLM(
            self.backbone,
            preprocessor=self.preprocessor,
        )

        self.raw_batch = tf.constant(
            ["the quick brown fox", "the slow brown fox"]
        )
        self.preprocessed_batch = self.preprocessor(self.raw_batch)[0]
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            self.raw_batch
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_masked_lm(self):
        self.masked_lm(self.preprocessed_batch)

    def test_classifier_predict(self):
        self.masked_lm.predict(self.raw_batch)
        self.masked_lm.preprocessor = None
        self.masked_lm.predict(self.preprocessed_batch)

    def test_classifier_fit(self):
        self.masked_lm.fit(self.raw_dataset)
        self.masked_lm.preprocessor = None
        self.masked_lm.fit(self.preprocessed_dataset)

    def test_classifier_fit_no_xla(self):
        self.masked_lm.preprocessor = None
        self.masked_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            jit_compile=False,
        )
        self.masked_lm.fit(self.preprocessed_dataset)

    def test_serialization(self):
        config = keras.utils.serialize_keras_object(self.masked_lm)
        new_classifier = keras.utils.deserialize_keras_object(config)
        self.assertEqual(
            new_classifier.get_config(),
            self.masked_lm.get_config(),
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large
    def test_saved_model(self, save_format, filename):
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.masked_lm.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, XLMRobertaMaskedLM)

        model_output = self.masked_lm(self.preprocessed_batch)
        restored_output = restored_model(self.preprocessed_batch)

        self.assertAllClose(model_output, restored_output)
