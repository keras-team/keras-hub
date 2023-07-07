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
"""Tests for BART causal LM model."""

import os
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.models.bart.bart_backbone import BartBackbone
from keras_nlp.models.bart.bart_seq_2_seq_lm import BartSeq2SeqLM
from keras_nlp.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_nlp.models.bart.bart_tokenizer import BartTokenizer
from keras_nlp.tests.test_case import TestCase


class BartSeq2SeqLMTest(TestCase):
    def setUp(self):
        # For DTensor.
        keras.backend.experimental.enable_tf_random_generator()
        keras.utils.set_random_seed(1337)

        self.vocab = {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "Ġair": 3,
            "plane": 4,
            "Ġat": 5,
            "port": 6,
            "Ġkoh": 7,
            "li": 8,
            "Ġis": 9,
            "Ġthe": 10,
            "Ġbest": 11,
            "<mask>": 12,
        }

        self.merges = ["Ġ a", "Ġ t", "Ġ k", "Ġ i", "Ġ b", "Ġa i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "o h", "l i", "Ġi s", "Ġb e"]
        self.merges += ["s t", "Ġt h", "Ġai r", "pl a", "Ġk oh", "Ġth e"]
        self.merges += ["Ġbe st", "po rt", "pla ne"]
        self.preprocessor = BartSeq2SeqLMPreprocessor(
            BartTokenizer(vocabulary=self.vocab, merges=self.merges),
            encoder_sequence_length=12,
            decoder_sequence_length=10,
        )
        self.backbone = BartBackbone(
            vocabulary_size=self.preprocessor.tokenizer.vocabulary_size(),
            num_layers=2,
            num_heads=2,
            hidden_dim=4,
            intermediate_dim=8,
            max_sequence_length=12,
        )
        self.seq_2_seq_lm = BartSeq2SeqLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )

        self.raw_batch = {
            "encoder_text": tf.constant(
                [" airplane at airport", " airplane at airport"]
            ),
            "decoder_text": tf.constant(
                [" kohli is the best", " kohli is the best"]
            ),
        }

        self.preprocessed_batch = self.preprocessor(self.raw_batch)[0]
        self.raw_dataset = tf.data.Dataset.from_tensor_slices(
            self.raw_batch
        ).batch(2)
        self.preprocessed_dataset = self.raw_dataset.map(self.preprocessor)

    def test_valid_call_seq_2_seq_lm(self):
        self.seq_2_seq_lm(self.preprocessed_batch)

    def test_predict(self):
        self.seq_2_seq_lm.predict(self.raw_batch)
        self.seq_2_seq_lm.preprocessor = None
        self.seq_2_seq_lm.predict(self.preprocessed_batch)

    def test_fit(self):
        self.seq_2_seq_lm.fit(self.raw_dataset)
        self.seq_2_seq_lm.preprocessor = None
        self.seq_2_seq_lm.fit(self.preprocessed_dataset)

    def test_fit_no_xla(self):
        self.seq_2_seq_lm.preprocessor = None
        self.seq_2_seq_lm.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            jit_compile=False,
        )
        self.seq_2_seq_lm.fit(self.preprocessed_dataset)

    def test_generate(self):
        # String input.
        inputs = {
            "encoder_text": " airplane at airport",
            "decoder_text": " kohli is the best",
        }
        output = self.seq_2_seq_lm.generate(inputs)
        self.assertTrue(" kohli is the best" in output)
        # String tensor input.
        self.assertIsInstance(
            self.seq_2_seq_lm.generate(self.raw_batch)[0], str
        )
        # String dataset input.
        self.assertIsInstance(
            self.seq_2_seq_lm.generate(self.raw_dataset)[0], str
        )

        # Int tensor input.
        self.seq_2_seq_lm.preprocessor = None
        preprocessed_batch = self.preprocessor.generate_preprocess(inputs)
        outputs = self.seq_2_seq_lm.generate(preprocessed_batch)
        # Assert prompt is in output in token id space.
        self.assertAllEqual(
            outputs["decoder_token_ids"][:, :5],
            preprocessed_batch["decoder_token_ids"][:, :5],
        )
        self.assertAllEqual(
            outputs["decoder_padding_mask"][:, :5],
            preprocessed_batch["decoder_padding_mask"][:, :5],
        )

    def test_generate_string_in_string_out(self):
        # String input.
        inputs = " airplane at airport"
        self.seq_2_seq_lm.generate(inputs)

        # String tensor input.
        self.assertIsInstance(
            self.seq_2_seq_lm.generate(
                [" airplane at airport", " airplane at airport"]
            )[0],
            str,
        )

        # String dataset input.
        raw_dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant([" airplane at airport", " airplane at airport"])
        ).batch(2)
        self.assertIsInstance(self.seq_2_seq_lm.generate(raw_dataset)[0], str)

    def test_early_stopping(self):
        call_decoder_with_cache = self.seq_2_seq_lm.call_decoder_with_cache

        def wrapper(*args, **kwargs):
            """Modify output logits to always favor end_token_id"""
            (
                logits,
                hidden_states,
                self_attention_cache,
                cross_attention_cache,
            ) = call_decoder_with_cache(*args, **kwargs)
            logits = np.zeros(logits.shape.as_list())
            logits[:, :, self.preprocessor.tokenizer.end_token_id] = 1.0e9
            return (
                logits,
                hidden_states,
                self_attention_cache,
                cross_attention_cache,
            )

        with patch.object(
            self.seq_2_seq_lm, "call_decoder_with_cache", wraps=wrapper
        ):
            inputs = {
                "encoder_text": [
                    " airplane at airport",
                    " airplane at airport",
                ],
                "decoder_text": [" kohli is the best", " kohli"],
            }
            output = self.seq_2_seq_lm.generate(inputs)

            # We should immediately abort and output the prompt.
            self.assertAllEqual(inputs["decoder_text"], output)
            self.assertEqual(
                self.seq_2_seq_lm.call_decoder_with_cache.call_count, 2
            )

    def test_beam_search(self):
        seq_2_seq_lm = BartSeq2SeqLM(
            backbone=self.backbone,
            preprocessor=self.preprocessor,
        )
        seq_2_seq_lm.compile(sampler="beam")
        seq_2_seq_lm.generate(self.raw_batch)

    def test_generate_compilation(self):
        # Assert we do not recompile with successive calls.
        self.seq_2_seq_lm.generate(self.raw_batch)
        first_fn = self.seq_2_seq_lm.generate_function
        self.seq_2_seq_lm.generate(self.raw_batch)
        second_fn = self.seq_2_seq_lm.generate_function
        self.assertEqual(first_fn, second_fn)
        # Assert we do recompile after compile is called.
        self.seq_2_seq_lm.compile(sampler="greedy")
        self.assertIsNone(self.seq_2_seq_lm.generate_function)

    def test_serialization(self):
        new_seq_2_seq_lm = keras.saving.deserialize_keras_object(
            keras.saving.serialize_keras_object(self.seq_2_seq_lm)
        )
        self.assertEqual(
            new_seq_2_seq_lm.get_config(), self.seq_2_seq_lm.get_config()
        )

    @pytest.mark.large
    def test_saved_model(self):
        keras.utils.set_random_seed(42)
        model_output = self.seq_2_seq_lm.predict(self.raw_batch)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        self.seq_2_seq_lm.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, BartSeq2SeqLM)

        # Check that output matches.
        keras.utils.set_random_seed(42)
        restored_output = restored_model.predict(self.raw_batch)
        self.assertAllClose(model_output, restored_output)
