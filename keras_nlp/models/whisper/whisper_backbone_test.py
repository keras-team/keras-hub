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
"""Test for Whisper backbone models."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.whisper.whisper_backbone import NUM_MELS
from keras_nlp.models.whisper.whisper_backbone import WhisperBackbone


class WhisperBackboneTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.model = WhisperBackbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_encoder_sequence_length=128,
            max_decoder_sequence_length=96,
        )
        self.batch_size = 8
        self.input_batch = {
            "encoder_features": tf.ones(
                (
                    self.batch_size,
                    self.model.max_encoder_sequence_length,
                    NUM_MELS,
                ),
                dtype="int32",
            ),
            "decoder_token_ids": tf.ones(
                (self.batch_size, self.model.max_decoder_sequence_length),
                dtype="int32",
            ),
            "decoder_padding_mask": tf.ones(
                (self.batch_size, self.model.max_decoder_sequence_length),
                dtype="int32",
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_whisper(self):
        self.model(self.input_batch)

        # Check default name passed through
        self.assertRegexpMatches(self.model.name, "whisper_backbone")

    def test_variable_sequence_length_call_whisper(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "encoder_features": tf.ones(
                    (self.batch_size, seq_length, NUM_MELS),
                    dtype="int32",
                ),
                "decoder_token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "decoder_padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            self.model(input_data)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compile(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_compile_batched_ds(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_dataset)

    def test_key_projection_bias_absence(self):
        # Check only for the first encoder layer and first decoder layer.
        self.assertIsNone(
            self.model.get_layer(
                "transformer_encoder_layer_0"
            )._self_attention_layer._key_dense.bias
        )
        self.assertIsNone(
            self.model.get_layer(
                "transformer_decoder_layer_0"
            )._self_attention_layer._key_dense.bias
        )
        self.assertIsNone(
            self.model.get_layer(
                "transformer_decoder_layer_0"
            )._cross_attention_layer._key_dense.bias
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model_output = self.model(self.input_batch)
        save_path = os.path.join(self.get_temp_dir(), filename)
        self.model.save(save_path, save_format=save_format)
        restored_model = keras.models.load_model(save_path)

        # Check we got the real object back.
        self.assertIsInstance(restored_model, WhisperBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(
            model_output["encoder_sequence_output"],
            restored_output["encoder_sequence_output"],
        )
        self.assertAllClose(
            model_output["decoder_sequence_output"],
            restored_output["decoder_sequence_output"],
        )


@pytest.mark.tpu
@pytest.mark.usefixtures("tpu_test_class")
class WhisperBackboneTPUTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        with self.tpu_strategy.scope():
            self.model = WhisperBackbone(
                vocabulary_size=1000,
                num_layers=2,
                num_heads=2,
                hidden_dim=64,
                intermediate_dim=128,
                max_encoder_sequence_length=128,
                max_decoder_sequence_length=64,
            )

        self.input_batch = {
            "encoder_features": tf.ones(
                (
                    8,
                    self.model.max_encoder_sequence_length,
                    NUM_MELS,
                ),
                dtype="int32",
            ),
            "decoder_token_ids": tf.ones(
                (8, self.model.max_decoder_sequence_length), dtype="int32"
            ),
            "decoder_padding_mask": tf.ones(
                (8, self.model.max_decoder_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_predict(self):
        self.model.compile()
        self.model.predict(self.input_dataset)
