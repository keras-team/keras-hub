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
        self.backbone = WhisperBackbone(
            vocabulary_size=10,
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_encoder_sequence_length=6,
            max_decoder_sequence_length=6,
        )
        self.input_batch = {
            "encoder_features": tf.ones((2, 5, NUM_MELS), dtype="int32"),
            "decoder_token_ids": tf.ones((2, 5), dtype="int32"),
            "decoder_padding_mask": tf.ones((2, 5), dtype="int32"),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_whisper(self):
        self.backbone(self.input_batch)

    def test_token_embedding(self):
        output = self.backbone.token_embedding(
            self.input_batch["decoder_token_ids"]
        )
        self.assertEqual(output.shape, (2, 5, 2))

    def test_name(self):
        # Check default name passed through
        self.assertRegexpMatches(self.backbone.name, "whisper_backbone")

    def test_variable_sequence_length_call_whisper(self):
        for seq_length in (2, 3, 4):
            input_data = {
                "encoder_features": tf.ones(
                    (2, seq_length, NUM_MELS), dtype="int32"
                ),
                "decoder_token_ids": tf.ones((2, seq_length), dtype="int32"),
                "decoder_padding_mask": tf.ones((2, seq_length), dtype="int32"),
            }
            self.backbone(input_data)

    def test_predict(self):
        self.backbone.predict(self.input_batch)
        self.backbone.predict(self.input_dataset)

    def test_serialization(self):
        new_backbone = keras.utils.deserialize_keras_object(
            keras.utils.serialize_keras_object(self.backbone)
        )
        self.assertEqual(new_backbone.get_config(), self.backbone.get_config())

    def test_key_projection_bias_absence(self):
        # Check only for the first encoder layer and first decoder layer.
        self.assertIsNone(
            self.backbone.get_layer(
                "transformer_encoder_layer_0"
            )._self_attention_layer._key_dense.bias
        )
        self.assertIsNone(
            self.backbone.get_layer(
                "transformer_decoder_layer_0"
            )._self_attention_layer._key_dense.bias
        )
        self.assertIsNone(
            self.backbone.get_layer(
                "transformer_decoder_layer_0"
            )._cross_attention_layer._key_dense.bias
        )

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    @pytest.mark.large  # Saving is slow, so mark these large.
    def test_saved_model(self, save_format, filename):
        model_output = self.backbone(self.input_batch)
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        self.backbone.save(path, save_format=save_format, **kwargs)
        restored_model = keras.models.load_model(path)

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
            self.backbone = WhisperBackbone(
                vocabulary_size=10,
                num_layers=2,
                num_heads=2,
                hidden_dim=2,
                intermediate_dim=4,
                max_encoder_sequence_length=6,
                max_decoder_sequence_length=6,
            )

        self.input_batch = {
            "encoder_features": tf.ones(
                (
                    8,
                    self.backbone.max_encoder_sequence_length,
                    NUM_MELS,
                ),
                dtype="int32",
            ),
            "decoder_token_ids": tf.ones(
                (8, self.backbone.max_decoder_sequence_length), dtype="int32"
            ),
            "decoder_padding_mask": tf.ones(
                (8, self.backbone.max_decoder_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_predict(self):
        self.backbone.compile()
        self.backbone.predict(self.input_dataset)
