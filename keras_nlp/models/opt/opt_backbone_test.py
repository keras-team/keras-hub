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
"""Test for OPT backbone models."""

import os

import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.models.opt.opt_backbone import OPTBackbone


class OPTTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        # For DTensor.
        keras.backend.experimental.enable_tf_random_generator()
        keras.utils.set_random_seed(1337)

        self.backbone = OPTBackbone(
            vocabulary_size=10,
            num_layers=2,
            num_heads=2,
            hidden_dim=2,
            intermediate_dim=4,
            max_sequence_length=5,
        )
        self.input_batch = {
            "token_ids": tf.ones((2, 5), dtype="int32"),
            "padding_mask": tf.ones((2, 5), dtype="int32"),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_opt(self):
        self.backbone(self.input_batch)

    def test_token_embedding(self):
        output = self.backbone.token_embedding(self.input_batch["token_ids"])
        self.assertEqual(output.shape, (2, 5, 2))

    def test_name(self):
        # Check default name passed through
        self.assertRegexpMatches(self.backbone.name, "opt_backbone")

    def test_variable_sequence_length_call_opt(self):
        for seq_length in (2, 3, 4):
            input_data = {
                "token_ids": tf.ones((2, seq_length), dtype="int32"),
                "padding_mask": tf.ones((2, seq_length), dtype="int32"),
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
        self.assertIsInstance(restored_model, OPTBackbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)

    def test_create_layout_map(self):
        mesh = tf.experimental.dtensor.create_mesh([("batch", 1), ("model", 1)])
        with OPTBackbone.create_layout_map(mesh).scope():
            OPTBackbone(
                vocabulary_size=10,
                num_layers=2,
                num_heads=2,
                hidden_dim=2,
                intermediate_dim=4,
                max_sequence_length=5,
            )
        # Using DTensor enables the mlir bridge as a side effect. Eventually
        # this will be default, but for now we have compile errors with the
        # bridge elsewhere and must disable. See
        # https://github.com/keras-team/keras-nlp/issues/1001
        tf.config.experimental.disable_mlir_bridge()


@pytest.mark.tpu
@pytest.mark.usefixtures("tpu_test_class")
class OPTBackboneTPUTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        with self.tpu_strategy.scope():
            self.backbone = OPTBackbone(
                vocabulary_size=1000,
                num_layers=2,
                num_heads=2,
                hidden_dim=32,
                intermediate_dim=128,
                max_sequence_length=128,
            )
        self.input_batch = {
            "token_ids": tf.ones((8, 128), dtype="int32"),
            "padding_mask": tf.ones((8, 128), dtype="int32"),
        }
        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_predict(self):
        self.backbone.compile()
        self.backbone.predict(self.input_dataset)
