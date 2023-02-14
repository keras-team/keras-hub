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
"""Test for GPT-2 backbone models."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras_nlp.models.gpt2.gpt2_backbone import GPT2Backbone


class GPT2Test(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        self.model = GPT2Backbone(
            vocabulary_size=1000,
            num_layers=2,
            num_heads=2,
            hidden_dim=64,
            intermediate_dim=128,
            max_sequence_length=128,
        )
        self.batch_size = 8
        self.input_batch = {
            "token_ids": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, self.model.max_sequence_length), dtype="int32"
            ),
        }

        self.input_dataset = tf.data.Dataset.from_tensor_slices(
            self.input_batch
        ).batch(2)

    def test_valid_call_gpt2(self):
        self.model(self.input_batch)

        # Check default name passed through
        self.assertRegexpMatches(self.model.name, "gpt2_backbone")

    def test_variable_sequence_length_call_gpt2(self):
        for seq_length in (25, 50, 75):
            input_data = {
                "token_ids": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
                "padding_mask": tf.ones(
                    (self.batch_size, seq_length), dtype="int32"
                ),
            }
            self.model(input_data)

    def test_cache_call_is_correct(self):
        initial_seq_len = 64
        max_seq_len = 80
        initial_inputs = {
            "token_ids": tf.ones(
                (self.batch_size, initial_seq_len), dtype="int32"
            ),
            "padding_mask": tf.ones(
                (self.batch_size, initial_seq_len), dtype="int32"
            ),
        }
        outputs, cache = self.model.build_initial_cache(
            initial_inputs,
            max_seq_len,
        )

        def foo(i, cache, outputs):
            def loop_body(i, cache, outputs):
                # Compute the rest tokens.
                output, cache = self.model.call_with_cache(
                    inputs=self.input_batch,
                    cache=cache,
                    current_index=i,
                )
                outputs = dynamic_update_slice(outputs, output, [0, i, 0])
                return i + 1, cache, outputs

            i, cache, cached_outputs = tf.while_loop(
                cond=lambda i, cache, outputs: i < max_seq_len,
                body=loop_body,
                loop_vars=[i, cache, outputs],
            )
            return cached_outputs

        cached_outputs = foo(initial_seq_len, cache, outputs)
        graph_foo = tf.function(foo)
        graph_cached_outputs = graph_foo(initial_seq_len, cache, outputs)
        normal_outputs = self.model(self.input_batch)
        normal_outputs = normal_outputs[:, :max_seq_len, :]

        self.assertAllClose(cached_outputs, normal_outputs)
        self.assertAllClose(graph_cached_outputs, normal_outputs)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_gpt2_compile(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_batch)

    @parameterized.named_parameters(
        ("jit_compile_false", False), ("jit_compile_true", True)
    )
    def test_gpt2_compile_batched_ds(self, jit_compile):
        self.model.compile(jit_compile=jit_compile)
        self.model.predict(self.input_dataset)

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
        self.assertIsInstance(restored_model, GPT2Backbone)

        # Check that output matches.
        restored_output = restored_model(self.input_batch)
        self.assertAllClose(model_output, restored_output)
