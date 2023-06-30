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
"""Tests for Transformer Decoder."""

import os

from absl.testing import parameterized

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling import transformer_decoder
from keras_nlp.tests.test_case import TestCase


class TransformerDecoderTest(TestCase):
    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_valid_call(self, normalize_first):
        encoder_input = keras.Input(shape=[4, 6])
        decoder_input = keras.Input(shape=[4, 6])
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
            normalize_first=normalize_first,
        )
        output = decoder(decoder_input, encoder_input)
        model = keras.Model(
            inputs=[decoder_input, encoder_input],
            outputs=output,
        )
        encoder_sequence = ops.random.uniform(shape=[2, 4, 6])
        decoder_sequence = ops.random.uniform(shape=[2, 4, 6])
        model([decoder_sequence, encoder_sequence])

    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_valid_call_without_cross_attention(self, normalize_first):
        decoder_input = keras.Input(shape=[4, 6])
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
            normalize_first=normalize_first,
        )
        output = decoder(decoder_input)
        model = keras.Model(
            inputs=decoder_input,
            outputs=output,
        )
        decoder_sequence = ops.random.uniform(shape=[2, 4, 6])
        model(decoder_sequence)

    def test_invalid_calls(self):
        encoder_input = ops.zeros((2, 4, 6))
        decoder_input = ops.zeros((2, 4, 6))

        # with cross-attention.
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder(decoder_input, encoder_input)
        # should raise ValueError if encoder_input is not provided
        with self.assertRaises(ValueError):
            decoder(decoder_input)

        # without cross-attention.
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder(decoder_input)
        # should raise ValueError if encoder_input is provided
        with self.assertRaises(ValueError):
            decoder(decoder_input, encoder_input)

    def test_get_config_and_from_config(self):
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
            kernel_initializer="HeNormal",
            bias_initializer="Zeros",
            normalize_first=True,
        )

        config = decoder.get_config()
        expected_config_subset = {
            "intermediate_dim": 4,
            "num_heads": 2,
            "dropout": 0,
            "activation": "relu",
            "layer_norm_epsilon": 1e-05,
            "kernel_initializer": keras.initializers.serialize(
                keras.initializers.HeNormal()
            ),
            "bias_initializer": keras.initializers.serialize(
                keras.initializers.Zeros()
            ),
            "normalize_first": True,
        }
        self.assertEqual(config, {**config, **expected_config_subset})
        self.assertEqual(config, {**config, **expected_config_subset})
        restored_decoder = transformer_decoder.TransformerDecoder.from_config(
            config,
        )
        self.assertEqual(
            restored_decoder.get_config(), {**config, **expected_config_subset}
        )

    def test_value_error_when_invalid_kernel_inititalizer(self):
        with self.assertRaises(ValueError):
            transformer_decoder.TransformerDecoder(
                intermediate_dim=4,
                num_heads=2,
                dropout=0.5,
                kernel_initializer="Invalid",
            )

    def test_one_training_step_of_transformer_with_cross_attention(self):
        decoder_input = keras.Input(shape=(4, 6))
        encoder_input = keras.Input(shape=(4, 6))
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4, num_heads=2
        )
        outputs = decoder(decoder_input, encoder_input)
        outputs = keras.layers.Dense(10, activation="softmax")(outputs)
        model = keras.Model((decoder_input, encoder_input), outputs)

        decoder_sequence = ops.random.uniform(shape=(2, 4, 6))
        encoder_sequence = ops.random.uniform(shape=(2, 4, 6))
        label = ops.random.randint(minval=0, maxval=10, shape=(2, 4, 1))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam()
        model.compile(loss=loss, optimizer=optimizer)
        loss = model.train_on_batch(
            x=(decoder_sequence, encoder_sequence), y=label
        )
        self.assertGreater(loss, 0)

    def test_one_training_step_of_transformer_without_cross_attention(self):
        decoder_input = keras.Input(shape=(4, 6))
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4, num_heads=2
        )
        outputs = decoder(decoder_input)
        outputs = keras.layers.Dense(10, activation="softmax")(outputs)
        model = keras.Model(decoder_input, outputs)

        decoder_sequence = ops.random.uniform(shape=(2, 4, 6))
        label = ops.random.randint(minval=0, maxval=10, shape=(2, 4, 1))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam()
        model.compile(loss=loss, optimizer=optimizer)
        loss = model.train_on_batch(x=decoder_sequence, y=label)
        self.assertGreater(loss, 0)

    def test_mask_propagation(self):
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder_sequence = ops.random.uniform(shape=[1, 4, 6])
        encoder_sequence = ops.random.uniform(shape=[1, 4, 6])
        mask = ops.array([[True, True, False, False]])
        decoder_sequence._keras_mask = mask
        outputs = decoder(decoder_sequence, encoder_sequence)
        self.assertAllEqual(outputs._keras_mask, mask)

    def test_mask_propagation_without_cross_attention(self):
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder_sequence = ops.random.uniform(shape=[1, 4, 6])
        mask = ops.array([[True, True, False, False]])
        decoder_sequence._keras_mask = mask
        outputs = decoder(decoder_sequence)
        self.assertAllEqual(outputs._keras_mask, mask)

    def test_cache_call_is_correct(self):
        batch_size = 2
        seq_len = 5
        num_heads = 2
        key_dim = 4
        hidden_dim = num_heads * key_dim

        input_shape = (batch_size, seq_len, hidden_dim)
        x = ops.random.uniform(shape=input_shape)
        input_cache = ops.zeros((batch_size, 2, seq_len, num_heads, key_dim))
        outputs = ops.zeros_like(x)

        layer = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=num_heads,
        )
        no_loop_outputs, no_loop_cache = layer(
            x,
            self_attention_cache=input_cache,
            self_attention_cache_update_index=0,
        )

        def loop_body(i, outputs, cache):
            # Compute the rest tokens.
            next_input = ops.slice(x, (0, i, 0), (batch_size, 1, hidden_dim))
            next_output, cache = layer(
                decoder_sequence=next_input,
                self_attention_cache=cache,
                self_attention_cache_update_index=i,
            )
            outputs = ops.slice_update(outputs, [0, i, 0], next_output)
            return i + 1, outputs, cache

        def call(outputs, cache):
            _, outputs, cache = ops.while_loop(
                cond=lambda i, outputs, cache: i < seq_len,
                body=loop_body,
                loop_vars=[0, outputs, cache],
            )
            return outputs, cache

        output, output_cache = call(outputs, input_cache)

        self.assertAllClose(output, no_loop_outputs)
        self.assertAllClose(output_cache, no_loop_cache)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        encoder_input = keras.Input(shape=[4, 6])
        decoder_input = keras.Input(shape=[4, 6])
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
            normalize_first=True,
        )
        output = decoder(encoder_input, decoder_input)
        model = keras.Model(
            inputs=[decoder_input, encoder_input],
            outputs=output,
        )
        encoder_sequence = ops.random.uniform(shape=[2, 4, 6])
        decoder_sequence = ops.random.uniform(shape=[2, 4, 6])
        model([decoder_sequence, encoder_sequence])
        path = os.path.join(self.get_temp_dir(), filename)
        # Don't save traces in the tf format, we check compilation elsewhere.
        kwargs = {"save_traces": False} if save_format == "tf" else {}
        model.save(path, save_format=save_format, **kwargs)

        loaded_model = keras.models.load_model(path)
        model_output = model([decoder_sequence, encoder_sequence])
        loaded_model_output = loaded_model([decoder_sequence, encoder_sequence])
        self.assertAllClose(model_output, loaded_model_output)

    def test_saved_model_without_cross_attention(self):
        decoder_input = keras.Input(shape=[4, 6])
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
            normalize_first=True,
        )
        output = decoder(decoder_input)
        model = keras.Model(
            inputs=decoder_input,
            outputs=output,
        )
        decoder_sequence = ops.random.uniform(shape=[2, 4, 6])
        model(decoder_sequence)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        loaded_model = keras.models.load_model(path)

        model_output = model(decoder_sequence)
        loaded_model_output = loaded_model(decoder_sequence)
        self.assertAllClose(model_output, loaded_model_output)
