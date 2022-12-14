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
"""Tests for FNet Encoder."""

import os

import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_nlp.layers import fnet_encoder


class FNetEncoderTest(tf.test.TestCase, parameterized.TestCase):
    def test_valid_call(self):
        encoder = fnet_encoder.FNetEncoder(intermediate_dim=4)
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                encoder,
            ]
        )
        input = tf.random.uniform(shape=[2, 4, 6])
        model(input)

    def test_get_config_and_from_config(self):
        encoder = fnet_encoder.FNetEncoder(
            intermediate_dim=4,
            kernel_initializer="HeNormal",
            bias_initializer="Zeros",
        )
        config = encoder.get_config()
        expected_config_subset = {
            "intermediate_dim": 4,
            "dropout": 0,
            "activation": "relu",
            "layer_norm_epsilon": 1e-5,
            "kernel_initializer": keras.initializers.serialize(
                keras.initializers.HeNormal()
            ),
            "bias_initializer": keras.initializers.serialize(
                keras.initializers.Zeros()
            ),
        }
        self.assertEqual(config, {**config, **expected_config_subset})

        restored_encoder = fnet_encoder.FNetEncoder.from_config(
            config,
        )
        self.assertEqual(
            restored_encoder.get_config(), {**config, **expected_config_subset}
        )

    def test_value_error_when_invalid_kernel_initializer(self):
        with self.assertRaises(ValueError):
            fnet_encoder.FNetEncoder(
                intermediate_dim=4,
                dropout=0.5,
                kernel_initializer="Invalid",
            )

    def test_one_training_step_of_fnet_encoder(self):
        encoder = fnet_encoder.FNetEncoder(intermediate_dim=4)
        inputs = keras.Input(shape=(4, 6))
        x = encoder(inputs)
        x = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=x)

        data = tf.random.uniform(shape=[2, 4, 6])
        label = tf.cast(data[:, :, 0] >= 0.5, dtype=tf.int32)

        loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            pred = model(data)
            loss = loss_fn(label, pred)
        grad = tape.gradient(loss, model.trainable_variables)
        self.assertGreater(len(grad), 1)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    def test_checkpointing_fnet_encoder(self):
        encoder1 = fnet_encoder.FNetEncoder(
            intermediate_dim=4,
        )

        encoder2 = fnet_encoder.FNetEncoder(
            intermediate_dim=4,
        )
        data = tf.random.uniform(shape=[2, 4, 6])
        encoder1(data)
        encoder2(data)
        # The weights of encoder1 and encoder2 are different.
        self.assertFalse(
            all(
                encoder1._output_dense.trainable_variables[0][0]
                == encoder2._output_dense.trainable_variables[0][0]
            )
        )
        checkpoint = tf.train.Checkpoint(encoder1)
        checkpoint2 = tf.train.Checkpoint(encoder2)
        temp_dir = self.get_temp_dir()
        save_path = checkpoint.save(temp_dir)
        checkpoint2.restore(save_path)

        encoder1_output = encoder1(data)
        encoder2_output = encoder2(data)
        self.assertAllClose(encoder1_output, encoder2_output)

    @parameterized.named_parameters(
        ("tf_format", "tf", "model"),
        ("keras_format", "keras_v3", "model.keras"),
    )
    def test_saved_model(self, save_format, filename):
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                fnet_encoder.FNetEncoder(
                    intermediate_dim=4,
                ),
            ]
        )
        data = tf.random.uniform(shape=[2, 4, 6])
        model(data)
        path = os.path.join(self.get_temp_dir(), filename)
        model.save(path, save_format=save_format)
        loaded_model = keras.models.load_model(path)

        model_output = model(data)
        loaded_model_output = loaded_model(data)
        self.assertAllClose(model_output, loaded_model_output)
