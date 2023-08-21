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

import os

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling import f_net_encoder
from keras_nlp.tests.test_case import TestCase


class FNetEncoderTest(TestCase):
    def test_valid_call(self):
        encoder = f_net_encoder.FNetEncoder(intermediate_dim=4)
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                encoder,
            ]
        )
        input = ops.random.uniform(shape=[2, 4, 6])
        model(input)

    def test_get_config_and_from_config(self):
        encoder = f_net_encoder.FNetEncoder(
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

        restored_encoder = f_net_encoder.FNetEncoder.from_config(
            config,
        )
        self.assertEqual(
            restored_encoder.get_config(), {**config, **expected_config_subset}
        )

    def test_value_error_when_invalid_kernel_initializer(self):
        with self.assertRaises(ValueError):
            f_net_encoder.FNetEncoder(
                intermediate_dim=4,
                dropout=0.5,
                kernel_initializer="Invalid",
            )

    def test_one_training_step_of_f_net_encoder(self):
        encoder = f_net_encoder.FNetEncoder(intermediate_dim=4)
        inputs = keras.Input(shape=(4, 6))
        x = encoder(inputs)
        x = keras.layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=x)

        data = ops.random.uniform(shape=[2, 4, 6])
        label = ops.random.randint(minval=0, maxval=2, shape=(2, 4, 1))

        loss = keras.losses.BinaryCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam()
        model.compile(loss=loss, optimizer=optimizer)
        loss = model.train_on_batch(x=data, y=label)
        self.assertGreater(loss, 0)

    def test_saved_model(self):
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                f_net_encoder.FNetEncoder(
                    intermediate_dim=4,
                ),
            ]
        )
        data = ops.random.uniform(shape=[2, 4, 6])
        model(data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        loaded_model = keras.models.load_model(path)

        model_output = model(data)
        loaded_model_output = loaded_model(data)
        self.assertAllClose(model_output, loaded_model_output)
