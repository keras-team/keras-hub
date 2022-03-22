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
"""Tests for Transformer Decoder."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import transformer_decoder


class TransformerDecoderTest(tf.test.TestCase):
    def test_valid_call(self):
        encoder_input = keras.Input(shape=[4, 6])
        decoder_input = keras.Input(shape=[4, 6])
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        output = decoder(encoder_input, decoder_input)
        model = keras.Model(
            inputs=[decoder_input, encoder_input],
            outputs=output,
        )

        encoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        decoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        model([encoder_sequence, decoder_sequence])

    def test_valid_call_with_mask(self):
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        encoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        decoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        encoder_padding_mask = encoder_sequence[:, :, 0] > 0.5
        decoder_padding_mask = decoder_sequence[:, :, 0] > 0.5
        decoder(
            decoder_sequence,
            encoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            encoder_padding_mask=encoder_padding_mask,
            use_causal_mask=True,
        )

    def test_get_config_and_from_config(self):
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
            kernel_initializer="HeNormal",
            bias_initializer="Zeros",
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

    def test_one_training_step_of_transformer_encoder(self):
        class MyModel(keras.Model):
            def __init__(self):
                super(MyModel, self).__init__()
                self._decoder = transformer_decoder.TransformerDecoder(
                    intermediate_dim=4, num_heads=2
                )
                self._dense = keras.layers.Dense(1, activation="sigmoid")

            def call(self, decoder_input, encoder_output):
                x = self._decoder(decoder_input, encoder_output)
                return self._dense(x)

        model = MyModel()

        decoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        encoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        label = tf.cast(decoder_sequence[:, :, 0] >= 0.5, dtype=tf.int32)

        loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            pred = model(decoder_sequence, encoder_sequence)
            loss = loss_fn(label, pred)
        grad = tape.gradient(loss, model.trainable_variables)
        self.assertGreater(len(grad), 1)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    def test_checkpointing_transformer_decoder(self):
        decoder1 = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )

        decoder2 = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        decoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        encoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        decoder1(decoder_sequence, encoder_sequence)
        decoder2(decoder_sequence, encoder_sequence)
        # The weights of decoder1 and decoder2 are different.
        self.assertFalse(
            all(
                decoder1._output_dense.trainable_variables[0][0]
                == decoder2._output_dense.trainable_variables[0][0]
            )
        )
        checkpoint = tf.train.Checkpoint(decoder1)
        checkpoint2 = tf.train.Checkpoint(decoder2)
        temp_dir = self.get_temp_dir()
        save_path = checkpoint.save(temp_dir)
        checkpoint2.restore(save_path)

        decoder1_output = decoder1(decoder_sequence, encoder_sequence)
        decoder2_output = decoder2(
            decoder_sequence,
            encoder_sequence,
        )
        self.assertAllClose(decoder1_output, decoder2_output)

    def test_save_model(self):
        encoder_input = keras.Input(shape=[4, 6])
        decoder_input = keras.Input(shape=[4, 6])
        decoder = transformer_decoder.TransformerDecoder(
            intermediate_dim=4,
            num_heads=2,
        )
        output = decoder(encoder_input, decoder_input)
        model = keras.Model(
            inputs=[decoder_input, encoder_input],
            outputs=output,
        )
        encoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        decoder_sequence = tf.random.uniform(shape=[2, 4, 6])
        model([encoder_sequence, decoder_sequence])
        path = os.path.join(self.get_temp_dir(), "model")
        model.save(path)
        loaded_model = keras.models.load_model(path)

        model_output = model([decoder_sequence, encoder_sequence])
        loaded_model_output = loaded_model([decoder_sequence, encoder_sequence])
        self.assertAllClose(model_output, loaded_model_output)
