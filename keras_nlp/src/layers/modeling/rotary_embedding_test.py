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
import keras
import numpy as np
from keras import ops
from keras import random

from keras_nlp.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_nlp.src.tests.test_case import TestCase


class RotaryEmbeddingTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=RotaryEmbedding,
            init_kwargs={
                "max_wavelength": 1000,
                "scaling_factor": 2.0,
                "sequence_axis": 1,
                "feature_axis": -1,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
        )

    def test_layer_behaviors_4d(self):
        self.run_layer_test(
            cls=RotaryEmbedding,
            init_kwargs={
                "max_wavelength": 1000,
            },
            input_data=random.uniform(shape=(2, 8, 4, 6)),
            expected_output_shape=(2, 8, 4, 6),
        )

    def test_dynamic_layer_output_shape(self):
        embedding_layer = RotaryEmbedding()
        hidden_size = 32
        inputs = keras.Input(shape=(None, hidden_size))
        outputs = embedding_layer(inputs)

        # When using dynamic positional encoding shapes, the output is expected
        # to be the same as the input shape in all dimensions but may be None.
        expected_output_shape = (None, None, hidden_size)
        self.assertEqual(expected_output_shape, outputs.shape)

    # do multi dimension before sequence length
    def test_multi_dimension_layer_output_shape(self):
        embedding_layer = RotaryEmbedding()
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(None, seq_length, hidden_size))
        outputs = embedding_layer(inputs)

        # When using multiple dimensions before sequence length, the output is
        # expected to be the same as the input shape in all dimensions.
        expected_output_shape = (None, None, seq_length, hidden_size)
        self.assertEqual(expected_output_shape, outputs.shape)

    def test_output_correct_values(self):
        embedding_layer = RotaryEmbedding()
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                embedding_layer,
            ]
        )
        input = ops.ones(shape=[1, 4, 6])
        output = model(input)

        # comapre position encoding values for position 0 and 3
        expected_0 = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
        expected_3 = [-1.1311, 0.8515, 0.9935, -0.8489, 1.1291, 1.0064]
        self.assertAllClose(output[0, 0, :], expected_0, atol=0.01, rtol=0.01)
        self.assertAllClose(output[0, 3, :], expected_3, atol=0.01, rtol=0.01)

    def test_start_index(self):
        batch_size, seq_length, feature_size = 2, 3, 4
        layer = RotaryEmbedding(seq_length)
        data = random.uniform(shape=(batch_size, seq_length, feature_size))
        full_output = layer(data)
        sequential_output = ops.zeros((batch_size, seq_length, feature_size))
        for i in range(seq_length):
            parial_output = layer(data[:, i : i + 1, :], start_index=i)
            sequential_output = ops.slice_update(
                sequential_output, (0, i, 0), parial_output
            )
        self.assertAllClose(full_output, sequential_output)

    def test_permuted_axes(self):
        batch_size, seq_length, feature_size = 2, 3, 4
        data = random.uniform(shape=(batch_size, seq_length, feature_size))
        layer = RotaryEmbedding(seq_length)
        outputs = layer(data)
        permuted_data = ops.transpose(data, (0, 2, 1))
        permuted_layer = RotaryEmbedding(
            seq_length, sequence_axis=-1, feature_axis=-2
        )
        permuted_outputs = permuted_layer(permuted_data)
        self.assertAllClose(outputs, ops.transpose(permuted_outputs, (0, 2, 1)))

    def test_float16_dtype(self):
        embedding_layer = RotaryEmbedding(dtype="float16")
        seq_length = 100
        hidden_size = 32
        inputs = keras.Input(shape=(seq_length, hidden_size))
        outputs = embedding_layer(inputs)

        # output dtype for this layer should be float16.
        self.assertEqual(outputs.dtype, "float16")

    def test_positions_array(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(size=(1, 2, 1, 16)).astype(np.float32)
        positions = ops.cast([0, 0], "float32")

        # Reference values computed using flax. Here's the code to generate
        # these numbers:
        # def _apply_flax_rope(
        #     inputs: jax.Array,    # [B, L]
        #     positions: jax.Array, # [B, L]
        #     head_dim: int,
        #     max_wavelength: int = 10_000.0,
        # ) -> jax.Array:
        #     """Applies RoPE."""
        #     fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
        #     timescale = max_wavelength**fraction

        #     sinusoid_inp = (
        #         positions[..., jnp.newaxis]
        #         / timescale[jnp.newaxis, jnp.newaxis, :]
        #     )
        #     sinusoid_inp = sinusoid_inp[..., jnp.newaxis, :]
        #     sin = jnp.sin(sinusoid_inp)
        #     cos = jnp.cos(sinusoid_inp)

        #     first_half, second_half = jnp.split(inputs, 2, axis=-1)
        #     first_part = first_half * cos - second_half * sin
        #     second_part = second_half * cos + first_half * sin
        #     out = jnp.concatenate([first_part, second_part], axis=-1)
        #     return out.astype(inputs.dtype)
        # fmt: off
        expected = np.array(
            [[[[0.12573022, -0.13210486, 0.64042264, 0.10490011,
                -0.5356694, 0.36159506, 1.304, 0.94708097,
                -0.70373523, -1.2654215, -0.62327445, 0.04132598,
                -2.3250308, -0.21879166, -1.245911, -0.7322674]],
              [[-0.544259, -0.31630015, 0.41163054, 1.0425134,
                -0.12853466, 1.3664634, -0.6651947, 0.35151008,
                0.90347016, 0.0940123, -0.7434993, -0.9217254,
                -0.45772582, 0.22019513, -1.0096182, -0.20917557]]]],
            dtype=np.float32
        )  # noqa
        # fmt: on

        layer = RotaryEmbedding()
        got = layer(x, positions=positions)

        np.testing.assert_allclose(expected, ops.convert_to_numpy(got))

    def test_rope_scaling(self):
        # Reference values computed from Huggingface llama implementation
        # With `scaling_factor` = 2.0
        # from transformers.models.llama.modeling_llama import (
        #     LlamaLinearScalingRotaryEmbedding,apply_rotary_pos_emb
        # )
        # import torch
        # torch.set_printoptions(precision=9)
        # rotary_emb = LlamaLinearScalingRotaryEmbedding(
        #     dim=4, max_position_embeddings=3, scaling_factor=2.0
        # )
        # query = torch.ones((1, 2, 3, 4)) # [bsz, num_heads, seq_len, head_dim]
        # cos, sin = rotary_emb(
        #     query, torch.unsqueeze(torch.arange(3, dtype=torch.int32), 0)
        # )
        # query, _ = apply_rotary_pos_emb(query, query, cos, sin)
        # print(query.transpose(1, 2))
        expected = [
            [
                [
                    [1.000000000, 1.000000000, 1.000000000, 1.000000000],
                    [1.000000000, 1.000000000, 1.000000000, 1.000000000],
                ],
                [
                    [0.398157001, 0.994987488, 1.357008100, 1.004987478],
                    [0.398157001, 0.994987488, 1.357008100, 1.004987478],
                ],
                [
                    [-0.301168621, 0.989950180, 1.381773233, 1.009949803],
                    [-0.301168621, 0.989950180, 1.381773233, 1.009949803],
                ],
            ]
        ]

        layer = RotaryEmbedding(scaling_factor=2.0)
        self.assertAllClose(
            layer(ops.ones((1, 3, 2, 4))),
            ops.convert_to_tensor(expected),
        )

    def test_rope_scaling_with_kv_cache(self):
        # Reference values computed from Huggingface llama implementation
        # With `scaling_factor` = 5.0
        # from transformers.models.llama.modeling_llama import (
        #     LlamaLinearScalingRotaryEmbedding,apply_rotary_pos_emb
        # )
        # import torch
        # torch.set_printoptions(precision=9)
        # rotary_emb = LlamaLinearScalingRotaryEmbedding(
        #     dim=4, max_position_embeddings=3, scaling_factor=5.0
        # )

        # query = torch.ones((1, 2, 1, 4)) # [bsz, num_heads, seq_len, head_dim]
        # cos, sin = rotary_emb(
        #     query, torch.unsqueeze(torch.arange(12, 13, dtype=torch.int32), 0)
        # )
        # query, _ = apply_rotary_pos_emb(query, query, cos, sin)
        # query.transpose(1, 2)
        expected = [
            [
                [
                    [-1.412856817, 0.975714266, -0.061930716, 1.023709655],
                    [-1.412856817, 0.975714266, -0.061930716, 1.023709655],
                ]
            ]
        ]

        layer = RotaryEmbedding(scaling_factor=5.0)
        self.assertAllClose(
            layer(ops.ones((1, 1, 2, 4)), start_index=12),
            ops.convert_to_tensor(expected),
        )
