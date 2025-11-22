# Copyright 2024 The KerasHub Authors
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

import pytest
from keras import ops

from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.tests.test_case import TestCase


class GptOssBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 4,
            "hidden_dim": 16,
            "intermediate_dim": 8,
            "num_experts": 2,
            "top_k": 2,
            "sliding_window": 2,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=GptOssBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=GptOssBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = GptOssBackbone(**self.init_kwargs)
        # Calculated based on the model architecture:
        # - Token embedding: vocabulary_size * hidden_dim
        # - Output projection: hidden_dim * vocabulary_size
        # - Transformer layers: num_layers * (attention + MoE block + LNs)
        # - Attention: q, k, v, o projections + sinks
        # - MoE: router (w+b) + experts (gate_up_proj (w+b), down_proj (w+b))
        # - Layer norms: hidden_dim each
        head_dim = 16 // 8  # hidden_dim / num_query_heads
        expected_params = (
            10 * 16  # Token embedding
            + 16 * 10  # Output projection
            + 2  # num_layers
            * (
                # Attention
                (16 * 8 * head_dim)
                + 8 * head_dim  # Query weight, bias
                + (16 * 4 * head_dim)
                + 4 * head_dim  # Key weight, bias
                + (16 * 4 * head_dim)
                + 4 * head_dim  # Value weight, bias
                + (8 * head_dim * 16)
                + 16  # Output weight, bias
                + 8  # Sinks
                # MoE
                + (16 * 2)
                + 2  # Router weight, bias
                + (2 * 16 * 2 * 8)  # Experts gate_up_proj weight
                + (2 * 2 * 8)  # Experts gate_up_proj bias
                + (2 * 8 * 16)  # Experts down_proj weight
                + (2 * 16)  # Experts down_proj bias
                # Layer Norms
                + 16  # Input LN
                + 16  # Post-attention LN
            )
            + 16  # Final layer norm
        )
        self.assertEqual(model.count_params(), expected_params)
