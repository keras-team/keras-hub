import copy

import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class DiffusionGemmaBackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 32

        self.init_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "image_size": None,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "use_sliding_window_attention": True,
            "sliding_window_size": 16,
            "sliding_window_pattern": 2,
            "attention_logit_soft_cap": None,
            "final_logit_soft_cap": None,
            "vision_encoder": None,
        }

        self.input_data = {
            "token_ids": np.random.randint(
                0,
                self.vocabulary_size,
                size=(self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "position_ids": np.tile(
                np.arange(self.text_sequence_length, dtype="int32")[
                    np.newaxis, :
                ],
                (self.batch_size, 1),
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=DiffusionGemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                self.init_kwargs["hidden_dim"],
            ),
            variable_length_data=[self.input_data],
            run_quantization_check=False,
        )

    def test_all_text_layers_have_both_scalars(self):
        backbone = DiffusionGemmaBackbone(**self.init_kwargs)
        for i, layer in enumerate(backbone.transformer_layers):
            self.assertTrue(
                hasattr(layer, "layer_scalar"),
                f"Layer {i} missing layer_scalar",
            )
            self.assertTrue(
                hasattr(layer, "encoder_layer_scalar"),
                f"Layer {i} missing encoder_layer_scalar",
            )

    def test_backbone_layer_attention_pattern(self):
        """With sliding_window_pattern=2, every 2nd layer is global."""
        backbone = DiffusionGemmaBackbone(**self.init_kwargs)
        pattern = self.init_kwargs["sliding_window_pattern"]
        for i, layer in enumerate(backbone.transformer_layers):
            expected_global = (i % pattern) == (pattern - 1)
            expected_sliding = (
                not expected_global
            ) and backbone.use_sliding_window_attention
            self.assertEqual(
                layer.use_sliding_window_attention,
                expected_sliding,
                f"Layer {i}: expected sliding={expected_sliding}",
            )

    def test_moe_architecture(self):
        init_kwargs = copy.deepcopy(self.init_kwargs)
        init_kwargs["enable_moe_block"] = True
        init_kwargs["num_experts"] = 4
        init_kwargs["expert_intermediate_dim"] = 8
        init_kwargs["num_experts_per_token"] = 2
        model = DiffusionGemmaBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape,
            (
                self.batch_size,
                self.text_sequence_length,
                init_kwargs["hidden_dim"],
            ),
        )

    def test_partial_rotary(self):
        init_kwargs = copy.deepcopy(self.init_kwargs)
        init_kwargs["global_rope_partial_rotary_factor"] = 0.25
        model = DiffusionGemmaBackbone(**init_kwargs)
        output = model(self.input_data)
        self.assertEqual(
            output.shape,
            (
                self.batch_size,
                self.text_sequence_length,
                init_kwargs["hidden_dim"],
            ),
        )

    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DiffusionGemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DiffusionGemmaBackbone.presets:
            self.run_preset_test(
                cls=DiffusionGemmaBackbone,
                preset=preset,
                input_data=self.input_data,
            )
