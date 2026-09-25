import copy

import numpy as np
import pytest
from absl.testing import parameterized

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.gemma4.gemma4_vision_encoder import (
    Gemma4VisionEncoder,
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

        # === Vision + Text Backbone ===
        # (image_size / patch_size)^2 / pool_size^2 = (16/4)^2 / 2^2 = 4
        # vision tokens per image.
        self.image_size = 16
        self.max_images_per_prompt = 2
        self.num_vision_tokens_per_image = 4
        vision_encoder = Gemma4VisionEncoder(
            image_size=self.image_size,
            patch_size=4,
            pool_size=2,
            num_layers=1,
            num_heads=2,
            head_dim=4,
            num_key_value_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=self.init_kwargs["hidden_dim"],
        )
        self.vision_init_kwargs = copy.deepcopy(self.init_kwargs)
        self.vision_init_kwargs["image_size"] = self.image_size
        self.vision_init_kwargs["vision_encoder"] = vision_encoder

        num_patches = int((self.image_size / 4) ** 2)
        patch_dim = 3 * 4 * 4
        total_vision_tokens = (
            self.max_images_per_prompt * self.num_vision_tokens_per_image
        )
        self.vision_input_data = dict(self.input_data)
        self.vision_input_data["pixel_values"] = np.random.rand(
            self.batch_size,
            self.max_images_per_prompt,
            num_patches,
            patch_dim,
        ).astype("float32")
        self.vision_input_data["pixel_position_ids"] = np.ones(
            (self.batch_size, self.max_images_per_prompt, num_patches, 2),
            dtype="int32",
        )
        # Index 0 is reserved (never overwritten), so vision tokens start
        # at position 1.
        self.vision_input_data["vision_indices"] = np.tile(
            np.arange(1, total_vision_tokens + 1, dtype="int32")[np.newaxis, :],
            (self.batch_size, 1),
        )

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

    def test_from_config_reinjects_self_conditioning_reference(self):
        backbone = DiffusionGemmaBackbone(**self.init_kwargs)
        restored = DiffusionGemmaBackbone.from_config(backbone.get_config())
        self.assertIs(
            restored.diffusion_self_conditioning._token_embedding_layer,
            restored.token_embedding,
        )

    def test_self_conditioning_weights_are_tracked(self):
        # Never called in this backbone's own forward pass, so its weights
        # only appear in backbone.weights via the dummy graph-wiring call.
        backbone = DiffusionGemmaBackbone(**self.init_kwargs)
        sc = backbone.diffusion_self_conditioning
        tracked_weight_ids = {id(w) for w in backbone.weights}
        for layer in (sc.pre_norm, sc.gate_proj, sc.up_proj, sc.down_proj):
            for w in layer.weights:
                self.assertIn(id(w), tracked_weight_ids)

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
                input_data=self.vision_input_data,
            )
