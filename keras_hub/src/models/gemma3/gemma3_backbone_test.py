import copy

import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
    Gemma3VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma3BackboneTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 16
        self.vision_tokens_per_image = int((self.image_size / 4) ** 2)
        self.max_images_per_prompt = 3

        # === Vision + Text Backbone ===
        vision_encoder = Gemma3VisionEncoder(
            image_size=self.image_size,
            patch_size=4,
            pool_size=2,
            num_layers=2,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            output_dim=8,
        )

        self.init_kwargs = {
            # vocabulary
            "vocabulary_size": self.vocabulary_size,
            # image
            "image_size": self.image_size,
            # model
            "num_layers": 6,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            # other model args
            "query_head_dim_normalize": True,
            "use_query_key_norm": True,
            "use_post_ffw_norm": True,
            "use_post_attention_norm": True,
            "final_logit_soft_cap": None,
            "attention_logit_soft_cap": None,
            "use_sliding_window_attention": True,
            "sliding_window_size": 1024,
            "vision_encoder": vision_encoder,
        }

        dummy_images = np.random.rand(
            self.batch_size,
            self.max_images_per_prompt,
            self.image_size,
            self.image_size,
            3,
        )
        dummy_text_token_ids = np.random.rand(
            self.batch_size, self.text_sequence_length
        )

        self.input_data = {
            "token_ids": dummy_text_token_ids,
            "images": dummy_images,
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }
        vision_mask_0 = [False] * 20 + [True] * 8 + [False] * 32 + [True] * 4
        vision_mask_1 = [False] * 16 + [True] * 8 + [False] * 36 + [True] * 4
        self.input_data["vision_mask"] = np.array(
            [vision_mask_0, vision_mask_1]
        )
        self.input_data["vision_indices"] = np.array(
            [
                list(range(20, 28)) + list(range(60, 64)),
                list(range(16, 24)) + list(range(60, 64)),
            ]
        )

        # === Text Backbone ===
        self.text_init_kwargs = copy.deepcopy(self.init_kwargs)
        del self.text_init_kwargs["vision_encoder"]

        self.text_backbone_input_data = copy.deepcopy(self.input_data)
        del self.text_backbone_input_data["images"]
        del self.text_backbone_input_data["vision_mask"]
        del self.text_backbone_input_data["vision_indices"]

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_backbone_basics(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        elif backbone_type == "text_only":
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data

        self.run_backbone_test(
            cls=Gemma3Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                8,
            ),
            variable_length_data=[input_data],
            run_quantization_check=False,
        )

    def test_embedding_model(self):
        embedding_dim = 16
        pooling_intermediate_dim = 32
        init_kwargs = self.text_init_kwargs.copy()
        input_data = self.text_backbone_input_data.copy()

        init_kwargs["is_embedding_model"] = True
        init_kwargs["embedding_dim"] = embedding_dim
        init_kwargs["pooling_intermediate_dim"] = pooling_intermediate_dim

        self.run_backbone_test(
            cls=Gemma3Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
            expected_output_shape={
                "sequence_output": (
                    self.batch_size,
                    self.text_sequence_length,
                    8,
                ),
                "pooled_output": (self.batch_size, embedding_dim),
            },
        )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision", 7560, 15),
        ("text_only", "text_only", 5752, 10),
    )
    def test_architecture_characteristics(
        self, backbone_type, num_params, num_layers
    ):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
        elif backbone_type == "text_only":
            init_kwargs = self.text_init_kwargs

        model = Gemma3Backbone(**init_kwargs)
        self.assertEqual(model.count_params(), num_params)
        self.assertEqual(len(model.layers), num_layers)

    def test_backbone_interleaved_attention(self):
        backbone = Gemma3Backbone(**self.init_kwargs)
        for i, layer in enumerate(backbone.transformer_layers):
            expected_sliding = i % 6 < 5
            self.assertEqual(
                layer.use_sliding_window_attention,
                expected_sliding,
                f"Layer {i} mismatch: expected sliding={expected_sliding}, but "
                "got {layer.use_sliding_window_attention}",
            )

    @parameterized.named_parameters(
        ("text_and_vision", "text_and_vision"), ("text_only", "text_only")
    )
    def test_saved_model(self, backbone_type):
        if backbone_type == "text_and_vision":
            init_kwargs = self.init_kwargs
            input_data = self.input_data
        elif backbone_type == "text_only":
            init_kwargs = self.text_init_kwargs
            input_data = self.text_backbone_input_data

        self.run_model_saving_test(
            cls=Gemma3Backbone,
            init_kwargs=init_kwargs,
            input_data=input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_text_preset(self):
        self.run_preset_test(
            cls=Gemma3Backbone,
            preset="gemma3_instruct_1b",
            input_data={
                "token_ids": ops.array([[651, 4320, 8426, 25341, 235265]]),
                "padding_mask": ops.ones((1, 5), dtype="int32"),
            },
            expected_output_shape=(1, 5, 1152),
            # The forward pass from a preset should be stable!
            expected_partial_output=ops.array(
                [-0.400391, -8.625, 0.605469, 1.726562, -1.507812]
            ),
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Gemma3Backbone.presets:
            self.run_preset_test(
                cls=Gemma3Backbone,
                preset=preset,
                input_data=self.text_backbone_input_data
                if "_text" in preset or "1b" in preset
                else self.input_data,
            )
