import copy

import numpy as np
import pytest
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_vision_encoder import (
    Gemma3VisionEncoder,
)
from keras_hub.src.models.gemma3.gemma3_backbone import (
    Gemma3EmbeddingModel,
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

class Gemma3EmbeddingModelTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.hidden_dim = 8
        self.embedding_dim = 16

        self.backbone = Gemma3Backbone(
            vocabulary_size=self.vocabulary_size,
            image_size=16,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=self.hidden_dim,
            intermediate_dim=32,
            head_dim=4,
            vision_encoder=None,
        )

        self.init_kwargs = {
            "backbone": self.backbone,
            "embedding_dim": self.embedding_dim,
            "normalize": True,
        }

        self.init_kwargs_no_norm = {
            "backbone": self.backbone,
            "embedding_dim": self.embedding_dim,
            "normalize": False,
        }

        dummy_text_token_ids = np.random.randint(
            0,
            self.vocabulary_size,
            (self.batch_size, self.text_sequence_length),
        )
        padding_mask = np.ones(
            (self.batch_size, self.text_sequence_length), dtype="int32"
        )
        padding_mask[0, -10:] = 0
        padding_mask[1, -5:] = 0

        self.input_data = {
            "token_ids": dummy_text_token_ids,
            "padding_mask": padding_mask,
        }

    def test_model_basics(self):
        """Test the model's forward pass and output shape."""
        model = Gemma3EmbeddingModel(**self.init_kwargs)
        output = model(self.input_data)
        expected_output_shape = (self.batch_size, self.embedding_dim)
        self.assertEqual(output.shape, expected_output_shape)

    @parameterized.named_parameters(
        ("normalize", True, 9),
        ("no_normalize", False, 8),
    )
    def test_architecture_characteristics(self, normalize, num_layers):
        """Test parameter and layer counts."""
        init_kwargs = self.init_kwargs if normalize else self.init_kwargs_no_norm
        model = Gemma3EmbeddingModel(**init_kwargs)

        backbone_params = self.backbone.count_params()
        projection_params = (
            self.hidden_dim * self.embedding_dim
        ) + self.embedding_dim
        expected_params = backbone_params + projection_params

        self.assertEqual(model.count_params(), expected_params)
        self.assertEqual(len(model.layers), num_layers)

    def test_normalization(self):
        """Test that the `normalize` flag works correctly."""
        model_norm = Gemma3EmbeddingModel(**self.init_kwargs)
        outputs_norm = model_norm(self.input_data)

        norms_squared = ops.sum(ops.square(outputs_norm), axis=1)
        norms = ops.sqrt(norms_squared)

        self.assertAllClose(norms, ops.ones(self.batch_size), atol=1e-5)

        model_no_norm = Gemma3EmbeddingModel(**self.init_kwargs_no_norm)
        outputs_no_norm = model_no_norm(self.input_data)
        
        norms_no_norm_squared = ops.sum(ops.square(outputs_no_norm), axis=1)
        norms_no_norm = ops.sqrt(norms_no_norm_squared)
        
        self.assertNotAllClose(norms_no_norm, ops.ones(self.batch_size))

    @parameterized.named_parameters(
        ("normalize", True),
        ("no_normalize", False),
    )
    def test_saved_model(self, normalize):
        init_kwargs = self.init_kwargs if normalize else self.init_kwargs_no_norm

        self.run_model_saving_test(
            cls=Gemma3EmbeddingModel,
            init_kwargs=init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_build_from_preset_backbone(self):
        backbone = Gemma3Backbone.from_preset("gemma3_instruct_1b_text")
        model = Gemma3EmbeddingModel(
            backbone=backbone,
            embedding_dim=768,
            normalize=True,
        )

        input_data = {
            "token_ids": ops.array([[651, 4320, 8426, 25341, 235265]]),
            "padding_mask": ops.ones((1, 5), dtype="int32"),
        }

        outputs = model(input_data)

        self.assertEqual(outputs.shape, (1, 768))
        norm = ops.vector_norm(outputs, axis=1)
        self.assertAllClose(norm, [1.0])