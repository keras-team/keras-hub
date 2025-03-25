import copy

import numpy as np
import pytest

from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_vit import Gemma3Vit
from keras_hub.src.tests.test_case import TestCase


class Gemma3BackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 16
        self.num_image_embeddings = int((self.image_size / 4) ** 2)
        self.image_max_length = 3

        # === Vision + Text Backbone ===
        vision_encoder = Gemma3Vit(
            **{
                "image_size": self.image_size,
                "patch_size": 4,
                "pool_size": 2,
                "num_layers": 2,
                "num_heads": 2,
                "hidden_dim": 8,
                "intermediate_dim": 16,
                "output_dim": 8,
            }
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
            self.image_max_length,
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
        text_mask_0 = [True] * 20 + [False] * 8 + [True] * 32 + [False] * 4
        text_mask_1 = (
            [False] * 4 + [True] * 16 + [False] * 4 + [True] * 36 + [False] * 4
        )
        self.input_data["text_mask"] = np.array([text_mask_0, text_mask_1])
        self.input_data["vision_indices"] = np.where(
            np.logical_not(np.reshape(self.input_data["text_mask"], -1))
        )[0]
        self.input_data["vision_indices"] = np.reshape(
            self.input_data["vision_indices"], (self.batch_size, -1)
        )

        empty_images = np.random.rand(
            self.batch_size,
            0,
            self.image_size,
            self.image_size,
            3,
        )

        self.text_only_input_data = {
            "token_ids": dummy_text_token_ids,
            "images": empty_images,
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "text_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "vision_indices": np.ones((self.batch_size, 0)),
        }

        # === Text Backbone ===
        self.text_init_kwargs = copy.deepcopy(self.init_kwargs)
        del self.text_init_kwargs["vision_encoder"]

        self.text_backbone_input_data = copy.deepcopy(self.input_data)
        del self.text_backbone_input_data["images"]
        del self.text_backbone_input_data["text_mask"]
        del self.text_backbone_input_data["vision_indices"]

    def test_text_only_backbone_basics(self):
        self.run_backbone_test(
            cls=Gemma3Backbone,
            init_kwargs=self.text_init_kwargs,
            input_data=self.text_backbone_input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                8,
            ),
            variable_length_data=[self.text_backbone_input_data],
            run_quantization_check=False,
        )

    def test_text_only_backbone_interleaved_attention(self):
        backbone = Gemma3Backbone(**self.text_init_kwargs)
        for i, layer in enumerate(backbone.transformer_layers):
            expected_sliding = i % 6 < 5
            self.assertEqual(
                layer.use_sliding_window_attention,
                expected_sliding,
                f"Layer {i} mismatch: expected sliding={expected_sliding}, but "
                "got {layer.use_sliding_window_attention}",
            )

    @pytest.mark.large
    def test_saved_text_model(self):
        self.run_model_saving_test(
            cls=Gemma3Backbone,
            init_kwargs=self.text_init_kwargs,
            input_data=self.text_backbone_input_data,
        )

    @pytest.mark.skipif(
        True,
        reason="disabled until the vision release.",
    )
    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Gemma3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                8,
            ),
            variable_length_data=[self.input_data],
            run_quantization_check=False,
        )

    @pytest.mark.skipif(
        True,
        reason="disabled until the vision release.",
    )
    def test_backbone_basics_text_only_input(self):
        self.run_backbone_test(
            cls=Gemma3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.text_only_input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length,
                8,
            ),
            variable_length_data=[self.text_only_input_data],
            run_quantization_check=False,
        )

    @pytest.mark.skipif(
        True,
        reason="disabled until the vision release.",
    )
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

    @pytest.mark.skipif(
        True,
        reason="disabled until the vision release.",
    )
    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Gemma3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
